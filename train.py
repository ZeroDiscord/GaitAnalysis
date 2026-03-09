import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import argparse
from tqdm import tqdm
import numpy as np
from dataset import create_dataloaders
from models.native_mamba import MambaGaitClassifier
from models.official_mamba import OfficialMambaGaitClassifier
from models.triton_mamba import HardwareMambaGaitClassifier
from models.gru_baseline import GRUAttentionGaitClassifier

def train_epoch(model, dataloader, criterion, optimizer, device, scaler, accum_steps=1):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for i, (features, masks, labels) in enumerate(pbar):
        features = features.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        # Forward pass with Automatic Mixed Precision
        with torch.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu'):
            outputs = model(features, masks)
            loss = criterion(outputs, labels) / accum_steps
        
        # Backward and optimize with Scaler
        scaler.scale(loss).backward()
        
        # Unscale and step if we've reached the accumulation boundary
        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            # Gradient clipping for stability in sequence models
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += (loss.item() * accum_steps) * features.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1

def evaluate(model, dataloader, criterion, device, num_classes, desc="Validating"):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for features, masks, labels in pbar:
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu'):
                outputs = model(features, masks)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Compute ROC-AUC based on number of classes
    try:
        all_probs_np = np.array(all_probs)
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs_np[:, 1])
        else:
            # Multi-class One-vs-Rest AUC
            auc = roc_auc_score(all_labels, all_probs_np, multi_class='ovr')
    except Exception as e:
        auc = 0.0 # Can happen if a class is entirely missing in a mini-test-set
        
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, auc, cm

def main():
    parser = argparse.ArgumentParser(description='Train Mamba Gait Classifier')
    parser.add_argument('--data_dir', type=str, default='Datasets/', help='Directory containing the dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (keep small, e.g., 2 or 4, for full sequences)')
    parser.add_argument('--accum_steps', type=int, default=8, help='Gradient accumulation steps to simulate larger batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=64, help='Model hidden dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of Mamba layers')
    parser.add_argument('--use_triton_mamba', action='store_true', help='Use True Fused Triton/CUDA Hardware-Accelerated Mamba Module')
    parser.add_argument('--use_official_mamba', action='store_true', help='Use the official mamba-ssm package (requires causal-conv1d and mamba-ssm to be pip installed)')
    parser.add_argument('--use_gru_baseline', action='store_true', help='Use the GRU+Attention baseline model instead of Mamba for comparison')
    parser.add_argument('--output_name', type=str, default='best_model.pth', help='Filename to save the best model weights (e.g., mamba_best.pth)')
    args = parser.parse_args()

    # Create dummy data if folder doesn't exist or is empty just for testing pipeline setup
    has_data = False
    if os.path.exists(args.data_dir):
        # Check if there are any CSV files in subdirectories
        for root, dirs, files in os.walk(args.data_dir):
            if any(f.endswith('.csv') for f in files):
                has_data = True
                break
                
    if not has_data:
        print(f"Creating dummy dataset at {args.data_dir} for testing...")
        os.makedirs(os.path.join(args.data_dir, "Normal"), exist_ok=True)
        os.makedirs(os.path.join(args.data_dir, "Pathological"), exist_ok=True)
        
        # Generate some random dummy CSVs (representing E_ant and E_ago time series)
        for i in range(20):
            seq_len = np.random.randint(50, 150)
            df_norm = pd.DataFrame({
                # Normal: High distinct separated peaks (made up)
                'E_ant': np.sin(np.linspace(0, 10, seq_len)) + np.random.normal(0, 0.1, seq_len),
                'E_ago': np.cos(np.linspace(0, 10, seq_len)) + np.random.normal(0, 0.1, seq_len)
            })
            df_norm.to_csv(os.path.join(args.data_dir, "Normal", f"sample_{i}.csv"), index=False)
            
            df_patho = pd.DataFrame({
                # Pathological: Co-contraction (high overlap)
                'E_ant': np.sin(np.linspace(0, 10, seq_len)) + np.random.normal(0.5, 0.2, seq_len),
                'E_ago': np.sin(np.linspace(0, 10, seq_len)) + np.random.normal(0.5, 0.2, seq_len)
            })
            df_patho.to_csv(os.path.join(args.data_dir, "Pathological", f"sample_{i}.csv"), index=False)

    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )
    classes = dataset.classes
    num_classes = len(classes)
    print(f"Detected {num_classes} classes: {classes}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Initialization
    if args.use_official_mamba:
        print("Initializing **OFFICIAL mamba-ssm** Classifier")
        model = OfficialMambaGaitClassifier(
            input_dim=4,
            num_classes=num_classes,
            d_model=args.d_model,
            n_layers=args.n_layers
        ).to(device)
    elif args.use_triton_mamba:
        print("Initializing **HARDWARE-ACCELERATED** Triton Mamba Classifier")
        model = HardwareMambaGaitClassifier(
            input_dim=4,
            num_classes=num_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            chunk_size=2048
        ).to(device)
    elif args.use_gru_baseline:
        print("Initializing **BASELINE** GRU+Attention Classifier")
        model = GRUAttentionGaitClassifier(
            input_dim=4,
            num_classes=num_classes,
            d_model=args.d_model,
            num_heads=4,
            n_layers=args.n_layers
        ).to(device)
    else:
        model = MambaGaitClassifier(
            input_dim=4, # [E_ant, E_ago, Torque, Stiffness]
            num_classes=num_classes,
            d_model=args.d_model,
            n_layers=args.n_layers
        ).to(device)
    
    # Calculate class weights for imbalanced datasets
    class_weights = dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Initialize Mixed Precision Scaler
    scaler = torch.amp.GradScaler(device.type if device.type != 'mps' else 'cpu')

    best_val_f1 = 0.0

    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, accum_steps=args.accum_steps
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_auc, _ = evaluate(model, val_loader, criterion, device, num_classes)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}")
              
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), args.output_name)
            print(f"  --> Saved new best model to {args.output_name}!")

    print("\nTraining Complete. Evaluating on Test Set...")
    # Load best model for testing
    model.load_state_dict(torch.load(args.output_name))
    test_loss, test_acc, test_f1, test_auc, test_cm = evaluate(model, test_loader, criterion, device, num_classes, desc="Testing")
    
    print("\n=== Final Test Results ===")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"ROC-AUC: {test_auc:.4f}")
    print(f"Confusion Matrix:\n{test_cm}")

if __name__ == "__main__":
    main()
