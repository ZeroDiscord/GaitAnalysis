import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from dataset import create_dataloaders
from feature_config import FeatureConfig
from models.native_mamba import MambaGaitClassifier
from models.official_mamba import OfficialMambaGaitClassifier
from models.triton_mamba import HardwareMambaGaitClassifier
from models.gru_baseline import GRUAttentionGaitClassifier

def train_epoch(model, dataloader, criterion, optimizer, device, scaler, scheduler, accum_steps=1, use_scaler=True):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    nan_batches = 0

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for i, (features, masks, labels) in enumerate(pbar):
        features = features.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        # Select AMP dtype: bfloat16 on H100 (no scaler needed), float16 elsewhere
        amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        with torch.cuda.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu', dtype=amp_dtype):  # type: ignore[attr-defined]
            outputs = model(features, masks)
            loss = criterion(outputs, labels) / accum_steps
        
        # NaN Guard: Skip corrupted batches to prevent permanent model poisoning
        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            optimizer.zero_grad() # Flush any accumulated NaN gradients
            continue
        
        # Backward pass: use GradScaler ONLY for float16, NOT bfloat16
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Step if we've reached the accumulation boundary
        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()

        running_loss += (loss.item() * accum_steps) * features.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    if nan_batches > 0:
        print(f"  WARNING: {nan_batches} batches had NaN/Inf loss and were skipped.")
        
    total_samples = len(dataloader.dataset) - nan_batches * dataloader.batch_size
    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division='0') if all_labels else 0.0
    
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

            amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
            with torch.cuda.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu', dtype=amp_dtype):  # type: ignore[attr-defined]
                outputs = model(features, masks)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Convert bfloat16 explicitly to float32 before passing to numpy
            all_probs.extend(probs.float().cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division='0')
    
    # Compute ROC-AUC based on number of classes
    try:
        all_probs_np = np.array(all_probs)
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs_np[:, 1])
        else:
            # Multi-class One-vs-Rest AUC explicitly requires the 'labels' parameter
            # to prevent crashing if a class is entirely missing in a mini-test-set.
            # We strictly enforce float64 precision and safe softmax accumulation here.
            # If a class is completely structurally missing from the *ground truth* of the split,
            # roc_auc_score will fail. We use a safe wrapper.
            valid_classes = np.unique(all_labels)
            if len(valid_classes) < 2:
                auc = 0.0 # Mathematically impossible to compute AUC with 1 ground truth class
            else:
                # Calculate AUC only on the classes that actually exist in the ground truth
                # to prevent shape mismatch errors.
                filtered_probs = all_probs_np[:, valid_classes]
                # Re-normalize probabilities for the valid classes to sum to 1
                row_sums = filtered_probs.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1e-9
                filtered_probs = filtered_probs / row_sums
                
                auc = roc_auc_score(
                    all_labels, 
                    filtered_probs, 
                    multi_class='ovr', 
                    labels=valid_classes
                )
    except Exception as e:
        auc = 0.0
        
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
    
    # Enhanced feature arguments
    parser.add_argument('--enhanced_features', action='store_true', help='Use enhanced feature extraction (17+ features)')
    parser.add_argument('--legacy_mode', action='store_true', help='Force legacy mode (5 features) even with enhanced_features flag')
    parser.add_argument('--enable_pca', action='store_true', help='Enable PCA dimensionality reduction')
    parser.add_argument('--pca_components', type=int, default=10, help='Number of PCA components')
    parser.add_argument('--enable_ica', action='store_true', help='Enable ICA dimensionality reduction')
    parser.add_argument('--ica_components', type=int, default=8, help='Number of ICA components')
    
    args = parser.parse_args()

    # Configure feature extraction
    if args.enhanced_features and not args.legacy_mode:
        feature_config = FeatureConfig(
            legacy_mode=False,
            include_base_features=True,
            include_time_domain=True,
            include_freq_domain=True,
            include_gait_cycle=True,
            enable_pca=args.enable_pca,
            pca_components=args.pca_components,
            enable_ica=args.enable_ica,
            ica_components=args.ica_components,
            enable_feature_caching=True
        )
        print("Using Enhanced Feature Mode:")
        print(f"  - Total features: {feature_config.get_total_feature_count()}")
        print(f"  - PCA: {'Enabled' if args.enable_pca else 'Disabled'}")
        print(f"  - ICA: {'Enabled' if args.enable_ica else 'Disabled'}")
    else:
        feature_config = None
        print("Using Legacy Feature Mode (5 features)")

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
        os.makedirs(os.path.join(args.data_dir, "01_Normal"), exist_ok=True)
        os.makedirs(os.path.join(args.data_dir, "02_Pathological"), exist_ok=True)
        
        # Generate some random dummy CSVs (representing E_ant and E_ago time series)
        for i in range(20):
            seq_len = np.random.randint(50, 150)
            df_norm = pd.DataFrame({
                # Normal: High distinct separated peaks (made up)
                'E_ant': np.sin(np.linspace(0, 10, seq_len)) + np.random.normal(0, 0.1, seq_len),
                'E_ago': np.cos(np.linspace(0, 10, seq_len)) + np.random.normal(0, 0.1, seq_len)
            })
            df_norm.to_csv(os.path.join(args.data_dir, "01_Normal", f"sample_{i}.csv"), index=False, header=False)
            
            df_patho = pd.DataFrame({
                # Pathological: Co-contraction (high overlap)
                'E_ant': np.sin(np.linspace(0, 10, seq_len)) + np.random.normal(0.5, 0.2, seq_len),
                'E_ago': np.sin(np.linspace(0, 10, seq_len)) + np.random.normal(0.5, 0.2, seq_len)
            })
            df_patho.to_csv(os.path.join(args.data_dir, "02_Pathological", f"sample_{i}.csv"), index=False, header=False)

    print("Initializing Data Loaders with Sliding Window...")
    # Smaller batch size default because Mamba has state memory, but each sample is now shorter
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size,
        window_size=2000,   # Slicing the 26000 sequence into 2000 frame chunks
        base_stride=1000,   # Generate a new slice every 1000 frames
        feature_config=feature_config
    )
    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"Detected {num_classes} classes: {classes}")
    
    # Get actual input dimension from dataset
    input_dim = train_dataset.get_feature_count()
    print(f"Input dimension: {input_dim} features per timestep")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Initialization
    # Defaulting to smaller parameter footprint for 5-sample dataset robustness 
    d_model_eff = args.d_model if args.d_model <= 32 else 32
    n_layers_eff = args.n_layers if args.n_layers <= 2 else 2
    
    if args.use_official_mamba:
        print(f"Initializing **OFFICIAL mamba-ssm** Classifier (d_model={d_model_eff}, layers={n_layers_eff})")
        model = OfficialMambaGaitClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model_eff,
            n_layers=n_layers_eff
        ).to(device)
    elif args.use_triton_mamba:
        print(f"Initializing **HARDWARE-ACCELERATED** Triton Mamba Classifier (d_model={d_model_eff}, layers={n_layers_eff})")
        model = HardwareMambaGaitClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model_eff,
            n_layers=n_layers_eff,
            chunk_size=2048
        ).to(device)
    elif args.use_gru_baseline:
        print(f"Initializing **BASELINE** GRU+Attention Classifier (d_model={d_model_eff}, layers={n_layers_eff})")
        model = GRUAttentionGaitClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model_eff,
            num_heads=4,
            n_layers=n_layers_eff
        ).to(device)
    else:
        model = MambaGaitClassifier(
            input_dim=input_dim,  # Dynamically loaded from FeatureConfig
            num_classes=num_classes,
            d_model=d_model_eff,
            n_layers=n_layers_eff
        ).to(device)
        
    # Standard Cross Entropy Loss. 
    # Class weights removed because Sliding Window dynamically balances the dataset batches!
    # Added label_smoothing to prevent overconfidence on the small dataset (improves generalization)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Initialize robust Learning Rate Scheduler (OneCycleLR for stable warm-up and cool-down)
    total_steps = int(args.epochs * np.ceil(len(train_loader) / args.accum_steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos'
    )
    
    # Mixed Precision Setup:
    # CRITICAL: GradScaler is ONLY for float16. bfloat16 has float32's exponent range
    # and does NOT need loss scaling. Using GradScaler with bfloat16 silently corrupts
    # Mamba's sequential state recurrence, causing permanent NaN.
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    if use_bf16:
        print("H100/bfloat16 detected: GradScaler DISABLED (not needed for bf16).")
        scaler = None
        use_scaler = False
    else:
        scaler = torch.cuda.amp.GradScaler(device.type if device.type != 'mps' else 'cpu')  # type: ignore[attr-defined]
        use_scaler = True

    best_val_f1 = 0.0

    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, scheduler, 
            accum_steps=args.accum_steps, use_scaler=use_scaler
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
