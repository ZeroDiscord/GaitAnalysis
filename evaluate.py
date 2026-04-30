import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, classification_report

from dataset import create_dataloaders
from feature_config import FeatureConfig
from models.native_mamba import MambaGaitClassifier
from models.official_mamba import OfficialMambaGaitClassifier
from models.triton_mamba import HardwareMambaGaitClassifier
from models.gru_baseline import GRUAttentionGaitClassifier

def load_model(args, input_dim, num_classes, device):
    """Initializes the correct model architecture based on flags."""
    if args.use_official_mamba:
        print("Loading **OFFICIAL mamba-ssm** Classifier...")
        model = OfficialMambaGaitClassifier(
            input_dim=input_dim, num_classes=num_classes, d_model=args.d_model, n_layers=args.n_layers
        )
    elif args.use_triton_mamba:
        print("Loading **HARDWARE-ACCELERATED** Triton Mamba Classifier...")
        model = HardwareMambaGaitClassifier(
            input_dim=input_dim, num_classes=num_classes, d_model=args.d_model, n_layers=args.n_layers, chunk_size=2048
        )
    elif args.use_gru_baseline:
        print("Loading **BASELINE** GRU+Attention Classifier...")
        model = GRUAttentionGaitClassifier(
            input_dim=input_dim, num_classes=num_classes, d_model=args.d_model, num_heads=4, n_layers=args.n_layers
        )
    else:
        print("Loading **NATIVE PYTORCH** Mamba Classifier...")
        model = MambaGaitClassifier(
            input_dim=input_dim, num_classes=num_classes, d_model=args.d_model, n_layers=args.n_layers
        )
        
    # Load weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file {args.model_path} not found!")
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights. Make sure the model architecture flags match the saved model.\n{e}")
        exit(1)
        
    return model.to(device)

def plot_confusion_matrix(cm, classes, save_path="confusion_matrix.png"):
    """Generates and saves a customized confusion matrix plot."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14})
    plt.title('Gait Pathology Classification - Confusion Matrix', fontsize=16)
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved confusion matrix plot to {save_path}")

def _count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_evaluation(model, dataloader, device, num_classes, class_names, benchmark=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    batch_times = []
    total_samples = 0

    # GPU warm-up (avoids cold-start skewing the first batch)
    if device.type == 'cuda':
        dummy = torch.randn(1, 100, 4, device=device)
        with torch.no_grad():
            try:
                model(dummy, torch.ones(1, 100, device=device))
            except Exception:
                pass
        torch.cuda.synchronize()

    print("\nStarting Evaluation...")
    wall_start = time.perf_counter()

    with torch.no_grad():
        for features, masks, labels in dataloader:
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            batch_size = features.size(0)
            total_samples += batch_size

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # AMP: GPU only. CPU autocast only supports bfloat16, and float16 autocast
            # on CPU silently produces NaN/garbage — so we skip AMP entirely on CPU.
            if device.type == 'cuda':
                amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    outputs = model(features, masks)
            else:
                outputs = model(features, masks)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_times.append((t1 - t0, batch_size))
            
            # Explicit bfloat16 to float32 cast BEFORE softmax so probabilities sum exactly to 1.0
            probs = torch.nn.functional.softmax(outputs.float(), dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    wall_end = time.perf_counter()
    wall_elapsed = wall_end - wall_start

    # Compute Final Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Compute ROC-AUC handling binary vs multi-class
    try:
        all_probs_np = np.array(all_probs)
        if num_classes == 2:
             # For binary, select probability of the positive class (index 1)
             # Note: ensure binary isn't missing a class entirely in the test set
             if len(np.unique(all_labels)) > 1:
                auc_score = roc_auc_score(all_labels, all_probs_np[:, 1])
             else:
                auc_score = float('nan')
        else:
            auc_score = roc_auc_score(all_labels, all_probs_np, multi_class='ovr')
    except Exception as e:
        print(f"Warning: ROC-AUC could not be calculated (often due to missing classes in test split): {e}")
        auc_score = float('nan')
        
    cm = confusion_matrix(all_labels, all_preds)

    # Print Report
    print("\n" + "="*50)
    print("                EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 Score (Wd):    {f1:.4f}")
    if not np.isnan(auc_score):
        print(f"ROC-AUC Score:    {auc_score:.4f}")
    else:
        print("ROC-AUC Score:    [N/A]")
    print("\nDetailed Classification Report:")
    # Ignore warnings for undefined metrics if a class was never predicted
    print(classification_report(all_labels, all_preds, labels=range(num_classes), target_names=class_names, zero_division=0))
    print("="*50)

    # ── Inference-Time Metrics ─────────────────────────────────────────
    total_param, trainable_param = _count_parameters(model)
    sum_batch_time = sum(t for t, _ in batch_times)
    per_sample_ms = (sum_batch_time / max(total_samples, 1)) * 1000
    throughput = total_samples / max(sum_batch_time, 1e-9)

    print("\n" + "="*50)
    print("             INFERENCE / EFFICIENCY METRICS")
    print("="*50)
    print(f"Total parameters:       {total_param:,}")
    print(f"Trainable parameters:   {trainable_param:,}")
    print(f"Model size (approx):    {total_param * 4 / 1024 / 1024:.2f} MB (FP32)")
    print(f"Total samples:          {total_samples}")
    print(f"Wall-clock time:        {wall_elapsed:.4f} s")
    print(f"Forward-pass time:      {sum_batch_time:.4f} s")
    print(f"Latency / sample:       {per_sample_ms:.3f} ms")
    print(f"Throughput:             {throughput:.1f} samples/s")

    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        print(f"Peak GPU memory:        {peak_mem:.1f} MB")

    if benchmark:
        # Detailed per-batch breakdown
        print("\n  Per-batch latency breakdown:")
        for i, (bt, bs) in enumerate(batch_times):
            print(f"    Batch {i:>3d}:  {bt*1000:7.2f} ms  ({bs} samples, {bt/bs*1000:.2f} ms/sample)")

    print("="*50)
    
    return all_labels, all_preds, all_probs, cm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Trained Gait Models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .pth weights file')
    parser.add_argument('--data_dir', type=str, default='Datasets/', help='Path to Datasets folder')
    parser.add_argument('--output_plot', type=str, default='confusion_matrix.png', help='Where to save the CM plot')
    
    # Model architecture parameters (must match the trained model!)
    parser.add_argument('--d_model', type=int, default=32, help='Hidden dimension used in training (default is now 32)')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers used in training')
    
    # Model selection flags (Only pass one)
    parser.add_argument('--use_triton_mamba', action='store_true', help='Evaluate Triton Mamba model')
    parser.add_argument('--use_official_mamba', action='store_true', help='Evaluate Official Mamba model')
    parser.add_argument('--use_gru_baseline', action='store_true', help='Evaluate GRU Baseline model')
    
    # Use testing batch size (can be larger up to memory limit since no gradients stored)
    parser.add_argument('--batch_size', type=int, default=4, help='Inference batch size')
    parser.add_argument('--benchmark', action='store_true', help='Print detailed per-batch inference timing breakdown')

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")

    # To ensure identical pre-processing, we use the same loader but only take the test_loader
    print(f"Scanning dataset from {args.data_dir} to extract classes...")
    try:
        _, _, test_loader, train_dataset = create_dataloaders(
            args.data_dir, 
            batch_size=args.batch_size,
            window_size=2000,
            base_stride=1000
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        exit(1)
        
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")
    print(f"Test Set Size: {len(test_loader.dataset)} windows")
    
    if len(test_loader.dataset) == 0:
         print("\nWARNING: Test set is empty! Your dataset might be too small for the validation split.")
         print("Generating dummy evaluation on training set instead for demonstration.")
         test_loader, _, _, _ = create_dataloaders(args.data_dir, batch_size=args.batch_size, random_seed=1)
         
    input_dim = train_dataset.get_feature_count()
    print(f"Input dimension: {input_dim} features per timestep")
    model = load_model(args, input_dim, num_classes, device)
    
    _, _, _, cm = run_evaluation(model, test_loader, device, num_classes, class_names, benchmark=args.benchmark)
    plot_confusion_matrix(cm, class_names, save_path=args.output_plot)
