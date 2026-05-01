"""
evaluate.py — Evaluate trained gait classification models.
===========================================================
Auto-detects model architecture from checkpoint config (saved by checkpoint_utils).
Falls back to manual --model_type flag for legacy checkpoints.

Usage:
  python evaluate.py --model_path checkpoints/best_fold_0.pth --data_dir Datasets/
  python evaluate.py --model_path best_model.pth --data_dir Datasets/ --benchmark
"""

import os
import argparse
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_auc_score, classification_report, recall_score)
from tqdm import tqdm

from dataset import GaitDataset, collate_fn_pad, discover_dataset, create_dataloaders
from checkpoint_utils import load_checkpoint
from models.bimamba_classifier import BiMambaGaitClassifier
from models.gru_baseline import GRUAttentionGaitClassifier

# Legacy models
try:
    from models.native_mamba import MambaGaitClassifier
    from models.triton_mamba import HardwareMambaGaitClassifier
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False


def build_model(config, device):
    """
    Build model from config dict (auto-detected from checkpoint).
    """
    model_type = config.get('model_type', 'bimamba')
    input_dim = config.get('input_dim', 5)
    num_classes = config.get('num_classes', 4)
    d_model = config.get('d_model', 64)
    n_layers = config.get('n_layers', 2)
    dropout = config.get('dropout', 0.1)

    if model_type == 'bimamba':
        model = BiMambaGaitClassifier(
            input_dim=input_dim, num_classes=num_classes,
            d_model=d_model, n_layers=n_layers, dropout=dropout,
        )
    elif model_type == 'gru':
        model = GRUAttentionGaitClassifier(
            input_dim=input_dim, num_classes=num_classes,
            d_model=d_model, n_layers=n_layers, dropout=dropout,
        )
    elif model_type == 'mamba' and LEGACY_AVAILABLE:
        model = MambaGaitClassifier(
            input_dim=input_dim, num_classes=num_classes,
            d_model=d_model, n_layers=n_layers,
        )
    elif model_type == 'triton_mamba' and LEGACY_AVAILABLE:
        model = HardwareMambaGaitClassifier(
            input_dim=input_dim, num_classes=num_classes,
            d_model=d_model, n_layers=n_layers,
        )
    else:
        raise ValueError(f"Unknown model type in config: {model_type}")

    # Handle prototype head if saved
    if config.get('use_prototype', False):
        from prototype_head import HybridClassifier
        model.classifier = HybridClassifier(d_model, num_classes, temperature=0.1)

    return model.to(device)


def plot_confusion_matrix(cm, classes, save_path="confusion_matrix.png"):
    """Generates and saves a confusion matrix plot."""
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
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_evaluation(model, dataloader, device, num_classes, class_names, benchmark=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    batch_times = []
    total_samples = 0

    # GPU warm-up
    if device.type == 'cuda':
        dummy = torch.randn(1, 100, 5, device=device)
        with torch.no_grad():
            try:
                model(dummy, torch.ones(1, 100, device=device))
            except Exception:
                pass
        torch.cuda.synchronize()

    print("\nStarting Evaluation...")
    wall_start = time.perf_counter()

    with torch.no_grad():
        for features, masks, labels in tqdm(dataloader, desc="Evaluating"):
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            batch_size = features.size(0)
            total_samples += batch_size

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

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

            probs = torch.nn.functional.softmax(outputs.float(), dim=1)
            _, preds = torch.max(outputs, 1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    wall_end = time.perf_counter()
    wall_elapsed = wall_end - wall_start

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    wf1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    recalls = recall_score(all_labels, all_preds, average=None,
                          labels=list(range(num_classes)), zero_division=0)

    try:
        all_probs_np = np.array(all_probs)
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs_np[:, 1]) if len(np.unique(all_labels)) > 1 else float('nan')
        else:
            auc = roc_auc_score(all_labels, all_probs_np, multi_class='ovr')
    except Exception:
        auc = float('nan')

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # Print results
    print("\n" + "=" * 50)
    print("                EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:           {acc:.4f} ({acc*100:.2f}%)")
    print(f"Weighted F1:        {wf1:.4f}")
    print(f"Macro F1:           {mf1:.4f}")
    print(f"Min Per-Class Recall: {np.min(recalls):.4f}")
    if not np.isnan(auc):
        print(f"ROC-AUC:            {auc:.4f}")
    print(f"\nPer-class recall: {dict(zip(class_names, recalls.round(3)))}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                labels=range(num_classes),
                                target_names=class_names, zero_division=0))

    # Efficiency metrics
    total_param, trainable_param = _count_parameters(model)
    sum_batch_time = sum(t for t, _ in batch_times)
    per_sample_ms = (sum_batch_time / max(total_samples, 1)) * 1000
    throughput = total_samples / max(sum_batch_time, 1e-9)

    print("=" * 50)
    print("             INFERENCE METRICS")
    print("=" * 50)
    print(f"Parameters:         {total_param:,}")
    print(f"Model size:         {total_param * 4 / 1024 / 1024:.2f} MB (FP32)")
    print(f"Samples:            {total_samples}")
    print(f"Wall time:          {wall_elapsed:.4f} s")
    print(f"Latency / sample:   {per_sample_ms:.3f} ms")
    print(f"Throughput:         {throughput:.1f} samples/s")

    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        print(f"Peak GPU memory:    {peak_mem:.1f} MB")

    if benchmark:
        print("\n  Per-batch breakdown:")
        for i, (bt, bs) in enumerate(batch_times):
            print(f"    Batch {i:>3d}:  {bt*1000:7.2f} ms  ({bs} samples)")

    print("=" * 50)
    return all_labels, all_preds, all_probs, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Gait Classification Models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--data_dir', type=str, default='Datasets/')
    parser.add_argument('--output_plot', type=str, default='confusion_matrix.png')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--benchmark', action='store_true')

    # Manual overrides (only needed for legacy checkpoints without embedded config)
    parser.add_argument('--model_type', type=str, default=None,
                        help='Override model type (auto-detected from checkpoint config)')
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.model_path}")
    state_dict, config, meta = load_checkpoint(args.model_path, device)

    # Apply manual overrides if config was inferred (legacy)
    if config.get('_inferred', False) or not config:
        print("WARNING: Legacy checkpoint — no embedded config. Using CLI flags or defaults.")
        config.setdefault('model_type', args.model_type or 'bimamba')
        config.setdefault('d_model', args.d_model or 64)
        config.setdefault('n_layers', args.n_layers or 2)
        config.setdefault('input_dim', 5)
    else:
        print(f"  Config: {config}")

    if args.model_type:
        config['model_type'] = args.model_type
    if args.d_model:
        config['d_model'] = args.d_model
    if args.n_layers:
        config['n_layers'] = args.n_layers

    # Load dataset to get class info and test split
    print(f"Loading dataset from {args.data_dir}...")
    _, _, test_loader, train_dataset = create_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        window_size=2000, base_stride=1000,
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)
    input_dim = train_dataset.get_feature_count()
    config['num_classes'] = num_classes
    config['input_dim'] = input_dim

    print(f"Classes: {class_names}")
    print(f"Test set: {len(test_loader.dataset)} windows")
    print(f"Input dim: {input_dim}")

    # Build model and load weights
    model = build_model(config, device)
    model.load_state_dict(state_dict)
    print("Weights loaded successfully.")

    # Run evaluation
    _, _, _, cm = run_evaluation(
        model, test_loader, device, num_classes, class_names,
        benchmark=args.benchmark,
    )
    plot_confusion_matrix(cm, class_names, save_path=args.output_plot)
