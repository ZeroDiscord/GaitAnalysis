"""
train.py — Leave-One-Patient-Out Cross-Validation Training
============================================================
Gold-standard evaluation for small biomedical cohorts.
Every patient gets exactly one turn as the test patient.
Results are aggregated across all folds with mean +/- std.

Supports:
  - BiMamba (bidirectional Mamba) -- recommended
  - GRU (bidirectional + non-causal attention)
  - Prototype classifier head
  - Focal loss with patient-level class weights
  - Advanced augmentation pipeline
  - Checkpoint config saving

Usage:
  python train.py --data_dir Datasets/ --model_type bimamba --epochs 60
  python train.py --data_dir Datasets/ --model_type gru --epochs 60
  python train.py --data_dir Datasets/ --model_type bimamba --use_prototype_head --epochs 60
  python train.py --data_dir Datasets/ --model_type bimamba --cv_mode kfold --n_folds 5
  python train.py --data_dir Datasets/ --model_type bimamba --single_fold
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, recall_score, roc_auc_score)
import argparse
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

from dataset import (GaitDataset, collate_fn_pad, discover_dataset,
                     patient_level_lopo_splits, patient_level_kfold_splits)
from augmentations import AugmentationPipeline
from checkpoint_utils import save_checkpoint, load_checkpoint, make_config
from prototype_head import PrototypeClassifier, HybridClassifier

# Model architectures
from models.bimamba_classifier import BiMambaGaitClassifier
from models.gru_baseline import GRUAttentionGaitClassifier

# Legacy models (still functional, kept in models/)
try:
    from models.native_mamba import MambaGaitClassifier
    from models.triton_mamba import HardwareMambaGaitClassifier
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.02):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            focal_w = (1 - probs) ** self.gamma
            loss = -focal_w * smooth * log_probs
            if self.weight is not None:
                loss = loss * self.weight.unsqueeze(0)
            return loss.sum(dim=1).mean()
        else:
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_w = (1 - p_t) ** self.gamma
            ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
            return (focal_w * ce).mean()


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def create_model(model_type: str, input_dim: int, num_classes: int,
                 d_model: int, n_layers: int, dropout: float = 0.1,
                 use_prototype_head: bool = False) -> nn.Module:
    """Create model by type string."""
    
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
        available = 'bimamba, gru'
        if LEGACY_AVAILABLE:
            available += ', mamba, triton_mamba'
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    
    # Replace classifier head with prototype head if requested
    if use_prototype_head and hasattr(model, 'classifier'):
        embed_dim = d_model
        model.classifier = HybridClassifier(embed_dim, num_classes, temperature=0.1)
        print(f"  Replaced classifier head with HybridClassifier (linear + prototype)")
    
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scheduler, device,
                accum_steps=1, use_prototype=False):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    nan_batches = 0
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"      Train", leave=False)
    for i, (features, masks, labels) in enumerate(pbar):
        features = features.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu',
                                 enabled=device.type == 'cuda'):
            if use_prototype and hasattr(model, 'get_embedding'):
                embeddings = model.get_embedding(features, masks)
                outputs = model.classifier(embeddings, labels)
            else:
                outputs = model(features, masks)
            loss = criterion(outputs, labels) / accum_steps
        
        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            optimizer.zero_grad()
            continue
        
        loss.backward()
        
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        current_loss = loss.item()
        running_loss += (current_loss * accum_steps) * features.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{current_loss:.4f}'})
    
    if nan_batches > 0:
        print(f"  WARNING: {nan_batches} NaN batches skipped")
    
    total = max(len(loader.dataset), 1)
    return running_loss / total, accuracy_score(all_labels, all_preds) if all_labels else 0.0


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, use_prototype=False):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    
    for features, masks, labels in tqdm(loader, desc="      Eval ", leave=False):
        features = features.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        if use_prototype and hasattr(model, 'get_embedding'):
            embeddings = model.get_embedding(features, masks)
            outputs = model.classifier(embeddings)
        else:
            outputs = model(features, masks)
        
        loss = criterion(outputs, labels)
        running_loss += loss.item() * features.size(0)
        
        probs = F.softmax(outputs.float(), dim=1)
        _, preds = torch.max(outputs, 1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    total = max(len(loader.dataset), 1)
    avg_loss = running_loss / total
    acc = accuracy_score(all_labels, all_preds)
    wf1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    recalls = recall_score(all_labels, all_preds, average=None,
                          labels=list(range(num_classes)), zero_division=0)
    min_recall = float(np.min(recalls)) if len(recalls) > 0 else 0.0
    
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    return {
        'loss': avg_loss, 'acc': acc, 'weighted_f1': wf1, 'macro_f1': mf1,
        'min_recall': min_recall, 'recalls': recalls, 'cm': cm,
        'labels': all_labels, 'preds': all_preds, 'probs': all_probs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='LOPO CV Training for Gait Classification')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='bimamba',
                        choices=['bimamba', 'gru', 'mamba', 'triton_mamba'])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--window_size', type=int, default=2000)
    parser.add_argument('--stride', type=int, default=1000)
    parser.add_argument('--use_prototype_head', action='store_true')
    parser.add_argument('--cv_mode', type=str, default='lopo', choices=['lopo', 'kfold'])
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for kfold mode')
    parser.add_argument('--single_fold', action='store_true', help='Run only first fold')
    parser.add_argument('--output_dir', type=str, default='checkpoints/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (set to ncpus-2 on HPC, e.g. 8)')
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {args.model_type} | d_model={args.d_model} | n_layers={args.n_layers}")
    print(f"CV mode: {args.cv_mode}")
    
    # Get splits
    if args.cv_mode == 'lopo':
        splits = list(patient_level_lopo_splits(args.data_dir))
    else:
        splits = list(patient_level_kfold_splits(args.data_dir, args.n_folds, args.seed))
    
    n_folds = 1 if args.single_fold else len(splits)
    all_fold_results = []
    
    # Augmentation pipeline
    train_aug = AugmentationPipeline.default_emg()
    
    for fold_idx in range(n_folds):
        # Clear GPU memory from previous fold
        torch.cuda.empty_cache()
        
        split = splits[fold_idx]
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, \
            class_to_idx, classes, fold_name = split
        
        num_classes = len(classes)
        
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx + 1}/{n_folds} -- {fold_name}")
        print(f"{'='*60}")
        print(f"  Train: {len(train_paths)} files | Val: {len(val_paths)} | Test: {len(test_paths)}")
        
        set_seed(args.seed + fold_idx)  # Different seed per fold for model init
        
        # Create datasets
        train_ds = GaitDataset(
            train_paths, train_labels, class_to_idx,
            window_size=args.window_size, base_stride=args.stride, mode='train',
            balance_classes=True, augmentation=train_aug,
        )
        
        g_mean, g_std = train_ds.global_mean, train_ds.global_std
        input_dim = train_ds.get_feature_count()
        
        val_ds = GaitDataset(
            val_paths, val_labels, class_to_idx,
            window_size=args.window_size, base_stride=args.stride, mode='val',
            global_mean=g_mean, global_std=g_std, balance_classes=False,
        )
        
        test_ds = GaitDataset(
            test_paths, test_labels, class_to_idx,
            window_size=args.window_size, base_stride=args.stride, mode='test',
            global_mean=g_mean, global_std=g_std, balance_classes=False,
        )
        
        pin = (device.type == 'cuda')
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn_pad,
                                  num_workers=args.num_workers, pin_memory=pin,
                                  persistent_workers=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn_pad,
                                num_workers=args.num_workers, pin_memory=pin,
                                persistent_workers=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn_pad,
                                 num_workers=args.num_workers, pin_memory=pin,
                                 persistent_workers=False)
        
        # Class weights from PATIENT counts (not window counts)
        patient_counts = train_ds.get_patient_class_counts()
        class_weights = 1.0 / (patient_counts.astype(float) + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_classes
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        print(f"  Patient-level class weights: {dict(zip(classes, class_weights.numpy().round(2)))}")
        
        # Loss
        criterion = FocalLoss(weight=class_weights.to(device), gamma=args.focal_gamma, label_smoothing=0.02)
        
        # Model
        model = create_model(
            args.model_type, input_dim, num_classes,
            args.d_model, args.n_layers, args.dropout,
            use_prototype_head=args.use_prototype_head,
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {param_count:,}")
        
        # Optimizer + scheduler
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
        steps_per_epoch = len(train_loader)
        total_optim_steps = max(args.epochs * (steps_per_epoch // max(args.accum_steps, 1)), 2)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_optim_steps,
            pct_start=0.1, anneal_strategy='cos',
        )
        
        # Training loop
        best_macro_f1 = -1.0
        patience_counter = 0
        ckpt_path = os.path.join(args.output_dir, f'best_{fold_name}.pth')
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device,
                accum_steps=args.accum_steps, use_prototype=args.use_prototype_head,
            )
            
            val_results = evaluate(
                model, val_loader, criterion, device, num_classes,
                use_prototype=args.use_prototype_head,
            )
            
            selection_score = val_results['macro_f1']
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                      f"Val Loss: {val_results['loss']:.4f} Acc: {val_results['acc']:.3f} "
                      f"MF1: {val_results['macro_f1']:.3f} MinRecall: {val_results['min_recall']:.3f}")
            
            if selection_score > best_macro_f1:
                best_macro_f1 = selection_score
                patience_counter = 0
                config = make_config(
                    args.model_type, input_dim, num_classes,
                    args.d_model, args.n_layers,
                    dropout=args.dropout, use_prototype=args.use_prototype_head,
                )
                save_checkpoint(model, config, ckpt_path, extra={
                    'epoch': epoch, 'best_macro_f1': best_macro_f1,
                    'fold': fold_name,
                })
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Test with best checkpoint
        state_dict, _, _ = load_checkpoint(ckpt_path, device)
        model.load_state_dict(state_dict)
        
        test_results = evaluate(
            model, test_loader, criterion, device, num_classes,
            use_prototype=args.use_prototype_head,
        )
        
        print(f"\n  Fold {fold_idx+1} Test Results:")
        print(f"  Acc={test_results['acc']:.3f} WF1={test_results['weighted_f1']:.3f} "
              f"MF1={test_results['macro_f1']:.3f} MinRecall={test_results['min_recall']:.3f}")
        print(f"  Per-class recall: {dict(zip(classes, test_results['recalls'].round(3)))}")
        print(classification_report(
            test_results['labels'], test_results['preds'],
            labels=list(range(num_classes)), target_names=classes, zero_division=0))
        
        all_fold_results.append({
            'fold': fold_name,
            'acc': test_results['acc'],
            'weighted_f1': test_results['weighted_f1'],
            'macro_f1': test_results['macro_f1'],
            'min_recall': test_results['min_recall'],
            'recalls': test_results['recalls'],
            'labels': test_results['labels'],
            'preds': test_results['preds'],
        })
    
    # -- Aggregate Results --
    print(f"\n{'='*60}")
    print(f"  AGGREGATED RESULTS ({n_folds} folds)")
    print(f"{'='*60}")
    
    accs = [r['acc'] for r in all_fold_results]
    wf1s = [r['weighted_f1'] for r in all_fold_results]
    mf1s = [r['macro_f1'] for r in all_fold_results]
    min_recalls = [r['min_recall'] for r in all_fold_results]
    
    print(f"  Accuracy:      {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"  Weighted F1:   {np.mean(wf1s):.3f} +/- {np.std(wf1s):.3f}")
    print(f"  Macro F1:      {np.mean(mf1s):.3f} +/- {np.std(mf1s):.3f}")
    print(f"  Min Recall:    {np.mean(min_recalls):.3f} +/- {np.std(min_recalls):.3f}")
    
    # Per-class recall aggregation
    if all_fold_results:
        all_recalls = np.array([r['recalls'] for r in all_fold_results])
        _, all_labels_list, class_to_idx, classes = discover_dataset(args.data_dir)
        print(f"\n  Per-class recall (mean +/- std):")
        for c_idx, c_name in enumerate(classes):
            if c_idx < all_recalls.shape[1]:
                print(f"    {c_name:20s}: {np.mean(all_recalls[:, c_idx]):.3f} +/- {np.std(all_recalls[:, c_idx]):.3f}")
    
    # Pooled classification report
    all_labels_pooled = []
    all_preds_pooled = []
    for r in all_fold_results:
        all_labels_pooled.extend(r['labels'])
        all_preds_pooled.extend(r['preds'])
    
    if all_labels_pooled:
        print(f"\n  Pooled Classification Report (all folds):")
        print(classification_report(
            all_labels_pooled, all_preds_pooled,
            labels=list(range(num_classes)), target_names=classes, zero_division=0))
    
    print(f"\n  Checkpoints saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
