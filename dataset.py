"""
dataset.py — Gait Pathology Dataset with Per-Window Features
=============================================================
Key features:
  1. Per-window feature extraction (not file-level tiled constants)
  2. Advanced augmentation pipeline integration (augmentations.py)
  3. Proper channel-aware augmentation (never scales gait_phase)
  4. Correct class weights from patient/file counts, not window counts
  5. LOPO and k-fold patient-level splitting utilities
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
from collections import defaultdict
from typing import Optional, List, Tuple, Dict

# Import gait phase module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from gait_phase import assign_gait_phase_continuous
except ImportError:
    def assign_gait_phase_continuous(ta, ga, fs=1000.0):
        return np.linspace(0, 100, len(ta), dtype=np.float32)

from augmentations import AugmentationPipeline


class GaitDataset(Dataset):
    """
    Gait Pathology Dataset with per-window feature extraction.
    
    Features:
      - 5 per-timestep features: [e_ant, e_ago, torque, stiffness, gait_phase]
      - Features computed PER WINDOW (not tiled file-level constants)
      - Augmentation pipeline applied BEFORE normalization
      - Only amplitude channels are augmented (never gait_phase)
      - Returns (features, label) with consistent shapes
    """
    
    def __init__(self, file_paths: List[str], labels: List[int],
                 class_to_idx: Dict[str, int],
                 window_size: int = 2000, base_stride: int = 1000,
                 mode: str = 'train',
                 global_mean: Optional[torch.Tensor] = None,
                 global_std: Optional[torch.Tensor] = None,
                 balance_classes: bool = True,
                 augmentation: Optional[AugmentationPipeline] = None,
                 fs: float = 1000.0):
        """
        Args:
            file_paths: List of absolute paths to CSV files
            labels: List of integer labels
            class_to_idx: Dict mapping class name -> int
            window_size: Samples per window (2000 = 2 seconds at 1kHz)
            base_stride: Stride between windows
            mode: 'train', 'val', or 'test'
            global_mean, global_std: Normalization stats from train set
            balance_classes: Dynamic stride to equalize class windows
            augmentation: AugmentationPipeline instance (None = no augmentation)
            fs: Sampling frequency (Hz)
        """
        self.window_size = window_size
        self.mode = mode
        self.fs = fs
        self.augmentation = augmentation
        self.global_mean = global_mean
        self.global_std = global_std
        self.classes = sorted(list(class_to_idx.keys()))
        self.num_classes = len(self.classes)
        self.input_dim = 5  # [e_ant, e_ago, torque, stiffness, gait_phase]
        
        # Pre-compute all features per file ONCE to avoid CPU bottleneck in Dataloader
        # and to ensure gait_phase spline has access to full file context for accurate peaks.
        self.base_samples = []
        for path, label in zip(file_paths, labels):
            df = pd.read_csv(path, header=None)
            e_ago = df.iloc[:, 0].values.astype(np.float64)  # TA
            e_ant = df.iloc[:, 1].values.astype(np.float64)  # GA
            e_ago = np.nan_to_num(e_ago, nan=0.0, posinf=0.0, neginf=0.0)
            e_ant = np.nan_to_num(e_ant, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Pre-compute
            torque = e_ant - e_ago
            stiffness = e_ant + e_ago
            try:
                gait_phase, _, _ = assign_gait_phase_continuous(e_ago, e_ant, self.fs)
            except Exception:
                gait_phase = np.linspace(0, 100, len(e_ago), dtype=np.float32)
            
            features = np.column_stack([e_ant, e_ago, torque, stiffness, gait_phase]).astype(np.float32)
            
            self.base_samples.append({
                'features': features,
                'label': label,
                'length': len(features),
            })
        
        # Sliding window segmentation with class balancing
        self.windows = []
        class_counts = defaultdict(int)
        for s in self.base_samples:
            class_counts[s['label']] += 1
        max_class_count = max(class_counts.values()) if class_counts else 1
        
        for idx, sample in enumerate(self.base_samples):
            L = sample['length']
            lbl = sample['label']
            
            if balance_classes and mode == 'train':
                multiplier = max_class_count / max(class_counts[lbl], 1)
                active_stride = max(int(base_stride / multiplier), 10)
            else:
                active_stride = base_stride
            
            start = 0
            while start + window_size <= L:
                self.windows.append({
                    'sample_idx': idx,
                    'start': start,
                    'label': lbl,
                })
                start += active_stride
            
            # Short files: use entire signal
            if start == 0:
                self.windows.append({
                    'sample_idx': idx,
                    'start': 0,
                    'label': lbl,
                })
        
        # Compute global normalization from training windows
        if mode == 'train' and (global_mean is None or global_std is None):
            self._compute_global_stats()
        
        print(f"Set '{mode}': {len(self.base_samples)} files -> {len(self.windows)} windows")
    
    def _extract_window_features(self, sample_idx: int, start: int, end: int) -> np.ndarray:
        """
        Get 5-feature vector per timestep from a pre-computed window.
        Returns: (T, 5) array: [e_ant, e_ago, torque, stiffness, gait_phase]
        """
        return self.base_samples[sample_idx]['features'][start:end].copy()
    
    def _compute_global_stats(self):
        """Compute global mean/std from training windows (sampled for efficiency)."""
        all_features = []
        # Sample up to 200 windows to avoid O(n^2) memory for large datasets
        sample_indices = list(range(min(200, len(self.windows))))
        
        for idx in sample_indices:
            win = self.windows[idx]
            sample = self.base_samples[win['sample_idx']]
            start = win['start']
            end = min(start + self.window_size, sample['length'])
            
            features = self._extract_window_features(win['sample_idx'], start, end)
            all_features.append(features)
        
        all_data = np.concatenate(all_features, axis=0)
        self.global_mean = torch.tensor(np.mean(all_data, axis=0, keepdims=True), dtype=torch.float32)
        self.global_std = torch.tensor(np.std(all_data, axis=0, keepdims=True), dtype=torch.float32)
        self.input_dim = all_data.shape[1]
        
        print(f"Global Train Mean: {self.global_mean.squeeze().tolist()}")
        print(f"Global Train Std:  {self.global_std.squeeze().tolist()}")
    
    def get_feature_count(self) -> int:
        return self.input_dim
    
    def get_patient_class_counts(self) -> np.ndarray:
        """Return class counts at the PATIENT/FILE level (not window level)."""
        counts = np.zeros(self.num_classes, dtype=np.int32)
        for s in self.base_samples:
            counts[s['label']] += 1
        return counts
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        win = self.windows[idx]
        sample = self.base_samples[win['sample_idx']]
        start = win['start']
        lbl = win['label']
        
        # Temporal jitter for training
        if self.mode == 'train':
            jitter = int(self.window_size * 0.05)
            shift = random.randint(-jitter, jitter)
            new_start = max(0, start + shift)
            end = min(new_start + self.window_size, sample['length'])
            new_start = max(0, end - self.window_size)
        else:
            new_start = start
            end = min(new_start + self.window_size, sample['length'])
        
        # Extract per-window features from pre-computed array
        features = self._extract_window_features(win['sample_idx'], new_start, end)  # (T, 5)
        
        # Apply augmentation BEFORE normalization (on raw amplitude scale)
        if self.mode == 'train' and self.augmentation is not None:
            features = self.augmentation(features)
        
        # Convert to tensor and normalize
        features = torch.tensor(features, dtype=torch.float32)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.global_mean is not None and self.global_std is not None:
            features = (features - self.global_mean) / (self.global_std + 1e-8)
        
        features = torch.clamp(features, -10.0, 10.0)
        
        return features, torch.tensor(lbl, dtype=torch.long)


def collate_fn_pad(batch):
    """Pad variable-length sequences and create attention masks."""
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    masks = torch.zeros(padded.shape[0], padded.shape[1], dtype=torch.float32)
    for i, seq in enumerate(sequences):
        masks[i, :len(seq)] = 1.0
    labels = torch.stack(labels)
    return padded, masks, labels


# ---------------------------------------------------------------------------
# Patient-level data splitting utilities
# ---------------------------------------------------------------------------

def discover_dataset(data_dir: str) -> Tuple[List[str], List[int], Dict[str, int], List[str]]:
    """
    Scan data_dir for class folders and CSV files.
    Returns (file_paths, labels, class_to_idx, class_names).
    """
    raw_folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    discovered_classes = set()
    for d in raw_folders:
        if '_' in d:
            discovered_classes.add(d.split('_', 1)[1])
    
    classes = sorted(list(discovered_classes))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    all_paths, all_labels = [], []
    for d in raw_folders:
        if '_' not in d:
            continue
        cls_name = d.split('_', 1)[1]
        cls_dir = os.path.join(data_dir, d)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.endswith('.csv'):
                all_paths.append(os.path.join(cls_dir, fname))
                all_labels.append(class_to_idx[cls_name])
    
    if not all_paths:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    return all_paths, all_labels, class_to_idx, classes


def patient_level_lopo_splits(data_dir: str):
    """
    Leave-One-Patient-Out cross-validation splits.
    
    Each patient takes one turn as the test set.
    Val = 1 patient per class from remaining (stratified).
    Train = everything else.
    
    Yields: (train_paths, train_labels, val_paths, val_labels,
             test_path, test_label, class_to_idx, classes, fold_name)
    """
    all_paths, all_labels, class_to_idx, classes = discover_dataset(data_dir)
    
    # Simple LOPO: each file is one "patient" (one fold per file)
    for test_idx in range(len(all_paths)):
        test_path = all_paths[test_idx]
        test_label = all_labels[test_idx]
        test_name = os.path.basename(test_path)
        
        # Remaining files
        remaining_paths = [p for i, p in enumerate(all_paths) if i != test_idx]
        remaining_labels = [l for i, l in enumerate(all_labels) if i != test_idx]
        
        # Stratified val: 1 file per class from remaining
        val_paths, val_labels = [], []
        train_paths, train_labels = [], []
        
        # Group remaining by class
        class_files = defaultdict(list)
        for p, l in zip(remaining_paths, remaining_labels):
            class_files[l].append(p)
        
        for cls_idx, files in class_files.items():
            if len(files) >= 2:
                val_paths.append(files[0])
                val_labels.append(cls_idx)
                for f in files[1:]:
                    train_paths.append(f)
                    train_labels.append(cls_idx)
            else:
                # Only 1 file: put in train, duplicate to val
                train_paths.extend(files)
                train_labels.extend([cls_idx] * len(files))
                val_paths.extend(files)
                val_labels.extend([cls_idx] * len(files))
        
        yield (train_paths, train_labels,
               val_paths, val_labels,
               [test_path], [test_label],
               class_to_idx, classes, test_name)


def patient_level_kfold_splits(data_dir: str, n_splits: int = 5, random_seed: int = 42):
    """
    Stratified k-fold patient-level splits.
    
    Yields: (train_paths, train_labels, val_paths, val_labels,
             test_paths, test_labels, class_to_idx, classes, fold_name)
    """
    all_paths, all_labels, class_to_idx, classes = discover_dataset(data_dir)
    
    # Group by class
    class_files = defaultdict(list)
    for p, l in zip(all_paths, all_labels):
        class_files[l].append(p)
    
    rng = random.Random(random_seed)
    for cls_idx in class_files:
        rng.shuffle(class_files[cls_idx])
    
    for fold in range(n_splits):
        train_paths, train_labels = [], []
        val_paths, val_labels = [], []
        test_paths, test_labels = [], []
        
        for cls_idx, files in class_files.items():
            n = len(files)
            if n <= 2:
                # Too few: duplicate
                train_paths.extend(files)
                train_labels.extend([cls_idx] * n)
                val_paths.append(files[0])
                val_labels.append(cls_idx)
                test_paths.append(files[-1])
                test_labels.append(cls_idx)
            else:
                test_idx = fold % n
                val_idx = (fold + 1) % n
                test_paths.append(files[test_idx])
                test_labels.append(cls_idx)
                val_paths.append(files[val_idx])
                val_labels.append(cls_idx)
                for i, f in enumerate(files):
                    if i != test_idx and i != val_idx:
                        train_paths.append(f)
                        train_labels.append(cls_idx)
        
        yield (train_paths, train_labels,
               val_paths, val_labels,
               test_paths, test_labels,
               class_to_idx, classes, f"fold_{fold}")


def manual_stratified_split(file_paths, labels, random_seed=42):
    """
    Allocate exactly 1 file per class to Test, 1 to Val, rest to Train.
    Handles tiny cohorts (1-2 files per class) gracefully.
    """
    random.seed(random_seed)
    class_groups = defaultdict(list)
    for path, lbl in zip(file_paths, labels):
        class_groups[lbl].append(path)

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for lbl, paths in class_groups.items():
        random.shuffle(paths)
        n = len(paths)
        if n == 1:
            train_paths.append(paths[0]); train_labels.append(lbl)
            val_paths.append(paths[0]);   val_labels.append(lbl)
            test_paths.append(paths[0]);  test_labels.append(lbl)
        elif n == 2:
            train_paths.append(paths[0]); train_labels.append(lbl)
            val_paths.append(paths[0]);   val_labels.append(lbl)
            test_paths.append(paths[1]);  test_labels.append(lbl)
        else:
            test_paths.append(paths.pop());  test_labels.append(lbl)
            val_paths.append(paths.pop());   val_labels.append(lbl)
            for p in paths:
                train_paths.append(p); train_labels.append(lbl)

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def create_dataloaders(data_dir: str, batch_size: int = 4,
                       window_size: int = 2000, base_stride: int = 1000,
                       random_seed: int = 42, use_augmentation: bool = True):
    """
    Create train/val/test dataloaders with stratified patient-level splitting.
    """
    all_paths, all_labels, class_to_idx, classes = discover_dataset(data_dir)
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = manual_stratified_split(
        all_paths, all_labels, random_seed=random_seed
    )

    aug = AugmentationPipeline.default_emg() if use_augmentation else None

    train_dataset = GaitDataset(
        train_p, train_l, class_to_idx,
        window_size=window_size, base_stride=base_stride, mode='train',
        balance_classes=True, augmentation=aug,
    )
    g_mean, g_std = train_dataset.global_mean, train_dataset.global_std

    val_dataset = GaitDataset(
        val_p, val_l, class_to_idx,
        window_size=window_size, base_stride=base_stride, mode='val',
        global_mean=g_mean, global_std=g_std, balance_classes=False,
    )
    test_dataset = GaitDataset(
        test_p, test_l, class_to_idx,
        window_size=window_size, base_stride=base_stride, mode='test',
        global_mean=g_mean, global_std=g_std, balance_classes=False,
    )

    train_dataset.classes = classes

    generator = torch.Generator()
    generator.manual_seed(random_seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_pad, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_pad)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn_pad)

    return train_loader, val_loader, test_loader, train_dataset
