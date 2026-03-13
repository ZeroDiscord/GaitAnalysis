import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random

class GaitPathologyDataset(Dataset):
    """
    Dataset for Gait Pathology Classification using time-series muscle activations.
    Implements Sliding Window Segmentation to multiply samples natively without data leakage,
    Global Normalization, and Temporal Jittering.
    """
    def __init__(self, file_paths, labels, class_to_idx, alpha=1.0, beta=1.0, 
                 window_size=2000, base_stride=1000, mode='train', 
                 global_mean=None, global_std=None, balance_classes=True):
        """
        Args:
            file_paths: List of absolute paths to CSV files.
            labels: List of integer labels corresponding to `file_paths`.
            class_to_idx: Dictionary mapping class name to integer.
            alpha, beta: Coefficients for Torque physics feature.
            window_size: Length of each extracted sub-sequence.
            base_stride: How much to shift the window.
            mode: 'train', 'val', or 'test'. Used to enable temporal jittering and control stride.
            global_mean, global_std: Normalization statistics pre-calculated from train set.
            balance_classes: If true, adjusts stride dynamically per file to explicitly balance the dataset.
        """
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.mode = mode
        
        self.global_mean = global_mean
        self.global_std = global_std
        
        self.classes = sorted(list(class_to_idx.keys()))
        self.num_classes = len(self.classes)
        
        # 1. First, load all base patient files and compute physics features immediately
        self.base_samples = []
        for path, label in zip(file_paths, labels):
            df = pd.read_csv(path, header=None)
            e_ago = df.iloc[:, 0].values  # Agonist is column 0
            e_ant = df.iloc[:, 1].values  # Antagonist is column 1
            
            torque = self.alpha * e_ant - self.beta * e_ago
            stiffness = e_ant + e_ago
            
            features = np.column_stack((e_ant, e_ago, torque, stiffness))
            features = torch.tensor(features, dtype=torch.float32)
            
            # Clean corrupt data
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.base_samples.append({
                'features': features,
                'label': label,
                'length': features.size(0)
            })

        # 2. Extract dataset-wide Global Mean/Std from Training Data BEFORE splitting into windows!
        if self.mode == 'train' and (self.global_mean is None or self.global_std is None):
            all_data = torch.cat([s['features'] for s in self.base_samples], dim=0)
            self.global_mean = torch.mean(all_data, dim=0, keepdim=True)
            self.global_std = torch.std(all_data, dim=0, keepdim=True)
            print(f"Computed Global Train Mean: {self.global_mean.squeeze().tolist()}")
            print(f"Computed Global Train Std:  {self.global_std.squeeze().tolist()}")

        # 3. Sliding Window Segmentation (With Native Balancing)
        self.windows = []
        
        # Calculate dynamic strides if balancing
        class_counts = {lbl: labels.count(lbl) for lbl in range(self.num_classes)}
        max_class_count = max(class_counts.values()) if class_counts else 1
        
        for sample in self.base_samples:
            feat = sample['features']
            lbl = sample['label']
            L = feat.size(0)
            
            # Dynamic stride allocation:
            # If a class has fewer files (e.g. 1 file vs 3 files), we shrink its stride 
            # so the 1 file generates 3x as many windows, natively balancing the dataset perfectly!
            if balance_classes and self.mode == 'train':
                multiplier = max_class_count / max(class_counts[lbl], 1)
                active_stride = max(int(base_stride / multiplier), 10) # never go lower than 10
            else:
                active_stride = base_stride
                
            # Generate the window pointers
            start_idx = 0
            while start_idx + self.window_size <= L:
                self.windows.append({
                    'parent_feat': feat,
                    'start': start_idx,
                    'label': lbl
                })
                start_idx += active_stride
                
            # If sequence was shorter than window size, take it entirely and pad later
            if start_idx == 0:
                 self.windows.append({
                    'parent_feat': feat,
                    'start': 0,
                    'label': lbl
                })

        print(f"Set '{self.mode}': Started with {len(self.base_samples)} Patient Files -> Expanded to {len(self.windows)} Windows.")

    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        window = self.windows[idx]
        parent = window['parent_feat']
        start = window['start']
        lbl = window['label']
        
        end = start + self.window_size
        
        # Temporal Jittering (Data Augmentation for robust Phase Invariance)
        if self.mode == 'train':
            # Randomly shift the window boundary by up to +/- 5% of the window size
            jitter = int(self.window_size * 0.05)
            shift = random.randint(-jitter, jitter)
            
            # Ensure we don't index out of bounds
            new_start = max(0, start + shift)
            new_end = new_start + self.window_size
            
            # If shift pushes us past the end, pull it back
            if new_end > parent.size(0):
                new_end = parent.size(0)
                new_start = max(0, new_end - self.window_size)
                
            window_feat = parent[new_start:new_end]
            
            # Additional Feature Augmentation: Random Scaling
            scale_factor = random.uniform(0.9, 1.1)
            window_feat = window_feat * scale_factor
        else:
            window_feat = parent[start:end]
            
        # Global Normalization (Applied IDENTICALLY to Val/Test using Training Stats)
        window_feat = (window_feat - self.global_mean) / (self.global_std + 1e-8)
        
        # Hard clamp extreme outliers (prevents nan loss spikes from exploding gradients)
        window_feat = torch.clamp(window_feat, min=-10.0, max=10.0)
            
        return window_feat, torch.tensor(lbl, dtype=torch.long)

def collate_fn_pad(batch):
    """
    Collate function handles variable sequence lengths.
    Since we use sliding windows, most sequences are perfectly identical in length,
    but edge cases/short patient files might need padding.
    """
    sequences, labels = zip(*batch)
    
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    attention_masks = torch.zeros(padded_sequences.shape[0], padded_sequences.shape[1], dtype=torch.float32)
    for i, seq in enumerate(sequences):
        attention_masks[i, :len(seq)] = 1.0
        
    labels = torch.stack(labels)
    
    return padded_sequences, attention_masks, labels

def manual_stratified_split(file_paths, labels, val_split=0.2, test_split=0.1, random_seed=42):
    """
    Given a list of file paths and their labels, forcefully allocate exactly 1 file per class 
    to the Test Set, 1 file per class to the Val set (if possible), and the rest to Train.
    This protects against `sklearn` failing when `n_samples < 3`.
    """
    random.seed(random_seed)
    
    # Group by class
    from collections import defaultdict
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
            # Extreme case: Warning, only 1 file exists for this class.
            # We MUST put it in Train, otherwise the model never learns it.
            # But we copy it to Test so evaluation math doesn't crash on missing classes.
            print(f"WARNING: Class {lbl} only has 1 sample! Duplicating to Train/Val/Test to prevent crash.")
            train_paths.append(paths[0])
            train_labels.append(lbl)
            val_paths.append(paths[0])
            val_labels.append(lbl)
            test_paths.append(paths[0])
            test_labels.append(lbl)
            
        elif n == 2:
            # 2 files: 1 train, 1 test (copy train to val)
            train_paths.append(paths[0])
            train_labels.append(lbl)
            val_paths.append(paths[0])
            val_labels.append(lbl)
            test_paths.append(paths[1])
            test_labels.append(lbl)
            
        else:
            # Normal distribution manually enforced
            test_paths.append(paths.pop())
            test_labels.append(lbl)
            
            val_paths.append(paths.pop())
            val_labels.append(lbl)
            
            for p in paths:
                train_paths.append(p)
                train_labels.append(lbl)
                
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def create_dataloaders(data_dir, batch_size=32, window_size=2000, base_stride=1000, random_seed=42):
    """
    Utility function to create separate train, validation, and test dataloaders
    using Sliding Windows and Strict Patient-Level Splitting.
    """
    raw_folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    discovered_classes = set()
    for d in raw_folders:
        if '_' in d:
            discovered_classes.add(d.split('_', 1)[1])
            
    classes = sorted(list(discovered_classes))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    all_paths = []
    all_labels = []
    for d in raw_folders:
        if '_' not in d: continue
        cls_name = d.split('_', 1)[1]
        cls_dir = os.path.join(data_dir, d)
        for file_name in sorted(os.listdir(cls_dir)):
            if file_name.endswith('.csv'):
                all_paths.append(os.path.join(cls_dir, file_name))
                all_labels.append(class_to_idx[cls_name])
                
    if not all_paths:
        raise ValueError(f"No valid CSV files found in {data_dir}.")

    # 1. Strict Patient-Level Split BEFORE Windowing
    # This prevents pieces of the exact same walk from appearing in both Train and Test.
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = manual_stratified_split(
        all_paths, all_labels, random_seed=random_seed
    )
    
    # 2. Initialize datasets. Train set calculates Global Norm natively!
    train_dataset = GaitPathologyDataset(
        train_p, train_l, class_to_idx, 
        window_size=window_size, base_stride=base_stride, mode='train',
        balance_classes=True
    )
    
    # Extract calculated norm to freeze and pass to val/test
    g_mean = train_dataset.global_mean
    g_std = train_dataset.global_std
    
    val_dataset = GaitPathologyDataset(
        val_p, val_l, class_to_idx, 
        window_size=window_size, base_stride=base_stride, mode='val',
        global_mean=g_mean, global_std=g_std, balance_classes=False
    )
    
    test_dataset = GaitPathologyDataset(
        test_p, test_l, class_to_idx, 
        window_size=window_size, base_stride=base_stride, mode='test',
        global_mean=g_mean, global_std=g_std, balance_classes=False
    )
    
    # Assign standard properties expected by train.py/evaluate.py
    train_dataset.classes = classes
    
    # 3. Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_pad)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pad)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pad)
    
    return train_loader, val_loader, test_loader, train_dataset