import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import sys

from gait_phase import assign_gait_phase_continuous
from feature_config import FeatureConfig

# Import EDA feature extraction functions
sys.path.append('EDA')
try:
    from EDA.eda_features import TIME_DOMAIN_FNS, _freq_features, _gait_cycle_feats, _GAIT_PHASE_AVAILABLE
except ImportError:
    # Fallback if EDA module not available
    TIME_DOMAIN_FNS = {}
    _GAIT_PHASE_AVAILABLE = False
    def _freq_features(x, fs=1000.0):
        return {'mean_freq': 0.0, 'spectral_entropy': 0.0, 'total_power': 0.0}
    def _gait_cycle_feats(ta_signal, ga_signal, fs):
        return {}

class GaitPathologyDataset(Dataset):
    """
    Enhanced Dataset for Gait Pathology Classification using time-series muscle activations.
    
    Supports both legacy mode (5 features) and enhanced mode (17+ features) with:
    - Advanced EMG feature extraction (time-domain, frequency-domain, gait cycle)
    - Sliding Window Segmentation with patient-level splitting
    - Feature caching for performance optimization
    - Dimensionality reduction integration
    - Global Normalization and Temporal Jittering
    """
    def __init__(self, file_paths, labels, class_to_idx, 
                 feature_config: Optional[FeatureConfig] = None,
                 alpha=1.0, beta=1.0, 
                 window_size=2000, base_stride=1000, mode='train', 
                 global_mean=None, global_std=None,
                 static_global_mean=None, static_global_std=None,
                 balance_classes=True, fs=1000.0):
        """
        Args:
            file_paths: List of absolute paths to CSV files.
            labels: List of integer labels corresponding to `file_paths`.
            class_to_idx: Dictionary mapping class name to integer.
            feature_config: Configuration for advanced features (None = legacy mode)
            alpha, beta: Coefficients for Torque physics feature.
            window_size: Length of each extracted sub-sequence.
            base_stride: How much to shift the window.
            mode: 'train', 'val', or 'test'. Used to enable temporal jittering and control stride.
            global_mean, global_std: Normalization statistics pre-calculated from train set.
            balance_classes: If true, adjusts stride dynamically per file to explicitly balance the dataset.
            fs: Sampling frequency for frequency-domain features.
        """
        # Initialize configuration (legacy mode if None)
        if feature_config is None:
            self.feature_config = FeatureConfig.create_legacy()
            self.legacy_mode = True
        else:
            self.feature_config = feature_config
            self.legacy_mode = feature_config.legacy_mode
        
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.mode = mode
        self.fs = fs
        
        self.global_mean = global_mean
        self.global_std = global_std
        self.static_global_mean = static_global_mean
        self.static_global_std = static_global_std
        
        self.classes = sorted(list(class_to_idx.keys()))
        self.num_classes = len(self.classes)
        
        # Initialize cache directory for enhanced features
        if not self.legacy_mode and self.feature_config.enable_feature_caching:
            os.makedirs(self.feature_config.cache_dir, exist_ok=True)
        
        # Feature extraction for base samples
        # In enhanced mode: features = temporal [seq_len, 5], static_features = [n_static]
        # In legacy mode:   features = temporal [seq_len, 5], static_features = zeros(0)
        self.base_samples = []
        for path, label in zip(file_paths, labels):
            if self.legacy_mode:
                features = self._extract_legacy_features(path)
                static_features = torch.zeros(0)
            else:
                features, static_features = self._load_or_compute_features(path, label)
            
            self.base_samples.append({
                'features': features,
                'static_features': static_features,
                'label': label,
                'length': features.size(0)
            })

        # 2. Extract dataset-wide Global Mean/Std from Training Data BEFORE splitting into windows!
        # Computed separately for temporal features and static features.
        if self.mode == 'train' and (self.global_mean is None or self.global_std is None):
            all_data = torch.cat([s['features'] for s in self.base_samples], dim=0)
            self.global_mean = torch.mean(all_data, dim=0, keepdim=True)
            self.global_std = torch.std(all_data, dim=0, keepdim=True)
            print(f"Computed Global Train Mean (temporal): {self.global_mean.squeeze().tolist()}")
            print(f"Computed Global Train Std  (temporal): {self.global_std.squeeze().tolist()}")
        
        # Static feature normalization stats (computed once across all files, not timesteps)
        if self.mode == 'train' and (self.static_global_mean is None or self.static_global_std is None):
            static_feats = [s['static_features'] for s in self.base_samples if s['static_features'].numel() > 0]
            if static_feats:
                all_static = torch.stack(static_feats, dim=0)  # [n_files, n_static]
                self.static_global_mean = torch.mean(all_static, dim=0)  # [n_static]
                self.static_global_std = torch.std(all_static, dim=0)    # [n_static]
                print(f"Computed Global Train Mean (static): {self.static_global_mean.tolist()}")
                print(f"Computed Global Train Std  (static): {self.static_global_std.tolist()}")
            else:
                self.static_global_mean = None
                self.static_global_std = None

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
                
            # Generate the window pointers (include static features per window)
            static_feat = sample['static_features']
            start_idx = 0
            while start_idx + self.window_size <= L:
                self.windows.append({
                    'parent_feat': feat,
                    'static_features': static_feat,
                    'start': start_idx,
                    'label': lbl
                })
                start_idx += active_stride
                
            # If sequence was shorter than window size, take it entirely and pad later
            if start_idx == 0:
                 self.windows.append({
                    'parent_feat': feat,
                    'static_features': static_feat,
                    'start': 0,
                    'label': lbl
                })

        print(f"Set '{self.mode}': Started with {len(self.base_samples)} Patient Files -> Expanded to {len(self.windows)} Windows.")

    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        window = self.windows[idx]
        parent = window['parent_feat']
        static_feat = window['static_features'].clone()
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
            
            # Additional Feature Augmentation: Random Scaling (temporal only)
            scale_factor = random.uniform(0.9, 1.1)
            window_feat = window_feat * scale_factor
        else:
            window_feat = parent[start:end]
            
        # --- Temporal Feature Normalization ---
        global_std_val = self.global_std if self.global_std is not None else 0.0
        window_feat = (window_feat - (self.global_mean if self.global_mean is not None else 0.0)) / (global_std_val + 1e-8)
        window_feat = torch.clamp(window_feat, min=-10.0, max=10.0)

        # --- Static Feature Normalization ---
        if static_feat.numel() > 0 and self.static_global_mean is not None:
            static_feat = (static_feat - self.static_global_mean) / (self.static_global_std + 1e-8)
            static_feat = torch.clamp(static_feat, min=-10.0, max=10.0)
            
        return window_feat, static_feat, torch.tensor(lbl, dtype=torch.long)

    def _extract_legacy_features(self, file_path: str) -> torch.Tensor:
        """Extract legacy 5-channel features (original pipeline)."""
        df = pd.read_csv(file_path, header=None)
        e_ago = df.iloc[:, 0].values  # TA (Tibialis Anterior) — col 0
        e_ant = df.iloc[:, 1].values  # GA (Gastrocnemius)     — col 1
        
        torque = self.alpha * e_ant - self.beta * e_ago
        stiffness = e_ant + e_ago
        
        # Gait Phase: TKEO + EMG burst detection → continuous [0, 100] per timestep
        try:
            gait_phase = assign_gait_phase_continuous(e_ago, e_ant)
        except Exception:
            # Graceful fallback if signal is too short or degenerate
            gait_phase = np.linspace(0.0, 100.0, len(e_ago), dtype=np.float32)
        
        # Stack all 5 channels: [e_ant, e_ago, torque, stiffness, gait_phase]
        features = np.column_stack((e_ant, e_ago, torque, stiffness, gait_phase))
        features = torch.tensor(features, dtype=torch.float32)
        
        # Clean corrupt data
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features

    def _get_cache_path(self, file_path: str) -> Optional[str]:
        """Get cache file path for a given input file."""
        if not self.feature_config.enable_feature_caching:
            return None
        
        # Create unique cache filename based on input file and config
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        config_hash = self.feature_config.get_config_hash()
        cache_filename = f"{os.path.basename(file_path)}_{file_hash}_{config_hash}.npz"
        return os.path.join(self.feature_config.cache_dir, cache_filename)
    
    def _load_or_compute_features(self, file_path: str, label: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load cached features or compute and cache new ones.
        Returns: (temporal_features [seq_len, 5], static_features [n_static])
        """
        cache_path = self._get_cache_path(file_path)
        
        # Try to load from cache first
        if cache_path and os.path.exists(cache_path):
            try:
                cached_data = np.load(cache_path)
                # Support both old single-array cache and new split cache
                if 'temporal' in cached_data and 'static' in cached_data:
                    temporal = torch.tensor(cached_data['temporal'], dtype=torch.float32)
                    static = torch.tensor(cached_data['static'], dtype=torch.float32)
                    return temporal, static
                else:
                    # Old cache format — discard and recompute
                    print(f"Old cache format detected for {file_path}. Recomputing...")
            except Exception as e:
                print(f"Cache load failed for {file_path}: {e}. Recomputing...")
        
        # Compute features from scratch
        temporal_features, static_features = self._extract_enhanced_features(file_path)
        
        # Cache the computed features
        if cache_path:
            try:
                np.savez_compressed(
                    cache_path,
                    temporal=temporal_features.numpy(),
                    static=static_features.numpy()
                )
            except Exception as e:
                print(f"Cache save failed for {file_path}: {e}")
        
        return temporal_features, static_features
    
    def _extract_enhanced_features(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract enhanced features from a single CSV file.
        
        Returns:
            temporal_features: [seq_len, 5] tensor of time-varying base features.
            static_features:   [n_advanced] tensor of whole-sequence statistical scalars.
        """
        # Load raw EMG data
        df = pd.read_csv(file_path, header=None)
        e_ago = df.iloc[:, 0].values.astype(np.float64)  # TA (Tibialis Anterior)
        e_ant = df.iloc[:, 1].values.astype(np.float64)  # GA (Gastrocnemius)
        
        # Clean corrupt data
        e_ant = np.nan_to_num(e_ant, nan=0.0, posinf=0.0, neginf=0.0)
        e_ago = np.nan_to_num(e_ago, nan=0.0, posinf=0.0, neginf=0.0)
        
        sequence_length = len(e_ant)
        
        # --- Temporal Stream: base features that vary per timestep ---
        if self.feature_config.include_base_features:
            torque = self.alpha * e_ant - self.beta * e_ago
            stiffness = e_ant + e_ago
            try:
                gait_phase = assign_gait_phase_continuous(e_ago, e_ant, self.fs)
            except Exception:
                gait_phase = np.linspace(0.0, 100.0, sequence_length, dtype=np.float32)
            temporal_np = np.column_stack((e_ant, e_ago, torque, stiffness, gait_phase))
        else:
            # Fallback: raw signals only
            temporal_np = np.column_stack((e_ant, e_ago))
        
        temporal_features = torch.tensor(temporal_np, dtype=torch.float32)
        temporal_features = torch.nan_to_num(temporal_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # --- Static Stream: whole-sequence statistical scalars ---
        # These are NOT broadcast. They are stored as a 1-D vector and routed
        # directly into the classifier head, bypassing the sequence model entirely.
        static_np = self._extract_advanced_features_per_sequence(e_ant, e_ago)
        static_features = torch.tensor(static_np, dtype=torch.float32)
        static_features = torch.nan_to_num(static_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return temporal_features, static_features
    
    def _extract_advanced_features_per_sequence(self, e_ant: np.ndarray, e_ago: np.ndarray) -> np.ndarray:
        """Extract advanced features for an entire sequence (to be broadcast per timestep)."""
        features = []
        
        # Time-domain features
        if self.feature_config.include_time_domain:
            channels = {'ant': e_ant, 'ago': e_ago}
            for ch_name, ch_data in channels.items():
                for feat_name in self.feature_config.time_domain_features:
                    if feat_name.endswith(f'_{ch_name}'):
                        base_feat_name = feat_name.replace(f'_{ch_name}', '')
                        if base_feat_name in TIME_DOMAIN_FNS:
                            feat_value = TIME_DOMAIN_FNS[base_feat_name](ch_data)
                            features.append(feat_value)
        
        # Frequency-domain features
        if self.feature_config.include_freq_domain:
            channels = {'ant': e_ant, 'ago': e_ago}
            for ch_name, ch_data in channels.items():
                freq_feats = _freq_features(ch_data, fs=self.fs)
                for feat_name in self.feature_config.freq_domain_features:
                    if feat_name.endswith(f'_{ch_name}'):
                        base_feat_name = feat_name.replace(f'_{ch_name}', '')
                        if base_feat_name in freq_feats:
                            features.append(freq_feats[base_feat_name])
        
        # Gait cycle features
        if self.feature_config.include_gait_cycle and _GAIT_PHASE_AVAILABLE:
            try:
                gait_feats = _gait_cycle_feats(e_ago, e_ant, self.fs)
                for feat_name in self.feature_config.gait_cycle_features:
                    gait_key = feat_name.replace('gp_', '')
                    if gait_key in gait_feats and isinstance(gait_feats[gait_key], (int, float)):
                        features.append(float(gait_feats[gait_key]))
            except Exception:
                # Add zeros for gait cycle features if extraction fails
                features.extend([0.0] * len(self.feature_config.gait_cycle_features))
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names in the order they appear in the feature vector."""
        if self.legacy_mode:
            return ['e_ant', 'e_ago', 'torque', 'stiffness', 'gait_phase']
        return self.feature_config.get_enabled_features()
    
    def get_feature_count(self) -> int:
        """Get number of TEMPORAL features per timestep (used as model input_dim)."""
        if self.legacy_mode:
            return 5
        # Only count the base (temporal) features — static features bypass the sequence model
        return 5 if self.feature_config.include_base_features else 0
    
    def get_static_feature_count(self) -> int:
        """Get number of STATIC features (used as classifier head auxiliary input)."""
        if self.legacy_mode or not self.base_samples:
            return 0
        return int(self.base_samples[0]['static_features'].shape[0])

def collate_fn_pad(batch):
    """
    Collate function handles variable sequence lengths.
    Returns a 4-tuple: (padded_sequences, attention_masks, static_features, labels)
    Static features are per-sample scalars; they do not need sequence padding.
    """
    sequences, static_feats, labels = zip(*batch)
    
    padded_sequences = pad_sequence(list(sequences), batch_first=True, padding_value=0.0)
    
    attention_masks = torch.zeros(padded_sequences.shape[0], padded_sequences.shape[1], dtype=torch.float32)
    for i, seq in enumerate(sequences):
        attention_masks[i, :len(seq)] = 1.0
    
    # Stack static features → [batch, n_static]. Works for empty (legacy) tensors too.
    static_batch = torch.stack(list(static_feats), dim=0)
    
    labels = torch.stack(labels)
    
    return padded_sequences, attention_masks, static_batch, labels

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
            # 2 files: train on paths[0], validate and test on paths[1].
            # CRITICAL: val must NEVER be the same file as train to avoid data leakage.
            # Sharing val and test is imperfect but honest — the model has not seen paths[1].
            train_paths.append(paths[0])
            train_labels.append(lbl)
            val_paths.append(paths[1])
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

def create_dataloaders(data_dir, batch_size=32, window_size=2000, base_stride=1000, random_seed=42, 
                      feature_config: Optional[FeatureConfig] = None):
    """
    Utility function to create separate train, validation, and test dataloaders
    using Sliding Windows and Strict Patient-Level Splitting.
    
    Args:
        feature_config: Configuration for enhanced features (None = legacy mode)
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
        feature_config=feature_config,
        window_size=window_size, base_stride=base_stride, mode='train',
        balance_classes=True
    )
    
    # Extract calculated normalization stats to freeze and pass to val/test
    # Both temporal and static norms are propagated so val/test are normalised identically.
    g_mean = train_dataset.global_mean
    g_std = train_dataset.global_std
    s_mean = train_dataset.static_global_mean
    s_std = train_dataset.static_global_std
    
    val_dataset = GaitPathologyDataset(
        val_p, val_l, class_to_idx, 
        feature_config=feature_config,
        window_size=window_size, base_stride=base_stride, mode='val',
        global_mean=g_mean, global_std=g_std,
        static_global_mean=s_mean, static_global_std=s_std,
        balance_classes=False
    )
    
    test_dataset = GaitPathologyDataset(
        test_p, test_l, class_to_idx, 
        feature_config=feature_config,
        window_size=window_size, base_stride=base_stride, mode='test',
        global_mean=g_mean, global_std=g_std,
        static_global_mean=s_mean, static_global_std=s_std,
        balance_classes=False
    )
    
    # Assign standard properties expected by train.py/evaluate.py
    train_dataset.classes = classes
    
    # 3. Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_pad)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pad)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pad)
    
    return train_loader, val_loader, test_loader, train_dataset

