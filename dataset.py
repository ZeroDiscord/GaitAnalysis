import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class GaitPathologyDataset(Dataset):
    """
    Dataset for Gait Pathology Classification using time-series muscle activations.
    Computes physics-informed features (Torque, Joint Stiffness) on the fly.
    """
    def __init__(self, data_dir, alpha=1.0, beta=1.0, transform=None, downsample=1):
        """
        Args:
            data_dir (str): Path to the root 'Datasets/' directory containing class folders.
            alpha (float): Coefficient for antagonist muscle activation in torque.
            beta (float): Coefficient for agonist muscle activation in torque.
            transform (callable, optional): Optional transform to be applied on a sample.
            downsample (int): Downsample factor to reduce very long sequences.
        """
        self.data_dir = data_dir
        self.alpha = alpha
        self.beta = beta
        self.transform = transform
        self.downsample = downsample
        
        self.samples = []
        self.labels = []
        
        # Discover base classes by stripping numeric prefixes (e.g., '1_Healthy' -> 'Healthy')
        raw_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        discovered_classes = set()
        for d in raw_folders:
            if '_' in d:
                cls_name = d.split('_', 1)[1]
                discovered_classes.add(cls_name)
                
        self.classes = sorted(list(discovered_classes))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Traverse directory to find CSV files
        for d in raw_folders:
            if '_' not in d:
                continue # Skip folders like 'raw data'
                
            cls_name = d.split('_', 1)[1]
            cls_dir = os.path.join(data_dir, d)
            
            for file_name in os.listdir(cls_dir):
                if file_name.endswith('.csv'):
                    self.samples.append(os.path.join(cls_dir, file_name))
                    self.labels.append(self.class_to_idx[cls_name])
                    
    def get_class_weights(self):
        import collections
        class_counts = collections.Counter(self.labels)
        total = sum(class_counts.values())
        weights = []
        for i in range(len(self.classes)):
            weight = total / (len(self.classes) * max(class_counts[i], 1))
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]
        
        # Read the CSV file without header since actual data has no headers
        df = pd.read_csv(file_path, header=None)
        
        # Ensure we take the first two columns as E_ant and E_ago and apply downsampling
        e_ant = df.iloc[::self.downsample, 0].values
        e_ago = df.iloc[::self.downsample, 1].values
            
        # Physics Feature Computation
        # Torque: τ(t) = α * E_ant(t) − β * E_ago(t)
        torque = self.alpha * e_ant - self.beta * e_ago
        
        # Joint stiffness: k(t) = E_ant(t) + E_ago(t)
        stiffness = e_ant + e_ago
        
        # Stack features: [E_ant, E_ago, τ, k]
        # Shape: (sequence_length, 4)
        features = np.column_stack((e_ant, e_ago, torque, stiffness))
        
        # Convert to tensor and float32 type
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Basic normalization (z-score along the time dimension per feature)
        # Note: In a real scenario, global dataset statistics are better, but per-sequence works as a baseline
        mean = torch.mean(features_tensor, dim=0, keepdim=True)
        std = torch.std(features_tensor, dim=0, keepdim=True)
        # Add epsilon to prevent division by zero
        features_tensor = (features_tensor - mean) / (std + 1e-8)
        
        if self.transform:
            features_tensor = self.transform(features_tensor)
            
        return features_tensor, torch.tensor(label, dtype=torch.long)

def collate_fn_pad(batch):
    """
    Collate function for DataLoader to handle variable sequence lengths.
    Pads sequences to the maximum length in the current batch.
    """
    sequences, labels = zip(*batch)
    
    # Pad sequences: returns tensor of shape (batch_size, max_seq_length, feature_dim)
    # batch_first=True makes standard dimension order
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # Create attention masks (1 for actual data, 0 for padding)
    # This helps models like Mamba ignore padded timesteps
    attention_masks = torch.zeros(padded_sequences.shape[0], padded_sequences.shape[1], dtype=torch.float32)
    for i, seq in enumerate(sequences):
        attention_masks[i, :len(seq)] = 1.0
        
    labels = torch.stack(labels)
    
    # Return padded sequences, their active masks, and the labels
    return padded_sequences, attention_masks, labels

def create_dataloaders(data_dir, batch_size=32, val_split=0.2, test_split=0.1, random_seed=42):
    """
    Utility function to create separate train, validation, and test dataloaders.
    """
    from torch.utils.data import random_split
    
    dataset = GaitPathologyDataset(data_dir=data_dir)
    
    total_size = len(dataset)
    if total_size == 0:
        raise ValueError(f"No valid CSV files found in {data_dir}. Please create 'Datasets/' directory with class subfolders containing CSVs.")
        
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_pad
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_pad
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_pad
    )
    
    return train_loader, val_loader, test_loader, dataset

if __name__ == "__main__":
    # Allow execution to create dummy data and test the loader directly
    print("Testing dataset loader implementation...")
    # Will add dummy data testing code below when building the whole pipeline.
