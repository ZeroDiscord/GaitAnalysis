"""
prototype_head.py — Prototypical Network Classifier Head
========================================================
Replaces the standard linear softmax head with a prototype-based classifier.

For each class, maintains a prototype (mean embedding of class samples).
Classification = nearest prototype in embedding space.

This naturally handles severe class imbalance: PIVD_Priformis with 3 patients
gets the same representation power as Healthy with 15 patients, because
each class has exactly one prototype vector.

Reference: Snell et al., "Prototypical Networks for Few-shot Learning" (arXiv:1703.05175)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PrototypeClassifier(nn.Module):
    """
    Prototype-based classification head.
    
    During training:
      - Maintains running class prototypes (EMA of class embeddings)
      - Loss = negative log-probability under prototype distances
      
    During inference:
      - Classify by nearest prototype in embedding space
      
    Advantages over linear softmax:
      - No bias toward majority classes (each class = 1 prototype)
      - Naturally handles few-shot classes (PIVD_Priformis)
      - Embedding space is metric (distances are interpretable)
    """
    
    def __init__(self, embed_dim: int, num_classes: int, 
                 temperature: float = 0.1, ema_decay: float = 0.99):
        """
        Args:
            embed_dim: Dimension of input embeddings (from backbone's pooled output)
            num_classes: Number of pathology classes
            temperature: Scaling factor for distance to probability conversion
                        Lower = sharper decisions. 0.1 works well for medical classification.
            ema_decay: Exponential moving average decay for prototype updates
                      0.99 = slow adaptation (stable), 0.9 = fast adaptation
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.ema_decay = ema_decay
        
        # Learnable prototypes (initialized randomly, updated via EMA + gradient)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.1)
        
        # Optional projection to a more discriminative space
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Running prototype estimates (not parameters, updated via EMA)
        self.register_buffer('running_prototypes', torch.zeros(num_classes, embed_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))
        self.register_buffer('initialized', torch.tensor(False))
    
    def forward(self, embeddings: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embeddings: (B, embed_dim) from backbone
            labels: (B,) class labels — only needed during training for EMA updates
            
        Returns:
            logits: (B, num_classes) — negative scaled distances (higher = more similar)
        """
        # Project embeddings
        projected = self.projection(embeddings)
        
        # Use learnable prototypes (updated by both gradient and EMA)
        prototypes = self.prototypes  # (num_classes, embed_dim)
        
        # Compute squared Euclidean distances
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b
        dists = torch.cdist(projected, prototypes, p=2)  # (B, num_classes)
        
        # Convert distances to logits (negative distance / temperature)
        logits = -dists / self.temperature
        
        # During training: update running prototypes via EMA
        if self.training and labels is not None:
            self._update_prototypes(projected.detach(), labels)
        
        return logits
    
    @torch.no_grad()
    def _update_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Update running prototype estimates via EMA."""
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                class_mean = embeddings[mask].mean(dim=0)
                
                if not self.initialized or self.prototype_counts[c] == 0:
                    self.running_prototypes[c] = class_mean
                    self.prototype_counts[c] = 1
                else:
                    self.running_prototypes[c] = (
                        self.ema_decay * self.running_prototypes[c] + 
                        (1 - self.ema_decay) * class_mean
                    )
                    self.prototype_counts[c] += 1
        
        # Periodically sync learnable prototypes with running estimates
        # (soft target — gradient still drives primary updates)
        if self.prototype_counts.sum() > 0:
            valid = self.prototype_counts > 0
            self.prototypes.data[valid] = (
                0.9 * self.prototypes.data[valid] + 
                0.1 * self.running_prototypes[valid]
            )
            self.initialized.fill_(True)
    
    def get_class_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return raw distances to each prototype (for visualization/debugging)."""
        projected = self.projection(embeddings)
        return torch.cdist(projected, self.prototypes, p=2)


class HybridClassifier(nn.Module):
    """
    Hybrid classifier: linear softmax + prototype, combined.
    
    Uses both a standard linear head and prototype distances,
    with a learnable gate that blends their logits.
    
    This hedges: if prototypes help the minority class, the gate
    can upweight them; if linear is better overall, it dominates.
    """
    
    def __init__(self, embed_dim: int, num_classes: int, temperature: float = 0.1):
        super().__init__()
        self.linear_head = nn.Linear(embed_dim, num_classes)
        self.prototype_head = PrototypeClassifier(embed_dim, num_classes, temperature)
        
        # Learnable blending gate
        self.gate = nn.Parameter(torch.tensor(0.5))  # sigmoid -> [0, 1]
    
    def forward(self, embeddings: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        linear_logits = self.linear_head(embeddings)
        proto_logits = self.prototype_head(embeddings, labels)
        
        alpha = torch.sigmoid(self.gate)
        return alpha * linear_logits + (1 - alpha) * proto_logits
