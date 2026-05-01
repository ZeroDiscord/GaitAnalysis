"""
models/gru_baseline.py — Bidirectional GRU + Non-Causal Attention
=================================================================
Bidirectional GRU with non-causal multi-head attention pooling.

Architecture:
    Input (B, T, input_dim)
    -> Linear projection -> (B, T, d_model)
    -> Bidirectional GRU stack (n_layers)
    -> Non-causal multi-head attention pooling
    -> Masked mean pool -> (B, d_model)
    -> Classifier head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class GRUAttentionGaitClassifier(nn.Module):
    """
    Bidirectional GRU with non-causal attention pooling for gait classification.
    
    Features:
      - Bidirectional GRU (concatenated, then projected back to d_model)
      - Non-causal self-attention (full window visibility)
      - Proper attention masking for padded sequences
      - LayerNorm + residual connections for training stability
    """
    
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 64,
                 num_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Bidirectional GRU stack
        # Each layer: BiGRU outputs 2*d_model, then projected back to d_model
        self.gru_layers = nn.ModuleList()
        self.gru_norms = nn.ModuleList()
        self.gru_projs = nn.ModuleList()
        
        for i in range(n_layers):
            self.gru_layers.append(nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if i < n_layers - 1 else 0.0,
            ))
            self.gru_norms.append(nn.LayerNorm(d_model))
            # Project 2*d_model (bidirectional concat) back to d_model
            self.gru_projs.append(nn.Linear(2 * d_model, d_model))
        
        # Non-causal multi-head attention for sequence pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"GRU+Attention initialized: {input_dim} -> {d_model}")
        print(f"  Bidirectional: Yes | Causal attention: No | Params: {total_params:,}")
    
    def _init_weights(self):
        """Xavier initialization for linear layers, orthogonal for GRU."""
        for name, param in self.named_parameters():
            if 'gru' in name and 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'gru' in name and 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'gru' in name and 'bias' in name:
                nn.init.zeros_(param)
            elif param.dim() >= 2:
                nn.init.xavier_uniform_(param)
    
    def get_embedding(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract the pooled embedding (before classifier head).
        Useful for prototype classifier or contrastive learning.
        
        Args:
            x: (B, T, input_dim) -- raw features
            mask: (B, T) -- 1.0 for real tokens, 0.0 for padding
            
        Returns:
            embedding: (B, d_model)
        """
        B, T, _ = x.shape
        
        # Input projection
        h = self.input_proj(x)  # (B, T, d_model)
        
        # Bidirectional GRU stack with residual connections
        for gru, norm, proj in zip(self.gru_layers, self.gru_norms, self.gru_projs):
            residual = h
            gru_out, _ = gru(h)  # (B, T, 2*d_model)
            h = proj(gru_out)    # (B, T, d_model)
            h = norm(h + residual)  # residual + LayerNorm
        
        # Non-causal self-attention with proper padding mask
        # key_padding_mask: True = IGNORE this position (PyTorch convention)
        if mask is not None:
            key_padding_mask = (mask == 0)  # True where padding
        else:
            key_padding_mask = None
        
        attn_out, _ = self.attention(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=False,  # CRITICAL: non-causal for classification
        )
        h = self.attn_norm(h + attn_out)  # residual
        
        # Masked mean pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
            h = (h * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            h = h.mean(dim=1)
        
        return h  # (B, d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
            mask: (B, T) -- 1.0 for real tokens, 0.0 for padding
            
        Returns:
            logits: (B, num_classes)
        """
        embedding = self.get_embedding(x, mask)
        return self.classifier(embedding)
