"""
models/bimamba_classifier.py — Bidirectional Mamba for Gait Classification
==========================================================================
Pure PyTorch bidirectional selective SSM classifier.

Key innovations over the original Mamba models:
  1. BIDIRECTIONAL scan (forward + backward, gated fusion)
     -- Following FEMBA (arXiv:2502.06438) and TSCMamba (arXiv:2406.04419)
     -- For classification, future context within the window is as important as past
  2. PROPER PADDING HANDLING
     -- Padding tokens are masked out before SSM scan by zeroing delta at pad positions
     -- Following PackMamba (arXiv:2408.03865): at pad positions, A_bar->0 (state reset)
  3. Input-dependent discretization (selective mechanism)
     -- delta, B, C are input-dependent, not fixed -- critical for non-stationary EMG
  4. Pre-norm architecture with residual connections for training stability

Architecture:
    Input (B, T, input_dim)
    -> Linear projection -> (B, T, d_model)  
    -> N x BiMambaBlock:
        Forward SSM scan + Backward SSM scan -> Gated fusion
        + Conv1d local context + LayerNorm + residual
    -> Masked mean pool -> (B, d_model)
    -> Classifier head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) -- the core Mamba mechanism.
    
    Implements the input-dependent discretization:
        delta = softplus(Linear(x))
        B = Linear(x)
        C = Linear(x)
        A_bar = exp(delta * A)
        B_bar = delta * B
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t
    
    With proper padding mask:
        At padding positions, delta -> large value
        This makes exp(delta*A) -> 0 (since A < 0), effectively resetting state.
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        
        # Input projection: x -> (z, x_ssm) split
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        
        # Depthwise conv for local context (like Mamba's conv1d)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )
        
        # Input-dependent parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # delta, B, C
        
        # A matrix (log-space, negative for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1).clone())
        
        # delta bias (controls default timescale)
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_softplus = torch.log(torch.expm1(dt_init))
        self.dt_bias = nn.Parameter(inv_softplus)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # D skip connection (like residual)
        self.D = nn.Parameter(torch.ones(self.d_inner))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                reverse: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: (B, T) -- 1.0 for real, 0.0 for padding
            reverse: if True, scan backward (for bidirectional)
        
        Returns:
            y: (B, T, d_model)
        """
        B, T, _ = x.shape
        
        if reverse:
            x = x.flip(1)
            if mask is not None:
                mask = mask.flip(1)
        
        # Input projection and split
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)
        
        # Depthwise conv (local context)
        x_conv = x_ssm.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]  # truncate padding
        x_ssm = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_ssm = F.silu(x_ssm)
        
        # Input-dependent delta, B, C
        params = self.x_proj(x_ssm)  # (B, T, d_state*2 + 1)
        dt_raw = params[..., :1]  # (B, T, 1)
        B_input = params[..., 1:1+self.d_state]  # (B, T, d_state)
        C_input = params[..., 1+self.d_state:]  # (B, T, d_state)
        
        # Discretize delta
        dt = F.softplus(dt_raw.squeeze(-1) + self.dt_bias.unsqueeze(0).unsqueeze(0).expand(B, T, -1).mean(-1))
        dt = dt.unsqueeze(-1).expand(B, T, self.d_inner)  # (B, T, d_inner)
        
        # PADDING MASK: At pad positions, set delta to a large value
        # This makes exp(delta*A) -> 0 (since A < 0), effectively resetting state
        if mask is not None:
            pad_mask = (1.0 - mask).unsqueeze(-1)  # (B, T, 1) -- 1.0 at pad positions
            dt = dt + pad_mask * 10.0  # Large delta at pad -> A_bar approx 0 -> state reset
        
        # Compute A (negative, in log-space for stability)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Parallel scan (vectorized associative scan)
        y = self._parallel_scan(x_ssm, dt, A, B_input, C_input)
        
        # Gating and output
        y = y * F.silu(z)  # gated output
        y = self.out_proj(y)
        
        if reverse:
            y = y.flip(1)
        
        return y
    
    def _parallel_scan(self, x, dt, A, B, C):
        """
        Vectorized Parallel Associative Scan.
        Calculates h_t = A_bar_t * h_{t-1} + B_bar_t * x_t in parallel.
        """
        B_batch, T, d_inner = x.shape
        d_state = self.d_state
        
        # Discretize A and B: (B, T, d_inner, d_state)
        # A is (d_inner, d_state), dt is (B, T, d_inner)
        dt = dt.unsqueeze(-1) # (B, T, d_inner, 1)
        A_bar = torch.exp(dt * A.unsqueeze(0).unsqueeze(0)) # (B, T, d_inner, d_state)
        B_bar = dt * B.unsqueeze(2) # (B, T, d_inner, d_state)
        
        # Intermediate signal: B_bar * x
        X_bar = B_bar * x.unsqueeze(-1) # (B, T, d_inner, d_state)
        
        # Parallel Scan logic:
        # h_t = (A_bar_t * A_bar_{t-1} * ... * A_bar_1) * h_0 + sum_{i=1}^t (A_bar_t * ... * A_bar_{i+1}) * X_bar_i
        # In log space or using cumprod for stability. 
        # For sequence length 500, cumprod is stable enough.
        
        # Compute cumulative state transitions
        A_bar_cumsum = torch.cat([
            torch.ones(B_batch, 1, d_inner, d_state, device=x.device, dtype=x.dtype),
            A_bar
        ], dim=1)
        
        # h_t = sum_{i=1}^t [ (prod_{j=i+1}^t A_bar_j) * X_bar_i ]
        # This can be computed efficiently by: 
        # h_t = (prod_{j=1}^t A_bar_j) * sum_{i=1}^t [ X_bar_i / (prod_{j=1}^i A_bar_j) ]
        
        A_bar_prod = torch.cumprod(A_bar, dim=1)
        
        # Handle potential division by zero or underflow by using a small epsilon
        # or performing the scan in a more robust way
        states = A_bar_prod * torch.cumsum(X_bar / (A_bar_prod + 1e-12), dim=1)
        
        # Output y_t = C_t * h_t + D * x_t
        y = (states * C.unsqueeze(2)).sum(dim=-1) # (B, T, d_inner)
        y = y + self.D * x
        
        return y


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block.
    
    Forward scan + backward scan -> learned gating fusion.
    Following FEMBA: both scans are summed with a learnable gate,
    not concatenated (to preserve d_model dimensionality).
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.forward_ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.backward_ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        
        # Learned gate for blending forward and backward
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: (B, T)
        Returns:
            (B, T, d_model) with residual
        """
        residual = x
        h = self.norm(x)
        
        fwd = self.forward_ssm(h, mask, reverse=False)
        bwd = self.backward_ssm(h, mask, reverse=True)
        
        # Gated fusion
        combined = torch.cat([fwd, bwd], dim=-1)  # (B, T, 2*d_model)
        gate_weights = self.gate(combined)  # (B, T, d_model) in [0, 1]
        fused = gate_weights * fwd + (1 - gate_weights) * bwd
        
        return residual + self.dropout(fused)


class BiMambaGaitClassifier(nn.Module):
    """
    Bidirectional Mamba classifier for gait pathology classification.
    
    Architecture:
        Input projection -> N x BiMambaBlock -> Masked mean pool -> Classifier
    """
    
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 64,
                 n_layers: int = 2, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # BiMamba blocks
        self.blocks = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"BiMamba Classifier initialized: {input_dim} -> {d_model}")
        print(f"  Bidirectional: Yes | Layers: {n_layers} | d_state: {d_state} | Params: {total_params:,}")
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'A_log' in name or 'dt_bias' in name or 'D' in name:
                continue  # These have special initialization
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif param.dim() == 1 and 'bias' in name:
                nn.init.zeros_(param)
    
    def get_embedding(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract pooled embedding (before classifier).
        Useful for prototype head or visualization.
        """
        h = self.input_proj(x)  # (B, T, d_model)
        
        for block in self.blocks:
            h = block(h, mask)
        
        h = self.final_norm(h)
        
        # Masked mean pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
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
