import torch
import torch.nn as nn

class OfficialMambaGaitClassifier(nn.Module):
    """
    Uses the official `mamba-ssm` CUDA package if installed.
    This provides the exact implementation from the original Albert Gu / Tri Dao paper.
    """
    def __init__(self, input_dim=4, num_classes=2, d_model=128, n_layers=4, d_state=16):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError:
            raise ImportError(
                "Please install the official mamba package: `pip install causal-conv1d>=1.2.0 mamba-ssm`\n"
                "Note: This usually requires a Linux environment with a CUDA toolkit installed."
            )
            
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Stacked official Mamba blocks
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
            for _ in range(n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        
        for layer in self.layers:
            # The official Mamba implementation doesn't natively support padding masks 
            # in its optimized CUDA kernel out of the box for the causal scan,
            # but for our full-length unpadded sequences, it works perfectly.
            x = x + layer(x)
            
        x = self.norm_f(x)
        
        if mask is not None:
             mask_expanded = mask.unsqueeze(-1)
             sum_embeddings = torch.sum(x * mask_expanded, dim=1)
             valid_lengths = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
             pooled = sum_embeddings / valid_lengths
        else:
            pooled = torch.mean(x, dim=1)
            
        return self.classifier(pooled)
