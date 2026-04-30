import torch
import torch.nn as nn

class SimplifiedMambaBlock(nn.Module):
    """
    A simplified PyTorch implementation of a Mamba-like State Space Model block.
    This provides the core recurrent dynamics of S4/Mamba without requiring 
    the complex CUDA extensions, making it fully portable.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 1D Convolution for local sequence context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # Linear layer for x input
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_state + self.d_inner, bias=False)
        
        # dt parameter (time step)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # State Space parameters
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # Learned transition matrix A
        self.D = nn.Parameter(torch.ones(self.d_inner)) # Skip connection parameter D
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x, mask=None):
        # CRITICAL: Force float32 for the entire Mamba recurrence to prevent
        # catastrophic floating-point drift over 2000 sequential timesteps in bfloat16.
        original_dtype = x.dtype
        x = x.float()
        
        batch, seq_len, _ = x.shape
        
        x_proj = self.in_proj(x)
        x_in, res = x_proj.chunk(2, dim=-1)
        
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len] 
        x_conv = x_conv.transpose(1, 2)
        
        x_act = self.activation(x_conv)
        
        x_params = self.x_proj(x_act)
        delta, B, C = torch.split(x_params, [self.d_inner, self.d_state, self.d_state], dim=-1)
        
        # Clamp delta projections before softplus/exp to prevent infinity
        delta_proj = self.dt_proj(delta)
        delta_proj = torch.clamp(delta_proj, min=-20.0, max=5.0)
        delta = nn.functional.softplus(delta_proj)
        
        # Stability clamp on transition matrix
        A = -torch.exp(self.A_log.float()) 
        A = torch.clamp(A, max=-1e-4)
        
        y = torch.zeros((batch, seq_len, self.d_inner), device=x.device, dtype=torch.float32)
        state = torch.zeros((batch, self.d_inner, self.d_state), device=x.device, dtype=torch.float32)
        
        for t in range(seq_len):
            if mask is not None:
                step_mask = mask[:, t].unsqueeze(-1).unsqueeze(-1).float()
            else:
                step_mask = 1.0

            dt_t = delta[:, t, :].unsqueeze(-1)
            
            A_bar = torch.exp(dt_t * A) 
            
            B_t = B[:, t, :].unsqueeze(1)
            x_t = x_act[:, t, :].unsqueeze(-1)
            B_bar = dt_t * B_t * x_t 
            
            new_state = A_bar * state + B_bar
            state = state * (1 - step_mask) + new_state * step_mask
            
            # Periodic state clamping to catch runaway accumulation
            if t % 100 == 99:
                state = torch.clamp(state, min=-1e4, max=1e4)
            
            C_t = C[:, t, :].unsqueeze(-1)
            y_t = torch.matmul(state, C_t).squeeze(-1) 
            y_t = y_t + self.D * x_act[:, t, :]
            
            y[:, t, :] = y_t
            
        y = y * self.activation(res)
        out = self.out_proj(y)
        return out.to(original_dtype)


class MambaGaitClassifier(nn.Module):
    """
    Complete Mamba-based architecture for Biomechanics-Informed Gait Pathology Classification.
    """
    def __init__(self, input_dim=4, num_classes=2, d_model=128, n_layers=4, d_state=16, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        self.layers = nn.ModuleList([
            SimplifiedMambaBlock(d_model=d_model, d_state=d_state)
            for _ in range(n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, mask=None):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = x + self.dropout(layer(x, mask=mask))
            
        x = self.norm_f(x)
        
        if mask is not None:
             mask_expanded = mask.unsqueeze(-1)
             sum_embeddings = torch.sum(x * mask_expanded, dim=1)
             valid_lengths = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
             pooled = sum_embeddings / valid_lengths
        else:
            pooled = torch.mean(x, dim=1)
            
        return self.classifier(pooled)
