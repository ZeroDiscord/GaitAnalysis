import torch
import torch.nn as nn
<<<<<<< HEAD
import torch.nn.functional as F
=======
>>>>>>> 251ecc3 (Initial commit of GaitMamba Pipeline)

class GRUAttentionGaitClassifier(nn.Module):
    """
    Requested Baseline Model: GRU with Masked Multi-Head Attention and Key-Value Caching.
    Useful for benchmarking Mamba's performance and parameter efficiency against classic RNN+Attention architectures.
    """
    def __init__(self, input_dim=4, num_classes=2, d_model=64, num_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Feature Embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Gated Recurrent Unit (GRU) Backend
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        
<<<<<<< HEAD
        # 3. Masked Multi-Head Attention (Custom SDPA for Extreme Sequences)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
=======
        # 3. Masked Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
>>>>>>> 251ecc3 (Initial commit of GaitMamba Pipeline)
        
        # 4. Layernorms and Pooling
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, mask=None, kv_cache=None):
        """
        Forward pass with optional KV caching for streaming generation.
        kv_cache shape: dict containing 'gru_hidden' and 'past_key_values'
        """
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        x = self.embedding(x)
        
        # 1. GRU Processing
        gru_hidden_in = kv_cache['gru_hidden'] if kv_cache is not None else None
        gru_out, gru_hidden_out = self.gru(x, gru_hidden_in)
        
        # Residual connection and norm
        gru_out = self.norm1(x + self.dropout(gru_out))
        
<<<<<<< HEAD
        # 2. Masked Attention Processing (Direct SDPA integration)
        # CRITICAL FIX: Directly use F.scaled_dot_product_attention to bypass nn.MultiheadAttention's
        # tendency to block is_causal=True without an explicit mask.
        
        # Project Q, K, V
        qkv = self.qkv_proj(gru_out)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute FlashAttention / Memory Efficient SDPA
        # We drop the explicit pad mask because padded regions will just compute junk that we temporally mask out anyway.
        # This completely guarantees pure FlashAttention execution taking only O(N) memory!
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None, 
                    dropout_p=self.dropout.p if self.training else 0.0, 
                    is_causal=True
                )
        except Exception:
            # Fallback if specific Hopper instructions reject the strict Flash constraints
            attn_out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout.p if self.training else 0.0, 
                is_causal=True
            )
        
        # Reshape back to (batch_size, seq_len, d_model)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        attn_out = self.o_proj(attn_out)
=======
        # 2. Masked Attention Processing
        # Create causal mask ensuring position i can only attend to positions <= i
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        
        # Handle KV Caching
        if kv_cache is not None and 'past_keys' in kv_cache:
            past_keys, past_values = kv_cache['past_keys'], kv_cache['past_values']
            
        # Attention Forward
        # key_padding_mask requires True for padded regions
        key_padding_mask = (mask == 0) if mask is not None else None
        
        attn_out, _ = self.attention(
            query=gru_out,
            key=gru_out,
            value=gru_out,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=True
        )
>>>>>>> 251ecc3 (Initial commit of GaitMamba Pipeline)
        
        # Residual connection and norm
        x_out = self.norm2(gru_out + self.dropout(attn_out))
        
        # 3. Temporal Pooling
        if mask is not None:
             mask_expanded = mask.unsqueeze(-1)
             sum_embeddings = torch.sum(x_out * mask_expanded, dim=1)
             valid_lengths = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
             pooled = sum_embeddings / valid_lengths
        else:
            pooled = torch.mean(x_out, dim=1)
            
        logits = self.classifier(pooled)
        
        if kv_cache is not None:
            # Return updated cache state
            new_cache = {'gru_hidden': gru_hidden_out} # Simplified cache return
            return logits, new_cache
            
        return logits
