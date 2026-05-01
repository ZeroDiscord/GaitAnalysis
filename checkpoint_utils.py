"""
checkpoint_utils.py — Save/Load model config alongside weights
==============================================================
Prevents the silent d_model cap mismatch between train.py and evaluate.py.
Every checkpoint now stores the exact architecture config that was used.
"""

import os
import json
import torch
from typing import Dict, Any, Optional


def save_checkpoint(model, config: Dict[str, Any], path: str, extra: Optional[Dict] = None):
    """
    Save model weights + architecture config as a single .pth file.
    
    Args:
        model: nn.Module to save
        config: Dict with architecture params (model_type, input_dim, num_classes, d_model, n_layers, etc.)
        path: Output .pth file path
        extra: Optional dict with training metadata (epoch, best_f1, optimizer_state, etc.)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
    }
    if extra:
        checkpoint['training_meta'] = extra
    torch.save(checkpoint, path)
    
    # Also save a human-readable JSON sidecar for inspection
    json_path = path.replace('.pth', '_config.json')
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_checkpoint(path: str, device: torch.device = None):
    """
    Load checkpoint and return (state_dict, config, training_meta).
    
    Handles both new-format (dict with 'config') and legacy (raw state_dict) checkpoints.
    """
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # New format: dict with 'model_state_dict' and 'config'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return (
            checkpoint['model_state_dict'],
            checkpoint.get('config', {}),
            checkpoint.get('training_meta', {})
        )
    
    # Legacy format: raw state_dict (no config)
    # Try to infer config from weight shapes
    if isinstance(checkpoint, dict):
        config = _infer_config_from_state_dict(checkpoint)
        return checkpoint, config, {}
    
    raise ValueError(f"Unrecognized checkpoint format in {path}")


def _infer_config_from_state_dict(state_dict: dict) -> dict:
    """
    Best-effort config inference from weight tensor shapes.
    Used for backward compatibility with legacy checkpoints.
    """
    config = {}
    
    # Try to find input projection layer to infer input_dim and d_model
    for key, tensor in state_dict.items():
        if 'input_proj' in key and 'weight' in key and tensor.dim() == 2:
            config['d_model'] = tensor.shape[0]
            config['input_dim'] = tensor.shape[1]
            break
        elif 'linear' in key.lower() and 'weight' in key and tensor.dim() == 2:
            # Fallback: first linear layer
            if 'input_dim' not in config:
                config['d_model'] = tensor.shape[0]
                config['input_dim'] = tensor.shape[1]
    
    # Try to find classifier head to infer num_classes
    for key, tensor in state_dict.items():
        if ('classifier' in key or 'fc' in key or 'head' in key) and 'weight' in key:
            config['num_classes'] = tensor.shape[0]
            break
    
    # Count layers by looking for repeated block patterns
    layer_indices = set()
    for key in state_dict.keys():
        parts = key.split('.')
        for i, part in enumerate(parts):
            if part.isdigit() and i > 0 and ('layer' in parts[i-1] or 'block' in parts[i-1]):
                layer_indices.add(int(part))
    if layer_indices:
        config['n_layers'] = max(layer_indices) + 1
    
    config['_inferred'] = True  # Flag that this was guessed, not saved
    return config


def make_config(model_type: str, input_dim: int, num_classes: int,
                d_model: int, n_layers: int, **kwargs) -> dict:
    """Convenience: build a standard config dict."""
    config = {
        'model_type': model_type,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'd_model': d_model,
        'n_layers': n_layers,
    }
    config.update(kwargs)
    return config
