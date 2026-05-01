import torch
import pandas as pd
import numpy as np
import os
import argparse
from dataset import assign_gait_phase_continuous
from checkpoint_utils import load_checkpoint
from models.bimamba_classifier import BiMambaGaitClassifier
from models.gru_baseline import GRUAttentionGaitClassifier

def build_model_from_checkpoint(path, device):
    """Loads checkpoint and builds the corresponding model architecture."""
    state_dict, config, meta = load_checkpoint(path, device)
    model_type = config.get('model_type', 'bimamba')
    input_dim = config.get('input_dim', 5)
    num_classes = config.get('num_classes', 5)
    d_model = config.get('d_model', 64)
    n_layers = config.get('n_layers', 2)
    
    if model_type == 'bimamba':
        model = BiMambaGaitClassifier(input_dim, num_classes, d_model, n_layers)
    else:
        model = GRUAttentionGaitClassifier(input_dim, num_classes, d_model, n_layers)
        
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, config, meta

def process_file(csv_path, window_size=500, stride=250):
    """Reads CSV or a directory of CSVs and prepares windows with pre-computed features."""
    if os.path.isdir(csv_path):
        # Stitch all CSVs in the directory
        csv_files = sorted([os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith('.csv')])
        print(f"Stitching {len(csv_files)} files from directory...")
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(csv_path)
    
    # Mapping for the 'final_stitched.csv' and raw 'serial' format:
    # tibialis_env -> TA (e_ago)
    # gastro_env -> GA (e_ant)
    if 'tibialis_env' in df.columns and 'gastro_env' in df.columns:
        e_ago = df['tibialis_env'].values.astype(np.float64)
        e_ant = df['gastro_env'].values.astype(np.float64)
    else:
        # Fallback to original 2-column no-header format
        e_ago = df.iloc[:, 0].values.astype(np.float64)
        e_ant = df.iloc[:, 1].values.astype(np.float64)
    
    e_ago = np.nan_to_num(e_ago)
    e_ant = np.nan_to_num(e_ant)
    
    # Pre-compute features matching the training pipeline
    torque = e_ant - e_ago
    stiffness = e_ant + e_ago
    try:
        gait_phase = assign_gait_phase_continuous(e_ago, e_ant, fs=1000.0)
    except:
        gait_phase = np.linspace(0, 100, len(e_ago))
    
    features = np.column_stack([e_ant, e_ago, torque, stiffness, gait_phase]).astype(np.float32)
    
    # Create windows
    windows = []
    for start in range(0, len(features) - window_size + 1, stride):
        win = features[start : start + window_size]
        windows.append(win)
    
    if not windows:
        windows.append(features)
        
    return torch.tensor(np.array(windows), dtype=torch.float32)

@torch.no_grad()
def run_inference(model, windows, device, class_names):
    """Performs inference and aggregates results across windows."""
    windows = windows.to(device)
    logits = model(windows)
    probs = torch.softmax(logits, dim=-1)
    
    # Aggregate: Average Probability across the entire file
    avg_probs = probs.mean(dim=0)
    pred_idx = torch.argmax(avg_probs).item()
    confidence = avg_probs[pred_idx].item()
    
    return class_names[pred_idx], confidence

def main():
    parser = argparse.ArgumentParser(description="Inference on a single gait CSV file.")
    parser.add_argument('--csv', type=str, required=True, help="Path to raw CSV file")
    parser.add_argument('--mamba_path', type=str, required=True, help="Path to BiMamba .pth")
    parser.add_argument('--gru_path', type=str, required=True, help="Path to GRU .pth")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Inference Node Initialized ---")
    print(f"Device: {device}")

    # Process data
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at {args.csv}")
        return

    print(f"Loading: {os.path.basename(args.csv)}")
    window_batch = process_file(args.csv)
    print(f"Extracted {len(window_batch)} windows for analysis.")

    # Inference - BiMamba
    if os.path.exists(args.mamba_path):
        model_mamba, _, meta_mamba = build_model_from_checkpoint(args.mamba_path, device)
        
        # Load stats and classes from metadata
        mu = torch.tensor(meta_mamba.get('global_mean', [0]*5), device=device)
        std = torch.tensor(meta_mamba.get('global_std', [1]*5), device=device)
        classes = meta_mamba.get('classes', ['Healthy', 'Hemiplegia', 'Osteoarthiritis', 'PIVD_Priformis', 'PIVD_RA'])
        
        # Normalize and run
        norm_windows = (window_batch.to(device) - mu) / (std + 1e-8)
        label_mamba, conf_mamba = run_inference(model_mamba, norm_windows, device, classes)
        
        print(f"\n[BiMamba Results]")
        print(f"  Prediction: {label_mamba}")
        print(f"  Confidence: {conf_mamba:.2%}")
    else:
        print(f"Warning: Mamba checkpoint not found at {args.mamba_path}")

    # Inference - GRU
    if os.path.exists(args.gru_path):
        model_gru, _, meta_gru = build_model_from_checkpoint(args.gru_path, device)
        
        # Load stats and classes from metadata
        mu = torch.tensor(meta_gru.get('global_mean', [0]*5), device=device)
        std = torch.tensor(meta_gru.get('global_std', [1]*5), device=device)
        classes = meta_gru.get('classes', ['Healthy', 'Hemiplegia', 'Osteoarthiritis', 'PIVD_Priformis', 'PIVD_RA'])
        
        # Normalize and run
        norm_windows = (window_batch.to(device) - mu) / (std + 1e-8)
        label_gru, conf_gru = run_inference(model_gru, norm_windows, device, classes)
        
        print(f"\n[GRU Results]")
        print(f"  Prediction: {label_gru}")
        print(f"  Confidence: {conf_gru:.2%}")
    else:
        print(f"Warning: GRU checkpoint not found at {args.gru_path}")

if __name__ == "__main__":
    main()
