import torch
import pandas as pd
import numpy as np
import os
import argparse
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# Internal imports
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

def process_file(csv_path, window_size=500, stride=250, swap_muscles=False, target_stds=[22.0, 13.0]):
    """Reads CSV/Directory and prepares features with 100% parity to dataset.py."""
    if os.path.isdir(csv_path):
        csv_files = sorted([os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith('.csv')])
        print(f"Stitching {len(csv_files)} files...")
        all_dfs = []
        current_time_offset = 0
        for f in csv_files:
            temp_df = pd.read_csv(f)
            temp_df.iloc[:, 0] = temp_df.iloc[:, 0] + current_time_offset
            current_time_offset = temp_df.iloc[-1, 0] + 40
            all_dfs.append(temp_df)
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        # Check if 2-column format (standard dataset)
        sample = pd.read_csv(csv_path, nrows=1, header=None)
        if sample.shape[1] == 2:
            print("  [Format] Standard Training Format (2-column)...")
            df = pd.read_csv(csv_path, header=None)
        else:
            print(f"  [Format] Local Hardware Format ({pd.read_csv(csv_path, nrows=1).shape[1]} columns)...")
            df = pd.read_csv(csv_path)

    # 1. Temporal Handling
    if df.shape[1] == 2:
        # Standard format is already 1000Hz
        e_ago = np.nan_to_num(df.iloc[:, 0].values.astype(np.float64))
        e_ant = np.nan_to_num(df.iloc[:, 1].values.astype(np.float64))
        fs = 1000.0
    else:
        # Resample local hardware to 1000Hz
        t = df.iloc[:, 0].values / 1000.0
        duration = t[-1] - t[0]
        fs = 1000.0
        num_samples = int(duration * fs)
        t_new = np.linspace(t[0], t[-1], num_samples)
        
        if 'tibialis_raw' in df.columns and 'gastro_raw' in df.columns:
            ago_raw = np.nan_to_num(df['tibialis_raw'].values.astype(np.float64))
            ant_raw = np.nan_to_num(df['gastro_raw'].values.astype(np.float64))
        else:
            ant_raw = np.nan_to_num(df.iloc[:, 13].values.astype(np.float64))
            ago_raw = np.nan_to_num(df.iloc[:, 16].values.astype(np.float64))
            
        f_ago = interp1d(t, ago_raw, kind='linear', fill_value="extrapolate")
        f_ant = interp1d(t, ant_raw, kind='linear', fill_value="extrapolate")
        e_ago = np.nan_to_num(f_ago(t_new))
        e_ant = np.nan_to_num(f_ant(t_new))
        
        # 3. Local Hardware Pre-processing (Filter noise + Scale to training power)
        def clean_local(signal, target_std, fs=1000.0):
            # Bandpass Filter (20-450Hz) - CRITICAL for local hardware noise
            nyq = 0.5 * fs
            low, high = 20.0 / nyq, 450.0 / nyq
            b, a = butter(4, [low, high], btype='band')
            signal = filtfilt(b, a, signal)
            # Remove outliers and scale
            std = signal.std() + 1e-8
            signal = np.clip(signal, -3 * std, 3 * std)
            signal = signal - signal.mean()
            return (signal / (signal.std() + 1e-8)) * target_std

        e_ago = clean_local(e_ago, target_stds[0], fs=fs)
        e_ant = clean_local(e_ant, target_stds[1], fs=fs)

    if swap_muscles:
        e_ago, e_ant = e_ant, e_ago

    # 2. Compute Features (Match dataset.py logic exactly)
    torque = e_ant - e_ago
    stiffness = e_ant + e_ago
    try:
        gait_phase, _, _ = assign_gait_phase_continuous(e_ago, e_ant, fs=fs)
    except:
        gait_phase = np.linspace(0, 100, len(e_ago))
    
    features = np.column_stack([e_ant, e_ago, torque, stiffness, gait_phase]).astype(np.float32)
    
    is_resampled = "Resampled" if df.shape[1] != 2 else "Original"
    print(f"  Signal Debug ({is_resampled} 1000Hz) - TA: Mean={e_ago.mean():.2f}, Range=[{e_ago.min():.2f}, {e_ago.max():.2f}]")
    print(f"  Signal Debug ({is_resampled} 1000Hz) - GA: Mean={e_ant.mean():.2f}, Range=[{e_ant.min():.2f}, {e_ant.max():.2f}]")

    # 3. Windowing
    windows = []
    for start in range(0, len(features) - window_size + 1, stride):
        windows.append(features[start : start + window_size])
    if not windows:
        windows.append(features)
        
    return torch.tensor(np.array(windows), dtype=torch.float32)

@torch.no_grad()
def run_inference(model, windows, device, meta, batch_size=16):
    """Performs inference with proper Z-score normalization from metadata."""
    mu = torch.tensor(meta.get('global_mean', [[0]*5]), device=device)
    std = torch.tensor(meta.get('global_std', [[1]*5]), device=device)
    class_names = meta.get('classes', [])
    
    all_logits = []
    for i in range(0, len(windows), batch_size):
        # Apply Z-score normalization
        batch = (windows[i : i + batch_size].to(device) - mu) / (std + 1e-8)
        logits = model(batch)
        all_logits.append(logits.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(all_logits, dim=-1)
    avg_probs = probs.mean(dim=0)
    pred_idx = torch.argmax(avg_probs).item()
    confidence = avg_probs[pred_idx].item()
    return class_names[pred_idx], confidence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--mamba_path', type=str, required=True)
    parser.add_argument('--gru_path', type=str, required=True)
    parser.add_argument('--swap', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Inference Node Initialized ---")
    
    window_batch = process_file(args.csv, swap_muscles=args.swap)
    
    # 1. BiMamba
    model_mamba, _, meta_mamba = build_model_from_checkpoint(args.mamba_path, device)
    label_mamba, conf_mamba = run_inference(model_mamba, window_batch, device, meta_mamba)
    print(f"\n[BiMamba Results]\n  Prediction: {label_mamba}\n  Confidence: {conf_mamba:.2%}")

    # 2. GRU
    model_gru, _, meta_gru = build_model_from_checkpoint(args.gru_path, device)
    label_gru, conf_gru = run_inference(model_gru, window_batch, device, meta_gru)
    print(f"\n[GRU Results]\n  Prediction: {label_gru}\n  Confidence: {conf_gru:.2%}")

if __name__ == "__main__":
    main()
