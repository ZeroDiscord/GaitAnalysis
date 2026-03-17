"""
EDA Feature Extraction Module
=============================
Mirrors the base feature engineering from dataset.py (e_ant, e_ago, torque, stiffness)
and extends it with time-domain and frequency-domain EMG features.

Does NOT import or modify dataset.py — replicates the same logic for independence.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import welch

# Gait phase module (lives one level up from EDA/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from gait_phase import extract_gait_cycle_features as _gait_cycle_feats
    _GAIT_PHASE_AVAILABLE = True
except ImportError:
    _GAIT_PHASE_AVAILABLE = False


# ---------------------------------------------------------------------------
# CSV discovery (mirrors create_dataloaders logic in dataset.py)
# ---------------------------------------------------------------------------

def discover_csv_files(data_dir):
    """
    Scan *data_dir* for class sub-folders whose names contain '_' (e.g. '01_Normal'),
    returning a list of (csv_path, class_name, patient_id) tuples.
    """
    records = []
    raw_folders = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    for folder in raw_folders:
        if '_' not in folder:
            continue
        class_name = folder.split('_', 1)[1]
        folder_path = os.path.join(data_dir, folder)
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith('.csv'):
                csv_path = os.path.join(folder_path, fname)
                patient_id = os.path.splitext(fname)[0]
                records.append((csv_path, class_name, patient_id))
    return records


# ---------------------------------------------------------------------------
# Time-domain features (per window, per channel)
# ---------------------------------------------------------------------------

def _rms(x):
    return np.sqrt(np.mean(x ** 2))


def _mav(x):
    return np.mean(np.abs(x))


def _variance(x):
    return np.var(x)


def _waveform_length(x):
    return np.sum(np.abs(np.diff(x)))


def _zero_crossing_rate(x):
    signs = np.sign(x)
    signs[signs == 0] = 1  # treat exact zero as positive
    return np.sum(np.abs(np.diff(signs)) > 0) / len(x)


def _slope_sign_changes(x):
    d = np.diff(x)
    signs = np.sign(d)
    signs[signs == 0] = 1
    return np.sum(np.abs(np.diff(signs)) > 0)


def _iemg(x):
    return np.sum(np.abs(x))


TIME_DOMAIN_FNS = {
    'rms': _rms,
    'mav': _mav,
    'variance': _variance,
    'wl': _waveform_length,
    'zcr': _zero_crossing_rate,
    'ssc': _slope_sign_changes,
    'iemg': _iemg,
}


# ---------------------------------------------------------------------------
# Frequency-domain features (per window, per channel)
# ---------------------------------------------------------------------------

def _freq_features(x, fs=1000.0):
    """
    Compute frequency-domain features using Welch's PSD estimate.
    Returns a dict of {feature_name: value}.
    """
    nperseg = min(256, len(x))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)

    total_power = np.sum(psd)
    if total_power == 0:
        return {
            'mean_freq': 0.0,
            'median_freq': 0.0,
            'spectral_entropy': 0.0,
            'peak_freq': 0.0,
            'total_power': 0.0,
        }

    # Mean frequency
    mean_freq = np.sum(freqs * psd) / total_power

    # Median frequency
    cumulative = np.cumsum(psd)
    median_freq = freqs[np.searchsorted(cumulative, total_power / 2)]

    # Spectral entropy
    psd_norm = psd / total_power
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))

    # Peak frequency
    peak_freq = freqs[np.argmax(psd)]

    return {
        'mean_freq': mean_freq,
        'median_freq': median_freq,
        'spectral_entropy': spectral_entropy,
        'peak_freq': peak_freq,
        'total_power': total_power,
    }


FREQ_FEATURE_NAMES = ['mean_freq', 'median_freq', 'spectral_entropy', 'peak_freq', 'total_power']


# ---------------------------------------------------------------------------
# Per-window feature row
# ---------------------------------------------------------------------------

def _extract_window_features(e_ant_win, e_ago_win, alpha, beta, fs=1000.0):
    """
    Given a single window of e_ant (GA) / e_ago (TA) values, compute ALL features and
    return a flat dict including gait cycle phase features.
    """
    row = {}

    # Base features (aggregated scalars to match window-level granularity)
    row['e_ant'] = np.mean(e_ant_win)
    row['e_ago'] = np.mean(e_ago_win)

    torque_signal = alpha * e_ant_win - beta * e_ago_win
    stiffness_signal = e_ant_win + e_ago_win
    row['torque'] = np.mean(torque_signal)
    row['stiffness'] = np.mean(stiffness_signal)

    # Time-domain features per channel
    channels = {'ant': e_ant_win, 'ago': e_ago_win}
    for ch_name, ch_data in channels.items():
        for feat_name, fn in TIME_DOMAIN_FNS.items():
            row[f'{feat_name}_{ch_name}'] = fn(ch_data)

    # Frequency-domain features per channel
    for ch_name, ch_data in channels.items():
        freq_feats = _freq_features(ch_data, fs=fs)
        for feat_name, val in freq_feats.items():
            row[f'{feat_name}_{ch_name}'] = val

    # Gait cycle phase features (TA = e_ago col0, GA = e_ant col1)
    if _GAIT_PHASE_AVAILABLE:
        try:
            phase_feats = _gait_cycle_feats(e_ago_win, e_ant_win, fs)
            # Flatten: prefix with 'gp_', skip non-numeric fields
            for k, v in phase_feats.items():
                if k == 'method':
                    continue
                row[f'gp_{k}'] = float(v)
        except Exception:
            pass  # Silently skip if window too short for gait phase

    return row


# ---------------------------------------------------------------------------
# Public API — build the full feature dataframe
# ---------------------------------------------------------------------------

def build_feature_dataframe(data_dir, alpha=1.0, beta=1.0,
                            window_size=2000, stride=1000, fs=1000.0):
    """
    Scan *data_dir*, extract per-window features from every patient CSV,
    and return a unified pandas DataFrame.

    Parameters
    ----------
    data_dir : str
        Root directory containing class sub-folders.
    alpha, beta : float
        Coefficients for torque = alpha * e_ant - beta * e_ago.
    window_size : int
        Number of samples per window.
    stride : int
        Step between consecutive windows.
    fs : float
        Assumed sampling frequency (Hz), used for frequency features.

    Returns
    -------
    pd.DataFrame
        One row per window, columns for every feature + metadata.
    """
    records = discover_csv_files(data_dir)
    if not records:
        raise ValueError(f"No CSV files found under {data_dir}")

    # Collect unique class names and build class_to_idx
    classes = sorted({r[1] for r in records})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    rows = []
    global_window_id = 0

    for csv_path, class_name, patient_id in records:
        df = pd.read_csv(csv_path, header=None)
        e_ago = df.iloc[:, 0].values.astype(np.float64)   # col 0 = agonist
        e_ant = df.iloc[:, 1].values.astype(np.float64)   # col 1 = antagonist

        # Replace NaN / Inf (mirrors torch.nan_to_num in dataset.py)
        e_ant = np.nan_to_num(e_ant, nan=0.0, posinf=0.0, neginf=0.0)
        e_ago = np.nan_to_num(e_ago, nan=0.0, posinf=0.0, neginf=0.0)

        L = len(e_ant)
        start = 0
        generated = False

        while start + window_size <= L:
            win_ant = e_ant[start:start + window_size]
            win_ago = e_ago[start:start + window_size]

            feat_row = _extract_window_features(win_ant, win_ago, alpha, beta, fs)
            feat_row['label'] = class_to_idx[class_name]
            feat_row['class_name'] = class_name
            feat_row['patient_id'] = patient_id
            feat_row['window_id'] = global_window_id

            rows.append(feat_row)
            global_window_id += 1
            start += stride
            generated = True

        # If the entire file is shorter than window_size, use it entirely
        if not generated:
            win_ant = e_ant
            win_ago = e_ago
            feat_row = _extract_window_features(win_ant, win_ago, alpha, beta, fs)
            feat_row['label'] = class_to_idx[class_name]
            feat_row['class_name'] = class_name
            feat_row['patient_id'] = patient_id
            feat_row['window_id'] = global_window_id
            rows.append(feat_row)
            global_window_id += 1

    feature_df = pd.DataFrame(rows)

    # Reorder columns: metadata first, then features
    meta_cols = ['patient_id', 'window_id', 'label', 'class_name']
    feat_cols = [c for c in feature_df.columns if c not in meta_cols]
    feature_df = feature_df[meta_cols + sorted(feat_cols)]

    return feature_df, classes
