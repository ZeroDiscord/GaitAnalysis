#!/usr/bin/env python
"""
EDA Runner
==========
Command-line entry point that orchestrates the full EDA pipeline:

    python EDA/eda_runner.py --data_dir path/to/data

Steps:
  1. Load all patient CSV files
  2. Extract time + frequency features (windowed)
  3. Build unified feature DataFrame
  4. Save feature table as CSV
  5. Run PCA / LDA
  6. Generate all plots
  7. Save outputs to EDA/outputs/
"""

import argparse
import os
import sys

# Ensure the EDA package can be imported when run from repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from EDA.eda_features import build_feature_dataframe
from EDA.eda_visualization import generate_all_plots


def main():
    parser = argparse.ArgumentParser(
        description='Run Exploratory Data Analysis on EMG gait data.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing class sub-folders with CSV files.')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(SCRIPT_DIR, 'outputs'),
                        help='Directory to save outputs (default: EDA/outputs/).')
    parser.add_argument('--window_size', type=int, default=2000,
                        help='Sliding window size in samples (default: 2000).')
    parser.add_argument('--stride', type=int, default=1000,
                        help='Stride between consecutive windows (default: 1000).')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha coefficient for torque (default: 1.0).')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta coefficient for torque (default: 1.0).')
    parser.add_argument('--fs', type=float, default=1000.0,
                        help='Sampling frequency in Hz (default: 1000).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1–3: Feature extraction ──────────────────────────────────────
    print('=' * 60)
    print('  EMG Gait EDA Pipeline')
    print('=' * 60)
    print(f'\nData directory : {args.data_dir}')
    print(f'Window size    : {args.window_size}')
    print(f'Stride         : {args.stride}')
    print(f'Alpha / Beta   : {args.alpha} / {args.beta}')
    print(f'Sampling freq  : {args.fs} Hz')
    print()

    print('Loading CSVs and extracting features …')
    df, classes = build_feature_dataframe(
        data_dir=args.data_dir,
        alpha=args.alpha,
        beta=args.beta,
        window_size=args.window_size,
        stride=args.stride,
        fs=args.fs,
    )
    print(f'   → {len(df)} windows from {df["patient_id"].nunique()} patients')
    print(f'   → {len(df.columns)} columns, Classes: {classes}')

    # ── Step 4: Save feature table ────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'eda_feature_table.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nFeature table saved → {csv_path}')

    # ── Step 5–6: Visualisations ──────────────────────────────────────────
    generate_all_plots(df, classes, args.output_dir)

    # ── Done ──────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  EDA complete — all artefacts in', args.output_dir)
    print('=' * 60)


if __name__ == '__main__':
    main()
