"""
Quick dataset inspection utility.
Prints shape and first few rows of one CSV from each class sub-folder.

Usage:
    python inspect_data.py --data_dir path/to/Datasets
"""

import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Inspect raw EMG CSV files.')
    parser.add_argument('--data_dir', type=str, default='Datasets/',
                        help='Root directory containing class sub-folders with CSV files.')
    args = parser.parse_args()

    folders = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    )

    for f in folders:
        f_path = os.path.join(args.data_dir, f)
        files = sorted(x for x in os.listdir(f_path) if x.endswith('.csv'))
        if files:
            first_file = os.path.join(f_path, files[0])
            # CSVs are headerless: col 0 = agonist (TA), col 1 = antagonist (GA)
            df = pd.read_csv(first_file, header=None)
            print(f"\n--- {f} / {files[0]} ---")
            print(f"Shape: {df.shape}")
            print(f"Files in folder: {len(files)}")
            print(df.head())


if __name__ == '__main__':
    main()
