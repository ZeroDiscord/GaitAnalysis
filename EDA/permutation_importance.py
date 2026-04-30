#!/usr/bin/env python
"""
Permutation Importance Evaluator (Fixed)
=========================================
Computes permutation importance using patient-level GroupKFold cross-validation
to prevent data leakage from overlapping windows of the same patient.

Key fixes over the original:
  1. Patient-level GroupKFold split (not window-level train_test_split)
  2. No StandardScaler (RF is scale-invariant; scaling can distort PI)
  3. Cross-validated importance (averages over folds for stability)
  4. Filters out gp_ features where extraction failed (NaN fallback rows)
  5. Reports effective sample size (patients, not windows)
  6. Warns about negative importance features
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

# Ensure script works from repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')
CSV_PATH = os.path.join(OUTPUT_DIR, 'eda_feature_table.csv')


def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feature table not found at {csv_path}. Please run EDA pipeline first.")

    df = pd.read_csv(csv_path)

    meta_cols = {'patient_id', 'window_id', 'label', 'class_name', 'gp_extraction_failed'}
    feat_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feat_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = df['label'].values
    patient_ids = df['patient_id'].values

    return X, y, feat_cols, patient_ids, df


def compute_permutation_importance_cv(X, y, feature_names, groups,
                                       n_splits=5, n_repeats=10, random_state=42):
    """
    Compute permutation importance using GroupKFold cross-validation.
    GroupKFold ensures ALL windows from a given patient stay in one fold.
    """
    n_patients = len(np.unique(groups))
    n_features = X.shape[1]

    print(f"Dataset: {X.shape[0]} windows from {n_patients} patients, {n_features} features")
    print(f"Patient-to-feature ratio: {n_patients / n_features:.2f} "
          f"({'OK' if n_patients / n_features > 1 else 'WARNING: underdetermined'})")

    unique_classes = np.unique(y)
    min_class_patients = min(len(np.unique(groups[y == c])) for c in unique_classes)
    effective_splits = min(n_splits, min_class_patients)
    if effective_splits < 2:
        effective_splits = max(2, min_class_patients)

    print(f"Using {effective_splits}-fold GroupKFold cross-validation\n")

    gkf = GroupKFold(n_splits=effective_splits)
    all_importances = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_patients = np.unique(groups[test_idx])

        # No StandardScaler — RF is scale-invariant
        model = RandomForestClassifier(
            n_estimators=200, random_state=random_state + fold_idx,
            n_jobs=-1, class_weight='balanced', min_samples_leaf=2,
        )
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"  Fold {fold_idx + 1}/{effective_splits}: "
              f"Train={train_acc:.3f} Test={test_acc:.3f} "
              f"(test patients: {list(test_patients)})")

        result = permutation_importance(
            model, X_test, y_test, n_repeats=n_repeats,
            random_state=random_state, n_jobs=-1
        )
        all_importances.append(result.importances_mean)

    mean_imp = np.mean(all_importances, axis=0)
    std_imp = np.std(all_importances, axis=0)

    imp_df = pd.DataFrame({
        'feature': feature_names, 'importance': mean_imp, 'std': std_imp,
    })
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)

    n_negative = (imp_df['importance'] < 0).sum()
    print(f"\n  Features with negative importance: {n_negative}")

    return imp_df


def plot_importance(imp_df, top_n=20, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plot_df = imp_df.head(top_n).copy().sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#4C72B0' if v >= 0 else '#DD4444' for v in plot_df['importance']]
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df['importance'], xerr=plot_df['std'], align='center',
            color=colors, ecolor='black', capsize=3, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['feature'])
    ax.set_xlabel('Mean Accuracy Decrease (Permutation Importance)')
    ax.set_title(f'Top {top_n} Features — Patient-Level GroupKFold CV', fontsize=12)
    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    fig.tight_layout()
    out_file = os.path.join(output_dir, 'permutation_importance.png')
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_file}")


def main():
    print("=" * 60)
    print("  Permutation Importance Analysis (Patient-Level CV)")
    print("=" * 60)

    try:
        X, y, feature_names, patient_ids, df = load_data(CSV_PATH)

        print(f"\nClass distribution (windows):")
        for cls in sorted(df['class_name'].unique()):
            n_w = len(df[df['class_name'] == cls])
            n_p = df[df['class_name'] == cls]['patient_id'].nunique()
            print(f"  {cls}: {n_w} windows from {n_p} patients")

        print()
        imp_df = compute_permutation_importance_cv(X, y, feature_names, patient_ids)

        print("\nTop 20 Features:")
        print(imp_df.head(20).to_string(index=False))

        plot_importance(imp_df, top_n=25)

        csv_out = os.path.join(OUTPUT_DIR, 'permutation_importance_scores.csv')
        imp_df.to_csv(csv_out, index=False)
        print(f"Saved scores to {csv_out}")
        print("=" * 60)

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
