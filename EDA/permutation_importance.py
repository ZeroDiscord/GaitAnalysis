#!/usr/bin/env python
"""
Permutation Importance Evaluator
================================
Loads the EDA feature table and trains a RandomForestClassifier to compute
permutation importance for all extracted features.
Visualises the top predictive features and saves output.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')

# Ensure script works from repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')
CSV_PATH = os.path.join(OUTPUT_DIR, 'eda_feature_table.csv')

def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feature table not found at {csv_path}. Please run EDA pipeline first.")
    
    df = pd.read_csv(csv_path)
    
    # Separate metadata and features
    meta_cols = {'patient_id', 'window_id', 'label', 'class_name'}
    feat_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feat_cols].values
    y = df['label'].values
    
    # Handle NaNs/Infs that might remain
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y, feat_cols

def compute_permutation_importance(X, y, feature_names, n_repeats=10, random_state=42):
    print(f"Dataset shape: {X.shape}")
    print("Training RandomForestClassifier...")
    
    # Train-test split mostly to evaluate generalization importance, though we could use OOB or full set.
    # Using hold-out set for more robust importance metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
    
    # Scale features (though RF doesn't strictly need it, it's good practice for numerical stability)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    print(f"RF Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
    
    print(f"Computing permutation importance ({n_repeats} repeats)...")
    result = permutation_importance(model, X_test_scaled, y_test, n_repeats=n_repeats, 
                                    random_state=random_state, n_jobs=-1)
    
    # Organize results
    importances = result.importances_mean
    std = result.importances_std
    
    # Create DataFrame for easy sorting
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'std': std
    })
    
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    return imp_df

def plot_importance(imp_df, top_n=20, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    
    # Select top N
    plot_df = imp_df.head(top_n).copy()
    
    # Sort ascending for horizontal bar plot
    plot_df = plot_df.sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df['importance'], xerr=plot_df['std'], align='center', 
            color='#4C72B0', ecolor='black', capsize=3, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['feature'])
    ax.set_xlabel('Mean Accuracy Decrease (Permutation Importance)')
    ax.set_title(f'Top {top_n} Features by Permutation Importance (Random Forest)')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    out_file = os.path.join(output_dir, 'permutation_importance.png')
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_file}")

def main():
    print("="*60)
    print(" Permutation Importance Analysis ")
    print("="*60)
    
    try:
        X, y, feature_names = load_data(CSV_PATH)
        imp_df = compute_permutation_importance(X, y, feature_names)
        
        print("\nTop 15 Features:")
        print(imp_df.head(15).to_string(index=False))
        
        plot_importance(imp_df, top_n=20)
        
        # Save exact scores to CSV
        csv_out = os.path.join(OUTPUT_DIR, 'permutation_importance_scores.csv')
        imp_df.to_csv(csv_out, index=False)
        print(f"Saved scores to {csv_out}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
