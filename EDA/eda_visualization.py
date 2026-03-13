"""
EDA Visualization Module
========================
Generates all EDA plots from a feature DataFrame produced by eda_features.py.
Every figure is saved to the *output_dir* path (default: EDA/outputs/).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_columns(df):
    """Return only numeric feature columns (exclude metadata)."""
    meta = {'patient_id', 'window_id', 'label', 'class_name'}
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


def _class_palette(classes):
    """Build a colour palette dict for the given class list."""
    base = sns.color_palette('Set2', n_colors=max(len(classes), 3))
    return {cls: base[i] for i, cls in enumerate(classes)}


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------

def plot_scatter(df, x_col, y_col, classes, output_dir):
    """Scatter plot of *x_col* vs *y_col*, coloured by class, with convex hull outlines."""
    palette = _class_palette(classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in classes:
        subset = df[df['class_name'] == cls]
        color = palette[cls]
        ax.scatter(subset[x_col], subset[y_col],
                   label=cls, alpha=0.6, s=30, color=color, edgecolors='w', linewidths=0.3)

        # Draw convex hull (need >= 3 points)
        pts = subset[[x_col, y_col]].dropna().values
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_vertices = np.append(hull.vertices, hull.vertices[0])  # close the polygon
                ax.plot(pts[hull_vertices, 0], pts[hull_vertices, 1],
                        color=color, linewidth=1.5, linestyle='--')
                hull_polygon = Polygon(pts[hull.vertices], closed=True,
                                       facecolor=color, alpha=0.1, edgecolor=color, linewidth=1.5)
                ax.add_patch(hull_polygon)
            except Exception:
                pass  # degenerate hull (collinear points, etc.)

    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(f'{y_col}  vs  {x_col}', fontsize=14)
    ax.legend(title='Class', fontsize=10)
    ax.grid(True, alpha=0.3)
    fname = f'scatter_{x_col}_vs_{y_col}.png'
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f'  ✓ Saved {fname}')


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df, output_dir):
    feat_cols = _get_feature_columns(df)
    corr = df[feat_cols].corr()

    g = sns.clustermap(corr, cmap='coolwarm', center=0,
                       linewidths=0.5, figsize=(max(12, len(feat_cols) * 0.45),
                                                max(10, len(feat_cols) * 0.4)))
    g.fig.suptitle('Feature Correlation Clustermap', fontsize=14, y=1.02)
    g.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(g.fig)
    print('  ✓ Saved feature_correlation_heatmap.png')


# ---------------------------------------------------------------------------
# PCA projection
# ---------------------------------------------------------------------------

def plot_pca(df, classes, output_dir):
    feat_cols = _get_feature_columns(df)
    X = df[feat_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    palette = _class_palette(classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in classes:
        mask = df['class_name'].values == cls
        ax.scatter(components[mask, 0], components[mask, 1],
                   label=cls, alpha=0.6, s=30, color=palette[cls], edgecolors='w', linewidths=0.3)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({ev[0] * 100:.1f}% var)', fontsize=12)
    ax.set_ylabel(f'PC2 ({ev[1] * 100:.1f}% var)', fontsize=12)
    ax.set_title('PCA Projection (PC1 vs PC2)', fontsize=14)
    ax.legend(title='Class', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_projection.png'), dpi=150)
    plt.close(fig)
    print('  ✓ Saved pca_projection.png')


# ---------------------------------------------------------------------------
# LDA projection
# ---------------------------------------------------------------------------

def plot_lda(df, classes, output_dir):
    feat_cols = _get_feature_columns(df)
    X = df[feat_cols].values
    y = df['label'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = StandardScaler().fit_transform(X)

    n_components = min(len(classes) - 1, X.shape[1], 2)
    if n_components < 1:
        print('  ⚠ Skipping LDA: need at least 2 classes.')
        return

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    components = lda.fit_transform(X, y)

    palette = _class_palette(classes)
    fig, ax = plt.subplots(figsize=(8, 6))

    if n_components == 1:
        # Only 1 LD axis available (2-class problem) — plot as 1-D strip
        for cls in classes:
            mask = df['class_name'].values == cls
            ax.scatter(components[mask, 0], np.zeros_like(components[mask, 0]),
                       label=cls, alpha=0.6, s=30, color=palette[cls], edgecolors='w', linewidths=0.3)
        ax.set_xlabel('LD1', fontsize=12)
        ax.set_yticks([])
        ax.set_title('LDA Projection (LD1)', fontsize=14)
    else:
        for cls in classes:
            mask = df['class_name'].values == cls
            ax.scatter(components[mask, 0], components[mask, 1],
                       label=cls, alpha=0.6, s=30, color=palette[cls], edgecolors='w', linewidths=0.3)
        ax.set_xlabel('LD1', fontsize=12)
        ax.set_ylabel('LD2', fontsize=12)
        ax.set_title('LDA Projection (LD1 vs LD2)', fontsize=14)

    ax.legend(title='Class', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'lda_projection.png'), dpi=150)
    plt.close(fig)
    print('  ✓ Saved lda_projection.png')


# ---------------------------------------------------------------------------
# Feature distribution plots (histograms + boxplots)
# ---------------------------------------------------------------------------

KEY_DISTRIBUTION_FEATURES = [
    'rms_ant', 'rms_ago',
    'torque', 'stiffness',
    'mean_freq_ant', 'mean_freq_ago',
]


def plot_feature_distributions(df, classes, output_dir):
    """
    For each key feature, plot a per-class histogram (top) and boxplot (bottom).
    """
    feats = [f for f in KEY_DISTRIBUTION_FEATURES if f in df.columns]
    if not feats:
        print('  ⚠ No key features found for distribution plots.')
        return

    n = len(feats)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    palette = _class_palette(classes)

    for i, feat in enumerate(feats):
        ax_hist = axes[i, 0]
        ax_box = axes[i, 1]

        # Histogram
        for cls in classes:
            subset = df.loc[df['class_name'] == cls, feat].dropna()
            ax_hist.hist(subset, bins=30, alpha=0.5, label=cls, color=palette[cls])
        ax_hist.set_title(f'{feat} — Histogram', fontsize=11)
        ax_hist.set_xlabel(feat)
        ax_hist.set_ylabel('Count')
        ax_hist.legend(fontsize=8)

        # Boxplot
        box_data = [df.loc[df['class_name'] == cls, feat].dropna().values for cls in classes]
        bp = ax_box.boxplot(box_data, labels=classes, patch_artist=True)
        for patch, cls in zip(bp['boxes'], classes):
            patch.set_facecolor(palette[cls])
            patch.set_alpha(0.6)
        ax_box.set_title(f'{feat} — Boxplot', fontsize=11)
        ax_box.set_ylabel(feat)

    fig.suptitle('Feature Distributions by Class', fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ Saved feature_distributions.png')


# ---------------------------------------------------------------------------
# Convenience: run everything
# ---------------------------------------------------------------------------

def generate_all_plots(df, classes, output_dir):
    """Generate every visualisation and save to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    print('\nGenerating EDA visualisations …')

    # Scatter plots
    scatter_pairs = [
        ('torque', 'stiffness'),
        ('rms_ant', 'rms_ago'),
    ]
    for x, y in scatter_pairs:
        if x in df.columns and y in df.columns:
            plot_scatter(df, x, y, classes, output_dir)

    plot_correlation_heatmap(df, output_dir)
    plot_pca(df, classes, output_dir)
    plot_lda(df, classes, output_dir)
    plot_feature_distributions(df, classes, output_dir)

    print('✅ All plots saved to', output_dir)
