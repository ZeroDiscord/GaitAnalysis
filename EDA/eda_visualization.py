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
# Signal waveform plots (raw time-series with interpretations)
# ---------------------------------------------------------------------------

# Consistent, colourblind-friendly palette for the two EMG channels
_EMG_COLORS = {
    'agonist':    '#2196F3',   # vivid blue
    'antagonist': '#F44336',   # vivid red
    'torque':     '#7C4DFF',   # deep purple
    'stiffness':  '#FF9800',   # amber
}


def _find_interesting_clip(signal, clip_len=1000):
    """
    Slide a window across *signal* and return the start index of the segment
    with the highest variance — this is typically where the most informative
    gait-cycle dynamics live.
    """
    if len(signal) <= clip_len:
        return 0
    best_start, best_var = 0, -1
    step = max(clip_len // 4, 1)
    for s in range(0, len(signal) - clip_len, step):
        v = np.var(signal[s:s + clip_len])
        if v > best_var:
            best_var = v
            best_start = s
    return best_start


def plot_signal_waveforms(data_dir, classes_with_paths, alpha, beta, output_dir,
                          clip_len=1000):
    """
    For each class, pick one representative patient, auto-select an
    interesting ~clip_len segment, and plot:

      Panel 1  —  Agonist + Antagonist overlaid (interaction view)
      Panel 2  —  Torque  (net activation proxy)
      Panel 3  —  Stiffness  (co-contraction proxy, filled)

    Also produces a comparison grid with one column per class.
    """
    all_classes = sorted(classes_with_paths.keys())

    # ── Per-class individual plots ──────────────────────────────────────
    for cls_name in all_classes:
        file_list = classes_with_paths[cls_name]
        if not file_list:
            continue
        csv_path, patient_id = file_list[0]

        df_raw = pd.read_csv(csv_path, header=None)
        e_ago_full = np.nan_to_num(df_raw.iloc[:, 0].values.astype(float))
        e_ant_full = np.nan_to_num(df_raw.iloc[:, 1].values.astype(float))

        # Smart clip: pick the most dynamic region
        combined = e_ant_full + e_ago_full
        start = _find_interesting_clip(combined, clip_len)
        end = start + clip_len
        e_ant = e_ant_full[start:end]
        e_ago = e_ago_full[start:end]
        torque = alpha * e_ant - beta * e_ago
        stiffness = e_ant + e_ago
        t = np.arange(len(e_ant))

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1, 1]})
        fig.suptitle(f'{cls_name}  —  samples {start}\u2013{end}  (patient: {patient_id})',
                     fontsize=13, fontweight='bold')

        # Panel 1: Agonist + Antagonist overlaid
        ax = axes[0]
        ax.plot(t, e_ago, color=_EMG_COLORS['agonist'], linewidth=0.8,
                alpha=0.85, label='Agonist (E_ago)')
        ax.plot(t, e_ant, color=_EMG_COLORS['antagonist'], linewidth=0.8,
                alpha=0.85, label='Antagonist (E_ant)')
        ax.fill_between(t, e_ago, e_ant, alpha=0.08, color='grey')
        ax.set_ylabel('EMG amplitude', fontsize=10)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.set_title('Agonist \u2013 Antagonist Interaction', fontsize=11, pad=4)
        ax.grid(True, alpha=0.2)
        # Interpretation box
        ax.text(0.01, 0.97,
                'Healthy: alternating peaks (reciprocal inhibition)\n'
                'Pathological: overlapping peaks (co-contraction)',
                transform=ax.transAxes, fontsize=7.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFFDE7', ec='#ccc', alpha=0.9))

        # Panel 2: Torque
        ax = axes[1]
        ax.plot(t, torque, color=_EMG_COLORS['torque'], linewidth=0.8)
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.fill_between(t, 0, torque, where=torque >= 0,
                        color=_EMG_COLORS['antagonist'], alpha=0.12, label='Ant-dominant')
        ax.fill_between(t, 0, torque, where=torque < 0,
                        color=_EMG_COLORS['agonist'], alpha=0.12, label='Ago-dominant')
        ax.set_ylabel('Torque', fontsize=10)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.text(0.01, 0.93,
                'Clear oscillation = normal motor control\n'
                'Flat / erratic = impaired control',
                transform=ax.transAxes, fontsize=7.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFFDE7', ec='#ccc', alpha=0.9))

        # Panel 3: Stiffness
        ax = axes[2]
        ax.fill_between(t, 0, stiffness, color=_EMG_COLORS['stiffness'], alpha=0.35)
        ax.plot(t, stiffness, color=_EMG_COLORS['stiffness'], linewidth=0.8)
        ax.set_ylabel('Stiffness', fontsize=10)
        ax.set_xlabel('Sample index', fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.text(0.01, 0.93,
                'High sustained stiffness = spasticity / co-contraction\n'
                'Low during swing = healthy gait',
                transform=ax.transAxes, fontsize=7.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFFDE7', ec='#ccc', alpha=0.9))

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        safe = cls_name.replace(' ', '_').lower()
        fname = f'waveform_{safe}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f'  \u2713 Saved {fname}')

    # ── Multi-class comparison grid ─────────────────────────────────────
    n_cls = len(all_classes)
    fig, axes = plt.subplots(3, n_cls, figsize=(5 * n_cls, 9), sharex='col',
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    if n_cls == 1:
        axes = axes.reshape(3, 1)
    fig.suptitle('EMG Waveform Comparison Across Classes', fontsize=14, fontweight='bold')

    for col, cls_name in enumerate(all_classes):
        file_list = classes_with_paths.get(cls_name, [])
        if not file_list:
            continue
        csv_path, pid = file_list[0]
        df_raw = pd.read_csv(csv_path, header=None)
        e_ago_full = np.nan_to_num(df_raw.iloc[:, 0].values.astype(float))
        e_ant_full = np.nan_to_num(df_raw.iloc[:, 1].values.astype(float))
        start = _find_interesting_clip(e_ant_full + e_ago_full, clip_len)
        e_ant = e_ant_full[start:start + clip_len]
        e_ago = e_ago_full[start:start + clip_len]
        torque = alpha * e_ant - beta * e_ago
        stiffness = e_ant + e_ago
        t = np.arange(len(e_ant))

        # Row 0 — EMG interaction
        ax = axes[0, col]
        ax.plot(t, e_ago, color=_EMG_COLORS['agonist'], lw=0.6, alpha=0.8, label='Ago')
        ax.plot(t, e_ant, color=_EMG_COLORS['antagonist'], lw=0.6, alpha=0.8, label='Ant')
        ax.fill_between(t, e_ago, e_ant, alpha=0.06, color='grey')
        ax.set_title(cls_name, fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('EMG', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

        # Row 1 — Torque
        ax = axes[1, col]
        ax.plot(t, torque, color=_EMG_COLORS['torque'], lw=0.6)
        ax.axhline(0, color='grey', lw=0.4, ls='--')
        ax.fill_between(t, 0, torque, where=torque >= 0,
                        color=_EMG_COLORS['antagonist'], alpha=0.1)
        ax.fill_between(t, 0, torque, where=torque < 0,
                        color=_EMG_COLORS['agonist'], alpha=0.1)
        if col == 0:
            ax.set_ylabel('Torque', fontsize=10)
        ax.grid(True, alpha=0.2)

        # Row 2 — Stiffness
        ax = axes[2, col]
        ax.fill_between(t, 0, stiffness, color=_EMG_COLORS['stiffness'], alpha=0.3)
        ax.plot(t, stiffness, color=_EMG_COLORS['stiffness'], lw=0.6)
        if col == 0:
            ax.set_ylabel('Stiffness', fontsize=10)
        ax.set_xlabel('Sample', fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, 'waveform_comparison.png'), dpi=150)
    plt.close(fig)
    print('  \u2713 Saved waveform_comparison.png')


# ---------------------------------------------------------------------------
# Convenience: run everything
# ---------------------------------------------------------------------------

def generate_all_plots(df, classes, output_dir, data_dir=None, alpha=1.0, beta=1.0):
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

    # Signal waveforms (needs raw CSV paths)
    if data_dir:
        from EDA.eda_features import discover_csv_files
        records = discover_csv_files(data_dir)
        classes_with_paths = {}
        for csv_path, cls_name, patient_id in records:
            classes_with_paths.setdefault(cls_name, []).append((csv_path, patient_id))
        plot_signal_waveforms(data_dir, classes_with_paths, alpha, beta, output_dir)
    else:
        print('  (Skipping waveform plots — no data_dir provided)')

    print('All plots saved to', output_dir)
