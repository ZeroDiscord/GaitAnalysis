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
from sklearn.decomposition import PCA, FastICA
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
    print(f'  [OK] Saved {fname}')


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df, output_dir):
    feat_cols = _get_feature_columns(df)
    # Aggregate to patient-level to avoid within-subject autocorrelation
    patient_df = df.groupby(['patient_id', 'class_name']).mean(numeric_only=True).reset_index()
    corr = patient_df[feat_cols].corr()
    # NaN correlations arise from gait phase features that failed extraction
    # (returned NaN).  Fill with 0.0 (uncorrelated) so clustering doesn't crash.
    corr = corr.fillna(0.0)

    g = sns.clustermap(corr, cmap='coolwarm', center=0,
                       linewidths=0.5, figsize=(max(12, len(feat_cols) * 0.45),
                                                max(10, len(feat_cols) * 0.4)))
    g.fig.suptitle('Feature Correlation Clustermap', fontsize=14, y=1.02)
    g.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(g.fig)
    print('  [OK] Saved feature_correlation_heatmap.png')


# ---------------------------------------------------------------------------
# PCA projection
# ---------------------------------------------------------------------------

def plot_pca(df, classes, output_dir):
    feat_cols = _get_feature_columns(df)
    # Aggregate to patient-level to prevent within-subject autocorrelation
    patient_df = df.groupby(['patient_id', 'class_name']).mean(numeric_only=True).reset_index()
    X_patient = patient_df[feat_cols].values
    X_patient = np.nan_to_num(X_patient, nan=0.0, posinf=0.0, neginf=0.0)
    X_patient = StandardScaler().fit_transform(X_patient)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_patient)

    palette = _class_palette(classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in classes:
        mask = patient_df['class_name'].values == cls
        ax.scatter(components[mask, 0], components[mask, 1],
                   label=cls, alpha=0.8, s=60, color=palette[cls], edgecolors='k', linewidths=0.5)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({ev[0] * 100:.1f}% var)', fontsize=12)
    ax.set_ylabel(f'PC2 ({ev[1] * 100:.1f}% var)', fontsize=12)
    ax.set_title('PCA Projection — Patient-Level (PC1 vs PC2)', fontsize=13)
    ax.legend(title='Class', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_projection.png'), dpi=150)
    plt.close(fig)
    print('  [OK] Saved pca_projection.png (patient-level)')


# ---------------------------------------------------------------------------
# LDA projection
# ---------------------------------------------------------------------------

def plot_lda(df, classes, output_dir):
    feat_cols = _get_feature_columns(df)
    # Aggregate to patient-level
    patient_df = df.groupby(['patient_id', 'class_name', 'label']).mean(numeric_only=True).reset_index()
    X = patient_df[feat_cols].values
    y = patient_df['label'].values
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
            mask = patient_df['class_name'].values == cls
            ax.scatter(components[mask, 0], np.zeros_like(components[mask, 0]),
                       label=cls, alpha=0.8, s=60, color=palette[cls], edgecolors='k', linewidths=0.5)
        ax.set_xlabel('LD1', fontsize=12)
        ax.set_yticks([])
        ax.set_title('LDA Projection (LD1)', fontsize=14)
    else:
        for cls in classes:
            mask = patient_df['class_name'].values == cls
            ax.scatter(components[mask, 0], components[mask, 1],
                       label=cls, alpha=0.8, s=60, color=palette[cls], edgecolors='k', linewidths=0.5)
        ax.set_xlabel('LD1', fontsize=12)
        ax.set_ylabel('LD2', fontsize=12)
        ax.set_title('LDA Projection — Patient-Level (LD1 vs LD2)', fontsize=13)

    ax.legend(title='Class', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'lda_projection.png'), dpi=150)
    plt.close(fig)
    print('  [OK] Saved lda_projection.png')


# ---------------------------------------------------------------------------
# ICA projection
# ---------------------------------------------------------------------------

def plot_ica(df, classes, output_dir, n_components=2):
    feat_cols = _get_feature_columns(df)
    # Aggregate to patient-level
    patient_df = df.groupby(['patient_id', 'class_name']).mean(numeric_only=True).reset_index()
    X = patient_df[feat_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = StandardScaler().fit_transform(X)

    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    try:
        components = ica.fit_transform(X)
    except Exception as e:
        print(f'  Warning: ICA failed ({e}), skipping.')
        return

    palette = _class_palette(classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in classes:
        mask = patient_df['class_name'].values == cls
        ax.scatter(components[mask, 0], components[mask, 1],
                   label=cls, alpha=0.8, s=60, color=palette[cls], edgecolors='k', linewidths=0.5)
    ax.set_xlabel('IC1', fontsize=12)
    ax.set_ylabel('IC2', fontsize=12)
    ax.set_title('ICA Projection — Patient-Level (IC1 vs IC2)', fontsize=13)
    ax.legend(title='Class', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ica_projection.png'), dpi=150)
    plt.close(fig)
    print('  [OK] Saved ica_projection.png')


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
    print('  [OK] Saved feature_distributions.png')


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


def _rolling_rms(signal, window=50):
    """
    Compute a rolling RMS envelope.
    RMS_i = sqrt( mean( x[i-w/2 : i+w/2]^2 ) )
    Uses a fast cumsum approach. Output length matches input.
    """
    x2 = np.asarray(signal, dtype=np.float64) ** 2
    # Pad to keep output length == input length
    pad = window // 2
    x2_padded = np.pad(x2, (pad, pad), mode='edge')
    cumsum = np.cumsum(x2_padded)
    rms = np.sqrt((cumsum[window:] - cumsum[:-window]) / window)
    return rms[:len(signal)]


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
                          clip_len=1000, rms_window=50, feature_df=None):
    """
    For each class, pick one representative patient, auto-select an
    interesting ~clip_len segment, and plot:

      Panel 1  —  Agonist + Antagonist RMS envelopes overlaid (interaction view)
      Panel 2  —  Torque RMS envelope  (net activation proxy)
      Panel 3  —  Stiffness RMS envelope  (co-contraction proxy, filled)

    Raw signals are shown as faint traces underneath for context.
    Also produces a comparison grid with one column per class.
    """
    all_classes = sorted(classes_with_paths.keys())

    # ── Per-class individual plots ──────────────────────────────────────
    for cls_name in all_classes:
        file_list = classes_with_paths[cls_name]
        if not file_list:
            continue
        # Pick median-representative patient instead of always first alphabetically
        csv_path, patient_id = file_list[0]  # fallback
        if feature_df is not None:
            from scipy.spatial.distance import cdist
            _fc = _get_feature_columns(feature_df)
            _cls_sub = feature_df[feature_df['class_name'] == cls_name]
            if not _cls_sub.empty and _fc:
                _centroid = _cls_sub[_fc].mean(numeric_only=True).values.reshape(1, -1)
                _centroid = np.nan_to_num(_centroid, nan=0.0)
                _pmeans = _cls_sub.groupby('patient_id')[_fc].mean()
                _dists = cdist(_centroid, np.nan_to_num(_pmeans.values, nan=0.0))[0]
                _best_pid = _pmeans.index[np.argmin(_dists)]
                for _cp, _pi in file_list:
                    if _pi == _best_pid:
                        csv_path, patient_id = _cp, _pi
                        break

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

        # RMS envelopes
        rms_ant = _rolling_rms(e_ant, rms_window)
        rms_ago = _rolling_rms(e_ago, rms_window)
        rms_torque = _rolling_rms(torque, rms_window)
        rms_stiffness = _rolling_rms(stiffness, rms_window)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1, 1]})
        fig.suptitle(f'{cls_name}  -  RMS envelope (window={rms_window})  -  '
                     f'samples {start}-{end}  (patient: {patient_id})',
                     fontsize=12, fontweight='bold')

        # Panel 1: Agonist + Antagonist RMS envelopes overlaid
        ax = axes[0]
        ax.plot(t, e_ago, color=_EMG_COLORS['agonist'], linewidth=0.3,
                alpha=0.15)  # faint raw
        ax.plot(t, e_ant, color=_EMG_COLORS['antagonist'], linewidth=0.3,
                alpha=0.15)  # faint raw
        ax.plot(t, rms_ago, color=_EMG_COLORS['agonist'], linewidth=1.8,
                alpha=0.9, label='Agonist RMS')
        ax.plot(t, rms_ant, color=_EMG_COLORS['antagonist'], linewidth=1.8,
                alpha=0.9, label='Antagonist RMS')
        ax.fill_between(t, rms_ago, rms_ant, alpha=0.10, color='grey')
        ax.set_ylabel('EMG amplitude (RMS)', fontsize=10)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.set_title('Agonist - Antagonist Interaction (RMS envelope)', fontsize=11, pad=4)
        ax.grid(True, alpha=0.2)
        # Interpretation box
        ax.text(0.01, 0.97,
                'Healthy: alternating peaks (reciprocal inhibition)\n'
                'Pathological: overlapping envelopes (co-contraction)',
                transform=ax.transAxes, fontsize=7.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFFDE7', ec='#ccc', alpha=0.9))

        # Panel 2: Torque RMS
        ax = axes[1]
        ax.plot(t, torque, color=_EMG_COLORS['torque'], linewidth=0.3, alpha=0.15)
        ax.plot(t, rms_torque, color=_EMG_COLORS['torque'], linewidth=1.8, label='Torque RMS')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.fill_between(t, 0, rms_torque,
                        color=_EMG_COLORS['torque'], alpha=0.12)
        ax.set_ylabel('Torque (RMS)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.text(0.01, 0.93,
                'Clear oscillation = normal motor control\n'
                'Flat / erratic = impaired control',
                transform=ax.transAxes, fontsize=7.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFFDE7', ec='#ccc', alpha=0.9))

        # Panel 3: Stiffness RMS
        ax = axes[2]
        ax.plot(t, stiffness, color=_EMG_COLORS['stiffness'], linewidth=0.3, alpha=0.15)
        ax.fill_between(t, 0, rms_stiffness, color=_EMG_COLORS['stiffness'], alpha=0.35)
        ax.plot(t, rms_stiffness, color=_EMG_COLORS['stiffness'], linewidth=1.8)
        ax.set_ylabel('Stiffness (RMS)', fontsize=10)
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
        print(f'  [OK] Saved {fname}')

    # ── Multi-class comparison grid ─────────────────────────────────────
    n_cls = len(all_classes)
    fig, axes = plt.subplots(3, n_cls, figsize=(5 * n_cls, 9), sharex='col',
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    if n_cls == 1:
        axes = axes.reshape(3, 1)
    fig.suptitle('EMG Waveform Comparison (RMS Envelope)', fontsize=14, fontweight='bold')

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

        rms_ant = _rolling_rms(e_ant, rms_window)
        rms_ago = _rolling_rms(e_ago, rms_window)
        rms_torque = _rolling_rms(torque, rms_window)
        rms_stiffness = _rolling_rms(stiffness, rms_window)

        # Row 0 — EMG interaction (RMS)
        ax = axes[0, col]
        ax.plot(t, e_ago, color=_EMG_COLORS['agonist'], lw=0.2, alpha=0.12)
        ax.plot(t, e_ant, color=_EMG_COLORS['antagonist'], lw=0.2, alpha=0.12)
        ax.plot(t, rms_ago, color=_EMG_COLORS['agonist'], lw=1.4, alpha=0.9, label='Ago RMS')
        ax.plot(t, rms_ant, color=_EMG_COLORS['antagonist'], lw=1.4, alpha=0.9, label='Ant RMS')
        ax.fill_between(t, rms_ago, rms_ant, alpha=0.08, color='grey')
        ax.set_title(cls_name, fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('EMG (RMS)', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

        # Row 1 — Torque RMS
        ax = axes[1, col]
        ax.plot(t, torque, color=_EMG_COLORS['torque'], lw=0.2, alpha=0.12)
        ax.plot(t, rms_torque, color=_EMG_COLORS['torque'], lw=1.4)
        ax.axhline(0, color='grey', lw=0.4, ls='--')
        ax.fill_between(t, 0, rms_torque, color=_EMG_COLORS['torque'], alpha=0.1)
        if col == 0:
            ax.set_ylabel('Torque (RMS)', fontsize=10)
        ax.grid(True, alpha=0.2)

        # Row 2 — Stiffness RMS
        ax = axes[2, col]
        ax.plot(t, stiffness, color=_EMG_COLORS['stiffness'], lw=0.2, alpha=0.12)
        ax.fill_between(t, 0, rms_stiffness, color=_EMG_COLORS['stiffness'], alpha=0.3)
        ax.plot(t, rms_stiffness, color=_EMG_COLORS['stiffness'], lw=1.4)
        if col == 0:
            ax.set_ylabel('Stiffness (RMS)', fontsize=10)
        ax.set_xlabel('Sample', fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, 'waveform_comparison.png'), dpi=150)
    plt.close(fig)
    print('  [OK] Saved waveform_comparison.png')


# ---------------------------------------------------------------------------
# Convenience: run everything
# ---------------------------------------------------------------------------

def plot_gait_phase_overlay(data_dir, classes_with_paths, output_dir, fs=1000.0, clip_len=2000):
    """
    For each class, plot:
      Panel 1 — TA + GA RMS envelopes (raw EMG activity)
      Panel 2 — Continuous gait_phase [0, 100] colored by phase region

    Phase color bands:
      Heel Strike / Loading (0–12%)   → Orange  (#FF6D00)
      Midstance / Terminal (12–62%)   → Green   (#00C853)
      Pre-Swing / Toe-Off  (62–75%)   → Purple  (#AA00FF)
      Swing                (75–100%)  → Teal    (#00B0FF)

    Also saves a multi-class comparison grid: gait_phase_comparison.png
    """
    try:
        import sys, os as _os
        sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
        from gait_phase import assign_gait_phase_continuous, emg_envelope, FSM_STATES
    except ImportError:
        print('  ⚠ gait_phase module not found — skipping gait phase overlay plots.')
        return

    # Cyclic colormap: interpolate phase 0–100 into RGBA
    _PHASE_COLORS = [
        (0,   '#FF6D00'),   # Heel Strike         0–12%
        (12,  '#00C853'),   # Midstance           12–62%
        (62,  '#AA00FF'),   # Pre-Swing           62–75%
        (75,  '#00B0FF'),   # Swing               75–100%
        (100, '#FF6D00'),   # Wrap around
    ]

    def _phase_color(pct):
        """Map a gait phase percentage to an RGBA colour."""
        import matplotlib.colors as mcolors
        for i in range(len(_PHASE_COLORS) - 1):
            lo_pct, lo_col = _PHASE_COLORS[i]
            hi_pct, hi_col = _PHASE_COLORS[i + 1]
            if lo_pct <= pct <= hi_pct:
                t = (pct - lo_pct) / max(hi_pct - lo_pct, 1e-6)
                lo_rgba = np.array(mcolors.to_rgba(lo_col))
                hi_rgba = np.array(mcolors.to_rgba(hi_col))
                return tuple(lo_rgba * (1 - t) + hi_rgba * t)
        return mcolors.to_rgba(_PHASE_COLORS[-1][1])

    all_classes = sorted(classes_with_paths.keys())

    # ── Individual per-class plots ─────────────────────────────────────────
    for cls_name in all_classes:
        file_list = classes_with_paths.get(cls_name, [])
        if not file_list:
            continue

        csv_path, patient_id = file_list[0]
        df_raw = pd.read_csv(csv_path, header=None)
        e_ta_full = np.nan_to_num(df_raw.iloc[:, 0].values.astype(float))
        e_ga_full = np.nan_to_num(df_raw.iloc[:, 1].values.astype(float))

        # Pick most dynamic clip
        start = _find_interesting_clip(e_ta_full + e_ga_full, clip_len)
        e_ta = e_ta_full[start:start + clip_len]
        e_ga = e_ga_full[start:start + clip_len]
        t = np.arange(len(e_ta))

        # Compute envelopes for visualization
        ta_env = _rolling_rms(np.abs(e_ta), 50)
        ga_env = _rolling_rms(np.abs(e_ga), 50)

        # Compute gait phase
        gait_phase, _, _ = assign_gait_phase_continuous(e_ta, e_ga, fs)

        # Detect cycle boundaries for vertical dashed lines (approx from phase resets)
        phase_diff = np.diff(gait_phase)
        cycle_boundaries = np.where(phase_diff < -50)[0]  # Large drops = new cycle

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                  gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle(
            f'{cls_name} — Gait Phase Overlay  (patient: {patient_id})',
            fontsize=13, fontweight='bold'
        )

        # --- Panel 1: EMG Envelopes ---
        ax = axes[0]
        ax.plot(t, e_ta, color='#2196F3', lw=0.3, alpha=0.15)
        ax.plot(t, e_ga, color='#F44336', lw=0.3, alpha=0.15)
        ax.plot(t, ta_env, color='#2196F3', lw=1.8, label='TA (Tibialis Anterior) RMS')
        ax.plot(t, ga_env, color='#F44336', lw=1.8, label='GA (Gastrocnemius) RMS')
        ax.fill_between(t, ta_env, ga_env, alpha=0.08, color='grey', label='Co-activation zone')

        for cb in cycle_boundaries:
            ax.axvline(cb, color='grey', lw=0.8, linestyle='--', alpha=0.6)

        ax.set_ylabel('EMG Amplitude (RMS)', fontsize=10)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.text(0.01, 0.96,
                'Each dashed line = detected gait cycle boundary',
                transform=ax.transAxes, fontsize=7.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFFDE7', ec='#ccc', alpha=0.9))

        # --- Panel 2: Gait Phase colored line ---
        ax = axes[1]
        # Plot phase as colored segments
        for i in range(len(t) - 1):
            color = _phase_color(float(gait_phase[i]))
            ax.plot(t[i:i+2], gait_phase[i:i+2], color=color, lw=2.2, solid_capstyle='round')

        for cb in cycle_boundaries:
            ax.axvline(cb, color='grey', lw=0.8, linestyle='--', alpha=0.6)

        ax.set_yticks([0, 12, 62, 75, 100])
        ax.set_yticklabels(['Heel\nStrike', 'Stance\nStart', 'Toe-\nOff', 'Swing\nStart', 'Cycle\nEnd'],
                            fontsize=7)
        ax.set_ylim(-5, 108)
        ax.set_ylabel('Gait Phase (%)', fontsize=10)
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.grid(True, alpha=0.2)

        # Phase legend patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6D00', label='Heel Strike / Loading (0–12%)'),
            Patch(facecolor='#00C853', label='Stance (12–62%)'),
            Patch(facecolor='#AA00FF', label='Pre-Swing / Toe-Off (62–75%)'),
            Patch(facecolor='#00B0FF', label='Swing (75–100%)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7.5,
                  framealpha=0.95, ncol=2)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        safe = cls_name.replace(' ', '_').lower()
        fname = f'gait_phase_{safe}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f'  [OK] Saved {fname}')

    # ── Multi-class comparison grid ────────────────────────────────────────
    n_cls = len(all_classes)
    if n_cls == 0:
        return

    fig, axes = plt.subplots(2, n_cls, figsize=(5 * n_cls, 7), sharex='col',
                              gridspec_kw={'height_ratios': [2, 1]})
    if n_cls == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle('Gait Phase Comparison Across Classes', fontsize=14, fontweight='bold')

    for col, cls_name in enumerate(all_classes):
        file_list = classes_with_paths.get(cls_name, [])
        if not file_list:
            continue
        csv_path, pid = file_list[0]
        df_raw = pd.read_csv(csv_path, header=None)
        e_ta = np.nan_to_num(df_raw.iloc[:, 0].values.astype(float))
        e_ga = np.nan_to_num(df_raw.iloc[:, 1].values.astype(float))
        start = _find_interesting_clip(e_ta + e_ga, clip_len)
        e_ta = e_ta[start:start + clip_len]
        e_ga = e_ga[start:start + clip_len]
        t = np.arange(len(e_ta))

        ta_env = _rolling_rms(np.abs(e_ta), 50)
        ga_env = _rolling_rms(np.abs(e_ga), 50)
        gait_phase, _, _ = assign_gait_phase_continuous(e_ta, e_ga, fs)
        cycle_boundaries = np.where(np.diff(gait_phase) < -50)[0]

        # Row 0 — EMG envelopes
        ax = axes[0, col]
        ax.plot(t, ta_env, color='#2196F3', lw=1.4, label='TA')
        ax.plot(t, ga_env, color='#F44336', lw=1.4, label='GA')
        ax.fill_between(t, ta_env, ga_env, alpha=0.08, color='grey')
        for cb in cycle_boundaries:
            ax.axvline(cb, color='grey', lw=0.6, linestyle='--', alpha=0.5)
        ax.set_title(cls_name, fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('EMG (RMS)', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

        # Row 1 — Gait phase
        ax = axes[1, col]
        for i in range(len(t) - 1):
            color = _phase_color(float(gait_phase[i]))
            ax.plot(t[i:i+2], gait_phase[i:i+2], color=color, lw=2.0, solid_capstyle='round')
        for cb in cycle_boundaries:
            ax.axvline(cb, color='grey', lw=0.6, linestyle='--', alpha=0.5)
        ax.set_yticks([0, 62, 100])
        ax.set_yticklabels(['0%', '62%', '100%'], fontsize=7)
        ax.set_ylim(-5, 108)
        if col == 0:
            ax.set_ylabel('Gait Phase (%)', fontsize=9)
        ax.set_xlabel('Sample', fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, 'gait_phase_comparison.png'), dpi=150)
    plt.close(fig)
    print('  [OK] Saved gait_phase_comparison.png')


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
    plot_ica(df, classes, output_dir)
    plot_feature_distributions(df, classes, output_dir)

    # Signal waveforms + gait phase overlay (needs raw CSV paths)
    if data_dir:
        from EDA.eda_features import discover_csv_files
        records = discover_csv_files(data_dir)
        classes_with_paths = {}
        for csv_path, cls_name, patient_id in records:
            classes_with_paths.setdefault(cls_name, []).append((csv_path, patient_id))
        plot_signal_waveforms(data_dir, classes_with_paths, alpha, beta, output_dir, feature_df=df)
        plot_gait_phase_overlay(data_dir, classes_with_paths, output_dir)
    else:
        print('  (Skipping waveform/gait-phase plots — no data_dir provided)')

    print('All plots saved to', output_dir)