***Note: Run `train.py` with the `--enhanced_features` flag to use all 22 top features***
---
# Biomechanics-Informed Gait Pathology Classification

> **A PyTorch pipeline using hardware-accelerated Mamba SSMs and physics-informed EMG features to classify clinical gait pathologies from extreme-length time-series.**

Classifies gait pathologies (**Healthy, Hemiplegia, PIVD-RA, PIVD-Piriformis, Osteoarthritis**) from raw dual-channel EMG recordings ($E\_{ant}$, $E\_{ago}$). The system automatically computes physics-derived features (Torque, Joint Stiffness), physiologically anchored gait phase, and trains end-to-end classifiers over sequences of $26{,}000+$ timesteps using State-Space Models (Mamba) and GRU+Attention baselines.

---

## Project Layout

```text
Gait Analysis/
â”œâ”€â”€ dataset.py                  # DataLoader, physics features, sliding window, normalization
â”œâ”€â”€ train.py                    # Training loop (Focal Loss, AMP, gradient accumulation)
â”œâ”€â”€ evaluate.py                 # Evaluation with inference benchmarking
â”œâ”€â”€ feature_config.py           # Feature selection config (legacy 5 / enhanced 22 features)
â”œâ”€â”€ gait_phase.py               # EMG-anchored gait cycle phase extraction
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ native_mamba.py         # Pure-PyTorch Mamba SSM (default, fully portable)
â”‚   â”œâ”€â”€ triton_mamba.py         # Fused Triton/CUDA Mamba with chunked streaming
â”‚   â”œâ”€â”€ official_mamba.py       # Wrapper for pip mamba-ssm (Gu & Dao)
â”‚   â””â”€â”€ gru_baseline.py        # GRU + FlashAttention baseline
â”œâ”€â”€ EDA/
â”‚   â”œâ”€â”€ eda_features.py         # Feature extraction (time + frequency domain)
â”‚   â”œâ”€â”€ eda_visualization.py    # Scatter, heatmap, PCA, LDA, gait phase plots
â”‚   â”œâ”€â”€ eda_runner.py           # CLI entry point for full EDA pipeline
â”‚   â””â”€â”€ outputs/                # Generated plots and CSV
â”œâ”€â”€ submit_gait.sh              # PBS job script for HPC (H100 GPU)
â”œâ”€â”€ connect_hpc.bat             # Windows SSH helper for HPC access
â”œâ”€â”€ Dockerfile                  # Container build
â””â”€â”€ Datasets/
    â”œâ”€â”€ 1_Healthy/              # Patient CSVs (2-column: E_ago, E_ant)
    â”œâ”€â”€ 3_Hemiplegia/
    â”œâ”€â”€ 5_PIVD_RA/
    â”œâ”€â”€ 6_PIVD_Priformis/
    â””â”€â”€ 8_Osteoarthiritis/
```

Each CSV contains two headerless columns: **column 0** = Agonist EMG ($E\_{ago}$), **column 1** = Antagonist EMG ($E\_{ant}$).

---

## Data Pipeline

### Physics-Informed Feature Engineering

From the two raw EMG channels, five input features are computed per timestep (legacy mode):

| Feature | Formula | Clinical Meaning |
|---------|---------|-----------------|
| $E\_{ant}$ | raw column 1 | Antagonist muscle activation |
| $E\_{ago}$ | raw column 0 | Agonist muscle activation |
| Torque | $\alpha \cdot E\_{ant} - \beta \cdot E\_{ago}$ | Net joint torque proxy |
| Stiffness | $E\_{ant} + E\_{ago}$ | Co-contraction / joint stiffness index |
| Gait Phase | EMG-anchored phase âˆˆ [0, 100] | Cycle position with physiological landmarks |

### Gait Phase Extraction (v3.0)

The gait phase signal is computed via a **two-stage EMG-anchored algorithm** in `gait_phase.py`:

**Stage 1 â€” EMG-driven phase velocity:**
$$v(t) = 0.15 + 0.85 \cdot \text{activity\_norm}(t)$$

where `activity_norm` is the cycle-peak-normalized sum of TA and GA raw RMS envelopes. The cumulative integral $\int v(t)dt$, normalized to [0, 100], produces a nonlinear phase that advances rapidly during gait events and slowly during quiet periods.

**Stage 2 â€” Physiological landmark anchoring:**

Four explicit gait events are detected from raw RMS envelopes and forced to their canonical phase positions via monotonic PCHIP (C1-continuous) re-normalization:

| Event | Detection Method | Anchored Phase |
|-------|-----------------|----------------|
| Heel strike | Cycle start (TA burst onset) | 0% |
| Loading response end | TA deactivation below 50% | 12% |
| Push-off (GA peak) | Max gastrocnemius RMS | 55% |
| Toe-off (GAâ†’TA crossover) | GA/(GA+TA) < 0.5 | 62% |

### Enhanced Feature Mode (22 features)

With `--enhanced_features`, the pipeline extracts 22 features per timestep selected by Random Forest Permutation Importance:

| Group | Count | Features |
|-------|-------|----------|
| Base | 5 | e_ant, e_ago, torque, stiffness, gait_phase |
| Time-domain | 10 | SSC, WL, ZCR, IEMG, Variance, MAV, RMS (per channel) |
| Frequency-domain | 6 | Spectral entropy, mean/median/peak freq, total power |
| Gait cycle | 1 | Propulsion phase duration |

Feature selection is configured in `feature_config.py`.

### Sliding Window Segmentation

Raw sequences ($26{,}000+$ timesteps) are sliced into $2{,}000$-sample windows:
- **Dynamic stride balancing**: Minority classes get a smaller stride, natively equalizing window counts without loss reweighting.
- **Temporal jittering**: $\pm 5\%$ boundary randomization during training for phase invariance.
- **Random scaling**: $0.9\text{â€“}1.1\times$ amplitude augmentation.

### Normalization & Splitting

- **Patient-level splitting**: Strict isolation of entire patient files into Train/Val/Test *before* windowing â€” zero data leakage.
- **Global normalization**: Mean/std computed on training windows only, frozen for val/test.
- **Hard clamping**: Outliers clamped to $[-10, 10]$ to prevent gradient explosions.

---

## Model Architectures

All models accept input shape `(batch, seq_len, input_dim)` and output class logits.

### 1. Native PyTorch Mamba *(default)*

`models/native_mamba.py` â€” Fully portable, no CUDA dependencies.

- Selective state-space recurrence with learnable $A$, $B$, $C$, $D$ matrices
- 1D depthwise convolution for local context
- SiLU-gated residual connections
- **Stability features**: float32 recurrence (prevents bfloat16 drift), periodic state clamping (every 100 steps), transition matrix clamping ($A \leq -10^{-4}$), dt softplus clamping $[-20, 5]$

### 2. Triton/CUDA Mamba

`models/triton_mamba.py` â€” Hardware-accelerated with chunked streaming.

- Fused Triton kernel for parallel associative scan (requires `triton`)
- Automatic fallback to chunked PyTorch scan if Triton unavailable
- Memory-efficient streaming across chunks (configurable `chunk_size`, default 2048)

### 3. Official Mamba SSM

`models/official_mamba.py` â€” Wrapper around the [mamba-ssm](https://github.com/state-spaces/mamba) package.

- Requires `pip install causal-conv1d>=1.2.0 mamba-ssm`
- Uses the exact CUDA kernels from the Gu & Dao paper

### 4. GRU + Attention Baseline

`models/gru_baseline.py` â€” Classical RNN+Attention for benchmarking.

- Multi-layer GRU â†’ Masked Multi-Head Attention
- Uses PyTorch 2.0 `F.scaled_dot_product_attention` (FlashAttention/memory-efficient backends)
- Causal masking with $O(N)$ memory
- Optional KV caching for streaming inference

---

## Requirements

```bash
pip install torch pandas numpy scikit-learn tqdm matplotlib seaborn scipy
```

For hardware-accelerated Mamba:

```bash
# Triton Mamba (Linux/WSL only)
pip install triton

# Official Mamba-SSM (requires CUDA compilation tools)
pip install causal-conv1d>=1.2.0 mamba-ssm
```

---

## Training

AMP (bfloat16 on H100, float16 elsewhere) and gradient accumulation are enabled by default. Uses **Focal Loss** with inverse-frequency class weights for PIVD separation.

```bash
# Native PyTorch Mamba â€” legacy features (5)
python train.py --data_dir "Datasets/" --epochs 50 --batch_size 2

# Native PyTorch Mamba â€” enhanced features (22)
python train.py --data_dir "Datasets/" --epochs 50 --batch_size 2 --enhanced_features

# GRU + Attention Baseline
python train.py --data_dir "Datasets/" --epochs 50 --batch_size 2 --use_gru_baseline

# Triton-Accelerated Mamba
python train.py --data_dir "Datasets/" --epochs 50 --batch_size 2 --use_triton_mamba

# Official Mamba-SSM
python train.py --data_dir "Datasets/" --epochs 50 --batch_size 2 --use_official_mamba
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 2 | Batch size (keep small for long sequences) |
| `--accum_steps` | 8 | Gradient accumulation steps |
| `--lr` | 1e-3 | Peak learning rate (OneCycleLR) |
| `--d_model` | 64 | Hidden dimension (capped at 32 for small datasets) |
| `--n_layers` | 2 | Number of SSM/GRU layers |
| `--focal_gamma` | 2.0 | Focal Loss gamma (0 = standard CE, 2 = hard-example focus) |
| `--enhanced_features` | off | Use 22-feature mode instead of legacy 5 |
| `--output_name` | `best_model.pth` | Saved weights filename |

The best model (by validation F1) is saved automatically.

---

## Evaluation

`evaluate.py` generates a full classification report, confusion matrix plot, and inference benchmarking metrics.

```bash
# Evaluate Native Mamba
python evaluate.py --model_path best_model.pth --data_dir "Datasets/"

# Evaluate GRU Baseline
python evaluate.py --model_path best_gru_baseline.pth --data_dir "Datasets/" --use_gru_baseline

# Detailed per-batch timing breakdown
python evaluate.py --model_path best_model.pth --data_dir "Datasets/" --benchmark
```

**Metrics reported:**

| Category | Metrics |
|----------|---------|
| Classification | Accuracy, F1 (weighted), ROC-AUC (OVR), per-class precision/recall |
| Inference | Total/trainable parameters, model size (MB), latency/sample (ms), throughput (samples/s), peak GPU memory |

> **Important**: Architecture flags (`--d_model`, `--n_layers`, `--use_*`) must match the trained model exactly.

---

## Exploratory Data Analysis (EDA)

A standalone analysis module that operates on the same raw CSV files without touching the training pipeline.

### Features Extracted

**Base features** (same as training): `e_ant`, `e_ago`, `torque`, `stiffness`

**Time-domain** (per channel, per window):
RMS, MAV, Variance, Waveform Length, Zero Crossing Rate, Slope Sign Changes, IEMG

**Frequency-domain** (per channel, per window via Welch PSD):
Mean Frequency, Median Frequency, Spectral Entropy, Peak Frequency, Total Spectral Power

### Running

```bash
python EDA/eda_runner.py --data_dir "path/to/Datasets"
```

Optional: `--window_size 2000`, `--stride 1000`, `--alpha 1.0`, `--beta 1.0`, `--fs 1000`

### Generated Outputs

All saved to `EDA/outputs/`:

| Output | Description |
|--------|-------------|
| `eda_feature_table.csv` | Unified feature dataframe (all windows, all patients) |
| `scatter_torque_vs_stiffness.png` | Scatter with convex hulls per class |
| `scatter_rms_ant_vs_rms_ago.png` | RMS channel comparison with hulls |
| `feature_correlation_heatmap.png` | Hierarchical clustermap of feature correlations |
| `pca_projection.png` | PC1 vs PC2 coloured by pathology |
| `lda_projection.png` | LD1 vs LD2 coloured by pathology |
| `feature_distributions.png` | Per-class histograms + boxplots for key features |
| `waveform_<class>.png` | Agonistâ€“antagonist interaction, torque, stiffness |
| `waveform_comparison.png` | Side-by-side waveform comparison across all classes |
| `gait_phase_<class>.png` | EMG overlay with anchored gait phase per class |
| `gait_phase_comparison.png` | Side-by-side gait phase comparison across all classes |

---

## HPC Deployment

### PBS Job Submission

```bash
qsub submit_gait.sh
```

The PBS script (`submit_gait.sh`) requests 1Ã— H100 GPU, 10 CPUs, 32 GB RAM and runs both Mamba and GRU training sequentially with enhanced features.

### Quick SSH Connect (Windows)

```cmd
connect_hpc.bat [username] [host]
connect_hpc.bat                      :: defaults to aantriksh.124259@10.16.1.50
```

Automatically loads CUDA, activates `gait_env`, and navigates to the project directory.

---

## Changelog

### v3.0 â€” EMG-Anchored Gait Phase & Focal Loss *(2026-04-30)*

**Added**
- `gait_phase.py`: Two-stage EMG-anchored gait phase algorithm
  - Stage 1: Phase velocity proportional to instantaneous EMG activity (TA + GA RMS)
  - Stage 2: PCHIP re-normalization to anchor physiological landmarks (heel strike â†’ 0%, loading end â†’ 12%, GA peak â†’ 55%, toe-off â†’ 62%)
  - Cycle boundary detection via TKEO + multi-method fusion (threshold, derivative, autocorrelation)
- `feature_config.py`: Feature selection configuration (legacy 5 / enhanced 22 features)
- Focal Loss (`train.py`): Î³=2.0 with inverse-frequency class weights for PIVD separation
- Per-class classification report printed at end of training
- Gait phase overlay visualizations in EDA (`gait_phase_*.png`)

**Changed**
- Label smoothing reduced from 0.1 â†’ 0.02 (less aggressive for minority classes)
- `evaluate()` now returns labels/preds for downstream reporting
- EDA visualization: NaN-safe correlation heatmap, co-activation zone overlays

### v2.1 â€” EDA Module & Inference Benchmarking *(2026-03-14)*

**Added**
- `EDA/` module â€” full exploratory data analysis pipeline
  - 7 time-domain + 5 frequency-domain EMG features per channel
  - Scatter plots with convex hulls, hierarchical correlation clustermap
  - PCA and LDA projections, per-class feature distributions
  - Signal waveform plots: agonistâ€“antagonist interaction with clinical interpretations, multi-class comparison grid, smart auto-clipping
- Inference benchmarking in `evaluate.py` â€” latency, throughput, parameter count, GPU memory (`--benchmark` flag)

### v2.0 â€” Numerical Stability & Mamba Fixes *(2026-03-13)*

**Fixed**
- **NaN loss elimination**: forced float32 for Mamba recurrence (prevents bfloat16 drift over 2000 timesteps)
- **GradScaler disabled for bfloat16**: H100 bfloat16 + GradScaler silently corrupted SSM state
- Periodic state clamping (every 100 steps) to prevent runaway accumulation
- Clamped transition matrix $A$ and dt projections for numerical stability

**Changed**
- `triton_mamba.py`: uses clamped dt and A matrices throughout chunked scan
- `native_mamba.py`: identical stability fixes
- All Mamba blocks cast to float32 internally, return to original dtype

### v1.0 â€” Initial Pipeline

**Core**
- Physics-informed feature engineering (Torque, Stiffness)
- Patient-level strict splitting with dynamic stride balancing
- Sliding window segmentation (2000-sample windows)
- Global normalization with hard outlier clamping
- Temporal jittering + random scaling augmentation

**Models**
- Native PyTorch Mamba SSM
- Triton/CUDA fused Mamba with chunked streaming
- Official mamba-ssm wrapper
- GRU + FlashAttention baseline

**Training**
- AMP (bfloat16/float16 auto-detect), gradient accumulation
- OneCycleLR scheduler, label smoothing (0.1)
- NaN batch skipping, gradient clipping

**Evaluation**
- Full classification report, confusion matrix visualization
- Multi-class ROC-AUC (OVR) with safe handling of missing classes

