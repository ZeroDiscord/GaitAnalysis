# Biomechanics-Informed Gait Pathology Classification

> **A PyTorch pipeline using hardware-accelerated Mamba SSMs and physics-informed EMG features to classify clinical gait pathologies from extreme-length time-series.**

Classifies gait pathologies (**Healthy, Hemiplegia, PIVD-RA, PIVD-Piriformis, Osteoarthritis**) from raw dual-channel EMG recordings ($E\_{ant}$, $E\_{ago}$). The system automatically computes physics-derived features (Torque, Joint Stiffness) and trains end-to-end classifiers over sequences of $26{,}000+$ timesteps using State-Space Models (Mamba) and GRU+Attention baselines.

---

## Project Layout

```text
Gait Analysis/
├── dataset.py                  # DataLoader, physics features, sliding window, normalization
├── train.py                    # Training loop (AMP, gradient accumulation, OneCycleLR)
├── evaluate.py                 # Evaluation with inference benchmarking
├── models/
│   ├── native_mamba.py         # Pure-PyTorch Mamba SSM (default, fully portable)
│   ├── triton_mamba.py         # Fused Triton/CUDA Mamba with chunked streaming
│   ├── official_mamba.py       # Wrapper for pip mamba-ssm (Gu & Dao)
│   └── gru_baseline.py        # GRU + FlashAttention baseline
├── EDA/
│   ├── eda_features.py         # Feature extraction (time + frequency domain)
│   ├── eda_visualization.py    # Scatter, heatmap, PCA, LDA, waveform plots
│   ├── eda_runner.py           # CLI entry point for full EDA pipeline
│   └── outputs/                # Generated plots and CSV
├── submit_gait.sh              # PBS job script for HPC (H100 GPU)
├── connect_hpc.bat             # Windows SSH helper for HPC access
├── Dockerfile                  # Container build
└── Datasets/
    ├── 1_Healthy/              # Patient CSVs (2-column: E_ago, E_ant)
    ├── 3_Hemiplegia/
    ├── 5_PIVD_RA/
    ├── 6_PIVD_Priformis/
    └── 8_Osteoarthiritis/
```

Each CSV contains two headerless columns: **column 0** = Agonist EMG ($E\_{ago}$), **column 1** = Antagonist EMG ($E\_{ant}$).

---

## Data Pipeline

### Physics-Informed Feature Engineering

From the two raw EMG channels, four input features are computed per timestep:

| Feature | Formula | Clinical Meaning |
|---------|---------|-----------------|
| $E\_{ant}$ | raw column 1 | Antagonist muscle activation |
| $E\_{ago}$ | raw column 0 | Agonist muscle activation |
| Torque | $\alpha \cdot E\_{ant} - \beta \cdot E\_{ago}$ | Net joint torque proxy |
| Stiffness | $E\_{ant} + E\_{ago}$ | Co-contraction / joint stiffness index |

### Sliding Window Segmentation

Raw sequences ($26{,}000+$ timesteps) are sliced into $2{,}000$-sample windows:
- **Dynamic stride balancing**: Minority classes get a smaller stride, natively equalizing window counts without loss reweighting.
- **Temporal jittering**: $\pm 5\%$ boundary randomization during training for phase invariance.
- **Random scaling**: $0.9\text{–}1.1\times$ amplitude augmentation.

### Normalization & Splitting

- **Patient-level splitting**: Strict isolation of entire patient files into Train/Val/Test *before* windowing — zero data leakage.
- **Global normalization**: Mean/std computed on training windows only, frozen for val/test.
- **Hard clamping**: Outliers clamped to $[-10, 10]$ to prevent gradient explosions.

---

## Model Architectures

All models accept input shape `(batch, seq_len, 4)` and output class logits.

### 1. Native PyTorch Mamba *(default)*

`models/native_mamba.py` — Fully portable, no CUDA dependencies.

- Selective state-space recurrence with learnable $A$, $B$, $C$, $D$ matrices
- 1D depthwise convolution for local context
- SiLU-gated residual connections
- **Stability features**: float32 recurrence (prevents bfloat16 drift), periodic state clamping (every 100 steps), transition matrix clamping ($A \leq -10^{-4}$), dt softplus clamping $[-20, 5]$

### 2. Triton/CUDA Mamba

`models/triton_mamba.py` — Hardware-accelerated with chunked streaming.

- Fused Triton kernel for parallel associative scan (requires `triton`)
- Automatic fallback to chunked PyTorch scan if Triton unavailable
- Memory-efficient streaming across chunks (configurable `chunk_size`, default 2048)

### 3. Official Mamba SSM

`models/official_mamba.py` — Wrapper around the [mamba-ssm](https://github.com/state-spaces/mamba) package.

- Requires `pip install causal-conv1d>=1.2.0 mamba-ssm`
- Uses the exact CUDA kernels from the Gu & Dao paper

### 4. GRU + Attention Baseline

`models/gru_baseline.py` — Classical RNN+Attention for benchmarking.

- Multi-layer GRU → Masked Multi-Head Attention
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

AMP (bfloat16 on H100, float16 elsewhere) and gradient accumulation are enabled by default.

```bash
# Native PyTorch Mamba (default)
python train.py --data_dir "Datasets/" --epochs 50 --batch_size 2

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
| `waveform_<class>.png` | Agonist–antagonist interaction, torque, stiffness |
| `waveform_comparison.png` | Side-by-side waveform comparison across all classes |

---

## HPC Deployment

### PBS Job Submission

```bash
qsub submit_gait.sh
```

The PBS script (`submit_gait.sh`) requests 1× H100 GPU, 10 CPUs, 32 GB RAM and runs both Mamba and GRU training sequentially.

### Quick SSH Connect (Windows)

```cmd
connect_hpc.bat [username] [host]
connect_hpc.bat                      :: defaults to aantriksh.124259@10.16.1.50
```

Automatically loads CUDA, activates `gait_env`, and navigates to the project directory.

---

## Changelog

### v2.1 — EDA Module & Inference Benchmarking *(2026-03-14)*

**Added**
- `EDA/` module — full exploratory data analysis pipeline
  - 7 time-domain + 5 frequency-domain EMG features per channel
  - Scatter plots with convex hulls, hierarchical correlation clustermap
  - PCA and LDA projections, per-class feature distributions
  - Signal waveform plots: agonist–antagonist interaction with clinical interpretations, multi-class comparison grid, smart auto-clipping
- Inference benchmarking in `evaluate.py` — latency, throughput, parameter count, GPU memory (`--benchmark` flag)

### v2.0 — Numerical Stability & Mamba Fixes *(2026-03-13)*

**Fixed**
- **NaN loss elimination**: forced float32 for Mamba recurrence (prevents bfloat16 drift over 2000 timesteps)
- **GradScaler disabled for bfloat16**: H100 bfloat16 + GradScaler silently corrupted SSM state
- Periodic state clamping (every 100 steps) to prevent runaway accumulation
- Clamped transition matrix $A$ and dt projections for numerical stability

**Changed**
- `triton_mamba.py`: uses clamped dt and A matrices throughout chunked scan
- `native_mamba.py`: identical stability fixes
- All Mamba blocks cast to float32 internally, return to original dtype

### v1.0 — Initial Pipeline

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
