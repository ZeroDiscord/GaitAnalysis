# GaitAnalysis — Research-Grade Gait Pathology Classification

A publication-ready deep learning pipeline for classifying lower-limb gait pathologies from EMG muscle activation signals. Built specifically for small biomedical cohorts with high class imbalance, this repository applies state-of-the-art sequence modelling (Bidirectional Mamba, Bidirectional GRU) with rigorous Leave-One-Patient-Out (LOPO) cross-validation to produce clinically trustworthy results.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture Overview](#architecture-overview)
- [Feature Extraction](#feature-extraction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Format](#dataset-format)
- [Training](#training)
- [Evaluation](#evaluation)
- [HPC Deployment](#hpc-deployment)
- [Design Decisions](#design-decisions)
- [License](#license)

---

## Problem Statement

Diagnosing gait pathologies (e.g., Piriformis Syndrome, Piriformis PIVD) from EMG data presents several compounding challenges:

1. **Tiny cohorts** — Each class may have only 5-10 patient recordings.
2. **Severe imbalance** — Minority pathologies can have 3-5x fewer samples than normal gait.
3. **Data leakage risk** — Naive random splits allow windows from the same patient to appear in both train and test sets, inflating accuracy.
4. **Minority class collapse** — Standard cross-entropy converges to predicting only the majority class.

This pipeline addresses all four issues with principled solutions at every stage of the ML lifecycle.

---

## Architecture Overview

Two primary model architectures are supported, both designed for whole-sequence classification on fixed-length sliding windows of EMG data.

### BiMamba (Recommended)

> `models/bimamba_classifier.py`

A bidirectional Mamba architecture that processes the input sequence in both the forward and backward temporal directions and fuses the two representations with a learned gate.

- **Forward scan** captures causal temporal patterns (e.g., loading response followed by push-off).
- **Backward scan** captures anti-causal patterns (context from future time-steps).
- **Gated fusion** (`tanh`/`sigmoid`) weights each direction per feature rather than naively summing.
- **Padding-aware**: `Ā` (the recurrent state matrix) is zeroed at pad positions, preventing leakage from padding into valid signal.

```
Input (B, T, 5)
  -> Linear projection -> (B, T, d_model)
  -> [Forward Mamba + Backward Mamba] fused -> (B, T, d_model)
  x N layers
  -> Masked mean pooling -> (B, d_model)
  -> Classifier head -> (B, num_classes)
```

### GRU + Non-Causal Attention (Baseline)

> `models/gru_baseline.py`

A bidirectional GRU with a non-causal multi-head self-attention pooling head.

- **Bidirectional GRU**: Each layer reads forward and backward simultaneously. Outputs are concatenated and projected back to `d_model`.
- **Non-causal attention**: Every position attends to every other position in the window (no causal mask). For a fully-observed 2-second window, this gives the pooling head maximal information.
- **Masked attention**: Padding tokens are excluded from the attention score computation.

```
Input (B, T, 5)
  -> Linear projection -> (B, T, d_model)
  -> Bidirectional GRU stack (n_layers)
  -> Non-causal MHA pooling with padding mask
  -> LayerNorm + residual -> (B, d_model)
  -> Classifier head -> (B, num_classes)
```

### Classifier Heads

Two head variants are supported, selectable with `--use_prototype_head`:

| Mode | Head | Best For |
|------|------|----------|
| Default | Linear softmax | Balanced datasets |
| `--use_prototype_head` | `HybridClassifier` (linear + prototypical distance) | Imbalanced, few-shot cohorts |

The `HybridClassifier` in `prototype_head.py` combines a softmax logit with a cosine-distance prototype score. This prevents minority classes from collapsing to zero probability by ensuring every class maintains a representative prototype in embedding space.

---

## Feature Extraction

> `dataset.py`

Each CSV file is segmented into sliding windows of `window_size` (default: 2000 samples = 2 seconds at 1kHz). For every window, **5 features** are computed per timestep:

| Feature | Description |
|---------|-------------|
| `e_ant` | Antagonist muscle envelope (Gaussian-smoothed rectified EMG) |
| `e_ago` | Agonist muscle envelope |
| `torque` | Computed as `e_ago - e_ant` (net driving torque proxy) |
| `stiffness` | Computed as `e_ago + e_ant` (co-contraction stiffness proxy) |
| `gait_phase` | Continuous gait phase (0-100%) computed via `gait_phase.py` |

**Global normalization** is calculated on the training set only and frozen before being applied to the validation and test sets — preventing any data leakage through the normalizer.

**Augmentation** (`augmentations.py`) is applied only to amplitude channels (`e_ant`, `e_ago`, `torque`, `stiffness`) during training. The `gait_phase` channel is never augmented. Techniques include:

- Time warping (random temporal stretching/compression)
- Magnitude warping (smooth gain perturbation)
- Channel dropout (entire channel zeroed with probability `p`)
- Additive Gaussian jitter

---

## Project Structure

```
GaitAnalysis/
|
├── models/
|   ├── __init__.py              # Exports all model classes
|   ├── bimamba_classifier.py    # Bidirectional Mamba (recommended)
|   ├── gru_baseline.py          # Bidirectional GRU + Non-Causal Attention
|   ├── native_mamba.py          # Unidirectional Mamba (legacy, functional)
|   ├── triton_mamba.py          # Hardware-accelerated Mamba (legacy, functional)
|   └── official_mamba.py        # Official mamba-ssm wrapper
|
├── legacy/                      # Archived prior implementations
|   ├── dataset.py               # FeatureConfig-based 22-feature dataset
|   ├── train.py                 # Random-split training script
|   ├── gru_baseline.py          # Original causal GRU
|   └── feature_config.py        # 22-feature EDA configuration
|
├── EDA/                         # Exploratory Data Analysis notebooks & tools
|
├── dataset.py                   # GaitDataset: windowing, augmentation, LOPO splits
├── train.py                     # Primary training entry point (LOPO/K-Fold CV)
├── evaluate.py                  # Checkpoint-aware evaluation (auto-detects config)
├── augmentations.py             # Physiological EMG augmentation pipeline
├── prototype_head.py            # PrototypeClassifier & HybridClassifier
├── checkpoint_utils.py          # Architecture-baked checkpoint save/load
├── gait_phase.py                # Continuous gait phase computation (PCHIP splines)
├── inspect_data.py              # Quick CSV inspection utility
|
├── submit_gait.sh               # PBS/Torque HPC job submission script
├── connect_hpc.bat              # SSH helper for Windows -> HPC
├── Dockerfile                   # Container definition
└── README.md
```

---

## Installation

```bash
conda create -n gait_env python=3.11
conda activate gait_env
pip install torch torchvision pandas numpy scikit-learn tqdm matplotlib seaborn

# Optional: hardware-accelerated Mamba (Linux/CUDA only)
pip install mamba-ssm triton
```

---

## Dataset Format

Organise your CSV files as follows. The folder name format is `{patient_id}_{class_name}`:

```
Datasets/
  01_Normal/
    patient_01.csv
    patient_02.csv
  02_Piriformis/
    patient_11.csv
    patient_12.csv
  03_PIVD/
    patient_21.csv
```

Each CSV should contain **at least 2 columns** of raw EMG signals (no header). The first column is treated as the antagonist (e.g., Tibialis Anterior) and the second as the agonist (e.g., Gastrocnemius). Sampling frequency is inferred automatically from timestamps if available, defaulting to 1000 Hz.

You can inspect your dataset at any time with:

```bash
python inspect_data.py --data_dir Datasets/
```

---

## Training

`train.py` is the single entry point for all training. It handles data loading, patient-level cross-validation, normalization, augmentation, and checkpointing automatically.

### Full CLI Reference

```
python train.py [OPTIONS]

Required:
  --data_dir PATH          Root directory of the dataset

Model:
  --model_type STR         Model architecture (default: bimamba)
                           Choices: bimamba | gru | mamba | triton_mamba
  --d_model INT            Hidden state dimension (default: 64)
  --n_layers INT           Number of stacked model layers (default: 2)
  --dropout FLOAT          Dropout probability (default: 0.1)
  --use_prototype_head     Use HybridClassifier instead of linear head

Training:
  --epochs INT             Max training epochs per fold (default: 60)
  --batch_size INT         Batch size (default: 4)
  --accum_steps INT        Gradient accumulation steps (default: 4)
  --lr FLOAT               Peak learning rate for OneCycleLR (default: 5e-4)
  --focal_gamma FLOAT      Focal loss gamma (default: 2.0)
  --patience INT           Early stopping patience in epochs (default: 20)
  --seed INT               Global random seed (default: 42)

Data:
  --window_size INT        Samples per sliding window (default: 2000)
  --stride INT             Window stride in samples (default: 1000)

Cross-Validation:
  --cv_mode STR            lopo (default) or kfold
  --n_folds INT            Number of folds for kfold mode (default: 5)
  --single_fold            Run only the first fold (useful for quick sanity checks)

Output:
  --output_dir PATH        Directory for saved checkpoints (default: checkpoints/)
```

### Common Recipes

```bash
# Gold standard: BiMamba LOPO (one fold per patient)
python train.py --data_dir Datasets/ --model_type bimamba --epochs 80

# For minority-class robustness, use the prototype head
python train.py --data_dir Datasets/ --model_type bimamba --use_prototype_head --epochs 80

# GRU baseline comparison
python train.py --data_dir Datasets/ --model_type gru --epochs 80

# Faster iteration: K-Fold instead of LOPO
python train.py --data_dir Datasets/ --model_type bimamba --cv_mode kfold --n_folds 5

# Quick import/environment sanity check (single fold, 2 epochs)
python train.py --data_dir Datasets/ --model_type bimamba --single_fold --epochs 2
```

### What the Training Loop Does

1. **Splits** are generated at the **patient level** (never the window level) to prevent data leakage.
2. **Global mean/std** is computed on the training split only and frozen for val/test.
3. **Class weights** are derived from **patient counts** (not window counts) to avoid over-weighting classes that simply have longer recordings.
4. **Focal Loss** (`gamma=2.0`) is used to down-weight easy (majority) examples.
5. The best checkpoint per fold is selected by **macro F1** on the validation set, not accuracy.
6. Epoch printouts report **Min Per-Class Recall** — the key metric for pathology detection.
7. After all folds, aggregated results are printed as `mean +/- std` across folds.

---

## Evaluation

`evaluate.py` is checkpoint-aware: it reads the model architecture (type, `d_model`, `n_layers`, etc.) directly from the `.pth` file's embedded config, so you never need to remember which flags you used during training.

```bash
# Standard evaluation with confusion matrix
python evaluate.py --model_path checkpoints/best_fold_0.pth --data_dir Datasets/

# Include per-batch inference latency breakdown
python evaluate.py --model_path checkpoints/best_fold_0.pth --data_dir Datasets/ --benchmark

# Override config for legacy checkpoints without embedded config
python evaluate.py \
  --model_path best_mamba_model.pth \
  --data_dir Datasets/ \
  --model_type triton_mamba \
  --d_model 32 --n_layers 2
```

### Evaluation Output

```
==================================================
                EVALUATION RESULTS
==================================================
Accuracy:           0.8200 (82.00%)
Weighted F1:        0.8155
Macro F1:           0.7890
Min Per-Class Recall: 0.6667

Per-class recall: {'Normal': 0.917, 'Piriformis': 0.750, 'PIVD': 0.667}

==================================================
             INFERENCE METRICS
==================================================
Parameters:         146,530
Model size:         0.56 MB (FP32)
Latency / sample:   4.2 ms
Throughput:         238.1 samples/s
Peak GPU memory:    26.4 MB
==================================================
```

A confusion matrix PNG is saved to `--output_plot` (default: `confusion_matrix.png`).

---

## HPC Deployment

Configure the paths at the top of `submit_gait.sh` to match your HPC environment, then submit:

```bash
# Edit data path and conda env name if needed
nano submit_gait.sh

# Submit to PBS queue
qsub submit_gait.sh
```

The script runs BiMamba LOPO training, GRU LOPO training, and evaluation sequentially on a single H100 GPU node. Checkpoints are saved to `checkpoints/` in the working directory.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **LOPO cross-validation** | The only statistically valid evaluation strategy when N < 20 patients. Treats each patient as an unseen test subject. |
| **Patient-level splits** | Window-level splits cause >30% accuracy inflation from recording-to-recording correlation within a patient. |
| **Focal Loss** | Heavily suppresses easy (correct majority-class) gradients. `gamma=2.0` is the standard clinical imbalance setting. |
| **Patient-count class weights** | Using window counts instead penalises classes whose patients have shorter recordings, not those with fewer patients. |
| **Non-causal attention** | For a fully-observed 2-second window, restricting attention to past-only context is pointless. Full non-causal attention gives the pooling head complete information. |
| **Bidirectional GRU/Mamba** | Pathology manifests in the co-activation pattern over the full gait cycle, not just causal temporal progressions. Both directions are needed. |
| **Checkpoint config embedding** | Embedding arch params in the `.pth` prevents silent architecture mismatches at evaluation time. |
| **Macro F1 model selection** | Accuracy is dominated by the majority class. Macro F1 ensures minority pathology detection is rewarded equally. |

---

## License

Research Use Only.
