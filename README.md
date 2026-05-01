# Gait Analysis — Robust Pathology Classification

A research-grade pipeline for classifying gait pathology (e.g., PIVD, Piriformis Syndrome) using EMG-driven muscle activation data. This repository implements state-of-the-art architectures including Bidirectional Mamba (BiMamba) and Attention-enhanced GRUs with a focus on robustness to small, imbalanced biomedical cohorts.

## 🚀 Key Features

*   **Unified Pipeline**: Clean, modular structure for training (`train.py`) and evaluation (`evaluate.py`).
*   **State-of-the-Art Models**:
    *   **BiMamba**: Bidirectional Mamba with padding-aware state resets and gated fusion.
    *   **GRU+Attention**: Bidirectional GRU with non-causal multi-head attention pooling.
*   **Robust Evaluation**: Built-in Leave-One-Patient-Out (LOPO) and Stratified K-Fold cross-validation to prevent data leakage.
*   **Imbalance Handling**: 
    *   **Focal Loss** with patient-level inverse-frequency weighting.
    *   **Hybrid Prototype Classifier**: Distance-based classification to prevent minority class collapse.
*   **Advanced Augmentation**: Physiological EMG augmentations (time-warping, channel dropout, magnitude warping) via `augmentations.py`.

---

## 📂 Project Structure

```
GaitAnalysis/
├── models/
│   ├── bimamba_classifier.py    # Bidirectional Mamba (Recommended)
│   ├── gru_baseline.py          # Bidirectional GRU + Attention
│   └── native_mamba.py          # Original Mamba architecture
├── legacy/                      # Archived research iterations
├── dataset.py                   # Per-window feature extraction & data loading
├── train.py                     # LOPO/K-Fold Training Entry Point
├── evaluate.py                  # Checkpoint-aware Evaluation (Auto-detects config)
├── augmentations.py             # EMG signal augmentation pipeline
├── prototype_head.py            # Prototypical network components
├── checkpoint_utils.py          # Architecture-baked checkpointing
├── gait_phase.py                # Physiological gait phase computation
└── submit_gait.sh               # HPC job submission script (PBS/Torque)
```

---

## 🛠️ Installation

```bash
conda create -n gait_env python=3.11
conda activate gait_env
pip install torch pandas numpy scikit-learn tqdm matplotlib seaborn
# Optional: pip install mamba-ssm triton (for hardware acceleration)
```

---

## 🏋️ Training

The training script automatically handles data splitting, normalization, and checkpointing.

### Recommended (BiMamba + LOPO)
```bash
python train.py --data_dir Datasets/ --model_type bimamba --epochs 80
```

### Prototypical Training (For high imbalance)
```bash
python train.py --data_dir Datasets/ --model_type bimamba --use_prototype_head
```

### Fast Iteration (K-Fold)
```bash
python train.py --data_dir Datasets/ --model_type gru --cv_mode kfold --n_folds 5
```

---

## 📊 Evaluation

The evaluation script auto-detects the architecture from the checkpoint file.

```bash
# Basic evaluation
python evaluate.py --model_path checkpoints/best_model.pth --data_dir Datasets/

# Benchmarking inference speed
python evaluate.py --model_path checkpoints/best_model.pth --data_dir Datasets/ --benchmark
```

---

## ☁️ HPC Deployment

To run on a cluster (H100 GPU), configure your paths in `submit_gait.sh` and run:

```bash
qsub submit_gait.sh
```

---

## 📝 License
Research Use Only.
