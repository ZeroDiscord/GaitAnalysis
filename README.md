# Biomechanics-Informed Gait Pathology Classification
> **A PyTorch pipeline to implement hardware-accelerated Mamba architectures and physics-informed features (Torque/Stiffness) to classify clinical gait pathologies over extreme-length time-series.**

This repository contains a full PyTorch pipeline to classify gait pathology datasets using State-Space Models (Mamba) and Baseline RNNs. The system ingests raw time-series muscle activations ($E_{ant}$ and $E_{ago}$), automatically computes predictive Physics features (Torque, Joint Stiffness), and trains an end-to-end classifier over extreme sequence lengths.

## 0. Prerequisites

The pipeline requires a standard Python environment with PyTorch.
```bash
# Basic requirements
pip install torch pandas numpy scikit-learn tqdm
```

To enable the hardware-accelerated Triton or Official Mamba kernels, you need an NVIDIA GPU with a compiled CUDA backend:
```bash
# To use --use_triton_mamba (Requires Linux or WSL)
pip install triton

# To use --use_official_mamba (Requires CUDA compilation tools)
pip install causal-conv1d>=1.2.0 mamba-ssm
```

## 1. Project Layout

The dataset loading expects a root directory (`Datasets/` by default) containing numerical folders matching disease states, populated with CSV files tracking muscle activation states.

```text
Gait Analysis/
├── dataset.py                # DataLoader and Physics computation
├── train.py                  # Main execution script
├── models/                   # Modular neural architectures
│   ├── native_mamba.py       # (Default) Compatible PyTorch Mamba
│   ├── triton_mamba.py       # Custom Fused CUDA/Triton Mamba
│   ├── official_mamba.py     # Wrapper for pip mamba-ssm
│   └── gru_baseline.py       # GRU + Masked Attention Baseline
└── Datasets/               
    ├── 1_Healthy/
    ├── 3_Hemiplegia/
    └── 8_Osteoarthiritis/
```

## 2. Training the Models

You can run the models using the following terminal commands. By default, **Automatic Mixed Precision (AMP) and Gradient Accumulation** are enabled to protect your GPU memory from out-of-memory errors over the 26,000 timestep un-downsampled sequences. 

> *Note: Change `--data_dir` to point to the actual path of your `Datasets` folder.*

**Train Native PyTorch Mamba (Highly Compatible / Default)**
```bash
python train.py --data_dir "Datasets/" --epochs 50 --batch_size 2 --accum_steps 8
```

**Train Baseline: GRU + KV Caching + Masked Multi-Head Attention**
```bash
python train.py --data_dir "Datasets/" --epochs 50 --batch_size 2 --accum_steps 8 --use_gru_baseline
```

**Train Custom Hardware Fused Mamba (Triton Acceleration)**
```bash
python train.py --data_dir "Datasets/" --epochs 50 --batch_size 2 --accum_steps 8 --use_triton_mamba
```

**Train Official State-Space Mamba (Albert Gu/Tri Dao Mamba-SSM)**
```bash
python train.py --data_dir "Datasets/" --epochs 50 --batch_size 2 --accum_steps 8 --use_official_mamba
```

## 3. Metrics Output

The training script automatically computes proportional class weights to counteract imbalanced datasets. Upon evaluation, it yields:
- `Loss` (CrossEntropy)
- `Accuracy`
- `F1 Score` (Weighted)
- `ROC-AUC` (OVR multi-class)
- `Confusion Matrix`

The top performing model (by highest validation F1 score) is saved to the filename specified. By default, this is `best_model.pth`. If training multiple models, you can prevent overwrites by using the `--output_name` parameter (e.g., `--output_name "gru_best.pth"`).
