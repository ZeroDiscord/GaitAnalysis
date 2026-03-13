# Biomechanics-Informed Gait Pathology Classification
> **A PyTorch pipeline to implement hardware-accelerated Mamba architectures and physics-informed features (Torque/Stiffness) to classify clinical gait pathologies over extreme-length time-series.**

This repository contains a full PyTorch pipeline to classify gait pathology datasets using State-Space Models (Mamba) and Baseline RNNs. The system ingests raw time-series muscle activations ($E_{ant}$ and $E_{ago}$), automatically computes predictive Physics features (Torque, Joint Stiffness), and trains an end-to-end classifier over extreme sequence lengths.

## Pipeline Architecture (H100 HPC Optimization)
The initial draft has been substantially upgraded to ensure mathematical convergence on extreme sequence lengths ($26,000+$ timesteps) and extreme low-sample biological datasets (e.g., 5 patients per class).

### 1. Robust Data Processing
*   **Patient-Level Strict Splitting**: Explicitly guarantees that biological samples (CSV files) belonging to the same subject are isolated into Train/Val/Test subsets *before* slicing. This mathematically guarantees zero data leakage of patient muscle signatures across evaluation sets.
*   **Sliding Window Segmentation**: $26,000+$ length raw sequences are sliced into smaller $2,000$ length context windows. A dataset of 5 patients is inherently multiplied into hundreds of robust training iterations, vastly stabilizing gradient descent.
*   **Dynamic Stride Class Balancing**: Replaces standard loss weighting. The dataloader dynamically shrinks the sliding window stride for minority classes, natively balancing the dataset perfectly by generating equal amounts of windows per class regardless of raw patient counts.
*   **Global Component Normalization**: To preserve absolute biological amplitude differences (which z-score destructively erases), `global_mean` and `global_std` are extracted strictly from the training patient subset and applied uniformly across Train/Val/Test.
*   **Temporal Phase Jittering**: Triggers $\pm 5\%$ boundary randomization when parsing training windows to instill phase-invariance to gait initiation.

### 2. Neural Stability
*   **OOM Protection**: The GRU Baseline acts without `nn.MultiheadAttention`, replaced with **PyTorch 2.0 FlashAttention** (`F.scaled_dot_product_attention`) for pure $O(N)$ execution. native `bfloat16` precision enforced to prevent `NaN` exploding gradients.

## Requirements
pip install torch pandas numpy scikit-learn tqdm

To enable the hardware-accelerated Triton or Official Mamba kernels, you need an NVIDIA GPU with a compiled CUDA backend:
```bash
# To use --use_triton_mamba (Requires Linux or WSL)
pip install triton

# To use --use_official_mamba (Requires CUDA compilation tools)
pip install causal-conv1d>=1.2.0 mamba-ssm
```

## HPC Deployment
To quickly connect to the HPC cluster and activate the PyTorch environment, a helper script is provided for Windows users (`connect_hpc.bat`) and Linux/macOS users (`connect_hpc.sh`):

**Windows (Command Prompt / PowerShell):**
```cmd
:: General usage
connect_hpc.bat [username] [hpc_address]

:: Example (Default uses aantriksh.124259 and 10.16.1.50)
connect_hpc.bat
```

**Linux / Mac / Git Bash:**
```bash
./connect_hpc.sh
```

This script will SSH into the node, load CUDA, activate the `gait_env` conda environment, and navigate to the project directory automatically. You can then submit your training jobs using `qsub submit_gait.sh`.


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

## 3. Metrics and Evaluation

The training script automatically computes proportional class weights to counteract imbalanced datasets. Upon evaluation, it yields:
- `Loss` (CrossEntropy)
- `Accuracy`
- `F1 Score` (Weighted)
- `ROC-AUC` (OVR multi-class)
- `Confusion Matrix`

The top performing model (by highest validation F1 score) is saved to the filename specified. By default, this is `best_model.pth`.

### Running Standalone Evaluation
You can evaluate a trained model using the `evaluate.py` script. The script generates a full classification report (precision/recall per class) and outputs a high-resolution confusion matrix image.

Ensure that the model architecture flags (`--d_model`, `--n_layers` and the `--use_*_mamba` flags) perfectly match how the model was trained!

```bash
# Evaluate Native PyTorch Mamba
python evaluate.py --model_path best_model.pth --data_dir Datasets/

# Evaluate GRU Baseline
python evaluate.py --model_path best_gru_baseline.pth --use_gru_baseline

# Evaluate Triton Mamba with custom plot name
python evaluate.py --model_path best_mamba_model.pth --use_triton_mamba --output_plot triton_cm.png
```
