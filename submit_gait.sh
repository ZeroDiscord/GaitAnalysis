#!/bin/bash
#PBS -N GaitMamba_H100
#PBS -q gpu
#PBS -l select=1:ncpus=10:ngpus=1:mem=32g
#PBS -j oe
#PBS -V 

# Navigate explicitly to the folder to bypass PBS_O_WORKDIR quote expansion errors
cd "/home/aantriksh.124259/Gait Analysis"

# Setup environment using Conda
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate gait_env
module load cuda

# 1. Run GRU Baseline Training First
echo "--- Starting GRU Baseline Training on H100 ---"
python3 train.py \
    --data_dir "/home/aantriksh.124259/Datasets" \
    --epochs 100 \
    --batch_size 16 \
    --accum_steps 2 \
    --use_gru_baseline \
    --output_name "best_gru_baseline.pth"

# 2. Run Mamba Training
echo "--- Starting Mamba Training on H100 ---"
python3 train.py \
    --data_dir "/home/aantriksh.124259/Datasets" \
    --epochs 100 \
    --batch_size 16 \
    --accum_steps 2 \
    --use_triton_mamba \
    --output_name "best_mamba_model.pth"
