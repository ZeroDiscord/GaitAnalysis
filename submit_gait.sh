#!/bin/bash
#PBS -N gaitAnalysis
#PBS -q gpu
#PBS -l select=1:ncpus=10:ngpus=1:mem=128g
#PBS -j oe
#PBS -V

cd "/home/aantriksh.124259/Gait Analysis"

source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate gait_env
module load cuda

# Makes CUDA errors synchronous — gives accurate stack traces if it crashes
export CUDA_LAUNCH_BLOCKING=1

# Auto-select the GPU with the most free memory
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{print $2}')
echo "Selected GPU: $CUDA_VISIBLE_DEVICES"

echo "--- GPU STATUS BEFORE TRAINING ---"
nvidia-smi
echo "----------------------------------"

# 1. BiMamba — LOPO Cross-Validation
echo "--- Starting BiMamba LOPO Training on H100 ---"
python3 train.py \
    --data_dir "/home/aantriksh.124259/Datasets" \
    --model_type bimamba \
    --cv_mode lopo \
    --epochs 80 \
    --batch_size 16 \
    --accum_steps 1 \
    --lr 5e-4 \
    --d_model 64 \
    --n_layers 2 \
    --dropout 0.1 \
    --patience 20 \
    --window_size 500 \
    --stride 250 \
    --num_workers 4 \
    --output_dir "checkpoints/"

# 2. GRU Baseline — LOPO Cross-Validation
echo "--- Starting GRU LOPO Training on H100 ---"
python3 train.py \
    --data_dir "/home/aantriksh.124259/Datasets" \
    --model_type gru \
    --cv_mode lopo \
    --epochs 80 \
    --batch_size 16 \
    --accum_steps 1 \
    --lr 5e-4 \
    --d_model 64 \
    --n_layers 2 \
    --dropout 0.1 \
    --patience 20 \
    --window_size 500 \
    --stride 250 \
    --num_workers 4 \
    --output_dir "checkpoints/"

# 3. Evaluate best BiMamba checkpoint
echo "--- Evaluating BiMamba ---"
CHECKPOINT=$(find checkpoints/ -name "best_*.pth" -print -quit 2>/dev/null)
if [ -n "$CHECKPOINT" ]; then
    echo "Evaluating checkpoint: $CHECKPOINT"
    python3 evaluate.py \
        --model_path "$CHECKPOINT" \
        --data_dir "/home/aantriksh.124259/Datasets" \
        --output_plot "confusion_bimamba.png" \
        --benchmark
else
    echo "WARNING: No checkpoint found — training may have failed. Skipping evaluation."
fi