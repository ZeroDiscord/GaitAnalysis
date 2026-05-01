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

# Use GPU 1 because GPU 0 is currently full on the node
export CUDA_VISIBLE_DEVICES=1

echo "--- Starting BiMamba LOPO Training on H100 ---"
python3 train.py \
    --data_dir "/home/aantriksh.124259/Datasets" \
    --model_type bimamba \
    --cv_mode lopo \
    --epochs 80 \
    --batch_size 16 \
    --accum_steps 2 \
    --lr 5e-4 \
    --d_model 64 \
    --n_layers 2 \
    --dropout 0.1 \
    --patience 20 \
    --output_dir "checkpoints/"

# 2. GRU Baseline — LOPO Cross-Validation
echo "--- Starting GRU LOPO Training on H100 ---"
python3 train.py \
    --data_dir "/home/aantriksh.124259/Datasets" \
    --model_type gru \
    --cv_mode lopo \
    --epochs 80 \
    --batch_size 16 \
    --accum_steps 2 \
    --lr 5e-4 \
    --d_model 64 \
    --n_layers 2 \
    --dropout 0.1 \
    --patience 20 \
    --output_dir "checkpoints/"

# 3. Evaluate best BiMamba checkpoint
echo "--- Evaluating BiMamba ---"
CHECKPOINT=$(ls checkpoints/best_*.pth | head -n 1)
echo "Evaluating checkpoint: $CHECKPOINT"
python3 evaluate.py \
    --model_path "$CHECKPOINT" \
    --data_dir "/home/aantriksh.124259/Datasets" \
    --output_plot "confusion_bimamba.png" \
    --benchmark
