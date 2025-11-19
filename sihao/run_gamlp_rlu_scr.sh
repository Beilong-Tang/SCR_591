#!/bin/bash

# GAMLP+RLU+SCR 训练脚本
# 这个脚本运行 GAMLP+RLU+SCR 版本，不使用 Mean Teacher

# 步骤 1: 数据预处理
echo "Step 1: Pre-processing data..."
echo "y" | python pre_processing.py --num_hops 5 --dataset ogbn-products

# 步骤 2: 训练 GAMLP+RLU+SCR
echo "Step 2: Training GAMLP+RLU+SCR..."
echo "y" | python main.py \
    --use-rlu \
    --method R_GAMLP_RLU \
    --stages 400 300 300 300 300 300 \
    --train-num-epochs 0 0 0 0 0 0 \
    --threshold 0.85 \
    --input-drop 0.2 \
    --att-drop 0.5 \
    --label-drop 0 \
    --pre-process \
    --residual \
    --dataset ogbn-products \
    --num-runs 10 \
    --eval-every 10 \
    --act leaky_relu \
    --batch_size 50000 \
    --patience 300 \
    --n-layers-1 4 \
    --n-layers-2 4 \
    --bns \
    --gama 0.1 \
    --consis \
    --tem 0.5 \
    --lam 0.1 \
    --hidden 512 \
    --ema \
    --lr 0.001 \
    --weight-decay 0

