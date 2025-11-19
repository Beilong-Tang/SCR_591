#!/bin/bash

# GAMLP+RLU+SCR with Entropy-based Attention 训练脚本
# 这个脚本运行 GAMLP+RLU+SCR 版本，使用基于熵权法的注意力机制

# 注意：代码已经修改为使用 consis_loss_entropy_attention
# 无需额外参数，已自动使用熵权法

# 步骤 2: 训练 GAMLP+RLU+SCR (数据预处理已完成，跳过)
echo "Step 2: Training GAMLP+RLU+SCR with Entropy-based Attention..."
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

