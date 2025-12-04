#!/bin/bash

echo "Training GAMLP+RLU+SCR with Multihead Attention (K=5)..."
echo "Features:"
echo "  - Learnable Multihead Attention"
echo "  - K = 5 forward passes (num_heads = 5)"
echo "  - Learnable query vector"
echo "  - num-runs: 3"
echo ""

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
    --num-runs 3 \
    --eval-every 10 \
    --act leaky_relu \
    --batch_size 50000 \
    --patience 300 \
    --n-layers-1 4 \
    --n-layers-2 4 \
    --bns \
    --gama 0.1 \
    --consis \
    --use-mha \
    --num-passes 5 \
    --tem 0.5 \
    --lam 0.1 \
    --hidden 512 \
    --ema \
    --lr 0.001 \
    --weight-decay 0


