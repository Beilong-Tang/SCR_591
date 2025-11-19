# GAMLP+RLU+SCR 参数配置指南

本文档说明如何通过传参运行 **GAMLP+RLU+SCR** 版本，而不使用 Mean Teacher 或其他变体。

## 关键参数说明

### 必需的核心参数（启用 GAMLP+RLU+SCR）

1. **`--use-rlu`**: 启用可靠标签蒸馏（Reliable Label Distillation）
2. **`--method R_GAMLP_RLU`**: 使用 R_GAMLP_RLU 模型架构
3. **`--consis`**: **重要！** 启用一致性正则化（Consistency Regularization），这是 SCR 的核心
4. **`--ema`**: 启用指数移动平均（Exponential Moving Average），仅在第一个 stage 使用

### 关键：不设置这些参数

- **不要设置 `--mean_teacher`**: 不启用 Mean Teacher 机制
- **不要设置 `--kl`**: 不使用 KL 散度损失
- **不要设置 `--adap`**: 不使用自适应 EMA decay

### 训练逻辑说明

代码会根据参数选择不同的训练路径：

```python
if args.mean_teacher == False:
    if stage == 0:
        # 第一阶段：仅使用有标签数据训练
        train(model, ...)
    elif stage != 0 and args.consis == True:
        # 后续阶段：RLU + 一致性损失（SCR）
        train_rlu_consis(model, ...)
    elif stage != 0 and args.consis == False:
        # 后续阶段：仅 RLU，无一致性损失
        train_rlu(model, ...)
```

当设置 `--consis` 且不设置 `--mean_teacher` 时，会走 `train_rlu_consis()` 路径，这就是 **GAMLP+RLU+SCR**。

## 完整命令示例

### 方法 1: 使用提供的脚本

```bash
chmod +x run_gamlp_rlu_scr.sh
./run_gamlp_rlu_scr.sh
```

### 方法 2: 直接运行命令

**步骤 1: 数据预处理**
```bash
python pre_processing.py --num_hops 5 --dataset ogbn-products
```

**步骤 2: 训练 GAMLP+RLU+SCR**
```bash
python main.py \
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
    --ema
```

## 参数详细说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--use-rlu` | - | 启用可靠标签蒸馏 |
| `--method` | `R_GAMLP_RLU` | 使用 GAMLP+RLU 模型 |
| `--consis` | - | **关键！** 启用一致性损失（SCR） |
| `--stages` | `400 300 300 300 300 300` | 6 个训练阶段，每个阶段的 epoch 数 |
| `--threshold` | `0.85` | 伪标签置信度阈值 |
| `--gama` | `0.1` | KL 散度损失权重 |
| `--tem` | `0.5` | 温度参数（用于一致性损失） |
| `--lam` | `0.1` | 一致性损失权重 |
| `--ema` | - | 启用 EMA（第一个 stage） |
| `--hidden` | `512` | 隐藏层维度 |
| `--n-layers-1` | `4` | 第一组层数 |
| `--n-layers-2` | `4` | 第二组层数 |
| `--bns` | - | 使用 BatchNorm |

## 版本对比

| 版本 | `--consis` | `--mean_teacher` | `--kl` | 训练函数 |
|------|------------|------------------|--------|----------|
| **GAMLP+RLU+SCR** | ✅ | ❌ | ❌ | `train_rlu_consis()` |
| GAMLP+SCR | ❌ | ❌ | ❌ | `train()` (单阶段) |
| GAMLP+SCR-m | ❌ | ✅ | ✅ | `train_mean_teacher()` |
| GAMLP+RLU | ❌ | ❌ | ❌ | `train_rlu()` |

## 验证是否正确配置

运行时应该看到：
1. 第一阶段：仅使用有标签数据训练
2. 后续阶段：显示 "This history model Train/Valid/Test ACC is ..."
3. **不显示** "start mean teacher" 或 "use mean_teacher"
4. 使用 `train_rlu_consis()` 函数（可以通过打印日志确认）

## 预期结果

根据 README，GAMLP+RLU+SCR 在 ogbn-products 上的结果：
- **Validation accuracy**: 0.9292±0.0005
- **Test accuracy**: 0.8505±0.0009

加上 C&S 后处理可达到：
- **Test accuracy**: 0.8520±0.0008

