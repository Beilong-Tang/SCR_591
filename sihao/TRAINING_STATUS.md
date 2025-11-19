# 训练状态报告

## 当前运行情况

### ✅ 新版本训练已启动（基于熵权法的注意力机制）

**进程信息：**
- PID: 2018335
- 启动时间: 2025-11-18 19:43:18
- 日志文件: `training_entropy_attention_20251118_194318.log`
- 状态: 正在初始化（计算标签嵌入）

**训练配置：**
- 方法: GAMLP+RLU+SCR with **Entropy-based Attention**
- 数据集: ogbn-products (240万节点)
- 阶段数: 6个 (400, 300, 300, 300, 300, 300 epochs)
- 运行次数: 10次

**关键改进：**
- ✅ 使用 `consis_loss_entropy_attention` 函数
- ✅ 基于信息熵的动态权重分配
- ✅ 低熵（高置信度）预测获得更高权重

### 📊 对比训练（原版本）

**进程信息：**
- PID: 2010443
- 状态: 运行中（已运行约16小时）
- 日志文件: `training.log`

---

## 监控命令

### 实时查看训练日志
```bash
cd /mnt/data/sliu78/SCR/ogbn-products
tail -f training_entropy_attention_20251118_194318.log
```

### 查看训练进度
```bash
bash monitor_training.sh
```

### 查看最新训练输出
```bash
tail -50 training_entropy_attention_20251118_194318.log | grep -E "Epoch|Train|Val|Test|Best|Params"
```

### 检查进程状态
```bash
ps aux | grep "python.*main.py.*R_GAMLP_RLU" | grep -v grep
```

---

## 预期输出

训练开始后，日志中应该看到：

1. **初始化阶段**（当前）:
   - 加载特征文件
   - 计算标签嵌入（9跳标签传播，可能需要几分钟）

2. **训练阶段**:
   ```
   GAMLP
   # Params: 3335831
   Epoch 0, Time(s): XX.XXXX, Train loss: X.XXXX, Train acc: XX.XXXX
   Validation: Time(s): XX.XXXX, Val X.XXXX, Best Epoch X, Val X.XXXX, Test X.XXXX
   ```

3. **多阶段训练**:
   - Stage 0: 仅使用有标签数据
   - Stage 1-5: 使用伪标签 + 一致性损失（熵权法）

---

## 代码修改说明

### 修改的文件
- `utils.py`: 
  - 新增 `consis_loss_entropy_attention()` 函数（第66-116行）
  - 修改 `train_rlu_consis()` 使用新函数（第161行）

### 核心改进
```python
# 原方法：简单平均
avg_p = torch.mean(ps, dim=2)

# 新方法：基于熵的加权平均
entropy = -torch.sum(ps * torch.log(ps + 1e-8), dim=1)
att_weights = F.softmax(-entropy, dim=1)
avg_p = torch.sum(ps * att_weights.unsqueeze(1), dim=2)
```

---

## 预期结果对比

| 版本 | 方法 | 预期 Test Acc |
|------|------|--------------|
| 原版本 | GAMLP+RLU+SCR (简单平均) | 0.8505±0.0009 |
| **新版本** | **GAMLP+RLU+SCR (熵权法)** | **待测试** |

---

## 注意事项

1. 标签嵌入计算需要时间：对于240万节点，9跳传播可能需要几分钟
2. 两个训练共享GPU：如果显存不足，可能需要等待原训练完成
3. 日志可能被缓冲：使用 `tail -f` 实时查看最新输出

---

**最后更新**: 2025-11-18 19:48

