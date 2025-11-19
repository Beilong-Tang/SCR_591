#!/bin/bash

# 监控训练进度的脚本

LOG_FILE="training_entropy_attention_20251118_194318.log"

echo "=== 训练监控 ==="
echo "日志文件: $LOG_FILE"
echo ""

# 检查进程
echo "--- 运行中的训练进程 ---"
ps aux | grep "python.*main.py.*R_GAMLP_RLU" | grep -v grep | awk '{print "PID:", $2, "| CPU:", $3"% | MEM:", $4"% | 运行时间:", $10}'
echo ""

# 查看最新日志
echo "--- 最新训练日志 (最后30行) ---"
if [ -f "$LOG_FILE" ]; then
    tail -30 "$LOG_FILE"
else
    echo "日志文件不存在"
fi

echo ""
echo "--- GPU 使用情况 ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "无法获取GPU信息"

