#!/bin/bash
# BTC 价格预测服务 - 停止脚本

echo "🛑 停止 BTC 预测服务..."

# 查找并终止进程
PIDS=$(pgrep -f "prediction_server.py" 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "⚠️ 未找到运行中的服务"
else
    for PID in $PIDS; do
        echo "   终止进程: $PID"
        kill "$PID" 2>/dev/null
    done
    echo "✅ 服务已停止"
fi

