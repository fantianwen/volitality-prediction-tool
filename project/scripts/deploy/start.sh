#!/bin/bash
# BTC 价格预测服务 - 启动脚本
# 
# 使用方法:
#   ./start.sh               # 前台运行
#   ./start.sh --background  # 后台运行
#   ./start.sh --test        # 测试模式 (执行一次预测后退出)

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 切换到项目目录
cd "$PROJECT_DIR"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
pip3 install -q schedule pandas numpy scikit-learn 2>/dev/null || true

# 加载环境变量 (如果存在 .env 文件)
if [ -f ".env" ]; then
    echo "📂 加载 .env 配置..."
    set -a
    source .env
    set +a
fi

# 检查模型文件
MODEL_PATH="${MODEL_PATH:-../models/regression_model_20251213_213205.pkl}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型文件不存在: $MODEL_PATH"
    echo "   请设置正确的 MODEL_PATH 或确保模型文件存在"
    exit 1
fi

echo "🚀 启动 BTC 价格预测服务..."
echo "   模型: $MODEL_PATH"
echo "   交易对: ${SYMBOL:-BTCUSDT}"

if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    echo "   Telegram: 已配置 ✅"
else
    echo "   Telegram: 未配置 (仅输出到控制台)"
fi

# 解析参数
EXTRA_ARGS=""
BACKGROUND=false

for arg in "$@"; do
    case $arg in
        --background)
            BACKGROUND=true
            shift
            ;;
        --test)
            EXTRA_ARGS="$EXTRA_ARGS --test"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

# 构建命令
CMD="python3 prediction_server.py --model '$MODEL_PATH'"
if [ -n "$SYMBOL" ]; then
    CMD="$CMD --symbol $SYMBOL"
fi
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    CMD="$CMD --telegram-token '$TELEGRAM_BOT_TOKEN'"
fi
if [ -n "$TELEGRAM_CHAT_ID" ]; then
    CMD="$CMD --telegram-chat-id '$TELEGRAM_CHAT_ID'"
fi
CMD="$CMD $EXTRA_ARGS"

# 运行
if [ "$BACKGROUND" = true ]; then
    echo "🔄 后台运行模式..."
    nohup bash -c "$CMD" > prediction_server.log 2>&1 &
    echo "✅ 服务已启动 (PID: $!)"
    echo "   日志: $PROJECT_DIR/prediction_server.log"
    echo "   停止: kill $!"
else
    eval "$CMD"
fi

