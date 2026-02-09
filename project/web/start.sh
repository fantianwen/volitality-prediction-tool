#!/bin/bash
# Web UI 启动脚本

cd "$(dirname "$0")"

echo "🚀 Starting BTC Prediction Web Dashboard..."
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# 检查依赖
echo "📦 Checking dependencies..."
if ! python3 -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask not found. Installing dependencies..."
    pip install -r requirements.txt
    pip install -r ../scripts/requirements.txt
fi

# 检查模型文件
MODEL_PATH="${MODEL_PATH:-../models/regression_model_20251213_213205.pkl}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Warning: Model file not found at $MODEL_PATH"
    echo "   Please set MODEL_PATH environment variable or ensure model exists"
    echo ""
fi

# 获取端口（从环境变量或使用默认值 9000）
PORT="${PORT:-9000}"

# 启动服务器
echo "✅ Starting Flask server..."
echo "📊 Dashboard will be available at: http://localhost:${PORT}"
echo ""
echo "💡 Tip: Set PORT environment variable to change port"
echo "   Example: PORT=3000 ./start.sh"
echo ""
echo "💡 To run in background:"
echo "   - Production: sudo systemctl start btc-predictor-web"
echo "   - Quick test: nohup python3 app.py --port ${PORT} > ../logs/web_ui.log 2>&1 &"
echo "   - Debug: tmux new -s webui (then run this script)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

PORT=$PORT python3 app.py

