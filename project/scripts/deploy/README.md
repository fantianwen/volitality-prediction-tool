# BTC 价格预测服务 - 部署指南

## 📋 功能概述

- 📊 实时从 Binance 获取多时间框架 K 线数据
- 🤖 使用 GBM 模型预测未来 20 小时价格走势
- ⏰ 每小时整点自动预测
- 📱 通过 Telegram 机器人发送预测报告
- 📈 输出涨跌方向、涨跌区间、置信度

## 🚀 快速开始

### 1. 安装依赖

```bash
cd project/scripts
pip install -r requirements.txt
```

### 2. 配置 Telegram 机器人

1. 在 Telegram 中搜索 `@BotFather`
2. 发送 `/newbot` 创建新机器人
3. 获取 Bot Token（格式：`123456789:ABCdef...`）
4. 发送消息给你的机器人
5. 访问 `https://api.telegram.org/bot<TOKEN>/getUpdates` 获取 Chat ID

### 3. 设置环境变量

```bash
# 方法1: 复制并编辑 .env 文件
cp .env.example .env
nano .env

# 方法2: 直接导出环境变量
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

### 4. 运行服务

```bash
# 测试模式 (执行一次预测后退出)
python prediction_server.py --model ../models/regression_model_20251213_213205.pkl --test

# 前台运行
python prediction_server.py --model ../models/regression_model_20251213_213205.pkl

# 后台运行 (使用脚本)
cd deploy
chmod +x start.sh stop.sh
./start.sh --background
```

## 📦 服务器部署

### 使用 Systemd (推荐)

```bash
# 1. 编辑服务文件
sudo cp deploy/btc-predictor.service /etc/systemd/system/

# 2. 修改配置
sudo nano /etc/systemd/system/btc-predictor.service
# - 修改 WorkingDirectory
# - 修改 ExecStart 中的路径
# - 配置环境变量

# 3. 启用并启动
sudo systemctl daemon-reload
sudo systemctl enable btc-predictor
sudo systemctl start btc-predictor

# 4. 查看状态
sudo systemctl status btc-predictor
journalctl -u btc-predictor -f
```

### 使用 Screen/Tmux

```bash
# Screen
screen -S btc-predictor
./deploy/start.sh
# Ctrl+A, D 分离

# Tmux
tmux new -s btc-predictor
./deploy/start.sh
# Ctrl+B, D 分离
```

### 使用 Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY project/scripts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY project/scripts /app/scripts
COPY project/models /app/models

WORKDIR /app/scripts
CMD ["python", "prediction_server.py", "--model", "../models/regression_model_20251213_213205.pkl"]
```

```bash
docker build -t btc-predictor .
docker run -d --name btc-predictor \
  -e TELEGRAM_BOT_TOKEN="your_token" \
  -e TELEGRAM_CHAT_ID="your_chat_id" \
  btc-predictor
```

## 📊 预测报告示例

```
🔮 BTC 价格预测报告
━━━━━━━━━━━━━━━━━━━━━━━━

📅 预测时间: 2024-12-14 15:00
🎯 预测目标: 2024-12-15 11:00 (20h后)

💰 当前价格: $102,345.67
💵 资金费率: 0.0060%

━━━━━━━━━━━━━━━━━━━━━━━━
📊 预测结果

📈 方向: 看涨
🟢 区间: 小涨 (0.5% ~ 2%)
📈 预测涨跌: +1.23%
🎯 置信度: 65%

━━━━━━━━━━━━━━━━━━━━━━━━
📈 市场状态

📊 信号强度: 2
📉 RSI(1h): 55.3
📈 ADX(1h): 28.5
⚡ 波动率: 1.25%

━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 仅供参考，不构成投资建议
```

## 📁 文件结构

```
project/
├── scripts/
│   ├── prediction_server.py  # 主预测服务
│   ├── data_collector.py     # 数据收集器
│   ├── train_model.py        # 模型训练
│   ├── config.py             # 配置管理
│   ├── requirements.txt      # 依赖
│   └── deploy/
│       ├── README.md         # 本文档
│       ├── start.sh          # 启动脚本
│       ├── stop.sh           # 停止脚本
│       └── btc-predictor.service  # Systemd 服务
├── models/
│   └── regression_model_*.pkl  # 训练好的模型
└── data/
    └── BTCUSDT_features_*.csv  # 历史数据
```

## ⚙️ 命令行参数

```bash
python prediction_server.py --help

参数:
  --model PATH          模型文件路径 (.pkl) [必需]
  --symbol SYMBOL       交易对 (默认: BTCUSDT)
  --telegram-token TOK  Telegram Bot Token
  --telegram-chat-id ID Telegram Chat ID
  --test                测试模式 (执行一次后退出)
```

## ❓ 常见问题

### Q: 如何更换模型？
A: 使用 `--model` 参数指定新模型路径，或修改 `.env` 中的 `MODEL_PATH`。

### Q: 如何修改预测频率？
A: 当前固定为每小时整点预测。如需修改，编辑 `prediction_server.py` 中的 `schedule` 配置。

### Q: 预测不准确怎么办？
A: 
1. 使用更长时间的历史数据重新训练
2. 尝试不同的模型（GBM, RF, LSTM）
3. 调整置信度阈值过滤低置信预测

### Q: SSL 证书错误？
A: 脚本已禁用 SSL 验证。如需启用，修改 `ssl_context` 配置。

## 📝 更新日志

- **v1.0.0** (2024-12-14)
  - 初始版本
  - 支持 GBM 模型预测
  - Telegram 通知
  - 每小时自动预测

