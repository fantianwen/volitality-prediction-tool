# BTC 价格预测服务器 - 使用指南

## 🎯 功能

- ✅ 使用 GBM 模型进行价格预测
- ✅ 实时从 Binance 获取多时间框架数据
- ✅ 每小时整点自动预测
- ✅ 输出涨跌方向、涨跌区间、置信度
- ✅ Telegram 机器人通知

## 🚀 快速开始

### 1. 安装依赖

```bash
cd project/scripts
pip install -r requirements.txt
```

### 2. 配置 Telegram（可选）

```bash
# 设置环境变量
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

### 3. 运行服务

```bash
# 测试模式（执行一次预测后退出）
python prediction_server.py --model ../models/regression_model_20251213_213205.pkl --test

# 前台运行（每小时整点预测）
python prediction_server.py --model ../models/regression_model_20251213_213205.pkl

# 后台运行
cd deploy
chmod +x start.sh
./start.sh --background
```

## 📊 预测报告示例

```
🔮 BTC 价格预测报告
━━━━━━━━━━━━━━━━━━━━━━━━

📅 预测时间: 2025-12-14 14:01
🎯 预测目标: 2025-12-15 10:01 (20h后)

💰 当前价格: $90,144.80
💵 资金费率: 0.0032%

━━━━━━━━━━━━━━━━━━━━━━━━
📊 预测结果

📉 方向: 看跌
🔴 区间: 小跌 (-2% ~ -0.5%)
📈 预测涨跌: -1.51%
🎯 置信度: 30%

━━━━━━━━━━━━━━━━━━━━━━━━
📈 市场状态

📊 信号强度: 0.0
📉 RSI(1h): 43.2
📈 ADX(1h): 25.2
⚡ 波动率: 43.65%

━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 仅供参考，不构成投资建议
```

## 📋 涨跌区间定义

- 🔴🔴 **大跌**: < -2%
- 🔴 **小跌**: -2% ~ -0.5%
- ⚖️ **横盘**: -0.5% ~ 0.5%
- 🟢 **小涨**: 0.5% ~ 2%
- 🟢🟢 **大涨**: > 2%

## ⚙️ 命令行参数

```bash
python prediction_server.py --help

必需参数:
  --model PATH          模型文件路径 (.pkl)

可选参数:
  --symbol SYMBOL       交易对 (默认: BTCUSDT)
  --telegram-token TOK  Telegram Bot Token
  --telegram-chat-id ID Telegram Chat ID
  --test                测试模式（执行一次后退出）
```

## 🔧 服务器部署

详见 `deploy/README.md`

