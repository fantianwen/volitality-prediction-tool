# Telegram Bot 命令处理器

## 📋 功能

Telegram Bot 支持以下命令：

- `/predict-now` - 立即执行预测并发送结果到 Telegram
- `/start` - 显示欢迎信息和命令列表
- `/help` - 显示帮助信息

## 🚀 使用方法

### 1. 独立运行 Bot（推荐）

Bot 可以独立运行，监听 Telegram 命令：

```bash
cd project/scripts
source ../venv/bin/activate  # 如果在虚拟环境中

python3 telegram_bot.py \
    --model ../models/regression_model_20251213_213205.pkl \
    --telegram-token YOUR_TOKEN \
    --telegram-chat-id YOUR_CHAT_ID
```

### 2. 与预测服务一起运行

Bot 和预测服务可以同时运行：

```bash
# 终端 1: 运行预测服务（每小时自动预测）
python3 prediction_server.py --model ../models/regression_model_20251213_213205.pkl

# 终端 2: 运行 Telegram Bot（监听命令）
python3 telegram_bot.py --model ../models/regression_model_20251213_213205.pkl
```

### 3. 使用 Systemd 服务

创建 systemd 服务文件 `/etc/systemd/system/telegram-bot.service`:

```ini
[Unit]
Description=Telegram Bot for BTC Prediction
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/btc-predictor/scripts
Environment="PATH=/home/ubuntu/btc-predictor/venv/bin:$PATH"
EnvironmentFile=/home/ubuntu/btc-predictor/scripts/.env

ExecStart=/home/ubuntu/btc-predictor/venv/bin/python3 telegram_bot.py \
    --model ../models/regression_model_20251213_213205.pkl

Restart=always
RestartSec=10

StandardOutput=append:/home/ubuntu/btc-predictor/logs/telegram_bot.log
StandardError=append:/home/ubuntu/btc-predictor/logs/telegram_bot.error.log

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable telegram-bot
sudo systemctl start telegram-bot
sudo systemctl status telegram-bot
```

## 📱 命令说明

### /predict-now

立即执行一次价格预测并发送详细报告。

**示例：**
```
用户: /predict-now

Bot: ⏳ 正在执行预测，请稍候...

Bot: [发送完整的预测报告]
```

### /start

显示欢迎信息和可用命令列表。

### /help

显示详细的帮助信息。

## 🔧 配置

Bot 使用与预测服务相同的配置：

- `TELEGRAM_BOT_TOKEN` - Telegram Bot Token
- `TELEGRAM_CHAT_ID` - Telegram Chat ID
- `MODEL_PATH` - 模型文件路径（通过命令行参数传递）

## 📊 工作流程

1. Bot 启动后，使用长轮询（long polling）监听 Telegram 消息
2. 当收到 `/predict-now` 命令时：
   - 发送"正在处理"消息
   - 调用预测服务器执行预测
   - 格式化预测结果
   - 发送完整报告到 Telegram

## ⚠️ 注意事项

1. **Chat ID 限制**: Bot 只响应配置的 Chat ID 发送的命令，忽略其他用户
2. **并发处理**: 当前实现是单线程的，同时收到多个命令会顺序处理
3. **错误处理**: 如果预测失败，会发送错误消息到 Telegram
4. **日志记录**: 所有命令和错误都会记录到日志

## 🐛 故障排查

### Bot 无法接收命令

1. 检查 Bot Token 是否正确
2. 检查 Chat ID 是否正确
3. 确保已向 Bot 发送过消息
4. 查看日志：`tail -f logs/telegram_bot.log`

### 命令无响应

1. 检查预测服务器是否正常工作
2. 检查模型文件是否存在
3. 查看错误日志：`tail -f logs/telegram_bot.error.log`

### 预测失败

1. 检查网络连接（需要访问 Binance API）
2. 检查模型文件路径
3. 查看预测服务器日志

## 📝 示例对话

```
用户: /start

Bot: 🤖 BTC 价格预测 Bot

可用命令:
/predict-now - 立即执行预测并发送结果
/help - 显示帮助信息

自动预测:
服务每小时整点自动执行预测并发送报告。

---

用户: /predict-now

Bot: ⏳ 正在执行预测，请稍候...

Bot: 🔮 BTC 价格预测报告
━━━━━━━━━━━━━━━━━━━━━━━━

📅 预测时间: 2025-12-14 17:30 (UTC+8)
🎯 预测目标: 2025-12-15 13:30 (UTC+8, 20h后)

💰 当前价格: $90,095.10
🎯 预测价格: $88,598.92
💵 资金费率: 0.0020%

...
```

