# 快速修复 Telegram Bot

## 🔍 诊断结果

根据诊断，问题已找到：
- ✅ 配置正确
- ✅ Bot API 连接正常
- ✅ 已收到 `/predict-now` 命令
- ❌ **Bot 进程未运行** ← 这是问题所在！

## 🚀 立即修复

### 在服务器上执行：

```bash
# 1. SSH 到服务器
ssh -i trading-bot.pem ubuntu@54.250.16.16

# 2. 进入项目目录
cd /home/ubuntu/btc-predictor/scripts

# 3. 激活虚拟环境
source ../venv/bin/activate

# 4. 启动 Bot（前台测试）
python3 telegram_bot.py --model ../models/regression_model_20251213_213205.pkl
```

### 如果测试成功，后台运行：

```bash
# 停止前台进程（Ctrl+C），然后：
nohup python3 telegram_bot.py \
    --model ../models/regression_model_20251213_213205.pkl \
    > ../logs/telegram_bot.log 2>&1 &

# 查看日志
tail -f ../logs/telegram_bot.log
```

## 📋 验证步骤

1. **启动 Bot 后**，在 Telegram 中发送：
   - `/start` - 应该收到欢迎消息
   - `/predict-now` - 应该收到预测报告

2. **检查日志**：
   ```bash
   tail -f /home/ubuntu/btc-predictor/logs/telegram_bot.log
   ```

3. **运行诊断**：
   ```bash
   cd /home/ubuntu/btc-predictor/scripts
   source ../venv/bin/activate
   python3 diagnose_bot.py
   ```

## 🔧 使用 Systemd 服务（推荐）

创建服务文件：

```bash
sudo nano /etc/systemd/system/telegram-bot.service
```

内容：

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

## ⚠️ 常见问题

### Bot 启动后立即退出

检查错误日志：
```bash
tail -50 /home/ubuntu/btc-predictor/logs/telegram_bot.error.log
```

可能原因：
- 依赖缺失：运行 `pip install -r requirements.txt`
- 模型文件不存在：检查模型路径
- 配置错误：检查 `.env` 文件

### 命令无响应

1. 确认 Bot 正在运行：`ps aux | grep telegram_bot`
2. 检查日志：`tail -f ../logs/telegram_bot.log`
3. 确认 Chat ID 匹配：运行 `diagnose_bot.py`

### 预测失败

1. 检查网络连接（需要访问 Binance API）
2. 检查模型文件是否存在
3. 查看详细错误信息

