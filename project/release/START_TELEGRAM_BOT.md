# 启动 Telegram Bot 服务

## 📋 已同步的文件

✅ `prediction_server.py` - 更新后的预测服务器（包含预测价格和 UTC+8 时间）
✅ `telegram_bot.py` - Telegram Bot 命令处理器
✅ `test_telegram.py` - Telegram 配置测试脚本

## 🚀 启动方式

### 方法 1: 手动启动（测试用）

```bash
# SSH 到服务器
ssh -i trading-bot.pem ubuntu@54.250.16.16

# 进入项目目录
cd /home/ubuntu/btc-predictor/scripts

# 激活虚拟环境
source ../venv/bin/activate

# 启动 Telegram Bot
python3 telegram_bot.py --model ../models/regression_model_20251213_213205.pkl
```

### 方法 2: 后台运行

```bash
# SSH 到服务器
ssh -i trading-bot.pem ubuntu@54.250.16.16

# 进入项目目录
cd /home/ubuntu/btc-predictor/scripts
source ../venv/bin/activate

# 后台运行
nohup python3 telegram_bot.py \
    --model ../models/regression_model_20251213_213205.pkl \
    > ../logs/telegram_bot.log 2>&1 &

# 查看进程
ps aux | grep telegram_bot

# 查看日志
tail -f ../logs/telegram_bot.log
```

### 方法 3: 使用 Systemd 服务（推荐）

创建服务文件：

```bash
ssh -i trading-bot.pem ubuntu@54.250.16.16

sudo nano /etc/systemd/system/telegram-bot.service
```

添加以下内容：

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

## 📱 使用命令

启动 Bot 后，在 Telegram 中向你的机器人发送：

- `/start` - 显示欢迎信息
- `/predict-now` - 立即执行预测并发送结果
- `/help` - 显示帮助信息

## 🔍 验证运行

### 检查 Bot 是否运行

```bash
# 查看进程
ps aux | grep telegram_bot

# 查看日志
tail -f /home/ubuntu/btc-predictor/logs/telegram_bot.log

# 查看 Systemd 服务状态
sudo systemctl status telegram-bot
```

### 测试命令

在 Telegram 中发送 `/predict-now`，应该收到：
1. "⏳ 正在执行预测，请稍候..."
2. 完整的预测报告（包含预测价格和 UTC+8 时间）

## 🔧 故障排查

### Bot 无法启动

```bash
# 检查依赖
cd /home/ubuntu/btc-predictor
source venv/bin/activate
pip list | grep -E "pandas|numpy|scikit-learn"

# 检查配置
cat scripts/.env | grep TELEGRAM

# 测试导入
python3 -c "from telegram_bot import TelegramBot; print('OK')"
```

### 命令无响应

1. 检查 Bot 是否正在运行
2. 检查日志中的错误信息
3. 验证 Telegram Token 和 Chat ID 是否正确

### 预测失败

1. 检查预测服务器是否正常工作
2. 检查网络连接（需要访问 Binance API）
3. 查看错误日志

## 📊 同时运行两个服务

可以同时运行预测服务和 Telegram Bot：

```bash
# 服务 1: 每小时自动预测
sudo systemctl start btc-predictor
sudo systemctl enable btc-predictor

# 服务 2: Telegram Bot（监听命令）
sudo systemctl start telegram-bot
sudo systemctl enable telegram-bot
```

## 📝 日志位置

- Bot 日志: `/home/ubuntu/btc-predictor/logs/telegram_bot.log`
- 错误日志: `/home/ubuntu/btc-predictor/logs/telegram_bot.error.log`
- Systemd 日志: `sudo journalctl -u telegram-bot -f`

