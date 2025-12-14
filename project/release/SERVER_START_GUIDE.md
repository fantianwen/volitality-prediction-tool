# AWS 服务器启动指南

## 🚀 快速启动

### 方法 1: 使用 Systemd 服务（推荐）

```bash
# SSH 连接到服务器
ssh -i trading-bot.pem ubuntu@54.250.16.16

# 启动服务
sudo systemctl start btc-predictor

# 设置开机自启
sudo systemctl enable btc-predictor

# 查看状态
sudo systemctl status btc-predictor

# 查看日志
tail -f /home/ubuntu/btc-predictor/logs/prediction_server.log
```

### 方法 2: 手动启动（测试用）

```bash
# SSH 连接到服务器
ssh -i trading-bot.pem ubuntu@54.250.16.16

# 进入项目目录
cd /home/ubuntu/btc-predictor/scripts

# 激活虚拟环境
source ../venv/bin/activate

# 测试运行
python3 prediction_server.py --model ../models/regression_model_20251213_213205.pkl --test

# 正式运行（前台）
python3 prediction_server.py --model ../models/regression_model_20251213_213205.pkl

# 后台运行
nohup python3 prediction_server.py --model ../models/regression_model_20251213_213205.pkl > ../logs/prediction_server.log 2>&1 &
```

## 📋 启动前检查清单

### 1. 检查环境变量配置

```bash
# 查看 .env 文件
cat /home/ubuntu/btc-predictor/scripts/.env

# 编辑 .env 文件（如果需要）
nano /home/ubuntu/btc-predictor/scripts/.env
```

确保以下配置正确：
- `TELEGRAM_BOT_TOKEN` - Telegram Bot Token
- `TELEGRAM_CHAT_ID` - Telegram Chat ID
- `MODEL_PATH` - 模型文件路径

### 2. 检查模型文件

```bash
ls -lh /home/ubuntu/btc-predictor/models/regression_model_20251213_213205.pkl
```

### 3. 检查 Python 依赖

```bash
cd /home/ubuntu/btc-predictor
source venv/bin/activate
pip list | grep -E "pandas|numpy|scikit-learn|schedule"
```

### 4. 检查网络连接

```bash
# 测试 Binance API
curl https://fapi.binance.com/fapi/v1/ping

# 测试 Telegram API
curl https://api.telegram.org/bot<YOUR_TOKEN>/getMe
```

## 🔧 服务管理命令

### 启动服务
```bash
sudo systemctl start btc-predictor
```

### 停止服务
```bash
sudo systemctl stop btc-predictor
```

### 重启服务
```bash
sudo systemctl restart btc-predictor
```

### 查看服务状态
```bash
sudo systemctl status btc-predictor
```

### 启用开机自启
```bash
sudo systemctl enable btc-predictor
```

### 禁用开机自启
```bash
sudo systemctl disable btc-predictor
```

### 查看服务日志
```bash
# Systemd 日志
sudo journalctl -u btc-predictor -f

# 应用日志
tail -f /home/ubuntu/btc-predictor/logs/prediction_server.log

# 错误日志
tail -f /home/ubuntu/btc-predictor/logs/prediction_server.error.log
```

## 🐛 故障排查

### 问题 1: 服务无法启动

```bash
# 查看详细错误
sudo journalctl -u btc-predictor -n 50 --no-pager

# 检查 Python 路径
which python3

# 检查虚拟环境
ls -la /home/ubuntu/btc-predictor/venv/bin/python3
```

### 问题 2: 依赖缺失

```bash
cd /home/ubuntu/btc-predictor
source venv/bin/activate
pip install -r scripts/requirements.txt
```

### 问题 3: 权限问题

```bash
# 检查文件权限
ls -la /home/ubuntu/btc-predictor/scripts/prediction_server.py

# 修复权限
chmod +x /home/ubuntu/btc-predictor/scripts/prediction_server.py
chown -R ubuntu:ubuntu /home/ubuntu/btc-predictor
```

### 问题 4: 环境变量未加载

```bash
# 检查 .env 文件
cat /home/ubuntu/btc-predictor/scripts/.env

# 手动测试加载
cd /home/ubuntu/btc-predictor/scripts
source ../venv/bin/activate
python3 -c "from config import config; print(config.telegram_enabled)"
```

## 📊 验证服务运行

### 1. 检查服务状态
```bash
sudo systemctl status btc-predictor
```
应该显示 `Active: active (running)`

### 2. 检查日志
```bash
tail -20 /home/ubuntu/btc-predictor/logs/prediction_server.log
```
应该看到预测记录

### 3. 检查 Telegram 通知
查看 Telegram 是否收到预测报告

### 4. 检查进程
```bash
ps aux | grep prediction_server
```

## 🔄 从本地管理服务器

### 启动服务
```bash
cd project/release
ssh -i trading-bot.pem ubuntu@54.250.16.16 "sudo systemctl start btc-predictor"
```

### 查看状态
```bash
ssh -i trading-bot.pem ubuntu@54.250.16.16 "sudo systemctl status btc-predictor"
```

### 查看日志
```bash
ssh -i trading-bot.pem ubuntu@54.250.16.16 "tail -f /home/ubuntu/btc-predictor/logs/prediction_server.log"
```

### 重启服务
```bash
ssh -i trading-bot.pem ubuntu@54.250.16.16 "sudo systemctl restart btc-predictor"
```

## 📝 完整启动流程示例

```bash
# 1. SSH 连接
ssh -i trading-bot.pem ubuntu@54.250.16.16

# 2. 检查配置
cd /home/ubuntu/btc-predictor/scripts
cat .env

# 3. 测试运行
source ../venv/bin/activate
python3 prediction_server.py --model ../models/regression_model_20251213_213205.pkl --test

# 4. 如果测试成功，启动服务
sudo systemctl start btc-predictor
sudo systemctl enable btc-predictor

# 5. 查看状态
sudo systemctl status btc-predictor

# 6. 查看日志
tail -f /home/ubuntu/btc-predictor/logs/prediction_server.log
```

## ⚠️ 注意事项

1. **首次启动前**：确保 `.env` 文件已正确配置 Telegram Token 和 Chat ID
2. **虚拟环境**：如果使用 Systemd 服务，确保服务文件中的 Python 路径指向虚拟环境
3. **网络连接**：确保服务器可以访问 Binance API 和 Telegram API
4. **日志轮转**：定期清理日志文件，避免占用过多磁盘空间

