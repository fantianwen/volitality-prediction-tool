# AWS 部署指南

## 📋 前置要求

1. **AWS EC2 实例**
   - Ubuntu 20.04+ 或 Amazon Linux 2
   - 至少 2GB RAM
   - 已配置安全组允许 SSH (端口 22)

2. **本地环境**
   - SSH 客户端
   - rsync (用于文件同步)
   - AWS PEM 密钥文件

3. **配置文件**
   - `aws_account.md` - 包含 IP 和 PEM 文件名
   - PEM 文件放在 `release/` 目录

## 🚀 快速部署

### 1. 配置 AWS 信息

编辑 `aws_account.md`:
```markdown
ip: 54.250.16.16
pem: trading-bot.pem
```

确保 PEM 文件在 `release/` 目录中，并设置正确的权限:
```bash
chmod 400 release/trading-bot.pem
```

### 2. 运行部署脚本

```bash
cd project/release
chmod +x deploy_to_aws.sh

# 测试连接
./deploy_to_aws.sh --test

# 完整部署
./deploy_to_aws.sh

# 仅重启服务（不重新部署）
./deploy_to_aws.sh --restart
```

### 3. 配置环境变量

SSH 到服务器并编辑 `.env` 文件:
```bash
ssh -i release/trading-bot.pem ubuntu@54.250.16.16
nano /home/ubuntu/btc-predictor/scripts/.env
```

填写 Telegram 配置:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
SYMBOL=BTCUSDT
MODEL_PATH=../models/regression_model_20251213_213205.pkl
```

### 4. 启动服务

```bash
# 启动服务
sudo systemctl start btc-predictor

# 设置开机自启
sudo systemctl enable btc-predictor

# 查看状态
sudo systemctl status btc-predictor
```

## 📁 远程目录结构

```
/home/ubuntu/btc-predictor/
├── scripts/
│   ├── prediction_server.py
│   ├── data_collector.py
│   ├── train_model.py
│   ├── requirements.txt
│   ├── .env                    # 需要手动配置
│   └── deploy/
├── web/
│   ├── app.py                  # Web UI 服务器
│   ├── templates/
│   │   └── index.html          # Web 界面
│   ├── requirements.txt
│   └── start.sh
├── models/
│   └── regression_model_*.pkl
├── data/
└── logs/
    ├── prediction_server.log
    ├── prediction_server.error.log
    ├── web_ui.log
    └── web_ui.error.log
```

## 🌐 Web UI 部署

Web UI 提供可视化的预测仪表板。详细部署指南请参考 [WEB_UI_DEPLOYMENT.md](WEB_UI_DEPLOYMENT.md)

### 快速启动 Web UI

```bash
# 部署脚本会自动安装和启动 Web UI
./deploy_to_aws.sh

# 手动启动
ssh -i release/trading-bot.pem ubuntu@YOUR_AWS_IP
sudo systemctl start btc-predictor-web
sudo systemctl enable btc-predictor-web
```

### 访问 Web UI

1. **配置 AWS 安全组**: 开放端口 8080
2. **访问地址**: `http://YOUR_AWS_IP:8080`

详细说明请参考 [WEB_UI_DEPLOYMENT.md](WEB_UI_DEPLOYMENT.md)

## 🔧 服务管理

### 查看服务状态
```bash
ssh -i release/trading-bot.pem ubuntu@54.250.16.16

# 预测服务
sudo systemctl status btc-predictor

# Web UI 服务
sudo systemctl status btc-predictor-web
```

### 查看日志
```bash
ssh -i release/trading-bot.pem ubuntu@54.250.16.16

# 预测服务日志
tail -f /home/ubuntu/btc-predictor/logs/prediction_server.log
tail -f /home/ubuntu/btc-predictor/logs/prediction_server.error.log

# Web UI 日志
tail -f /home/ubuntu/btc-predictor/logs/web_ui.log
tail -f /home/ubuntu/btc-predictor/logs/web_ui.error.log

# Systemd 日志
sudo journalctl -u btc-predictor -f
sudo journalctl -u btc-predictor-web -f
```

### 重启服务
```bash
ssh -i release/trading-bot.pem ubuntu@54.250.16.16

# 重启所有服务
sudo systemctl restart btc-predictor btc-predictor-web

# 或分别重启
sudo systemctl restart btc-predictor
sudo systemctl restart btc-predictor-web
```

### 停止服务
```bash
sudo systemctl stop btc-predictor
```

### 禁用开机自启
```bash
sudo systemctl disable btc-predictor
```

## 🔄 更新部署

### 更新代码
```bash
cd project/release
./deploy_to_aws.sh
```

### 仅更新模型
```bash
rsync -avz -e "ssh -i release/trading-bot.pem" \
    project/models/ \
    ubuntu@54.250.16.16:/home/ubuntu/btc-predictor/models/
```

### 重启服务
```bash
ssh -i release/trading-bot.pem ubuntu@54.250.16.16 \
    'sudo systemctl restart btc-predictor'
```

## 🐛 故障排查

### 1. 服务无法启动

检查日志:
```bash
sudo journalctl -u btc-predictor -n 50
```

常见问题:
- Python 依赖未安装 → 运行 `pip3 install -r requirements.txt`
- 模型文件不存在 → 检查 `MODEL_PATH` 配置
- 环境变量未设置 → 检查 `.env` 文件

### 2. 无法连接 Binance API

检查网络连接:
```bash
curl https://fapi.binance.com/fapi/v1/ping
```

如果失败，检查:
- 安全组是否允许出站 HTTPS (443)
- 服务器时间是否正确 (`date`)

### 3. Telegram 通知不工作

测试 Telegram API:
```bash
python3 -c "
import os
import urllib.request
token = 'YOUR_TOKEN'
chat_id = 'YOUR_CHAT_ID'
url = f'https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text=Test'
urllib.request.urlopen(url)
"
```

### 4. 权限问题

确保用户有权限:
```bash
sudo chown -R ubuntu:ubuntu /home/ubuntu/btc-predictor
chmod +x /home/ubuntu/btc-predictor/scripts/*.py
```

## 📊 监控

### 查看服务运行时间
```bash
systemctl status btc-predictor | grep Active
```

### 查看资源使用
```bash
# CPU 和内存
top -p $(pgrep -f prediction_server.py)

# 磁盘空间
df -h
du -sh /home/ubuntu/btc-predictor
```

### 查看预测历史
```bash
tail -100 /home/ubuntu/btc-predictor/logs/prediction_server.log | grep "预测完成"
```

## 🔒 安全建议

1. **PEM 文件安全**
   ```bash
   chmod 400 release/trading-bot.pem
   # 不要将 PEM 文件提交到 Git
   ```

2. **环境变量安全**
   - 不要将 `.env` 文件提交到 Git
   - 使用 AWS Systems Manager Parameter Store 存储敏感信息

3. **防火墙配置**
   - 只开放必要的端口 (SSH 22)
   - 使用安全组限制访问

4. **定期更新**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

## 📝 检查清单

部署前:
- [ ] AWS EC2 实例已创建并运行
- [ ] 安全组已配置 (SSH 22)
- [ ] PEM 文件已下载并设置权限
- [ ] `aws_account.md` 已配置
- [ ] 模型文件已训练并可用

部署后:
- [ ] 预测服务已启动 (`systemctl status btc-predictor`)
- [ ] Web UI 服务已启动 (`systemctl status btc-predictor-web`)
- [ ] 日志正常 (`tail -f logs/prediction_server.log`)
- [ ] Web UI 可访问 (`http://YOUR_IP:8080`)
- [ ] Telegram 通知正常
- [ ] 预测任务按时执行
- [ ] 开机自启已配置
- [ ] AWS 安全组已开放端口 8080

## 🆘 获取帮助

如果遇到问题:
1. 查看日志文件
2. 检查服务状态
3. 验证网络连接
4. 确认配置文件正确

