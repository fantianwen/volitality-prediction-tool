# AWS 部署 - 快速开始

## ⚡ 5 分钟部署

### 1. 准备文件

确保以下文件存在:
```bash
release/
├── aws_account.md          # AWS 配置
├── trading-bot.pem         # SSH 密钥 (chmod 400)
└── deploy_to_aws.sh        # 部署脚本
```

### 2. 配置 AWS 信息

编辑 `aws_account.md`:
```markdown
ip: 54.250.16.16
pem: trading-bot.pem
```

### 3. 部署

```bash
cd project/release
chmod +x deploy_to_aws.sh
./deploy_to_aws.sh
```

### 4. 配置 Telegram

```bash
ssh -i trading-bot.pem ubuntu@54.250.16.16
nano /home/ubuntu/btc-predictor/scripts/.env
```

填写:
```
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 5. 启动服务

```bash
sudo systemctl restart btc-predictor
sudo systemctl status btc-predictor
```

## 📋 常用命令

```bash
# 查看日志
ssh -i trading-bot.pem ubuntu@54.250.16.16 \
    'tail -f /home/ubuntu/btc-predictor/logs/prediction_server.log'

# 重启服务
ssh -i trading-bot.pem ubuntu@54.250.16.16 \
    'sudo systemctl restart btc-predictor'

# 查看状态
ssh -i trading-bot.pem ubuntu@54.250.16.16 \
    'sudo systemctl status btc-predictor'

# 更新代码
cd project/release
./deploy_to_aws.sh
```

## ✅ 验证部署

1. **服务运行**: `systemctl status btc-predictor` 显示 `active (running)`
2. **日志正常**: 每小时整点有预测记录
3. **Telegram 通知**: 收到预测报告

## 🆘 问题?

查看详细文档: `AWS_DEPLOYMENT.md`

