# Web UI AWS 部署指南

## 📋 概述

Web UI 部署在 AWS EC2 上，提供可视化的 BTC 价格预测仪表板。

## 🚀 快速部署

### 1. 使用部署脚本（推荐）

```bash
cd project/release
./deploy_to_aws.sh
```

部署脚本会自动：
- 同步 Web UI 代码到服务器
- 安装 Flask 和 Web UI 依赖
- 创建 systemd 服务
- 启动 Web UI 服务

### 2. 配置 AWS 安全组

**重要**: 需要开放端口 8080 以访问 Web UI

1. 登录 AWS EC2 控制台
2. 选择您的实例
3. 点击"安全"标签 -> 安全组
4. 编辑入站规则
5. 添加规则：
   - **类型**: 自定义 TCP
   - **端口**: 8080
   - **来源**: 
     - `0.0.0.0/0` (允许所有IP，仅用于测试)
     - 或您的特定 IP 地址（推荐用于生产环境）

### 3. 访问 Web UI

部署完成后，在浏览器中访问：

```
http://YOUR_AWS_IP:8080
```

例如：
```
http://54.250.16.16:8080
```

## 🔧 服务管理

### 启动服务

```bash
ssh -i release/trading-bot.pem ubuntu@YOUR_AWS_IP
sudo systemctl start btc-predictor-web
```

### 停止服务

```bash
sudo systemctl stop btc-predictor-web
```

### 查看状态

```bash
sudo systemctl status btc-predictor-web
```

### 查看日志

```bash
# 实时日志
tail -f /home/ubuntu/btc-predictor/logs/web_ui.log

# 错误日志
tail -f /home/ubuntu/btc-predictor/logs/web_ui.error.log

# Systemd 日志
sudo journalctl -u btc-predictor-web -f
```

### 重启服务

```bash
sudo systemctl restart btc-predictor-web
```

### 设置开机自启

```bash
sudo systemctl enable btc-predictor-web
```

## ⚙️ 配置

### 更改端口

编辑 systemd 服务文件：

```bash
sudo nano /etc/systemd/system/btc-predictor-web.service
```

修改 `Environment="PORT=8080"` 为所需端口，然后：

```bash
sudo systemctl daemon-reload
sudo systemctl restart btc-predictor-web
```

### 更改绑定地址

默认绑定到 `0.0.0.0`（所有接口）。如果只想本地访问：

```bash
sudo nano /etc/systemd/system/btc-predictor-web.service
```

修改 `ExecStart` 行，添加 `--host 127.0.0.1`：

```
ExecStart=/home/ubuntu/btc-predictor/venv/bin/python3 /home/ubuntu/btc-predictor/web/app.py --port 8080 --host 127.0.0.1
```

然后重启服务。

## 🐛 故障排查

### 1. 无法访问 Web UI

**检查服务状态**:
```bash
sudo systemctl status btc-predictor-web
```

**检查端口是否监听**:
```bash
sudo netstat -tlnp | grep 8080
# 或
sudo ss -tlnp | grep 8080
```

**检查防火墙**:
```bash
# Ubuntu/Debian
sudo ufw status
sudo ufw allow 8080/tcp

# CentOS/RHEL
sudo firewall-cmd --list-ports
sudo firewall-cmd --add-port=8080/tcp --permanent
sudo firewall-cmd --reload
```

**检查 AWS 安全组**:
- 确保安全组规则允许端口 8080 的入站流量

### 2. 服务无法启动

**查看错误日志**:
```bash
sudo journalctl -u btc-predictor-web -n 50
tail -50 /home/ubuntu/btc-predictor/logs/web_ui.error.log
```

**常见问题**:
- Flask 未安装 → `pip install flask flask-cors`
- 端口被占用 → 更改端口或停止占用端口的进程
- 权限问题 → 检查文件权限和用户

### 3. 预测功能不工作

**检查预测服务**:
```bash
sudo systemctl status btc-predictor
```

Web UI 依赖预测服务正常运行。确保：
- 预测服务已启动
- 模型文件存在
- `.env` 文件配置正确

### 4. 依赖缺失

**重新安装依赖**:
```bash
cd /home/ubuntu/btc-predictor
source venv/bin/activate
pip install -r web/requirements.txt
pip install -r scripts/requirements.txt
```

## 📊 监控

### 查看服务运行时间

```bash
systemctl status btc-predictor-web | grep Active
```

### 查看资源使用

```bash
# CPU 和内存
top -p $(pgrep -f "app.py")

# 磁盘空间
df -h
du -sh /home/ubuntu/btc-predictor
```

### 查看访问日志

```bash
tail -f /home/ubuntu/btc-predictor/logs/web_ui.log | grep "GET /"
```

## 🔒 安全建议

### 1. 使用 HTTPS（推荐）

使用 Nginx 作为反向代理，配置 SSL 证书：

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 2. 限制访问 IP

在 AWS 安全组中，只允许特定 IP 访问端口 8080。

### 3. 使用防火墙

```bash
# 只允许特定 IP
sudo ufw allow from YOUR_IP to any port 8080
```

### 4. 更改默认端口

使用非标准端口（如 9000）可以减少扫描攻击。

## 📝 检查清单

部署前:
- [ ] AWS EC2 实例运行中
- [ ] 安全组已配置端口 8080
- [ ] PEM 文件权限正确 (chmod 400)
- [ ] 预测服务已部署并运行

部署后:
- [ ] Web UI 服务已启动
- [ ] 可以访问 http://YOUR_IP:8080
- [ ] 预测功能正常工作
- [ ] 日志正常输出
- [ ] 开机自启已配置

## 🆘 获取帮助

如果遇到问题:
1. 查看服务状态: `sudo systemctl status btc-predictor-web`
2. 查看日志: `tail -f /home/ubuntu/btc-predictor/logs/web_ui.error.log`
3. 检查网络连接: `curl http://localhost:8080/api/status`
4. 验证依赖: `pip list | grep -E "flask|flask-cors"`

