# Web UI 后台运行指南

## 🎯 推荐方法：使用 Systemd 服务（生产环境）

这是最推荐的方法，适合生产环境，支持自动重启和日志管理。

### 启动服务

```bash
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

### 设置开机自启

```bash
sudo systemctl enable btc-predictor-web
```

### 禁用开机自启

```bash
sudo systemctl disable btc-predictor-web
```

### 重启服务

```bash
sudo systemctl restart btc-predictor-web
```

---

## 🔧 其他后台运行方法

### 方法 1: 使用 nohup（临时运行）

```bash
cd /home/ubuntu/btc-predictor/web
source ../venv/bin/activate
nohup python3 app.py --port 8080 > ../logs/web_ui_nohup.log 2>&1 &
```

查看进程：
```bash
ps aux | grep "app.py"
```

停止进程：
```bash
# 找到进程 ID
ps aux | grep "app.py" | grep -v grep

# 停止进程（替换 PID 为实际进程 ID）
kill <PID>
```

---

### 方法 2: 使用 screen（适合调试）

```bash
# 安装 screen（如果未安装）
sudo apt-get install screen -y

# 创建新的 screen 会话
screen -S webui

# 在 screen 中启动 Web UI
cd /home/ubuntu/btc-predictor/web
source ../venv/bin/activate
python3 app.py --port 8080

# 按 Ctrl+A 然后按 D 来分离会话（后台运行）

# 重新连接到会话
screen -r webui

# 列出所有会话
screen -ls

# 终止会话
screen -X -S webui quit
```

---

### 方法 3: 使用 tmux（推荐用于调试）

```bash
# 安装 tmux（如果未安装）
sudo apt-get install tmux -y

# 创建新的 tmux 会话
tmux new -s webui

# 在 tmux 中启动 Web UI
cd /home/ubuntu/btc-predictor/web
source ../venv/bin/activate
python3 app.py --port 8080

# 按 Ctrl+B 然后按 D 来分离会话（后台运行）

# 重新连接到会话
tmux attach -t webui

# 列出所有会话
tmux ls

# 终止会话
tmux kill-session -t webui
```

---

### 方法 4: 使用 & 符号（简单后台运行）

```bash
cd /home/ubuntu/btc-predictor/web
source ../venv/bin/activate
python3 app.py --port 8080 > ../logs/web_ui.log 2>&1 &
```

查看后台任务：
```bash
jobs
```

将后台任务转到前台：
```bash
fg %1  # 1 是任务编号
```

---

## 📊 检查 Web UI 是否运行

### 检查端口是否监听

```bash
sudo netstat -tlnp | grep 8080
# 或
sudo ss -tlnp | grep 8080
```

### 测试 Web UI

```bash
curl http://localhost:8080/api/status
```

### 检查进程

```bash
ps aux | grep "app.py" | grep -v grep
```

---

## 🐛 常见问题

### 1. 端口已被占用

```bash
# 查找占用端口的进程
sudo lsof -i :8080
# 或
sudo netstat -tlnp | grep 8080

# 停止占用端口的进程
sudo kill <PID>
```

### 2. 服务无法启动

```bash
# 查看详细错误
sudo journalctl -u btc-predictor-web -n 50

# 手动测试启动
cd /home/ubuntu/btc-predictor/web
source ../venv/bin/activate
python3 app.py --port 8080
```

### 3. 权限问题

```bash
# 确保文件有执行权限
chmod +x /home/ubuntu/btc-predictor/web/app.py
chmod +x /home/ubuntu/btc-predictor/web/start.sh
```

---

## 📝 推荐配置

### 生产环境
✅ **使用 Systemd 服务** - 自动重启、日志管理、开机自启

### 开发/调试环境
✅ **使用 tmux 或 screen** - 方便查看实时输出和调试

### 临时测试
✅ **使用 nohup** - 简单快速

---

## 🔄 完整示例（Systemd）

```bash
# 1. 启动服务
sudo systemctl start btc-predictor-web

# 2. 检查状态
sudo systemctl status btc-predictor-web

# 3. 查看日志
tail -f /home/ubuntu/btc-predictor/logs/web_ui.log

# 4. 测试访问
curl http://localhost:8080/api/status

# 5. 设置开机自启
sudo systemctl enable btc-predictor-web
```

---

## 📚 相关文档

- [Web UI 部署指南](../release/WEB_UI_DEPLOYMENT.md)
- [Web UI README](README.md)
- [快速开始](QUICKSTART.md)

