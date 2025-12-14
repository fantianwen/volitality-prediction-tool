# 修复 Telegram 404 错误

## 🔍 问题诊断

如果遇到 `HTTP Error 404: Not Found`，通常是因为：

1. **Chat ID 不正确** - 最常见原因
2. **Bot Token 格式错误**
3. **未向机器人发送过消息**

## 🛠️ 解决步骤

### 方法 1: 使用测试脚本（推荐）

在服务器上运行测试脚本：

```bash
# SSH 到服务器
ssh -i trading-bot.pem ubuntu@54.250.16.16

# 进入项目目录
cd /home/ubuntu/btc-predictor/scripts

# 激活虚拟环境
source ../venv/bin/activate

# 运行测试脚本
python3 test_telegram.py
```

### 方法 2: 手动验证配置

#### 步骤 1: 验证 Bot Token

```bash
# 替换 YOUR_TOKEN 为实际的 Bot Token
curl "https://api.telegram.org/botYOUR_TOKEN/getMe"
```

应该返回类似：
```json
{
  "ok": true,
  "result": {
    "id": 123456789,
    "is_bot": true,
    "first_name": "Your Bot",
    "username": "your_bot"
  }
}
```

#### 步骤 2: 获取正确的 Chat ID

```bash
# 1. 向你的机器人发送任意消息（在 Telegram 中）

# 2. 获取更新
curl "https://api.telegram.org/botYOUR_TOKEN/getUpdates"
```

在返回的 JSON 中查找：
```json
{
  "message": {
    "chat": {
      "id": 8157443482,  // 这就是你的 Chat ID
      ...
    }
  }
}
```

#### 步骤 3: 更新 .env 文件

```bash
# 编辑 .env 文件
nano /home/ubuntu/btc-predictor/scripts/.env
```

确保配置正确：
```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=8157443482
```

**注意**：
- Bot Token 格式：`数字:字母数字组合`
- Chat ID 必须是纯数字
- 不要有引号或空格

#### 步骤 4: 测试发送消息

```bash
# 在服务器上测试
cd /home/ubuntu/btc-predictor/scripts
source ../venv/bin/activate
python3 test_telegram.py
```

### 方法 3: 从本地测试

```bash
# 在本地项目目录
cd project/scripts
python3 test_telegram.py
```

## 🔧 常见错误及解决方法

### 错误 1: 404 Not Found

**原因**: Chat ID 不正确或未向机器人发送过消息

**解决**:
1. 向机器人发送消息
2. 使用 `getUpdates` API 获取正确的 Chat ID
3. 更新 `.env` 文件

### 错误 2: 401 Unauthorized

**原因**: Bot Token 无效或已过期

**解决**:
1. 检查 Bot Token 是否正确复制
2. 在 @BotFather 中重新生成 Token
3. 更新 `.env` 文件

### 错误 3: 400 Bad Request

**原因**: Chat ID 格式不正确

**解决**:
1. 确保 Chat ID 是纯数字
2. 不要有引号或空格
3. 确保是私聊的 Chat ID（不是群组 ID）

## 📋 检查清单

- [ ] Bot Token 格式正确（`数字:字母数字`）
- [ ] Chat ID 是纯数字
- [ ] 已向机器人发送过至少一条消息
- [ ] `.env` 文件中没有引号
- [ ] `.env` 文件中没有多余空格
- [ ] 使用 `test_telegram.py` 测试通过

## 🚀 验证修复

修复后，重启服务并检查日志：

```bash
# 重启服务
sudo systemctl restart btc-predictor

# 查看日志
tail -f /home/ubuntu/btc-predictor/logs/prediction_server.log
```

应该看到：
```
✅ Telegram 消息发送成功
```

而不是：
```
❌ Telegram HTTP 错误 404: Not Found
```

