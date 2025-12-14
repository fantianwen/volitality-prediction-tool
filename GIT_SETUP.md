# Git 仓库设置指南

## ✅ 已配置的 .gitignore

以下文件和目录已被忽略，不会提交到 Git：

### 敏感文件
- `.env` - 环境变量（包含 Telegram Token 等）
- `*.pem`, `*.key` - SSH 密钥文件
- `project/release/aws_account.md` - AWS 服务器信息

### 日志和临时文件
- `*.log` - 所有日志文件
- `__pycache__/` - Python 缓存
- `*.pyc`, `*.pyo` - Python 编译文件

### 虚拟环境
- `venv/`, `env/`, `ENV/` - Python 虚拟环境

### 数据文件（可选）
- `project/data/*.csv` - 历史数据文件（可能很大）
- `project/models/*_results_*.json` - 训练结果文件

### IDE 和系统文件
- `.vscode/`, `.idea/` - IDE 配置
- `.DS_Store` - macOS 系统文件

## 📋 准备提交到 GitHub

### 1. 检查敏感文件

```bash
# 确认敏感文件被忽略
git check-ignore project/scripts/.env
git check-ignore project/release/aws_account.md
git check-ignore project/release/*.pem
```

### 2. 查看将要提交的文件

```bash
git status
```

### 3. 提交代码

```bash
# 添加所有文件（.gitignore 会自动排除敏感文件）
git add .

# 查看将要提交的文件
git status

# 提交
git commit -m "Initial commit: BTC price prediction system

- Multi-timeframe technical analysis
- GBM model for price prediction
- Telegram bot integration
- Position management system
- AWS deployment scripts"

# 添加远程仓库（替换为你的 GitHub 仓库 URL）
git remote add origin https://github.com/yourusername/volitality-prediction-tool.git

# 推送到 GitHub
git push -u origin main
```

## ⚠️ 提交前检查清单

- [ ] `.env` 文件不在暂存区
- [ ] `*.pem` 文件不在暂存区
- [ ] `aws_account.md` 不在暂存区
- [ ] `*.log` 文件不在暂存区
- [ ] 虚拟环境目录不在暂存区
- [ ] 敏感信息已从代码中移除

## 🔒 安全建议

1. **不要提交敏感信息**:
   - Telegram Bot Token
   - Chat ID
   - AWS 服务器 IP
   - SSH 密钥

2. **使用环境变量**:
   - 所有敏感配置通过 `.env` 文件管理
   - `.env` 文件已在 `.gitignore` 中

3. **模型文件**:
   - 如果模型文件很大，考虑使用 Git LFS
   - 或单独存储模型文件

4. **数据文件**:
   - 历史数据文件不提交（已在 `.gitignore` 中）
   - 如果需要示例数据，使用小样本

## 📝 提交信息模板

```
feat: Add position management system

- Add position_manager.py for position sizing
- Integrate position recommendations in reports
- Add risk level configuration (conservative/moderate/aggressive)
- Update prediction frequency to 30 minutes
```

## 🔄 更新远程仓库

```bash
# 拉取最新更改
git pull origin main

# 推送本地更改
git push origin main
```

