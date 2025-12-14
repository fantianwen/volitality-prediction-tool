# Web Dashboard 快速开始

## 🚀 快速启动

### 方法 1: 使用启动脚本（推荐）

```bash
cd project/web
./start.sh
```

### 方法 2: 手动启动

```bash
cd project/web

# 安装依赖（首次运行）
pip install -r requirements.txt
pip install -r ../scripts/requirements.txt

# 启动服务器
python3 app.py
```

## 📊 访问 Dashboard

打开浏览器访问：**http://localhost:8080** (默认端口)

## ⚙️ 配置

### 设置模型路径（可选）

```bash
export MODEL_PATH=../models/regression_model_20251213_213205.pkl
python3 app.py
```

### 更改端口（可选）

**方法 1: 使用环境变量**
```bash
PORT=3000 python3 app.py
```

**方法 2: 使用命令行参数**
```bash
python3 app.py --port 3000
```

**方法 3: 使用启动脚本**
```bash
PORT=3000 ./start.sh
```

## 🎯 功能说明

### 1. 价格预测卡片
- 显示当前 BTC 价格
- 显示预测价格（20小时后）
- 显示价格变化百分比

### 2. 预测结果卡片
- 方向：看涨/看跌/横盘
- 价格区间
- 预测时间（UTC+8）

### 3. 置信度卡片
- 置信度百分比（0-100%）
- 信号强度（-12 到 +12）
- 信号类型（看涨/看跌/中性）

### 4. 仓位建议卡片
- 建议仓位大小（%）
- 建议杠杆倍数
- 风险评分
- 交易建议原因

### 5. 市场状态卡片
- RSI (1小时)
- ADX (1小时)
- 波动率 (1小时)

### 6. 预测历史卡片
- 最近50次预测记录
- 时间、方向、置信度
- 信号强度

## 🔄 自动刷新

点击页面上的 "Auto Refresh (30s)" 开关，系统会每30秒自动执行一次预测并更新界面。

## 🐛 常见问题

### 1. 模型文件未找到

**错误**: `Warning: Model file not found`

**解决**: 
```bash
# 检查模型文件是否存在
ls -lh ../models/regression_model_*.pkl

# 设置正确的模型路径
export MODEL_PATH=../models/your_model.pkl
```

### 2. 端口被占用

**错误**: `Address already in use`

**解决**:
```bash
# 使用其他端口（方法1：环境变量）
PORT=3000 python3 app.py

# 使用其他端口（方法2：命令行参数）
python3 app.py --port 3000

# 使用其他端口（方法3：启动脚本）
PORT=3000 ./start.sh
```

### 3. 依赖缺失

**错误**: `ModuleNotFoundError: No module named 'flask'`

**解决**:
```bash
pip install -r requirements.txt
pip install -r ../scripts/requirements.txt
```

### 4. 预测失败

**错误**: `Prediction failed`

**可能原因**:
- Binance API 连接问题
- 模型文件损坏
- 特征提取失败

**解决**: 检查网络连接和模型文件完整性

## 📝 注意事项

- 首次预测可能需要 3-5 秒
- 所有时间显示为 UTC+8（北京时间）
- 仓位建议仅供参考，不构成投资建议
- 自动刷新间隔：30秒

## 🔗 相关文档

- [完整 README](README.md)
- [预测服务器文档](../scripts/README_PREDICTION_SERVER.md)
- [仓位管理指南](../scripts/POSITION_MANAGEMENT_GUIDE.md)

