#!/bin/bash
# BTC 价格预测服务 - AWS 部署脚本
#
# 使用方法:
#   ./deploy_to_aws.sh                    # 部署到 AWS
#   ./deploy_to_aws.sh --test             # 测试连接
#   ./deploy_to_aws.sh --restart          # 重启服务

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 读取 AWS 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_CONFIG_FILE="$SCRIPT_DIR/aws_account.md"

if [ ! -f "$AWS_CONFIG_FILE" ]; then
    echo -e "${RED}❌ 错误: 未找到 AWS 配置文件: $AWS_CONFIG_FILE${NC}"
    exit 1
fi

# 解析配置
AWS_IP=$(grep "^ip:" "$AWS_CONFIG_FILE" | awk '{print $2}')
AWS_PEM=$(grep "^pem:" "$AWS_CONFIG_FILE" | awk '{print $2}')

if [ -z "$AWS_IP" ] || [ -z "$AWS_PEM" ]; then
    echo -e "${RED}❌ 错误: AWS 配置不完整${NC}"
    echo "   请检查 $AWS_CONFIG_FILE"
    exit 1
fi

# 检查 PEM 文件
PEM_PATH="$SCRIPT_DIR/$AWS_PEM"
if [ ! -f "$PEM_PATH" ]; then
    echo -e "${YELLOW}⚠️  警告: PEM 文件不存在: $PEM_PATH${NC}"
    echo "   请确保 PEM 文件在 release 目录中"
    read -p "   是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 项目配置
PROJECT_NAME="btc-predictor"
REMOTE_USER="ubuntu"
REMOTE_DIR="/home/$REMOTE_USER/$PROJECT_NAME"
LOCAL_PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${GREEN}🚀 开始部署到 AWS${NC}"
echo "   IP: $AWS_IP"
echo "   PEM: $AWS_PEM"
echo "   远程目录: $REMOTE_DIR"
echo ""

# 测试连接
if [ "$1" == "--test" ]; then
    echo -e "${YELLOW}🔍 测试 SSH 连接...${NC}"
    ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no "$REMOTE_USER@$AWS_IP" "echo '✅ 连接成功'"
    exit 0
fi

# 检查远程 Python 版本
echo -e "${YELLOW}📋 检查远程环境...${NC}"
ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no -o BatchMode=yes "$REMOTE_USER@$AWS_IP" "python3 --version || echo '⚠️  Python3 未安装'; uname -a; df -h / | tail -1"

# 创建远程目录
echo -e "${YELLOW}📁 创建远程目录...${NC}"
ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no "$REMOTE_USER@$AWS_IP" "
    mkdir -p $REMOTE_DIR/{scripts,models,data,logs,web/templates}
"

# 同步项目文件
echo -e "${YELLOW}📦 同步项目文件...${NC}"
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='*.log' \
    --exclude='data/*.csv' \
    --exclude='models/classification_model_*.pkl' \
    --exclude='models/ensemble_model_*.pkl' \
    --exclude='models/*.keras' \
    --exclude='models/*_results_*.json' \
    --timeout=60 \
    -e "ssh -i $PEM_PATH -o StrictHostKeyChecking=no -o ServerAliveInterval=30" \
    "$LOCAL_PROJECT_ROOT/project/" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/"

# 同步模型文件（只同步需要的回归模型）
echo -e "${YELLOW}🤖 同步模型文件...${NC}"
rsync -avz --progress \
    --include='regression_model_20251213_213205.pkl' \
    --exclude='*.pkl' \
    --exclude='*.keras' \
    --exclude='*.json' \
    --timeout=60 \
    -e "ssh -i $PEM_PATH -o StrictHostKeyChecking=no -o ServerAliveInterval=30" \
    "$LOCAL_PROJECT_ROOT/project/models/" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/models/"

# 创建 .env 文件（如果不存在）
echo -e "${YELLOW}⚙️  配置环境变量...${NC}"
ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no -o BatchMode=yes "$REMOTE_USER@$AWS_IP" "
if [ ! -f $REMOTE_DIR/scripts/.env ]; then
    echo '📝 创建 .env 文件模板...'
    mkdir -p $REMOTE_DIR/scripts
    cat > $REMOTE_DIR/scripts/.env << 'ENVEOF'
# Telegram 配置
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# 交易对
SYMBOL=BTCUSDT

# 模型路径
MODEL_PATH=../models/regression_model_20251213_213205.pkl
ENVEOF
    echo '⚠️  请编辑 $REMOTE_DIR/scripts/.env 并填写 Telegram 配置'
else
    echo '✅ .env 文件已存在'
fi
"

# 安装依赖
echo -e "${YELLOW}📦 安装 Python 依赖...${NC}"
ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no -o BatchMode=yes "$REMOTE_USER@$AWS_IP" "
cd $REMOTE_DIR
# 创建虚拟环境（如果不存在）
if [ ! -d 'venv' ]; then
    echo '创建 Python 虚拟环境...'
    python3 -m venv venv
fi
# 激活虚拟环境并安装依赖
source venv/bin/activate
pip install --upgrade pip -q
pip install -r scripts/requirements.txt -q
# 安装 Web UI 依赖（如果存在）
if [ -f 'web/requirements.txt' ]; then
    pip install -r web/requirements.txt -q
fi
echo '✅ 依赖安装完成'
"

# 设置执行权限
echo -e "${YELLOW}🔧 设置执行权限...${NC}"
ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no "$REMOTE_USER@$AWS_IP" "
    chmod +x $REMOTE_DIR/scripts/deploy/*.sh
    if [ -f '$REMOTE_DIR/web/start.sh' ]; then
        chmod +x $REMOTE_DIR/web/start.sh
    fi
"

# 安装 systemd 服务
if [ "$1" != "--skip-service" ]; then
    echo -e "${YELLOW}⚙️  安装 Systemd 服务...${NC}"
    
    # 生成预测服务文件
    PREDICTION_SERVICE=$(cat << EOF
[Unit]
Description=BTC Price Prediction Service
After=network.target

[Service]
Type=simple
User=$REMOTE_USER
WorkingDirectory=$REMOTE_DIR/scripts

# Python 虚拟环境路径
Environment="PATH=$REMOTE_DIR/venv/bin:\$PATH"

# 加载环境变量
EnvironmentFile=$REMOTE_DIR/scripts/.env

# 启动命令
ExecStart=$REMOTE_DIR/venv/bin/python3 $REMOTE_DIR/scripts/prediction_server.py --model ../models/regression_model_20251213_213205.pkl

# 自动重启
Restart=always
RestartSec=10

# 日志
StandardOutput=append:$REMOTE_DIR/logs/prediction_server.log
StandardError=append:$REMOTE_DIR/logs/prediction_server.error.log

[Install]
WantedBy=multi-user.target
EOF
)
    
    # 生成 Web UI 服务文件
    WEB_SERVICE=$(cat << EOF
[Unit]
Description=BTC Price Prediction Web Dashboard
After=network.target

[Service]
Type=simple
User=$REMOTE_USER
WorkingDirectory=$REMOTE_DIR/web

# Python 虚拟环境路径
Environment="PATH=$REMOTE_DIR/venv/bin:\$PATH"
Environment="PORT=8080"

# 加载环境变量（如果存在）
EnvironmentFile=$REMOTE_DIR/scripts/.env

# 启动命令
ExecStart=$REMOTE_DIR/venv/bin/python3 $REMOTE_DIR/web/app.py --port 8080

# 自动重启
Restart=always
RestartSec=10

# 日志
StandardOutput=append:$REMOTE_DIR/logs/web_ui.log
StandardError=append:$REMOTE_DIR/logs/web_ui.error.log

[Install]
WantedBy=multi-user.target
EOF
)
    
    # 上传服务文件
    echo "$PREDICTION_SERVICE" | ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no "$REMOTE_USER@$AWS_IP" "sudo tee /etc/systemd/system/btc-predictor.service > /dev/null"
    echo "$WEB_SERVICE" | ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no "$REMOTE_USER@$AWS_IP" "sudo tee /etc/systemd/system/btc-predictor-web.service > /dev/null"
    
    # 重载 systemd
    ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no "$REMOTE_USER@$AWS_IP" "sudo systemctl daemon-reload"
    
    echo -e "${GREEN}✅ Systemd 服务已安装${NC}"
    echo ""
    echo "📊 预测服务管理:"
    echo "   sudo systemctl start btc-predictor"
    echo "   sudo systemctl stop btc-predictor"
    echo "   sudo systemctl status btc-predictor"
    echo "   sudo systemctl enable btc-predictor  # 开机自启"
    echo ""
    echo "🌐 Web UI 服务管理:"
    echo "   sudo systemctl start btc-predictor-web"
    echo "   sudo systemctl stop btc-predictor-web"
    echo "   sudo systemctl status btc-predictor-web"
    echo "   sudo systemctl enable btc-predictor-web  # 开机自启"
fi

# 重启服务
if [ "$1" == "--restart" ] || [ "$1" != "--skip-service" ]; then
    echo -e "${YELLOW}🔄 重启服务...${NC}"
    ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no -o BatchMode=yes "$REMOTE_USER@$AWS_IP" "
        sudo systemctl stop btc-predictor 2>/dev/null || true
        sudo systemctl stop btc-predictor-web 2>/dev/null || true
        sudo systemctl start btc-predictor
        sudo systemctl start btc-predictor-web
        sleep 2
        echo '📊 预测服务状态:'
        sudo systemctl status btc-predictor --no-pager | head -5
        echo ''
        echo '🌐 Web UI 服务状态:'
        sudo systemctl status btc-predictor-web --no-pager | head -5
    "
fi

echo ""
echo -e "${GREEN}✅ 部署完成!${NC}"
echo ""
echo "📋 后续步骤:"
echo "   1. 编辑远程 .env 文件:"
echo "      ssh -i $PEM_PATH $REMOTE_USER@$AWS_IP 'nano $REMOTE_DIR/scripts/.env'"
echo ""
echo "   2. 配置 AWS 安全组，开放端口 8080 (Web UI):"
echo "      - 进入 AWS EC2 控制台"
echo "      - 选择实例 -> 安全组"
echo "      - 添加入站规则: 类型=自定义TCP, 端口=8080, 来源=0.0.0.0/0 (或您的IP)"
echo ""
echo "   3. 重启服务:"
echo "      ssh -i $PEM_PATH $REMOTE_USER@$AWS_IP 'sudo systemctl restart btc-predictor btc-predictor-web'"
echo ""
echo "   4. 查看日志:"
echo "      ssh -i $PEM_PATH $REMOTE_USER@$AWS_IP 'tail -f $REMOTE_DIR/logs/prediction_server.log'"
echo "      ssh -i $PEM_PATH $REMOTE_USER@$AWS_IP 'tail -f $REMOTE_DIR/logs/web_ui.log'"
echo ""
echo "   5. 查看服务状态:"
echo "      ssh -i $PEM_PATH $REMOTE_USER@$AWS_IP 'sudo systemctl status btc-predictor btc-predictor-web'"
echo ""
echo "   6. 访问 Web UI:"
echo "      http://$AWS_IP:8080"
echo ""

