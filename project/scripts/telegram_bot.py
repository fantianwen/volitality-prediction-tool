#!/usr/bin/env python3
"""
Telegram Bot 命令处理器

支持命令:
- /predict-now: 立即执行预测并发送结果
"""

import os
import sys
import json
import time
import ssl
import urllib.request
import threading
from pathlib import Path
from datetime import datetime

# 加载 .env
try:
    from config import load_dotenv
    load_dotenv()
except ImportError:
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    os.environ.setdefault(key, value)

# 导入预测服务器
from prediction_server import PredictionServer, TelegramNotifier

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram Bot 命令处理器"""
    
    def __init__(self, bot_token: str, chat_id: str, prediction_server: PredictionServer):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.prediction_server = prediction_server
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # SSL 配置
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # 命令处理映射（支持多种格式）
        self.commands = {
            'predict-now': self.handle_predict_now,
            'predict_now': self.handle_predict_now,  # 支持下划线格式
            'start': self.handle_start,
            'help': self.handle_help,
        }
    
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """发送消息到 Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            
            json_data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=json_data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=30) as response:
                result = json.loads(response.read().decode())
                return result.get('ok', False)
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False
    
    def handle_start(self, message: dict):
        """处理 /start 命令"""
        help_text = """
<b>🤖 BTC 价格预测 Bot</b>

<b>可用命令:</b>
/predict-now - 立即执行预测并发送结果
/help - 显示帮助信息

<b>自动预测:</b>
服务每小时整点自动执行预测并发送报告。
"""
        self.send_message(help_text)
    
    def handle_help(self, message: dict):
        """处理 /help 命令"""
        help_text = """
<b>📋 命令说明</b>

<b>/predict-now</b>
立即执行一次价格预测并发送详细报告到 Telegram。

<b>/help</b>
显示此帮助信息。

<b>自动预测</b>
服务会在每小时整点自动执行预测并发送报告。
"""
        self.send_message(help_text)
    
    def handle_predict_now(self, message: dict):
        """处理 /predict-now 命令"""
        try:
            # 发送处理中消息
            self.send_message("⏳ 正在执行预测，请稍候...")
            
            # 执行预测
            logger.info("收到 /predict-now 命令，开始预测...")
            result = self.prediction_server.predict()
            
            if result:
                # 格式化报告
                report = self.prediction_server.format_report(result)
                # 发送报告
                success = self.send_message(report)
                if success:
                    logger.info("✅ 预测结果已发送到 Telegram")
                else:
                    logger.error("❌ 发送预测结果失败")
            else:
                self.send_message("❌ 预测失败，请检查服务器日志")
                
        except Exception as e:
            logger.error(f"处理 /predict-now 命令失败: {e}")
            import traceback
            traceback.print_exc()
            self.send_message(f"❌ 执行预测时出错: {str(e)}")
    
    def process_update(self, update: dict):
        """处理 Telegram 更新"""
        if 'message' not in update:
            return
        
        message = update['message']
        chat_id = str(message.get('chat', {}).get('id', ''))
        
        # 只处理指定 Chat ID 的消息
        if chat_id != self.chat_id:
            logger.warning(f"收到来自其他 Chat ID 的消息: {chat_id}")
            return
        
        # 检查是否是命令
        if 'text' in message:
            text = message['text'].strip()
            
            # 处理命令
            if text.startswith('/'):
                command = text.split()[0][1:].lower()  # 移除 '/' 并转小写
                
                if command in self.commands:
                    logger.info(f"收到命令: /{command}")
                    self.commands[command](message)
                else:
                    self.send_message(f"❓ 未知命令: /{command}\n使用 /help 查看可用命令")
    
    def get_updates(self, offset: int = None) -> list:
        """获取 Telegram 更新"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {"timeout": 30}
            if offset:
                params["offset"] = offset
            
            query = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{url}?{query}"
            
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=35) as response:
                result = json.loads(response.read().decode())
                if result.get('ok'):
                    return result.get('result', [])
                return []
        except Exception as e:
            logger.error(f"获取更新失败: {e}")
            return []
    
    def run_polling(self):
        """运行轮询模式"""
        logger.info("🤖 Telegram Bot 开始运行（轮询模式）")
        logger.info(f"   监听 Chat ID: {self.chat_id}")
        
        offset = None
        
        while True:
            try:
                updates = self.get_updates(offset)
                
                for update in updates:
                    # 更新 offset
                    update_id = update.get('update_id')
                    if update_id:
                        offset = update_id + 1
                    
                    # 处理更新
                    self.process_update(update)
                
                # 短暂休眠避免过于频繁的请求
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("收到停止信号，退出...")
                break
            except Exception as e:
                logger.error(f"轮询错误: {e}")
                time.sleep(5)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Telegram Bot 命令处理器')
    parser.add_argument('--model', type=str, required=True,
                        help='模型文件路径 (.pkl)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='交易对 (默认: BTCUSDT)')
    parser.add_argument('--telegram-token', type=str, default=None,
                        help='Telegram Bot Token (或设置环境变量 TELEGRAM_BOT_TOKEN)')
    parser.add_argument('--telegram-chat-id', type=str, default=None,
                        help='Telegram Chat ID (或设置环境变量 TELEGRAM_CHAT_ID)')
    
    args = parser.parse_args()
    
    # 从环境变量获取配置
    telegram_token = args.telegram_token or os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = args.telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID')
    
    if not telegram_token or not telegram_chat_id:
        logger.error("❌ Telegram 配置不完整")
        logger.error("   请设置 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID")
        sys.exit(1)
    
    # 创建预测服务器（不启用 Telegram 通知，因为我们会手动发送）
    prediction_server = PredictionServer(
        model_path=args.model,
        symbol=args.symbol,
        telegram_token=None,  # 不自动发送，由 bot 处理
        telegram_chat_id=None
    )
    
    # 创建 Telegram Bot
    bot = TelegramBot(telegram_token, telegram_chat_id, prediction_server)
    
    # 运行轮询
    try:
        bot.run_polling()
    except KeyboardInterrupt:
        logger.info("Bot 已停止")


if __name__ == "__main__":
    main()

