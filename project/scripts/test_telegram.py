#!/usr/bin/env python3
"""
Telegram 配置测试脚本

用于测试 Telegram Bot Token 和 Chat ID 是否正确
"""

import os
import sys
import json
import urllib.request
import ssl
from pathlib import Path

# 加载 .env
try:
    from config import load_dotenv
    load_dotenv()
except ImportError:
    # 手动加载 .env
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

token = os.getenv('TELEGRAM_BOT_TOKEN', '')
chat_id = os.getenv('TELEGRAM_CHAT_ID', '')

if not token or not chat_id:
    print('❌ 错误: Telegram 配置不完整')
    print(f'   TELEGRAM_BOT_TOKEN: {"已设置" if token else "未设置"}')
    print(f'   TELEGRAM_CHAT_ID: {"已设置" if chat_id else "未设置"}')
    sys.exit(1)

print('=' * 60)
print('📱 Telegram 配置测试')
print('=' * 60)
print()

# SSL 配置
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# 测试 1: 验证 Bot Token
print('1️⃣ 测试 Bot Token...')
try:
    url = f"https://api.telegram.org/bot{token}/getMe"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
        result = json.loads(response.read().decode())
        if result.get('ok'):
            bot_info = result.get('result', {})
            print(f'   ✅ Bot Token 有效')
            print(f'   Bot 名称: {bot_info.get("first_name", "N/A")}')
            print(f'   Bot 用户名: @{bot_info.get("username", "N/A")}')
        else:
            print(f'   ❌ Bot Token 无效: {result.get("description", "未知错误")}')
            sys.exit(1)
except urllib.error.HTTPError as e:
    error_body = e.read().decode()
    try:
        error_data = json.loads(error_body)
        print(f'   ❌ HTTP {e.code}: {error_data.get("description", "未知错误")}')
        if e.code == 401:
            print('   💡 Bot Token 无效或已过期')
    except:
        print(f'   ❌ HTTP {e.code}: {error_body}')
    sys.exit(1)
except Exception as e:
    print(f'   ❌ 连接失败: {e}')
    sys.exit(1)

print()

# 测试 2: 发送测试消息
print('2️⃣ 发送测试消息...')
test_message = """
🧪 <b>Telegram 配置测试</b>

这是一条测试消息，用于验证 BTC 预测服务的 Telegram 通知功能。

✅ 如果收到此消息，说明配置正确！
"""

url = f"https://api.telegram.org/bot{token}/sendMessage"
data = {
    "chat_id": chat_id,
    "text": test_message,
    "parse_mode": "HTML"
}

try:
    json_data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=json_data,
        headers={'Content-Type': 'application/json'}
    )
    
    with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
        result = json.loads(response.read().decode())
        
        if result.get('ok'):
            print('   ✅ 测试消息发送成功！')
            msg_info = result.get('result', {})
            print(f'   消息 ID: {msg_info.get("message_id", "N/A")}')
            print(f'   发送时间: {msg_info.get("date", "N/A")}')
            print()
            print('🎉 Telegram 配置完全正确！')
        else:
            print('   ❌ 发送失败')
            print(f'   错误: {result.get("description", "未知错误")}')
            sys.exit(1)
            
except urllib.error.HTTPError as e:
    error_body = e.read().decode()
    try:
        error_data = json.loads(error_body)
        error_desc = error_data.get('description', '未知错误')
        error_code = error_data.get('error_code', e.code)
        print(f'   ❌ HTTP {error_code}: {error_desc}')
        
        if error_code == 404:
            print('   💡 可能原因:')
            print('      - Chat ID 不正确')
            print('      - 未向机器人发送过消息')
            print('      - Bot Token 格式错误')
            print()
            print('   🔧 解决方法:')
            print('      1. 向你的机器人发送任意消息')
            print('      2. 访问: https://api.telegram.org/bot<TOKEN>/getUpdates')
            print('      3. 在返回的 JSON 中找到 "chat":{"id": 123456789}')
        elif error_code == 400:
            print('   💡 可能原因: Chat ID 格式不正确（应该是数字）')
        elif error_code == 401:
            print('   💡 可能原因: Bot Token 无效')
    except:
        print(f'   ❌ HTTP {e.code}: {error_body}')
    sys.exit(1)
    
except Exception as e:
    print(f'   ❌ 连接失败: {e}')
    sys.exit(1)

print('=' * 60)

