#!/usr/bin/env python3
"""
Telegram Bot 诊断脚本

用于诊断 Bot 不工作的原因
"""

import os
import sys
import json
import ssl
import urllib.request
from pathlib import Path

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

token = os.getenv('TELEGRAM_BOT_TOKEN', '')
chat_id = os.getenv('TELEGRAM_CHAT_ID', '')

print('=' * 60)
print('🔍 Telegram Bot 诊断')
print('=' * 60)
print()

# 1. 检查配置
print('1️⃣ 检查配置...')
if not token:
    print('   ❌ TELEGRAM_BOT_TOKEN 未设置')
    sys.exit(1)
else:
    print(f'   ✅ TELEGRAM_BOT_TOKEN: {token[:10]}...')

if not chat_id:
    print('   ❌ TELEGRAM_CHAT_ID 未设置')
    sys.exit(1)
else:
    print(f'   ✅ TELEGRAM_CHAT_ID: {chat_id}')
print()

# 2. 测试 Bot API
print('2️⃣ 测试 Bot API...')
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

try:
    url = f"https://api.telegram.org/bot{token}/getMe"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
        result = json.loads(response.read().decode())
        if result.get('ok'):
            bot_info = result.get('result', {})
            print(f'   ✅ Bot 连接成功')
            print(f'   Bot 名称: {bot_info.get("first_name", "N/A")}')
            print(f'   Bot 用户名: @{bot_info.get("username", "N/A")}')
        else:
            print(f'   ❌ Bot API 错误: {result.get("description", "未知错误")}')
            sys.exit(1)
except Exception as e:
    print(f'   ❌ 连接失败: {e}')
    sys.exit(1)
print()

# 3. 获取最近的更新
print('3️⃣ 检查最近的更新...')
try:
    url = f"https://api.telegram.org/bot{token}/getUpdates?timeout=1"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ssl_context, timeout=5) as response:
        result = json.loads(response.read().decode())
        if result.get('ok'):
            updates = result.get('result', [])
            print(f'   ✅ 获取到 {len(updates)} 条更新')
            
            if updates:
                print('   📋 最近的更新:')
                for i, update in enumerate(updates[-3:], 1):  # 只显示最后3条
                    if 'message' in update:
                        msg = update['message']
                        msg_chat_id = str(msg.get('chat', {}).get('id', ''))
                        msg_text = msg.get('text', 'N/A')
                        print(f'      {i}. Chat ID: {msg_chat_id}, 消息: {msg_text[:50]}')
                        
                        # 检查 Chat ID 是否匹配
                        if msg_chat_id != chat_id:
                            print(f'      ⚠️  Chat ID 不匹配! 配置: {chat_id}, 消息来自: {msg_chat_id}')
            else:
                print('   ⚠️  没有收到任何更新')
                print('   💡 提示: 向 Bot 发送任意消息来生成更新')
        else:
            print(f'   ❌ 获取更新失败: {result.get("description", "未知错误")}')
except Exception as e:
    print(f'   ❌ 获取更新失败: {e}')
print()

# 4. 测试命令格式
print('4️⃣ 检查命令格式...')
test_commands = ['/predict-now', '/start', '/help', 'predict-now']
print('   支持的命令格式:')
for cmd in test_commands:
    if cmd.startswith('/'):
        cmd_name = cmd[1:].split()[0].lower()
        print(f'      {cmd} -> 命令名: "{cmd_name}"')
print()

# 5. 检查 Bot 是否在运行
print('5️⃣ 检查进程...')
import subprocess
try:
    result = subprocess.run(['pgrep', '-f', 'telegram_bot'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        pids = result.stdout.strip().split('\n')
        print(f'   ✅ Bot 进程正在运行 (PID: {", ".join(pids)})')
    else:
        print('   ⚠️  Bot 进程未运行')
        print('   💡 提示: 需要启动 Bot 才能接收命令')
except:
    print('   ⚠️  无法检查进程状态')
print()

# 6. 建议
print('=' * 60)
print('📋 诊断总结')
print('=' * 60)
print()
print('如果命令不工作，请检查:')
print('  1. Bot 是否正在运行')
print('  2. Chat ID 是否正确（必须与发送消息的 Chat ID 匹配）')
print('  3. 命令格式是否正确（应该是 /predict-now，不是 predict-now）')
print('  4. 是否已向 Bot 发送过消息（首次需要先发送消息）')
print()
print('启动 Bot:')
print('  python3 telegram_bot.py --model ../models/regression_model_20251213_213205.pkl')
print('=' * 60)

