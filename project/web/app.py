#!/usr/bin/env python3
"""
BTC 价格预测 - Web 前端服务器

提供 REST API 和 Web 界面显示预测信息
"""

import os
import sys
import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# 添加 scripts 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

try:
    from prediction_server import PredictionServer
    from config import load_dotenv
    load_dotenv()
except ImportError as e:
    print(f"Warning: Could not import prediction_server: {e}")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

# 全局预测服务器实例
prediction_server = None


def init_prediction_server():
    """初始化预测服务器"""
    global prediction_server
    
    if prediction_server is not None:
        return prediction_server
    
    # 从环境变量或默认路径加载模型
    model_path = os.getenv('MODEL_PATH', '../models/regression_model_20251213_213205.pkl')
    if not os.path.isabs(model_path):
        model_path = str(Path(__file__).parent.parent / 'models' / Path(model_path).name)
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found: {model_path}")
        return None
    
    try:
        prediction_server = PredictionServer(
            model_path=model_path,
            symbol='BTCUSDT',
            telegram_token=None,  # Web UI 不需要 Telegram
            telegram_chat_id=None,
            risk_level='moderate'
        )
        print(f"✅ Prediction server initialized with model: {model_path}")
        return prediction_server
    except Exception as e:
        print(f"Error initializing prediction server: {e}")
        return None


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    """执行预测 API"""
    try:
        server = init_prediction_server()
        if server is None:
            return jsonify({
                'success': False,
                'error': 'Prediction server not initialized'
            }), 500
        
        # 执行预测
        result = server.predict()
        
        if result is None:
            return jsonify({
                'success': False,
                'error': 'Prediction failed'
            }), 500
        
        # 计算预测价格
        current_price = result['current_price']
        prediction_pct = result['prediction_pct']
        predicted_price = current_price * (1 + prediction_pct / 100)
        
        # UTC+8 时间
        utc8 = timezone(timedelta(hours=8))
        now_utc8 = datetime.now(utc8)
        prediction_time_utc8 = (datetime.now() + timedelta(hours=20)).replace(tzinfo=timezone.utc).astimezone(utc8)
        
        # 仓位建议（如果可用）
        position_info = None
        if server.position_manager:
            position_info = server.position_manager.calculate_position_size(
                signal_strength=result['features_summary']['signal_strength'],
                confidence=result['confidence'],
                prediction_pct=result['prediction_pct']
            )
        
        # 格式化响应
        response = {
            'success': True,
            'timestamp': now_utc8.isoformat(),
            'prediction_time': prediction_time_utc8.isoformat(),
            'current_price': current_price,
            'predicted_price': predicted_price,
            'prediction_pct': prediction_pct,
            'direction': result['direction'],
            'direction_emoji': result['direction_emoji'],
            'range': result['range'],
            'confidence': result['confidence'],
            'signal_strength': result['features_summary']['signal_strength'],
            'market_status': {
                'rsi_1h': result['features_summary']['rsi_1h'],
                'adx_1h': result['features_summary']['adx_1h'],
                'volatility_1h': result['features_summary']['volatility_1h'] * 100,
            },
            'funding_rate': result['funding_rate'] * 100,
            'position_recommendation': position_info
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """获取预测历史"""
    try:
        server = init_prediction_server()
        if server is None:
            return jsonify({'success': False, 'error': 'Server not initialized'}), 500
        
        # 检查是否有历史记录
        if not hasattr(server, 'prediction_history') or not server.prediction_history:
            return jsonify({
                'success': True,
                'history': []
            })
        
        history = server.prediction_history[-50:]  # 最近50条
        
        # 格式化历史数据
        formatted_history = []
        for item in history:
            try:
                utc8 = timezone(timedelta(hours=8))
                # 处理时间戳（可能是字符串或datetime对象）
                if isinstance(item.get('timestamp'), str):
                    timestamp = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                else:
                    timestamp = item.get('timestamp', datetime.now())
                
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                
                timestamp_utc8 = timestamp.astimezone(utc8)
                
                formatted_history.append({
                    'timestamp': timestamp_utc8.isoformat(),
                    'current_price': item.get('current_price', 0),
                    'prediction_pct': item.get('prediction_pct', 0),
                    'direction': item.get('direction', 'Unknown'),
                    'confidence': item.get('confidence', 0),
                    'signal_strength': item.get('features_summary', {}).get('signal_strength', 0),
                })
            except Exception as e:
                # 跳过格式错误的历史记录
                continue
        
        return jsonify({
            'success': True,
            'history': formatted_history
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """服务器状态"""
    server = init_prediction_server()
    return jsonify({
        'success': True,
        'server_initialized': server is not None,
        'model_loaded': server is not None,
        'timestamp': datetime.now(timezone(timedelta(hours=8))).isoformat()
    })


if __name__ == '__main__':
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='BTC Price Prediction Web Dashboard')
    parser.add_argument('--port', '-p', type=int, default=None,
                        help='Web server port (default: 8080 or from PORT env var)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable Flask debug mode')
    args = parser.parse_args()
    
    # 初始化预测服务器
    init_prediction_server()
    
    # 确定端口：命令行参数 > 环境变量 > 默认值
    port = args.port or int(os.getenv('PORT', 8080))
    debug = args.debug or (os.getenv('FLASK_DEBUG', 'False').lower() == 'true')
    
    print(f"🚀 Starting Web UI server on http://localhost:{port}")
    print(f"📊 Dashboard: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{port}")
    app.run(host=args.host, port=port, debug=debug)

