#!/usr/bin/env python3
"""
BTC 价格预测 - Web 前端服务器

提供 REST API 和 Web 界面显示预测信息
"""

import os
import sys
import json
import pickle
import ssl
import urllib.request
import numpy as np
import pandas as pd
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS


# ============ VZO Indicator Functions ============

def fetch_binance_klines(symbol: str = 'BTCUSDT', interval: str = '15m', limit: int = 100) -> pd.DataFrame:
    """Fetch klines from Binance Futures API"""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    req = urllib.request.Request(url)
    
    with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
        data = json.loads(response.read().decode())
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_vzo(df: pd.DataFrame, period: int = 14, ma_len: int = 9) -> pd.DataFrame:
    """
    Calculate Volume Zone Oscillator (VZO)
    
    VZO = 100 * EMA(signed_volume) / EMA(total_volume)
    where signed_volume = +volume if close > open, else -volume
    
    Args:
        df: DataFrame with 'open', 'close', 'volume' columns
        period: EMA period for VZO calculation (default 14)
        ma_len: Moving average length for VZO smoothing (default 9)
    
    Returns:
        DataFrame with VZO values and MA
    """
    # Signed volume: positive when close > open (bullish), negative otherwise (bearish)
    signed_vol = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
    
    # EMA of signed volume and total volume
    vp = calculate_ema(pd.Series(signed_vol), period)
    tv = calculate_ema(df['volume'], period)
    
    # VZO = 100 * EMA(signed_vol) / EMA(total_vol)
    vzo = 100 * vp / tv
    
    # Moving average of VZO
    vzo_ma = calculate_ema(vzo, ma_len)
    
    result = pd.DataFrame({
        'timestamp': df['timestamp'],
        'vzo': vzo,
        'vzo_ma': vzo_ma,
        'close': df['close']
    })
    
    return result


def calculate_slope(values: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Calculate slope using linear regression over lookback period
    
    Args:
        values: Array of values
        lookback: Number of periods to calculate slope over
    
    Returns:
        Array of slopes (same length as input, with NaN for initial values)
    """
    slopes = np.full(len(values), np.nan)
    x = np.arange(lookback)
    
    for i in range(lookback - 1, len(values)):
        y = values[i - lookback + 1:i + 1]
        if not np.any(np.isnan(y)):
            # Linear regression: slope = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
            slope, _ = np.polyfit(x, y, 1)
            slopes[i] = slope
    
    return slopes


def get_vzo_signal(slope: float, vzo: float, threshold: float = 1.5,
                   prev_slope: float = None) -> dict:
    """
    Generate trading signal based on VZO slope
    
    Args:
        slope: Current VZO slope
        vzo: Current VZO value
        threshold: Minimum absolute slope to trigger signal
        prev_slope: Previous slope value for zero-cross detection
    
    Returns:
        Signal dict with direction, strength, and crossing info
    """
    if np.isnan(slope):
        return {
            'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Insufficient data',
            'slope_crossed': False, 'cross_direction': None, 'zone_warning': False,
        }
    
    abs_slope = abs(slope)
    
    # Detect slope zero-cross
    slope_crossed = False
    cross_direction = None  # 'bullish' or 'bearish'
    if prev_slope is not None and not np.isnan(prev_slope):
        if prev_slope <= 0 < slope:
            slope_crossed = True
            cross_direction = 'bullish'  # slope turned positive -> potential ENTER
        elif prev_slope >= 0 > slope:
            slope_crossed = True
            cross_direction = 'bearish'  # slope turned negative -> potential EXIT
    
    # Zone warning: overbought with negative slope
    zone_warning = (vzo > 40 and slope < 0) or (vzo < -40 and slope > 0)
    
    if abs_slope < threshold:
        result = {
            'signal': 'NEUTRAL',
            'strength': abs_slope / threshold,
            'reason': f'Weak momentum (slope: {slope:.2f})'
        }
    elif slope > threshold:
        strength = min(abs_slope / (threshold * 3), 1.0)
        zone = 'overbought zone' if vzo > 40 else ('bullish zone' if vzo > 0 else 'recovery')
        result = {
            'signal': 'LONG',
            'strength': strength,
            'reason': f'Strong buying acceleration in {zone} (slope: {slope:.2f}, VZO: {vzo:.1f})'
        }
    else:  # slope < -threshold
        strength = min(abs_slope / (threshold * 3), 1.0)
        zone = 'oversold zone' if vzo < -40 else ('bearish zone' if vzo < 0 else 'decline')
        result = {
            'signal': 'SHORT',
            'strength': strength,
            'reason': f'Strong selling acceleration in {zone} (slope: {slope:.2f}, VZO: {vzo:.1f})'
        }
    
    result['slope_crossed'] = slope_crossed
    result['cross_direction'] = cross_direction
    result['zone_warning'] = zone_warning
    return result


# ============ Trade Signal Manager ============

class TradeSignalManager:
    """
    Stateful trade signal manager that uses 15m VZO/slope data
    to generate ENTER/EXIT/HOLD/WARNING signals.
    
    Combines three signal layers:
    1. Slope zero-cross detection (primary)
    2. VZO-slope divergence (early warning)
    3. Slope acceleration / 2nd derivative (urgency)
    """
    
    HISTORY_SIZE = 20  # number of recent data points to track
    
    def __init__(self):
        self.vzo_history = []     # list of float
        self.slope_history = []   # list of float
        self._loaded_from_db = False
    
    def load_from_db(self):
        """Load recent VZO/slope history from the prediction database."""
        if self._loaded_from_db:
            return
        try:
            from prediction_db import fetch_recent_vzo_slopes
            records = fetch_recent_vzo_slopes(limit=self.HISTORY_SIZE)
            for rec in records:
                self.vzo_history.append(rec.vzo_15m)
                self.slope_history.append(rec.slope_15m)
            self._loaded_from_db = True
            print(f"  TradeSignalManager: loaded {len(records)} historical VZO/slope records from DB")
        except Exception as e:
            print(f"  TradeSignalManager: could not load from DB: {e}")
            self._loaded_from_db = True  # don't retry
    
    def _append(self, vzo: float, slope: float):
        """Add a new data point and trim history."""
        self.vzo_history.append(vzo)
        self.slope_history.append(slope)
        if len(self.vzo_history) > self.HISTORY_SIZE:
            self.vzo_history = self.vzo_history[-self.HISTORY_SIZE:]
            self.slope_history = self.slope_history[-self.HISTORY_SIZE:]
    
    def _detect_slope_cross(self, slope: float) -> dict:
        """Detect if slope crossed zero since previous reading."""
        if len(self.slope_history) < 1:
            return {'crossed': False, 'direction': None}
        
        prev_slope = self.slope_history[-1]
        if prev_slope <= 0 < slope:
            return {'crossed': True, 'direction': 'bullish'}
        elif prev_slope >= 0 > slope:
            return {'crossed': True, 'direction': 'bearish'}
        return {'crossed': False, 'direction': None}
    
    def _detect_divergence(self, vzo: float, slope: float) -> dict:
        """
        Detect VZO-slope divergence:
        - Bearish divergence: VZO rising but slope falling (for 2+ bars)
        - Bullish divergence: VZO falling but slope rising (for 2+ bars)
        """
        if len(self.vzo_history) < 2 or len(self.slope_history) < 2:
            return {'divergence': False, 'type': None, 'bars': 0}
        
        # Check bearish divergence: VZO up, slope down
        bearish_bars = 0
        for i in range(len(self.vzo_history) - 1, max(len(self.vzo_history) - 5, -1), -1):
            if i <= 0:
                break
            if self.vzo_history[i] > self.vzo_history[i - 1] and self.slope_history[i] < self.slope_history[i - 1]:
                bearish_bars += 1
            else:
                break
        
        # Check if current bar continues the divergence
        if len(self.vzo_history) >= 1:
            if vzo > self.vzo_history[-1] and slope < self.slope_history[-1]:
                bearish_bars += 1
        
        if bearish_bars >= 2:
            return {'divergence': True, 'type': 'bearish_divergence', 'bars': bearish_bars}
        
        # Check bullish divergence: VZO down, slope up
        bullish_bars = 0
        for i in range(len(self.vzo_history) - 1, max(len(self.vzo_history) - 5, -1), -1):
            if i <= 0:
                break
            if self.vzo_history[i] < self.vzo_history[i - 1] and self.slope_history[i] > self.slope_history[i - 1]:
                bullish_bars += 1
            else:
                break
        
        if len(self.vzo_history) >= 1:
            if vzo < self.vzo_history[-1] and slope > self.slope_history[-1]:
                bullish_bars += 1
        
        if bullish_bars >= 2:
            return {'divergence': True, 'type': 'bullish_divergence', 'bars': bullish_bars}
        
        return {'divergence': False, 'type': None, 'bars': 0}
    
    def _compute_slope_acceleration(self, slope: float) -> float:
        """
        Compute 2nd derivative of slope (slope of slopes).
        Positive = accelerating, Negative = decelerating.
        """
        if len(self.slope_history) < 2:
            return 0.0
        
        # Use last 3 slope values (including current) for acceleration
        recent = list(self.slope_history[-2:]) + [slope]
        if len(recent) < 3:
            return recent[-1] - recent[-2] if len(recent) >= 2 else 0.0
        
        # Simple 2nd derivative: (s[2] - s[1]) - (s[1] - s[0])
        accel = (recent[2] - recent[1]) - (recent[1] - recent[0])
        return accel
    
    def _count_consecutive_negative_slopes(self, slope: float) -> int:
        """Count how many consecutive bars have negative slope (including current)."""
        count = 1 if slope < 0 else 0
        for s in reversed(self.slope_history):
            if s < 0:
                count += 1
            else:
                break
        return count
    
    def update(self, vzo: float, slope: float, ml_direction: str = '',
               ml_confidence: float = 0.0, threshold: float = 1.5) -> dict:
        """
        Process a new 15m VZO/slope data point and return a trade signal.
        
        Args:
            vzo: Current 15m VZO value
            slope: Current 15m VZO slope
            ml_direction: ML model direction ('看涨', '看跌', '震荡')
            ml_confidence: ML model confidence (0-100)
            threshold: Slope threshold for signal generation
            
        Returns:
            dict with keys:
              - action: 'ENTER' | 'EXIT' | 'HOLD' | 'WARNING'
              - reason: str
              - urgency: float (0-1)
              - slope_cross: dict from _detect_slope_cross
              - divergence: dict from _detect_divergence
              - slope_acceleration: float
        """
        self.load_from_db()
        
        cross = self._detect_slope_cross(slope)
        divergence = self._detect_divergence(vzo, slope)
        accel = self._compute_slope_acceleration(slope)
        neg_bars = self._count_consecutive_negative_slopes(slope)
        
        # VZO is now INFORMATIONAL ONLY - no entry/exit triggers
        # Just describe the current VZO state
        action = 'INFO'
        reason = 'VZO monitoring (informational only)'
        urgency = 0.0
        
        # Describe VZO zone
        if vzo > 40:
            zone_desc = f'Overbought zone (VZO: {vzo:.1f})'
        elif vzo > 5:
            zone_desc = f'Bullish zone (VZO: {vzo:.1f})'
        elif vzo > -5:
            zone_desc = f'Neutral zone (VZO: {vzo:.1f})'
        elif vzo > -40:
            zone_desc = f'Bearish zone (VZO: {vzo:.1f})'
        else:
            zone_desc = f'Oversold zone (VZO: {vzo:.1f})'
        
        # Describe slope direction
        if slope > threshold:
            slope_desc = f'Rising ({slope:.2f})'
        elif slope < -threshold:
            slope_desc = f'Falling ({slope:.2f})'
        else:
            slope_desc = f'Flat ({slope:.2f})'
        
        reason = f'{zone_desc}, Slope: {slope_desc}'
        
        # Add cross info if detected
        if cross['crossed']:
            reason += f' | Slope crossed {cross["direction"]}'
        
        # Add divergence info if detected
        if divergence['divergence']:
            reason += f' | {divergence["type"]} detected'
        
        # Append to history AFTER analysis (so next call can compare)
        self._append(vzo, slope)
        
        return {
            'action': action,
            'reason': reason,
            'urgency': round(min(urgency, 1.0), 2),
            'slope_cross': cross,
            'divergence': divergence,
            'slope_acceleration': round(accel, 3),
            'consecutive_negative_bars': neg_bars,
        }


# Global trade signal manager instance
trade_signal_manager = TradeSignalManager()


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

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
predict_lock = threading.Lock()

try:
    from prediction_db import init_db as init_prediction_db
    from prediction_db import save_prediction, fetch_recent_predictions, fetch_recent_vzo_slopes
    PREDICTION_DB_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import prediction_db: {e}")
    PREDICTION_DB_AVAILABLE = False


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


def _compute_15m_vzo_slope(symbol: str = 'BTCUSDT') -> dict:
    """
    Compute current 15m VZO and slope values.
    
    Returns dict with keys: vzo_15m, slope_15m, vzo_signal_15m, prev_slope_15m
    or None values on failure.
    """
    try:
        df = fetch_binance_klines(symbol, '15m', 100)
        vzo_df = calculate_vzo(df, period=14, ma_len=9)
        vzo_values = vzo_df['vzo'].values
        slopes = calculate_slope(vzo_values, lookback=5)
        
        current_vzo = float(vzo_values[-1]) if not np.isnan(vzo_values[-1]) else None
        current_slope = float(slopes[-1]) if not np.isnan(slopes[-1]) else None
        prev_slope = float(slopes[-2]) if len(slopes) >= 2 and not np.isnan(slopes[-2]) else None
        
        # Get VZO signal
        vzo_signal = 'NEUTRAL'
        if current_vzo is not None and current_slope is not None:
            sig = get_vzo_signal(current_slope, current_vzo, threshold=1.5, prev_slope=prev_slope)
            vzo_signal = sig['signal']
        
        return {
            'vzo_15m': current_vzo,
            'slope_15m': current_slope,
            'prev_slope_15m': prev_slope,
            'vzo_signal_15m': vzo_signal,
        }
    except Exception as e:
        print(f"Warning: failed to compute 15m VZO/slope: {e}")
        return {'vzo_15m': None, 'slope_15m': None, 'prev_slope_15m': None, 'vzo_signal_15m': None}


def _persist_prediction_if_possible(result: dict, vzo_data: dict = None,
                                     trade_action: str = None) -> None:
    if not PREDICTION_DB_AVAILABLE or not isinstance(result, dict):
        return
    try:
        current_price = float(result.get("current_price", 0.0))
        prediction_pct = float(result.get("prediction_pct", 0.0))
        predicted_price = current_price * (1 + prediction_pct / 100)
        
        vzo_15m = None
        slope_15m = None
        vzo_signal_15m = None
        if vzo_data:
            vzo_15m = vzo_data.get('vzo_15m')
            slope_15m = vzo_data.get('slope_15m')
            vzo_signal_15m = vzo_data.get('vzo_signal_15m')
        
        save_prediction(
            result=result,
            predicted_price=predicted_price,
            vzo_15m=vzo_15m,
            slope_15m=slope_15m,
            vzo_signal_15m=vzo_signal_15m,
            trade_action=trade_action,
        )
    except Exception as e:
        print(f"Warning: failed to persist prediction: {e}")


def _start_background_predict_loop():
    """
    Start a background thread that runs predictions every N seconds (default 60s).

    Env vars:
      - AUTO_PREDICT: true/false (default true)
      - PREDICT_INTERVAL_SECONDS: default 60
    """
    auto = os.getenv("AUTO_PREDICT", "true").lower() in ("1", "true", "yes", "y", "on")
    if not auto:
        print("ℹ️ AUTO_PREDICT disabled; background loop not started.")
        return

    interval = int(os.getenv("PREDICT_INTERVAL_SECONDS", "60"))
    interval = max(10, interval)

    def loop():
        print(f"⏱️ Background prediction loop started (every {interval}s).")
        # small initial delay so server boots before heavy work
        time.sleep(2)
        while True:
            try:
                server = init_prediction_server()
                if server is None:
                    time.sleep(interval)
                    continue
                with predict_lock:
                    result = server.predict()
                if result:
                    # Compute 15m VZO/slope and trade signal
                    vzo_data = _compute_15m_vzo_slope(
                        symbol=result.get('symbol', 'BTCUSDT')
                    )
                    trade_action_result = None
                    if vzo_data.get('vzo_15m') is not None and vzo_data.get('slope_15m') is not None:
                        trade_action_result = trade_signal_manager.update(
                            vzo=vzo_data['vzo_15m'],
                            slope=vzo_data['slope_15m'],
                            ml_direction=result.get('direction', ''),
                            ml_confidence=result.get('confidence', 0.0),
                        )
                    ta = trade_action_result['action'] if trade_action_result else None
                    _persist_prediction_if_possible(result, vzo_data=vzo_data, trade_action=ta)
            except Exception as e:
                print(f"Warning: background prediction loop error: {e}")
            time.sleep(interval)

    t = threading.Thread(target=loop, name="background_predict_loop", daemon=True)
    t.start()


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
        with predict_lock:
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
        
        # Compute 15m VZO/slope
        vzo_data = _compute_15m_vzo_slope(
            symbol=result.get('symbol', 'BTCUSDT')
        )
        
        # Run trade signal manager
        trade_action_result = None
        if vzo_data.get('vzo_15m') is not None and vzo_data.get('slope_15m') is not None:
            trade_action_result = trade_signal_manager.update(
                vzo=vzo_data['vzo_15m'],
                slope=vzo_data['slope_15m'],
                ml_direction=result.get('direction', ''),
                ml_confidence=result.get('confidence', 0.0),
            )
        
        # 仓位建议（如果可用）
        position_info = None
        vzo_trade_action = trade_action_result['action'] if trade_action_result else None
        if server.position_manager:
            position_info = server.position_manager.calculate_position_size(
                signal_strength=result['features_summary']['signal_strength'],
                confidence=result['confidence'],
                prediction_pct=result['prediction_pct'],
                vzo_trade_action=vzo_trade_action,
            )
        
        # 格式化响应
        response = {
            'success': True,
            'timestamp': now_utc8.isoformat(),
            'prediction_time': prediction_time_utc8.isoformat(),
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'prediction_pct': float(prediction_pct),
            'direction': result['direction'],
            'direction_emoji': result['direction_emoji'],
            'range': result['range'],
            'confidence': float(result['confidence']),
            'signal_strength': float(result['features_summary']['signal_strength']),
            'market_status': {
                'rsi_1h': float(result['features_summary']['rsi_1h']),
                'adx_1h': float(result['features_summary']['adx_1h']),
                'volatility_1h': float(result['features_summary']['volatility_1h'] * 100),
            },
            'funding_rate': float(result['funding_rate'] * 100),
            'position_recommendation': convert_numpy_types(position_info),
            'vzo_15m': {
                'vzo': vzo_data.get('vzo_15m'),
                'slope': vzo_data.get('slope_15m'),
                'signal': vzo_data.get('vzo_signal_15m'),
            },
            'trade_signal': trade_action_result,
        }

        # Persist to SQL DB (best-effort)
        _persist_prediction_if_possible(
            result,
            vzo_data=vzo_data,
            trade_action=vzo_trade_action,
        )
        
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
        # Prefer SQL DB-backed history
        if PREDICTION_DB_AVAILABLE:
            items = fetch_recent_predictions(limit=50, symbol=request.args.get("symbol"))
            utc8 = timezone(timedelta(hours=8))
            formatted = []
            for item in items:
                try:
                    ts = datetime.fromisoformat(item.timestamp.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    ts_utc8 = ts.astimezone(utc8)
                except Exception:
                    ts_utc8 = datetime.now(utc8)

                formatted.append(
                    {
                        "timestamp": ts_utc8.isoformat(),
                        "current_price": item.current_price,
                        "prediction_pct": item.prediction_pct,
                        "direction": item.direction or "Unknown",
                        "confidence": item.confidence,
                        "signal_strength": item.signal_strength,
                        "vzo_15m": item.vzo_15m,
                        "slope_15m": item.slope_15m,
                        "vzo_signal_15m": item.vzo_signal_15m,
                        "trade_action": item.trade_action,
                    }
                )

            return jsonify({"success": True, "history": formatted})

        # Fallback: in-memory history
        server = init_prediction_server()
        if server is None:
            return jsonify({'success': False, 'error': 'Server not initialized'}), 500

        if not hasattr(server, "prediction_history") or not server.prediction_history:
            return jsonify({"success": True, "history": []})

        history = server.prediction_history[-50:]  # 最近50条
        formatted_history = []
        for item in history:
            try:
                utc8 = timezone(timedelta(hours=8))
                if isinstance(item.get("timestamp"), str):
                    timestamp = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                else:
                    timestamp = item.get("timestamp", datetime.now())
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                timestamp_utc8 = timestamp.astimezone(utc8)

                formatted_history.append(
                    {
                        "timestamp": timestamp_utc8.isoformat(),
                        "current_price": item.get("current_price", 0),
                        "prediction_pct": item.get("prediction_pct", 0),
                        "direction": item.get("direction", "Unknown"),
                        "confidence": item.get("confidence", 0),
                        "signal_strength": item.get("features_summary", {}).get("signal_strength", 0),
                    }
                )
            except Exception:
                continue

        return jsonify({"success": True, "history": formatted_history})
        
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


@app.route('/api/vzo', methods=['GET'])
def get_vzo():
    """
    Get VZO indicator data with slope calculations
    
    Query params:
        - symbol: Trading pair (default: BTCUSDT)
        - interval: Candle interval (default: 15m)
        - limit: Number of candles (default: 100, max: 500)
        - period: VZO EMA period (default: 14)
        - ma_len: VZO MA length (default: 9)
        - slope_lookback: Periods for slope calculation (default: 5)
        - slope_threshold: Minimum slope for signal (default: 1.5)
    """
    try:
        # Parse query parameters
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '15m')
        limit = min(int(request.args.get('limit', 100)), 500)
        period = int(request.args.get('period', 14))
        ma_len = int(request.args.get('ma_len', 9))
        slope_lookback = int(request.args.get('slope_lookback', 5))
        slope_threshold = float(request.args.get('slope_threshold', 1.5))
        
        # Fetch klines data
        df = fetch_binance_klines(symbol, interval, limit)
        
        # Calculate VZO
        vzo_df = calculate_vzo(df, period, ma_len)
        
        # Calculate slope
        vzo_values = vzo_df['vzo'].values
        slopes = calculate_slope(vzo_values, slope_lookback)
        vzo_df['slope'] = slopes
        
        # Generate signals for each point (with prev_slope for cross detection)
        signals = []
        for i in range(len(vzo_df)):
            prev_s = slopes[i - 1] if i > 0 and not np.isnan(slopes[i - 1]) else None
            sig = get_vzo_signal(slopes[i], vzo_values[i], slope_threshold, prev_slope=prev_s)
            signals.append(sig['signal'])
        vzo_df['signal'] = signals
        
        # Get current signal with prev_slope
        current_vzo = float(vzo_values[-1])
        current_slope = float(slopes[-1]) if not np.isnan(slopes[-1]) else 0
        prev_slope_val = float(slopes[-2]) if len(slopes) >= 2 and not np.isnan(slopes[-2]) else None
        current_signal = get_vzo_signal(current_slope, current_vzo, slope_threshold, prev_slope=prev_slope_val)
        
        # Compute slope acceleration (2nd derivative)
        slope_accel = 0.0
        valid_slopes = [float(s) for s in slopes[-3:] if not np.isnan(s)]
        if len(valid_slopes) >= 3:
            slope_accel = (valid_slopes[2] - valid_slopes[1]) - (valid_slopes[1] - valid_slopes[0])
        
        # Prepare response data
        data_points = []
        for idx, row in vzo_df.iterrows():
            data_points.append({
                'timestamp': row['timestamp'].isoformat(),
                'vzo': round(float(row['vzo']), 2) if not np.isnan(row['vzo']) else None,
                'vzo_ma': round(float(row['vzo_ma']), 2) if not np.isnan(row['vzo_ma']) else None,
                'slope': round(float(row['slope']), 3) if not np.isnan(row['slope']) else None,
                'signal': row['signal'],
                'price': round(float(row['close']), 2)
            })
        
        # UTC+8 时间
        utc8 = timezone(timedelta(hours=8))
        now_utc8 = datetime.now(utc8)
        
        return jsonify({
            'success': True,
            'timestamp': now_utc8.isoformat(),
            'symbol': symbol,
            'interval': interval,
            'parameters': {
                'period': period,
                'ma_len': ma_len,
                'slope_lookback': slope_lookback,
                'slope_threshold': slope_threshold
            },
            'current': {
                'vzo': round(current_vzo, 2),
                'slope': round(current_slope, 3),
                'signal': current_signal['signal'],
                'signal_strength': round(current_signal['strength'], 2),
                'reason': current_signal['reason'],
                'slope_crossed': current_signal.get('slope_crossed', False),
                'cross_direction': current_signal.get('cross_direction'),
                'zone_warning': current_signal.get('zone_warning', False),
                'slope_acceleration': round(slope_accel, 3),
            },
            'zones': {
                'overbought': 40,
                'oversold': -40
            },
            'data': data_points
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='BTC Price Prediction Web Dashboard')
    parser.add_argument('--port', '-p', type=int, default=None,
                        help='Web server port (default: 9000 or from PORT env var)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable Flask debug mode')
    args = parser.parse_args()
    
    # 确定端口：命令行参数 > 环境变量 > 默认值
    port = args.port or int(os.getenv('PORT', 9000))
    debug = args.debug or (os.getenv('FLASK_DEBUG', 'False').lower() == 'true')

    # 初始化预测服务器
    init_prediction_server()
    if PREDICTION_DB_AVAILABLE:
        try:
            init_prediction_db()
            print("🗄️ Prediction SQL DB initialized")
        except Exception as e:
            print(f"Warning: failed to init prediction DB: {e}")

    # 启动后台每分钟预测
    # (Avoid double-start when Flask debug reloader is enabled)
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _start_background_predict_loop()
    
    print(f"🚀 Starting Web UI server on http://localhost:{port}")
    print(f"📊 Dashboard: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{port}")
    app.run(host=args.host, port=port, debug=debug)

