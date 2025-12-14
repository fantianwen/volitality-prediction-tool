"""
BTC 价格预测服务器

功能:
1. 加载训练好的 GBM 模型
2. 实时从 Binance 获取多时间框架 K 线数据
3. 每30分钟预测未来 20 小时价格走势
4. 通过 Telegram 机器人发送预测报告

使用方法:
    python prediction_server.py --model ../models/regression_model_xxx.pkl

环境变量:
    TELEGRAM_BOT_TOKEN: Telegram 机器人 Token
    TELEGRAM_CHAT_ID: 接收消息的 Chat ID
"""

import os
import sys
import json
import pickle
import time
import ssl
import urllib.request
import argparse
import logging
import threading
import schedule
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# 导入仓位管理器
try:
    from position_manager import PositionManager, explain_signal_strength
    POSITION_MANAGER_AVAILABLE = True
except ImportError:
    POSITION_MANAGER_AVAILABLE = False

# 加载 .env 文件（如果存在）
try:
    from config import load_dotenv
    load_dotenv()
except ImportError:
    # 如果 config 模块不存在，手动加载 .env
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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prediction_server.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Telegram 通知器"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # SSL 配置
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        发送消息到 Telegram
        
        Args:
            text: 消息内容
            parse_mode: 解析模式 (HTML, Markdown, MarkdownV2)
        
        Returns:
            是否发送成功
        """
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
                if result.get('ok'):
                    logger.info("✅ Telegram 消息发送成功")
                    return True
                else:
                    logger.error(f"❌ Telegram 发送失败: {result}")
                    return False
                    
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if hasattr(e, 'read') else ''
            try:
                error_data = json.loads(error_body) if error_body else {}
                error_desc = error_data.get('description', '未知错误')
                error_code = error_data.get('error_code', e.code)
            except:
                error_desc = str(e)
                error_code = e.code
            
            logger.error(f"❌ Telegram HTTP 错误 {error_code}: {error_desc}")
            if error_code == 404:
                logger.error("   💡 可能原因: Bot Token 或 Chat ID 不正确")
                logger.error(f"   💡 请检查: Bot Token 是否正确，Chat ID 是否为数字")
            elif error_code == 401:
                logger.error("   💡 可能原因: Bot Token 无效或已过期")
            elif error_code == 400:
                logger.error("   💡 可能原因: Chat ID 不正确，或未向机器人发送过消息")
            return False
        except Exception as e:
            logger.error(f"❌ Telegram 发送异常: {e}")
            logger.error(f"   💡 错误类型: {type(e).__name__}")
            return False


class BinanceDataFetcher:
    """Binance 数据获取器"""
    
    def __init__(self, symbol: str = 'BTCUSDT'):
        self.symbol = symbol.upper()
        self.base_url = "https://fapi.binance.com"
        
        # SSL 配置
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # 支持的时间周期
        self.timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
    
    def _request(self, endpoint: str, params: dict = None) -> dict:
        """发送 REST API 请求"""
        url = f"{self.base_url}{endpoint}"
        if params:
            query = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{url}?{query}"
        
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=self.ssl_context, timeout=30) as response:
            return json.loads(response.read().decode())
    
    def get_klines(self, interval: str, limit: int = 200) -> pd.DataFrame:
        """获取 K 线数据"""
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": min(limit, 1500)
        }
        
        data = self._request("/fapi/v1/klines", params)
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def get_funding_rate(self) -> dict:
        """获取资金费率"""
        data = self._request("/fapi/v1/premiumIndex", {"symbol": self.symbol})
        return {
            'funding_rate': float(data.get('lastFundingRate', 0)),
            'mark_price': float(data.get('markPrice', 0)),
            'index_price': float(data.get('indexPrice', 0)),
        }
    
    def get_all_timeframes(self, limit: int = 200) -> Dict[str, pd.DataFrame]:
        """获取所有时间周期的数据"""
        data_dict = {}
        for tf in self.timeframes:
            logger.debug(f"获取 {tf} K 线数据...")
            data_dict[tf] = self.get_klines(tf, limit)
            time.sleep(0.1)
        return data_dict
    
    def get_current_price(self) -> float:
        """获取当前价格"""
        data = self._request("/fapi/v1/ticker/price", {"symbol": self.symbol})
        return float(data.get('price', 0))


class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()
    
    @staticmethod
    def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> tuple:
        low_n = df['low'].rolling(window=n).min()
        high_n = df['high'].rolling(window=n).max()
        
        rsv = (df['close'] - low_n) / (high_n - low_n) * 100
        rsv = rsv.fillna(50)
        
        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return k, d, j
    
    @staticmethod
    def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        ema_fast = TechnicalIndicators.calculate_ema(close, fast)
        ema_slow = TechnicalIndicators.calculate_ema(close, slow)
        
        macd = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd, signal)
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    @staticmethod
    def detect_crossover(fast: pd.Series, slow: pd.Series) -> tuple:
        golden = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        death = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        return golden, death
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        rsi = rsi.fillna(50)
        rsi = rsi.replace([np.inf, -np.inf], 50)
        
        return rsi
    
    @staticmethod
    def calculate_roc(close: pd.Series, period: int = 10) -> pd.Series:
        roc = (close - close.shift(period)) / close.shift(period) * 100
        return roc.fillna(0)
    
    @staticmethod
    def calculate_momentum(close: pd.Series, period: int = 10) -> pd.Series:
        mom = close - close.shift(period)
        return mom.fillna(0)
    
    @staticmethod
    def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_n = df['high'].rolling(window=period).max()
        low_n = df['low'].rolling(window=period).min()
        
        wr = (high_n - df['close']) / (high_n - low_n) * -100
        return wr.fillna(-50)
    
    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        cci = (tp - sma_tp) / (0.015 * mad)
        cci = cci.fillna(0)
        cci = cci.replace([np.inf, -np.inf], 0)
        
        return cci
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> tuple:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        adx = adx.fillna(25).replace([np.inf, -np.inf], 25)
        plus_di = plus_di.fillna(25).replace([np.inf, -np.inf], 25)
        minus_di = minus_di.fillna(25).replace([np.inf, -np.inf], 25)
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def calculate_stoch_rsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> tuple:
        rsi = TechnicalIndicators.calculate_rsi(close, rsi_period)
        
        rsi_low = rsi.rolling(window=stoch_period).min()
        rsi_high = rsi.rolling(window=stoch_period).max()
        
        stoch_rsi_k = (rsi - rsi_low) / (rsi_high - rsi_low) * 100
        stoch_rsi_d = stoch_rsi_k.rolling(window=3).mean()
        
        stoch_rsi_k = stoch_rsi_k.fillna(50)
        stoch_rsi_d = stoch_rsi_d.fillna(50)
        
        return stoch_rsi_k, stoch_rsi_d


class FeatureExtractor:
    """特征提取器 - 与训练时保持一致"""
    
    def __init__(self):
        self.timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        self.indicators = TechnicalIndicators()
    
    def extract_features(self, df_dict: Dict[str, pd.DataFrame]) -> pd.Series:
        """从多时间周期数据中提取特征"""
        features = {}
        
        for tf in self.timeframes:
            if tf not in df_dict or len(df_dict[tf]) < 50:
                continue
                
            df = df_dict[tf]
            
            # 1. KDJ
            k, d, j = self.indicators.calculate_kdj(df)
            
            # 2. MACD
            macd, signal, hist = self.indicators.calculate_macd(df['close'])
            
            # 3. RSI
            rsi_7 = self.indicators.calculate_rsi(df['close'], period=7)
            rsi_14 = self.indicators.calculate_rsi(df['close'], period=14)
            rsi_21 = self.indicators.calculate_rsi(df['close'], period=21)
            
            # 4. 交叉信号
            kdj_golden, kdj_death = self.indicators.detect_crossover(k, d)
            macd_golden, macd_death = self.indicators.detect_crossover(macd, signal)
            
            # 5. 波动率
            returns = df['close'].pct_change()
            periods_per_year = {'5m': 288*365, '15m': 96*365, '30m': 48*365, '1h': 24*365, '4h': 6*365, '1d': 365}
            volatility = returns.std() * np.sqrt(periods_per_year.get(tf, 365))
            
            # 6. 成交量特征
            volume = df['volume']
            volume_ma5 = volume.rolling(5).mean()
            volume_ma10 = volume.rolling(10).mean()
            volume_ma20 = volume.rolling(20).mean()
            
            vol_ratio_ma5 = volume.iloc[-1] / volume_ma5.iloc[-1] if volume_ma5.iloc[-1] > 0 else 1
            vol_ratio_ma10 = volume.iloc[-1] / volume_ma10.iloc[-1] if volume_ma10.iloc[-1] > 0 else 1
            vol_ratio_ma20 = volume.iloc[-1] / volume_ma20.iloc[-1] if volume_ma20.iloc[-1] > 0 else 1
            
            vol_change_1 = (volume.iloc[-1] - volume.iloc[-2]) / volume.iloc[-2] * 100 if volume.iloc[-2] > 0 else 0
            vol_change_5 = (volume.iloc[-1] - volume.iloc[-6]) / volume.iloc[-6] * 100 if len(volume) > 5 and volume.iloc[-6] > 0 else 0
            
            vol_trend = (volume_ma5.iloc[-1] - volume_ma20.iloc[-1]) / volume_ma20.iloc[-1] * 100 if volume_ma20.iloc[-1] > 0 else 0
            
            vol_high_20 = volume.tail(20).max()
            vol_low_20 = volume.tail(20).min()
            vol_position = (volume.iloc[-1] - vol_low_20) / (vol_high_20 - vol_low_20) if vol_high_20 > vol_low_20 else 0.5
            
            vol_spike = 1 if vol_ratio_ma20 > 2 else 0
            vol_shrink = 1 if vol_ratio_ma20 < 0.5 else 0
            
            price_up = 1 if df['close'].iloc[-1] > df['close'].iloc[-2] else 0
            vol_up = 1 if volume.iloc[-1] > volume.iloc[-2] else 0
            vol_price_divergence = 1 if price_up != vol_up else 0
            
            # 7. 价格位置
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            price_position = (df['close'].iloc[-1] - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
            
            # 8. 趋势强度
            ma20 = df['close'].rolling(20).mean()
            trend_strength = (df['close'].iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1] * 100 if ma20.iloc[-1] > 0 else 0
            
            # 9. RSI 衍生特征
            rsi_14_value = rsi_14.iloc[-1]
            rsi_overbought = 1 if rsi_14_value > 70 else 0
            rsi_oversold = 1 if rsi_14_value < 30 else 0
            rsi_trend = rsi_14.iloc[-1] - rsi_14.iloc[-5] if len(rsi_14) > 5 else 0
            
            # 10. 动量指标
            roc_5 = self.indicators.calculate_roc(df['close'], period=5)
            roc_10 = self.indicators.calculate_roc(df['close'], period=10)
            roc_20 = self.indicators.calculate_roc(df['close'], period=20)
            
            mom_10 = self.indicators.calculate_momentum(df['close'], period=10)
            mom_20 = self.indicators.calculate_momentum(df['close'], period=20)
            
            williams_r = self.indicators.calculate_williams_r(df, period=14)
            cci = self.indicators.calculate_cci(df, period=20)
            adx, plus_di, minus_di = self.indicators.calculate_adx(df, period=14)
            stoch_rsi_k, stoch_rsi_d = self.indicators.calculate_stoch_rsi(df['close'])
            
            cci_value = cci.iloc[-1]
            cci_overbought = 1 if cci_value > 100 else 0
            cci_oversold = 1 if cci_value < -100 else 0
            
            adx_value = adx.iloc[-1]
            adx_strong_trend = 1 if adx_value > 25 else 0
            adx_weak_trend = 1 if adx_value < 20 else 0
            trend_bullish = 1 if plus_di.iloc[-1] > minus_di.iloc[-1] else 0
            
            # 构建特征字典
            tf_features = {
                f'{tf}_kdj_k': k.iloc[-1],
                f'{tf}_kdj_d': d.iloc[-1],
                f'{tf}_kdj_j': j.iloc[-1],
                f'{tf}_kdj_golden': int(kdj_golden.iloc[-1]) if not pd.isna(kdj_golden.iloc[-1]) else 0,
                f'{tf}_kdj_death': int(kdj_death.iloc[-1]) if not pd.isna(kdj_death.iloc[-1]) else 0,
                
                f'{tf}_macd': macd.iloc[-1],
                f'{tf}_macd_signal': signal.iloc[-1],
                f'{tf}_macd_hist': hist.iloc[-1],
                f'{tf}_macd_golden': int(macd_golden.iloc[-1]) if not pd.isna(macd_golden.iloc[-1]) else 0,
                f'{tf}_macd_death': int(macd_death.iloc[-1]) if not pd.isna(macd_death.iloc[-1]) else 0,
                
                f'{tf}_volatility': volatility,
                
                f'{tf}_vol_ratio_ma5': vol_ratio_ma5,
                f'{tf}_vol_ratio_ma10': vol_ratio_ma10,
                f'{tf}_vol_ratio_ma20': vol_ratio_ma20,
                f'{tf}_vol_change_1': vol_change_1,
                f'{tf}_vol_change_5': vol_change_5,
                f'{tf}_vol_trend': vol_trend,
                f'{tf}_vol_position': vol_position,
                f'{tf}_vol_spike': vol_spike,
                f'{tf}_vol_shrink': vol_shrink,
                f'{tf}_vol_price_divergence': vol_price_divergence,
                
                f'{tf}_price_position': price_position,
                f'{tf}_trend_strength': trend_strength,
                
                f'{tf}_rsi_7': rsi_7.iloc[-1],
                f'{tf}_rsi_14': rsi_14.iloc[-1],
                f'{tf}_rsi_21': rsi_21.iloc[-1],
                f'{tf}_rsi_overbought': rsi_overbought,
                f'{tf}_rsi_oversold': rsi_oversold,
                f'{tf}_rsi_trend': rsi_trend,
                
                f'{tf}_roc_5': roc_5.iloc[-1],
                f'{tf}_roc_10': roc_10.iloc[-1],
                f'{tf}_roc_20': roc_20.iloc[-1],
                
                f'{tf}_mom_10': mom_10.iloc[-1],
                f'{tf}_mom_20': mom_20.iloc[-1],
                
                f'{tf}_williams_r': williams_r.iloc[-1],
                
                f'{tf}_cci': cci.iloc[-1],
                f'{tf}_cci_overbought': cci_overbought,
                f'{tf}_cci_oversold': cci_oversold,
                
                f'{tf}_adx': adx.iloc[-1],
                f'{tf}_plus_di': plus_di.iloc[-1],
                f'{tf}_minus_di': minus_di.iloc[-1],
                f'{tf}_adx_strong_trend': adx_strong_trend,
                f'{tf}_adx_weak_trend': adx_weak_trend,
                f'{tf}_trend_bullish': trend_bullish,
                
                f'{tf}_stoch_rsi_k': stoch_rsi_k.iloc[-1],
                f'{tf}_stoch_rsi_d': stoch_rsi_d.iloc[-1],
            }
            
            # 清洗特征值
            for key, value in tf_features.items():
                if pd.isna(value) or np.isinf(value):
                    tf_features[key] = 0
            
            features.update(tf_features)
        
        # 多周期共振特征
        golden_count = sum([features.get(f'{tf}_kdj_golden', 0) + features.get(f'{tf}_macd_golden', 0) for tf in self.timeframes])
        death_count = sum([features.get(f'{tf}_kdj_death', 0) + features.get(f'{tf}_macd_death', 0) for tf in self.timeframes])
        
        features['multi_tf_golden_count'] = golden_count
        features['multi_tf_death_count'] = death_count
        features['signal_strength'] = golden_count - death_count
        
        # 时间特征
        now = datetime.now()
        features['hour'] = now.hour
        features['day_of_week'] = now.weekday()
        features['is_weekend'] = 1 if now.weekday() >= 5 else 0
        
        return pd.Series(features)


class PredictionServer:
    """预测服务器"""
    
    # 涨跌区间定义
    CHANGE_RANGES = {
        'large_drop': {'min': -np.inf, 'max': -2, 'label': '大跌 (< -2%)', 'emoji': '🔴🔴'},
        'small_drop': {'min': -2, 'max': -0.5, 'label': '小跌 (-2% ~ -0.5%)', 'emoji': '🔴'},
        'sideways': {'min': -0.5, 'max': 0.5, 'label': '横盘 (-0.5% ~ 0.5%)', 'emoji': '⚖️'},
        'small_rise': {'min': 0.5, 'max': 2, 'label': '小涨 (0.5% ~ 2%)', 'emoji': '🟢'},
        'large_rise': {'min': 2, 'max': np.inf, 'label': '大涨 (> 2%)', 'emoji': '🟢🟢'},
    }
    
    def __init__(self, model_path: str, symbol: str = 'BTCUSDT',
                 telegram_token: str = None, telegram_chat_id: str = None,
                 risk_level: str = 'moderate'):
        """
        初始化预测服务器
        
        Args:
            model_path: 模型文件路径 (.pkl)
            symbol: 交易对
            telegram_token: Telegram Bot Token
            telegram_chat_id: Telegram Chat ID
            risk_level: 风险等级 ('conservative', 'moderate', 'aggressive')
        """
        self.symbol = symbol
        self.model_path = model_path
        
        # 加载模型
        self.model, self.model_info = self._load_model(model_path)
        
        # 初始化组件
        self.fetcher = BinanceDataFetcher(symbol)
        self.feature_extractor = FeatureExtractor()
        
        # Telegram 通知器
        self.notifier = None
        if telegram_token and telegram_chat_id:
            self.notifier = TelegramNotifier(telegram_token, telegram_chat_id)
            logger.info("✅ Telegram 通知已启用")
        else:
            logger.warning("⚠️ Telegram 未配置，仅输出到控制台")
        
        # 仓位管理器
        if POSITION_MANAGER_AVAILABLE:
            self.position_manager = PositionManager(risk_level=risk_level)
            logger.info(f"✅ 仓位管理器已启用 (风险等级: {risk_level})")
        else:
            self.position_manager = None
            logger.warning("⚠️ 仓位管理器不可用")
        
        # 预测历史
        self.prediction_history = []
        
    def _load_model(self, model_path: str) -> Tuple:
        """加载模型"""
        logger.info(f"📦 加载模型: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        feature_names = None
        
        # 兼容不同的保存格式
        if isinstance(model_data, dict):
            # 如果是字典格式 (包含多个模型)
            if 'best_model' in model_data:
                model = model_data['best_model']
                feature_names = model_data.get('feature_names', None)
                model_info = {'type': type(model).__name__, 'source': 'dict', 'feature_names': feature_names}
            elif 'gbm' in model_data:
                model = model_data['gbm']
                feature_names = model_data.get('feature_names', None)
                model_info = {'type': 'gbm', 'source': 'dict', 'feature_names': feature_names}
            elif 'model' in model_data:
                model = model_data['model']
                feature_names = model_data.get('feature_names', None)
                model_info = model_data.get('info', {})
                model_info['feature_names'] = feature_names
            else:
                # 取第一个可用的模型
                for key, value in model_data.items():
                    if hasattr(value, 'predict'):
                        model = value
                        feature_names = model_data.get('feature_names', None)
                        model_info = {'type': key, 'source': 'dict', 'feature_names': feature_names}
                        break
                else:
                    raise ValueError("无法从字典中找到有效模型")
        else:
            # 直接是模型对象
            model = model_data
            model_info = {'type': type(model).__name__, 'source': 'direct', 'feature_names': None}
        
        logger.info(f"   模型类型: {type(model).__name__}")
        if feature_names:
            logger.info(f"   特征数量: {len(feature_names)}")
        
        return model, model_info
    
    def _classify_prediction(self, pred_pct: float) -> dict:
        """根据预测值分类"""
        for range_name, range_info in self.CHANGE_RANGES.items():
            if range_info['min'] < pred_pct <= range_info['max']:
                return {
                    'range_name': range_name,
                    'label': range_info['label'],
                    'emoji': range_info['emoji']
                }
        return {'range_name': 'unknown', 'label': '未知', 'emoji': '❓'}
    
    def _calculate_confidence(self, features: pd.Series, prediction: float) -> float:
        """
        计算预测置信度
        
        基于多个因素：
        1. 多周期信号一致性
        2. 趋势强度
        3. ADX 趋势确认
        """
        confidence_factors = []
        
        # 1. 信号强度一致性 (0-1)
        signal_strength = abs(features.get('signal_strength', 0))
        max_signal = 12  # 最大可能信号数
        signal_confidence = min(signal_strength / max_signal, 1.0)
        confidence_factors.append(signal_confidence * 0.3)
        
        # 2. 预测方向与信号方向一致性
        signal = features.get('signal_strength', 0)
        if (prediction > 0 and signal > 0) or (prediction < 0 and signal < 0):
            direction_confidence = 0.3
        elif signal == 0:
            direction_confidence = 0.1
        else:
            direction_confidence = 0.0
        confidence_factors.append(direction_confidence)
        
        # 3. ADX 趋势强度
        adx_1h = features.get('1h_adx', 25)
        if adx_1h > 25:
            adx_confidence = min((adx_1h - 25) / 50, 1.0) * 0.2
        else:
            adx_confidence = 0.05
        confidence_factors.append(adx_confidence)
        
        # 4. RSI 不处于极端区域 (更可靠)
        rsi_1h = features.get('1h_rsi_14', 50)
        if 30 < rsi_1h < 70:
            rsi_confidence = 0.1
        else:
            rsi_confidence = 0.05
        confidence_factors.append(rsi_confidence)
        
        # 5. 预测幅度合理性 (极端预测可能不可靠)
        pred_abs = abs(prediction)
        if pred_abs < 5:
            magnitude_confidence = 0.1
        else:
            magnitude_confidence = max(0.1 - (pred_abs - 5) * 0.01, 0)
        confidence_factors.append(magnitude_confidence)
        
        # 总置信度 (归一化到 0-100)
        total_confidence = sum(confidence_factors) * 100
        return min(max(total_confidence, 10), 90)  # 限制在 10-90 之间
    
    def predict(self) -> dict:
        """执行一次预测"""
        try:
            logger.info("🔮 开始预测...")
            
            # 1. 获取数据
            logger.info("   📊 获取市场数据...")
            df_dict = self.fetcher.get_all_timeframes(limit=200)
            current_price = self.fetcher.get_current_price()
            funding = self.fetcher.get_funding_rate()
            
            # 2. 提取特征
            logger.info("   🔧 提取特征...")
            features = self.feature_extractor.extract_features(df_dict)
            features['funding_rate'] = funding['funding_rate']
            features['mark_price'] = funding['mark_price']
            features['index_price'] = funding['index_price']
            
            # 3. 准备模型输入
            feature_df = pd.DataFrame([features])
            
            # 移除非特征列
            exclude_cols = ['timestamp', 'base_timestamp', 'target_regression', 
                           'target_classification', 'target_direction', 'close_price']
            model_features = feature_df.drop(columns=[c for c in exclude_cols if c in feature_df.columns], errors='ignore')
            
            # 确保特征对齐（处理模型训练时的特征列）
            expected_features = None
            
            # 优先使用模型信息中保存的特征列表
            if self.model_info.get('feature_names'):
                expected_features = self.model_info['feature_names']
            # 其次使用模型的 feature_names_in_ 属性
            elif hasattr(self.model, 'feature_names_in_'):
                expected_features = self.model.feature_names_in_
            
            if expected_features:
                # 使用字典构建，然后一次性创建 DataFrame（更高效）
                aligned_data = {}
                missing_features = []
                
                for col in expected_features:
                    if col in model_features.columns:
                        aligned_data[col] = model_features[col].values[0]
                    else:
                        missing_features.append(col)
                        aligned_data[col] = 0.0
                
                if missing_features:
                    logger.warning(f"   ⚠️ {len(missing_features)} 个特征缺失，已填充为 0")
                
                model_features = pd.DataFrame([aligned_data])
            else:
                logger.warning("   ⚠️ 未找到特征名称列表，使用所有提取的特征")
            
            # 处理 NaN/Inf
            model_features = model_features.replace([np.inf, -np.inf], 0).fillna(0)
            
            # 确保数据类型正确
            model_features = model_features.astype(float)
            
            # 4. 预测
            logger.info("   🎯 模型预测...")
            prediction = self.model.predict(model_features)[0]
            
            # 5. 分类预测结果
            classification = self._classify_prediction(prediction)
            
            # 6. 计算置信度
            confidence = self._calculate_confidence(features, prediction)
            
            # 7. 确定方向
            if prediction > 0.5:
                direction = '看涨'
                direction_emoji = '📈'
            elif prediction < -0.5:
                direction = '看跌'
                direction_emoji = '📉'
            else:
                direction = '震荡'
                direction_emoji = '↔️'
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'current_price': current_price,
                'prediction_pct': prediction,
                'direction': direction,
                'direction_emoji': direction_emoji,
                'range': classification,
                'confidence': confidence,
                'funding_rate': funding['funding_rate'],
                'features_summary': {
                    'signal_strength': features.get('signal_strength', 0),
                    'rsi_1h': features.get('1h_rsi_14', 50),
                    'adx_1h': features.get('1h_adx', 25),
                    'volatility_1h': features.get('1h_volatility', 0),
                }
            }
            
            # 保存到历史
            self.prediction_history.append(result)
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            logger.info(f"   ✅ 预测完成: {direction} ({prediction:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def format_report(self, result: dict = None) -> str:
        """
        格式化预测报告 (HTML 格式用于 Telegram)
        
        Args:
            result: 预测结果字典，如果为 None 则使用最新的预测结果
        """
        if result is None:
            # 如果没有提供结果，使用最新的预测历史
            if self.prediction_history:
                result = self.prediction_history[-1]
            else:
                return "❌ 预测失败，请检查日志"
        
        if not isinstance(result, dict):
            return "❌ 预测失败，请检查日志"
        
        # 计算预测目标时间 (20小时后)
        prediction_time = datetime.now() + timedelta(hours=20)
        
        # 计算预测价格
        current_price = result['current_price']
        prediction_pct = result['prediction_pct']
        predicted_price = current_price * (1 + prediction_pct / 100)
        
        # UTC+8 时区
        from datetime import timezone
        utc8 = timezone(timedelta(hours=8))
        now_utc8 = datetime.now(utc8)
        # 将 prediction_time 转换为 UTC+8
        if prediction_time.tzinfo is None:
            prediction_time = prediction_time.replace(tzinfo=timezone.utc)
        prediction_time_utc8 = prediction_time.astimezone(utc8)
        
        report = f"""
<b>🔮 BTC 价格预测报告</b>
<code>━━━━━━━━━━━━━━━━━━━━━━━━</code>

📅 <b>预测时间:</b> {now_utc8.strftime('%Y-%m-%d %H:%M')} (UTC+8)
🎯 <b>预测目标:</b> {prediction_time_utc8.strftime('%Y-%m-%d %H:%M')} (UTC+8, 20h后)

<b>💰 当前价格:</b> ${current_price:,.2f}
<b>🎯 预测价格:</b> ${predicted_price:,.2f}
<b>💵 资金费率:</b> {result['funding_rate']*100:.4f}%

<code>━━━━━━━━━━━━━━━━━━━━━━━━</code>
<b>📊 预测结果</b>

{result['direction_emoji']} <b>方向:</b> {result['direction']}
{result['range']['emoji']} <b>区间:</b> {result['range']['label']}
📈 <b>预测涨跌:</b> {result['prediction_pct']:+.2f}%
🎯 <b>置信度:</b> {result['confidence']:.0f}%

<code>━━━━━━━━━━━━━━━━━━━━━━━━</code>
<b>📈 市场状态</b>

📊 信号强度: {result['features_summary']['signal_strength']}
📉 RSI(1h): {result['features_summary']['rsi_1h']:.1f}
📈 ADX(1h): {result['features_summary']['adx_1h']:.1f}
⚡ 波动率: {result['features_summary']['volatility_1h']*100:.2f}%
"""
        
        # 添加仓位建议（如果可用）
        if self.position_manager:
            position_info = self.position_manager.calculate_position_size(
                signal_strength=result['features_summary']['signal_strength'],
                confidence=result['confidence'],
                prediction_pct=result['prediction_pct']
            )
            position_recommendation = self.position_manager.format_recommendation(
                position_info, result['direction']
            )
            report += f"""
<code>━━━━━━━━━━━━━━━━━━━━━━━━</code>
{position_recommendation}
"""
        
        report += """
<code>━━━━━━━━━━━━━━━━━━━━━━━━</code>
<i>⚠️ 仅供参考，不构成投资建议</i>
"""
        return report
    
    def send_prediction_report(self):
        """执行预测并发送报告"""
        logger.info("=" * 50)
        logger.info("⏰ 定时预测任务开始")
        
        # 执行预测
        result = self.predict()
        
        # 格式化报告
        report = self.format_report(result)
        
        # 输出到控制台
        print("\n" + "=" * 50)
        print(report.replace('<b>', '').replace('</b>', '')
              .replace('<code>', '').replace('</code>', '')
              .replace('<i>', '').replace('</i>', ''))
        print("=" * 50 + "\n")
        
        # 发送到 Telegram
        if self.notifier:
            self.notifier.send_message(report)
        
        return result
    
    def run(self, test_mode: bool = False):
        """
        运行预测服务器
        
        Args:
            test_mode: 如果为 True，立即执行一次预测然后退出
        """
        logger.info("=" * 50)
        logger.info("🚀 BTC 预测服务器启动")
        logger.info(f"   交易对: {self.symbol}")
        logger.info(f"   模型: {self.model_path}")
        logger.info("=" * 50)
        
        if test_mode:
            # 测试模式：立即预测一次
            logger.info("📋 测试模式：执行一次预测")
            self.send_prediction_report()
            return
        
        # 立即执行一次
        logger.info("📋 首次预测...")
        self.send_prediction_report()
        
        # 设置定时任务：每30分钟执行一次
        schedule.every(30).minutes.do(self.send_prediction_report)
        
        logger.info("⏰ 定时任务已设置：每30分钟预测一次")
        logger.info("🔄 等待下一个30分钟间隔...")
        
        # 主循环
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n👋 服务器已停止")


def main():
    parser = argparse.ArgumentParser(description='BTC 价格预测服务器')
    parser.add_argument('--model', type=str, required=True, 
                        help='模型文件路径 (.pkl)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='交易对 (默认: BTCUSDT)')
    parser.add_argument('--telegram-token', type=str, default=None,
                        help='Telegram Bot Token (或设置环境变量 TELEGRAM_BOT_TOKEN)')
    parser.add_argument('--telegram-chat-id', type=str, default=None,
                        help='Telegram Chat ID (或设置环境变量 TELEGRAM_CHAT_ID)')
    parser.add_argument('--test', action='store_true',
                        help='测试模式：执行一次预测后退出')
    parser.add_argument('--risk-level', type=str, 
                        choices=['conservative', 'moderate', 'aggressive'],
                        default='moderate',
                        help='风险等级 (默认: moderate)')
    
    args = parser.parse_args()
    
    # 从环境变量获取 Telegram 配置
    telegram_token = args.telegram_token or os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = args.telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID')
    
    # 创建服务器
    server = PredictionServer(
        model_path=args.model,
        symbol=args.symbol,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        risk_level=args.risk_level
    )
    
    # 运行服务器
    server.run(test_mode=args.test)


if __name__ == "__main__":
    main()

