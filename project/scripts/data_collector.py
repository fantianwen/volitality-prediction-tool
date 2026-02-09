"""
BTC 价格变动预测 - 数据收集与特征提取

基于 KDJ/MACD 多时间周期交叉信号预测价格变动

功能:
1. 收集多时间周期 K 线数据 (5m, 15m, 30m, 1h, 4h, 1d)
2. 计算技术指标 (KDJ, MACD, RSI, ROC, MOM, Williams %R, CCI, ADX, Stochastic RSI)
3. 检测金叉/死叉信号
4. 提取市场状态特征
5. 生成价格变动标签

使用方法:
    python data_collector.py --symbol BTCUSDT --output ../data
"""

import pandas as pd
import numpy as np
import json
import time
import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional
import urllib.request
import ssl


class BinanceDataFetcher:
    """Binance 数据获取器"""
    
    def __init__(self, symbol: str = 'BTCUSDT'):
        self.symbol = symbol.upper()
        self.base_url = "https://fapi.binance.com"
        
        # SSL 配置（禁用证书验证以解决某些网络环境的问题）
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
    
    def get_klines(self, interval: str, limit: int = 500, 
                    start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取 K 线数据
        
        Args:
            interval: 时间周期 (5m, 15m, 1h, 4h, 1d)
            limit: 获取的 K 线数量 (单次最大 1500)
            start_time: 开始时间戳 (毫秒)
            end_time: 结束时间戳 (毫秒)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": min(limit, 1500)  # Binance 单次最大 1500
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = self._request("/fapi/v1/klines", params)
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # 转换数据类型
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
    
    def get_klines_historical(self, interval: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        获取历史 K 线数据（自动分批获取）
        
        Args:
            interval: 时间周期 (5m, 15m, 1h, 4h, 1d)
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)，默认为当前时间
            
        Returns:
            完整的历史 K 线 DataFrame
        """
        from datetime import datetime, timedelta
        
        # 解析日期
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        
        # 计算每个周期的毫秒数
        interval_ms = {
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        
        ms_per_candle = interval_ms.get(interval, 60 * 60 * 1000)
        batch_size = 1000  # 每批获取的 K 线数
        batch_ms = batch_size * ms_per_candle
        
        all_data = []
        current_start = start_ms
        
        while current_start < end_ms:
            current_end = min(current_start + batch_ms, end_ms)
            
            df = self.get_klines(interval, limit=batch_size, 
                                start_time=current_start, end_time=current_end)
            
            if len(df) == 0:
                break
                
            all_data.append(df)
            
            # 下一批的开始时间
            last_timestamp = df['timestamp'].iloc[-1]
            current_start = int(last_timestamp.timestamp() * 1000) + ms_per_candle
            
            time.sleep(0.1)  # 避免请求过快
        
        if not all_data:
            return pd.DataFrame()
        
        # 合并所有数据
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        return result
    
    def get_all_timeframes(self, limit: int = 500) -> Dict[str, pd.DataFrame]:
        """获取所有时间周期的数据"""
        data_dict = {}
        for tf in self.timeframes:
            print(f"  📊 获取 {tf} K 线数据...")
            data_dict[tf] = self.get_klines(tf, limit)
            time.sleep(0.2)  # 避免请求过快
        return data_dict
    
    def get_all_timeframes_historical(self, start_date: str, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """获取所有时间周期的历史数据"""
        data_dict = {}
        for tf in self.timeframes:
            print(f"  📊 获取 {tf} 历史 K 线数据 ({start_date} ~ {end_date or '现在'})...")
            data_dict[tf] = self.get_klines_historical(tf, start_date, end_date)
            print(f"      获取到 {len(data_dict[tf])} 根 K 线")
            time.sleep(0.3)
        return data_dict


class TechnicalIndicators:
    """技术指标计算器（不依赖 TA-Lib）"""
    
    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """计算 EMA"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """计算 SMA"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> tuple:
        """
        计算 KDJ 指标
        
        Args:
            df: 包含 high, low, close 的 DataFrame
            n: RSV 周期
            m1: K 值平滑周期
            m2: D 值平滑周期
            
        Returns:
            (K, D, J) Series tuple
        """
        low_n = df['low'].rolling(window=n).min()
        high_n = df['high'].rolling(window=n).max()
        
        # RSV = (Close - Low_n) / (High_n - Low_n) * 100
        rsv = (df['close'] - low_n) / (high_n - low_n) * 100
        rsv = rsv.fillna(50)  # 处理 NaN
        
        # K = SMA(RSV, m1)
        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        
        # D = SMA(K, m2)
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        
        # J = 3K - 2D
        j = 3 * k - 2 * d
        
        return k, d, j
    
    @staticmethod
    def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """
        计算 MACD 指标
        
        Args:
            close: 收盘价 Series
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            (MACD, Signal, Histogram) Series tuple
        """
        ema_fast = TechnicalIndicators.calculate_ema(close, fast)
        ema_slow = TechnicalIndicators.calculate_ema(close, slow)
        
        macd = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd, signal)
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    @staticmethod
    def detect_crossover(fast: pd.Series, slow: pd.Series) -> tuple:
        """
        检测金叉/死叉
        
        Returns:
            (golden_cross, death_cross) bool Series tuple
        """
        # 金叉：快线从下往上穿越慢线
        golden = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        
        # 死叉：快线从上往下穿越慢线
        death = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        
        return golden, death
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        计算 RSI (Relative Strength Index) 指标
        
        Args:
            close: 收盘价 Series
            period: RSI 周期 (默认 14)
            
        Returns:
            RSI Series (0-100)
        """
        # 计算价格变动
        delta = close.diff()
        
        # 分离涨跌
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # 计算平均涨跌 (使用 EMA)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        # 计算 RS
        rs = avg_gain / avg_loss
        
        # 计算 RSI
        rsi = 100 - (100 / (1 + rs))
        
        # 处理除零情况
        rsi = rsi.fillna(50)
        rsi = rsi.replace([np.inf, -np.inf], 50)
        
        return rsi
    
    @staticmethod
    def calculate_roc(close: pd.Series, period: int = 10) -> pd.Series:
        """
        计算 ROC (Rate of Change) - 价格变化率
        
        Args:
            close: 收盘价 Series
            period: 计算周期
            
        Returns:
            ROC Series (百分比)
        """
        roc = (close - close.shift(period)) / close.shift(period) * 100
        return roc.fillna(0)
    
    @staticmethod
    def calculate_momentum(close: pd.Series, period: int = 10) -> pd.Series:
        """
        计算 MOM (Momentum) - 动量
        
        Args:
            close: 收盘价 Series
            period: 计算周期
            
        Returns:
            Momentum Series
        """
        mom = close - close.shift(period)
        return mom.fillna(0)
    
    @staticmethod
    def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算 Williams %R - 威廉指标
        
        Args:
            df: 包含 high, low, close 的 DataFrame
            period: 计算周期
            
        Returns:
            Williams %R Series (-100 to 0)
        """
        high_n = df['high'].rolling(window=period).max()
        low_n = df['low'].rolling(window=period).min()
        
        wr = (high_n - df['close']) / (high_n - low_n) * -100
        return wr.fillna(-50)
    
    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        计算 CCI (Commodity Channel Index) - 商品通道指数
        
        Args:
            df: 包含 high, low, close 的 DataFrame
            period: 计算周期
            
        Returns:
            CCI Series
        """
        # 典型价格
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # CCI = (TP - SMA(TP)) / (0.015 * MAD)
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        cci = (tp - sma_tp) / (0.015 * mad)
        cci = cci.fillna(0)
        cci = cci.replace([np.inf, -np.inf], 0)
        
        return cci
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> tuple:
        """
        计算 ADX (Average Directional Index) - 平均趋向指数
        
        Args:
            df: 包含 high, low, close 的 DataFrame
            period: 计算周期
            
        Returns:
            (ADX, +DI, -DI) Series tuple
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Smoothed TR and DM
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        # 处理异常值
        adx = adx.fillna(25)
        plus_di = plus_di.fillna(25)
        minus_di = minus_di.fillna(25)
        
        adx = adx.replace([np.inf, -np.inf], 25)
        plus_di = plus_di.replace([np.inf, -np.inf], 25)
        minus_di = minus_di.replace([np.inf, -np.inf], 25)
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def calculate_stoch_rsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> tuple:
        """
        计算 Stochastic RSI - 随机 RSI
        
        Args:
            close: 收盘价 Series
            rsi_period: RSI 周期
            stoch_period: 随机指标周期
            
        Returns:
            (StochRSI_K, StochRSI_D) Series tuple
        """
        rsi = TechnicalIndicators.calculate_rsi(close, rsi_period)
        
        rsi_low = rsi.rolling(window=stoch_period).min()
        rsi_high = rsi.rolling(window=stoch_period).max()
        
        stoch_rsi_k = (rsi - rsi_low) / (rsi_high - rsi_low) * 100
        stoch_rsi_d = stoch_rsi_k.rolling(window=3).mean()
        
        stoch_rsi_k = stoch_rsi_k.fillna(50)
        stoch_rsi_d = stoch_rsi_d.fillna(50)
        
        return stoch_rsi_k, stoch_rsi_d


def calculate_vzo(df: pd.DataFrame, period: int = 14, ma_len: int = 9) -> tuple:
    """
    Volume Zone Oscillator.
    Returns (vzo, vzo_ma) Series.
    """
    signed_vol = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
    vp = pd.Series(signed_vol, index=df.index).ewm(span=period, adjust=False).mean()
    tv = df['volume'].ewm(span=period, adjust=False).mean()
    vzo = (100 * vp / tv).fillna(0).replace([np.inf, -np.inf], 0)
    vzo_ma = vzo.ewm(span=ma_len, adjust=False).mean()
    return vzo, vzo_ma


def calculate_vzo_slope(vzo: pd.Series, lookback: int = 5) -> pd.Series:
    """Linear-regression slope of VZO over *lookback* periods."""
    x = np.arange(lookback, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _lr_slope(window):
        if len(window) < lookback:
            return np.nan
        y = window.values
        return ((x - x_mean) * (y - y.mean())).sum() / x_var

    return vzo.rolling(window=lookback).apply(_lr_slope, raw=False).fillna(0)


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self):
        self.timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        self.indicators = TechnicalIndicators()
    
    def extract_features(self, df_dict: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        从多时间周期数据中提取特征
        
        Args:
            df_dict: {'5m': df_5m, '15m': df_15m, ...}
            
        Returns:
            特征 Series
        """
        features = {}
        
        for tf in self.timeframes:
            if tf not in df_dict or len(df_dict[tf]) < 50:
                continue
                
            df = df_dict[tf]
            
            # 1. 计算 KDJ
            k, d, j = self.indicators.calculate_kdj(df)
            
            # 2. 计算 MACD
            macd, signal, hist = self.indicators.calculate_macd(df['close'])
            
            # 3. 计算 RSI (多周期)
            rsi_7 = self.indicators.calculate_rsi(df['close'], period=7)
            rsi_14 = self.indicators.calculate_rsi(df['close'], period=14)
            rsi_21 = self.indicators.calculate_rsi(df['close'], period=21)
            
            # 4. 检测交叉信号
            kdj_golden, kdj_death = self.indicators.detect_crossover(k, d)
            macd_golden, macd_death = self.indicators.detect_crossover(macd, signal)
            
            # 5. 计算波动率 (年化)
            returns = df['close'].pct_change()
            periods_per_year = {'5m': 288*365, '15m': 96*365, '30m': 48*365, '1h': 24*365, '4h': 6*365, '1d': 365}
            volatility = returns.std() * np.sqrt(periods_per_year.get(tf, 365))
            
            # 6. 成交量特征 (Volume Features)
            volume = df['volume']
            volume_ma5 = volume.rolling(5).mean()
            volume_ma10 = volume.rolling(10).mean()
            volume_ma20 = volume.rolling(20).mean()
            
            # 成交量比率 (当前成交量 / MA)
            vol_ratio_ma5 = volume.iloc[-1] / volume_ma5.iloc[-1] if volume_ma5.iloc[-1] > 0 else 1
            vol_ratio_ma10 = volume.iloc[-1] / volume_ma10.iloc[-1] if volume_ma10.iloc[-1] > 0 else 1
            vol_ratio_ma20 = volume.iloc[-1] / volume_ma20.iloc[-1] if volume_ma20.iloc[-1] > 0 else 1
            
            # 成交量变化率
            vol_change_1 = (volume.iloc[-1] - volume.iloc[-2]) / volume.iloc[-2] * 100 if volume.iloc[-2] > 0 else 0
            vol_change_5 = (volume.iloc[-1] - volume.iloc[-6]) / volume.iloc[-6] * 100 if len(volume) > 5 and volume.iloc[-6] > 0 else 0
            
            # 成交量趋势 (MA5 vs MA20)
            vol_trend = (volume_ma5.iloc[-1] - volume_ma20.iloc[-1]) / volume_ma20.iloc[-1] * 100 if volume_ma20.iloc[-1] > 0 else 0
            
            # 成交量位置 (相对于最近20根K线的高低点)
            vol_high_20 = volume.tail(20).max()
            vol_low_20 = volume.tail(20).min()
            vol_position = (volume.iloc[-1] - vol_low_20) / (vol_high_20 - vol_low_20) if vol_high_20 > vol_low_20 else 0.5
            
            # 成交量放大信号 (是否超过2倍MA)
            vol_spike = 1 if vol_ratio_ma20 > 2 else 0
            
            # 成交量萎缩信号 (是否低于0.5倍MA)
            vol_shrink = 1 if vol_ratio_ma20 < 0.5 else 0
            
            # 量价背离检测
            price_up = 1 if df['close'].iloc[-1] > df['close'].iloc[-2] else 0
            vol_up = 1 if volume.iloc[-1] > volume.iloc[-2] else 0
            vol_price_divergence = 1 if price_up != vol_up else 0  # 价涨量跌 或 价跌量涨
            
            # 7. 价格位置 (相对于最近 N 根 K 线)
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            price_position = (df['close'].iloc[-1] - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
            
            # 8. 趋势强度 (价格与均线的偏离)
            ma20 = df['close'].rolling(20).mean()
            trend_strength = (df['close'].iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1] * 100 if ma20.iloc[-1] > 0 else 0
            
            # 9. RSI 衍生特征
            rsi_14_value = rsi_14.iloc[-1]
            rsi_overbought = 1 if rsi_14_value > 70 else 0      # 超买
            rsi_oversold = 1 if rsi_14_value < 30 else 0        # 超卖
            rsi_trend = rsi_14.iloc[-1] - rsi_14.iloc[-5] if len(rsi_14) > 5 else 0  # RSI 趋势
            
            # 10. 额外动量指标
            # ROC - 价格变化率
            roc_5 = self.indicators.calculate_roc(df['close'], period=5)
            roc_10 = self.indicators.calculate_roc(df['close'], period=10)
            roc_20 = self.indicators.calculate_roc(df['close'], period=20)
            
            # MOM - 动量
            mom_10 = self.indicators.calculate_momentum(df['close'], period=10)
            mom_20 = self.indicators.calculate_momentum(df['close'], period=20)
            
            # Williams %R
            williams_r = self.indicators.calculate_williams_r(df, period=14)
            
            # CCI - 商品通道指数
            cci = self.indicators.calculate_cci(df, period=20)
            
            # ADX - 平均趋向指数
            adx, plus_di, minus_di = self.indicators.calculate_adx(df, period=14)
            
            # Stochastic RSI
            stoch_rsi_k, stoch_rsi_d = self.indicators.calculate_stoch_rsi(df['close'])
            
            # 动量指标衍生特征
            cci_value = cci.iloc[-1]
            cci_overbought = 1 if cci_value > 100 else 0
            cci_oversold = 1 if cci_value < -100 else 0
            
            adx_value = adx.iloc[-1]
            adx_strong_trend = 1 if adx_value > 25 else 0  # 强趋势
            adx_weak_trend = 1 if adx_value < 20 else 0    # 弱趋势/盘整
            
            # 趋势方向 (+DI vs -DI)
            trend_bullish = 1 if plus_di.iloc[-1] > minus_di.iloc[-1] else 0
            
            # 11. VZO & VZO Slope features (multi-timeframe)
            vzo, vzo_ma = calculate_vzo(df, period=14, ma_len=9)
            vzo_slope = calculate_vzo_slope(vzo, lookback=5)
            slope_accel = vzo_slope.diff().fillna(0)  # 2nd derivative
            
            # Per-timeframe rolling z-score normalization (window=20)
            vzo_roll_mean = vzo.rolling(20).mean().bfill()
            vzo_roll_std = vzo.rolling(20).std().fillna(1).replace(0, 1)
            vzo_zscore = ((vzo - vzo_roll_mean) / vzo_roll_std).fillna(0).replace([np.inf, -np.inf], 0)
            
            slope_roll_mean = vzo_slope.rolling(20).mean().bfill()
            slope_roll_std = vzo_slope.rolling(20).std().fillna(1).replace(0, 1)
            slope_zscore = ((vzo_slope - slope_roll_mean) / slope_roll_std).fillna(0).replace([np.inf, -np.inf], 0)
            
            # VZO zone classification: bullish (>40), bearish (<-40), neutral
            vzo_last = vzo.iloc[-1]
            if vzo_last > 40:
                vzo_zone = 2  # strong bullish
            elif vzo_last > 15:
                vzo_zone = 1  # bullish
            elif vzo_last > -15:
                vzo_zone = 0  # neutral
            elif vzo_last > -40:
                vzo_zone = -1  # bearish
            else:
                vzo_zone = -2  # strong bearish
            
            # Slope direction sign
            slope_last = vzo_slope.iloc[-1]
            slope_sign = 1 if slope_last > 0 else (-1 if slope_last < 0 else 0)
            
            # 构建特征
            tf_features = {
                # KDJ 指标
                f'{tf}_kdj_k': k.iloc[-1],
                f'{tf}_kdj_d': d.iloc[-1],
                f'{tf}_kdj_j': j.iloc[-1],
                f'{tf}_kdj_golden': int(kdj_golden.iloc[-1]) if not pd.isna(kdj_golden.iloc[-1]) else 0,
                f'{tf}_kdj_death': int(kdj_death.iloc[-1]) if not pd.isna(kdj_death.iloc[-1]) else 0,
                
                # MACD 指标
                f'{tf}_macd': macd.iloc[-1],
                f'{tf}_macd_signal': signal.iloc[-1],
                f'{tf}_macd_hist': hist.iloc[-1],
                f'{tf}_macd_golden': int(macd_golden.iloc[-1]) if not pd.isna(macd_golden.iloc[-1]) else 0,
                f'{tf}_macd_death': int(macd_death.iloc[-1]) if not pd.isna(macd_death.iloc[-1]) else 0,
                
                # 波动率
                f'{tf}_volatility': volatility,
                
                # 成交量特征 (Volume Features)
                f'{tf}_vol_ratio_ma5': vol_ratio_ma5,      # 成交量/MA5
                f'{tf}_vol_ratio_ma10': vol_ratio_ma10,    # 成交量/MA10
                f'{tf}_vol_ratio_ma20': vol_ratio_ma20,    # 成交量/MA20
                f'{tf}_vol_change_1': vol_change_1,        # 1根K线成交量变化%
                f'{tf}_vol_change_5': vol_change_5,        # 5根K线成交量变化%
                f'{tf}_vol_trend': vol_trend,              # 成交量趋势 (MA5 vs MA20)
                f'{tf}_vol_position': vol_position,        # 成交量位置 (0-1)
                f'{tf}_vol_spike': vol_spike,              # 放量信号
                f'{tf}_vol_shrink': vol_shrink,            # 缩量信号
                f'{tf}_vol_price_divergence': vol_price_divergence,  # 量价背离
                
                # 价格状态
                f'{tf}_price_position': price_position,
                f'{tf}_trend_strength': trend_strength,
                
                # RSI 指标
                f'{tf}_rsi_7': rsi_7.iloc[-1],              # RSI(7) - 短周期
                f'{tf}_rsi_14': rsi_14.iloc[-1],            # RSI(14) - 标准周期
                f'{tf}_rsi_21': rsi_21.iloc[-1],            # RSI(21) - 长周期
                f'{tf}_rsi_overbought': rsi_overbought,     # 超买信号 (RSI > 70)
                f'{tf}_rsi_oversold': rsi_oversold,         # 超卖信号 (RSI < 30)
                f'{tf}_rsi_trend': rsi_trend,               # RSI 趋势变化
                
                # ROC - 价格变化率
                f'{tf}_roc_5': roc_5.iloc[-1],              # ROC(5)
                f'{tf}_roc_10': roc_10.iloc[-1],            # ROC(10)
                f'{tf}_roc_20': roc_20.iloc[-1],            # ROC(20)
                
                # MOM - 动量
                f'{tf}_mom_10': mom_10.iloc[-1],            # Momentum(10)
                f'{tf}_mom_20': mom_20.iloc[-1],            # Momentum(20)
                
                # Williams %R
                f'{tf}_williams_r': williams_r.iloc[-1],    # Williams %R (-100 to 0)
                
                # CCI - 商品通道指数
                f'{tf}_cci': cci.iloc[-1],                  # CCI 值
                f'{tf}_cci_overbought': cci_overbought,     # CCI 超买 (>100)
                f'{tf}_cci_oversold': cci_oversold,         # CCI 超卖 (<-100)
                
                # ADX - 平均趋向指数
                f'{tf}_adx': adx.iloc[-1],                  # ADX 值
                f'{tf}_plus_di': plus_di.iloc[-1],          # +DI
                f'{tf}_minus_di': minus_di.iloc[-1],        # -DI
                f'{tf}_adx_strong_trend': adx_strong_trend, # 强趋势信号 (ADX > 25)
                f'{tf}_adx_weak_trend': adx_weak_trend,     # 弱趋势信号 (ADX < 20)
                f'{tf}_trend_bullish': trend_bullish,       # 趋势看涨 (+DI > -DI)
                
                # Stochastic RSI
                f'{tf}_stoch_rsi_k': stoch_rsi_k.iloc[-1],  # Stoch RSI K
                f'{tf}_stoch_rsi_d': stoch_rsi_d.iloc[-1],  # Stoch RSI D
                
                # VZO & Slope features (with normalization)
                f'{tf}_vzo': vzo.iloc[-1],                   # Raw VZO value
                f'{tf}_vzo_ma': vzo_ma.iloc[-1],             # VZO moving average
                f'{tf}_vzo_slope': vzo_slope.iloc[-1],       # VZO slope (linear regression)
                f'{tf}_vzo_zscore': vzo_zscore.iloc[-1],     # VZO z-score (rolling 20-period)
                f'{tf}_slope_zscore': slope_zscore.iloc[-1], # Slope z-score (rolling 20-period)
                f'{tf}_slope_accel': slope_accel.iloc[-1],   # Slope acceleration (2nd derivative)
                f'{tf}_vzo_zone': vzo_zone,                  # VZO zone (-2 to +2)
                f'{tf}_slope_sign': slope_sign,              # Slope direction (+1/0/-1)
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
        features['signal_strength'] = golden_count - death_count  # 正数=看涨，负数=看跌
        
        # VZO 多周期共振特征
        vzo_tfs = ['5m', '15m', '30m', '1h', '4h']
        vzo_bullish_count = sum(1 for tf in vzo_tfs if features.get(f'{tf}_vzo_zone', 0) > 0)
        vzo_bearish_count = sum(1 for tf in vzo_tfs if features.get(f'{tf}_vzo_zone', 0) < 0)
        slope_bullish_count = sum(1 for tf in vzo_tfs if features.get(f'{tf}_slope_sign', 0) > 0)
        slope_bearish_count = sum(1 for tf in vzo_tfs if features.get(f'{tf}_slope_sign', 0) < 0)
        
        features['vzo_multi_tf_bullish'] = vzo_bullish_count       # 多少个TF的VZO看涨
        features['vzo_multi_tf_bearish'] = vzo_bearish_count       # 多少个TF的VZO看跌
        features['vzo_multi_tf_consensus'] = vzo_bullish_count - vzo_bearish_count  # VZO多周期共识
        features['slope_multi_tf_bullish'] = slope_bullish_count   # 多少个TF的slope看涨
        features['slope_multi_tf_bearish'] = slope_bearish_count   # 多少个TF的slope看跌
        features['slope_multi_tf_consensus'] = slope_bullish_count - slope_bearish_count  # Slope多周期共识
        
        # VZO/Slope 短期 vs 长期 divergence
        # Short-term (5m, 15m avg) vs long-term (1h, 4h avg) VZO
        short_vzo = np.mean([features.get(f'{tf}_vzo', 0) for tf in ['5m', '15m']])
        long_vzo = np.mean([features.get(f'{tf}_vzo', 0) for tf in ['1h', '4h']])
        features['vzo_short_long_diff'] = short_vzo - long_vzo    # 短期vs长期VZO差值
        
        short_slope = np.mean([features.get(f'{tf}_vzo_slope', 0) for tf in ['5m', '15m']])
        long_slope = np.mean([features.get(f'{tf}_vzo_slope', 0) for tf in ['1h', '4h']])
        features['slope_short_long_diff'] = short_slope - long_slope  # 短期vs长期slope差值
        
        # 时间特征
        now = datetime.now()
        features['hour'] = now.hour
        features['day_of_week'] = now.weekday()
        features['is_weekend'] = 1 if now.weekday() >= 5 else 0
        
        return pd.Series(features)


class LabelGenerator:
    """标签生成器"""
    
    @staticmethod
    def generate_labels(df: pd.DataFrame, lookforward_periods: int = 20) -> pd.DataFrame:
        """
        生成未来价格变动标签
        
        Args:
            df: 包含 close 的 DataFrame
            lookforward_periods: 预测未来多少根 K 线
            
        Returns:
            添加了标签的 DataFrame
        """
        df = df.copy()
        
        # 未来价格
        future_price = df['close'].shift(-lookforward_periods)
        current_price = df['close']
        
        # 价格变动百分比
        price_change_pct = (future_price - current_price) / current_price * 100
        
        # 回归目标
        df['target_regression'] = price_change_pct
        
        # 分类目标
        def classify_change(pct):
            if pd.isna(pct):
                return np.nan
            elif pct < -2:
                return 0  # 大跌
            elif pct < -0.5:
                return 1  # 小跌
            elif pct < 0.5:
                return 2  # 横盘
            elif pct < 2:
                return 3  # 小涨
            else:
                return 4  # 大涨
        
        df['target_classification'] = price_change_pct.apply(classify_change)
        
        # 方向目标 (简化版)
        df['target_direction'] = np.where(price_change_pct > 0, 1, np.where(price_change_pct < 0, -1, 0))
        
        return df


class DataCollector:
    """主数据收集器"""
    
    def __init__(self, symbol: str = 'BTCUSDT', output_dir: str = './data'):
        self.symbol = symbol
        self.output_dir = output_dir
        
        self.fetcher = BinanceDataFetcher(symbol)
        self.feature_extractor = FeatureExtractor()
        self.label_generator = LabelGenerator()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_snapshot(self) -> dict:
        """收集当前时刻的数据快照"""
        print(f"\n🚀 开始收集 {self.symbol} 数据快照...")
        print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 获取所有时间周期数据
        print("\n📊 获取 K 线数据...")
        df_dict = self.fetcher.get_all_timeframes(limit=200)
        
        # 2. 获取资金费率
        print("\n💰 获取资金费率...")
        funding = self.fetcher.get_funding_rate()
        print(f"   资金费率: {funding['funding_rate']*100:.4f}%")
        print(f"   标记价格: {funding['mark_price']:,.2f}")
        print(f"   指数价格: {funding['index_price']:,.2f}")
        
        # 3. 提取特征
        print("\n🔧 提取特征...")
        features = self.feature_extractor.extract_features(df_dict)
        features['funding_rate'] = funding['funding_rate']
        features['mark_price'] = funding['mark_price']
        features['index_price'] = funding['index_price']
        features['timestamp'] = datetime.now().isoformat()
        
        # 4. 打印信号摘要
        self._print_signal_summary(features)
        
        return {
            'features': features,
            'klines': df_dict,
            'funding': funding
        }
    
    def _print_signal_summary(self, features: pd.Series):
        """打印信号摘要"""
        print("\n" + "=" * 60)
        print("📈 信号摘要")
        print("=" * 60)
        
        # KDJ/MACD 交叉信号
        for tf in ['5m', '15m', '1h', '4h', '1d']:
            kdj_g = features.get(f'{tf}_kdj_golden', 0)
            kdj_d = features.get(f'{tf}_kdj_death', 0)
            macd_g = features.get(f'{tf}_macd_golden', 0)
            macd_d = features.get(f'{tf}_macd_death', 0)
            
            signals = []
            if kdj_g: signals.append("KDJ金叉🟢")
            if kdj_d: signals.append("KDJ死叉🔴")
            if macd_g: signals.append("MACD金叉🟢")
            if macd_d: signals.append("MACD死叉🔴")
            
            if signals:
                print(f"  {tf:>3}: {', '.join(signals)}")
        
        # 多周期共振
        signal_strength = features.get('signal_strength', 0)
        if signal_strength > 0:
            print(f"\n  🔥 多周期共振: 看涨 (+{signal_strength})")
        elif signal_strength < 0:
            print(f"\n  🔥 多周期共振: 看跌 ({signal_strength})")
        else:
            print(f"\n  ⚖️ 多周期共振: 中性 (0)")
        
        print("=" * 60)
    
    def collect_historical_features(self, base_timeframe: str = '1h', 
                                     lookforward: int = 20,
                                     lookback: int = 100,
                                     start_date: str = None,
                                     end_date: str = None) -> pd.DataFrame:
        """
        收集历史特征数据用于模型训练
        
        Args:
            base_timeframe: 基准时间周期
            lookforward: 预测未来多少根 K 线
            lookback: 回溯多少个数据点
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            包含特征和标签的 DataFrame
        """
        print(f"\n🚀 开始收集历史特征数据...")
        print(f"   基准周期: {base_timeframe}")
        print(f"   预测周期: {lookforward} 根 K 线")
        if start_date:
            print(f"   日期范围: {start_date} ~ {end_date or '现在'}")
        
        # 获取数据
        if start_date:
            df_dict = self.fetcher.get_all_timeframes_historical(start_date, end_date)
        else:
            df_dict = self.fetcher.get_all_timeframes(limit=500)
        
        # 基准时间周期数据
        base_df = df_dict[base_timeframe].copy()
        
        # 生成标签
        base_df = self.label_generator.generate_labels(base_df, lookforward)
        
        # 收集每个时间点的特征
        features_list = []
        valid_indices = range(100, len(base_df) - lookforward)
        
        print(f"\n📊 提取 {len(valid_indices)} 个时间点的特征...")
        
        for i, idx in enumerate(valid_indices):
            # 截取历史数据
            current_df_dict = {}
            for tf, df in df_dict.items():
                # 简化处理：使用固定窗口
                end_idx = min(idx + 1, len(df))
                start_idx = max(0, end_idx - 100)
                current_df_dict[tf] = df.iloc[start_idx:end_idx]
            
            # 提取特征
            features = self.feature_extractor.extract_features(current_df_dict)
            
            # 添加标签
            features['target_regression'] = base_df.iloc[idx]['target_regression']
            features['target_classification'] = base_df.iloc[idx]['target_classification']
            features['target_direction'] = base_df.iloc[idx]['target_direction']
            features['base_timestamp'] = base_df.iloc[idx]['timestamp']
            features['close_price'] = base_df.iloc[idx]['close']
            
            features_list.append(features)
            
            # 进度显示
            if (i + 1) % 50 == 0:
                print(f"   已处理: {i + 1}/{len(valid_indices)}")
        
        result_df = pd.DataFrame(features_list)
        
        # 保存到文件
        filename = f"{self.symbol}_features_{base_timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        result_df.to_csv(filepath, index=False)
        print(f"\n💾 数据已保存到: {filepath}")
        print(f"   样本数量: {len(result_df)}")
        print(f"   特征数量: {len(result_df.columns)}")
        
        # 打印标签分布
        if 'target_classification' in result_df.columns:
            print("\n📊 标签分布:")
            labels = {0: '大跌', 1: '小跌', 2: '横盘', 3: '小涨', 4: '大涨'}
            for label, name in labels.items():
                count = (result_df['target_classification'] == label).sum()
                pct = count / len(result_df) * 100
                print(f"   {name}: {count} ({pct:.1f}%)")
        
        return result_df


def main():
    parser = argparse.ArgumentParser(description='BTC 价格变动预测 - 数据收集器')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='交易对')
    parser.add_argument('--output', type=str, default='../data', help='数据保存目录')
    parser.add_argument('--mode', type=str, choices=['snapshot', 'historical'], default='snapshot',
                        help='模式: snapshot=当前快照, historical=历史特征')
    parser.add_argument('--timeframe', type=str, default='1h', help='历史模式的基准时间周期')
    parser.add_argument('--lookforward', type=int, default=20, help='预测未来多少根K线')
    parser.add_argument('--start-date', type=str, default=None, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='结束日期 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    collector = DataCollector(symbol=args.symbol, output_dir=args.output)
    
    if args.mode == 'snapshot':
        # 收集当前快照
        data = collector.collect_snapshot()
        
        # 保存特征快照
        features_df = pd.DataFrame([data['features']])
        filename = f"{args.symbol}_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(args.output, filename)
        features_df.to_csv(filepath, index=False)
        print(f"\n💾 快照已保存到: {filepath}")
        
    elif args.mode == 'historical':
        # 收集历史特征
        collector.collect_historical_features(
            base_timeframe=args.timeframe,
            lookforward=args.lookforward,
            start_date=args.start_date,
            end_date=args.end_date
        )


if __name__ == "__main__":
    main()

