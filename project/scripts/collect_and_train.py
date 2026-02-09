"""
Fast vectorized data collection + model training pipeline.

Collects multi-TF historical data from Binance, computes ALL features
(including VZO/slope with z-score normalization) in a vectorized manner,
generates labels, then trains models.

Usage:
    python collect_and_train.py --start-date 2024-01-01 --end-date 2026-02-07
    python collect_and_train.py --start-date 2024-01-01 --train-only  # skip collection
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Import shared components ──
from data_collector import BinanceDataFetcher, TechnicalIndicators

# ====================================================================
# VZO / Slope functions
# ====================================================================

def calculate_vzo(df: pd.DataFrame, period: int = 14, ma_len: int = 9):
    signed_vol = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
    vp = pd.Series(signed_vol, index=df.index).ewm(span=period, adjust=False).mean()
    tv = df['volume'].ewm(span=period, adjust=False).mean()
    vzo = (100 * vp / tv).fillna(0).replace([np.inf, -np.inf], 0)
    vzo_ma = vzo.ewm(span=ma_len, adjust=False).mean()
    return vzo, vzo_ma


def calculate_vzo_slope(vzo: pd.Series, lookback: int = 5) -> pd.Series:
    x = np.arange(lookback, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _lr_slope(window):
        if len(window) < lookback:
            return np.nan
        y = window.values
        return ((x - x_mean) * (y - y.mean())).sum() / x_var

    return vzo.rolling(window=lookback).apply(_lr_slope, raw=False).fillna(0)


# ====================================================================
# Vectorized feature extraction for a single timeframe
# ====================================================================

def extract_tf_features_vectorized(df: pd.DataFrame, tf: str, ti: TechnicalIndicators) -> pd.DataFrame:
    """
    Compute ALL features for a single timeframe in a fully vectorized way.
    Returns a DataFrame with one row per candle, columns prefixed by tf.
    """
    n = len(df)
    if n < 50:
        return pd.DataFrame()

    # ── Technical indicators ──
    k, d, j = ti.calculate_kdj(df)
    macd, signal, hist = ti.calculate_macd(df['close'])
    rsi_7 = ti.calculate_rsi(df['close'], 7)
    rsi_14 = ti.calculate_rsi(df['close'], 14)
    rsi_21 = ti.calculate_rsi(df['close'], 21)
    kdj_golden, kdj_death = ti.detect_crossover(k, d)
    macd_golden, macd_death = ti.detect_crossover(macd, signal)

    returns = df['close'].pct_change()
    periods_map = {'5m': 288*365, '15m': 96*365, '30m': 48*365,
                   '1h': 24*365, '4h': 6*365, '1d': 365}
    ann = np.sqrt(periods_map.get(tf, 24*365))
    volatility = returns.rolling(24).std() * ann

    vol = df['volume']
    vol_ma5 = vol.rolling(5).mean()
    vol_ma10 = vol.rolling(10).mean()
    vol_ma20 = vol.rolling(20).mean()
    vol_ratio_ma5 = (vol / vol_ma5).fillna(1).replace([np.inf, -np.inf], 1)
    vol_ratio_ma10 = (vol / vol_ma10).fillna(1).replace([np.inf, -np.inf], 1)
    vol_ratio_ma20 = (vol / vol_ma20).fillna(1).replace([np.inf, -np.inf], 1)
    vol_change_1 = vol.pct_change(1).fillna(0).replace([np.inf, -np.inf], 0) * 100
    vol_change_5 = vol.pct_change(5).fillna(0).replace([np.inf, -np.inf], 0) * 100
    vol_trend = ((vol_ma5 - vol_ma20) / vol_ma20 * 100).fillna(0).replace([np.inf, -np.inf], 0)
    vol_high_20 = vol.rolling(20).max()
    vol_low_20 = vol.rolling(20).min()
    vol_position = ((vol - vol_low_20) / (vol_high_20 - vol_low_20)).fillna(0.5)
    vol_spike = (vol_ratio_ma20 > 2).astype(int)
    vol_shrink = (vol_ratio_ma20 < 0.5).astype(int)
    price_up = (df['close'] > df['close'].shift(1)).astype(int)
    vol_up = (vol > vol.shift(1)).astype(int)
    vol_price_div = (price_up != vol_up).astype(int)

    recent_high = df['high'].rolling(20).max()
    recent_low = df['low'].rolling(20).min()
    price_position = ((df['close'] - recent_low) / (recent_high - recent_low)).fillna(0.5)
    ma20 = df['close'].rolling(20).mean()
    trend_strength = ((df['close'] - ma20) / ma20 * 100).fillna(0).replace([np.inf, -np.inf], 0)

    rsi_overbought = (rsi_14 > 70).astype(int)
    rsi_oversold = (rsi_14 < 30).astype(int)
    rsi_trend = (rsi_14 - rsi_14.shift(5)).fillna(0)

    roc_5 = ti.calculate_roc(df['close'], 5)
    roc_10 = ti.calculate_roc(df['close'], 10)
    roc_20 = ti.calculate_roc(df['close'], 20)
    mom_10 = ti.calculate_momentum(df['close'], 10)
    mom_20 = ti.calculate_momentum(df['close'], 20)
    williams_r = ti.calculate_williams_r(df, 14)
    cci = ti.calculate_cci(df, 20)
    adx, plus_di, minus_di = ti.calculate_adx(df, 14)
    stoch_rsi_k, stoch_rsi_d = ti.calculate_stoch_rsi(df['close'])

    cci_overbought = (cci > 100).astype(int)
    cci_oversold = (cci < -100).astype(int)
    adx_strong = (adx > 25).astype(int)
    adx_weak = (adx < 20).astype(int)
    trend_bullish = (plus_di > minus_di).astype(int)

    # ── VZO & Slope ──
    vzo, vzo_ma = calculate_vzo(df)
    slope = calculate_vzo_slope(vzo, lookback=5)
    slope_accel = slope.diff().fillna(0)

    vzo_roll_mean = vzo.rolling(20).mean().bfill()
    vzo_roll_std = vzo.rolling(20).std().fillna(1).replace(0, 1)
    vzo_zscore = ((vzo - vzo_roll_mean) / vzo_roll_std).fillna(0).replace([np.inf, -np.inf], 0)

    slope_roll_mean = slope.rolling(20).mean().bfill()
    slope_roll_std = slope.rolling(20).std().fillna(1).replace(0, 1)
    slope_zscore = ((slope - slope_roll_mean) / slope_roll_std).fillna(0).replace([np.inf, -np.inf], 0)

    vzo_zone = pd.Series(0, index=df.index)
    vzo_zone = vzo_zone.where(~(vzo > 40), 2)
    vzo_zone = vzo_zone.where(~((vzo > 15) & (vzo <= 40)), 1)
    vzo_zone = vzo_zone.where(~((vzo >= -40) & (vzo < -15)), -1)
    vzo_zone = vzo_zone.where(~(vzo < -40), -2)

    slope_sign = pd.Series(0, index=df.index)
    slope_sign = slope_sign.where(~(slope > 0), 1)
    slope_sign = slope_sign.where(~(slope < 0), -1)

    # ── Assemble ──
    feat = pd.DataFrame({
        f'{tf}_kdj_k': k, f'{tf}_kdj_d': d, f'{tf}_kdj_j': j,
        f'{tf}_kdj_golden': kdj_golden.astype(int).fillna(0),
        f'{tf}_kdj_death': kdj_death.astype(int).fillna(0),
        f'{tf}_macd': macd, f'{tf}_macd_signal': signal, f'{tf}_macd_hist': hist,
        f'{tf}_macd_golden': macd_golden.astype(int).fillna(0),
        f'{tf}_macd_death': macd_death.astype(int).fillna(0),
        f'{tf}_volatility': volatility.fillna(0),
        f'{tf}_vol_ratio_ma5': vol_ratio_ma5, f'{tf}_vol_ratio_ma10': vol_ratio_ma10,
        f'{tf}_vol_ratio_ma20': vol_ratio_ma20,
        f'{tf}_vol_change_1': vol_change_1, f'{tf}_vol_change_5': vol_change_5,
        f'{tf}_vol_trend': vol_trend, f'{tf}_vol_position': vol_position,
        f'{tf}_vol_spike': vol_spike, f'{tf}_vol_shrink': vol_shrink,
        f'{tf}_vol_price_divergence': vol_price_div,
        f'{tf}_price_position': price_position, f'{tf}_trend_strength': trend_strength,
        f'{tf}_rsi_7': rsi_7, f'{tf}_rsi_14': rsi_14, f'{tf}_rsi_21': rsi_21,
        f'{tf}_rsi_overbought': rsi_overbought, f'{tf}_rsi_oversold': rsi_oversold,
        f'{tf}_rsi_trend': rsi_trend,
        f'{tf}_roc_5': roc_5, f'{tf}_roc_10': roc_10, f'{tf}_roc_20': roc_20,
        f'{tf}_mom_10': mom_10, f'{tf}_mom_20': mom_20,
        f'{tf}_williams_r': williams_r,
        f'{tf}_cci': cci, f'{tf}_cci_overbought': cci_overbought,
        f'{tf}_cci_oversold': cci_oversold,
        f'{tf}_adx': adx, f'{tf}_plus_di': plus_di, f'{tf}_minus_di': minus_di,
        f'{tf}_adx_strong_trend': adx_strong, f'{tf}_adx_weak_trend': adx_weak,
        f'{tf}_trend_bullish': trend_bullish,
        f'{tf}_stoch_rsi_k': stoch_rsi_k, f'{tf}_stoch_rsi_d': stoch_rsi_d,
        # VZO features
        f'{tf}_vzo': vzo, f'{tf}_vzo_ma': vzo_ma,
        f'{tf}_vzo_slope': slope,
        f'{tf}_vzo_zscore': vzo_zscore, f'{tf}_slope_zscore': slope_zscore,
        f'{tf}_slope_accel': slope_accel,
        f'{tf}_vzo_zone': vzo_zone, f'{tf}_slope_sign': slope_sign,
    }, index=df.index)

    # Add timestamp for merge_asof
    feat['timestamp'] = df['timestamp'] if 'timestamp' in df.columns else df.index
    return feat.fillna(0).replace([np.inf, -np.inf], 0)


# ====================================================================
# Build full feature matrix aligned to base timeframe
# ====================================================================

def build_feature_matrix(df_dict: Dict[str, pd.DataFrame],
                         base_tf: str = '1h',
                         lookforward: int = 20) -> pd.DataFrame:
    """
    Build a full feature matrix aligned to the base timeframe.
    All other TF features are merged via merge_asof (forward-fill).
    """
    ti = TechnicalIndicators()
    base_df = df_dict[base_tf].copy()
    base_df['timestamp'] = pd.to_datetime(base_df['timestamp'])

    print(f"  Base timeframe: {base_tf}, {len(base_df)} candles")

    # Extract base TF features
    print(f"  Computing {base_tf} features...")
    base_feat = extract_tf_features_vectorized(base_df, base_tf, ti)

    # Merge other timeframes
    all_tfs = ['5m', '15m', '30m', '1h', '4h']
    for tf in all_tfs:
        if tf == base_tf:
            continue
        if tf not in df_dict or len(df_dict[tf]) < 50:
            print(f"  Skipping {tf} (insufficient data)")
            continue

        print(f"  Computing {tf} features ({len(df_dict[tf])} candles)...")
        tf_df = df_dict[tf].copy()
        tf_df['timestamp'] = pd.to_datetime(tf_df['timestamp'])
        tf_feat = extract_tf_features_vectorized(tf_df, tf, ti)

        # merge_asof: align to base timestamps
        tf_feat['timestamp'] = pd.to_datetime(tf_feat['timestamp'])
        base_feat_sorted = base_feat.sort_values('timestamp').reset_index(drop=True)
        tf_feat_sorted = tf_feat.sort_values('timestamp').reset_index(drop=True)

        merged = pd.merge_asof(
            base_feat_sorted[['timestamp']],
            tf_feat_sorted,
            on='timestamp',
            direction='backward',
        )
        # Bring back the merged columns
        for col in tf_feat.columns:
            if col != 'timestamp' and col not in base_feat.columns:
                base_feat[col] = merged[col].values

    # ── Cross-timeframe VZO consensus features ──
    vzo_tfs = [t for t in all_tfs if f'{t}_vzo_zone' in base_feat.columns]
    if len(vzo_tfs) >= 2:
        zone_df = base_feat[[f'{t}_vzo_zone' for t in vzo_tfs]]
        sign_df = base_feat[[f'{t}_slope_sign' for t in vzo_tfs]]

        base_feat['vzo_multi_tf_bullish'] = (zone_df > 0).sum(axis=1)
        base_feat['vzo_multi_tf_bearish'] = (zone_df < 0).sum(axis=1)
        base_feat['vzo_multi_tf_consensus'] = (
            base_feat['vzo_multi_tf_bullish'] - base_feat['vzo_multi_tf_bearish'])
        base_feat['slope_multi_tf_bullish'] = (sign_df > 0).sum(axis=1)
        base_feat['slope_multi_tf_bearish'] = (sign_df < 0).sum(axis=1)
        base_feat['slope_multi_tf_consensus'] = (
            base_feat['slope_multi_tf_bullish'] - base_feat['slope_multi_tf_bearish'])

        # Short vs long term divergence
        short_tfs = [t for t in ['5m', '15m'] if f'{t}_vzo' in base_feat.columns]
        long_tfs = [t for t in ['1h', '4h'] if f'{t}_vzo' in base_feat.columns]
        if short_tfs and long_tfs:
            base_feat['vzo_short_long_diff'] = (
                base_feat[[f'{t}_vzo' for t in short_tfs]].mean(axis=1) -
                base_feat[[f'{t}_vzo' for t in long_tfs]].mean(axis=1))
            base_feat['slope_short_long_diff'] = (
                base_feat[[f'{t}_vzo_slope' for t in short_tfs]].mean(axis=1) -
                base_feat[[f'{t}_vzo_slope' for t in long_tfs]].mean(axis=1))

    # ── Cross signals ──
    golden_cols = [f'{t}_kdj_golden' for t in all_tfs if f'{t}_kdj_golden' in base_feat.columns]
    golden_cols += [f'{t}_macd_golden' for t in all_tfs if f'{t}_macd_golden' in base_feat.columns]
    death_cols = [f'{t}_kdj_death' for t in all_tfs if f'{t}_kdj_death' in base_feat.columns]
    death_cols += [f'{t}_macd_death' for t in all_tfs if f'{t}_macd_death' in base_feat.columns]

    base_feat['multi_tf_golden_count'] = base_feat[golden_cols].sum(axis=1) if golden_cols else 0
    base_feat['multi_tf_death_count'] = base_feat[death_cols].sum(axis=1) if death_cols else 0
    base_feat['signal_strength'] = base_feat['multi_tf_golden_count'] - base_feat['multi_tf_death_count']

    # ── Time features ──
    ts = pd.to_datetime(base_feat['timestamp'])
    base_feat['hour'] = ts.dt.hour
    base_feat['day_of_week'] = ts.dt.weekday
    base_feat['is_weekend'] = (ts.dt.weekday >= 5).astype(int)

    # ── Labels ──
    future_price = base_df['close'].shift(-lookforward)
    price_change_pct = (future_price - base_df['close']) / base_df['close'] * 100
    base_feat['target_regression'] = price_change_pct.values
    base_feat['target_classification'] = price_change_pct.apply(
        lambda p: 0 if p < -2 else (1 if p < -0.5 else (2 if p < 0.5 else (3 if p < 2 else 4)))
    ).values
    base_feat['close_price'] = base_df['close'].values
    base_feat['base_timestamp'] = base_df['timestamp'].values

    # Drop rows with NaN targets
    base_feat = base_feat.dropna(subset=['target_regression']).reset_index(drop=True)

    # Drop the timestamp column used for merging
    base_feat = base_feat.drop(columns=['timestamp'], errors='ignore')

    # Final cleanup
    base_feat = base_feat.fillna(0).replace([np.inf, -np.inf], 0)

    print(f"  Feature matrix: {base_feat.shape[0]} samples x {base_feat.shape[1]} columns")
    return base_feat


# ====================================================================
# Data cleaning (from train_model.py)
# ====================================================================

def prepare_features(df: pd.DataFrame):
    """Clean and prepare features for training."""
    exclude_cols = ['target_regression', 'target_classification', 'target_direction',
                    'base_timestamp', 'timestamp', 'close_price']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y_reg = df['target_regression'] if 'target_regression' in df.columns else None

    n_orig = len(X)
    print(f"\n  Data cleaning...")
    print(f"    Original: {n_orig} samples, {len(X.columns)} features")

    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop high-NaN columns
    nan_ratio = X.isna().sum() / len(X)
    high_nan = nan_ratio[nan_ratio > 0.5].index.tolist()
    if high_nan:
        print(f"    Dropped {len(high_nan)} high-NaN columns")
        X = X.drop(columns=high_nan)

    # Fill NaN with median
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            X[col] = X[col].fillna(med if not pd.isna(med) else 0)

    # Clip outliers (1st-99th percentile)
    for col in X.select_dtypes(include=[np.number]).columns:
        q01, q99 = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col] = X[col].clip(q01, q99)

    # Drop constant columns
    const_cols = X.columns[X.std() == 0].tolist()
    if const_cols:
        print(f"    Dropped {len(const_cols)} constant columns")
        X = X.drop(columns=const_cols)

    # Remove NaN labels
    if y_reg is not None:
        valid = ~y_reg.isna()
        X = X[valid].reset_index(drop=True)
        y_reg = y_reg[valid].reset_index(drop=True)

    print(f"    Final: {len(X)} samples, {len(X.columns)} features")
    return X, y_reg


# ====================================================================
# Model training
# ====================================================================

def train_models(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """Train RF, GBM, Ridge and pick the best by direction accuracy."""

    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'rf': RandomForestRegressor(n_estimators=150, max_depth=10,
                                    min_samples_split=10, random_state=42, n_jobs=-1),
        'gbm': GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                         max_depth=6, random_state=42),
        'ridge': Ridge(alpha=1.0),
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    for name, model in models.items():
        print(f"\n  {'='*50}")
        print(f"  Training: {name}")
        print(f"  {'='*50}")

        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            dir_acc = np.mean(np.sign(y_te) == np.sign(y_pred))
            mae = mean_absolute_error(y_te, y_pred)
            returns = np.where(y_pred > 0, y_te, -y_te)
            total_ret = np.sum(returns)
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)
                      if np.std(returns) > 0 else 0)

            fold_metrics.append({
                'direction_accuracy': dir_acc, 'mae': mae,
                'total_return': total_ret, 'sharpe_ratio': sharpe,
            })
            print(f"    Fold {fold+1}: dir_acc={dir_acc:.2%}, MAE={mae:.4f}, "
                  f"return={total_ret:.2f}%")

        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        std = {f'{k}_std': np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        results[name] = {**avg, **std}

        print(f"\n    AVG: dir_acc={avg['direction_accuracy']:.2%} "
              f"(+/-{std['direction_accuracy_std']:.2%}), "
              f"MAE={avg['mae']:.4f}, return={avg['total_return']:.2f}%, "
              f"sharpe={avg['sharpe_ratio']:.2f}")

    # Pick best model by direction accuracy
    best_name = max(results, key=lambda x: results[x]['direction_accuracy'])
    best_model = models[best_name]
    print(f"\n  Best model: {best_name} "
          f"(dir_acc={results[best_name]['direction_accuracy']:.2%})")

    # Retrain on full data
    print(f"  Retraining {best_name} on full dataset...")
    best_model.fit(X_scaled, y.values)

    # Feature importance
    feat_imp = None
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = dict(zip(feature_names, best_model.feature_importances_))
        top10 = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]
        print(f"\n  Top 15 features:")
        for i, (f, imp) in enumerate(top10, 1):
            print(f"    {i:2d}. {f}: {imp:.4f}")

    return best_model, best_name, scaler, feature_names, feat_imp, results


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='Fast data collection + training')
    parser.add_argument('--start-date', type=str, default='2024-01-01')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (default: today)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--base-tf', type=str, default='1h')
    parser.add_argument('--lookforward', type=int, default=20)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--output', type=str, default='../models')
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--train-only', action='store_true',
                        help='Skip data collection, use existing CSV')
    args = parser.parse_args()

    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Step 1: Data collection ──
    csv_path = None
    if not args.train_only:
        print("\n" + "="*60)
        print(f"Step 1: Collecting {args.symbol} data ({args.start_date} ~ {end_date})")
        print("="*60)

        fetcher = BinanceDataFetcher(args.symbol)
        print("\n  Downloading multi-timeframe K-line data...")

        df_dict = {}
        for tf in ['5m', '15m', '30m', '1h', '4h']:
            print(f"    Fetching {tf}...")
            t0 = time.time()
            df_dict[tf] = fetcher.get_klines_historical(tf, args.start_date, end_date)
            elapsed = time.time() - t0
            print(f"      -> {len(df_dict[tf])} candles ({elapsed:.1f}s)")

        # Build feature matrix
        print("\n  Building feature matrix (vectorized)...")
        t0 = time.time()
        feature_df = build_feature_matrix(df_dict, base_tf=args.base_tf,
                                          lookforward=args.lookforward)
        elapsed = time.time() - t0
        print(f"  Feature matrix built in {elapsed:.1f}s")

        # Save
        csv_path = os.path.join(args.data_dir,
                                f'{args.symbol}_features_{args.base_tf}_{timestamp}.csv')
        feature_df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}")
        print(f"  {feature_df.shape[0]} samples, {feature_df.shape[1]} columns")

        # Label distribution
        if 'target_classification' in feature_df.columns:
            labels = {0: 'Big drop (<-2%)', 1: 'Small drop', 2: 'Sideways',
                      3: 'Small rise', 4: 'Big rise (>2%)'}
            print("\n  Label distribution:")
            for lbl, name in labels.items():
                cnt = (feature_df['target_classification'] == lbl).sum()
                pct = cnt / len(feature_df) * 100
                print(f"    {name}: {cnt} ({pct:.1f}%)")
    else:
        # Find most recent CSV
        pattern = os.path.join(args.data_dir, f'{args.symbol}_features_{args.base_tf}_*.csv')
        files = glob.glob(pattern)
        if not files:
            print(f"ERROR: No data files found matching {pattern}")
            sys.exit(1)
        csv_path = max(files, key=os.path.getmtime)
        print(f"\n  Using existing data: {csv_path}")

    # ── Step 2: Training ──
    print("\n" + "="*60)
    print("Step 2: Model Training")
    print("="*60)

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} samples, {len(df.columns)} columns")

    X, y_reg = prepare_features(df)

    if y_reg is None or len(X) == 0:
        print("ERROR: No valid training data")
        sys.exit(1)

    best_model, best_name, scaler, feature_names, feat_imp, results = \
        train_models(X, y_reg, n_splits=args.cv_splits)

    # ── Step 3: Save model ──
    print("\n" + "="*60)
    print("Step 3: Saving model")
    print("="*60)

    model_path = os.path.join(args.output, f'regression_model_{timestamp}.pkl')
    model_data = {
        'best_model': best_model,
        'best_model_name': best_name,
        'scaler': scaler,
        'feature_names': feature_names,
        'feature_importance': feat_imp,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  Model saved: {model_path}")

    # Save results
    results_path = os.path.join(args.output, f'regression_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2)
    print(f"  Results saved: {results_path}")

    # ── Summary ──
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"  Best model: {best_name}")
    print(f"  Direction accuracy: {results[best_name]['direction_accuracy']:.2%}")
    print(f"  Total features: {len(feature_names)}")

    # Count VZO features
    vzo_feats = [f for f in feature_names if 'vzo' in f or 'slope' in f]
    print(f"  VZO/slope features: {len(vzo_feats)}")
    print(f"\n  Model path (use in BTCVZOBacktest config):")
    print(f"    {os.path.abspath(model_path)}")


if __name__ == '__main__':
    main()
