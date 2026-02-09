"""
2-Hour Volatility Prediction Model
====================================

Predicts the next 2 hours' volatility using multi-TF features (5m, 15m, 30m, 1h, 4h).

Targets (computed from 5m data for precision):
  - target_vol_realized: realized volatility = std(5m returns) over next 24 candles (2h)
  - target_range_pct:    (max_high - min_low) / close * 100 over next 2h
  - target_vol_class:    binary high/low volatility (above/below rolling median)

Usage:
    python collect_and_train_vol.py --start-date 2024-01-01
    python collect_and_train_vol.py --train-only   # use existing CSV
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
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from data_collector import BinanceDataFetcher, TechnicalIndicators

# ── Reuse from collect_and_train.py ──
from collect_and_train import (build_feature_matrix, calculate_vzo,
                               calculate_vzo_slope,
                               extract_tf_features_vectorized)


# ====================================================================
# Volatility-specific feature engineering
# ====================================================================

def add_vol_specific_features(base_feat: pd.DataFrame,
                              df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Add features specifically useful for volatility prediction:
    - Recent realized volatility (lookback) at multiple horizons
    - ATR (Average True Range)
    - Bollinger Band width
    - Intraday vol pattern features
    """
    # ── From 1h data ──
    if '1h' in df_dict:
        df = df_dict['1h'].copy()
        close = df['close']
        high = df['high']
        low = df['low']
        returns = close.pct_change()

        # Recent realized vol at multiple lookback windows
        for w in [6, 12, 24, 48]:  # 6h, 12h, 24h, 48h
            rv = returns.rolling(w).std().fillna(0)
            base_feat[f'recent_vol_{w}h'] = rv.values[:len(base_feat)]

        # Vol-of-vol (how much vol itself is changing)
        vol_24 = returns.rolling(24).std()
        base_feat['vol_of_vol'] = vol_24.rolling(12).std().fillna(0).values[:len(base_feat)]

        # Vol ratio: recent 6h vol / 24h vol (vol regime detection)
        vol_6 = returns.rolling(6).std().fillna(0)
        vol_24_safe = vol_24.replace(0, 1e-8)
        base_feat['vol_ratio_6_24'] = (vol_6 / vol_24_safe).fillna(1).replace(
            [np.inf, -np.inf], 1).values[:len(base_feat)]

        # ATR (Average True Range) at multiple periods
        for period in [7, 14, 20]:
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().fillna(0)
            atr_pct = (atr / close * 100).fillna(0).replace([np.inf, -np.inf], 0)
            base_feat[f'atr_{period}_pct'] = atr_pct.values[:len(base_feat)]

        # ATR expansion/contraction (current ATR vs longer-term ATR)
        atr_7 = (high - low).rolling(7).mean()
        atr_20 = (high - low).rolling(20).mean()
        atr_ratio = (atr_7 / atr_20.replace(0, 1e-8)).fillna(1).replace([np.inf, -np.inf], 1)
        base_feat['atr_expansion'] = atr_ratio.values[:len(base_feat)]

        # Bollinger Band width (proxy for vol)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_width = (2 * std20 / ma20.replace(0, 1e-8) * 100).fillna(0).replace([np.inf, -np.inf], 0)
        base_feat['bb_width_pct'] = bb_width.values[:len(base_feat)]

        # BB width z-score (is current BB width high or low vs recent history?)
        bb_mean = bb_width.rolling(48).mean().bfill()
        bb_std = bb_width.rolling(48).std().fillna(1).replace(0, 1)
        base_feat['bb_width_zscore'] = ((bb_width - bb_mean) / bb_std).fillna(0).replace(
            [np.inf, -np.inf], 0).values[:len(base_feat)]

        # Range-based volatility (Parkinson)
        log_hl = np.log(high / low.replace(0, 1e-8))
        parkinson = np.sqrt(log_hl ** 2 / (4 * np.log(2)))
        park_ma = parkinson.rolling(12).mean().fillna(0)
        base_feat['parkinson_vol_12'] = park_ma.values[:len(base_feat)]

        # Candle body ratio: |close-open| / (high-low) -- doji candles = indecision = vol
        body = (close - df['open']).abs()
        wick = (high - low).replace(0, 1e-8)
        body_ratio = (body / wick).fillna(0.5)
        base_feat['body_ratio'] = body_ratio.values[:len(base_feat)]
        # Rolling avg body ratio (many small bodies = compression before expansion)
        base_feat['body_ratio_ma5'] = body_ratio.rolling(5).mean().fillna(0.5).values[:len(base_feat)]

    # ── From 5m data (high-frequency vol signals) ──
    if '5m' in df_dict:
        df5 = df_dict['5m'].copy()
        df5['timestamp'] = pd.to_datetime(df5['timestamp'])
        ret5 = df5['close'].pct_change()

        # 5m realized vol at multiple windows
        for w, label in [(12, '1h'), (24, '2h'), (72, '6h')]:
            rv = ret5.rolling(w).std().fillna(0)
            # Map to 1h timestamps via merge_asof
            rv_df = pd.DataFrame({'timestamp': df5['timestamp'], f'rv5m_{label}': rv})
            if 'base_timestamp' in base_feat.columns:
                base_ts = pd.DataFrame({'timestamp': pd.to_datetime(base_feat['base_timestamp'])})
                merged = pd.merge_asof(
                    base_ts.sort_values('timestamp'),
                    rv_df.sort_values('timestamp'),
                    on='timestamp', direction='backward')
                base_feat[f'rv5m_{label}'] = merged[f'rv5m_{label}'].values

        # 5m vol trend: is 1h vol increasing vs 2h vol?
        if 'rv5m_1h' in base_feat.columns and 'rv5m_2h' in base_feat.columns:
            rv1 = base_feat['rv5m_1h'].replace(0, 1e-8)
            base_feat['rv5m_trend'] = (base_feat['rv5m_1h'] / base_feat['rv5m_2h'].replace(0, 1e-8)).fillna(1)

    base_feat = base_feat.fillna(0).replace([np.inf, -np.inf], 0)
    return base_feat


# ====================================================================
# Build volatility targets from 5m data
# ====================================================================

def build_vol_targets(df_1h: pd.DataFrame, df_5m: pd.DataFrame,
                      horizon_hours: int = 2) -> pd.DataFrame:
    """
    For each 1h candle, compute volatility of the NEXT `horizon_hours` hours
    using 5m data for precision.

    Returns DataFrame with columns: base_timestamp, target_vol_realized,
    target_range_pct, target_vol_class.
    """
    n_5m = horizon_hours * 12  # 12 five-minute candles per hour

    df_5m = df_5m.copy()
    df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
    df_5m = df_5m.sort_values('timestamp').reset_index(drop=True)
    df_5m['return'] = df_5m['close'].pct_change()

    df_1h = df_1h.copy()
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)

    # Build lookup: for each 1h timestamp, find the next n_5m 5m-candles
    targets = []
    ts_5m = df_5m['timestamp'].values

    for _, row in df_1h.iterrows():
        t0 = row['timestamp']
        # Find 5m candles in window [t0, t0 + horizon]
        mask = (df_5m['timestamp'] > t0) & \
               (df_5m['timestamp'] <= t0 + pd.Timedelta(hours=horizon_hours))
        window = df_5m.loc[mask]

        if len(window) < n_5m * 0.8:  # need at least 80% coverage
            targets.append({
                'base_timestamp': t0,
                'target_vol_realized': np.nan,
                'target_range_pct': np.nan,
            })
            continue

        # Realized vol: std of 5m returns, annualized to 2h
        rv = window['return'].std()

        # Range: (max_high - min_low) / close_at_t0
        range_pct = (window['high'].max() - window['low'].min()) / row['close'] * 100

        targets.append({
            'base_timestamp': t0,
            'target_vol_realized': rv,
            'target_range_pct': range_pct,
        })

    target_df = pd.DataFrame(targets)

    # Binary classification: high/low vol relative to rolling median
    median_rv = target_df['target_vol_realized'].rolling(168, min_periods=48).median()  # ~1 week
    target_df['target_vol_class'] = (target_df['target_vol_realized'] > median_rv).astype(float)
    target_df.loc[target_df['target_vol_realized'].isna(), 'target_vol_class'] = np.nan

    return target_df


# ====================================================================
# Data preparation
# ====================================================================

def prepare_vol_features(df: pd.DataFrame):
    """Clean features for volatility model."""
    exclude_cols = ['target_vol_realized', 'target_range_pct', 'target_vol_class',
                    'target_regression', 'target_classification',
                    'base_timestamp', 'timestamp', 'close_price']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()

    print(f"\n  Data cleaning...")
    print(f"    Original: {len(X)} samples, {len(X.columns)} features")

    X = X.replace([np.inf, -np.inf], np.nan)

    nan_ratio = X.isna().sum() / len(X)
    high_nan = nan_ratio[nan_ratio > 0.5].index.tolist()
    if high_nan:
        print(f"    Dropped {len(high_nan)} high-NaN columns")
        X = X.drop(columns=high_nan)

    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            X[col] = X[col].fillna(med if not pd.isna(med) else 0)

    for col in X.select_dtypes(include=[np.number]).columns:
        q01, q99 = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col] = X[col].clip(q01, q99)

    const_cols = X.columns[X.std() == 0].tolist()
    if const_cols:
        print(f"    Dropped {len(const_cols)} constant columns")
        X = X.drop(columns=const_cols)

    print(f"    Final: {len(X)} samples, {len(X.columns)} features")
    return X


# ====================================================================
# Volatility model training
# ====================================================================

def train_vol_regression(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """Train regression models to predict volatility magnitude."""

    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'rf': RandomForestRegressor(n_estimators=200, max_depth=12,
                                    min_samples_split=10, random_state=42, n_jobs=-1),
        'gbm': GradientBoostingRegressor(n_estimators=250, learning_rate=0.05,
                                         max_depth=6, random_state=42),
        'ridge': Ridge(alpha=1.0),
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    for name, model in models.items():
        print(f"\n  {'='*50}")
        print(f"  Training vol regression: {name}")
        print(f"  {'='*50}")

        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            mae = mean_absolute_error(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            r2 = r2_score(y_te, y_pred)
            # Correlation between predicted and actual vol
            corr = np.corrcoef(y_te, y_pred)[0, 1] if len(y_te) > 1 else 0

            # Direction accuracy: did we predict high-vol vs low-vol correctly?
            median_y = np.median(y_tr)  # use training median as threshold
            dir_acc = np.mean((y_te > median_y) == (y_pred > median_y))

            fold_metrics.append({
                'mae': mae, 'rmse': rmse, 'r2': r2,
                'correlation': corr, 'vol_dir_accuracy': dir_acc,
            })
            print(f"    Fold {fold+1}: MAE={mae:.6f}, R2={r2:.4f}, "
                  f"corr={corr:.4f}, vol_dir_acc={dir_acc:.2%}")

        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        std = {f'{k}_std': np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        results[name] = {**avg, **std}

        print(f"\n    AVG: MAE={avg['mae']:.6f}, R2={avg['r2']:.4f}, "
              f"corr={avg['correlation']:.4f}, vol_dir={avg['vol_dir_accuracy']:.2%}")

    # Pick best model by correlation (best vol predictor)
    best_name = max(results, key=lambda x: results[x]['correlation'])
    best_model = models[best_name]
    print(f"\n  Best vol regression: {best_name} "
          f"(corr={results[best_name]['correlation']:.4f})")

    # Retrain on full data
    best_model.fit(X_scaled, y.values)

    feat_imp = None
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = dict(zip(feature_names, best_model.feature_importances_))

    return best_model, best_name, scaler, feature_names, feat_imp, results


def train_vol_classifier(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """Train binary classifier: high-vol (1) vs low-vol (0)."""

    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'rf_cls': RandomForestClassifier(n_estimators=200, max_depth=10,
                                         class_weight='balanced', random_state=42, n_jobs=-1),
        'gbm_cls': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                              max_depth=5, random_state=42),
        'logistic': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000,
                                       random_state=42),
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    for name, model in models.items():
        print(f"\n  {'='*50}")
        print(f"  Training vol classifier: {name}")
        print(f"  {'='*50}")

        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            acc = accuracy_score(y_te, y_pred)

            # Precision for high-vol prediction
            high_vol_mask = y_pred == 1
            if high_vol_mask.sum() > 0:
                high_vol_precision = np.mean(y_te[high_vol_mask] == 1)
            else:
                high_vol_precision = 0

            fold_metrics.append({'accuracy': acc, 'high_vol_precision': high_vol_precision})
            print(f"    Fold {fold+1}: acc={acc:.2%}, high_vol_precision={high_vol_precision:.2%}")

        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        std = {f'{k}_std': np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        results[name] = {**avg, **std}

        print(f"\n    AVG: acc={avg['accuracy']:.2%}, "
              f"high_vol_prec={avg['high_vol_precision']:.2%}")

    best_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = models[best_name]
    print(f"\n  Best vol classifier: {best_name} "
          f"(acc={results[best_name]['accuracy']:.2%})")

    best_model.fit(X_scaled, y.values)

    feat_imp = None
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = dict(zip(feature_names, best_model.feature_importances_))

    return best_model, best_name, scaler, feature_names, feat_imp, results


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='2h Volatility Prediction Training')
    parser.add_argument('--start-date', type=str, default='2024-01-01')
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--horizon', type=int, default=2, help='Prediction horizon in hours')
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--output', type=str, default='../models')
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--train-only', action='store_true')
    args = parser.parse_args()

    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    csv_path = None

    if not args.train_only:
        # ══════════════════════════════════════════════════════
        # STEP 1: Data Collection
        # ══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print(f"Step 1: Collecting {args.symbol} data ({args.start_date} ~ {end_date})")
        print("="*60)

        fetcher = BinanceDataFetcher(args.symbol)
        df_dict = {}
        for tf in ['5m', '15m', '30m', '1h', '4h']:
            print(f"  Fetching {tf}...")
            t0 = time.time()
            df_dict[tf] = fetcher.get_klines_historical(tf, args.start_date, end_date)
            print(f"    -> {len(df_dict[tf])} candles ({time.time()-t0:.1f}s)")

        # ══════════════════════════════════════════════════════
        # STEP 2: Build Features + Vol Targets
        # ══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("Step 2: Building feature matrix + volatility targets")
        print("="*60)

        # Build standard feature matrix (aligned to 1h)
        print("\n  Building multi-TF feature matrix...")
        t0 = time.time()
        # Use dummy lookforward=2 (we'll replace targets with vol targets)
        feature_df = build_feature_matrix(df_dict, base_tf='1h', lookforward=2)
        print(f"  Feature matrix: {feature_df.shape} in {time.time()-t0:.1f}s")

        # Add vol-specific features
        print("\n  Adding volatility-specific features...")
        feature_df = add_vol_specific_features(feature_df, df_dict)
        print(f"  After vol features: {feature_df.shape[1]} columns")

        # Build volatility targets from 5m data
        print(f"\n  Computing {args.horizon}h volatility targets from 5m data...")
        vol_targets = build_vol_targets(df_dict['1h'], df_dict['5m'],
                                        horizon_hours=args.horizon)
        print(f"  Vol targets: {len(vol_targets)} rows, "
              f"{vol_targets['target_vol_realized'].notna().sum()} valid")

        # Merge vol targets into feature matrix
        feature_df['base_timestamp'] = pd.to_datetime(feature_df['base_timestamp'])
        vol_targets['base_timestamp'] = pd.to_datetime(vol_targets['base_timestamp'])
        feature_df = feature_df.merge(vol_targets, on='base_timestamp', how='left',
                                      suffixes=('', '_vol'))

        # Drop dummy direction targets
        feature_df = feature_df.drop(columns=['target_regression', 'target_classification'],
                                     errors='ignore')

        # Drop NaN vol targets
        before = len(feature_df)
        feature_df = feature_df.dropna(subset=['target_vol_realized']).reset_index(drop=True)
        print(f"  Dropped {before - len(feature_df)} rows with NaN vol targets")

        # Save
        csv_path = os.path.join(args.data_dir,
                                f'{args.symbol}_vol_features_{args.horizon}h_{timestamp}.csv')
        feature_df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}")
        print(f"  Shape: {feature_df.shape}")

        # Target stats
        rv = feature_df['target_vol_realized']
        rng = feature_df['target_range_pct']
        cls = feature_df['target_vol_class']
        print(f"\n  Target statistics:")
        print(f"    Realized vol:  mean={rv.mean():.6f}, std={rv.std():.6f}, "
              f"median={rv.median():.6f}")
        print(f"    Range (pct):   mean={rng.mean():.2f}%, std={rng.std():.2f}%, "
              f"median={rng.median():.2f}%")
        print(f"    Vol class:     high={cls.sum():.0f} ({cls.mean()*100:.1f}%), "
              f"low={len(cls)-cls.sum():.0f} ({(1-cls.mean())*100:.1f}%)")

    else:
        pattern = os.path.join(args.data_dir, f'{args.symbol}_vol_features_*.csv')
        files = glob.glob(pattern)
        if not files:
            print(f"ERROR: No vol data files found: {pattern}")
            sys.exit(1)
        csv_path = max(files, key=os.path.getmtime)
        print(f"\n  Using existing data: {csv_path}")

    # ══════════════════════════════════════════════════════
    # STEP 3: Train Regression Model (predict vol magnitude)
    # ══════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("Step 3: Training Volatility Regression Model")
    print("="*60)

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} samples, {len(df.columns)} columns")

    X = prepare_vol_features(df)

    # Align targets
    valid = df['target_vol_realized'].notna()
    X = X[valid].reset_index(drop=True)
    y_vol = df.loc[valid, 'target_vol_realized'].reset_index(drop=True)
    y_range = df.loc[valid, 'target_range_pct'].reset_index(drop=True)
    y_class = df.loc[valid, 'target_vol_class'].reset_index(drop=True)

    # --- Regression on realized vol ---
    print(f"\n  Training target: realized volatility ({len(X)} samples)")
    reg_model, reg_name, reg_scaler, reg_features, reg_imp, reg_results = \
        train_vol_regression(X, y_vol, n_splits=args.cv_splits)

    # --- Regression on range_pct ---
    print(f"\n  Training target: range percentage ({len(X)} samples)")
    rng_model, rng_name, rng_scaler, rng_features, rng_imp, rng_results = \
        train_vol_regression(X, y_range, n_splits=args.cv_splits)

    # ══════════════════════════════════════════════════════
    # STEP 4: Train Binary Classifier (high/low vol)
    # ══════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("Step 4: Training Volatility Classifier (High/Low)")
    print("="*60)

    valid_cls = y_class.notna()
    X_cls = X[valid_cls].reset_index(drop=True)
    y_cls = y_class[valid_cls].astype(int).reset_index(drop=True)
    print(f"  Classifier samples: {len(X_cls)} "
          f"(high={y_cls.sum()}, low={len(y_cls)-y_cls.sum()})")

    cls_model, cls_name, cls_scaler, cls_features, cls_imp, cls_results = \
        train_vol_classifier(X_cls, y_cls, n_splits=args.cv_splits)

    # ══════════════════════════════════════════════════════
    # STEP 5: Save Models
    # ══════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("Step 5: Saving Models")
    print("="*60)

    # Save vol regression model
    vol_model_path = os.path.join(args.output, f'vol_regression_{args.horizon}h_{timestamp}.pkl')
    with open(vol_model_path, 'wb') as f:
        pickle.dump({
            'model': reg_model, 'model_name': reg_name,
            'scaler': reg_scaler, 'feature_names': reg_features,
            'feature_importance': reg_imp, 'target': 'realized_vol',
            'horizon_hours': args.horizon,
        }, f)
    print(f"  Vol regression: {vol_model_path}")

    # Save range regression model
    rng_model_path = os.path.join(args.output, f'range_regression_{args.horizon}h_{timestamp}.pkl')
    with open(rng_model_path, 'wb') as f:
        pickle.dump({
            'model': rng_model, 'model_name': rng_name,
            'scaler': rng_scaler, 'feature_names': rng_features,
            'feature_importance': rng_imp, 'target': 'range_pct',
            'horizon_hours': args.horizon,
        }, f)
    print(f"  Range regression: {rng_model_path}")

    # Save vol classifier
    cls_model_path = os.path.join(args.output, f'vol_classifier_{args.horizon}h_{timestamp}.pkl')
    with open(cls_model_path, 'wb') as f:
        pickle.dump({
            'model': cls_model, 'model_name': cls_name,
            'scaler': cls_scaler, 'feature_names': cls_features,
            'feature_importance': cls_imp, 'target': 'vol_class',
            'horizon_hours': args.horizon,
        }, f)
    print(f"  Vol classifier: {cls_model_path}")

    # Save all results
    all_results = {
        'vol_regression': reg_results,
        'range_regression': rng_results,
        'vol_classifier': cls_results,
    }
    results_path = os.path.join(args.output, f'vol_results_{args.horizon}h_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump({k: {kk: {kkk: float(vvv) for kkk, vvv in vv.items()}
                       for kk, vv in v.items()} for k, v in all_results.items()}, f, indent=2)
    print(f"  Results: {results_path}")

    # Print top features
    for label, imp in [('Vol regression', reg_imp), ('Range regression', rng_imp),
                       ('Vol classifier', cls_imp)]:
        if imp:
            top = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
            print(f"\n  {label} - Top 15 features:")
            for i, (fname, fimp) in enumerate(top, 1):
                tag = ''
                if 'vzo' in fname or 'slope' in fname:
                    tag = ' [VZO]'
                elif 'vol' in fname or 'atr' in fname or 'bb_' in fname or 'rv' in fname or 'parkinson' in fname:
                    tag = ' [VOL]'
                print(f"    {i:2d}. {fname}: {fimp:.4f}{tag}")

    # ══════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("DONE! 2h Volatility Model Summary")
    print("="*60)
    print(f"  Horizon: {args.horizon}h")
    print(f"  Features: {len(reg_features)}")
    print(f"  Vol regression:  {reg_name}, corr={reg_results[reg_name]['correlation']:.4f}, "
          f"R2={reg_results[reg_name]['r2']:.4f}")
    print(f"  Range regression: {rng_name}, corr={rng_results[rng_name]['correlation']:.4f}, "
          f"R2={rng_results[rng_name]['r2']:.4f}")
    print(f"  Vol classifier:  {cls_name}, acc={cls_results[cls_name]['accuracy']:.2%}")

    vzo_feats = [f for f in reg_features if 'vzo' in f or 'slope' in f]
    vol_feats = [f for f in reg_features
                 if any(k in f for k in ['vol', 'atr', 'bb_', 'rv5m', 'parkinson', 'body_ratio'])]
    print(f"  VZO/slope features: {len(vzo_feats)}")
    print(f"  Vol-specific features: {len(vol_feats)}")


if __name__ == '__main__':
    main()
