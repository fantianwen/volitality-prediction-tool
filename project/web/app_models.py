#!/usr/bin/env python3
"""
BTC Live Prediction Dashboard (Port 9001)

Fetches real-time Binance data, extracts multi-TF features using the same
vectorized pipeline from training, runs inference through all saved models
(direction + volatility), and serves a live dashboard UI.
"""

import glob
import json
import os
import pickle
import sys
import threading
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify
from flask_cors import CORS

warnings.filterwarnings('ignore')

# ── Add scripts directory to path so we can reuse training code ──
SCRIPTS_DIR = str(Path(__file__).parent.parent / 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from data_collector import BinanceDataFetcher, TechnicalIndicators
from collect_and_train import (
    extract_tf_features_vectorized,
    calculate_vzo,
    calculate_vzo_slope,
)
from collect_and_train_vol import add_vol_specific_features

# ── Constants ──
MODELS_DIR = Path(__file__).parent.parent / 'models'
UTC8 = timezone(timedelta(hours=8))
ALL_TFS = ['5m', '15m', '30m', '1h', '4h']
BASE_TF = '1h'

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

# ── Global state ──
fetcher = BinanceDataFetcher('BTCUSDT')
ti = TechnicalIndicators()
models = {}           # name -> {model, scaler, feature_names, meta}
results_cache = {}    # direction_results, vol_results  (from JSON)
predict_lock = threading.Lock()
last_prediction = {}  # cached latest prediction result
prediction_history = []  # list of recent predictions (kept in memory)
HISTORY_MAX = 48         # keep last 48 predictions (~48 hours)


# ====================================================================
# Model loading
# ====================================================================

def _load_pkl(pattern: str) -> dict | None:
    matches = sorted(glob.glob(str(MODELS_DIR / pattern)))
    if not matches:
        return None
    path = matches[-1]
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        data['_path'] = Path(path).name
        return data
    except Exception as e:
        print(f"  Warning: failed to load {path}: {e}")
        return None


def _load_json(pattern: str) -> dict | None:
    matches = sorted(glob.glob(str(MODELS_DIR / pattern)))
    if not matches:
        return None
    with open(matches[-1], 'r') as f:
        return json.load(f)


def load_all_models():
    """Load all model pkl files and result JSONs at startup."""
    global models, results_cache

    # Direction model
    d = _load_pkl('regression_model_*.pkl')
    if d:
        models['direction'] = {
            'model': d['best_model'],
            'scaler': d['scaler'],
            'feature_names': d['feature_names'],
            'model_name': d.get('best_model_name', '?'),
            'file': d['_path'],
        }
        print(f"  Loaded direction model: {d['_path']} "
              f"({d.get('best_model_name')}, {len(d['feature_names'])} features)")

    # Vol regression
    v = _load_pkl('vol_regression_*.pkl')
    if v:
        models['vol_regression'] = {
            'model': v['model'],
            'scaler': v['scaler'],
            'feature_names': v['feature_names'],
            'model_name': v.get('model_name', '?'),
            'file': v['_path'],
        }
        print(f"  Loaded vol regression: {v['_path']} "
              f"({v.get('model_name')}, {len(v['feature_names'])} features)")

    # Range regression
    r = _load_pkl('range_regression_*.pkl')
    if r:
        models['range_regression'] = {
            'model': r['model'],
            'scaler': r['scaler'],
            'feature_names': r['feature_names'],
            'model_name': r.get('model_name', '?'),
            'file': r['_path'],
        }
        print(f"  Loaded range regression: {r['_path']} "
              f"({r.get('model_name')}, {len(r['feature_names'])} features)")

    # Vol classifier
    c = _load_pkl('vol_classifier_*.pkl')
    if c:
        models['vol_classifier'] = {
            'model': c['model'],
            'scaler': c['scaler'],
            'feature_names': c['feature_names'],
            'model_name': c.get('model_name', '?'),
            'file': c['_path'],
        }
        print(f"  Loaded vol classifier: {c['_path']} "
              f"({c.get('model_name')}, {len(c['feature_names'])} features)")

    # Result JSONs (for model-info display)
    results_cache['direction'] = _load_json('regression_results_*.json')
    results_cache['volatility'] = _load_json('vol_results_*.json')

    print(f"  Total models loaded: {len(models)}")


# ====================================================================
# Live feature extraction  (replicates build_feature_matrix logic)
# ====================================================================

def build_live_features(df_dict: dict) -> pd.DataFrame:
    """
    Build a full feature matrix from live kline data, aligned to 1h base.
    Returns a DataFrame with the latest rows ready for inference.
    """
    base_df = df_dict[BASE_TF].copy()
    base_df['timestamp'] = pd.to_datetime(base_df['timestamp'])

    # Extract base TF features
    base_feat = extract_tf_features_vectorized(base_df, BASE_TF, ti)

    # Merge other timeframes via merge_asof
    for tf in ALL_TFS:
        if tf == BASE_TF:
            continue
        if tf not in df_dict or len(df_dict[tf]) < 50:
            continue
        tf_df = df_dict[tf].copy()
        tf_df['timestamp'] = pd.to_datetime(tf_df['timestamp'])
        tf_feat = extract_tf_features_vectorized(tf_df, tf, ti)

        tf_feat['timestamp'] = pd.to_datetime(tf_feat['timestamp'])
        base_sorted = base_feat.sort_values('timestamp').reset_index(drop=True)
        tf_sorted = tf_feat.sort_values('timestamp').reset_index(drop=True)

        merged = pd.merge_asof(
            base_sorted[['timestamp']],
            tf_sorted,
            on='timestamp',
            direction='backward',
        )
        for col in tf_feat.columns:
            if col != 'timestamp' and col not in base_feat.columns:
                base_feat[col] = merged[col].values

    # Cross-TF VZO consensus features
    vzo_tfs = [t for t in ALL_TFS if f'{t}_vzo_zone' in base_feat.columns]
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

        short_tfs = [t for t in ['5m', '15m'] if f'{t}_vzo' in base_feat.columns]
        long_tfs = [t for t in ['1h', '4h'] if f'{t}_vzo' in base_feat.columns]
        if short_tfs and long_tfs:
            base_feat['vzo_short_long_diff'] = (
                base_feat[[f'{t}_vzo' for t in short_tfs]].mean(axis=1) -
                base_feat[[f'{t}_vzo' for t in long_tfs]].mean(axis=1))
            base_feat['slope_short_long_diff'] = (
                base_feat[[f'{t}_vzo_slope' for t in short_tfs]].mean(axis=1) -
                base_feat[[f'{t}_vzo_slope' for t in long_tfs]].mean(axis=1))

    # Cross signals
    golden_cols = [f'{t}_kdj_golden' for t in ALL_TFS if f'{t}_kdj_golden' in base_feat.columns]
    golden_cols += [f'{t}_macd_golden' for t in ALL_TFS if f'{t}_macd_golden' in base_feat.columns]
    death_cols = [f'{t}_kdj_death' for t in ALL_TFS if f'{t}_kdj_death' in base_feat.columns]
    death_cols += [f'{t}_macd_death' for t in ALL_TFS if f'{t}_macd_death' in base_feat.columns]

    base_feat['multi_tf_golden_count'] = base_feat[golden_cols].sum(axis=1) if golden_cols else 0
    base_feat['multi_tf_death_count'] = base_feat[death_cols].sum(axis=1) if death_cols else 0
    base_feat['signal_strength'] = base_feat['multi_tf_golden_count'] - base_feat['multi_tf_death_count']

    # Time features
    ts = pd.to_datetime(base_feat['timestamp'])
    base_feat['hour'] = ts.dt.hour
    base_feat['day_of_week'] = ts.dt.weekday
    base_feat['is_weekend'] = (ts.dt.weekday >= 5).astype(int)

    # Add base_timestamp and close_price for vol-specific features
    base_feat['close_price'] = base_df['close'].values[:len(base_feat)]
    base_feat['base_timestamp'] = base_df['timestamp'].values[:len(base_feat)]

    # Add vol-specific features
    base_feat = add_vol_specific_features(base_feat, df_dict)

    # Drop timestamp column, clean up
    base_feat = base_feat.drop(columns=['timestamp'], errors='ignore')
    base_feat = base_feat.fillna(0).replace([np.inf, -np.inf], 0)

    return base_feat


def run_inference(feature_row: pd.DataFrame, model_key: str) -> float | None:
    """
    Run inference for a single model.
    Aligns feature columns, scales, and predicts.
    """
    if model_key not in models:
        return None

    m = models[model_key]
    expected_features = m['feature_names']

    # Build a single-row DataFrame aligned to expected features
    row = feature_row.copy()
    for col in expected_features:
        if col not in row.columns:
            row[col] = 0
    row = row[expected_features]

    # Replace any remaining NaN/inf
    row = row.fillna(0).replace([np.inf, -np.inf], 0)

    X = m['scaler'].transform(row.values)
    return m['model'].predict(X)


# ====================================================================
# VZO extraction for display
# ====================================================================

def extract_vzo_readings(df_dict: dict) -> list:
    """Extract current VZO and slope for each timeframe."""
    readings = []
    for tf in ALL_TFS:
        if tf not in df_dict or len(df_dict[tf]) < 20:
            readings.append({'tf': tf, 'vzo': None, 'slope': None, 'zone': 'N/A'})
            continue
        df = df_dict[tf].copy()
        vzo, vzo_ma = calculate_vzo(df)
        slope = calculate_vzo_slope(vzo, lookback=5)

        v = float(vzo.iloc[-1]) if not np.isnan(vzo.iloc[-1]) else 0
        s = float(slope.iloc[-1]) if not np.isnan(slope.iloc[-1]) else 0

        if v > 40:
            zone = 'Overbought'
        elif v > 15:
            zone = 'Bullish'
        elif v > -15:
            zone = 'Neutral'
        elif v > -40:
            zone = 'Bearish'
        else:
            zone = 'Oversold'

        readings.append({
            'tf': tf,
            'vzo': round(v, 2),
            'vzo_ma': round(float(vzo_ma.iloc[-1]), 2) if not np.isnan(vzo_ma.iloc[-1]) else 0,
            'slope': round(s, 3),
            'zone': zone,
        })
    return readings


# ====================================================================
# Routes
# ====================================================================

@app.route('/')
def index():
    return render_template('models_dashboard.html')


def _run_prediction() -> dict:
    """Core prediction logic. Returns result dict."""
    t0 = time.time()

    # 1. Fetch live klines
    df_dict = {}
    for tf in ALL_TFS:
        df_dict[tf] = fetcher.get_klines(tf, limit=200)

    current_price = float(df_dict['1h']['close'].iloc[-1])
    fetch_time = time.time() - t0

    # Determine the last 1h candle close timestamp (UTC)
    last_1h_ts = pd.to_datetime(df_dict['1h']['timestamp'].iloc[-1])
    hour_utc = last_1h_ts.hour
    is_4h_boundary = (hour_utc % 4 == 0)

    # 2. Build features
    t1 = time.time()
    feat_df = build_live_features(df_dict)
    latest = feat_df.iloc[[-1]].copy()
    feat_time = time.time() - t1

    # 3. Run inference
    t2 = time.time()
    result = {
        'success': True,
        'timestamp': datetime.now(UTC8).isoformat(),
        'candle_timestamp': last_1h_ts.isoformat(),
        'current_price': round(current_price, 2),
        'is_4h_boundary': is_4h_boundary,
        'predictions': {},
        'vzo_readings': extract_vzo_readings(df_dict),
        'timing': {},
    }

    # Direction
    dir_pred = run_inference(latest, 'direction')
    if dir_pred is not None:
        pct_change = float(dir_pred[0])
        direction = 'Bullish' if pct_change > 0.5 else ('Bearish' if pct_change < -0.5 else 'Sideways')
        result['predictions']['direction'] = {
            'price_change_pct': round(pct_change, 4),
            'direction': direction,
            'predicted_price': round(current_price * (1 + pct_change / 100), 2),
            'model': models['direction']['model_name'],
        }

    # Vol regression
    vol_pred = run_inference(latest, 'vol_regression')
    if vol_pred is not None:
        result['predictions']['vol_regression'] = {
            'realized_vol': round(float(vol_pred[0]), 8),
            'model': models['vol_regression']['model_name'],
        }

    # Range regression
    range_pred = run_inference(latest, 'range_regression')
    if range_pred is not None:
        result['predictions']['range_regression'] = {
            'range_pct': round(float(range_pred[0]), 4),
            'range_usd': round(current_price * float(range_pred[0]) / 100, 2),
            'model': models['range_regression']['model_name'],
        }

    # Vol classifier
    cls_pred = run_inference(latest, 'vol_classifier')
    if cls_pred is not None:
        cls_val = int(cls_pred[0])
        prob = None
        try:
            m = models['vol_classifier']
            row = latest.copy()
            for col in m['feature_names']:
                if col not in row.columns:
                    row[col] = 0
            row = row[m['feature_names']].fillna(0).replace([np.inf, -np.inf], 0)
            X = m['scaler'].transform(row.values)
            proba = m['model'].predict_proba(X)
            prob = round(float(proba[0][cls_val]), 4)
        except Exception:
            pass

        result['predictions']['vol_classifier'] = {
            'class': 'High Volatility' if cls_val == 1 else 'Low Volatility',
            'class_int': cls_val,
            'probability': prob,
            'model': models['vol_classifier']['model_name'],
        }

    infer_time = time.time() - t2
    result['timing'] = {
        'fetch_s': round(fetch_time, 2),
        'features_s': round(feat_time, 2),
        'inference_s': round(infer_time, 2),
        'total_s': round(time.time() - t0, 2),
    }

    return result


def _store_prediction(result: dict):
    """Save to in-memory history (most recent first)."""
    global last_prediction
    last_prediction = result
    prediction_history.insert(0, {
        'timestamp': result['timestamp'],
        'candle_timestamp': result.get('candle_timestamp'),
        'is_4h_boundary': result.get('is_4h_boundary', False),
        'current_price': result['current_price'],
        'predictions': result['predictions'],
    })
    # Trim history
    while len(prediction_history) > HISTORY_MAX:
        prediction_history.pop()


# ---------------------------------------------------------------------------
# Background prediction loop -- runs at each 1h candle close
# ---------------------------------------------------------------------------

def _start_background_predict_loop():
    """
    Predict once at startup, then schedule predictions aligned to 1h candle
    closes (at HH:01:00 UTC to allow the candle to finalize).
    """
    def loop():
        # Initial prediction after short startup delay
        time.sleep(3)
        try:
            with predict_lock:
                result = _run_prediction()
                _store_prediction(result)
                tag = " [4H BOUNDARY]" if result.get('is_4h_boundary') else ""
                print(f"  Prediction @ startup: ${result['current_price']}{tag}")
        except Exception as e:
            print(f"  Warning: startup prediction failed: {e}")

        while True:
            try:
                # Wait until ~1 minute past the next hour (UTC)
                now = datetime.now(timezone.utc)
                next_hour = now.replace(minute=1, second=0, microsecond=0)
                if next_hour <= now:
                    next_hour += timedelta(hours=1)
                wait_secs = (next_hour - now).total_seconds()
                print(f"  Next prediction in {wait_secs:.0f}s "
                      f"(at {next_hour.strftime('%H:%M')} UTC)")
                time.sleep(wait_secs)

                # Run prediction
                with predict_lock:
                    result = _run_prediction()
                    _store_prediction(result)

                is_4h = result.get('is_4h_boundary', False)
                tag = " *** 4H BOUNDARY ***" if is_4h else ""
                p = result.get('predictions', {})
                dir_str = ''
                if 'direction' in p:
                    d = p['direction']
                    dir_str = f" | {d['direction']} {d['price_change_pct']:+.3f}%"

                print(f"  Prediction @ {result['candle_timestamp']}: "
                      f"${result['current_price']}{dir_str}{tag}")

            except Exception as e:
                print(f"  Warning: prediction loop error: {e}")
                time.sleep(60)  # backoff on error

    t = threading.Thread(target=loop, name="predict_loop", daemon=True)
    t.start()


@app.route('/api/predict')
def predict():
    """Return the latest cached prediction, or run one now if none exists."""
    global last_prediction
    if last_prediction and last_prediction.get('success'):
        return jsonify(last_prediction)

    # No cached prediction yet -- run one now
    try:
        with predict_lock:
            result = _run_prediction()
            _store_prediction(result)
            return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict/now')
def predict_now():
    """Force a fresh prediction (manual trigger)."""
    try:
        with predict_lock:
            result = _run_prediction()
            _store_prediction(result)
            return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict/history')
def predict_history():
    """Return recent prediction history."""
    return jsonify({
        'success': True,
        'count': len(prediction_history),
        'predictions': prediction_history,
    })


@app.route('/api/model-status')
def model_status():
    """Return info about loaded models."""
    info = {}
    for key, m in models.items():
        info[key] = {
            'file': m['file'],
            'model_name': m['model_name'],
            'feature_count': len(m['feature_names']),
        }
    return jsonify({
        'success': True,
        'models': info,
        'training_results': {
            'direction': results_cache.get('direction'),
            'volatility': results_cache.get('volatility'),
        },
        'timestamp': datetime.now(UTC8).isoformat(),
    })


@app.route('/api/vzo-multi-tf')
def vzo_multi_tf():
    """Get current VZO/slope across all timeframes."""
    try:
        df_dict = {}
        for tf in ALL_TFS:
            df_dict[tf] = fetcher.get_klines(tf, limit=100)
        readings = extract_vzo_readings(df_dict)
        return jsonify({
            'success': True,
            'readings': readings,
            'timestamp': datetime.now(UTC8).isoformat(),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ====================================================================
# Main
# ====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BTC Live Prediction Dashboard')
    parser.add_argument('--port', '-p', type=int, default=9001)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print(f"Models directory: {MODELS_DIR}")
    print("Loading models...")
    load_all_models()

    # Start background prediction loop (aligned to 1h candle closes)
    if not args.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _start_background_predict_loop()

    print(f"\nStarting Live Prediction Dashboard on http://localhost:{args.port}")
    print("  Predictions run automatically at each 1h candle close (HH:01 UTC)")
    print("  4h boundary predictions (00, 04, 08, 12, 16, 20 UTC) are highlighted")
    app.run(host=args.host, port=args.port, debug=args.debug)
