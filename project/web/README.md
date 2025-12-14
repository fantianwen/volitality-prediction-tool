# BTC Price Prediction - Web Dashboard

A beautiful web interface to visualize BTC price predictions and trading recommendations.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd project/web
pip install -r requirements.txt

# Also install shared dependencies
pip install -r ../scripts/requirements.txt
```

### 2. Configure Environment

Set the model path (optional, defaults to `../models/regression_model_20251213_213205.pkl`):

```bash
export MODEL_PATH=../models/regression_model_20251213_213205.pkl
```

### 3. Run Web Server

```bash
python3 app.py
```

The dashboard will be available at: **http://localhost:8080** (default port)

## 📊 Features

- **Real-time Prediction**: Execute predictions on demand
- **Price Display**: Current and predicted prices with visual indicators
- **Confidence Metrics**: Signal strength and confidence scores
- **Position Recommendations**: Position size and leverage suggestions
- **Market Status**: RSI, ADX, and volatility indicators
- **Prediction History**: View last 50 predictions
- **Auto Refresh**: Optional 30-second auto-refresh

## 🎨 UI Components

### Price Card
- Current BTC price
- Predicted price (20 hours ahead)
- Price change percentage with color coding

### Prediction Card
- Direction (Bullish/Bearish/Sideways)
- Price change range
- Prediction timestamps (UTC+8)

### Confidence Card
- Confidence score (0-100%)
- Signal strength (-12 to +12)
- Visual progress bars

### Position Recommendation Card
- Recommended position size (%)
- Suggested leverage (x)
- Risk score
- Trading recommendation level

### Market Status Card
- RSI (1h) with overbought/oversold indicators
- ADX (1h) trend strength
- Volatility percentage

### History Card
- Last 50 predictions
- Timestamp, direction, confidence
- Signal strength

## 🔧 Configuration

### Environment Variables

- `MODEL_PATH`: Path to the trained model file
- `PORT`: Web server port (default: 8080)
- `FLASK_DEBUG`: Enable debug mode (default: False)

### Command Line Arguments

```bash
python3 app.py --port 3000          # Set port to 3000
python3 app.py --host 127.0.0.1      # Bind to localhost only
python3 app.py --debug               # Enable debug mode
python3 app.py -p 5000 --debug       # Combine options
```

### API Endpoints

- `GET /` - Web dashboard
- `GET /api/predict` - Execute prediction
- `GET /api/history` - Get prediction history
- `GET /api/status` - Server status

## 📱 Usage

1. Open http://localhost:5000 in your browser
2. Click "Run Prediction Now" to execute a prediction
3. Enable "Auto Refresh" for automatic updates every 30 seconds
4. View prediction history and market status

## 🎯 Features in Detail

### Signal Strength Calculation

Signal strength = (Golden Cross Count) - (Death Cross Count)

- **Range**: -12 to +12
- **Positive**: Bullish signals (more golden crosses)
- **Negative**: Bearish signals (more death crosses)
- **Zero**: Neutral (equal signals)

### Position Recommendations

Based on:
- Signal strength (40% weight)
- Confidence level (40% weight)
- Prediction magnitude (20% weight)

Risk levels:
- **Conservative**: Reduced position, lower leverage
- **Moderate**: Standard configuration
- **Aggressive**: Increased position, higher leverage

## 🐛 Troubleshooting

### Model Not Found

Ensure the model file exists at the specified path:
```bash
ls -lh ../models/regression_model_*.pkl
```

### Port Already in Use

Change the port:
```bash
PORT=8080 python3 app.py
```

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
pip install -r ../scripts/requirements.txt
```

## 🔄 Running in Background

### Using Systemd Service (Recommended for Production)

```bash
# Start service
sudo systemctl start btc-predictor-web

# Check status
sudo systemctl status btc-predictor-web

# View logs
tail -f /home/ubuntu/btc-predictor/logs/web_ui.log

# Enable auto-start on boot
sudo systemctl enable btc-predictor-web
```

### Using nohup (Quick Test)

```bash
cd /home/ubuntu/btc-predictor/web
source ../venv/bin/activate
nohup python3 app.py --port 8080 > ../logs/web_ui.log 2>&1 &
```

### Using tmux/screen (For Debugging)

```bash
# Create tmux session
tmux new -s webui

# Start Web UI
cd /home/ubuntu/btc-predictor/web
source ../venv/bin/activate
python3 app.py --port 8080

# Detach: Ctrl+B then D
# Reattach: tmux attach -t webui
```

For more details, see [RUN_BACKGROUND.md](RUN_BACKGROUND.md)

## 📝 Notes

- Predictions are executed in real-time (may take 3-5 seconds)
- All times displayed in UTC+8 (Asia/Shanghai)
- Position recommendations are for reference only
- Auto-refresh interval: 30 seconds

