# BTC Price Prediction Tool

A machine learning-based BTC price movement prediction system using multi-timeframe technical indicators and GBM models.

## 📋 Features

- 📊 **Multi-Timeframe Analysis**: 5m, 15m, 30m, 1h, 4h, 1d
- 🔧 **Technical Indicators**: KDJ, MACD, RSI, ROC, MOM, Williams %R, CCI, ADX, Stochastic RSI
- 🤖 **ML Models**: GBM, Random Forest, LSTM, Ensemble
- ⏰ **Auto Prediction**: Automatic prediction every 30 minutes with reports
- 📱 **Telegram Integration**: Real-time prediction reports and trading recommendations
- 💰 **Position Management**: Position sizing and leverage recommendations based on signal strength and confidence

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd project/scripts
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
nano .env
```

Fill in Telegram configuration:
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. Run Prediction Service

```bash
# Test mode
python3 prediction_server.py --model ../models/regression_model_20251213_213205.pkl --test

# Production mode (predict every 30 minutes)
python3 prediction_server.py --model ../models/regression_model_20251213_213205.pkl
```

### 4. Start Telegram Bot (Optional)

```bash
# Listen for commands (e.g., /predict-now)
python3 telegram_bot.py --model ../models/regression_model_20251213_213205.pkl
```

## 📁 Project Structure

```
.
├── readMe.md                          # Project design document
├── project/
│   ├── scripts/                       # Main scripts
│   │   ├── data_collector.py          # Data collection and feature extraction
│   │   ├── train_model.py             # Model training
│   │   ├── prediction_server.py       # Prediction server
│   │   ├── telegram_bot.py            # Telegram Bot
│   │   ├── position_manager.py       # Position management
│   │   └── requirements.txt           # Dependencies
│   ├── models/                        # Trained models (not in Git)
│   ├── data/                          # Historical data (not in Git)
│   └── release/                       # Deployment files
│       ├── deploy_to_aws.sh           # AWS deployment script
│       └── AWS_DEPLOYMENT.md         # Deployment documentation
```

## 📊 Prediction Report Contents

- Current price and predicted price
- Prediction direction (Bullish/Bearish/Sideways)
- Price change range (Large Drop/Small Drop/Sideways/Small Rise/Large Rise)
- Predicted percentage change
- Confidence score
- Signal strength
- Market status (RSI, ADX, Volatility)
- **Position and leverage recommendations** (based on signal strength and confidence)

## 🔧 Main Features

### Data Collection

```bash
# Collect historical data
python3 data_collector.py --mode historical \
    --start-date 2024-01-01 \
    --timeframe 1h \
    --lookforward 20
```

### Model Training

```bash
# Train model
python3 train_model.py \
    --data ../data/BTCUSDT_features_1h_*.csv \
    --output ../models \
    --models gbm \
    --task regression
```

### Prediction Service

```bash
# Start prediction service
python3 prediction_server.py \
    --model ../models/regression_model_xxx.pkl \
    --risk-level moderate
```

### Telegram Bot Commands

- `/predict-now` - Execute prediction immediately
- `/start` - Show welcome message
- `/help` - Show help information

## 📖 Documentation

- [Project Design Document](readMe.md)
- [Position Management Guide](project/scripts/POSITION_MANAGEMENT_GUIDE.md)
- [Telegram Bot Usage](project/scripts/TELEGRAM_BOT_README.md)
- [AWS Deployment Guide](project/release/AWS_DEPLOYMENT.md)

## ⚙️ Configuration Options

### Risk Levels

- `conservative` - Conservative: Reduced position size and lower leverage
- `moderate` - Moderate: Default configuration (recommended)
- `aggressive` - Aggressive: Increased position size and higher leverage

### Prediction Frequency

Default: Every 30 minutes. Can be modified in the code.

## ⚠️ Important Notes

1. **Environment Variables**: Do not commit `.env` files to Git
2. **Sensitive Information**: `aws_account.md` and `*.pem` files are ignored
3. **Model Files**: Model files are large, consider using Git LFS or separate storage
4. **Data Files**: Historical data files are not committed to Git

## 📝 License

This project is for learning and research purposes only.

## 🤝 Contributing

Issues and Pull Requests are welcome.
