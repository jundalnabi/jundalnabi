# PyQuotex - AI-Powered Trading Bot with Neural Network!

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

An advanced AI-powered trading bot for Quotex platform featuring real-time neural network learning, pattern recognition, and comprehensive monitoring.

## 🚀 Features

### 🧠 Advanced AI & Machine Learning
- **Deep Neural Network**: 5-layer architecture with 20,000+ parameters
- **Real-time Learning**: Learns from every trade instantly
- **Pattern Recognition**: Identifies market patterns and trends
- **Confidence Scoring**: Knows how confident each prediction is
- **Adaptive Training**: Automatically retrains based on performance
- **Incremental Learning**: Continuous learning without forgetting

### 📊 Comprehensive Monitoring
- **Rolling Accuracy**: Tracks success rate of last 50 trades
- **Profit Tracking**: Real-time profit/loss monitoring
- **Streak Analysis**: Win/loss streak tracking
- **CSV Export**: All data saved for analysis
- **Performance Insights**: Detailed statistics and recommendations

### 🔄 Smart Trading Features
- **Auto-trading**: Automated trade execution
- **Risk Management**: Martingale strategy with smart recovery
- **Multiple Assets**: Support for various trading pairs
- **Technical Indicators**: RSI, MACD, Stochastic, SMA, EMA
- **Strategy Adaptation**: Aggressive/Balanced/Conservative modes

## 📋 Requirements

- Python 3.8 or higher
- Windows 10/11 (PowerShell)
- Quotex account
- Internet connection
- VPN for bot and website

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/pyquotex.git
cd pyquotex
```

### 2. Create Virtual Environment
```bash
# Windows PowerShell
python -m venv myenv
myenv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Credentials
Edit `pyquotex/config.py` and add your Quotex credentials:
```python
def credentials():
    return "your_email@example.com", "your_password"
```

## 🚀 Quick Start

### Basic Trading
```bash
# Start auto-trading
python app.py auto-trade --amount 50 --asset EURUSD_otc --duration 60 --interval 60

# Check account balance
python app.py balance

# View real-time prices
python app.py realtime-price --asset EURJPY_otc
```

### AI & Machine Learning
```bash
# Check AI status
python app.py ml-status

# View learned patterns
python app.py ml-patterns

# Analyze performance
python app.py ml-analyze

# Neural network insights
python app.py nn-insights

# Monitor trading statistics
python app.py monitor-stats
```

## 📖 Commands Reference

### Trading Commands
| Command | Description | Example |
|---------|-------------|---------|
| `auto-trade` | Start automated trading | `python app.py auto-trade --amount 50` |
| `balance` | Check account balance | `python app.py balance` |
| `realtime-price` | View live prices | `python app.py realtime-price --asset EURUSD_otc` |
| `signals` | Get trading signals | `python app.py signals` |

### AI & ML Commands
| Command | Description | Example |
|---------|-------------|---------|
| `ml-status` | Show AI learning status | `python app.py ml-status` |
| `ml-analyze` | Analyze trading performance | `python app.py ml-analyze` |
| `ml-patterns` | View learned patterns | `python app.py ml-patterns` |
| `ml-save` | Save AI model | `python app.py ml-save --name "my_model"` |
| `ml-save-all` | Save all learning data | `python app.py ml-save-all` |
| `nn-insights` | Neural network analysis | `python app.py nn-insights` |
| `monitor-stats` | Trading statistics | `python app.py monitor-stats` |

### Advanced ML Commands
| Command | Description | Example |
|---------|-------------|---------|
| `ml-incremental` | Manage incremental learning | `python app.py ml-incremental --replay` |
| `ml-incremental --buffer-size 2000` | Set memory buffer size | `python app.py ml-incremental --buffer-size 2000` |

## 🧠 Neural Network Architecture

### Model Structure
```
Input Layer (36 features)
    ↓
Batch Normalization
    ↓
Dense Layer (128 units) + Dropout(0.3) + BatchNorm
    ↓
Dense Layer (64 units) + Dropout(0.2) + BatchNorm
    ↓
Dense Layer (32 units) + Dropout(0.2) + BatchNorm
    ↓
Dense Layer (16 units) + Dropout(0.1)
    ↓
Output Layer (1 unit, sigmoid)
```

### Features (36 total)
- **Candle Data**: Open, Close, High, Low, Volume (5 candles × 5 = 25)
- **Technical Indicators**: SMA, EMA, RSI (3 each = 9)
- **MACD**: Signal, Histogram (2)
- **Previous Trade**: Result, Profit (2)
- **Time**: Hour of day (1)

### Advanced Features
- **Batch Normalization**: Prevents internal covariate shift
- **Dropout Regularization**: Prevents overfitting
- **Learning Rate Decay**: Adaptive learning rate
- **Early Stopping**: Prevents overtraining
- **Model Checkpointing**: Saves best model

## 📊 Monitoring & Analytics

### Real-time Display
```
[Neural Network] Prediction: CALL (confidence: 85%)
📊 [2024-09-14] Rolling Accuracy (last 5 trades): 80.00%
💰 [2024-09-14] Last Trade Profit: R$ 5.00, Total Profit: R$ 25.00
📈 [2024-09-14] Trade #5 | Win Streak: 3 | Loss Streak: 0
🧠 [2024-09-14] Strategy: AGGRESSIVE | Learning Confidence: 75%
```

### CSV Export
All trading data is automatically saved to `trade_metrics.csv`:
- Timestamp
- Rolling accuracy
- Total profit
- Last trade profit
- Trade count
- Win/loss streaks
- Strategy
- Learning confidence

## ⚙️ Configuration

### Trading Parameters
```python
# In auto_trade method
amount = 50          # Trade amount
asset = "EURUSD_otc" # Trading pair
duration = 60        # Trade duration (seconds)
interval = 60        # Time between trades (seconds)
```

### AI Parameters
```python
# Neural network settings
feature_size = 36    # Input features
memory_buffer_size = 1000  # Experience replay buffer
incremental_batch_size = 32  # Mini-batch size
learning_rate = 0.001  # Initial learning rate
```

## 🔧 Advanced Usage

### Custom Model Training
```python
# The bot automatically:
# 1. Learns from every trade
# 2. Retrains when performance drops
# 3. Saves models automatically
# 4. Adapts strategy based on results
```

### Pattern Learning
The bot learns and recognizes:
- **Candle Patterns**: Doji, Hammer, Engulfing
- **Indicator Patterns**: RSI divergence, MACD crossovers
- **Time Patterns**: Best trading hours
- **Market Conditions**: Volatility, trends

### Strategy Adaptation
- **Aggressive**: High confidence, larger positions
- **Balanced**: Moderate risk, steady growth
- **Conservative**: Low risk, small positions

## 📁 Project Structure

```
pyquotex/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── README.md             # This file
├── LICENSE               # MIT License
├── pyquotex/            # Core library
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   └── ...
├── examples/            # Example scripts
├── docs/               # Documentation
└── models/             # Saved AI models (auto-created)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This software is for educational purposes only. Trading involves risk and you may lose money. Use at your own risk. The authors are not responsible for any financial losses.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Email**: alet8319@gmail.com
- **Phone**: +92-3478042183
- **Issues**: [GitHub Issues](https://github.com/jundalnabi/jundalnabi/issues)

## 🙏 Acknowledgments

- Team Jund Al Nabi for the original PyQuotex library
- TensorFlow team for the machine learning framework
- The open-source community for various dependencies

---

**⭐ If you find this project helpful, please give it a star!**
