# PyQuotex Setup Guide

This guide will help you set up PyQuotex AI Trading Bot on your system.

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Linux Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Internet**: Stable broadband connection

### Recommended Requirements
- **OS**: Windows 11 or macOS 12+
- **Python**: 3.9 or 3.10
- **RAM**: 8GB or more
- **Storage**: 5GB free space
- **Internet**: High-speed connection

## 📦 Installation Methods

### Method 1: Quick Install (Recommended)

1. **Download the repository**:
   ```bash
   git clone https://github.com/yourusername/pyquotex.git
   cd pyquotex
   ```

2. **Run the setup script**:
   ```bash
   # Windows
   setup.bat
   
   # Linux/Mac
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Configure your credentials**:
   ```bash
   python app.py config
   ```

### Method 2: Manual Install

#### Step 1: Install Python

**Windows**:
1. Download Python from [python.org](https://python.org)
2. Run installer and check "Add Python to PATH"
3. Verify installation: `python --version`

**macOS**:
```bash
# Using Homebrew
brew install python

# Or download from python.org
```

**Linux**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# CentOS/RHEL
sudo yum install python3 python3-pip
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# Windows
myenv\Scripts\activate

# Linux/Mac
source myenv/bin/activate
```

#### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or install individually
pip install tensorflow numpy pandas requests websocket-client
```

#### Step 4: Configure Credentials

1. **Edit config file**:
   ```bash
   # Open config file
   notepad pyquotex/config.py  # Windows
   nano pyquotex/config.py     # Linux/Mac
   ```

2. **Add your credentials**:
   ```python
   def credentials():
       return "your_email@example.com", "your_password"
   ```

#### Step 5: Test Installation

```bash
# Test basic functionality
python app.py balance

# Test AI features
python app.py ml-status
```

## ⚙️ Configuration

### Basic Configuration

Edit `pyquotex/config.py`:

```python
def credentials():
    """Return your Quotex login credentials."""
    return "your_email@example.com", "your_password"

def trading_settings():
    """Default trading parameters."""
    return {
        "default_amount": 50,
        "default_asset": "EURUSD_otc",
        "default_duration": 60,
        "default_interval": 60
    }
```

### Advanced Configuration

Create `settings/config.ini`:

```ini
[trading]
default_amount = 50
default_asset = EURUSD_otc
default_duration = 60
default_interval = 60

[neural_network]
feature_size = 36
learning_rate = 0.001
batch_size = 32
epochs = 10

[monitoring]
rolling_window = 50
csv_file = trade_metrics.csv
log_level = INFO
```

## 🚀 First Run

### 1. Check Account Balance

```bash
python app.py balance
```

Expected output:
```
💰 Account Balance: R$ 100.00
```

### 2. Test Real-time Prices

```bash
python app.py realtime-price --asset EURUSD_otc
```

Expected output:
```
📊 EURUSD_otc: R$ 1.0850 (Live)
```

### 3. Start AI Learning

```bash
python app.py ml-status
```

Expected output:
```
🧠 AI Learning Status: Ready
📊 Model: Neural Network (36 features)
```

### 4. Begin Auto-trading

```bash
python app.py auto-trade --amount 10 --asset EURUSD_otc --duration 60 --interval 120
```

## 🔧 Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution**:
```bash
# Activate virtual environment
myenv\Scripts\activate  # Windows
source myenv/bin/activate  # Linux/Mac

# Install TensorFlow
pip install tensorflow
```

#### Issue: "ConnectionError: Failed to connect"

**Solution**:
1. Check internet connection
2. Verify credentials in `pyquotex/config.py`
3. Try different asset: `python app.py auto-trade --asset EURJPY_otc`

#### Issue: "AttributeError: 'PyQuotexCLI' object has no attribute 'model'"

**Solution**:
```bash
# Initialize the model
python app.py ml-status
```

#### Issue: "PermissionError: [Errno 13] Permission denied"

**Solution**:
```bash
# Run as administrator (Windows)
# Or fix permissions (Linux/Mac)
chmod +x app.py
```

### Performance Issues

#### Slow Neural Network Training

**Solution**:
1. **Reduce batch size**:
   ```python
   # In app.py, modify:
   self.incremental_batch_size = 16  # Instead of 32
   ```

2. **Use CPU optimization**:
   ```bash
   export TF_ENABLE_ONEDNN_OPTS=0
   ```

#### High Memory Usage

**Solution**:
1. **Reduce memory buffer**:
   ```python
   self.memory_buffer_size = 500  # Instead of 1000
   ```

2. **Clear old models**:
   ```bash
   python app.py ml-cleanup
   ```

### Network Issues

#### Connection Timeouts

**Solution**:
1. **Check firewall settings**
2. **Use VPN if needed**
3. **Try different time of day**

#### API Rate Limits

**Solution**:
1. **Increase interval between trades**:
   ```bash
   python app.py auto-trade --interval 300  # 5 minutes
   ```

2. **Use smaller amounts**:
   ```bash
   python app.py auto-trade --amount 25
   ```

## 📊 Monitoring Setup

### Enable CSV Logging

The bot automatically creates `trade_metrics.csv` with:
- Timestamp
- Rolling accuracy
- Total profit
- Trade details

### View Statistics

```bash
# View trading statistics
python app.py monitor-stats

# View AI performance
python app.py ml-analyze

# View neural network insights
python app.py nn-insights
```

### Set Up Alerts

Create `alerts.py`:

```python
import smtplib
from email.mime.text import MIMEText

def send_alert(message):
    # Configure your email settings
    sender = "your_email@gmail.com"
    password = "your_app_password"
    recipient = "your_email@gmail.com"
    
    msg = MIMEText(message)
    msg['Subject'] = 'PyQuotex Alert'
    msg['From'] = sender
    msg['To'] = recipient
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender, password)
    server.send_message(msg)
    server.quit()
```

## 🔒 Security

### Credential Security

1. **Never commit credentials** to version control
2. **Use environment variables**:
   ```bash
   export QUOTEX_EMAIL="your_email@example.com"
   export QUOTEX_PASSWORD="your_password"
   ```

3. **Use config files** with restricted permissions:
   ```bash
   chmod 600 pyquotex/config.py
   ```

### API Security

1. **Use strong passwords**
2. **Enable 2FA on Quotex account**
3. **Monitor account activity**
4. **Use VPN for additional security**

## 📈 Optimization

### Trading Optimization

1. **Start with small amounts** (R$ 10-25)
2. **Use longer intervals** (2-5 minutes)
3. **Monitor performance** regularly
4. **Adjust strategy** based on results

### AI Optimization

1. **Let it learn** for 50+ trades
2. **Monitor confidence** scores
3. **Adjust learning rate** if needed
4. **Save models** regularly

## 🆘 Getting Help

### Documentation
- **README.md**: Main documentation
- **CONTRIBUTING.md**: Contribution guidelines
- **docs/**: Detailed documentation

### Support Channels
- **GitHub Issues**: Bug reports and feature requests
- **Email**: alet8319@gmail.com
- **Phone**: +92-3478042183

### Community
- **Discussions**: GitHub Discussions
- **Wiki**: Community wiki
- **Discord**: Community server (if available)

## ✅ Verification

### Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Credentials configured
- [ ] Balance check successful
- [ ] Real-time prices working
- [ ] AI status showing
- [ ] First trade completed

### Performance Checklist

- [ ] Neural network training
- [ ] Pattern learning active
- [ ] Monitoring data saved
- [ ] No error messages
- [ ] Stable connection
- [ ] Profitable trades

---

**🎉 Congratulations! You're ready to start AI-powered trading!**

For more advanced usage, see the [README.md](README.md) and [docs/](docs/) directory.
