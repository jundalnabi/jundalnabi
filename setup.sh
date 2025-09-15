#!/bin/bash

echo "========================================"
echo "PyQuotex AI Trading Bot Setup"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "Python found:"
python3 --version
echo

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv myenv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source myenv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p models patterns settings

# Create config file if it doesn't exist
if [ ! -f "pyquotex/config.py" ]; then
    echo "Creating config file..."
    cat > pyquotex/config.py << EOF
def credentials():
    return "your_email@example.com", "your_password"
EOF
fi

echo
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Edit pyquotex/config.py with your Quotex credentials"
echo "2. Run: python app.py balance"
echo "3. Start trading: python app.py auto-trade --amount 10"
echo
echo "For help, run: python app.py --help"
echo
