@echo off
echo ========================================
echo PyQuotex AI Trading Bot Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv myenv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call myenv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Create necessary directories
echo Creating directories...
if not exist "models" mkdir models
if not exist "patterns" mkdir patterns
if not exist "settings" mkdir settings

REM Create config file if it doesn't exist
if not exist "pyquotex\config.py" (
    echo Creating config file...
    echo def credentials(): > pyquotex\config.py
    echo     return "your_email@example.com", "your_password" >> pyquotex\config.py
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Edit pyquotex\config.py with your Quotex credentials
echo 2. Run: python app.py balance
echo 3. Start trading: python app.py auto-trade --amount 10
echo.
echo For help, run: python app.py --help
echo.
pause
