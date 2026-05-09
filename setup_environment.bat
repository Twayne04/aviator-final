@echo off
echo ========================================
echo Aviator Predictor - Environment Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing required packages...
pip install -r requirements.txt

echo [4/4] Verifying installation...
python -c "import streamlit; import pandas; import sklearn; print('All packages installed successfully!')"

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To run the Aviator Predictor:
echo   1. Activate the environment: venv\Scripts\activate
echo   2. Run the app: streamlit run aviator_ui.py
echo.
pause