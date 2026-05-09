@echo off
REM Playwright & Streamlit Setup Script for Aviator Predictor

echo.
echo ========================================
echo Aviator Predictor - Playwright Setup
echo ========================================
echo.

REM Check if venv exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv
        pause
        exit /b 1
    )
)

REM Activate venv
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate venv
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip -q

REM Install requirements
echo Installing base requirements (streamlit, pandas, scikit-learn, numpy)...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

REM Install Playwright and nest_asyncio
echo Installing Playwright and nest_asyncio...
pip install -q playwright nest_asyncio
if errorlevel 1 (
    echo ERROR: Failed to install Playwright and nest_asyncio
    echo Try installing manually: pip install playwright nest_asyncio
    pause
    exit /b 1
)

REM Install Playwright browser binaries
echo Downloading Firefox for Playwright (this may take a few minutes)...
playwright install firefox
if errorlevel 1 (
    echo WARNING: Playwright browser installation had issues
    echo You may need to run: playwright install firefox
    echo.
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To run the app:
echo   streamlit run aviator_ui_revised.py
echo.
echo For Playwright setup help, see: PLAYWRIGHT_SETUP.md
echo.
pause
