# ✈️ Aviator Predictor - Integration Complete

## What's New

Your Streamlit app (`aviator_ui_revised.py`) now includes **real Playwright-based browser automation** for Spin City login and multiplier scraping.

### Updated Files

1. **`aviator_ui_revised.py`** - Main app with integrated Playwright login
2. **`requirements.txt`** - Updated with `playwright` and `nest_asyncio`
3. **`PLAYWRIGHT_SETUP.md`** - Detailed setup instructions
4. **`setup_playwright.bat`** - One-click setup script (Windows)

## Key Features

✅ **Dual Prediction Models**
- Random Forest Regressor
- Neural Network (MLPRegressor)
- Combined average prediction with customizable threshold

✅ **Real Browser Automation**
- Uses Playwright Firefox headless browser
- Logs into Spin City with your credentials
- Navigates to Aviator game
- Captures live multiplier values
- Saves debug screenshots

✅ **Lightweight, Scroll-Friendly UI**
- Compact login section
- Clear prediction metrics
- Dataset summary with histogram
- Easy-to-use controls

✅ **Data Management**
- Auto-appends scraped values to `live_multipliers.csv`
- Trains models on historical data
- Retrains button for quick model updates

## Quick Start

### Option 1: Automatic Setup (Windows)
```bash
setup_playwright.bat
```

### Option 2: Manual Setup
```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Install Playwright & dependencies
pip install playwright nest_asyncio
playwright install firefox

# Run the app
streamlit run aviator_ui_revised.py
```

## Using the App

1. **Login & Scrape**
   - Enter your Spin City phone number (without +263)
   - Enter your password
   - Click "🚀 Login + Scrape Multipliers"
   - Wait 30-40 seconds for scraping to complete

2. **Make Predictions**
   - Paste recent multiplier values (comma-separated)
   - Click "Predict now"
   - See RF, NN, and Combined predictions
   - Check recommendation (HIGH/LOW) based on threshold

3. **Retrain Models**
   - Click "🔄 Retrain models" in sidebar
   - Models update with all historical + scraped data

## File Structure

```
aviator - final/
├── aviator_ui_revised.py          # ← Main Streamlit app (updated)
├── aviator_ui.py                  # Original UI (optional backup)
├── train_models_simple.py          # Model trainer utility
├── inspect_data.py                # Data inspector
├── requirements.txt                # Updated dependencies
├── PLAYWRIGHT_SETUP.md            # Setup instructions
├── setup_playwright.bat           # Windows setup script
├── compiled_spincity_data.csv     # Historical data
├── live_multipliers.csv           # Scraped data (auto-appended)
└── [model files] *.pkl            # Trained models (auto-saved)
```

## Technical Details

### Login Flow
```
1. Browser launches headless Firefox
2. Navigate to https://spincity.co.zw/login
3. Fill phone field with your number
4. Fill password field
5. Click LOGIN button
6. Wait for login to process
7. Navigate to Aviator game
8. Click PLAY button
9. Wait for game to load
10. Scrape multiplier values from page DOM
11. Return captured values for prediction
```

### Data Processing
- Raw multipliers → Feature engineering (rolling window stats)
- Train/test split (80/20, no shuffle for time series)
- StandardScaler normalization
- Models trained on 20 engineered features
- Predictions clamped to minimum 1.0x

### Async Handling
- `nest_asyncio.apply()` allows nested event loops in Streamlit
- `perform_login_scrape()` wraps async Playwright code
- Handles Streamlit's synchronous execution model

## Troubleshooting

### Q: "Playwright not installed"
**A:** Run `pip install playwright` then `playwright install firefox`

### Q: "No multiplier values found"
**A:** Check `logged_in_plane.png` in project folder for debugging. Verify credentials.

### Q: Connection timeout
**A:** Your network may be slow or Spin City server is slow. Wait longer or try during off-peak hours.

### Q: Models not predicting
**A:** Click "🔄 Retrain models" in sidebar to train with your data.

## Security & Privacy

- Credentials are **never stored** or logged
- Only transmitted to Spin City's official servers
- Headless browser session is isolated and temporary
- No third-party data collection

## Next Steps

1. Run `setup_playwright.bat` to install everything
2. Test the login with your Spin City credentials
3. Verify `logged_in_plane.png` shows the Aviator game
4. Scrape a batch of multipliers
5. Train models with "🔄 Retrain models"
6. Make your first prediction!

Enjoy! ✈️
