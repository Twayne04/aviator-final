# Playwright Setup Guide

The updated `aviator_ui_revised.py` includes real browser automation via **Playwright** to login and scrape multiplier data from Spin City.

## Installation

### 1. Install Playwright and nest_asyncio

```bash
# Activate your virtual environment first
.\.venv\Scripts\activate

# Then install the packages
pip install playwright nest_asyncio
```

### 2. Install Browser Binaries

After pip install, you must download the browser binaries:

```bash
playwright install firefox
```

This downloads Firefox (~200MB) which is required for automated login.

## Running the App

```bash
# Activate venv
.\.venv\Scripts\activate

# Run the Streamlit app
streamlit run aviator_ui_revised.py
```

## Using the Login Feature

1. Open the app in your browser (usually `http://localhost:8501`)
2. In the **🔐 Login & Scrap Data** section, enter:
   - **Phone Number:** Your Spin City phone (without the +263 prefix)
     - Example: If your number is `+263771234567`, enter `771234567`
   - **Password:** Your Spin City account password
3. Click **🚀 Login + Scrape Multipliers**
4. Wait 30-40 seconds for the bot to:
   - Login to Spin City
   - Navigate to the Aviator game
   - Capture recent multiplier values
5. A screenshot (`logged_in_plane.png`) will be saved for debugging

## Troubleshooting

### `ModuleNotFoundError: No module named 'playwright'`
- Run `pip install playwright`
- Then run `playwright install firefox`

### `Playwright not installed` error in Streamlit
- Same as above—the packages need to be installed in your `.venv`

### Scraper captures no multipliers
- Check `logged_in_plane.png` in the project folder
- Verify your phone number and password are correct
- Make sure you're using the exact phone format (no +263)
- Network connectivity to Spin City may be required

### Playwright browser download fails
- If network is slow, try:
  ```bash
  pip install --index-url https://pypi.org/simple playwright
  playwright install firefox --with-deps
  ```

## What Happens Behind the Scenes

1. Browser launches in **headless mode** (no visible window)
2. Navigates to `https://spincity.co.zw/login`
3. Fills phone field with your credentials
4. Clicks the LOGIN button
5. Waits for login to complete
6. Navigates to Aviator game at `https://spincity.co.zw/games/Aviator-6094`
7. Clicks PLAY to start the engine
8. Scrapes multiplier values from the page
9. Returns captured multipliers for prediction

All multipliers are stored in `live_multipliers.csv` and used by the RF/NN models.

## Security Note

Your credentials are **not stored** by the app—they're only used for the Playwright browser session and never logged or transmitted anywhere except to Spin City's login endpoint.
