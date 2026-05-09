# 🧪 Testing Login & Scrape Function

## Prerequisites

Before running the test, you need Playwright and Firefox installed:

```bash
# 1. Install Playwright and nest_asyncio
pip install playwright nest_asyncio

# 2. Install Firefox browser for Playwright
playwright install firefox
```

## Running the Test

### Option 1: Direct Run
```bash
# Activate your virtual environment
.\.venv\Scripts\activate

# Run the test script
python test_login_scrape.py
```

### Option 2: From Project Root
```bash
.\.venv\Scripts\python.exe test_login_scrape.py
```

## What the Test Does

The test script will:

1. ✅ Launch a headless Firefox browser
2. ✅ Navigate to Spin City login page
3. ✅ Fill in your phone number (without +263)
4. ✅ Fill in your password
5. ✅ Click the LOGIN button
6. ✅ Wait for authentication (8 seconds)
7. ✅ Navigate to Aviator game
8. ✅ Click PLAY button
9. ✅ Wait for game to load (20 seconds)
10. ✅ Scrape multiplier values from the page
11. ✅ Save debug screenshots
12. ✅ Return captured multipliers

## Test Output

The test will show you:
- Real-time status updates with emojis
- Each step as it completes
- Number of multiplier values found
- List of top 10 captured values
- Screenshot file locations

### Example Output:
```
============================================================
Testing Spin City Login & Scraping
============================================================

1️⃣  Navigating to login page...
   ✅ Login page loaded

2️⃣  Filling phone number...
   ✅ Phone filled: 771234567

3️⃣  Filling password...
   ✅ Password filled

4️⃣  Clicking LOGIN button...
   ✅ LOGIN button clicked

5️⃣  Waiting for login to process...
   ✅ Login processed (waiting complete)

6️⃣  Navigating to Aviator game...
   ✅ Aviator game page loaded

7️⃣  Clicking PLAY button...
   ✅ PLAY button clicked

8️⃣  Waiting for game to load...
   ✅ Game load complete

9️⃣  Scraping multiplier values...
   ✅ Found 25 items in frame
   ✅ Parsed 25 multiplier values

✅ SUCCESS! Scraped 18 unique multiplier values
============================================================

Top 10 values: [23.45, 19.82, 15.33, 12.01, 10.55, 8.22, 6.11, 4.88, 3.45, 2.67]
```

## Debug Screenshots

If the test fails or produces unexpected results, it saves screenshots:

- `test_after_login.png` - Page after login attempt
- `test_aviator_game.png` - Aviator game page
- `test_phone_error.png` - If phone field filling fails
- `test_password_error.png` - If password field filling fails
- `test_login_button_error.png` - If login button click fails
- `test_game_nav_error.png` - If game navigation fails

Open these images to see what the browser actually sees.

## Troubleshooting

### ❌ "Playwright not installed"
```bash
pip install playwright
playwright install firefox
```

### ❌ "No multiplier values found"
- Check `test_after_login.png` - did login work?
- Check `test_aviator_game.png` - did game load?
- Website selectors may have changed
- Try increasing wait times (20 → 30 seconds)

### ❌ "Login button not found"
- Phone number might be wrong
- Credentials might be incorrect
- Check `test_after_login.png` for the actual page

### ❌ Browser won't launch
- Ensure Firefox is installed: `playwright install firefox`
- Check system has enough resources
- Try running as administrator

### ⏳ Test is very slow
- Normal for first run (Firefox install)
- Network connectivity to Spin City
- Server response time

## Success Criteria

The test is successful if:
1. ✅ It prints "SUCCESS!"
2. ✅ Shows a list of multiplier values
3. ✅ Returns at least 1 unique multiplier
4. ✅ You can proceed to use the app

## Next Steps

Once the test passes:

1. Run the main app:
   ```bash
   streamlit run aviator_ui_revised.py
   ```

2. Use the app's login section with the same credentials

3. Data will be scraped and used for predictions

## Advanced: Modify Selectors

If the test fails but the page loads correctly in screenshots, the selectors may have changed.

Edit `test_login_scrape.py` and update these selectors:

```python
# Phone field
await page.fill("input[placeholder*='Phone']", phone_number)

# Password field
await page.fill("input[type='password']", password)

# Login button
await page.click("button.button.primary:has-text('LOGIN')")

# Play button
await page.click("div.button.primary:has-text('PLAY')")

# Multiplier scraping
Array.from(document.querySelectorAll('.multiplier-item, .bubble-item, .payouts-block'))
```

Use browser developer tools (F12) in screenshots to identify the correct CSS selectors.

---

**Good luck! 🚀** Let me know if the test succeeds or if you need help debugging.
