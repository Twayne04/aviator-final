import time
import re
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.microsoft import EdgeChromiumDriverManager

def login_and_capture_selenium(phone_number, password):
    """Selenium-based login and multiplier scraper for Spin City."""

    print("🚀 Starting Selenium browser automation with Microsoft Edge...")

    # Set up Edge options
    edge_options = Options()
    edge_options.use_chromium = True
    # Use visible Edge for debugging and to avoid headless compatibility issues
    # edge_options.add_argument("--headless=new")
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--disable-dev-shm-usage")
    edge_options.add_argument("--window-size=1920,1080")

    try:
        # Initialize the driver
        try:
            service = Service(EdgeChromiumDriverManager().install())
        except Exception as e:
            print(f"WebDriver download failed: {e}")
            print("Trying local msedgedriver.exe...")
            service = Service("msedgedriver.exe")
        driver = webdriver.Edge(service=service, options=edge_options)
        wait = WebDriverWait(driver, 20)

        print("🌐 Navigating to Spin City login...")
        driver.get('https://spincity.co.zw/login')
        time.sleep(5)
        driver.save_screenshot("debug_01_login_page_loaded.png")
        print("📸 Screenshot: debug_01_login_page_loaded.png")

        # Dismiss cookie banner if it exists
        try:
            print("🍪 Checking for cookie banner...")
            cookie_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'GOT IT!')]")))
            cookie_button.click()
            print("🍪 Cookie banner dismissed.")
            time.sleep(2)
        except TimeoutException:
            print("🍪 No cookie banner found or already dismissed.")

        driver.save_screenshot("debug_02_after_cookie.png")
        print("📸 Screenshot: debug_02_after_cookie.png")

        # Step 1: Fill Phone Number
        print(f"📱 Filling phone: {phone_number}")
        try:
            phone_input = wait.until(EC.presence_of_element_located((By.XPATH, "//input[contains(@name, 'phone') or contains(@id, 'phone') or contains(@placeholder, 'Phone') or contains(@placeholder, 'phone') or contains(@aria-label, 'Phone') or contains(@type, 'tel')]")))
            phone_input.clear()
            phone_input.send_keys(phone_number)
            print("   ✅ Phone filled")
        except TimeoutException:
            print("   ❌ Phone input not found")
            driver.save_screenshot("phone_error.png")
            return [], "Phone input not found"

        driver.save_screenshot("debug_03_phone_filled.png")
        print("📸 Screenshot: debug_03_phone_filled.png")

        # Step 2: Fill Password
        print("🔑 Filling password...")
        try:
            password_input = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='password' or contains(@name, 'password') or contains(@id, 'password') or contains(@placeholder, 'Password')]")))
            password_input.clear()
            password_input.send_keys(password)
            print("   ✅ Password filled")
        except TimeoutException:
            print("   ❌ Password input not found")
            driver.save_screenshot("password_error.png")
            return [], "Password input not found"
        driver.save_screenshot("debug_04_password_filled.png")
        print("📸 Screenshot: debug_04_password_filled.png")
        # Step 3: Click Login Button
        print("🔘 Clicking LOGIN...")
        try:
            login_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(translate(normalize-space(.), 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'LOGIN')] | //button[contains(translate(normalize-space(.), 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'SIGN IN')] | //button[contains(@class, 'login') or contains(@class, 'btn') or contains(@class, 'submit')]") ))
            login_button.click()
            print("   ✅ LOGIN clicked")
        except TimeoutException:
            print("   ❌ LOGIN button not found")
            driver.save_screenshot("login_button_error.png")
            return [], "LOGIN button not found"

        driver.save_screenshot("debug_05_login_clicked.png")
        print("📸 Screenshot: debug_05_login_clicked.png")

        # Wait for login to process
        print("⏳ Waiting for login to process...")
        time.sleep(8)
        print("   ✅ Login processing complete")

        driver.save_screenshot("debug_06_after_login_wait.png")
        print("📸 Screenshot: debug_06_after_login_wait.png")

        # Step 4: Navigate to Aviator
        print("🎮 Navigating to Aviator game...")
        driver.get('https://spincity.co.zw/games/Aviator-6094')
        time.sleep(5)

        driver.save_screenshot("debug_07_aviator_loaded.png")
        print("📸 Screenshot: debug_07_aviator_loaded.png")

        # Save page source for inspection
        with open('aviator_page_source.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        print("📄 Saved Aviator page source to 'aviator_page_source.html'")

        # Step 5: Click PLAY button
        print("🔘 Clicking PLAY button...")
        try:
            play_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(translate(normalize-space(.), 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'PLAY')] | //div[contains(@class, 'button') and contains(translate(normalize-space(.), 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'PLAY')]") ))
            play_button.click()
            print("   ✅ PLAY clicked")
        except TimeoutException:
            print("   ⚠️ PLAY button not found (may not be needed)")

        # Wait for game to load
        print("⏳ Waiting for game to load...")
        time.sleep(20)
        print("   ✅ Game load complete")

        driver.save_screenshot("debug_08_game_loaded.png")
        print("📸 Screenshot: debug_08_game_loaded.png")

        # Take screenshot
        driver.save_screenshot("logged_in_plane.png")
        print("📸 Screenshot saved: logged_in_plane.png")

        # Step 6: Scrape multipliers
        print("🔍 Scraping multiplier values...")
        multipliers = []

        # Try different selectors
        selectors = [
            '.multiplier-item',
            '.bubble-item',
            '.payouts-block',
            '[class*="multiplier"]',
            '[class*="payout"]'
        ]

        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.strip()
                    if text and 'x' in text:
                        multipliers.append(text)
            except Exception:
                continue

        # Fallback: regex on page source
        if not multipliers:
            print("   ⚠️ DOM scraping found nothing, trying regex...")
            page_text = driver.page_source
            matches = re.findall(r'([0-9]+\.[0-9]{2})x', page_text)
            multipliers = [f"{match}x" for match in matches]

        # Parse and clean multipliers
        parsed_multipliers = []
        for m in multipliers:
            try:
                # Extract number from "1.23x" format
                match = re.search(r'([0-9]+\.[0-9]{2})', m)
                if match:
                    parsed_multipliers.append(float(match.group(1)))
            except:
                continue

        driver.quit()

        if parsed_multipliers:
            unique_values = sorted(set(parsed_multipliers), reverse=True)
            print(f"💰 SUCCESS! Captured {len(unique_values)} unique multiplier values")
            print(f"Top values: {unique_values[:10]}")
            return unique_values, "✅ Scraping successful!"
        else:
            print("⚠️ No multiplier values found")
            return [], "⚠️ No multiplier values found. Check screenshots for debugging."

    except Exception as e:
        print(f"❌ Error: {e}")
        try:
            driver.save_screenshot("selenium_error.png")
        except:
            pass
        try:
            driver.quit()
        except:
            pass
        return [], f"❌ Selenium error: {str(e)}"

# Test with provided credentials
PHONE = "785409934"
PWD = "0426"

print("="*60)
print("🧪 Testing Spin City Login & Scraping (Selenium)")
print("="*60)

multipliers, message = login_and_capture_selenium(PHONE, PWD)
print(f"\nResult: {message}")

if multipliers:
    print(f"\n✅ SUCCESS! Found {len(multipliers)} multipliers:")
    for i, mult in enumerate(multipliers[:20], 1):
        print("2.2f")
    if len(multipliers) > 20:
        print(f"... and {len(multipliers) - 20} more")
else:
    print("\n❌ FAILED: No multipliers captured")
    print("Check the screenshot files for debugging.")