"""
Test script for Playwright login and scraping function.
Run this directly to test the Spin City login and multiplier scraping.

Usage:
    python test_login_scrape.py
"""

import asyncio
import sys
from pathlib import Path

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    print("⚠️  nest_asyncio not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nest_asyncio"])
    import nest_asyncio
    nest_asyncio.apply()

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("❌ Playwright not installed!")
    print("Install it with:")
    print("  pip install playwright")
    print("  playwright install firefox")
    sys.exit(1)

import re


async def test_login_and_scrape(phone_number, password):
    """Test the Playwright login and scraping function."""
    
    print(f"\n{'='*60}")
    print("🧪 Testing Spin City Login & Scraping")
    print(f"{'='*60}\n")
    
    print(f"📱 Phone: {phone_number}")
    print(f"🔑 Password: {'*' * len(password)}")
    print()
    
    try:
        async with async_playwright() as p:
            print("🌐 Launching Firefox browser...")
            browser = await p.firefox.launch(headless=True)
            
            print("📄 Creating new page context...")
            context = await browser.new_context(
                viewport={'width': 390, 'height': 844},
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
            )
            page = await context.new_page()
            
            # Step 1: Navigate to login
            print("\n1️⃣  Navigating to login page...")
            await page.goto('https://spincity.co.zw/login', wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)
            print("   ✅ Login page loaded")
            
            # Step 2: Fill phone number
            print("\n2️⃣  Filling phone number...")
            try:
                await page.fill("input[placeholder*='Phone']", phone_number)
                print(f"   ✅ Phone filled: {phone_number}")
            except Exception as e:
                print(f"   ❌ Failed to fill phone: {e}")
                await page.screenshot(path="test_phone_error.png")
                return [], f"Phone field error: {e}"
            
            await asyncio.sleep(1)
            
            # Step 3: Fill password
            print("\n3️⃣  Filling password...")
            try:
                await page.fill("input[type='password']", password)
                print("   ✅ Password filled")
            except Exception as e:
                print(f"   ❌ Failed to fill password: {e}")
                await page.screenshot(path="test_password_error.png")
                return [], f"Password field error: {e}"
            
            await asyncio.sleep(1)
            
            # Step 4: Click login button
            print("\n4️⃣  Clicking LOGIN button...")
            try:
                await page.click("button.button.primary:has-text('LOGIN'), .btn-login, button:has-text('LOGIN')")
                print("   ✅ LOGIN button clicked")
            except Exception as e:
                print(f"   ⚠️  Login button selector failed, trying alternate...")
                try:
                    await page.click("button:has-text('LOGIN')")
                    print("   ✅ LOGIN button clicked (alternate selector)")
                except Exception as e2:
                    print(f"   ❌ All login button selectors failed: {e2}")
                    await page.screenshot(path="test_login_button_error.png")
                    return [], f"Login button error: {e2}"
            
            print("\n5️⃣  Waiting for login to process...")
            await asyncio.sleep(8)
            print("   ✅ Login processed (waiting complete)")
            
            # Save screenshot after login attempt
            await page.screenshot(path="test_after_login.png")
            print("   📸 Screenshot saved: test_after_login.png")
            
            # Step 5: Navigate to Aviator game
            print("\n6️⃣  Navigating to Aviator game...")
            try:
                await page.goto('https://spincity.co.zw/games/Aviator-6094', wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(3)
                print("   ✅ Aviator game page loaded")
            except Exception as e:
                print(f"   ❌ Failed to navigate to game: {e}")
                await page.screenshot(path="test_game_nav_error.png")
                return [], f"Game navigation error: {e}"
            
            # Step 6: Click PLAY button
            print("\n7️⃣  Clicking PLAY button...")
            try:
                await page.click("div.button.primary:has-text('PLAY'), button:has-text('PLAY')")
                print("   ✅ PLAY button clicked")
            except Exception as e:
                print(f"   ⚠️  PLAY button not found (may not be needed): {e}")
            
            print("\n8️⃣  Waiting for game to load...")
            await asyncio.sleep(20)
            print("   ✅ Game load complete")
            
            # Save screenshot of game
            await page.screenshot(path="test_aviator_game.png")
            print("   📸 Screenshot saved: test_aviator_game.png")
            
            # Step 7: Scrape multipliers
            print("\n9️⃣  Scraping multiplier values...")
            multipliers = []
            
            # Try multiple methods
            for frame in page.frames:
                try:
                    data = await frame.evaluate("""() => {
                        const items = Array.from(document.querySelectorAll('.multiplier-item, .bubble-item, .payouts-block, [class*="multiplier"], [class*="payout"]'));
                        return items.map(i => i.innerText.trim()).filter(v => v && v.includes('x'));
                    }""")
                    if data:
                        print(f"   ✅ Found {len(data)} items in frame")
                        multipliers.extend(data)
                except:
                    pass
            
            # Fallback: regex on page content
            if not multipliers:
                print("   ⚠️  DOM scraping found nothing, trying regex fallback...")
                try:
                    page_text = await page.content()
                    matches = re.findall(r'([0-9]+\.[0-9]{2})x', page_text)
                    print(f"   ✅ Regex found {len(matches)} matches")
                    multipliers = [float(m) for m in matches]
                except Exception as e:
                    print(f"   ❌ Regex failed: {e}")
            else:
                # Parse multiplier strings
                try:
                    parsed = []
                    for m in multipliers:
                        try:
                            val = float(re.search(r'([0-9]+\.[0-9]{2})', m).group(1))
                            parsed.append(val)
                        except:
                            pass
                    multipliers = parsed
                    print(f"   ✅ Parsed {len(multipliers)} multiplier values")
                except Exception as e:
                    print(f"   ⚠️  Parse error: {e}")
            
            await browser.close()
            
            print(f"\n{'='*60}")
            if multipliers:
                unique = sorted(set(multipliers), reverse=True)
                print(f"✅ SUCCESS! Scraped {len(unique)} unique multiplier values")
                print(f"{'='*60}\n")
                print("Top 10 values:", unique[:10])
                print("\nAll unique values:", unique)
                return unique, "✅ Scraping successful!"
            else:
                print("⚠️  WARNING: No multiplier values found!")
                print(f"{'='*60}\n")
                print("This could mean:")
                print("  1. Login failed - check test_after_login.png")
                print("  2. Game didn't load - check test_aviator_game.png")
                print("  3. Multipliers are loaded via JavaScript after page load")
                print("  4. Selectors have changed on the website")
                print("\n👉 Check the screenshot files above for debugging.")
                return [], "⚠️  No multiplier values found. Check screenshots for debugging."
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return [], f"❌ Test failed: {str(e)}"


def main():
    print("\n" + "="*60)
    print("Spin City Playwright Login & Scrape Tester")
    print("="*60 + "\n")
    
    # Get credentials from user
    phone = input("📱 Enter phone number (without +263): ").strip()
    if not phone:
        print("❌ Phone number cannot be empty!")
        return
    
    password = input("🔑 Enter password: ").strip()
    if not password:
        print("❌ Password cannot be empty!")
        return
    
    print("\n⏳ Starting test...\n")
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        multipliers, message = loop.run_until_complete(
            test_login_and_scrape(phone, password)
        )
        
        print(message)
        
        if multipliers:
            print("\n✅ Test passed! You can now use the app.")
            print("Run: streamlit run aviator_ui_revised.py")
        else:
            print("\n❌ Test failed. Check the screenshots for debugging.")
            print("Screenshots saved:")
            print("  - test_after_login.png")
            print("  - test_aviator_game.png")
            print("  - test_*_error.png (if errors occurred)")
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
