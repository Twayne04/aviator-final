!pip install playwright
!playwright install
!playwright install-deps

import nest_asyncio
import asyncio
from playwright.async_api import async_playwright

nest_asyncio.apply()

async def login_and_capture(phone_number, password):
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 390, 'height': 844},
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
        )
        page = await context.new_page()

        try:
            print("🚀 Navigating to Spin City...")
            await page.goto('https://spincity.co.zw/login', wait_until="domcontentloaded")
            await asyncio.sleep(5)

            # Dismiss the cookie banner if it exists
            try:
                print("🍪 Checking for and dismissing cookie banner...")
                await page.click("button:has-text('GOT IT!')", timeout=2000)
                print("🍪 Cookie banner dismissed.")
                await asyncio.sleep(2) # Give some time for the banner to disappear
            except Exception:
                print("🍪 No cookie banner found or already dismissed.")
                pass # Banner didn't appear or already dismissed

            # Step 1: Fill Phone Number
            # We target the input that has 'Phone' in the placeholder
            print(f"📱 Filling phone: {phone_number}")
            await page.fill("input[placeholder*='Phone']", phone_number)

            # Step 2: Fill Password
            print("🔑 Filling password...")
            await page.fill("input[type='password']", password)

            # Step 3: Click the Yellow Login Button
            print("🔘 Clicking LOGIN...")
            await page.click("button.button.primary:has-text('LOGIN'), .btn-login, button:has-text('LOGIN')")

            # Wait for login to process
            await asyncio.sleep(8)
            print("✅ Login pulse check...")

            # Step 4: Go to Aviator
            await page.goto('https://spincity.co.zw/games/Aviator-6094', wait_until="domcontentloaded")
            await asyncio.sleep(5)

            # Save the page content to a file for inspection
            html_content = await page.content()
            with open('aviator_page_source.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            print("📄 Saved Aviator page source to 'aviator_page_source.html' for inspection.")

            # Step 5: Click the Yellow "PLAY" button to start the engine
            print("🔘 Triggering the Plane...")
            await page.click("div.button.primary:has-text('PLAY'), button:has-text('PLAY')")
            await asyncio.sleep(20) # Heavy wait for Spribe

            # Step 6: Verify and Scrape
            await page.screenshot(path="logged_in_plane.png")

            multipliers = []
            for frame in page.frames:
                data = await frame.evaluate("""() => {
                    const items = Array.from(document.querySelectorAll('.multiplier-item, .bubble-item, .payouts-block'));
                    return items.map(i => i.innerText.trim()).filter(v => v.includes('x'));
                }""")
                if data:
                    multipliers.extend(data)

            if multipliers:
                print(f"💰 SUCCESS! Data Captured: {multipliers}")
            else:
                print("⚠️ Logged in, but multipliers hidden. Check 'logged_in_plane.png' and 'aviator_page_source.html'.")

        except Exception as e:
            print(f"❌ Error: {e}")
            await page.screenshot(path="login_failure.png")

        await browser.close()

# USE YOUR REAL PHONE NUMBER (WITHOUT THE +263) AND PASSWORD
# Since the +263 is already in the box, just put the rest of the number
PHONE = "785409934"
PWD = "0426"

asyncio.run(login_and_capture(PHONE, PWD))