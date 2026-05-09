import requests
from bs4 import BeautifulSoup
import re
import time
import json

def login_and_scrape_requests(phone_number, password):
    """Simple requests-based login and scraping for Spin City."""

    print("🌐 Starting requests-based login...")

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1'
    })

    try:
        # Step 1: Get login page to extract any CSRF tokens or form data
        print("1️⃣  Getting login page...")
        login_page = session.get('https://spincity.co.zw/login', timeout=30)
        login_page.raise_for_status()
        print("   ✅ Login page loaded")

        # Parse the login page
        soup = BeautifulSoup(login_page.text, 'html.parser')

        # Look for form data, CSRF tokens, etc.
        csrf_token = None
        for input_tag in soup.find_all('input', {'type': 'hidden'}):
            if 'csrf' in input_tag.get('name', '').lower() or 'token' in input_tag.get('name', '').lower():
                csrf_token = input_tag.get('value')
                break

        if csrf_token:
            print(f"   📝 Found CSRF token: {csrf_token[:10]}...")

        # Step 2: Prepare login data
        login_data = {
            'phone': phone_number,
            'password': password,
        }

        if csrf_token:
            login_data['_token'] = csrf_token

        # Alternative: try different field names
        alt_login_data = {
            'username': phone_number,
            'password': password,
        }

        # Step 3: Attempt login
        print("2️⃣  Attempting login...")
        login_response = session.post('https://spincity.co.zw/login', data=login_data, timeout=30)

        if login_response.status_code == 200:
            print("   ✅ Login request successful")
        else:
            print(f"   ⚠️  Login returned status {login_response.status_code}")

        # Check if we're redirected to dashboard or still on login
        if 'login' not in login_response.url.lower():
            print("   ✅ Redirected away from login page - likely successful")
        else:
            print("   ⚠️  Still on login page - may have failed")

        # Save login response for debugging
        with open('login_response.html', 'w', encoding='utf-8') as f:
            f.write(login_response.text)
        print("   📄 Saved login response to 'login_response.html'")

        # Step 4: Try to access Aviator game
        print("3️⃣  Accessing Aviator game...")
        game_response = session.get('https://spincity.co.zw/games/Aviator-6094', timeout=30)

        if game_response.status_code == 200:
            print("   ✅ Game page accessed")
        else:
            print(f"   ⚠️  Game page returned status {game_response.status_code}")

        # Save game page
        with open('aviator_page_source.html', 'w', encoding='utf-8') as f:
            f.write(game_response.text)
        print("   📄 Saved game page to 'aviator_page_source.html'")

        # Step 5: Extract multipliers from both pages
        print("4️⃣  Extracting multipliers...")

        all_text = login_response.text + game_response.text
        multipliers = []

        # Try different regex patterns
        patterns = [
            r'([0-9]+\.[0-9]{2})x',  # 1.23x
            r'multiplier["\s:]+([0-9]+\.[0-9]{2})',  # multiplier: 1.23
            r'payout["\s:]+([0-9]+\.[0-9]{2})',  # payout: 1.23
            r'([0-9]+\.[0-9]{2})\s*x',  # 1.23 x
        ]

        for pattern in patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                multipliers.extend([float(m) for m in matches])

        # Also try to find JSON data
        try:
            json_matches = re.findall(r'\{[^}]*"multiplier"[^}]*\}', all_text)
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    if 'multiplier' in data:
                        multipliers.append(float(data['multiplier']))
                except:
                    pass
        except:
            pass

        # Remove duplicates and filter reasonable values
        unique_multipliers = list(set(multipliers))
        # Filter to reasonable Aviator multipliers (1.0 to 100.0)
        filtered_multipliers = [m for m in unique_multipliers if 1.0 <= m <= 100.0]

        if filtered_multipliers:
            sorted_multipliers = sorted(filtered_multipliers, reverse=True)
            print(f"   ✅ Found {len(sorted_multipliers)} unique multipliers")
            print(f"   📊 Range: {min(sorted_multipliers):.2f}x - {max(sorted_multipliers):.2f}x")
            return sorted_multipliers, "✅ Scraping successful!"
        else:
            print("   ⚠️  No multipliers found in page content")
            return [], "⚠️  No multiplier values found. Check saved HTML files."

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Network error: {e}")
        return [], f"❌ Network error: {str(e)}"
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return [], f"❌ Unexpected error: {str(e)}"

# Test with provided credentials
PHONE = "785409934"
PWD = "0426"

print("="*60)
print("🧪 Testing Spin City Login & Scraping (Requests)")
print("="*60)

multipliers, message = login_and_scrape_requests(PHONE, PWD)
print(f"\nResult: {message}")

if multipliers:
    print(f"\n✅ SUCCESS! Found {len(multipliers)} multipliers:")
    for i, mult in enumerate(multipliers[:20], 1):
        print("2.2f")
    if len(multipliers) > 20:
        print(f"... and {len(multipliers) - 20} more")
else:
    print("\n❌ FAILED: No multipliers captured")
    print("Check the saved HTML files for debugging.")