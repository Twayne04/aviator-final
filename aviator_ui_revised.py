import os
import re
import glob
import pickle
import asyncio
import warnings
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

warnings.filterwarnings('ignore')

MODEL_DIR = Path(__file__).parent
DATA_PATTERN = str(MODEL_DIR / "*.csv")
LIVE_FILE = MODEL_DIR / "live_multipliers.csv"
DEFAULT_WINDOW = 10

st.set_page_config(
    page_title="Aviator Predictor Lite",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# Data helpers
# --------------------------------------------------

def load_all_data():
    files = sorted(glob.glob(DATA_PATTERN))
    df_list = []

    for file in files:
        try:
            df = pd.read_csv(file)
            if 'multiplier' not in df.columns:
                df.columns = ['multiplier'] + list(df.columns[1:])
            df = df[['multiplier']].copy()
            df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
            df_list.append(df.dropna(subset=['multiplier']))
        except Exception:
            continue

    if not df_list:
        return pd.DataFrame(columns=['multiplier']), np.array([])

    df = pd.concat(df_list, ignore_index=True).drop_duplicates().reset_index(drop=True)
    df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
    df = df.dropna(subset=['multiplier']).reset_index(drop=True)
    return df, df['multiplier'].values


def append_live_data(values):
    if not values:
        return
    LIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    header = not LIVE_FILE.exists()
    with open(LIVE_FILE, 'a', encoding='utf-8') as f:
        if header:
            f.write('multiplier\n')
        for value in values:
            f.write(f"{value}\n")


def create_features(multipliers, window=DEFAULT_WINDOW):
    features, targets = [], []
    last_high = -1

    for i in range(len(multipliers)):
        if i >= window:
            window_values = np.array(multipliers[i - window:i])
            series = pd.Series(window_values)
            if last_high < 0:
                rounds_since_high = window
            else:
                rounds_since_high = i - last_high

            diffs = series.diff().dropna()
            gains = diffs[diffs > 0].sum()
            losses = -diffs[diffs < 0].sum()
            rsi = 100 - (100 / (1 + gains / (losses if losses != 0 else 1e-9)))

            features.append([
                window_values.mean(),
                window_values.std(),
                window_values.min(),
                window_values.max(),
                window_values.median(),
                multipliers[i - 1],
                multipliers[i - 2],
                multipliers[i - 3],
                window_values[-1] - window_values[-2],
                np.sum(window_values >= 10) / window,
                np.sum(window_values >= 2) / window,
                np.sum(window_values < 1.5) / window,
                rounds_since_high,
                series.ewm(span=5).mean().iloc[-1],
                series.skew(),
                series.kurt(),
                series.var(),
                rsi,
                series.tail(3).mean(),
                np.sum(window_values < 1.2) / window,
            ])
            targets.append(multipliers[i])

        if multipliers[i] >= 10:
            last_high = i

    return np.array(features), np.array(targets)


# --------------------------------------------------
# Model helpers
# --------------------------------------------------

def train_models(df):
    multipliers = df['multiplier'].values
    if len(multipliers) < DEFAULT_WINDOW + 10:
        return None, None, None, {"error": "Not enough historical data."}

    X, y = create_features(multipliers)
    if len(X) < 50:
        return None, None, None, {"error": "Not enough feature rows to train models."}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    rf_mae = mean_absolute_error(y_test, rf_preds)

    nn_model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        early_stopping=True,
        max_iter=400,
        learning_rate_init=1e-3,
        random_state=42,
    )
    nn_model.fit(X_train_scaled, y_train)
    nn_preds = nn_model.predict(X_test_scaled)
    nn_mae = mean_absolute_error(y_test, nn_preds)

    with open(MODEL_DIR / 'rf_regressor.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open(MODEL_DIR / 'nn_regressor.pkl', 'wb') as f:
        pickle.dump(nn_model, f)
    with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    metrics = {
        'rf_mae': float(rf_mae),
        'nn_mae': float(nn_mae),
        'records': int(len(multipliers)),
        'trained_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
    }
    with open(MODEL_DIR / 'model_metadata.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    return rf_model, nn_model, scaler, metrics


def load_models():
    try:
        with open(MODEL_DIR / 'rf_regressor.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open(MODEL_DIR / 'nn_regressor.pkl', 'rb') as f:
            nn_model = pickle.load(f)
        with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(MODEL_DIR / 'model_metadata.pkl', 'rb') as f:
            metrics = pickle.load(f)
        return rf_model, nn_model, scaler, metrics
    except Exception:
        return None, None, None, None


def predict_models(recent_multipliers, rf_model, nn_model, scaler):
    values = [float(v.strip()) for v in recent_multipliers if str(v).strip()]
    if len(values) < DEFAULT_WINDOW:
        return None

    last_window = np.array(values[-DEFAULT_WINDOW:]).reshape(1, -1)
    features, _ = create_features(np.array(values))
    if len(features) == 0:
        return None

    X = scaler.transform(features)
    latest = X[-1].reshape(1, -1)

    rf_pred = rf_model.predict(latest)[0]
    nn_pred = nn_model.predict(latest)[0]
    return {
        'rf_prediction': max(1.0, float(rf_pred)),
        'nn_prediction': max(1.0, float(nn_pred)),
        'average_prediction': max(1.0, float((rf_pred + nn_pred) / 2.0)),
    }


# --------------------------------------------------
# Login and scrape helpers
# --------------------------------------------------

async def perform_login_scrape_async(
    phone_number,
    password,
    game_url='https://spincity.co.zw/games/Aviator-6094',
    login_url='https://spincity.co.zw/login',
):
    """Playwright-based login and multiplier scraper for Spin City.
    
    Args:
        phone_number: Phone number without +263 prefix (e.g., '771234567')
        password: Account password
        game_url: URL to the Aviator game
        login_url: Login page URL
    
    Returns:
        (multipliers_list, status_message)
    """
    if not HAS_PLAYWRIGHT:
        return [], "Playwright not installed. Run: pip install playwright"
    
    if not phone_number or not password:
        return [], "Phone number and password required."
    
    multipliers = []
    status_msg = ""
    
    try:
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 390, 'height': 844},
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
            )
            page = await context.new_page()
            
            # Navigate to login
            await page.goto(login_url, wait_until="domcontentloaded")
            await asyncio.sleep(3)
            
            # Fill phone number
            await page.fill("input[placeholder*='Phone']", phone_number)
            await asyncio.sleep(1)
            
            # Fill password
            await page.fill("input[type='password']", password)
            await asyncio.sleep(1)
            
            # Click login button
            try:
                await page.click("button.button.primary:has-text('LOGIN'), .btn-login, button:has-text('LOGIN')")
            except:
                await page.click("button:has-text('LOGIN')")
            
            await asyncio.sleep(8)
            status_msg = "✅ Login successful. "
            
            # Navigate to Aviator game
            await page.goto(game_url, wait_until="domcontentloaded")
            await asyncio.sleep(5)
            
            # Click Play button
            try:
                await page.click("div.button.primary:has-text('PLAY'), button:has-text('PLAY')")
            except:
                pass
            
            await asyncio.sleep(20)
            
            # Try to scrape multipliers from various possible selectors
            try:
                for frame in page.frames:
                    data = await frame.evaluate("""() => {
                        const items = Array.from(document.querySelectorAll('.multiplier-item, .bubble-item, .payouts-block, [class*="multiplier"], [class*="payout"]'));
                        return items.map(i => i.innerText.trim()).filter(v => v && v.includes('x'));
                    }""")
                    if data:
                        multipliers.extend(data)
            except:
                pass
            
            # Fallback: regex search on page text
            if not multipliers:
                page_text = await page.content()
                matches = re.findall(r'([0-9]+\.[0-9]{2})x', page_text)
                multipliers = [float(m) for m in matches]
            else:
                # Parse multiplier strings (e.g., "1.23x" -> 1.23)
                parsed = []
                for m in multipliers:
                    try:
                        val = float(re.search(r'([0-9]+\.[0-9]{2})', m).group(1))
                        parsed.append(val)
                    except:
                        pass
                multipliers = parsed
            
            # Save screenshot for debugging
            try:
                await page.screenshot(path=MODEL_DIR / "logged_in_plane.png")
            except:
                pass
            
            await browser.close()
            
            if multipliers:
                unique_values = sorted(set(multipliers), reverse=True)
                status_msg += f"Captured {len(unique_values)} unique multiplier values."
                return unique_values, status_msg
            else:
                return [], status_msg + "No multiplier values found. Check screenshot."
    
    except Exception as e:
        return [], f"❌ Scraping error: {str(e)}"


def perform_login_scrape(phone_number, password):
    """Synchronous wrapper for async login+scrape.
    
    Args:
        phone_number: Phone without +263 prefix
        password: Account password
    
    Returns:
        (multipliers_list, status_message)
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(
            perform_login_scrape_async(phone_number, password)
        )
    except Exception as e:
        return [], f"Error running scraper: {str(e)}"


# --------------------------------------------------
# UI helpers
# --------------------------------------------------

def init_session():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'login_message' not in st.session_state:
        st.session_state.login_message = ''
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = []
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None


def render_login_card():
    st.markdown('## 🔐 Login & Scrap Data (Spin City)')
    st.info('Provide your Spin City credentials to login and extract recent Aviator multipliers.')

    col1, col2 = st.columns(2)
    with col1:
        phone_number = st.text_input(
            'Phone Number (without +263)',
            placeholder='e.g., 771234567',
            help='Enter your phone digits without the +263 prefix'
        )
    
    with col2:
        password = st.text_input('Password', type='password', placeholder='Your password')
    
    login_button = st.button('🚀 Login + Scrape Multipliers')

    if login_button:
        if not phone_number or not password:
            st.error('Please enter both phone number and password.')
        else:
            with st.spinner('🔄 Logging in and scraping data... This may take 30-40 seconds.'):
                scraped_values, message = perform_login_scrape(phone_number, password)
                st.session_state.scraped_data = scraped_values
                st.session_state.login_message = message
                if scraped_values:
                    st.session_state.authenticated = True

    if st.session_state.login_message:
        if '❌' in st.session_state.login_message or 'error' in st.session_state.login_message.lower():
            st.error(st.session_state.login_message)
        elif '✅' in st.session_state.login_message:
            st.success(st.session_state.login_message)
        else:
            st.warning(st.session_state.login_message)

    if st.session_state.authenticated and st.session_state.scraped_data:
        st.success('✅ Successfully scraped data! Ready to predict.')
        st.metric('Multiplier values captured', len(st.session_state.scraped_data))
        if st.checkbox('Show captured multipliers', value=False):
            st.write(sorted(st.session_state.scraped_data, reverse=True)[:20])

    return st.session_state.authenticated


def render_prediction_section(multipliers, rf_model, nn_model, scaler, model_metrics):
    st.markdown('## 📊 Prediction')

    if rf_model is None or nn_model is None or scaler is None:
        st.warning('No trained models are loaded. Use the retrain button in the sidebar.')
        return

    default_input = ', '.join([f'{x:.2f}' for x in multipliers[-DEFAULT_WINDOW:]]) if len(multipliers) >= DEFAULT_WINDOW else ''
    raw_input = st.text_area('Recent multipliers (comma separated)', value=default_input, height=120)
    threshold = st.slider('High recommendation threshold', 0.1, 0.9, 0.5, 0.05)

    if st.button('Predict now'):
        recent_values = raw_input.split(',')
        prediction = predict_models(recent_values, rf_model, nn_model, scaler)
        if prediction is None:
            st.error(f'Please enter at least {DEFAULT_WINDOW} valid multipliers.')
            return

        st.session_state.prediction_result = prediction

    if st.session_state.prediction_result:
        prediction = st.session_state.prediction_result
        rf_card, nn_card, avg_card = st.columns(3)
        with rf_card:
            st.metric('RF Prediction', f"{prediction['rf_prediction']:.2f}x")
            st.caption('Random Forest regressor output')
        with nn_card:
            st.metric('NN Prediction', f"{prediction['nn_prediction']:.2f}x")
            st.caption('Neural network regressor output')
        with avg_card:
            st.metric('Combined Avg', f"{prediction['average_prediction']:.2f}x")
            st.caption('Average of both models')

        recommendation = 'HIGH' if prediction['average_prediction'] >= threshold else 'LOW'
        rec_color = '✅' if recommendation == 'HIGH' else '⚠️'
        st.markdown(f'### {rec_color} Recommendation: **{recommendation}**')

    if model_metrics:
        with st.expander('Model metrics', expanded=False):
            st.write('**Random Forest MAE:**', f"{model_metrics['rf_mae']:.3f}x")
            st.write('**Neural Network MAE:**', f"{model_metrics['nn_mae']:.3f}x")
            st.write('**Records used:**', model_metrics['records'])
            st.write('**Last trained:**', model_metrics['trained_at'])


def render_data_summary(df, multipliers):
    st.markdown('## 📁 Dataset summary')
    if len(multipliers) == 0:
        st.warning('No multiplier data found. Upload CSV or scrape data to proceed.')
        return

    high_count = int(np.sum(multipliers >= 10))
    st.metric('Total records', len(multipliers))
    st.metric('High (10x+)', f'{high_count} ({high_count / len(multipliers) * 100:.1f}%)')

    with st.expander('Recent multipliers', expanded=False):
        st.write(pd.DataFrame({'multiplier': multipliers[-20:][::-1]}))

    with st.expander('Histogram', expanded=False):
        hist = pd.cut(multipliers, bins=[0, 1.5, 2, 5, 10, 20, 50, 100]).value_counts().sort_index()
        st.bar_chart(hist)


# --------------------------------------------------
# App layout
# --------------------------------------------------

def main():
    init_session()
    st.title('✈️ Aviator Predictor Lite')
    st.markdown('A compact, scroll-friendly prediction app with two regression engines and real-time scraping from Spin City.')

    logged_in = render_login_card()
    
    # Append scraped data to live file if we just scraped
    if logged_in and st.session_state.scraped_data:
        append_live_data(st.session_state.scraped_data)
        st.session_state.scraped_data = []  # Clear to avoid re-appending
    
    st.markdown('---')

    df, multipliers = load_all_data()
    
    if st.sidebar.button('🔄 Retrain models'):
        with st.spinner('Training models, this may take up to a minute...'):
            rf_model, nn_model, scaler, metrics = train_models(df)
            if metrics and 'error' not in metrics:
                st.success('✅ Models retrained successfully.')
                st.session_state.model_metrics = metrics
            else:
                st.error(metrics.get('error', 'Training failed.'))
    
    if st.sidebar.button('🔁 Refresh data'):
        df, multipliers = load_all_data()
        st.rerun()

    with st.sidebar.expander('⚙️ Advanced options', expanded=False):
        if st.button('Show raw dataset'):
            st.dataframe(df, use_container_width=True)

    rf_model, nn_model, scaler, model_metrics = load_models()

    render_data_summary(df, multipliers)
    st.markdown('---')
    render_prediction_section(multipliers, rf_model, nn_model, scaler, model_metrics)

    st.markdown('---')
    st.caption(f'📊 Data points: {len(multipliers)} | Last updated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
