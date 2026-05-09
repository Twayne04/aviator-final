"""
Aviator Multiplier Prediction UI
Streamlit Web Interface for Aviator Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from pathlib import Path
import os
import io
import glob
import warnings

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Aviator Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# AUTHENTICATION
# ============================================
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "aviator123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔐 Aviator Predictor Login")
        st.text_input("Enter Access Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔐 Aviator Predictor Login")
        st.text_input("Enter Access Password", type="password", on_change=password_entered, key="password")
        st.error("❌ Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    """Load the trained models"""
    base_path = Path(__file__).parent    
    lstm_model = None

    # Try loading TensorFlow/LSTM if available
    try:
        from tensorflow.keras.models import load_model
        # Try multiple common names for the LSTM model
        for model_name in ['aviator_lstm_classifier.h5', 'aviator_model.h5', 'best_model.h5']:
            lstm_path = base_path / model_name
            if lstm_path.exists():
                lstm_model = load_model(lstm_path)
                break
    except Exception:
        pass # Fallback to SKLearn only

    try:
        # Load classifier
        clf_path = base_path / 'aviator_classifier.pkl'
        classifier = pickle.load(open(clf_path, 'rb'))
        
        # Load regressor
        reg_path = base_path / 'aviator_regressor.pkl'
        regressor = pickle.load(open(reg_path, 'rb'))
        
        # Load scaler
        scaler_path = base_path / 'aviator_scaler.pkl'
        scaler = pickle.load(open(scaler_path, 'rb'))
        
        return classifier, regressor, scaler, lstm_model
    except Exception as e:
        return None, None, None, None

# ============================================
# RETRAINING UTILITIES
# ============================================
def create_features_train(multipliers, window=10):
    features, targets, targets_class = [], [], []
    last_high_idx = -1
    for i in range(len(multipliers)):
        if i >= window:
            window_series = pd.Series(multipliers[i-window:i])
            rounds_since_high = i - last_high_idx if last_high_idx != -1 else window
            
            # Calculate RSI-like indicator
            diffs = window_series.diff().dropna()
            gains = diffs[diffs > 0].sum()
            losses = -diffs[diffs < 0].sum()
            rsi = 100 - (100 / (1 + gains / (losses if losses != 0 else 1e-9)))

            feature = [
                window_series.mean(), window_series.std(), window_series.min(),
                window_series.max(), window_series.median(), multipliers[i-1],
                multipliers[i-2], multipliers[i-3], multipliers[i-1] - multipliers[i-2],
                np.sum(window_series >= 10) / window, np.sum(window_series >= 2) / window,
                np.sum(window_series < 1.5) / window, rounds_since_high,
                # New technical indicators
                window_series.ewm(span=5).mean().iloc[-1], # EMA 5
                window_series.skew(), # Skewness
                window_series.kurt(), # Kurtosis
                window_series.var(), # Variance
                rsi, # RSI-like
                window_series.tail(3).mean(), # Short-term trend
                np.sum(window_series < 1.2) / window # Extreme low frequency
            ]
            feature = [0 if pd.isna(x) else x for x in feature]

            features.append(feature)
            targets.append(multipliers[i])
            targets_class.append(1 if multipliers[i] >= 10 else 0)
        if multipliers[i] >= 10:
            last_high_idx = i
    return np.array(features), np.array(targets), np.array(targets_class)

def run_full_retraining():
    """Concludes the bot by running a deep hyperparameter search and saving metrics"""
    base_path = Path(__file__).parent
    data_files = glob.glob(str(base_path / "*.csv")) + glob.glob(str(base_path / "*.txt"))
    if not data_files: return False, "No data files found."
    
    df_list = [pd.read_csv(f) for f in data_files]
    for d in df_list:
        if 'multiplier' not in d.columns: d.columns = ['multiplier'] + list(d.columns[1:])
    
    df = pd.concat(df_list, ignore_index=True)
    df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
    df = df.dropna(subset=['multiplier']).drop_duplicates()
    
    # Sort by timestamp if it exists to preserve chronological order
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

    multipliers_data = df['multiplier'].values
    
    X, y_regress, y_class = create_features_train(multipliers_data)
    
    # TimeSeriesSplit is essential for sequence-based data like Aviator rounds
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- 1. Deep Training for Classifier ---
    # Using ExtraTrees for better variance reduction on noisy game data
    clf_base = ExtraTreesClassifier(class_weight='balanced', random_state=42)
    clf_params = {
        'n_estimators': [100, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    
    grid_clf = GridSearchCV(
        estimator=clf_base,
        param_grid=clf_params,
        cv=tscv,
        scoring='f1', # F1 balances precision and recall for rare 10x+ events
        n_jobs=-1
    )
    grid_clf.fit(X_scaled, y_class)
    best_classifier = grid_clf.best_estimator_

    # --- 2. Deep Training for Regressor ---
    reg_base = GradientBoostingRegressor(random_state=42)
    reg_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5],
        'loss': ['huber', 'absolute_error'] # Robust to extreme multiplier outliers
    }
    
    grid_reg = GridSearchCV(
        estimator=reg_base,
        param_grid=reg_params,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_reg.fit(X_scaled, y_regress)
    best_regressor = grid_reg.best_estimator_
    
    # Save metadata for the UI to display
    metrics = {
        "f1_score": grid_clf.best_score_,
        "mae": -grid_reg.best_score_,
        "records_used": len(multipliers_data),
        "last_trained": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    }
    
    # Save best models and the scaler
    with open(base_path / 'aviator_classifier.pkl', 'wb') as f: pickle.dump(best_classifier, f)
    with open(base_path / 'aviator_regressor.pkl', 'wb') as f: pickle.dump(best_regressor, f)
    with open(base_path / 'aviator_scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    with open(base_path / 'bot_metadata.pkl', 'wb') as f: pickle.dump(metrics, f)
    
    msg = f"Deep Retraining Successful! Best Clf F1: {grid_clf.best_score_:.4f}, Best Reg MAE: {-grid_reg.best_score_:.4f}"
    return True, msg

def load_all_data():
    """Helper to load and combine all available multiplier CSV files"""
    base_path = Path(__file__).parent
    data_files = glob.glob(str(base_path / "*.csv")) + glob.glob(str(base_path / "*.txt"))
    if not data_files: 
        return pd.DataFrame(columns=['multiplier']), np.array([])
    
    df_list = []
    for f in data_files:
        try:
            d = pd.read_csv(f).rename(columns=str.lower)
            # Handle common variations of headers
            if 'multiplier' not in d.columns and len(d.columns) > 0:
                d = d.rename(columns={d.columns[0]: 'multiplier'})
            df_list.append(d)
        except:
            continue
            
    if not df_list: return pd.DataFrame(columns=['multiplier']), np.array([])
    # Drop duplicates across different files and ensure numeric
    df = pd.concat(df_list, ignore_index=True).drop_duplicates()
    df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
    df = df.dropna(subset=['multiplier'])
    return df, df['multiplier'].values

@st.cache_data
def convert_df_to_csv(df):
    """Cache the conversion to CSV to prevent redundant processing"""
    return df.to_csv(index=False).encode('utf-8')

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_multiplier(full_history_multipliers, recent_multipliers_input, classifier, gb_regressor, scaler, lstm_model=None, window=10, seq_len=5, threshold=0.3):
    """
    Hybrid prediction using LSTM sequence if available, falling back to SKLearn
    """
    if lstm_model:
        required_len = window + (seq_len - 1)
        if len(recent_multipliers_input) < required_len:
            # Not enough recent values for the LSTM sequence; fallback to SKLearn.
            lstm_model = None
    else:
        required_len = window

    if len(recent_multipliers_input) < required_len:
        return None
    
    sequence_features = []
    for s in range(seq_len if lstm_model else 1):
        # Determine the correct window for this step in the sequence
        end_idx = len(recent_multipliers_input) - (seq_len - 1 if lstm_model else 0) + s
        start_idx = end_idx - window
        sub_window = recent_multipliers_input[start_idx:end_idx]
        window_series = pd.Series(sub_window)
        
        # Calculate relative history for rounds_since_high
        hist_cutoff = len(full_history_multipliers) - (seq_len - 1 if lstm_model else 0) + s
        hist_slice = full_history_multipliers[:hist_cutoff]
        high_indices = np.where(np.array(hist_slice) >= 10)[0]
        rounds_since_high = hist_cutoff - high_indices[-1] if len(high_indices) > 0 else window

        # RSI Calculation
        diffs = window_series.diff().dropna()
        gains = diffs[diffs > 0].sum()
        losses = -diffs[diffs < 0].sum()
        rsi = 100 - (100 / (1 + gains / (losses if losses != 0 else 1e-9)))

        feature = [
            window_series.mean(), window_series.std(), window_series.min(), window_series.max(),
            window_series.median(), sub_window[-1], sub_window[-2], sub_window[-3],
            sub_window[-1] - sub_window[-2],
            np.sum(window_series >= 10) / window, np.sum(window_series >= 2) / window,
            np.sum(window_series < 1.5) / window, rounds_since_high,
            window_series.ewm(span=5).mean().iloc[-1], window_series.skew(),
            window_series.kurt(), window_series.var(), rsi,
            window_series.tail(3).mean(), np.sum(window_series < 1.2) / window
        ]
        sequence_features.append([0 if pd.isna(x) else x for x in feature])

    # Scale and Predict
    features_scaled = scaler.transform(sequence_features)
    
    # Regression uses the latest feature vector
    predicted_value = gb_regressor.predict(features_scaled[-1].reshape(1, -1))[0]

    # Classification uses LSTM if available, else SKLearn
    if lstm_model:
        lstm_input = features_scaled.reshape(1, seq_len, features_scaled.shape[1])
        high_prob = lstm_model.predict(lstm_input, verbose=0)[0][0]
        engine = 'LSTM'
    else:
        high_prob = classifier.predict_proba(features_scaled[-1].reshape(1, -1))[0][1]
        engine = 'SKLearn'

    return {
        'high_multiplier_probability': float(high_prob),
        'predicted_value': max(1.0, float(predicted_value)),
        'recommendation': 'HIGH' if high_prob >= threshold else 'LOW',
        'threshold': threshold,
        'engine': engine
    }

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("✈️ Aviator Predictor")

# --- Theme Toggle ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

st.session_state.dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

if st.session_state.dark_mode:
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        [data-testid="stSidebar"] { background-color: #1a1c24; }
        header[data-testid="stHeader"] { background: rgba(0,0,0,0); }
        </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.header("📊 Model Info")
st.sidebar.info("""
This model predicts high multipliers (10x+) for the Aviator game.

**Features Used:**
- Rolling statistics (mean, std, min, max, median)
- Lag features (last 3 multipliers)
- Trend analysis
- High multiplier ratio
- Above 2x ratio
- Low multiplier ratio
""")

# ============================================
# DATA INSPECTOR (SIDEBAR)
# ============================================
with st.sidebar.expander("📂 Data File Inspector"):
    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "*.csv"))
    if files:
        for f in files:
            fname = os.path.basename(f)
            try:
                fdf = pd.read_csv(f)
                st.write(f"**{fname}**")
                st.text(f"Rows: {len(fdf)} | Last: {fdf.iloc[-1].values[0]}x")
            except:
                st.error(f"Error reading {fname}")
    else:
        st.write("No data files detected.")

# ============================================
# MAIN CONTENT
# ============================================
st.title("✈️ Aviator Multiplier Predictor")
st.markdown("### Predict High Multipliers (10x+)")

# Load all available data
df, multipliers = load_all_data()
st.success(f"✅ Loaded {len(multipliers)} multiplier records from all datasets")

# Load models
classifier, regressor, scaler, lstm_model = load_models()

st.sidebar.markdown("---")
with st.sidebar.expander("🔧 Model Diagnostics"):
    st.write(f"**Data Shape:** {df.shape}")
    st.write(f"**Last Multiplier:** {multipliers[-1] if len(multipliers) > 0 else 'N/A'}")
    if classifier:
        st.write("**Model Status:** ✅ Loaded")
    
    # Show Training Health
    try:
        with open(Path(__file__).parent / 'bot_metadata.pkl', 'rb') as f:
            meta = pickle.load(f)
            st.divider()
            st.write(f"**Model Health (F1):** {meta['f1_score']:.2%}")
            st.write(f"**Prediction Error (MAE):** {meta['mae']:.2f}x")
            st.write(f"**Last Trained:** {meta['last_trained']}")
    except:
        st.write("Retrain to see performance metrics.")

    st.sidebar.markdown('---')
    prediction_threshold = st.sidebar.slider(
        'High multiplier probability threshold',
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help='Raise this threshold to reduce false HIGH recommendations and make predictions more conservative.'
    )

# ============================================
with st.sidebar.expander("🛠️ Model Architecture"):
    st.subheader("Scikit-Learn Models")
    st.write("- **Classifier:** RandomForestClassifier (Detects 10x+ patterns)")
    st.write("- **Regressor:** GradientBoostingRegressor (Predicts multiplier values)")    
    # Note: TensorFlow/LSTM is conditionally loaded if model file exists and is in requirements.txt.
    # This project prioritizes Scikit-learn for its robustness and lower overhead.
    st.info("Uses Scikit-learn models. LSTM is an optional dependency if its model file is present.")
    
    if classifier:
        st.subheader("Feature Importance (Classifier)")
        # Define feature names in the same order as they are created
        feature_names = [
            "Mean (Window)", "Std (Window)", "Min (Window)", "Max (Window)", "Median (Window)",
            "Last Multiplier", "2nd Last Multiplier", "3rd Last Multiplier",
            "Diff Last 2 Multipliers", "Freq >= 10x", "Freq >= 2x", "Freq < 1.5x",
            "Rounds Since High", "EMA 5", "Skewness", "Kurtosis", "Variance",
            "RSI-like", "Mean (Last 3)", "Freq < 1.2x"
        ]
        
        # Ensure the number of feature names matches the number of importances
        if len(feature_names) == len(classifier.feature_importances_):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': classifier.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            st.bar_chart(importance_df.set_index('Feature'))
        else:
            st.warning("Could not display feature importance: Mismatch between feature names and importance values.")
    else:
        st.info("Train the models first to see feature importances.")

# ============================================
# LIVE DATA ENTRY
# ============================================
st.markdown("---")
st.subheader("📡 Live Data Entry")
live_col1, live_col2 = st.columns([1, 1])

with live_col1:
    new_val = st.number_input("Last multiplier seen:", min_value=1.0, value=1.0, step=0.01, format="%.2f")
    if st.button("➕ Add Multiplier to History"):
        # Save to a dedicated live file
        live_file = Path(__file__).parent / 'live_multipliers.csv'
        write_header = not live_file.exists()
        with open(live_file, 'a') as f:
            if write_header: f.write("multiplier\n")
            f.write(f"{new_val}\n")
        st.toast(f"Added {new_val}x!")
        st.rerun()

with live_col2:
    st.info("""
    Adding a multiplier here saves it to `live_multipliers.csv` and automatically updates the prediction input below.
    """)

# ============================================
# RETRAINING SECTION (SIDEBAR)
# ============================================
st.sidebar.markdown("---")
st.sidebar.header("🔄 Model Management")
uploaded_files = st.sidebar.file_uploader("Upload New Data (CSV)", type="csv", accept_multiple_files=True)
if uploaded_files:
    for up_file in uploaded_files:
        save_path = Path(__file__).parent / up_file.name
        with open(save_path, "wb") as buffer:
            buffer.write(up_file.getbuffer())
    st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s)")

if st.sidebar.button("🚀 Start Retraining"):
    with st.sidebar.status("Retraining models...", expanded=True) as status:
        success, msg = run_full_retraining()
        if success:
            status.update(label="✅ Retraining Successful!", state="complete")
            st.cache_resource.clear()
            st.rerun()
        else:
            status.update(label=f"❌ Error: {msg}", state="error")

# --- Manual Rerun Button ---
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Rerun App"):
    st.rerun()

# Define constants (must match training script)
WINDOW_SIZE = 10
LSTM_SEQUENCE_LENGTH = 5

# ============================================
# INPUT SECTION
# ============================================
st.markdown("---")
st.subheader("📝 Enter Recent Multipliers")

# Auto-generate the last 10 multipliers string from loaded data
default_input = ", ".join([f"{x:.2f}" for x in multipliers[-10:]]) if len(multipliers) >= 10 else ""

col1, col2 = st.columns([3, 1])

with col1:
    # Text input for multipliers
    multiplier_input = st.text_area(
        "Enter last 10 multipliers (comma separated):",
        value=default_input,
        height=100
    )

with col2:
    st.markdown("### Quick Stats")
    st.metric("Total Records", len(multipliers))
    high_count = np.sum(multipliers >= 10)
    st.metric("High (10x+)", f"{high_count} ({high_count/len(multipliers)*100:.1f}%)")

# Parse input
try:
    recent_multipliers = [float(x.strip()) for x in multiplier_input.split(',')]
except:
    st.error("Please enter valid numbers separated by commas")
    st.stop()

# ============================================
# PREDICTION SECTION
# ============================================
if classifier is not None and len(recent_multipliers) >= 10:
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    # Combine historical data with user input for full_history_multipliers
    if len(recent_multipliers) == len(multipliers) and np.array_equal(np.array(recent_multipliers), multipliers[-len(recent_multipliers):]):
        combined_history = multipliers.tolist()
    else:
        combined_history = multipliers.tolist() + recent_multipliers
    
    result = predict_multiplier(
        full_history_multipliers=np.array(combined_history),
        recent_multipliers_input=recent_multipliers,
        classifier=classifier,
        gb_regressor=regressor,
        scaler=scaler,
        lstm_model=lstm_model,
        window=WINDOW_SIZE,
        seq_len=LSTM_SEQUENCE_LENGTH,
        threshold=prediction_threshold
    )

    if result is None:
        st.error("Not enough recent multiplier history to make a reliable prediction. Enter at least 10 values and try again.")
    else:
        # Store prediction in session state for syncing with the calculator
        st.session_state['last_prediction'] = result
        
        # Display results in columns
        # Using 2x2 grid for better mobile visibility
        res_row1_col1, res_row1_col2 = st.columns(2)
        res_row2_col1, res_row2_col2 = st.columns(2)
        
        with res_row1_col1:
            st.metric("Predicted", f"{result['predicted_value']:.2f}x")
        
        with res_row1_col2:
            st.metric("High Prob", f"{result['high_multiplier_probability']*100:.1f}%")
        
        with res_row2_col1:
            rec_color = "green" if result['recommendation'] == 'HIGH' else "red"
            st.markdown(f"**Recommendation:** :{rec_color}[{result['recommendation']}]")
        
        with res_row2_col2:
            confidence = "HIGH" if result['high_multiplier_probability'] > 0.5 else "MEDIUM" if result['high_multiplier_probability'] > 0.3 else "LOW"
            st.metric("Confidence", confidence)
    
    # Detailed analysis
    st.markdown("---")
    st.subheader("📈 Detailed Analysis")
    
    # Show recent multipliers chart
    chart_data = pd.DataFrame({
        'Round': range(1, len(recent_multipliers) + 1),
        'Multiplier': recent_multipliers
    })
    st.line_chart(chart_data.set_index('Round'))
    
    # Statistics
    st.write("**Recent Statistics:**")
    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
    with stats_col1:
        st.metric("Mean", f"{np.mean(recent_multipliers):.2f}")
    with stats_col2:
        st.metric("Std", f"{np.std(recent_multipliers):.2f}")
    with stats_col3:
        st.metric("Min", f"{np.min(recent_multipliers):.2f}")
    with stats_col4:
        st.metric("Max", f"{np.max(recent_multipliers):.2f}")
    with stats_col5:
        high_ratio = np.sum(np.array(recent_multipliers) >= 10) / len(recent_multipliers)
        st.metric("High Ratio", f"{high_ratio*100:.0f}%")
    
    # Recommendation
    st.markdown("---")
    if result['recommendation'] == 'HIGH':
        st.success("🎉 **RECOMMENDATION: PLACE BET** - High multiplier probability detected!")
    else:
        st.warning("⚠️ **RECOMMENDATION: WAIT** - Low multiplier probability. Consider waiting for better opportunity.")
elif classifier is None:
    st.info("🎯 Prediction will be available once models are trained. Use the sidebar to 'Start Retraining'.")
elif len(recent_multipliers) < 10:
    st.error(f"Please enter at least 10 multipliers. You entered {len(recent_multipliers)}.")

# ============================================
st.markdown("---")
st.subheader("💰 Bankroll Management")
bank_col1, bank_col2 = st.columns(2)
with bank_col1:
    total_bank = st.number_input("Current Balance", min_value=0.0, value=100.0, step=10.0, help="Your total available budget")
    risk_pct = st.slider("Risk Tolerance (%)", 0.5, 5.0, 1.0, 0.5, help="Professional traders usually risk 1-2% per round")
    
with bank_col2:
    base_bet = total_bank * (risk_pct / 100)
    st.metric("Recommended Bet", f"${base_bet:.2f}")
    st.info(f"Safe Streak: You can sustain **{int(total_bank/base_bet) if base_bet > 0 else 0}** consecutive losses.")

st.divider()
st.write("**🎯 Goal-Based Target**")

# Check if a prediction exists to allow syncing
has_prediction = 'last_prediction' in st.session_state
sync_with_model = False

if has_prediction:
    sync_with_model = st.checkbox("🔄 Sync Target with Model Prediction", value=False)

goal_col1, goal_col2 = st.columns(2)

with goal_col1:
    if sync_with_model:
        # Auto-calculate profit based on predicted multiplier: Profit = Bet * (Multiplier - 1)
        pred_mult = st.session_state['last_prediction']['predicted_value']
        calc_profit = base_bet * (pred_mult - 1)
        desired_profit = st.number_input("Target Profit (Synced)", value=max(0.01, float(calc_profit)), format="%.2f", disabled=True)
        st.caption(f"Based on predicted {pred_mult:.2f}x")
    else:
        desired_profit = st.number_input("Target Profit per Bet", min_value=0.1, value=10.0, step=5.0)

with goal_col2:
    if base_bet > 0:
        needed_multiplier = (desired_profit / base_bet) + 1
        st.write(f"To earn ${desired_profit:.2f} with a ${base_bet:.2f} bet:")
        st.markdown(f"### Required Exit: **{needed_multiplier:.2f}x**")
    else:
        st.write("Enter balance to calculate targets.")

# ============================================
# BACKTESTING SECTION
# ============================================
st.markdown("---")
st.subheader("🧪 Model Backtesting")
if classifier is None:
    st.info("Backtesting requires a trained model.")
else:
    with st.expander("Run Backtest on Historical Data", expanded=False):
        st.write("This simulates the model's performance on the most recent historical rounds.")
        
        # Determine safe limit for backtesting
        required_len = WINDOW_SIZE + (LSTM_SEQUENCE_LENGTH - 1 if lstm_model else 0)
        max_backtest = len(multipliers) - required_len
        
        if max_backtest > 5:
            num_rounds = st.number_input("Number of rounds to backtest", min_value=5, max_value=max_backtest, value=min(20, max_backtest))
            
            if st.button("▶️ Run Backtest"):
                backtest_results = []
                progress_bar = st.progress(0)
                
                end_idx = len(multipliers)
                start_idx = end_idx - num_rounds
                
                for i in range(start_idx, end_idx):
                    hist_at_t = multipliers[:i]
                    recent_at_t = multipliers[i - required_len : i]
                    actual_val = multipliers[i]
                    
                    pred = predict_multiplier(
                        full_history_multipliers=hist_at_t,
                        recent_multipliers_input=recent_at_t.tolist(),
                        classifier=classifier,
                        gb_regressor=regressor,
                        scaler=scaler,
                        lstm_model=lstm_model,
                        window=WINDOW_SIZE,
                        seq_len=LSTM_SEQUENCE_LENGTH
                    )
                    
                    if pred:
                        is_correct = (pred['recommendation'] == 'HIGH' and actual_val >= 10) or \
                                     (pred['recommendation'] == 'LOW' and actual_val < 10)
                        
                        backtest_results.append({
                            "Round": i,
                            "Actual": f"{actual_val:.2f}x",
                            "Predicted Value": f"{pred['predicted_value']:.2f}x",
                            "High Prob": f"{pred['high_multiplier_probability']:.1%}",
                            "Rec": pred['recommendation'],
                            "Result": "✅ Match" if is_correct else "❌ Miss"
                        })
                    progress_bar.progress((i - start_idx + 1) / num_rounds)
                
                if backtest_results:
                    res_df = pd.DataFrame(backtest_results)
                    matches = res_df[res_df["Result"] == "✅ Match"].shape[0]
                    accuracy = matches / len(backtest_results)
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("Backtest Accuracy", f"{accuracy:.1%}")
                    col_b.metric("Total Rounds", len(backtest_results))
                    st.dataframe(res_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Not enough data to run backtest. Need more historical records.")
    with bank_col1:
        total_bank = st.number_input("Current Balance", min_value=0.0, value=100.0, step=10.0, help="Your total available budget")
        risk_pct = st.slider("Risk Tolerance (%)", 0.5, 5.0, 1.0, 0.5, help="Professional traders usually risk 1-2% per round")
        
    with bank_col2:
        base_bet = total_bank * (risk_pct / 100)
        st.metric("Recommended Bet", f"${base_bet:.2f}")
        st.info(f"Safe Streak: You can sustain **{int(total_bank/base_bet) if base_bet > 0 else 0}** consecutive losses.")

    st.divider()
    st.write("**🎯 Goal-Based Target**")
    
    # Check if a prediction exists to allow syncing
    has_prediction = 'last_prediction' in st.session_state
    sync_with_model = False
    
    if has_prediction:
        sync_with_model = st.checkbox("🔄 Sync Target with Model Prediction", value=False)

    goal_col1, goal_col2 = st.columns(2)
    
    with goal_col1:
        if sync_with_model:
            # Auto-calculate profit based on predicted multiplier: Profit = Bet * (Multiplier - 1)
            pred_mult = st.session_state['last_prediction']['predicted_value']
            calc_profit = base_bet * (pred_mult - 1)
            desired_profit = st.number_input("Target Profit (Synced)", value=max(0.01, float(calc_profit)), format="%.2f", disabled=True)
            st.caption(f"Based on predicted {pred_mult:.2f}x")
        else:
            desired_profit = st.number_input("Target Profit per Bet", min_value=0.1, value=10.0, step=5.0)
    
    with goal_col2:
        if base_bet > 0:
            needed_multiplier = (desired_profit / base_bet) + 1
            st.write(f"To earn ${desired_profit:.2f} with a ${base_bet:.2f} bet:")
            st.markdown(f"### Required Exit: **{needed_multiplier:.2f}x**")
        else:
            st.write("Enter balance to calculate targets.")

# ============================================
# HISTORICAL DATA SECTION
# ============================================
st.markdown("---")
st.subheader("📊 Historical Data Preview")

if len(multipliers) > 0:
    # Feature: Download Combined Data
    csv_bytes = convert_df_to_csv(df)
    st.download_button(
        label="📥 Download Combined Dataset (CSV)",
        data=csv_bytes,
        file_name=f"aviator_combined_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    # Show last 20 multipliers
    st.write("Last 20 multipliers:")
    st.dataframe(pd.DataFrame({
        'Index': range(len(multipliers)-20, len(multipliers)),
        'Multiplier': multipliers[-20:]
    }))
    
    # Distribution
    st.write("Multiplier Distribution:")
    hist_data = pd.cut(multipliers, bins=[0, 1.5, 2, 5, 10, 20, 50, 100]).value_counts().sort_index()
    st.bar_chart(hist_data)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("*Note: This is a prediction model for educational purposes. Gamble responsibly.*")