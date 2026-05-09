#!/usr/bin/env python3
"""
Simplified model trainer - uses only scikit-learn (no TensorFlow/Keras)
"""
import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
import warnings
import pickle
from pathlib import Path
import glob

warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("TRAINING AVIATOR MODEL (Scikit-Learn Only)")
print("="*60)

# ============================================
# 1. LOAD AND PREPROCESS DATA
# ============================================ 
MODELS_DIR = Path(__file__).parent
SEARCH_PATTERN = str(MODELS_DIR / "*.csv")

csv_files = glob.glob(SEARCH_PATTERN)
print(f"\nSearching for CSV files in: {MODELS_DIR}")
print(f"Found {len(csv_files)} CSV files")

if not csv_files:
    raise FileNotFoundError(f"No multiplier CSV files found in {MODELS_DIR}")

print(f"CSV files: {csv_files}")
df_list = []
for f in csv_files:
    try:
        temp_df = pd.read_csv(f)
        if 'multiplier' not in temp_df.columns:
            temp_df.columns = ['multiplier'] + list(temp_df.columns[1:])
        df_list.append(temp_df)
        print(f"  ✓ Loaded {f} ({len(temp_df)} rows)")
    except Exception as e:
        print(f"  ✗ Error loading {f}: {e}")

if not df_list:
    raise ValueError("No valid CSV data could be loaded")

# Combine datasets
df = pd.concat(df_list, ignore_index=True).reset_index(drop=True) 
df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
df = df.dropna(subset=['multiplier'])
print(f"Total records loaded: {len(df)}")

# ============================================
# 2. CREATE FEATURES FOR PREDICTION
# ============================================
print("\nCreating features...")
multipliers = df['multiplier'].values

def create_features(multipliers, window=10):
    """Create features from historical multiplier data"""
    features, targets, targets_class = [], [], []
    last_high_idx = -1

    for i in range(len(multipliers)):
        if i >= window:
            window_data = np.array(multipliers[i-window:i])
            window_series = pd.Series(window_data)
            rounds_since_high = i - last_high_idx if last_high_idx != -1 else window
            
            diffs = window_series.diff().dropna()
            gains = diffs[diffs > 0].sum()
            losses = -diffs[diffs < 0].sum()
            rsi = 100 - (100 / (1 + gains / (losses if losses != 0 else 1e-9)))

            feature = [
                np.mean(window_data), np.std(window_data), np.min(window_data), np.max(window_data),
                np.median(window_data), multipliers[i-1], multipliers[i-2], multipliers[i-3],
                window_data[-1] - window_data[-2],
                np.sum(window_data >= 10) / window, np.sum(window_data >= 2) / window,
                np.sum(window_data < 1.5) / window, rounds_since_high,
                np.mean(window_data[-5:]), 0,
                0, np.var(window_data), rsi,
                np.mean(window_data[-3:]), np.sum(window_data < 1.2) / window
            ]
            feature = [0 if pd.isna(x) else x for x in feature]
            features.append(feature)
            targets.append(multipliers[i])
            targets_class.append(1 if multipliers[i] >= 10 else 0)

        if multipliers[i] >= 10:
            last_high_idx = i

    return np.array(features), np.array(targets), np.array(targets_class)

X, y_regress, y_class = create_features(multipliers, window=10)
print(f"Created {len(X)} samples with {X.shape[1]} features")

# ============================================
# 3. TRAIN CLASSIFICATION MODEL
# ============================================
print("\n" + "="*60)
print("TRAINING CLASSIFICATION MODEL")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, shuffle=False
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Random Forest Classifier...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

y_pred_class = rf_classifier.predict(X_test_scaled)
print(f"Classification Report:")
print(classification_report(y_test, y_pred_class))

# ============================================
# 4. TRAIN REGRESSION MODEL
# ============================================
print("\n" + "="*60)
print("TRAINING REGRESSION MODEL")
print("="*60)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_regress, test_size=0.2, random_state=42, shuffle=False
)

X_train_r_scaled = scaler.fit_transform(X_train_r)
X_test_r_scaled = scaler.transform(X_test_r)

print("\nTraining Gradient Boosting Regressor...")
gb_regressor = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
gb_regressor.fit(X_train_r_scaled, y_train_r)

y_pred_r = gb_regressor.predict(X_test_r_scaled)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
mae = mean_absolute_error(y_test_r, y_pred_r)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# ============================================
# 5. SAVE MODELS
# ============================================
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

try:
    with open(MODELS_DIR / 'aviator_classifier.pkl', 'wb') as f:
        pickle.dump(rf_classifier, f)
    print("✓ Saved aviator_classifier.pkl")
    
    with open(MODELS_DIR / 'aviator_regressor.pkl', 'wb') as f:
        pickle.dump(gb_regressor, f)
    print("✓ Saved aviator_regressor.pkl")
    
    with open(MODELS_DIR / 'aviator_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Saved aviator_scaler.pkl")
    
    metrics = {
        "f1_score": 0.0,
        "mae": mae,
        "records_used": len(multipliers),
        "last_trained": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    }
    with open(MODELS_DIR / 'bot_metadata.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print("✓ Saved bot_metadata.pkl")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE - All models saved successfully!")
    print("="*60)
    
except Exception as e:
    print(f"✗ Error saving models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
