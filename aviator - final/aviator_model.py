"""
Aviator Multiplier Prediction Model
Predicts high multipliers (10x+) for the Aviator game
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
import warnings
try:
    import keras_tuner as kt
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except Exception as ex:
    print(f"TensorFlow/Keras not available: {ex}")
    kt = None
    tf = None
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    TF_AVAILABLE = False
from pathlib import Path
import glob
warnings.filterwarnings('ignore')

def predict_next_multiplier(history, recent_window, lstm_model, gb_regressor, scaler, sequence_length=5, window=10):
    """
    Predict the next multiplier based on recent history
    history: full list of multipliers for calculating 'rounds since high'
    recent_window: list of last 15 multipliers (needed to create sequence for LSTM)
    """
    required_len = window + sequence_length - 1
    if len(recent_window) < required_len:
        print(f"Need at least {required_len} recent multipliers for LSTM sequence!")
        return None
    
    # We need to generate a SEQUENCE of feature vectors for the LSTM
    sequence_features = []
    for s in range(sequence_length):
        # Sliding window for each step in the sequence
        start_idx = len(recent_window) - (window + sequence_length - 1) + s
        sub_window = recent_window[start_idx : start_idx + window]
        window_data = np.array(sub_window)
        window_series = pd.Series(window_data)
        
        # Calculate rounds since high relative to this step in history
        current_hist_len = len(history) - (sequence_length - 1) + s
        hist_slice = history[:current_hist_len]
        high_indices = np.where(np.array(hist_slice) >= 10)[0]
        rounds_since_high = current_hist_len - high_indices[-1] if len(high_indices) > 0 else 10

        # RSI Calculation
        diffs = window_series.diff().dropna()
        gains = diffs[diffs > 0].sum()
        losses = -diffs[diffs < 0].sum()
        rsi = 100 - (100 / (1 + gains / (losses if losses != 0 else 1e-9)))

        feature = [
            np.mean(window_data), np.std(window_data), np.min(window_data), np.max(window_data),
            np.median(window_data), sub_window[-1], sub_window[-2], sub_window[-3],
            window_data[-1] - window_data[-2],
            np.sum(window_data >= 10) / window, np.sum(window_data >= 2) / window,
            np.sum(window_data < 1.5) / window, rounds_since_high,
            window_series.ewm(span=5).mean().iloc[-1], window_series.skew(),
            window_series.kurt(), window_series.var(), rsi,
            window_series.tail(3).mean(), np.sum(window_data < 1.2) / window
        ]
        sequence_features.append([0 if pd.isna(x) else x for x in feature])
    
    # Scale the sequence
    sequence_scaled = scaler.transform(sequence_features)
    
    # Get classification prediction (LSTM expects shape [1, 5, features])
    high_prob = lstm_model.predict(sequence_scaled.reshape(1, sequence_length, sequence_scaled.shape[1]), verbose=0)[0][0]
    
    # Get regression prediction (Uses only the most recent feature vector)
    predicted_value = gb_regressor.predict(sequence_scaled[-1].reshape(1, -1))[0]
    
    return {
        'high_multiplier_probability': high_prob,
        'predicted_value': max(1.0, predicted_value),
        'recommendation': 'HIGH' if high_prob > 0.3 else 'LOW'
    }

def train_my_models():
    # ============================================
    # 1. LOAD AND PREPROCESS DATA
    # ============================================ 
    MODELS_DIR = Path(__file__).parent # Use relative path to the script's directory
    SEARCH_PATTERN = str(MODELS_DIR / "*.csv")

    csv_files = glob.glob(SEARCH_PATTERN)
    if not csv_files:
        raise FileNotFoundError(f"No multiplier CSV files found in {MODELS_DIR}")

    print(f"Found {len(csv_files)} datasets. Merging...")
    df_list = []
    for f in csv_files:
        temp_df = pd.read_csv(f)
        if 'multiplier' not in temp_df.columns:
            temp_df.columns = ['multiplier'] + list(temp_df.columns[1:])
        df_list.append(temp_df)

    # Combine datasets without sorting by value to preserve the sequence order
    df = pd.concat(df_list, ignore_index=True).reset_index(drop=True) 
    df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
    df = df.dropna(subset=['multiplier'])
    print(f"Total records loaded: {len(df)}")

    # ============================================
    # 2. CREATE FEATURES FOR PREDICTION
    # ============================================
    print("\nCreating features...")

    # Convert to numpy array for processing
    if 'multiplier' not in df.columns:
        raise ValueError("DataFrame must contain a 'multiplier' column.")

    multipliers = df['multiplier'].values

    # Create lag features (previous multipliers)
    def create_features(multipliers, window=10):
        """Create features from historical multiplier data"""
        features, targets, targets_class = [], [], []
        last_high_idx = -1

        for i in range(len(multipliers)):
            # Target index is i, features come from i-window to i-1
            if i >= window:
                window_data = np.array(multipliers[i-window:i])
                window_series = pd.Series(window_data)
                rounds_since_high = i - last_high_idx if last_high_idx != -1 else window
                
                # Calculate RSI-like indicator
                diffs = window_series.diff().dropna()
                gains = diffs[diffs > 0].sum()
                losses = -diffs[diffs < 0].sum()
                rsi = 100 - (100 / (1 + gains / (losses if losses != 0 else 1e-9)))

                feature = [
                    np.mean(window_data),           # Mean of last 10
                    np.std(window_data),            # Std of last 10
                    np.min(window_data),            # Min of last 10
                    np.max(window_data),            # Max of last 10
                    np.median(window_data),         # Median of last 10
                    multipliers[i-1],              # Last multiplier
                    multipliers[i-2],              # 2nd last
                    multipliers[i-3],              # 3rd last
                    window_data[-1] - window_data[-2],  # Trend
                    np.sum(window_data >= 10) / window,  # High multiplier ratio
                    np.sum(window_data >= 2) / window,   # Above 2x ratio
                    np.sum(window_data < 1.5) / window,  # Low multiplier ratio
                    rounds_since_high,               # Feature: distance since last big win
                    # New technical indicators (matching UI)
                    window_series.ewm(span=5).mean().iloc[-1], # EMA 5
                    window_series.skew(), # Skewness
                    window_series.kurt(), # Kurtosis
                    window_series.var(),  # Variance
                    rsi,                  # RSI-like
                    window_series.tail(3).mean(), # Short-term trend
                    np.sum(window_data < 1.2) / window # Extreme low frequency
                ]
                # Clean NaNs (common in skew/kurt for constant windows)
                feature = [0 if pd.isna(x) else x for x in feature]
                features.append(feature)
                targets.append(multipliers[i])
                targets_class.append(1 if multipliers[i] >= 10 else 0)

            if multipliers[i] >= 10:
                last_high_idx = i

        return np.array(features), np.array(targets), np.array(targets_class)

    # Create features with a window of 10
    # X contains individual feature vectors
    # y_regress contains the actual next multiplier value
    # y_class contains 0/1 indicating if the next multiplier is >= 10
    X, y_regress, y_class = create_features(multipliers, window=10)
    print(f"Created {len(X)} samples with {X.shape[1]} features")

    # ============================================
    # 3. TRAIN CLASSIFICATION MODEL (High vs Low)
    # ============================================
    print("\n" + "=" * 60)
    print("TRAINING CLASSIFICATION MODEL")
    print("=" * 60)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, shuffle=False
    )

    # Scale individual features (used for both LSTM timesteps and regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ============================================
    # LSTM MODEL FOR CLASSIFICATION
    # ============================================
    LSTM_SEQUENCE_LENGTH = 5  # Number of previous feature vectors to consider for LSTM
    history = None
    accuracy = None

    if TF_AVAILABLE:
        def create_lstm_dataset(X_features, y_class_targets, sequence_length):
            X_lstm, y_lstm = [], []
            for i in range(len(X_features) - sequence_length):
                X_lstm.append(X_features[i : (i + sequence_length)])
                y_lstm.append(y_class_targets[i + sequence_length])
            return np.array(X_lstm), np.array(y_lstm)

        # Create LSTM datasets
        X_lstm, y_lstm_class = create_lstm_dataset(X_train_scaled, y_train, LSTM_SEQUENCE_LENGTH)
        X_test_lstm, y_test_lstm = create_lstm_dataset(X_test_scaled, y_test, LSTM_SEQUENCE_LENGTH)

        print(f"\nCreated {len(X_lstm)} LSTM training samples with sequence length {LSTM_SEQUENCE_LENGTH}")
        print(f"Created {len(X_test_lstm)} LSTM test samples with sequence length {LSTM_SEQUENCE_LENGTH}")

        # Define the Keras Tuner model-building function
        def build_lstm_model(hp):
            num_features_per_timestep = X_lstm.shape[2]
            model = Sequential()
            
            # Tune the number of LSTM layers
            for i in range(hp.Int('num_lstm_layers', 1, 3)):
                model.add(LSTM(units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=128, step=32),
                               activation='relu',
                               return_sequences=True if i < hp.Int('num_lstm_layers', 1, 3) - 1 else False,
                               input_shape=(LSTM_SEQUENCE_LENGTH, num_features_per_timestep) if i == 0 else None))
                model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
            
            model.add(Dense(units=1, activation='sigmoid'))
            
            # Tune the learning rate for the optimizer
            learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            return model

        # Instantiate the Keras Tuner
        tuner = kt.Hyperband(
            build_lstm_model,
            objective='val_accuracy',
            max_epochs=50,
            factor=3,
            directory='keras_tuner_dir',
            project_name='aviator_lstm_tuning',
            overwrite=True
        )

        # Define early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        print("\nStarting Hyperparameter Tuning for LSTM Classifier...")
        tuner.search(X_lstm, y_lstm_class,
                     epochs=50,
                     validation_split=0.1,
                     callbacks=[early_stopping],
                     verbose=1)

        # Get the optimal hyperparameters and the best model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the best model and train it to capture the history object
        lstm_model = tuner.hypermodel.build(best_hps)
        print("\nTraining final LSTM model with best hyperparameters...")
        history = lstm_model.fit(X_lstm, y_lstm_class, epochs=50, validation_split=0.1, callbacks=[early_stopping], verbose=0)

        print(f"\nOptimal LSTM Hyperparameters:")
        print(f"  Number of LSTM layers: {best_hps.get('num_lstm_layers')}")
        for i in range(best_hps.get('num_lstm_layers')):
            print(f"  LSTM units in layer {i+1}: {best_hps.get(f'lstm_units_{i}')}")
            print(f"  Dropout rate in layer {i+1}: {best_hps.get(f'dropout_{i}'):.2f}")
        print(f"  Learning rate: {best_hps.get('learning_rate')}")

        # Evaluate the best LSTM model
        print("\nEvaluating the best LSTM Classifier...")
        loss, accuracy = lstm_model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2%}")

        y_pred_lstm_prob = lstm_model.predict(X_test_lstm)
        y_pred_lstm_class = (y_pred_lstm_prob > 0.5).astype(int)
        print("\nLSTM Classification Report (Best Model):")
        print(classification_report(y_test_lstm, y_pred_lstm_class))
    else:
        print("\nTensorFlow not available; skipping LSTM classifier tuning and training.")
        lstm_model = None

    # ============================================
    # 4. TRAIN REGRESSION MODEL (Predict Value)
    # ============================================
    print("\n" + "=" * 60)
    print("TRAINING REGRESSION MODEL")
    print("=" * 60)

    # Split for regression
    # For regression, we still use the individual feature vectors, not sequences.
    # We need to ensure the regression targets align with the classification targets if we want to use the same test set.
    # For simplicity, let's use the same split as the original classification, but with y_regress.
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_regress, test_size=0.2, random_state=42, shuffle=False)
    # Scale
    X_train_r_scaled = scaler.fit_transform(X_train_r)
    X_test_r_scaled = scaler.transform(X_test_r)

    # Train Gradient Boosting Regressor
    print("\nTraining Gradient Boosting Regressor...")
    gb_regressor = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    gb_regressor.fit(X_train_r_scaled, y_train_r)

    # Evaluate regressor
    y_pred_r = gb_regressor.predict(X_test_r_scaled)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    mae = mean_absolute_error(y_test_r, y_pred_r)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # ============================================
    # 6. TEST WITH SAMPLE DATA
    # ============================================
    print("\n" + "=" * 60)
    print("TESTING MODEL WITH RECENT DATA")
    print("=" * 60)

    # Use last 15 multipliers as test
    test_recent = multipliers[-15:].tolist()
    print(f"\nRecent multipliers: {test_recent}")

    # Make prediction for each position
    history_so_far = multipliers[:-15].tolist()
    if lstm_model is None:
        print("\nSkipping example LSTM test because TensorFlow/LSTM is unavailable.")
    else:
        for i in range(10, len(test_recent)):
            recent = test_recent[i-10:i]
            result = predict_next_multiplier(history_so_far + test_recent[:i], recent, lstm_model, gb_regressor, scaler, LSTM_SEQUENCE_LENGTH, window=10)
            actual = test_recent[i]
            
            print(f"\nPosition {i}:")
            print(f"  Recent: {recent[-5:]}")
            print(f"  Predicted: {result['predicted_value']:.2f} (High prob: {result['high_multiplier_probability']:.2%})")
            print(f"  Actual: {actual:.2f}")
            print(f"  Recommendation: {result['recommendation']}")

    # ============================================
    # 7. SAVE MODEL
    # ============================================
    import pickle

    # 1. Save LSTM (for advanced use)
    if lstm_model is not None:
        lstm_model.save(MODELS_DIR / 'aviator_lstm_classifier.h5')

    # 2. Save Scikit-Learn Classifier (Required by UI)
    print("Training Scikit-Learn Classifier for UI compatibility...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)
    with open(MODELS_DIR / 'aviator_classifier.pkl', 'wb') as f:
        pickle.dump(rf_classifier, f)

    # 3. Save Regressor and Scaler
    with open(MODELS_DIR / 'aviator_regressor.pkl', 'wb') as f:
        pickle.dump(gb_regressor, f)
    with open(MODELS_DIR / 'aviator_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 4. Save Metadata (Required by UI Diagnostics)
    history_df = pd.DataFrame(history.history) if history is not None else pd.DataFrame()
    metrics = {
        "f1_score": float(accuracy) if accuracy is not None else 0.0,
        "mae": mae,
        "records_used": len(multipliers),
        "last_trained": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    }
    with open(MODELS_DIR / 'bot_metadata.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    print("\n" + "=" * 60)
    print("MODELS SAVED SUCCESSFULLY!")
    print(f"Files saved to {MODELS_DIR}:")
    print("  - aviator_classifier.pkl (Scikit-Learn)")
    if lstm_model is not None:
        print("  - aviator_lstm_classifier.h5 (TensorFlow)")
    print("  - aviator_regressor.pkl")
    print("  - aviator_scaler.pkl")
    print("  - bot_metadata.pkl")

if __name__ == "__main__":
    try:
        train_my_models()
    except Exception as e:
        print("\n" + "!" * 60)
        print("CRITICAL ERROR: Training script failed.")
        print(f"Error details: {e}")
        print("!" * 60)
        import traceback
        traceback.print_exc()