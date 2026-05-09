#!/usr/bin/env python3
"""Minimal training using only numpy and sklearn"""
import sys
import csv
from pathlib import Path
import pickle

# Simple CSV reader - multiplier is in column 2
def load_csv_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    val = float(parts[1])
                    if val > 0:
                        data.append(val)
                except:
                    pass
    return data

# Find CSV files
csv_dir = Path(__file__).parent
multipliers = []
for csv_file in csv_dir.glob("*.csv"):
    print(f"Loading {csv_file.name}...")
    multipliers.extend(load_csv_data(str(csv_file)))

if not multipliers:
    print("ERROR: No CSV data found!")
    sys.exit(1)

print(f"Loaded {len(multipliers)} multiplier records")

# Create simple features
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

multipliers = np.array(multipliers)
X, y_regress, y_class = [], [], []
window = 10

for i in range(window, len(multipliers)):
    window_data = multipliers[i-window:i]
    # Simple features
    feature = [
        np.mean(window_data),
        np.std(window_data),
        np.min(window_data),
        np.max(window_data),
        np.median(window_data),
        multipliers[i-1],
        multipliers[i-2] if i >= 2 else multipliers[i-1],
        multipliers[i-3] if i >= 3 else multipliers[i-1],
        window_data[-1] - window_data[-2],
        np.sum(window_data >= 10) / window,
        np.sum(window_data >= 2) / window,
        np.sum(window_data < 1.5) / window,
        window - i,  # rounds since high
        np.mean(window_data[-5:]),
        0, 0, np.var(window_data), 0,
        np.mean(window_data[-3:]),
        np.sum(window_data < 1.2) / window
    ]
    X.append(feature)
    y_regress.append(multipliers[i])
    y_class.append(1 if multipliers[i] >= 10 else 0)

X = np.array(X)
y_regress = np.array(y_regress)
y_class = np.array(y_class)

print(f"Created {len(X)} samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, shuffle=False
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
print("Training classifier...")
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train_scaled, y_train)
score = clf.score(X_test_scaled, y_test)
print(f"  Classifier accuracy: {score:.2%}")

# Train regressor
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_regress, test_size=0.2, random_state=42, shuffle=False
)
X_train_r_scaled = scaler.fit_transform(X_train_r)
X_test_r_scaled = scaler.transform(X_test_r)

print("Training regressor...")
reg = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
reg.fit(X_train_r_scaled, y_train_r)
score_r = reg.score(X_test_r_scaled, y_test_r)
print(f"  Regressor score: {score_r:.3f}")

# Save models
models_dir = Path(__file__).parent
print("\nSaving models...")
with open(models_dir / 'aviator_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("✓ aviator_classifier.pkl")

with open(models_dir / 'aviator_regressor.pkl', 'wb') as f:
    pickle.dump(reg, f)
print("✓ aviator_regressor.pkl")

with open(models_dir / 'aviator_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ aviator_scaler.pkl")

metrics = {"f1_score": 0.0, "mae": 0.0, "records_used": len(multipliers), "last_trained": "now"}
with open(models_dir / 'bot_metadata.pkl', 'wb') as f:
    pickle.dump(metrics, f)
print("✓ bot_metadata.pkl")

print("\n✅ Training complete!")
