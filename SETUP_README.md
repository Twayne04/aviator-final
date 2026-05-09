# Aviator Multiplier Predictor - Setup Guide

## Quick Start (Recommended)

### Option 1: Using the Setup Script (Easiest)
1. Double-click `setup_environment.bat` in your Downloads folder
2. Follow the on-screen instructions
3. After setup completes, run: `streamlit run aviator_ui.py`

### Option 2: Manual Setup (If you prefer)
Open a terminal in the Downloads folder and run:

```powershell
# Create virtual environment
python -m venv venv

# Activate the environment
.\venv\Scripts\Activate

# Install dependencies
pip install streamlit pandas scikit-learn

# Run the app
streamlit run aviator_ui.py
```

---

## Troubleshooting

### Issue: "Install 'conda' to create conda environments"
**Solution:** This is just a warning. You don't need Conda - the virtual environment (venv) approach works perfectly fine. Run `setup_environment.bat` to create a proper environment.

### Issue: "Select Interpreter" warning in VS Code
**Solution:** 
1. Click on the Python version in the bottom right of VS Code
2. Select `venv\Scripts\python.exe` as your interpreter
3. If you don't see it, restart VS Code after running the setup script

### Issue: TensorFlow/Keras errors
**Solution:** This version has been migrated to Scikit-Learn only. TensorFlow is no longer required.

### Issue: Data file not found
**Solution:** Ensure `all_spincity_multipliers_combined.csv` is in the same folder as `aviator_ui.py`. Both files should be in your Downloads folder.

---

## Project Structure

```
Downloads/
├── aviator_ui.py              # Main Streamlit app
├── aviator_model.py           # Model training script
├── all_spincity_multipliers_combined.csv  # Data file
├── aviator_classifier.pkl     # Trained Scikit-learn classifier
├── aviator_regressor.pkl      # Trained regressor
├── aviator_scaler.pkl         # Fitted scaler
├── venv/                      # Virtual environment (created by setup)
├── setup_environment.bat     # Setup script
└── requirements.txt          # Package requirements
```

---

## Common Commands

| Action | Command |
|--------|---------|
| Activate environment | `.\venv\Scripts\Activate` |
| Run the app | `streamlit run aviator_ui.py` |
| Check installed packages | `pip list` |
| Update packages | `pip install --upgrade -r requirements.txt` |

---

## Notes

- The virtual environment (`venv`) keeps your project dependencies isolated
- You need to activate the environment every time you open a new terminal
- If you see NaN errors in predictions, the data cleaning code in aviator_ui.py (line ~307) handles this automatically