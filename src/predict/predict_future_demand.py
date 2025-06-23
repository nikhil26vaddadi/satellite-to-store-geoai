"""
Predict demand for future weather rows (those missing 'demand').
Reads engineered feature file and XGBoost model, writes predictions CSV.
"""

import os, joblib, pandas as pd
from xgboost import XGBRegressor

FEATURES_PATH   = "data/processed/features.csv"
MODEL_PATH      = "models/xgb_demand_model.pkl"
PREDICT_OUT     = "data/predictions/future_demand_48h.csv"

os.makedirs(os.path.dirname(PREDICT_OUT), exist_ok=True)

# Load engineered features
df_feat = pd.read_csv(FEATURES_PATH)
future_rows = df_feat[df_feat["demand"].isna()].copy()
if future_rows.empty:
    print("❌ No future rows found (all rows already have demand).")
    quit()

# Prepare X matrix
drop_cols = ["time", "datetime", "city", "demand"]
X = future_rows.drop(columns=drop_cols, errors="ignore")

# Load model & predict
model: XGBRegressor = joblib.load(MODEL_PATH)
future_rows["predicted_demand"] = model.predict(X)

# Save
future_rows[["city", "datetime", "predicted_demand"]].to_csv(PREDICT_OUT, index=False)
print(f"✅ Future demand forecast saved → {PREDICT_OUT}")
