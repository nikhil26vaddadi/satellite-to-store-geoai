"""
Train XGBoost model to predict store demand based on weather features.
Author: Sai Nikhil Vaddadi
"""

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ------------ CONFIG ------------
DATA_PATH = "data/processed/features.csv"
MODEL_PATH = "models/xgb_demand_model.pkl"
TARGET = "demand"
DROP_COLS = ["time", "datetime", "city", TARGET]

# ------------ MAIN PIPELINE ------------
def main():
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded data: {df.shape}")

    # Clean rows with missing demand
    df = df.dropna(subset=[TARGET])

    # Define features and target
    X = df.drop(DROP_COLS, axis=1)
    y = df[TARGET]

    print("✅ Loaded features:", df.shape)
    print(df[["city", "datetime", "demand"]].dropna().head())
    print("❌ Missing demand rows:", df["demand"].isna().sum())


    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ MAE: {mae:.2f}")
    print(f"✅ R² Score: {r2:.3f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
