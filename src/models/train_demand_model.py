import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def main():
    # 1. Load processed features
    features_path = os.path.join("data", "processed", "weather_features.csv")
    df = pd.read_csv(features_path)

    # 2. Simulate demand (if not present)
    if "demand" not in df.columns:
        np.random.seed(42)
        df["demand"] = np.random.poisson(lam=200, size=len(df))
    
    print(df.columns.tolist())


    # 3. Feature / target split
    X = df.drop(["time", "datetime", "city", "demand"], axis=1)

    y = df["demand"]

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate
    score = model.score(X_test, y_test)
    print(f"Model R² score: {score:.3f}")

    # 7. Save model + predictions
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/demand_model.pkl")

    os.makedirs("data/predictions", exist_ok=True)
    df["predicted_demand"] = model.predict(X)
    df.to_csv("data/predictions/forecasted_demand.csv", index=False)

if __name__ == "__main__":
    main()
