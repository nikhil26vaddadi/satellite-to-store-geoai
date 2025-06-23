"""
Feature engineering on hourly weather CSVs.
Author: Sai Nikhil Vaddadi
"""
import os
import pandas as pd
from glob import glob

# ------------ CONFIG ------------
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
LAG_HOURS = 24
ROLLING_HOURS = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------ HELPERS ------------
def load_weather_csvs(input_dir):
    files = sorted(glob(os.path.join(input_dir, "*_[A-Z][A-Z].csv")))
    if not files:
        print("❌ No weather files found in:", input_dir)
        return pd.DataFrame()

    dfs = []
    for f in files:
        # Extract city from filename (e.g., '20250623T122700Z_Berlin_DE.csv' → 'Berlin_DE')
        city = f.split("_")[-2] + "_" + f.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(f)
        df["datetime"] = pd.to_datetime(df["time"])
        df["city"] = city
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ------------ FEATURE BUILDING ------------
def build_features(df):
    df = df.sort_values(["city", "datetime"])
    for var in ["temperature_2m", "precipitation", "cloudcover", "windspeed_10m"]:
        df[f"{var}_lag24"] = df.groupby("city")[var].shift(LAG_HOURS)
        df[f"{var}_rollmean6"] = df.groupby("city")[var].transform(lambda x: x.rolling(ROLLING_HOURS).mean())
        df[f"{var}_high_flag"] = (df[var] > df[var].quantile(0.9)).astype(int)
    return df.dropna()

# ------------ MAIN PIPELINE ------------
def main():
    df = load_weather_csvs(INPUT_DIR)
    print(f"✅ Loaded {len(df)} rows of weather data")

    if df.empty:
        print("❌ Exiting: No weather data found.")
        return

    # Feature engineering
    df_feat = build_features(df)
    print(f"✅ After feature engineering: {len(df_feat)} rows")

    # Load demand data from SQLite
    import sqlite3
    conn = sqlite3.connect(os.path.join(INPUT_DIR, "geoai_store_demand.db"))
    demand_df = pd.read_sql_query("SELECT * FROM store_demand", conn)
    conn.close()

    # Ensure datetime columns match
    demand_df["datetime"] = pd.to_datetime(demand_df["datetime"])
    df_feat["datetime"] = pd.to_datetime(df_feat["datetime"])

    # Merge demand into weather features
    df_merged = df_feat.merge(demand_df, on=["city", "datetime"], how="left")

    # Final output
    out_path = os.path.join(OUTPUT_DIR, "features.csv")
    df_merged.to_csv(out_path, index=False)
    print(f"✅ Saved features+labels to {out_path} ({len(df_merged)} rows)")


if __name__ == "__main__":
    main()
