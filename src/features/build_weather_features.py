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
    files = sorted(glob(os.path.join(input_dir, "*.csv")))
    dfs = []
    for f in files:
        city = f.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(f)
        df["datetime"] = pd.to_datetime(df["time"])
        df["city"] = city
        dfs.append(df)
    return pd.concat(dfs)

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
    print(f"Loaded {len(df)} rows")
    df_feat = build_features(df)
    out_path = os.path.join(OUTPUT_DIR, "weather_features.csv")
    df_feat.to_csv(out_path, index=False)
    print(f"Saved features to {out_path} ({len(df_feat)} rows)")

if __name__ == "__main__":
    main()
