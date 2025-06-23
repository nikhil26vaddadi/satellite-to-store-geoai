"""
Simulate 48-hour future weather for five cities.
Creates CSVs in the same format as existing raw files so the
feature-engineering pipeline can process them unchanged.
"""

import os, random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

CITIES = {
    "Cork_IE":   (51.8985, -8.4756),
    "Dublin_IE": (53.3498, -6.2603),
    "London_UK": (51.5072, -0.1276),
    "Berlin_DE": (52.5200, 13.4050),
    "Madrid_ES": (40.4168, -3.7038),
}

# Simple seasonal baselines (°C, mm, %, m/s)
BASES = {
    "temperature_2m":    {"mean": 18, "std": 3},
    "precipitation":     {"mean":  0.5, "std": 0.6},
    "cloudcover":        {"mean": 40, "std": 30},
    "windspeed_10m":     {"mean":  4, "std": 1.5},
}

start_dt = datetime.utcnow()
hours     = 48            # 48-hour horizon

for city in CITIES.keys():
    rows = []
    for h in range(hours):
        ts = start_dt + timedelta(hours=h)
        row = {
            "time": ts.strftime("%Y-%m-%dT%H:%M"),
            "temperature_2m": np.random.normal(BASES["temperature_2m"]["mean"], BASES["temperature_2m"]["std"]),
            "precipitation":  max(0, np.random.normal(BASES["precipitation"]["mean"], BASES["precipitation"]["std"])),
            "cloudcover":     np.clip(np.random.normal(BASES["cloudcover"]["mean"], BASES["cloudcover"]["std"]), 0, 100),
            "windspeed_10m":  max(0, np.random.normal(BASES["windspeed_10m"]["mean"], BASES["windspeed_10m"]["std"]))
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # File name pattern: <ISOstamp>_<City>.csv  (matches your glob "*_[A-Z][A-Z].csv")
    iso_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname     = f"{iso_stamp}_{city}.csv"
    df.to_csv(os.path.join(RAW_DIR, fname), index=False)
    print(f"✅ 48-h weather sim saved → {fname}")
