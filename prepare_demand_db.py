import os
import pandas as pd
import sqlite3
import numpy as np

# ------------ CONFIG ------------
os.makedirs("data/raw", exist_ok=True)

cities = {
    "Cork_IE": {"base": 180, "boost_hour": [11, 17]},
    "Dublin_IE": {"base": 200, "boost_hour": [12, 18]},
    "London_UK": {"base": 220, "boost_hour": [10, 16]},
    "Berlin_DE": {"base": 210, "boost_hour": [13, 19]},
    "Madrid_ES": {"base": 230, "boost_hour": [14, 20]}
}

datetime_range = pd.date_range(start="2025-06-22 00:00:00", periods=168, freq="h")

data = []
for city, config in cities.items():
    for dt in datetime_range:
        base_demand = config["base"]

        # Weekend effect
        if dt.weekday() >= 5:
            base_demand *= 1.15

        # Afternoon boost
        if dt.hour in config["boost_hour"]:
            base_demand += 20

        # Add noise
        demand = np.random.normal(loc=base_demand, scale=10)
        demand = max(0, round(demand, 2))  # ensure non-negative
        data.append([city, dt, demand])

df = pd.DataFrame(data, columns=["city", "datetime", "demand"])
print("🧾 Generated realistic demand for cities:", df["city"].nunique())
print(df.groupby("city")["demand"].describe())

# ------------ Save to SQLite ------------
conn = sqlite3.connect("data/raw/geoai_store_demand.db")
df.to_sql("store_demand", conn, index=False, if_exists="replace")
conn.close()

print("✅ SQLite DB updated with realistic demand patterns.")
