"""
Quick & clean weather downloader for the Satellite-to-Store GeoAI project.
Author: Sai Nikhil Vaddadi
"""
from datetime import datetime
import os
import requests
import pandas as pd

# -------------------------------------------------------------------
# 1. CONFIG – edit these lists later as you add more store locations
# -------------------------------------------------------------------
LOCATIONS = {
    "Cork_IE": (51.8985, -8.4756),
    "Dublin_IE": (53.3498, -6.2603),
    "London_UK": (51.5072, -0.1276),
    "Berlin_DE": (52.5200, 13.4050),
    "Madrid_ES": (40.4168, -3.7038)
}

HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "cloudcover",
    "windspeed_10m"
]

FORECAST_DAYS = 3          # how many days ahead
OUT_DIR = "data/raw"   # keep relative to this script

# -------------------------------------------------------------------
# 2. HELPER – build URL for one location
# -------------------------------------------------------------------
def build_url(lat: float, lon: float) -> str:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": FORECAST_DAYS,
        "timezone": "auto"
    }
    return f"{base}?{'&'.join(f'{k}={v}' for k,v in params.items())}"

# -------------------------------------------------------------------
# 3. MAIN – loop over locations, fetch, save CSV
# -------------------------------------------------------------------
def main() -> None:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(OUT_DIR, exist_ok=True)

    for loc_name, (lat, lon) in LOCATIONS.items():
        url = build_url(lat, lon)
        print(f"Fetching {loc_name}: {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        data = resp.json()["hourly"]
        df = pd.DataFrame(data)
        csv_path = os.path.join(OUT_DIR, f"{ts}_{loc_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved → {csv_path}  ({len(df)} rows)")

if __name__ == "__main__":
    main()
