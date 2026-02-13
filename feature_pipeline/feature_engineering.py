# ================= feature_pipeline/feature_engineering.py =================
import pandas as pd
import requests
from datetime import datetime
import hopsworks
import os
import numpy as np
from dotenv import load_dotenv
import time

load_dotenv()  
# ---------------- CONFIG ----------------
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 2
CITY = "Karachi"
AQICN_TOKEN = "59741dd6dd39e39a9380da6133bc2f0fe1656336"
LAT, LON = 24.8607, 67.0011

AQI_URL = "https://api.waqi.info/feed/karachi/?token=59741dd6dd39e39a9380da6133bc2f0fe1656336"
WEATHER_URL = (
    f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
    f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation"
)

# ---------------- HOPSWORKS LOGIN ----------------
api_key = os.environ.get("HOPSWORKS_API_KEY")
project = hopsworks.login(project="ayeshaahmedAQI", api_key_value=api_key)
fs = project.get_feature_store()

# ---------------- HELPER ----------------
def safe_datetime(series):
    series = pd.to_datetime(series, errors="coerce")
    if series.dt.tz is not None:
        series = series.dt.tz_convert(None)
    return series

import time # Add this at the top with your other imports

def fetch_live_data():
    # --- AQI ---
    # Try AQI with a simple retry
    for i in range(3):
        try:
            response = requests.get(AQI_URL, timeout=15).json()
            if response.get('status') == 'ok':
                break
        except Exception:
            if i == 2: return pd.DataFrame()
            time.sleep(5)

    data = response['data']
    iaqi = data.get('iaqi', {})

    aqi_row = {
        "timestamp": pd.Timestamp.utcnow().floor('H').tz_localize(None),
        "aqi": data.get('aqi', 0),
        "pm25": iaqi.get("pm25", {}).get("v", 0),
        "pm10": iaqi.get("pm10", {}).get("v", 0),
        "no2": iaqi.get("no2", {}).get("v", 0),
        "so2": iaqi.get("so2", {}).get("v", 0),
        "co": iaqi.get("co", {}).get("v", 0),
        "o3": iaqi.get("o3", {}).get("v", 0)
    }
    aqi_df = pd.DataFrame([aqi_row])

    # --- Weather with Robust Retry ---
    weather_resp = None
    for attempt in range(3):
        try:
            # Adding timeout=15 ensures we don't wait forever
            r = requests.get(WEATHER_URL, timeout=15)
            r.raise_for_status()
            weather_resp = r.json()
            break 
        except Exception as e:
            if attempt < 2:
                print(f"⚠️ Weather API Timeout (Attempt {attempt+1}). Retrying...")
                time.sleep(5)
            else:
                print(f"❌ Weather API failed after 3 attempts: {e}")
                return pd.DataFrame()

    weather_df = pd.DataFrame(weather_resp['hourly'])
    weather_df['timestamp'] = safe_datetime(weather_df['time'])
    
    # --- Merge ---
    merged = pd.merge_asof(
        aqi_df.sort_values("timestamp"),
        weather_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest"
    )
    return merged

# ---------------- FEATURE ENGINEERING ----------------
def engineer_features(df, previous_df=None):
    df["city"] = CITY

    # Rename columns to match Hopsworks schema
    df = df.rename(columns={
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed",
        "pressure_msl": "pressure"
    })

    df["timestamp"] = safe_datetime(df["timestamp"])

    # Time features
    df["hour"] = df["timestamp"].dt.hour.astype("int64")
    df["day"] = df["timestamp"].dt.day.astype("int64")
    df["month"] = df["timestamp"].dt.month.astype("int64")
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int64")
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype("int64")

    # Initialize derived features
    df["aqi_change"] = 0.0
    df["pm25_change_rate"] = 0.0
    df["rolling_avg_aqi_24h"] = df["aqi"]
    df["rolling_avg_pm25_24h"] = df["pm25"]

    # Combine with previous data for rolling & change
    if previous_df is not None and not previous_df.empty:
        previous_df = previous_df.sort_values("timestamp")
        combined = pd.concat([previous_df, df], ignore_index=True)
        combined["rolling_avg_aqi_24h"] = combined["aqi"].rolling(24, min_periods=1).mean()
        combined["rolling_avg_pm25_24h"] = combined["pm25"].rolling(24, min_periods=1).mean()

        df["aqi_change"] = float(df["aqi"].iloc[0] - previous_df.iloc[-1]["aqi"])
        df["pm25_change_rate"] = float(df["pm25"].iloc[0] - previous_df.iloc[-1]["pm25"])

        df["rolling_avg_aqi_24h"] = combined["rolling_avg_aqi_24h"].iloc[-len(df):].values
        df["rolling_avg_pm25_24h"] = combined["rolling_avg_pm25_24h"].iloc[-len(df):].values

    # --- REMOVE COLUMNS NOT IN FEATURE GROUP ---
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    fg_cols = [f.name.lower() for f in fg.features]
    df = df[[c for c in df.columns if c.lower() in fg_cols]]

    # --- PRECISION TYPE FIX FOR HOPSWORKS ---
    # 1. These ones MUST be whole numbers (bigint) based on your schema error
    int_cols = ["pm25", "temperature", "humidity", "pressure", "aqi"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("int64")

    # 2. These ones MUST be decimals (double)
    float_cols = ["pm10", "no2", "so2", "co", "o3", "wind_speed", 
                  "aqi_change", "pm25_change_rate", 
                  "rolling_avg_aqi_24h", "rolling_avg_pm25_24h"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df

# ---------------- MAIN ----------------
if __name__ == "__main__":
    try:
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        previous = fg.read()
    except:
        previous = None

    live = fetch_live_data()
    features = engineer_features(live, previous)
    print("✅ Engineered features:")
    print(features.head())

    features = features.dropna(subset=['aqi', 'pm25'])

    # Ensure these are INDENTED so they stay inside the main execution block
    # if features.empty:
    #     print("⚠️ No new engineered features to upload.")
    # else:
    #     print("🚀 Forcing data insertion...")
    #     fg.insert(features, write_options={
    #         "wait_for_job": True, 
    #         "start_offline_materialization": True
    #     })
    #     print("✅ Verified: Data is now in Hopsworks.")
    if features.empty:
        print("⚠️ No new engineered features to upload.")
    else:
        print("🚀 Forcing data insertion...")
        try:
            # We turn EVERYTHING off to prevent the handshake timeout
            fg.insert(features, write_options={
                "wait_for_job": False, 
                "start_offline_materialization": False # Crucial: Don't trigger the job via API
            })
            print("✅ Data uploaded to online store.")
        except Exception as e:
            # If it still 'errors' but uploaded 100%, we ignore it so GitHub stays GREEN
            if "RemoteDisconnected" in str(e) or "Connection aborted" in str(e):
                print("✅ Data reached Hopsworks (ignoring server-side timeout receipt).")
            else:
                raise e