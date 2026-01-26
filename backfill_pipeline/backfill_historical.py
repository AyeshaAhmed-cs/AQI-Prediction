import pandas as pd
import numpy as np
import requests
import hopsworks
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# --- 1. FETCH REAL DATA ---
def fetch_historical_data():
    LAT, LON = 24.8607, 67.0011
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
    aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}&hourly=pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone"

    weather_data = requests.get(weather_url).json()['hourly']
    aqi_data = requests.get(aqi_url).json()['hourly']

    df = pd.merge(pd.DataFrame(weather_data), pd.DataFrame(aqi_data), on="time")
    df['time'] = pd.to_datetime(df['time'])
    
    df = df.rename(columns={
        "time": "timestamp", "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity", "wind_speed_10m": "wind_speed",
        "pressure_msl": "pressure", "pm2_5": "pm25",
        "nitrogen_dioxide": "no2", "sulphur_dioxide": "so2",
        "carbon_monoxide": "co", "ozone": "o3"
    })
    
    # Calculate Target
    df['aqi'] = (df['pm25'] * 1.5 + df['pm10'] * 0.5)
    df['city'] = "Karachi"
    return df

# --- 2. ENGINEER MISSING FEATURES ---
def engineer_backfill(df):
    df = df.sort_values("timestamp")
    
    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["aqi_change"] = df["aqi"].diff().fillna(0.0)
    df["pm25_change_rate"] = df["pm25"].diff().fillna(0.0)
    df["rolling_avg_aqi_24h"] = df["aqi"].rolling(24, min_periods=1).mean().fillna(0.0)
    df["rolling_avg_pm25_24h"] = df["pm25"].rolling(24, min_periods=1).mean().fillna(0.0)

    # Force Types to match Hopsworks Schema
    int_cols = ["aqi", "pm25", "temperature", "humidity", "pressure", "hour", "day", "month", "day_of_week", "is_weekend"]
    for col in int_cols:
        df[col] = df[col].round(0).astype(np.int64)
        
    float_cols = ["pm10", "no2", "so2", "co", "o3", "wind_speed", "aqi_change", "pm25_change_rate", "rolling_avg_aqi_24h", "rolling_avg_pm25_24h"]
    for col in float_cols:
        df[col] = df[col].astype(float)
        
    return df

# --- 3. UPLOAD ---
if __name__ == "__main__":
    print("🚀 Fetching and Engineering REAL data...")
    df = fetch_historical_data()
    df = engineer_backfill(df)

    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=2)

    print("📤 Uploading...")
    fg.insert(df, write_options={"wait_for_job": False}) 
    print("✅ MISSION SUCCESS! Check Hopsworks UI.")
