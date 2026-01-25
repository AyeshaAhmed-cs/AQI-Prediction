# ================= backfill_pipeline/backfill_one_month.py =================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()
# ---------------- CONFIG ----------------
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 2
CITY = "Karachi"
HOURS_BACK = 24 * 30  # 1 month of hourly data

# ---------------- GENERATE HISTORICAL TIMESTAMPS ----------------
end_time = datetime.now()
timestamps = [end_time - timedelta(hours=i) for i in range(HOURS_BACK)]
timestamps.reverse()  # oldest first

# ---------------- SIMULATE AQI + WEATHER DATA ----------------
np.random.seed(42)
df = pd.DataFrame({
    "city": CITY,
    "timestamp": pd.to_datetime(timestamps),
    "aqi": np.random.randint(50, 200, size=HOURS_BACK),
    "pm25": np.random.randint(10, 150, size=HOURS_BACK),
    "pm10": np.random.uniform(20, 200, size=HOURS_BACK),
    "no2": np.random.uniform(10, 100, size=HOURS_BACK),
    "so2": np.random.uniform(5, 50, size=HOURS_BACK),
    "co": np.random.uniform(0.2, 2.0, size=HOURS_BACK),
    "o3": np.random.uniform(10, 80, size=HOURS_BACK),
    "temperature": np.random.uniform(15, 40, size=HOURS_BACK),
    "humidity": np.random.uniform(30, 90, size=HOURS_BACK),
    "pressure": np.random.uniform(950, 1050, size=HOURS_BACK),
    "wind_speed": np.random.uniform(0, 15, size=HOURS_BACK)
})

# ---------------- TIME FEATURES ----------------
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# ---------------- DERIVED FEATURES ----------------
df["aqi_change"] = df["aqi"].diff().fillna(0)
df["pm25_change_rate"] = df["pm25"].diff().fillna(0)
df["rolling_avg_aqi_24h"] = df["aqi"].rolling(24, min_periods=1).mean()
df["rolling_avg_pm25_24h"] = df["pm25"].rolling(24, min_periods=1).mean()

# ---------------- SORT & REMOVE DUPLICATES ----------------
df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

# ---------------- FORCE TYPES TO MATCH HOPSWORKS ----------------
for col in ["pm10", "co", "o3", "rolling_avg_aqi_24h", "rolling_avg_pm25_24h", "aqi_change", "pm25_change_rate"]:
    df[col] = df[col].astype(np.float64)

for col in ["temperature", "humidity", "pressure", "hour", "day", "month", "day_of_week", "is_weekend", "aqi", "pm25"]:
    df[col] = df[col].round(0).astype(np.int64)

# ---------------- HOPSWORKS LOGIN ----------------
api_key = os.environ["HOPSWORKS_API_KEY"]
project = hopsworks.login(project="ayeshaahmedAQI", api_key_value=api_key)
fs = project.get_feature_store()

# ---------------- GET FEATURE GROUP ----------------
fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

# ---------------- INSERT INTO FEATURE GROUP ----------------
try:
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"✅ Successfully backfilled {len(df)} rows for 1 month")
except Exception as e:
    print("❌ ERROR during backfill:")
    print(e)
