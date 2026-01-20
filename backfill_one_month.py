import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hopsworks

# ---------------- CONFIG ----------------
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 2
CITY = "Karachi"
HOURS_BACK = 24 * 30  # 1 month of hourly data
# ---------------------------------------

# 1️⃣ Generate timestamps for the past month
end_time = datetime.now()
timestamps = [end_time - timedelta(hours=i) for i in range(HOURS_BACK)]
timestamps.reverse()  # oldest first

# 2️⃣ Simulate realistic AQI & pollutants
np.random.seed(42)
aqi = np.random.randint(50, 200, size=HOURS_BACK)
pm25 = np.random.randint(10, 150, size=HOURS_BACK)
pm10 = np.random.uniform(20, 200, size=HOURS_BACK)        # double
no2 = np.random.uniform(10, 100, size=HOURS_BACK)
so2 = np.random.uniform(5, 50, size=HOURS_BACK)
co = np.random.uniform(0.2, 2.0, size=HOURS_BACK)
o3 = np.random.uniform(10, 80, size=HOURS_BACK)
temperature = np.random.uniform(15, 40, size=HOURS_BACK)  # will convert to int
humidity = np.random.uniform(30, 90, size=HOURS_BACK)     # will convert to int
pressure = np.random.uniform(950, 1050, size=HOURS_BACK)  # will convert to int
wind_speed = np.random.uniform(0, 15, size=HOURS_BACK)

# 3️⃣ Build DataFrame
df = pd.DataFrame({
    "city": CITY,
    "timestamp": pd.to_datetime(timestamps),
    "aqi": aqi,
    "pm25": pm25,
    "pm10": pm10,
    "no2": no2,
    "so2": so2,
    "co": co,
    "o3": o3,
    "temperature": temperature,
    "humidity": humidity,
    "pressure": pressure,
    "wind_speed": wind_speed
})

# 4️⃣ Time-based features
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# 5️⃣ Derived features
df["aqi_change"] = df["aqi"].diff().fillna(0)
df["pm25_change_rate"] = df["pm25"].diff().fillna(0)
df["rolling_avg_aqi_24h"] = df["aqi"].rolling(24, min_periods=1).mean()
df["rolling_avg_pm25_24h"] = df["pm25"].rolling(24, min_periods=1).mean()

# 6️⃣ Sort & deduplicate
df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

# 7️⃣ FORCE TYPES TO MATCH HOPSWORKS
df["pm10"] = df["pm10"].astype(np.float64)
df["temperature"] = df["temperature"].round(0).astype(np.int64)
df["humidity"] = df["humidity"].round(0).astype(np.int64)
df["pressure"] = df["pressure"].round(0).astype(np.int64)
df["hour"] = df["hour"].astype(np.int64)
df["day"] = df["day"].astype(np.int64)
df["month"] = df["month"].astype(np.int64)
df["day_of_week"] = df["day_of_week"].astype(np.int64)

# 8️⃣ Connect to Hopsworks
project = hopsworks.login(
    project="ayeshaahmedAQI",
    api_key_value="m0Gtiak8ESFhLCN4.GH2AXrrUWpmj7kygOmdLNXBwOVRG5YLVjvVxRT3mz5VF5DrkCqV0CQKrZ7az5UBS"
)
fs = project.get_feature_store()

# 9️⃣ Get feature group
fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

# 🔟 Insert data into Feature Group
try:
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"✅ Successfully backfilled {len(df)} rows for 1 month")
except Exception as e:
    print("❌ ERROR during backfill:")
    print(e)

