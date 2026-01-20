import pandas as pd
import os
from fetch_aqi_hourly import fetch_aqi

CSV_PATH = "aqi_features.csv"

# Define exact column order
COLUMNS = [
    "city", "timestamp", "aqi", "pm25", "pm10", "no2", "so2",
    "co", "o3", "temperature", "humidity", "pressure", "wind_speed",
    "hour", "day", "month", "day_of_week", "is_weekend",
    "aqi_change", "pm25_change_rate", "rolling_avg_aqi_24h", "rolling_avg_pm25_24h"
]

def generate_features():
    row = fetch_aqi()
    if row is None:
        print("❌ Failed to fetch AQI")
        return None

    df = pd.DataFrame([row])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Initialize derived features
    df["aqi_change"] = 0.0
    df["pm25_change_rate"] = 0.0
    df["rolling_avg_aqi_24h"] = df["aqi"]
    df["rolling_avg_pm25_24h"] = df["pm25"]

    # Only compute changes if CSV exists and has at least 1 row
    if os.path.exists(CSV_PATH):
        try:
            prev = pd.read_csv(CSV_PATH)

            # Ensure previous CSV has correct columns
            for col in ["aqi", "pm25"]:
                if col not in prev.columns:
                    prev[col] = 0.0

            df["aqi_change"] = df["aqi"] - prev.iloc[-1]["aqi"]
            df["pm25_change_rate"] = df["pm25"] - prev.iloc[-1]["pm25"]

            combined = pd.concat([prev, df], ignore_index=True)

            # Rolling averages
            df["rolling_avg_aqi_24h"] = combined["aqi"].rolling(24, min_periods=1).mean().iloc[-1]
            df["rolling_avg_pm25_24h"] = combined["pm25"].rolling(24, min_periods=1).mean().iloc[-1]
        except Exception as e:
            print(f"⚠ Warning: Could not read previous CSV: {e}")

    # Ensure all columns exist in final CSV
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = 0.0  # default value for missing columns

    # Reorder columns
    df = df[COLUMNS]

    return df


if __name__ == "__main__":
    features = generate_features()
    if features is not None:
        # Save CSV safely
        if os.path.exists(CSV_PATH):
            features.to_csv(CSV_PATH, mode="a", header=False, index=False)
        else:
            features.to_csv(CSV_PATH, index=False)
        print("✅ Features saved successfully")



