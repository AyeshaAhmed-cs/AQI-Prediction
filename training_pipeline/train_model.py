# ================= training_pipeline/train_model.py =================
import os
import hopsworks
import pandas as pd
import numpy as np
import joblib
import shutil
import time
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# 1. SETUP & LOGIN (Only once!)
load_dotenv()
project = hopsworks.login(project="ayeshaahmedAQI")
fs = project.get_feature_store()

# 2. GET OR CREATE FEATURE VIEW
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 2

fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

try:
    fv = fs.get_or_create_feature_view(
        name="aqi_training_view",
        version=1,
        description="AQI features ready for ML",
        query=fg
    )
    print("✅ Feature View ready!")
except Exception as e:
    print(f"Checking existing Feature View...")
    fv = fs.get_feature_view(name="aqi_training_view", version=1)

# 3. READ DATA (With Retry logic for Error 255)
print("Reading data from Hopsworks...")
try:
    df = fv.query.read()
except Exception as e:
    print(f"Server busy (Error 255), waiting 20 seconds...")
    time.sleep(20)
    df = fv.query.read()

# 4. CLEAN DATA
df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
# Drop non-numeric for training
training_df = df.select_dtypes(include=[np.number])
if "aqi" not in training_df.columns:
    # If aqi was dropped because it wasn't numeric, we have a problem
    print("Error: Target column 'aqi' is missing or non-numeric!")
    exit()

X = training_df.drop(columns=["aqi"])
y = training_df["aqi"]

# 5. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy="median")

# 6. DEFINE & TRAIN MODELS
models = {
    "RandomForest": Pipeline([
        ("imputer", imputer),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    "Ridge": Pipeline([
        ("imputer", imputer),
        ("model", Ridge(alpha=1.0))
    ])
}

results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    results[name] = {
        "model": pipe,
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

# 7. MODEL SELECTION
best_model_name = min(results, key=lambda k: results[k]["RMSE"])
best_model = results[best_model_name]["model"]
print(f"🏆 Best model: {best_model_name} with RMSE: {results[best_model_name]['RMSE']:.2f}")

# 8. SAVE & REGISTER
os.makedirs("model_artifact", exist_ok=True)
joblib.dump(best_model, "model_artifact/model.pkl")
shutil.make_archive("aqi_model", "zip", "model_artifact")

# Filter metrics to only include the numbers, not the model object
final_metrics = {
    "RMSE": float(results[best_model_name]["RMSE"]),
    "MAE": float(results[best_model_name]["MAE"]),
    "R2": float(results[best_model_name]["R2"])
}

print(f"Uploading model with metrics: {final_metrics}")

mr = project.get_model_registry()
model_meta = mr.python.create_model(
    name="karachi_aqi_model",
    metrics=final_metrics,  # Now sending only the numbers!
    description="Random Forest Regressor for Karachi AQI"
)

# Upload the zip file
model_meta.save("aqi_model.zip")
print("🔥 SUCCESS: Training pipeline completed & model registered in Hopsworks!")
