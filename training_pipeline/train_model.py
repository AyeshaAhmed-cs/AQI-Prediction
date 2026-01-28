import os
import hopsworks
import pandas as pd
import numpy as np
import joblib
import shutil
import time
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# TensorFlow for ANN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- 1. SETUP & LOGIN ---
load_dotenv()
project = hopsworks.login(project="ayeshaahmedAQI")
fs = project.get_feature_store()

# --- 2. FETCH DATA ---
print("📡 Fetching data from Feature Store...")
fg = fs.get_feature_group(name="aqi_features", version=2)
df = fg.read(online=False)

# --- 3. DATA PREPARATION (The NaN-Proof Version) ---
# Drop non-numeric and ensure aqi exists
training_df = df.select_dtypes(include=[np.number]).dropna(subset=["aqi"])

X = training_df.drop(columns=["aqi"])
y = training_df["aqi"]

# 🛠️ CRITICAL FIX: Fill missing values in Features (X) BEFORE scaling
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Scaling for Ridge and ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. EXPERIMENT WITH VARIOUS ML MODELS ---

# A. Ridge Regression
print("📈 Training Ridge Regression...")
ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)

# B. Ensemble Models
print("🌲 Training Ensemble Models...")
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
gb = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)

# C. Artificial Neural Network (ANN)
print("🧠 Training ANN (TensorFlow)...")
def build_ann(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1) 
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

ann = build_ann(X_train.shape[1])
ann.fit(X_train_scaled, y_train, epochs=50, batch_size=8, verbose=0)

# --- 5. EVALUATE PERFORMANCE ---
models = {
    "Ridge": (ridge.predict(X_test_scaled), ridge),
    "RandomForest": (rf.predict(X_test), rf),
    "GradientBoosting": (gb.predict(X_test), gb),
    "ANN": (ann.predict(X_test_scaled).flatten(), ann)
}

results = {}
for name, (preds, model_obj) in models.items():
    results[name] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "object": model_obj
    }
    print(f"📊 {name} -> RMSE: {results[name]['RMSE']:.2f}, R2: {results[name]['R2']:.2f}")

# --- 6. STORE TRAINED MODEL ---
best_name = min(results, key=lambda k: results[k]["RMSE"])
print(f"🏆 Best Performing Model: {best_name}")

artifact_path = "model_artifact"
try:
    if os.path.exists(artifact_path):
        for filename in os.listdir(artifact_path):
            file_path = os.path.join(artifact_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"⚠️ Could not delete {file_path}: {e}")
    else:
        os.makedirs(artifact_path, exist_ok=True)
except Exception:
    pass 

# Save the Best Model
if best_name == "ANN":
    results[best_name]["object"].save(os.path.join(artifact_path, "saved_model"))
else:
    joblib.dump(results[best_name]["object"], os.path.join(artifact_path, "model.pkl"))

# Save Preprocessing Tools (Vital for app.py!)
joblib.dump(scaler, os.path.join(artifact_path, "scaler.pkl"))
joblib.dump(imputer, os.path.join(artifact_path, "imputer.pkl"))

# Zip the artifacts
if os.path.exists("aqi_model.zip"):
    os.remove("aqi_model.zip")
shutil.make_archive("aqi_model", "zip", artifact_path)

# Register with Hopsworks
mr = project.get_model_registry()
model_meta = mr.python.create_model(
    name="karachi_aqi_model",
    metrics={
        "RMSE": float(results[best_name]["RMSE"]), 
        "R2": float(results[best_name]["R2"]),
        "MAE": float(results[best_name]["MAE"]) 
    },
    description=f"Winner: {best_name}. MAE included."
)
model_meta.save("aqi_model.zip")
print(f"🔥 SUCCESS: {best_name} registered with MAE, RMSE, and R2!")