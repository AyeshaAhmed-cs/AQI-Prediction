# ===================== IMPORTS =====================
import hopsworks
import pandas as pd
import numpy as np
import joblib
import shutil
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ===================== LOGIN =====================
project = hopsworks.login(
    project="ayeshaahmedAQI",
    api_key_value="m0Gtiak8ESFhLCN4.GH2AXrrUWpmj7kygOmdLNXBwOVRG5YLVjvVxRT3mz5VF5DrkCqV0CQKrZ7az5UBS"
)
fs = project.get_feature_store()

# ===================== FETCH DATA FROM FEATURE VIEW =====================
fv = fs.get_feature_view(name="aqi_training_view", version=1)

df = fv.query.read()   # ✅ THIS IS THE CORRECT METHOD FOR YOU
print(f"✅ Feature View fetched: {df.shape[0]} rows, {df.shape[1]} columns")


print(f"✅ Training dataset fetched: {df.shape[0]} rows, {df.shape[1]} columns")

# ===================== CLEAN DATA =====================
df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
if non_numeric_cols:
    print(f"⚠ Dropping non-numeric columns: {non_numeric_cols}")
    df = df.drop(columns=non_numeric_cols)

# ===================== SPLIT FEATURES / TARGET =====================
TARGET = "aqi"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ===================== TRAIN / TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== IMPUTER =====================
imputer = SimpleImputer(strategy="median")

# ===================== MODELS =====================
models = {
    "RandomForest": Pipeline([
        ("imputer", imputer),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]),
    "Ridge": Pipeline([
        ("imputer", imputer),
        ("model", Ridge(alpha=1.0))
    ])
}

results = {}

# ===================== TRAIN SKLEARN MODELS =====================
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    results[name] = {
        "model": pipe,
        "RMSE": mean_squared_error(y_test, preds) ** 0.5,
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

# ===================== ANN MODEL =====================
X_train_nn = imputer.fit_transform(X_train)
X_test_nn = imputer.transform(X_test)

ann = Sequential([
    Input(shape=(X_train_nn.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])

ann.compile(optimizer="adam", loss="mse", metrics=["mae"])
ann.fit(X_train_nn, y_train, epochs=40, batch_size=32, verbose=0)

ann_preds = ann.predict(X_test_nn).flatten()

results["ANN"] = {
    "model": ann,
    "RMSE": mean_squared_error(y_test, ann_preds) ** 0.5,
    "MAE": mean_absolute_error(y_test, ann_preds),
    "R2": r2_score(y_test, ann_preds)
}

# ===================== MODEL SELECTION =====================
def score(m):
    return m["RMSE"] + m["MAE"] - m["R2"]

best_model_name = min(results, key=lambda k: score(results[k]))
best_model = results[best_model_name]["model"]

print(f"🏆 Best model: {best_model_name}")
print("Metrics:", results[best_model_name])

# ===================== SAVE MODEL LOCALLY =====================
os.makedirs("model_artifact", exist_ok=True)

if best_model_name == "ANN":
    best_model.save("model_artifact/model.h5")
else:
    joblib.dump(best_model, "model_artifact/model.pkl")

# ===================== ZIP MODEL (UPLOAD FIX) =====================
shutil.make_archive("aqi_model", "zip", "model_artifact")
print("✅ Model zipped for reliable upload")

# ===================== REGISTER MODEL =====================
mr = project.get_model_registry()

model = mr.python.create_model(
    name=f"{best_model_name}_AQI_predictor",
    metrics={
        "RMSE": float(results[best_model_name]["RMSE"]),
        "MAE": float(results[best_model_name]["MAE"]),
        "R2": float(results[best_model_name]["R2"])
    },
    description="AQI prediction model trained using Feature View data"
)

# ===================== UPLOAD ZIPPED MODEL =====================
model.save("aqi_model.zip")

print("🔥 Training pipeline completed & model registered successfully!")
