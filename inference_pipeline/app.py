import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
import requests
import glob
import zipfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG & STYLING ---
CITY = "Karachi"
PRIMARY_COLOR = "#E63946"  # Professional Red
SECONDARY_COLOR = "#1D3557" # Deep Navy

st.set_page_config(page_title="Karachi AQI AI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a "Premium" Look
st.markdown(f"""
    <style>
    .main {{ background-color: #f8f9fa; }}
    .stMetric {{ background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid {PRIMARY_COLOR}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
    .stTabs [data-baseweb="tab"] {{ height: 50px; white-space: pre-wrap; background-color: #f1f3f5; border-radius: 10px 10px 0px 0px; gap: 1px; padding: 10px; }}
    .stTabs [aria-selected="true"] {{ background-color: {PRIMARY_COLOR} !important; color: white !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    project = hopsworks.login()
    mr = project.get_model_registry()
    model_meta = mr.get_model("karachi_aqi_model", version=1)
    model_dir = model_meta.download() 
    
    zip_files = glob.glob(os.path.join(model_dir, "*.zip"))
    for z in zip_files:
        with zipfile.ZipFile(z, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
            
    pkl_files = glob.glob(os.path.join(model_dir, "**/model.pkl"), recursive=True)
    model = joblib.load(pkl_files[0])
    
    scaler_files = glob.glob(os.path.join(model_dir, "**/scaler.pkl"), recursive=True)
    imputer_files = glob.glob(os.path.join(model_dir, "**/imputer.pkl"), recursive=True)
    
    scaler = joblib.load(scaler_files[0]) if scaler_files else None
    imputer = joblib.load(imputer_files[0]) if imputer_files else None
    
    return model, scaler, imputer

# --- 2. DATA FETCHING ---
def get_forecast_with_features():
    url = "https://api.open-meteo.com/v1/forecast?latitude=24.8607&longitude=67.0011&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl&forecast_days=3"
    response = requests.get(url).json()
    hourly = response['hourly']
    
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly['time']),
        "temperature": hourly['temperature_2m'],
        "humidity": hourly['relative_humidity_2m'],
        "wind_speed": hourly['wind_speed_10m'],
        "pressure": hourly['pressure_msl']
    })
    
    # Feature engineering (Keeping your logic exactly as is)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    for col in ["aqi_change", "pm25_change_rate"]: df[col] = 0.0
    df["rolling_avg_aqi_24h"] = 125.0 
    df["rolling_avg_pm25_24h"] = 65.0 
    for col in ["pm25", "pm10", "no2", "so2", "co", "o3", "month", "day"]: df[col] = 50.0
    
    return df

# --- 3. SIDEBAR (Presentable Time & Info) ---
with st.sidebar:
    st.markdown(f"<h1 style='color: {PRIMARY_COLOR};'>🛡️ AQI Monitor</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("📍 Station Details")
    st.write(f"**City:** {CITY}, Pakistan")
    st.write(f"**Latitude:** 24.86° N")
    st.write(f"**Longitude:** 67.00° E")
    st.markdown("---")
    st.subheader("🕒 System Status")
    st.write(f"**Last Sync:** {datetime.now().strftime('%H:%M:%S')}")
    st.write(f"**Date:** {datetime.now().strftime('%d %b, %Y')}")
    st.success("System: Online")

# --- 4. MAIN DASHBOARD ---
st.markdown(f"<h1 style='text-align: center;'>🌍 Karachi Air Quality <span style='color:{PRIMARY_COLOR};'>Intelligence</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Machine Learning powered 72-hour pollution forecasting</p>", unsafe_allow_html=True)

try:
    model, scaler, imputer = load_assets()
    forecast_df = get_forecast_with_features()
    
    # Prediction Logic
    X = forecast_df.select_dtypes(include=[np.number])
    if hasattr(model, 'feature_names_in_'): X = X[model.feature_names_in_]
    if imputer: X = imputer.transform(X)
    if scaler: X = scaler.transform(X)
    preds = model.predict(X).flatten()
    forecast_df['predicted_aqi'] = preds

    # --- TOP METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Predicted AQI", f"{preds.mean():.0f}")
    m2.metric("Peak Level", f"{preds.max():.0f}", delta="High" if preds.max() > 150 else "Safe")
    m3.metric("Current Temp", f"{forecast_df['temperature'].iloc[0]}°C")
    m4.metric("Humidity", f"{forecast_df['humidity'].iloc[0]}%")

    # --- PRESENTABLE LINE GRAPH ---
    st.markdown("### 📈 Pollution Trend Analysis")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df['timestamp'], y=forecast_df['predicted_aqi'],
        mode='lines', name='AQI Level',
        line=dict(color=PRIMARY_COLOR, width=4),
        fill='tozeroy', fillcolor='rgba(230, 57, 70, 0.1)'
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        xaxis_title="Time Horizon",
        yaxis_title="AQI Score",
        plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- TABS FOR ORGANIZED DATA ---
    st.markdown("### 🔬 In-Depth Insights")
    tab1, tab2, tab3 = st.tabs(["💡 Health Advisory", "📊 Environmental EDA", "⚠️ Hazard Alerts"])

    with tab1:
        st.subheader("👨‍⚕️ AI-Generated Health Recommendations")
        avg_aqi = preds.mean()
        
        # Enhanced description logic
        if avg_aqi > 150:
            st.error("### 🔴 Critical Level: Unhealthy Air")
            st.write(f"The average predicted AQI is **{avg_aqi:.1f}**. This is a significant health concern for the citizens of Karachi.")
            st.markdown("""
            **Recommended Actions:**
            * **Mask Mandate:** Wear an N95 mask if you must go outdoors.
            * **Household:** Keep windows closed to prevent dust and PM2.5 from entering.
            * **Exercise:** Avoid all outdoor physical activities; move workouts indoors.
            * **Air Purifiers:** Run air purifiers on high settings if available.
            """)
        elif avg_aqi > 100:
            st.warning("### 🟠 Warning: Moderate Risk")
            st.write(f"The average predicted AQI is **{avg_aqi:.1f}**. Some individuals may experience health effects.")
            st.markdown("""
            **Recommended Actions:**
            * **Sensitive Groups:** Children, elderly, and those with asthma should limit outdoor time.
            * **Ventilation:** Best to keep windows closed during peak traffic hours (8 AM - 10 AM).
            * **Hydration:** Stay well-hydrated to help your body clear inhaled particulates.
            """)
        else:
            st.success("### 🟢 Clean Air Notice: Low Risk")
            st.write(f"The average predicted AQI is **{avg_aqi:.1f}**. Air quality is considered satisfactory.")
            st.markdown("""
            **Recommended Actions:**
            * **Outdoor Activities:** A great time for park visits or outdoor sports at Sea View.
            * **Ventilation:** Open windows to allow fresh air circulation in your home.
            """)
        
    with tab2:
        st.subheader("Weather Correlation Study")
        var = st.selectbox("Select Variable:", ["temperature", "humidity", "wind_speed"])
        fig_eda = px.scatter(forecast_df, x=var, y="predicted_aqi", color="predicted_aqi", 
                             color_continuous_scale="RdBu_r", title=f"Impact of {var.title()} on AQI")
        st.plotly_chart(fig_eda, use_container_width=True)

    with tab3:
        st.subheader("🚨 Hazardous Level Monitor (72h Log)")
        danger_zones = forecast_df[forecast_df['predicted_aqi'] > 150]
        
        if not danger_zones.empty:
            st.error(f"### ⚠️ {len(danger_zones)} Hazardous Spikes Detected")
            st.write("Our AI model has identified specific time windows where pollution levels exceed safe limits. Avoid being outdoors during these exact times:")
            
            # Formatting the table for better readability
            display_df = danger_zones[['timestamp', 'predicted_aqi']].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%b %d, %H:%M')
            display_df.columns = ['Danger Window', 'Predicted AQI Value']
            st.table(display_df.head(10))
            
            st.markdown("> **Note:** These spikes are often caused by stagnant wind speeds and high humidity trapping vehicular emissions.")
        else:
            st.info("### ✅ No Hazardous Spikes Predicted")
            st.write("Excellent news! The machine learning model predicts that for the next 72 hours, Karachi's air will remain below the hazardous threshold of 150 AQI.")
            st.write("This stability is likely due to sufficient **wind speed** (the 'Karachi Sea Breeze') preventing the accumulation of pollutants at ground level.")

except Exception as e:
    st.error(f"⚠️ UI Loading Error: {e}")