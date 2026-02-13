import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hopsworks
import joblib
import requests
import os
import zipfile 
from datetime import datetime, timedelta
from scipy.interpolate import make_interp_spline
import time
from dotenv import load_dotenv

# --- 1. STYLING & CONFIG ---
st.set_page_config(page_title="Karachi AI AQI Dashboard", layout="wide")
load_dotenv()

st.markdown("""
    <style>
    /* 1. Global Dark Theme */
    .stApp, [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #0E1117 !important;
    }

    /* 2. Sidebar Visibility Fixes (FORCED VISIBILITY) */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }

    /* MAKE BUTTON ALWAYS VISIBLE */
    .stButton>button {
        background-color: rgba(0, 242, 255, 0.15) !important;
        color: #00F2FF !important;
        border: 1px solid #00F2FF !important;
        width: 100%;
        font-weight: bold;
    }

    /* MAKE EXPANDER HEADER ALWAYS VISIBLE */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(0, 242, 255, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderHeader p {
        color: #00F2FF !important;
        font-weight: bold !important;
    }

    /* 3. Centered Main Title */
    .main-title {
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        color: #00F2FF !important;
        padding: 10px 0;
        letter-spacing: 2px;
        font-weight: bold;
        text-shadow: 0px 0px 15px rgba(0, 242, 255, 0.4);
    }

    /* 4. Centered Digital Clock */
    .digital-clock {
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        color: #00F2FF;
        font-size: 1.1rem;
        margin-bottom: 25px;
        text-shadow: 0px 0px 10px rgba(0, 242, 255, 0.3);
    }

    /* 5. Headings Visibility */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
    }

    h3 {
        border-left: 4px solid #00F2FF;
        padding-left: 15px;
        margin-top: 30px !important;
    }
    
    /* 6. Metrics Styling */
    .metric-row { display: flex; justify-content: center; gap: 12px; margin-bottom: 30px; flex-wrap: wrap; }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 242, 255, 0.2);
        border-radius: 12px;
        padding: 15px; min-width: 110px; text-align: center;
    }
    .m-label { color: #888; font-size: 0.7rem; text-transform: uppercase; }
    .m-value { color: #00F2FF; font-size: 1.1rem; font-family: 'Orbitron'; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 2. THE CLOCK COMPONENT (CENTERED) ---
@st.fragment(run_every="1s")
def sync_clock():
    # Force Karachi Time (UTC + 5 hours)
    karachi_now = datetime.now() + timedelta(hours=5)
    now_str = karachi_now.strftime("%A, %b %d, %Y | %I:%M:%S %p")
    st.markdown(f'<div class="digital-clock">🕒 {now_str} (PKT)</div>', unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>🏙️ KARACHI AIR QUALITY INDEX</h1>", unsafe_allow_html=True)
sync_clock()

# --- 3. LOADING ---
@st.cache_data(ttl=300) # Use cache_data for the actual dataframe
def fetch_latest_logs(_project):
    fs = _project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=2)
    # Force a fresh read by sorting by timestamp descending
    hist_df = fg.read(read_options={"use_hive": True}).sort_values('timestamp').tail(100)
    return hist_df

@st.cache_resource(ttl=60)
def init_hopsworks():
    project = hopsworks.login(project="ayeshaahmedAQI")
    mr = project.get_model_registry()
    
    # 1. Fetch ALL available versions
    all_models = mr.get_models("karachi_aqi_model")
    
    # 2. FILTER: Ignore models with "No value" and handle case sensitivity
    valid_models = [
        m for m in all_models 
        if m.training_metrics and (m.training_metrics.get('R2') is not None or m.training_metrics.get('r2') is not None)
    ]
    
    if not valid_models:
        st.error("No models with valid training metrics (R2) found!")
        st.stop()

    # 3. Find BEST: Prioritizes highest R2 Score, then the highest Version number
    best_model_meta = max(
        valid_models, 
        key=lambda m: (float(m.training_metrics.get('R2', m.training_metrics.get('r2', 0))), m.version)
    )
    
    current_version = best_model_meta.version
    training_metrics = best_model_meta.training_metrics
    
    # 4. Download and load logic
    download_path = best_model_meta.download()
    for root, dirs, files in os.walk(download_path):
        for file in files:
            if file.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(root, file), 'r') as z:
                    z.extractall(root)

    model_file_path = next((os.path.join(r, "model.pkl") for r, d, f in os.walk(download_path) if "model.pkl" in f), None)
    
    if not model_file_path:
        st.error(f"Model file missing in v{current_version}")
        st.stop()

    loaded_model = joblib.load(model_file_path)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=2)
    # This fix uses the Hive connector to avoid the 'Binder Error' column mismatch
    hist_df = fg.read(read_options={"use_hive": True}).tail(100)
    return loaded_model, hist_df, current_version, training_metrics
try:
    model, historical_data, current_v, metrics = init_hopsworks()
except Exception as e:
    st.error(f"Dashboard failed to connect to Hopsworks: {e}")
    st.stop()

# --- 4. LIVE DATA FETCHING ---
def get_live_weather():
    try:
        w_url = "https://api.open-meteo.com/v1/forecast?latitude=24.8607&longitude=67.0011&hourly=temperature_2m,relativehumidity_2m,surface_pressure,windspeed_10m&daily=sunrise,sunset&current_weather=true&timezone=auto"
        res = requests.get(w_url).json()
      
        current_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:00")
        
        try:
            # Match the index so we don't accidentally grab "future" data (like 01:00 AM)
            h_idx = res['hourly']['time'].index(current_utc)
        except (ValueError, KeyError):
            h_idx = 0 

        curr = res['current_weather']
        
        # 3. Calculate Day/Night status
        now_time = datetime.now().strftime("%Y-%m-%dT%H:%M")
        sunrise = res['daily']['sunrise'][0]
        sunset = res['daily']['sunset'][0]
        is_daylight = sunrise <= now_time <= sunset

        return {
            'temp': curr['temperature'],
            'wind': curr['windspeed'],
            'hum': res['hourly']['relativehumidity_2m'][h_idx],
            'pres': res['hourly']['surface_pressure'][h_idx],
            'sunrise': datetime.fromisoformat(sunrise).strftime("%I:%M %p"),
            'sunset': datetime.fromisoformat(sunset).strftime("%I:%M %p"),
            'is_day': is_daylight,
            'forecast': res['hourly']
        }
    except Exception as e:
        print(f"Weather Fetch Error: {e}")
        return {
            'temp': 25, 'wind': 10, 'hum': 50, 'pres': 1013,
            'sunrise': '07:00 AM', 'sunset': '06:00 PM', 'is_day': True, 'forecast': {}
        }

weather_data = get_live_weather()

# --- 5. RECURSIVE REAL-TIME PREDICTION ENGINE ---
def predict_aqi_recursive(forecast_data, historical_data, model):
    future_preds = []
    
    # 1. Initialize "current_state" with the absolute latest REAL data from Hopsworks
    latest_real = historical_data.iloc[-1].copy()
    
    # We maintain a sliding window of AQI and PM2.5 to update rolling averages
    aqi_window = historical_data['aqi'].tail(24).tolist()
    pm25_window = historical_data['pm25'].tail(24).tolist()
    
    # Reference for SHAP (we'll capture the first iteration)
    df_for_shap = None
    feat_cols = ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", 
                 "wind_speed", "pressure", "hour", "day", "month", "day_of_week", 
                 "is_weekend", "aqi_change", "pm25_change_rate", "rolling_avg_aqi_24h", "rolling_avg_pm25_24h"]

    base_time = datetime.now() + timedelta(hours=5)

    for i in range(72):
        p_time = base_time + timedelta(hours=i)
        
        # 2. Dynamic Weather Features from Open-Meteo
        f_temp = forecast_data['temperature_2m'][i]
        f_hum = forecast_data['relativehumidity_2m'][i]
        f_wind = forecast_data['windspeed_10m'][i]
        f_pres = weather_data['pres'] # Constant for simplicity or pull from forecast if available

        # 3. Recursive Feature Update
        current_rolling_aqi = sum(aqi_window[-24:]) / 24
        current_rolling_pm25 = sum(pm25_window[-24:]) / 24

        input_data = {
            "pm25": float(pm25_window[-1]), 
            "pm10": float(latest_real['pm10']),
            "no2": float(latest_real['no2']),
            "so2": float(latest_real['so2']),
            "co": float(latest_real['co']),
            "o3": float(latest_real['o3']),
            "temperature": float(f_temp),
            "humidity": float(f_hum),
            "wind_speed": float(f_wind),
            "pressure": float(f_pres),
            "hour": int(p_time.hour),
            "day": int(p_time.day),
            "month": int(p_time.month),
            "day_of_week": int(p_time.weekday()),
            "is_weekend": int(1 if p_time.weekday() >= 5 else 0),
            "aqi_change": float(aqi_window[-1] - aqi_window[-2]),
            "pm25_change_rate": float(pm25_window[-1] - pm25_window[-2]),
            "rolling_avg_aqi_24h": float(current_rolling_aqi),
            "rolling_avg_pm25_24h": float(current_rolling_pm25)
        }
        
        # Create DataFrame in correct order
        input_df = pd.DataFrame([input_data])[feat_cols]
        
        # 4. Predict
        prediction = float(model.predict(input_df.values)[0])
        future_preds.append(prediction)
        
        # 5. FEEDBACK LOOP: Update the windows for the next hour
        aqi_window.append(prediction)
        pm25_window.append(pm25_window[-1]) 

        if i == 0:
            df_for_shap = input_df

    return future_preds, df_for_shap, feat_cols

# Execution
future_preds, df_for_shap, feats_for_shap = predict_aqi_recursive(weather_data['forecast'], historical_data, model)

# Card Assignments
live_prediction = future_preds[0] 
today_card_val = future_preds[12]      # 12h
tomorrow_card_val = future_preds[36]   # 36h
next_day_card_val = future_preds[71]   # 72h (Last index of the 72-hour forecast)

# --- 6. DASHBOARD UI ---

if live_prediction > 150:
    st.warning(f"🚨 HAZARDOUS LEVEL ALERT: Current AI Estimate is {live_prediction}. Avoid outdoor activities.")
elif live_prediction > 100:
    st.info(f"⚠️ MODERATE POLLUTION: Current AI Estimate is {live_prediction}. Sensitive groups should take care.")

# --- 7. TOP METRICS ROW (FULLY DYNAMIC) ---
val_mae = metrics.get('MAE', metrics.get('mae', 0.0))
val_rmse = metrics.get('RMSE', metrics.get('rmse', 0.0))
val_r2 = metrics.get('R2', metrics.get('r2', 0.0))

st.markdown(f"""
    <div class="metric-row">
        <div class="glass-card">
            <div class="m-label">TEMP</div>
            <div class="m-value">{weather_data.get('temp', 'N/A')}°C</div>
        </div>
        <div class="glass-card">
            <div class="m-label">HUMIDITY</div>
            <div class="m-value">{weather_data.get('hum', 'N/A')}%</div>
        </div>
        <div class="glass-card">
            <div class="m-label">PRESSURE</div>
            <div class="m-value">{weather_data.get('pres', 'N/A')} hPa</div>
        </div>
        <div class="glass-card">
            <div class="m-label">WIND</div>
            <div class="m-value">{weather_data.get('wind', 'N/A')} km/h</div>
        </div>
        <div class="glass-card">
            <div class="m-label">MAE</div>
            <div class="m-value">{float(val_mae):.4f}</div>
        </div>
        <div class="glass-card">
            <div class="m-label">RMSE</div>
            <div class="m-value">{float(val_rmse):.4f}</div>
        </div>
        <div class="glass-card">
            <div class="m-label">R² SCORE</div>
            <div class="m-value">{float(val_r2):.4f}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 8. OUTLOOK GRID (BEAUTIFIED) ---
st.markdown("### 📅 3-Day Forecast (AI Predictions)")
c1, c2, c3 = st.columns(3)

card_values = [today_card_val, tomorrow_card_val, next_day_card_val]
card_labels = ["Today (Avg)", "Tomorrow", "Day After"]

for col, val, label in zip([c1, c2, c3], card_values, card_labels):
    rounded_val = int(round(val))
    # Change background opacity to 0.08 and add a subtle box-shadow for a "glow"
    col.markdown(f"""
        <div style="
            text-align:center; 
            padding:25px; 
            border:1px solid rgba(0, 242, 255, 0.4); 
            border-radius:15px; 
            background: rgba(255, 255, 255, 0.08); 
            box-shadow: 0px 4px 15px rgba(0, 242, 255, 0.1);
        ">
            <p style="color:#BBB; margin-bottom:5px; font-weight: bold; letter-spacing: 1px;">{label}</p>
            <h1 style="color:#00F2FF; margin:0; font-size: 3rem; text-shadow: 0px 0px 10px rgba(0, 242, 255, 0.3);">{rounded_val}</h1>
        </div>
    """, unsafe_allow_html=True)

# --- 9. ANALYTICS ROW: GAUGE & TRAJECTORY ---
# 1. Create Columns for "Face-to-Face" Alignment
col_gauge, col_graph = st.columns([1, 2]) 

with col_gauge:
    st.markdown("### LIVE AQI LEVEL")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = live_prediction, 
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 300], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00F2FF"},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 50], 'color': '#00E400'},
                {'range': [51, 100], 'color': '#FFFF00'},
                {'range': [101, 150], 'color': '#FF7E00'},
                {'range': [151, 200], 'color': '#FF0000'},
                {'range': [201, 300], 'color': '#8F3F97'}
            ]
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Orbitron"}, height=400,
        margin=dict(l=30, r=30, t=50, b=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_graph:
    st.markdown("### 📈 72-HOUR AQI TRAJECTORY")
    
    # 1. Base time and raw X-axis
    base_time = datetime.now() + timedelta(hours=5)
    x_raw = np.arange(len(future_preds))
    # Raw dates for the scatter points
    raw_dates = [base_time + timedelta(hours=int(h)) for h in x_raw]
    
    # 2. Smooth curve logic (Spline)
    x_smooth = np.linspace(x_raw.min(), x_raw.max(), 300)
    spl = make_interp_spline(x_raw, future_preds, k=3)
    y_smooth = spl(x_smooth)
    smooth_dates = [base_time + timedelta(hours=float(h)) for h in x_smooth]

    fig_traj = px.line(
        x=smooth_dates, 
        y=y_smooth,
        labels={'x': 'Timeline', 'y': 'AQI Level'},
        template="plotly_dark"
    )

    fig_traj.add_scatter(
        x=raw_dates, 
        y=future_preds, 
        mode='markers',
        name='Hourly Prediction',
        marker=dict(color='#00F2FF', size=4, opacity=0.5),
        hovertemplate="Time: %{x}<br>AQI: %{y:.2f}<extra></extra>"
    )
    
    fig_traj.update_traces(line_color='#00F2FF', line_width=3, selector=dict(type='scatter', mode='lines'))
    fig_traj.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(showgrid=False, title="Next 3 Days (Karachi Time)"), 
        yaxis=dict(showgrid=True, gridcolor='#333', title="Predicted AQI")
    )
    st.plotly_chart(fig_traj, use_container_width=True)

# --- 10. ANALYSIS & HEALTH ---
tab1, tab2 = st.tabs(["🔥 Deep Analytics", "🩺 Health Advisory"])

with tab1:
    ch, ci = st.columns([1.2, 0.8])
    
    # --- DATA BRIDGE: Prepare the features for SHAP ---
    # We grab the latest row from historical data and prepare it exactly like your predict_aqi function
    latest_row = historical_data.iloc[-1:]
    feat_cols = ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", 
                 "wind_speed", "pressure", "hour", "day", "month", "day_of_week", 
                 "is_weekend", "aqi_change", "pm25_change_rate", "rolling_avg_aqi_24h", "rolling_avg_pm25_24h"]
    
    # 1. Clean data for correlation
    nums = historical_data.select_dtypes(include=[np.number])
    nums_filtered = nums.loc[:, nums.nunique() > 1] 
    
    with ch:
        st.markdown("### Feature Correlation")
        if not nums_filtered.empty:
            corr_matrix = nums_filtered.corr().dropna(how='all', axis=0).dropna(how='all', axis=1)
            fig_heat = px.imshow(
                corr_matrix, 
                text_auto=".2f", 
                color_continuous_scale='RdBu_r', 
                template="plotly_dark",
                aspect="auto"
            )
            fig_heat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Orbitron")
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("Not enough variable data for heatmap.")
        
    with ci:
        st.markdown("### 🧬 SHAP Explainability")
        try:
            # Prepare Model
            rf_model = model.steps[-1][1] if hasattr(model, 'steps') else model
            explainer = shap.TreeExplainer(rf_model)
            
            shap_values = explainer.shap_values(df_for_shap, check_additivity=False)
            
            sv = shap_values[0] if isinstance(shap_values, list) else shap_values

            # Create Plot
            fig_shap, ax = plt.subplots(figsize=(6, 4))
            shap.bar_plot(sv[0], 
                          features=df_for_shap.iloc[0], 
                          feature_names=feats_for_shap, 
                          show=False)
            
            # Styling
            fig_shap.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.xaxis.label.set_color('white')
            
            st.pyplot(fig_shap)
            st.caption("How features influenced the current AQI prediction.")

        except Exception as e: 
            st.error(f"SHAP Error: {e}")

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
 
    aqi_val = live_prediction 

    if aqi_val <= 50:
        status, hazard_msg, color = "GOOD", "OPTIMAL DISPERSION", "#00E400"
        clinical = """
            <li><b>Respiratory Impact:</b> Minimal to none; air is clear.</li>
            <li><b>Outdoor Activities:</b> Fully recommended, even intense exercise.</li>
            <li><b>Ventilation:</b> Open windows to refresh indoor air.</li>
            <li><b>Masking:</b> Not required for any groups.</li>
        """
        analysis = "Atmospheric conditions are scrubbed clean. PM2.5 concentrations are well below safety thresholds. Ideal for Karachi residents to be outdoors."
    
    elif aqi_val <= 100:
        status, hazard_msg, color = "MODERATE", "STABLE AIRFLOW", "#FFFF00"
        clinical = """
            <li><b>Sensitive Groups:</b> May experience slight irritation after long exposure.</li>
            <li><b>Outdoor Activities:</b> Safe for most; reduce heavy exertion if feeling tired.</li>
            <li><b>Pediatric Care:</b> Monitor children with existing asthma.</li>
            <li><b>Ventilation:</b> Generally safe for Karachi homes today.</li>
        """
        analysis = "Air quality is acceptable. A small number of individuals unusually sensitive to Karachi's dust or humidity might feel minor discomfort."
    
    elif aqi_val <= 150:
        status, hazard_msg, color = "SENSITIVE", "PARTICULATE ACCUMULATION", "#FF7E00"
        clinical = """
            <li><b>Primary Warning:</b> Vulnerable groups should limit outdoor time.</li>
            <li><b>Masking:</b> Standard masks recommended for long commutes.</li>
            <li><b>Home Care:</b> Keep windows closed during peak traffic hours.</li>
            <li><b>Symptoms:</b> Throat irritation possible for asthma patients.</li>
        """
        analysis = "Increasing particulates detected. Risk is higher for lung disease patients and children. System indicates a buildup of combustion pollutants."
    
    elif aqi_val <= 200:
        status, hazard_msg, color = "UNHEALTHY", "ATMOSPHERIC STAGNATION", "#FF0000"
        clinical = """
            <li><b>Everyone:</b> Wear N95/KN95 respirators for all outdoor tasks.</li>
            <li><b>Physical Exertion:</b> Avoid outdoor jogging or sports.</li>
            <li><b>Air Purifiers:</b> Activate filtration in indoor spaces.</li>
            <li><b>Vulnerable:</b> Stay indoors with filtered ventilation.</li>
        """
        analysis = "Stagnant air is trapping Karachi's industrial emissions at ground level. General public may feel health effects; sensitive groups face serious risk."
    
    elif aqi_val <= 300:
        status, hazard_msg, color = "VERY UNHEALTHY", "SEVERE STAGNATION", "#8F3F97"
        clinical = """
            <li><b>Health Alert:</b> Significant risk to the entire population.</li>
            <li><b>Restriction:</b> Outdoor parks and schools should be avoided.</li>
            <li><b>Masking:</b> N95 respirators are mandatory for safety.</li>
            <li><b>Medical:</b> Monitor for chest pain or breathing difficulty.</li>
        """
        analysis = "Near-emergency conditions. Thermal inversion is likely preventing pollutants from escaping. Acute respiratory responses are expected."
    
    else:
        status, hazard_msg, color = "HAZARDOUS", "TOXIC ENTRAPMENT", "#7E0023"
        clinical = """
            <li><b>Emergency:</b> Health warning of emergency conditions.</li>
            <li><b>Critical Action:</b> Stay indoors; use air filtration.</li>
            <li><b>Equipment:</b> Use professional-grade respirators if transit is vital.</li>
            <li><b>Alert:</b> Serious risk of respiratory collapse.</li>
        """
        analysis = "Life-threatening toxicity level. Karachi's atmosphere is completely locked, creating a dangerous trap for residents at ground level."

    # --- UI LAYOUT ---
    h_col1, h_col2 = st.columns(2)
    
    with h_col1:
        st.markdown(f"""
            <div style="border: 1px solid {color}; padding: 25px; border-radius: 15px; height: 380px; background: {color}05;">
                <h3 style="color: {color}; font-family: 'Orbitron'; font-size:1.1rem;">📋 CLINICAL STATUS</h3>
                <p style="font-size: 1rem; color:#888;">Condition: <b style="color:{color}; font-family:'Orbitron'; font-size:1.3rem;">{status}</b></p>
                <hr style="border: 0.1px solid {color}30; margin: 15px 0;">
                <ul style="color: #DDD; line-height: 1.8; font-size:0.95rem;">{clinical}</ul>
            </div>
        """, unsafe_allow_html=True)

    with h_col2:
        st.markdown(f"""
            <div style="border: 1px solid {color}; padding: 25px; border-radius: 15px; height: 380px; background: {color}03;">
                <h3 style="color: {color}; font-family: 'Orbitron'; font-size:1.1rem;">⚠️ HAZARD ANALYSIS</h3>
                <p style="font-size: 1rem; color:#888;">System: <b style="color:#00F2FF; font-family:'Orbitron'; font-size:1.3rem;">{hazard_msg}</b></p>
                <hr style="border: 0.1px solid rgba(0, 242, 255, 0.2); margin: 15px 0;">
                <p style="color: #BBB; font-size: 0.95rem; line-height: 1.6;">{analysis}</p>
                <div style="margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.05); padding-top:10px;">
                    <p style="font-size: 0.75rem; color: #555; margin:0;">Station: Karachi US-Consulate</p>
                    <p style="font-size: 0.75rem; color: #555; margin:0;">AI Core: Model v{current_v} Sync</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- 11. ENHANCED SIDEBAR UI ---
st.sidebar.markdown(f"<h2 style='text-align: center; color: #00F2FF; font-size: 1.5rem;'>🛰️ STATION STATUS</h2>", unsafe_allow_html=True)

# 1. Action Buttons (Clear Cache)
if st.sidebar.button("🔄 Clear Cache & Sync"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# --- 2. Sunrise/Sunset with dynamic styling ---

karachi_hour = (datetime.utcnow() + timedelta(hours=5)).hour

is_actually_day = 6 <= karachi_hour < 18 

sun_icon = "☀️" if is_actually_day else "🌙"
day_status = "DAYTIME" if is_actually_day else "NIGHTTIME"
status_color = "#00F2FF" if is_actually_day else "#8F3F97"
glow_shadow = "0 0 10px #00F2FF" if is_actually_day else "0 0 10px #8F3F97"

st.sidebar.markdown(f"""
    <div style="padding:20px; border-radius:15px; background: rgba(255,255,255,0.03); 
                border: 1px solid {status_color}40; box-shadow: {glow_shadow}30; margin-bottom: 20px;">
        <h4 style="margin:0; color:{status_color}; letter-spacing: 2px; text-shadow: {glow_shadow};">{sun_icon} {day_status}</h4>
        <p style="margin:10px 0 0 0; font-size:0.9rem; color: #BBB;">🌅 Sunrise: <span style="color:white;">{weather_data['sunrise']}</span></p>
        <p style="margin:0; font-size:0.9rem; color: #BBB;">🌇 Sunset: <span style="color:white;">{weather_data['sunset']}</span></p>
    </div>
""", unsafe_allow_html=True)

# 3. Interactive AQI Scale
st.sidebar.markdown("### 📊 AQI REFERENCE")
pointer_pos = min(max((live_prediction / 300) * 100, 0), 100)

st.sidebar.markdown(f"""
    <div style="width:100%; height:12px; border-radius:6px; 
        background: linear-gradient(to right, #00E400, #FFFF00, #FF7E00, #FF0000, #8F3F97, #7E0023);
        position: relative; margin: 25px 0 10px 0;">
        <div style="position: absolute; left: {pointer_pos}%; top: -8px; width: 4px; height: 28px; 
                    background: white; border: 1px solid black; box-shadow: 0 0 15px #FFF; border-radius:2px;">
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #888; font-weight: bold;">
        <span>0</span><span>100</span><span>200</span><span>300+</span>
    </div>
""", unsafe_allow_html=True)

# 4. Model Performance Expander
with st.sidebar.expander("🛠️ Model Performance Details"):
    r2_val = metrics.get('r2', metrics.get('R2', 0.0))
    mae_val = metrics.get('mae', metrics.get('MAE', 0.0))
    
    st.markdown(f"""
        <p style='color:white; margin:0;'><b>Selected Version:</b> v{current_v}</p>
        <p style='color:white; margin:0;'><b>R² Score:</b> {float(r2_val):.4f}</p>
        <p style='color:white; margin:0;'><b>MAE:</b> {float(mae_val):.4f}</p>
        <p style='color:gray; font-size:0.8rem; margin-top:5px;'>Auto-selected based on peak accuracy.</p>
    """, unsafe_allow_html=True)

# 5. System Automation & Sync Status
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 SYSTEM AUTOMATION")

if not historical_data.empty:
    # Treat Hopsworks timestamp as UTC
    utc_time = pd.to_datetime(historical_data['timestamp'].iloc[-1])
    # Add 5 hours for Karachi Time (PKT)
    local_time = (utc_time + pd.Timedelta(hours=5)).strftime("%I:%M %p")
    
    st.sidebar.success(f"✅ Data Freshness: {local_time} (PKT)")
else:
    st.sidebar.warning("⚠️ Waiting for pipeline sync...")

# Dashboard metadata
st.sidebar.caption(f"Last UI Sync: {datetime.now().strftime('%I:%M %p')}")