# 🌍 Karachi AQI Intelligence System
### **Comparative AI Forecasting: Statistical vs. Deep Learning Approaches**

**Developed by:** ✨ AYESHA AHMED ✨  
**Status:** 🟢 System Online | **Cycle:** Hourly Updates  

---

## 📌 Project Overview
This repository hosts a production-grade machine learning pipeline designed to predict the Air Quality Index (AQI) for Karachi, Pakistan. The system leverages automated data engineering via Hopsworks and an interactive Streamlit dashboard to provide actionable environmental insights for the general public.

---

## 🔬 Advanced Analytics Features
This system transcends basic prediction by implementing a robust data science framework:

* **Exploratory Data Analysis (EDA):** Interactive multivariate correlation studies analyzing the interplay between humidity, atmospheric pressure, temperature, and AQI levels.
* **Model Explainability:** Integrated feature importance tracking to explain AI decision-making (e.g., how wind speed affects particulate dispersal).
* **Public Health Alerts:** An automated hazard detection system that triggers critical warnings for AQI spikes exceeding the 150 (Unhealthy) threshold.
* **Multi-Model Support:** Architecture designed to support both Statistical Regressors and Deep Learning models via a decoupled inference pipeline.

---

## 🛠️ System Architecture


The project is divided into three primary automated pipelines:

### 1. 🏗️ Feature Pipeline (`feature_pipeline.py`)
* **Source:** Real-time data ingestion from Open-Meteo & Air Quality APIs.
* **Logic:** Fetches and prepares atmospheric data hourly.
* **Feature Store:** Synchronizes processed features to **Hopsworks Feature Store** for model consumption.

### 2. 🧠 Training Pipeline (`training_pipeline.py`)
* **Training:** Automatically retrains ML models on a daily schedule.
* **Version Control:** Logs model artifacts (Pickle files) to the **Hopsworks Model Registry**.
* **Optimization:** Implements feature scaling and missing value imputation for high-accuracy forecasting.

### 3. 📈 Inference & Dashboard (`app.py`)
* **Real-time Prediction:** Pulls the latest model from the registry to forecast the next 72 hours.
* **UX/UI:** A premium Streamlit dashboard featuring:
    * **Health Advisory:** Plain-English recommendations for "normal people."
    * **Expert EDA:** Heatmaps, Boxplots, and Scatter plots for multivariate statistical analysis.
    * **Hazard Monitor:** A dedicated log for identifying dangerous pollution windows.

---

# 🚀 Installation & Setup
### 1. Clone the repository
git clone https://github.com/YourUsername/AQI-Prediction.git
cd AQI-Prediction


### 2. Install dependencies
pip install -r requirements.txt


### 3. Configure environment
#### Create a .env file with your Hopsworks API key
##### HOPSWORKS_API_KEY="your_api_key_here"

The system prioritizes **Random Forest Regression** due to its ability to capture non-linear relationships between weather variables and pollution.
Benchmark models include:
- Ridge Regression
- Artificial Neural Networks (ANNs)


> “Providing data-driven clarity for a cleaner, safer Karachi.”
