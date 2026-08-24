"""
 Dashboard
A Streamlit web app that shows the current AQI for Rawalpindi, a 3-day-ahead
forecast using the trained Ridge model, a hazardous AQI alert, and a recent
trend chart.

Data flow:
  Hopsworks Feature Store (latest features) --> Ridge model (Model Registry)
  --> 3-day-ahead prediction --> displayed here

AQI scale used (OpenWeather standard, NOT the US 0-500 scale):
  1 = Good, 2 = Fair, 3 = Moderate, 4 = Poor, 5 = Very Poor
"""

import os
import joblib
import pandas as pd
import streamlit as st
import hopsworks


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Aerocast - Rawalpindi AQI Forecast",
    page_icon="🌫️",
    layout="centered"
)

AQI_LABELS = {
    1: ("Good", "🟢"),
    2: ("Fair", "🟡"),
    3: ("Moderate", "🟠"),
    4: ("Poor", "🔴"),
    5: ("Very Poor", "🟣"),
}

HAZARDOUS_THRESHOLD = 4  # AQI of 4 (Poor) or 5 (Very Poor) triggers the alert


def get_aqi_label(aqi_value):
    """Convert a raw AQI number (1-5) into a readable label + emoji."""
    rounded = max(1, min(5, round(aqi_value)))
    return AQI_LABELS[rounded]


# ---------------------------------------------------------------------------
# Connect to Hopsworks and load data + model
# Cached so this doesn't reconnect on every single interaction
# ---------------------------------------------------------------------------
def get_hopsworks_api_key():
    """
    Reads the Hopsworks API key.
    Works both locally (via .env / os.environ) and on Streamlit Community
    Cloud (via st.secrets), without needing two different code versions.
    """
    if "HOPSWORKS_API_KEY" in st.secrets:
        return st.secrets["HOPSWORKS_API_KEY"]
    return os.environ.get("HOPSWORKS_API_KEY")


@st.cache_resource(ttl=3600)  # reconnect at most once per hour
def load_data_and_model():
    api_key = get_hopsworks_api_key()
    project = hopsworks.login(api_key_value=api_key, project="Aero_cast")
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # Get the most recent rows from the feature store
    feature_group = fs.get_feature_group(name="aqi_features", version=1)
    df = feature_group.read()
    df = df.sort_values("datetime").reset_index(drop=True)

    
    # Download the trained Ridge model from the Model Registry.
    # get_model with version=None doesn't actually fetch the latest version -
    # it defaults to version 1. So we fetch all versions and pick the highest
    # number ourselves to make sure we always use the most recently trained model.
    all_versions = mr.get_models(name="aqi_ridge_model")
    latest_version = max(m.version for m in all_versions)
    model_entry = mr.get_model(name="aqi_ridge_model", version=latest_version)
    model_dir = model_entry.download()
    model = joblib.load(os.path.join(model_dir, "ridge_model.pkl"))

    return df, model



# ---------------------------------------------------------------------------
# Build the page
# ---------------------------------------------------------------------------
st.title("🌫️ Aerocast")
st.caption("3-day AQI forecast for Rawalpindi")

with st.spinner("Loading latest data and model..."):
    try:
        df, model = load_data_and_model()
        load_error = None
    except Exception as e:
        df, model = None, None
        load_error = str(e)

if load_error:
    st.error(f"Could not load data or model right now. Please try again shortly.\n\nDetails: {load_error}")
    st.stop()

# --- Prepare the most recent row for prediction ---
df["aqi_rolling_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
df["pm2_5_rolling_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()

latest_row = df.iloc[-1]

feature_cols = ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
                 "hour", "day", "month", "aqi_change_rate",
                 "aqi_rolling_24h", "pm2_5_rolling_24h"]

X_latest = latest_row[feature_cols].to_frame().T
predicted_aqi = model.predict(X_latest)[0]

current_aqi = latest_row["aqi"]
current_label, current_emoji = get_aqi_label(current_aqi)
predicted_label, predicted_emoji = get_aqi_label(predicted_aqi)

# --- Hazardous alert banner ---
if round(predicted_aqi) >= HAZARDOUS_THRESHOLD:
    st.error(
        f"⚠️ Hazardous air quality expected in 3 days: **{predicted_label}** "
        f"(AQI ≈ {predicted_aqi:.1f}). Consider limiting outdoor activity."
    )

# --- Current vs forecast, side by side ---
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"Current AQI ({latest_row['datetime'].strftime('%d %b, %H:%M')})",
        value=f"{current_aqi:.0f} {current_emoji}",
        help=current_label
    )
    st.caption(current_label)

with col2:
    st.metric(
        label="Forecast (3 days ahead)",
        value=f"{predicted_aqi:.1f} {predicted_emoji}",
        help=predicted_label
    )
    st.caption(predicted_label)

# --- Recent trend chart ---
st.subheader("Recent AQI trend")
recent_df = df.tail(24 * 7)  # last 7 days
chart_df = recent_df.set_index("datetime")[["aqi"]]
st.line_chart(chart_df)

st.caption(
    "AQI scale: 1 = Good, 2 = Fair, 3 = Moderate, 4 = Poor, 5 = Very Poor "
    "(OpenWeather Air Pollution Index)."
)