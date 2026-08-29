"""
Dashboard
A Streamlit web app that shows the current AQI for Rawalpindi, a 3-day-ahead
forecast using the trained Ridge model, a hazardous AQI alert, current
weather conditions, and a recent trend chart.

Data flow:
  Hopsworks Feature Store (latest features) --> Ridge model (Model Registry)
  --> 3-day-ahead prediction --> displayed here
  OpenWeather Weather API --> live temperature/humidity/pressure (display only,
  not used by the model, not stored in Hopsworks)

AQI scale used (OpenWeather standard, NOT the US 0-500 scale):
  1 = Good, 2 = Fair, 3 = Moderate, 4 = Poor, 5 = Very Poor
"""

import os
import joblib
import time
import requests
import pandas as pd
import streamlit as st
import hopsworks
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Aerocast - Rawalpindi AQI Forecast",
    page_icon="🌫️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0 0 160 160'%3E%3Cg stroke='%231D9E75' stroke-width='1.5' fill='none' opacity='0.14'%3E%3Cg transform='translate(15,15) scale(0.7)'%3E%3Cpath d='M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0z'/%3E%3C/g%3E%3Cg transform='translate(95,15) scale(0.7)'%3E%3Cpath d='M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2'/%3E%3C/g%3E%3Cg transform='translate(15,95) scale(0.7)'%3E%3Cpath d='M12 2C12 2 5 10 5 15a7 7 0 0 0 14 0c0-5-7-13-7-13z'/%3E%3C/g%3E%3Cg transform='translate(95,95) scale(0.7)'%3E%3Cpath d='M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    background-repeat: repeat;
    background-size: 240px 240px;
}
</style>
""", unsafe_allow_html=True)


AQI_LABELS = {
    1: ("Good", "🟢"),
    2: ("Fair", "🟡"),
    3: ("Moderate", "🟠"),
    4: ("Poor", "🔴"),
    5: ("Very Poor", "🟣"),
}

HAZARDOUS_THRESHOLD = 4  # AQI of 4 (Poor) or 5 (Very Poor) triggers the alert

RAWALPINDI_LAT = 33.6007
RAWALPINDI_LON = 73.0679


def get_aqi_label(aqi_value):
    """Convert a raw AQI number (1-5) into a readable label + emoji."""
    rounded = max(1, min(5, round(aqi_value)))
    return AQI_LABELS[rounded]

def current_emoji_color(aqi_value):
    """Return a hex color matching the AQI category, for custom HTML circles."""
    rounded = max(1, min(5, round(aqi_value)))
    colors = {
        1: "#2ECC71",  # green
        2: "#F1C40F",  # yellow
        3: "#E67E22",  # orange
        4: "#E24B4A",  # red
        5: "#9B59B6",  # purple
    }
    return colors[rounded]



# ---------------------------------------------------------------------------
# API keys - work both locally (.env / os.environ) and on Streamlit Cloud
# ---------------------------------------------------------------------------
def get_hopsworks_api_key():
    if "HOPSWORKS_API_KEY" in st.secrets:
        return st.secrets["HOPSWORKS_API_KEY"]
    return os.environ.get("HOPSWORKS_API_KEY")

def get_openweather_api_key():
    if "OPENWEATHER_API_KEY" in st.secrets:
        return st.secrets["OPENWEATHER_API_KEY"]
    return os.environ.get("OPENWEATHER_API_KEY")


# ---------------------------------------------------------------------------
# Connect to Hopsworks and load data + model
# Cached so this doesn't reconnect on every single interaction
# ---------------------------------------------------------------------------
@st.cache_resource(ttl=3600)  # reconnect at most once per hour
def load_data_and_models(max_retries=3):
    api_key = get_hopsworks_api_key()

    for attempt in range(1, max_retries + 1):
        try:
            project = hopsworks.login(api_key_value=api_key, project="Aero_cast")
            fs = project.get_feature_store()
            mr = project.get_model_registry()

            feature_group = fs.get_feature_group(name="aqi_features", version=1)
            df = feature_group.read()
            df = df.sort_values("datetime").reset_index(drop=True)

            model_registry_names = {
                "1day": "aqi_ridge_1day",
                "2day": "aqi_ridge_2day",
                "3day": "aqi_ridge_3day",
            }
            models = {}
            for horizon, reg_name in model_registry_names.items():
                all_versions = mr.get_models(name=reg_name)
                latest_version = max(m.version for m in all_versions)
                model_entry = mr.get_model(name=reg_name, version=latest_version)
                model_dir = model_entry.download()
                models[horizon] = joblib.load(os.path.join(model_dir, f"ridge_model_{horizon}.pkl"))

            return df, models  # success

        except Exception as e:
            if attempt < max_retries:
                wait_seconds = 5 * attempt
                time.sleep(wait_seconds)
            else:
                raise


# ---------------------------------------------------------------------------
# Live weather (display only - not stored in Hopsworks, not a model input)
# Cached for 30 minutes so we don't hit OpenWeather on every rerun
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def get_live_weather():
    api_key = get_openweather_api_key()
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": RAWALPINDI_LAT,
        "lon": RAWALPINDI_LON,
        "appid": api_key,
        "units": "metric",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
        }
    except Exception:
        return None


with st.spinner("Loading latest data and models..."):
    try:
        df, models = load_data_and_models()
        load_error = None
    except Exception as e:
        df, models = None, None
        load_error = str(e)

if load_error:
    st.error(f"Could not load data or model right now. Please try again shortly.\n\nDetails: {load_error}")
    st.stop()

weather = get_live_weather()

# --- Prepare the most recent row for prediction ---
df["aqi_rolling_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
df["pm2_5_rolling_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()

latest_row = df.iloc[-1]

feature_cols = ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
                 "hour", "day", "month", "aqi_change_rate",
                 "aqi_rolling_24h", "pm2_5_rolling_24h"]

X_latest = latest_row[feature_cols].to_frame().T

predictions = {}
for horizon, model in models.items():
    predictions[horizon] = model.predict(X_latest)[0]

current_aqi = latest_row["aqi"]
current_label, current_emoji = get_aqi_label(current_aqi)


# ---------------------------------------------------------------------------
# Layout: main content on the left, AQI scale as a vertical strip on the right
# ---------------------------------------------------------------------------
main_col, scale_col = st.columns([4, 1.3])

with main_col:
    worst_horizon = max(predictions, key=predictions.get)
    worst_value = predictions[worst_horizon]
    worst_label, _ = get_aqi_label(worst_value)
    worst_day_text = {"1day": "tomorrow", "2day": "in 2 days", "3day": "in 3 days"}[worst_horizon]

    header_col1, header_col2 = st.columns([3, 2])

    with header_col1:
        st.markdown("""
        <style>
        @keyframes shine {
            to { background-position: 200% center; }
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(226,75,74,0.4); }
            50% { box-shadow: 0 0 0 12px rgba(226,75,74,0); }
        }
        .aerocast-title {
            font-size: 70px;
            font-weight: 700;
            background: linear-gradient(90deg, #1D9E75, #378ADD, #1D9E75);
            background-size: 200% auto;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: shine 4s linear infinite;
            margin-bottom: 0;
        }
        </style>
        <div class="aerocast-title">🌫️ Aerocast</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <p style="color:#AAA;font-size:16px;margin-top:2px;margin-bottom:0px;">
            3-day AQI forecast for Rawalpindi
        </p>
        """, unsafe_allow_html=True)

    with header_col2:
        if round(worst_value) >= HAZARDOUS_THRESHOLD:
            st.markdown(f"""
                   <div style="
            background: #3B1414;
            border: 1px solid #E24B4A;
            border-radius: 12px;
            padding: 14px 18px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            width: 210px;
            height: 210px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-size:20px;">⚠️</span>
                    <span style="font-weight:600;color:#E24B4A;">Hazardous Air Quality</span>
                </div>
                <p style="margin:6px 0 0 0;font-size:14px;color:#DDD;">
                    Expected {worst_day_text}: <b>{worst_label}</b> (AQI ≈ {worst_value:.1f}).
                    Consider limiting outdoor activity.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # --- Current AQI ---
    st.markdown(f"""
        <p style="color:#888;font-size:16px;margin-top:6px;margin-bottom:4px;">
        Current AQI ({latest_row['datetime'].strftime('%d %b, %H:%M')})
    </p>
    <div style="display:flex;align-items:center;gap:12px;">
        <span style="font-size:40px;font-weight:600;">{current_aqi:.0f}</span>
        <div style="width:32px;height:32px;border-radius:50%;background:{current_emoji_color(current_aqi)};
             animation: pulse 2s ease-in-out infinite;"></div>
    </div>
    <p style="color:#888;font-size:14px;margin-top:4px;">{current_label}</p>
    """, unsafe_allow_html=True)

        # --- Current Conditions: Temperature, Wind, Humidity, Pressure, PM2.5 ---
    st.markdown("<p style='color:#888;font-size:14px;margin-top:20px;margin-bottom:8px;'>Current Conditions</p>", unsafe_allow_html=True)
    cond_cols = st.columns(5)

    pm25_value = latest_row["pm2_5"]

    if weather is not None:
        cond_values = [
            ("Temperature", f"{weather['temperature']:.1f}°C", "🌡️"),
            ("Wind", f"{weather['wind_speed']:.1f} m/s", "🌬️"),
            ("Humidity", f"{weather['humidity']:.0f}%", "💧"),
            ("Pressure", f"{weather['pressure']:.0f} hPa", "🧭"),
            ("PM2.5", f"{pm25_value:.1f} µg/m³", "🌫️"),
        ]
    else:
        cond_values = [
            ("Temperature", "N/A", "🌡️"),
            ("Wind", "N/A", "🌬️"),
            ("Humidity", "N/A", "💧"),
            ("Pressure", "N/A", "🧭"),
            ("PM2.5", f"{pm25_value:.1f} µg/m³", "🌫️"),
        ]

    for i, (label, value, icon) in enumerate(cond_values):
        with cond_cols[i]:
            st.markdown(f"""
            <div style="background:#1A1D24;border:1px solid #1D9E75;border-radius:12px;
                 padding:16px 10px;text-align:center;">
                <div style="width:44px;height:44px;border-radius:50%;background:rgba(29,158,117,0.15);
                     display:flex;align-items:center;justify-content:center;margin:0 auto 10px auto;
                     font-size:20px;">{icon}</div>
                <p style="color:#888;font-size:11px;letter-spacing:0.5px;margin-bottom:4px;">{label.upper()}</p>
                <p style="font-size:18px;font-weight:600;color:#EEE;margin:0;">{value}</p>
            </div>
            """, unsafe_allow_html=True)

    # --- 3-day forecast, one card per day ---
    st.markdown("<p style='color:#888;font-size:14px;margin-top:24px;margin-bottom:8px;'>3-Day Forecast</p>", unsafe_allow_html=True)

    day_labels = {"1day": "Tomorrow", "2day": "In 2 Days", "3day": "In 3 Days"}
    card_cols = st.columns(3)

    for i, horizon in enumerate(["1day", "2day", "3day"]):
        pred_value = predictions[horizon]
        pred_label, _ = get_aqi_label(pred_value)
        pred_color = current_emoji_color(pred_value)

        with card_cols[i]:
            st.markdown(f"""
            <div style="background:#1A1D24;border:1px solid {pred_color};border-radius:12px;
                 padding:16px;text-align:center;">
                <p style="color:#888;font-size:12px;letter-spacing:1px;margin-bottom:8px;">
                    {day_labels[horizon].upper()}
                </p>
                <p style="font-size:32px;font-weight:700;color:{pred_color};margin:0;">
                    {pred_value:.1f}
                </p>
                <p style="color:#AAA;font-size:14px;margin-top:6px;">{pred_label}</p>
            </div>
            """, unsafe_allow_html=True)

    # --- Recent trend chart (history + forecast) ---
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    st.subheader("Recent AQI trend")
    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

    recent_df = df.tail(24 * 7).copy()  # last 7 days
    last_time = recent_df["datetime"].iloc[-1]
    last_aqi = recent_df["aqi"].iloc[-1]

    forecast_times = [last_time + timedelta(hours=24 * i) for i in (1, 2, 3)]
    forecast_values = [predictions["1day"], predictions["2day"], predictions["3day"]]

    hist_part = pd.DataFrame({
        "datetime": recent_df["datetime"],
        "Historical AQI": recent_df["aqi"],
        "Forecast AQI": [None] * len(recent_df),
    })
    forecast_part = pd.DataFrame({
        "datetime": [last_time] + forecast_times,
        "Historical AQI": [None] * 4,
        "Forecast AQI": [last_aqi] + forecast_values,
    })

    chart_df = pd.concat([hist_part, forecast_part], ignore_index=True).set_index("datetime")

    st.line_chart(chart_df, color=["#378ADD", "#1D9E75"])

with scale_col:
    st.markdown("""
    <div style="background:#1A1D24;border:1px solid #333;border-radius:12px;
         padding:22px 18px;padding-top: 24px; margin-left: 12px;">
        <p style="color:#888;font-size:15px;letter-spacing:1px;margin-bottom:18px;">AQI SCALE</p>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <div style="width:18px;height:18px;border-radius:50%;background:#2ECC71;"></div>
            <span style="font-size:15px;color:#CCC;">1 - Good</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <div style="width:18px;height:18px;border-radius:50%;background:#F1C40F;"></div>
            <span style="font-size:15px;color:#CCC;">2 - Fair</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <div style="width:18px;height:18px;border-radius:50%;background:#E67E22;"></div>
            <span style="font-size:15px;color:#CCC;">3 - Moderate</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <div style="width:18px;height:18px;border-radius:50%;background:#E24B4A;"></div>
            <span style="font-size:15px;color:#CCC;">4 - Poor</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <div style="width:18px;height:18px;border-radius:50%;background:#9B59B6;"></div>
            <span style="font-size:15px;color:#CCC;">5 - Very Poor</span>
        </div>
        <p style="color:#666;font-size:12px;margin-top:20px;">OpenWeather Air Pollution Index</p>
    </div>
    """, unsafe_allow_html=True)