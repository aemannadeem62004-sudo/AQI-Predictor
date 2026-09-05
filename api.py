"""
Aerocast API (Flask)

A minimal, standalone Flask API exposing the same AQI predictions shown on
the Aerocast Streamlit dashboard, in JSON format — added to satisfy the
project's Flask/FastAPI requirement without restructuring the dashboard.

This file is fully independent of app.py: it does not import from it, and
running one has no effect on the other. It duplicates the small amount of
logic needed (Hopsworks connection, feature loading, EPA AQI conversion)
so each file can be understood and run on its own.

Endpoints:
  GET /api/current   -> current EPA AQI, label, and PM2.5
  GET /api/forecast   -> 1/2/3-day-ahead EPA AQI predictions

Run locally:
  python api.py
  (then visit http://127.0.0.1:5000/api/current in a browser or via curl)
"""

import os
import math
import joblib
import pandas as pd
import hopsworks
from flask import Flask, jsonify

app = Flask(__name__)


# ---------------------------------------------------------------------------
# EPA AQI conversion helpers (same logic as app.py)
# ---------------------------------------------------------------------------
def get_epa_aqi_from_pm25(pm25):
    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 400),
        (325.5, 500.4, 401, 500),
    ]
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(((aqi_high - aqi_low) / (c_high - c_low)) * (pm25 - c_low) + aqi_low)
    return 500


def get_epa_aqi_label(aqi_value):
    if aqi_value <= 50: return "Good"
    elif aqi_value <= 100: return "Moderate"
    elif aqi_value <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200: return "Unhealthy"
    elif aqi_value <= 300: return "Very Unhealthy"
    else: return "Hazardous"


def build_category_pm25_map(df):
    return df.groupby("aqi")["pm2_5"].mean().to_dict()


def estimate_pm25_from_category(predicted_category, category_map):
    keys = sorted(category_map.keys())
    low = max(min(int(math.floor(predicted_category)), max(keys)), min(keys))
    high = max(min(int(math.ceil(predicted_category)), max(keys)), min(keys))
    if low == high:
        return category_map[low]
    frac = predicted_category - low
    return category_map[low] + frac * (category_map[high] - category_map[low])


# ---------------------------------------------------------------------------
# Hopsworks connection + data/model loading (runs once at startup)
# ---------------------------------------------------------------------------
def get_hopsworks_api_key():
    return os.environ.get("HOPSWORKS_API_KEY")


print("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=get_hopsworks_api_key(), project="Aero_cast")
fs = project.get_feature_store()
mr = project.get_model_registry()

feature_group = fs.get_feature_group(name="aqi_features", version=1)
df = feature_group.read()
df = df.sort_values("datetime").reset_index(drop=True)
df["aqi_rolling_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
df["pm2_5_rolling_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()

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

category_pm25_map = build_category_pm25_map(df)
print("Ready.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/api/current", methods=["GET"])
def current_aqi():
    latest_row = df.iloc[-1]
    epa_aqi = get_epa_aqi_from_pm25(latest_row["pm2_5"])
    return jsonify({
        "datetime": str(latest_row["datetime"]),
        "epa_aqi": epa_aqi,
        "label": get_epa_aqi_label(epa_aqi),
        "pm2_5": round(float(latest_row["pm2_5"]), 2),
    })


@app.route("/api/forecast", methods=["GET"])
def forecast():
    latest_row = df.iloc[-1]
    feature_cols = ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
                     "hour", "day", "month", "aqi_change_rate",
                     "aqi_rolling_24h", "pm2_5_rolling_24h"]
    X_latest = latest_row[feature_cols].to_frame().T

    results = {}
    for horizon, model in models.items():
        cat_value = model.predict(X_latest)[0]
        estimated_pm25 = estimate_pm25_from_category(cat_value, category_pm25_map)
        epa_aqi = get_epa_aqi_from_pm25(estimated_pm25)
        results[horizon] = {
            "epa_aqi": epa_aqi,
            "label": get_epa_aqi_label(epa_aqi),
        }

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)