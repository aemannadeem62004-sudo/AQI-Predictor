"""
hourly_feature_update.py

Aerocast - Week 4 Hourly Feature Pipeline
Fetches the last 2 hours of AQI/pollution data from OpenWeather and
inserts it into the existing Hopsworks Feature Store (aqi_features, v1).

This is DIFFERENT from feature_pipeline.py:
- feature_pipeline.py = one-time historical backfill (2 years of data)
- hourly_feature_update.py = ongoing small updates (last 2 hours only),
  meant to run automatically every hour via GitHub Actions

Fetching 2 hours (not just 1) gives us enough data to calculate
aqi_change_rate (which needs a "previous hour" value to compare against).
Hopsworks uses "timestamp" as the primary key, so re-inserting an hour
that's already stored just safely updates it - it won't create duplicates.
"""

import os
import requests
import pandas as pd
import hopsworks
import time
from datetime import datetime, timedelta
from config import LATITUDE, LONGITUDE, OPENWEATHER_API_KEY, CITY_NAME


def fetch_recent_data(hours_back=2):
    """Fetch the last N hours of AQI/pollution data from OpenWeather."""
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours_back)

    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())

    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "start": start_unix,
        "end": end_unix,
        "appid": OPENWEATHER_API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()["list"]
    else:
        print(f"Fetch failed: {response.status_code} - {response.text}")
        return []


def records_to_dataframe(records):
    """Convert raw OpenWeather records into a clean dataframe."""
    rows = []
    for record in records:
        row = {
            "timestamp": record["dt"],
            "city": CITY_NAME,
            "aqi": record["main"]["aqi"],
            "co": record["components"]["co"],
            "no": record["components"]["no"],
            "no2": record["components"]["no2"],
            "o3": record["components"]["o3"],
            "so2": record["components"]["so2"],
            "pm2_5": record["components"]["pm2_5"],
            "pm10": record["components"]["pm10"],
            "nh3": record["components"]["nh3"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def add_features(df):
    """Add the same time-based and derived features used in training."""
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month

    df["aqi_change_rate"] = df["aqi"].diff()
    df["aqi_change_rate"] = df["aqi_change_rate"].fillna(0)

    return df


def upload_to_feature_store(df, max_retries=3):
    """
    Insert the new rows into the existing Hopsworks feature group.
    Includes retry logic - Hopsworks' free-tier servers occasionally
    drop the connection briefly, so we retry a few times before
    actually failing.
    """
    HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Connecting to Hopsworks (attempt {attempt}/{max_retries})...")
            project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="Aero_cast")
            fs = project.get_feature_store()

            print("Getting feature group...")
            feature_group = fs.get_feature_group(name="aqi_features", version=1)

            print(f"Inserting {len(df)} rows...")
            feature_group.insert(df)
            print("Hourly update complete!")
            return  # success - exit the function

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait_seconds = 10 * attempt  # waits 10s, then 20s
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                print("All retry attempts failed. Raising the error.")
                raise


if __name__ == "__main__":
    print("Fetching recent data...")
    records = fetch_recent_data(hours_back=2)
    print(f"Got {len(records)} raw records")

    if len(records) == 0:
        print("No new data fetched - exiting without upload.")
    else:
        print("Converting to dataframe...")
        df = records_to_dataframe(records)

        print("Adding features...")
        df = add_features(df)

        # Same cleaning approach as the backfill script - catch any
        # negative/sentinel values before uploading
        pollutant_cols = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
        for col in pollutant_cols:
            df.loc[df[col] < 0, col] = None
        df[pollutant_cols] = df[pollutant_cols].interpolate(method="linear")

        print(df)
        upload_to_feature_store(df)