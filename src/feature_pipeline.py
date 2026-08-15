import requests
import time
import pandas as pd
import hopsworks
from datetime import datetime, timedelta
from config import LATITUDE, LONGITUDE, OPENWEATHER_API_KEY, CITY_NAME

CHUNK_SIZE_DAYS = 90

def fetch_chunk(start_date, end_date):
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
        print(f"Chunk failed ({start_date.date()} to {end_date.date()}): {response.status_code}")
        return []

def fetch_raw_historical_data(days_back=730):
    all_records = []
    end_date = datetime.now()
    cutoff = end_date - timedelta(days=days_back)

    current_end = end_date
    while current_end > cutoff:
        current_start = max(current_end - timedelta(days=CHUNK_SIZE_DAYS), cutoff)
        chunk_records = fetch_chunk(current_start, current_end)
        all_records.extend(chunk_records)
        current_end = current_start
        time.sleep(1)

    return all_records

def raw_records_to_dataframe(records):
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
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month

    df["aqi_change_rate"] = df["aqi"].diff()
    df["aqi_change_rate"] = df["aqi_change_rate"].fillna(0)

    return df
def upload_to_feature_store(df):
    print("\nConnecting to Hopsworks...")
    project = hopsworks.login(project="Aero_cast")
    fs = project.get_feature_store()

    print("Creating/getting feature group...")
    feature_group = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="Hourly AQI and pollution features for Rawalpindi",
        primary_key=["timestamp"],
        event_time="datetime",
        time_travel_format="HUDI"
    )

    print("Inserting data (this may take a few minutes for 16,992 rows)...")
    feature_group.insert(df)
    print("Upload complete!")

if __name__ == "__main__":
    print("Fetching raw historical data...")
    records = fetch_raw_historical_data(days_back=730)
    print(f"Got {len(records)} raw records")

    print("Converting to dataframe...")
    df = raw_records_to_dataframe(records)

    print("Adding features...")
    df = add_features(df)

    print("\nChecking for suspicious sentinel values...")
    for col in ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]:
        suspicious_count = (df[col] < 0).sum()
        if suspicious_count > 0:
            print(f"  {col}: {suspicious_count} negative/suspicious values found")

    print("\nCleaning suspicious values...")
    pollutant_cols = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
    for col in pollutant_cols:
        df.loc[df[col] < 0, col] = None   # mark bad values as missing
    df[pollutant_cols] = df[pollutant_cols].interpolate(method="linear")

    print("Re-checking after cleaning...")
    for col in pollutant_cols:
        remaining = (df[col] < 0).sum()
        if remaining > 0:
            print(f"  {col}: still has {remaining} negative values!")
        else:
            print(f"  {col}: clean")
        
    print(df.head())
    print(f"\nTotal rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    df.to_csv("aerocast_features_raw.csv", index=False)
    print("\nSaved to aerocast_features_raw.csv")
    upload_to_feature_store(df)