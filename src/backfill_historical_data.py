import requests
import time
from datetime import datetime, timedelta
from config import LATITUDE, LONGITUDE, OPENWEATHER_API_KEY, CITY_NAME

CHUNK_SIZE_DAYS = 90   # safely under the ~171 day limit we discovered

def fetch_chunk(start_date, end_date):
    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())

    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "start": start_unix,
        "end": end_unix,
        "appid": OPENWEATHER_API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data["list"]
    else:
        print(f"Chunk failed ({start_date.date()} to {end_date.date()}): {response.status_code}")
        return []

def backfill_two_years():
    all_records = []
    end_date = datetime.now()
    two_years_ago = end_date - timedelta(days=730)

    current_end = end_date
    while current_end > two_years_ago:
        current_start = max(current_end - timedelta(days=CHUNK_SIZE_DAYS), two_years_ago)

        print(f"Fetching {current_start.date()} to {current_end.date()}...")
        chunk_records = fetch_chunk(current_start, current_end)
        print(f"  -> got {len(chunk_records)} records")

        all_records.extend(chunk_records)

        current_end = current_start
        time.sleep(1)   # small pause to be polite to the API between calls

    print(f"\nTotal records collected: {len(all_records)}")
    return all_records

if __name__ == "__main__":
    records = backfill_two_years()