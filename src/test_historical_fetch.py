import requests
from datetime import datetime, timedelta
from config import LATITUDE, LONGITUDE, OPENWEATHER_API_KEY, CITY_NAME

def fetch_historical_air_pollution(start_date, end_date):
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
        print(f"Fetched {len(data['list'])} hourly records for {CITY_NAME}")
        print("First record:", data['list'][0])
        print("Last record:", data['list'][-1])
    else:
        print(f"Request failed. Status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    end = datetime.now()
    start = end - timedelta(days=180)   # test with 6 months
    fetch_historical_air_pollution(start, end)