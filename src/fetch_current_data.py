import requests
from config import LATITUDE, LONGITUDE, OPENWEATHER_API_KEY, CITY_NAME

def fetch_current_air_pollution():
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "appid": OPENWEATHER_API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        print(f"Current air pollution data for {CITY_NAME}:")
        print(data)
    else:
        print(f"Request failed. Status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    fetch_current_air_pollution()