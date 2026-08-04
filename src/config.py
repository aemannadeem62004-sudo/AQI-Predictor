import os
from dotenv import load_dotenv

load_dotenv()

CITY_NAME = "Rawalpindi"
LATITUDE = 33.5651
LONGITUDE = 73.0169

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")