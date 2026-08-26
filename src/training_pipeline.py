"""
training_pipeline.py

Aerocast - Day-by-day Training Pipeline (1-day, 2-day, 3-day ahead)
Pulls processed AQI features from the Hopsworks Feature Store, builds THREE
prediction targets (24h, 48h, 72h ahead), trains a separate Ridge model for
each horizon (Ridge was already established as the best-performing model
type in earlier testing), and saves all three to the Model Registry.

This replaces the single 3-day-only model from Week 3 with three models,
one per forecast day, so the dashboard can show "Tomorrow / In 2 days /
In 3 days" separately instead of one combined number.
"""

import os
import hopsworks
import time
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# STEP 1: Connect to Hopsworks and pull the processed feature data
# ---------------------------------------------------------------------------
HOPSWORKS_API_KEY = os.environ.get('HOPSWORKS_API_KEY')

print("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="Aero_cast")
fs = project.get_feature_store()

print("Fetching feature group...")
feature_group = fs.get_feature_group(name="aqi_features", version=1)

print("Reading data...")
max_retries = 3
for attempt in range(1, max_retries + 1):
    try:
        df = feature_group.read()
        break  # success
    except Exception as e:
        print(f"Read attempt {attempt} failed: {e}")
        if attempt < max_retries:
            wait_seconds = 15 * attempt
            print(f"Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)
        else:
            print("All retry attempts failed.")
            raise

print(f"Total rows: {len(df)}")


# ---------------------------------------------------------------------------
# STEP 2: Build THREE prediction targets - 1 day, 2 days, 3 days ahead
# ---------------------------------------------------------------------------
df = df.sort_values("datetime").reset_index(drop=True)

df["aqi_target_1day"] = df["aqi"].shift(-24)
df["aqi_target_2day"] = df["aqi"].shift(-48)
df["aqi_target_3day"] = df["aqi"].shift(-72)

# Drop rows missing the 3-day target (the most restrictive) so all three
# targets are available for the same set of rows - keeps train/test
# consistent across all three models
df = df.dropna(subset=["aqi_target_1day", "aqi_target_2day", "aqi_target_3day"])

df["aqi_rolling_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
df["pm2_5_rolling_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()

print(f"Rows after processing: {len(df)}")


# ---------------------------------------------------------------------------
# STEP 3: Train/test split - CHRONOLOGICAL, not random
# ---------------------------------------------------------------------------
feature_cols = ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
                 "hour", "day", "month", "aqi_change_rate",
                 "aqi_rolling_24h", "pm2_5_rolling_24h"]

target_cols = {
    "1day": "aqi_target_1day",
    "2day": "aqi_target_2day",
    "3day": "aqi_target_3day",
}

split_index = int(len(df) * 0.85)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

print(f"Training set: {len(X_train)} rows")
print(f"Test set: {len(X_test)} rows")


# ---------------------------------------------------------------------------
# STEP 4: Train one Ridge model per horizon (same inputs, different targets)
# ---------------------------------------------------------------------------
trained_models = {}
results = {}

for horizon_name, target_col in target_cols.items():
    print(f"\nTraining Ridge model for {horizon_name}...")
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results[horizon_name] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }
    trained_models[horizon_name] = model

    print(f"  RMSE: {results[horizon_name]['RMSE']:.4f}")
    print(f"  MAE:  {results[horizon_name]['MAE']:.4f}")
    print(f"  R2:   {results[horizon_name]['R2']:.4f}")


# ---------------------------------------------------------------------------
# STEP 5: Save all three models to the Hopsworks Model Registry
# ---------------------------------------------------------------------------
print("\nConnecting to Model Registry...")
mr = project.get_model_registry()

model_names = {
    "1day": "aqi_ridge_1day",
    "2day": "aqi_ridge_2day",
    "3day": "aqi_ridge_3day",
}

for horizon_name, model in trained_models.items():
    reg_name = model_names[horizon_name]
    local_filename = f"ridge_model_{horizon_name}.pkl"

    joblib.dump(model, local_filename)

    print(f"Uploading {reg_name}...")
    model_entry = mr.python.create_model(
        name=reg_name,
        metrics={
            "rmse": results[horizon_name]["RMSE"],
            "mae": results[horizon_name]["MAE"],
            "r2": results[horizon_name]["R2"]
        },
        description=f"Ridge Regression model predicting AQI {horizon_name.replace('day', ' day(s)')} "
                     f"ahead for Rawalpindi."
    )
    model_entry.save(local_filename)
    print(f"{reg_name} saved to registry!")

print("\nAll three models trained and saved successfully!")