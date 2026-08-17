"""
training_pipeline.py

Aerocast - Week 3 Training Pipeline
Pulls processed AQI features from the Hopsworks Feature Store, builds the
3-day-ahead prediction target, trains and compares 3 models (Ridge,
Random Forest, TensorFlow), saves the best model (Ridge) to the Hopsworks
Model Registry, and generates a SHAP explainability summary plot.

NOTE: This script was developed and run in Google Colab (not locally),
because of persistent Hopsworks/Delta dependency issues on Windows.
It is saved here as a record of the working pipeline for the internship
report and GitHub history. To actually run it, use the Colab notebook.
"""

from google.colab import userdata
import hopsworks
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import shap


# ---------------------------------------------------------------------------
# STEP 1: Connect to Hopsworks and pull the processed feature data
# ---------------------------------------------------------------------------
HOPSWORKS_API_KEY = userdata.get('HOPSWORKS_API_KEY')

print("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="Aero_cast")
fs = project.get_feature_store()

print("Fetching feature group...")
feature_group = fs.get_feature_group(name="aqi_features", version=1)

print("Reading data...")
df = feature_group.read()

print(f"Total rows: {len(df)}")


# ---------------------------------------------------------------------------
# STEP 2: Build the prediction target and extra trend features
# ---------------------------------------------------------------------------
# Sort by actual time order first - this matters for both the shift below
# and for the chronological train/test split later
df = df.sort_values("datetime").reset_index(drop=True)

# The target is "AQI 3 days (72 hours) from now" - shift the aqi column
# backwards by 72 rows so each row's target is its own future value
HOURS_AHEAD = 3 * 24
df["aqi_target_3day"] = df["aqi"].shift(-HOURS_AHEAD)

# The last 72 rows have no future value to predict, so drop them
df = df.dropna(subset=["aqi_target_3day"])

# Rolling averages give the model a sense of recent trend, not just a
# single snapshot in time
df["aqi_rolling_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
df["pm2_5_rolling_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()

print(f"Rows after processing: {len(df)}")
print(df[["datetime", "aqi", "aqi_target_3day", "aqi_rolling_24h"]].head())


# ---------------------------------------------------------------------------
# STEP 3: Train/test split - CHRONOLOGICAL, not random
# ---------------------------------------------------------------------------
# Random splitting would leak future information into training. Since this
# is a time series, train on the earlier 85% and test on the most recent 15%.
feature_cols = ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
                 "hour", "day", "month", "aqi_change_rate",
                 "aqi_rolling_24h", "pm2_5_rolling_24h"]
target_col = "aqi_target_3day"

split_index = int(len(df) * 0.85)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

print(f"Training set: {len(X_train)} rows")
print(f"Test set: {len(X_test)} rows")


# ---------------------------------------------------------------------------
# STEP 4: Train and compare 3 models
# ---------------------------------------------------------------------------
results = {}

# --- Model 1: Ridge Regression ---
print("Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_preds = ridge_model.predict(X_test)
results["Ridge Regression"] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, ridge_preds)),
    "MAE": mean_absolute_error(y_test, ridge_preds),
    "R2": r2_score(y_test, ridge_preds)
}

# --- Model 2: Random Forest ---
# max_depth and min_samples_leaf are set to limit overfitting - an
# earlier untuned version overfit badly on this data
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20,
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
results["Random Forest"] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, rf_preds)),
    "MAE": mean_absolute_error(y_test, rf_preds),
    "R2": r2_score(y_test, rf_preds)
}

print("\nResults so far:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# --- Model 3: TensorFlow Neural Network ---
# Neural networks need scaled inputs (mean 0, std 1) to train properly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Building TensorFlow model...")
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

tf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Training TensorFlow model...")
history = tf_model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=32,
    verbose=1
)

tf_preds = tf_model.predict(X_test_scaled).flatten()
results["TensorFlow Neural Network"] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, tf_preds)),
    "MAE": mean_absolute_error(y_test, tf_preds),
    "R2": r2_score(y_test, tf_preds)
}

print("\nAll results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# CONCLUSION: Ridge Regression performed best (lowest RMSE/MAE, highest R2).
# 3-day-ahead AQI prediction from a single time snapshot is a genuinely hard
# problem - the simpler linear model generalized better than Random Forest
# or the neural network, both of which showed signs of overfitting.


# ---------------------------------------------------------------------------
# STEP 5: Save the best model (Ridge) to the Hopsworks Model Registry
# ---------------------------------------------------------------------------
# Save the trained model to a local file first
joblib.dump(ridge_model, "ridge_model.pkl")

print("Connecting to Model Registry...")
mr = project.get_model_registry()

# Register the model along with its performance metrics
model = mr.python.create_model(
    name="aqi_ridge_model",
    metrics={
        "rmse": results["Ridge Regression"]["RMSE"],
        "mae": results["Ridge Regression"]["MAE"],
        "r2": results["Ridge Regression"]["R2"]
    },
    description="Ridge Regression model predicting AQI 3 days ahead for Rawalpindi. "
                "Best of 3 models compared (Ridge, Random Forest, TensorFlow)."
)

print("Uploading model...")
model.save("ridge_model.pkl")
print("Model saved to registry!")


# ---------------------------------------------------------------------------
# STEP 6: SHAP explainability - which features matter most, and how
# ---------------------------------------------------------------------------
print("Creating SHAP explainer...")
explainer = shap.LinearExplainer(ridge_model, X_train)
shap_values = explainer(X_test)

print("Generating summary plot...")
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)