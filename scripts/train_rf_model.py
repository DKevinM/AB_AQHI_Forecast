import pandas as pd
import numpy as np
import os
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------------
# CONFIG
# -----------------------------------

DATA_FILE = "/tmp/training_dataset.csv.gz"

os.makedirs("models", exist_ok=True)

DIAG_DIR = Path("diagnostics")
DIAG_DIR.mkdir(exist_ok=True)

# -----------------------------------
# Load Dataset
# -----------------------------------

print("Loading training dataset...")

data = pd.read_csv(DATA_FILE, compression="gzip")

print("Rows:", len(data))
print("Columns:", len(data.columns))

data["datetime"] = pd.to_datetime(data["datetime"])
data = data.sort_values("datetime").reset_index(drop=True)

# -----------------------------------
# RH MODEL FIELD
# -----------------------------------

data["RH_model"] = data["RH"]

if "RH_filled" in data.columns:
    rh_filled_max = data["RH_filled"].max(skipna=True)

    if rh_filled_max > 1:
        print("Using RH_filled as filled RH value.")
        data["RH_model"] = data["RH_model"].fillna(data["RH_filled"])
    else:
        print("RH_filled appears to be a flag, not a filled RH value.")

data["RH_was_missing"] = data["RH"].isna().astype(int)

print("RH missing before RH_model fill:", data["RH"].isna().sum())
print("RH_model missing after fill:", data["RH_model"].isna().sum())

# -----------------------------------
# Diagnostics before final drop
# -----------------------------------

print("Rows before final model drop:", len(data))

station_counts = (
    data.groupby("station")
      .size()
      .reset_index(name="training_rows")
      .sort_values("training_rows")
)

station_counts.to_csv(
    DIAG_DIR / "station_rows_in_training_dataset.csv",
    index=False
)

print("\nStations in raw training dataset:")
print("Count:", data["station"].nunique())
print(station_counts.head(20))

MASTER_FILE = Path("data/AB_master.csv.gz")

if MASTER_FILE.exists():
    master = pd.read_csv(
        MASTER_FILE,
        compression="gzip",
        usecols=["station"]
    )

    master_stations = set(master["station"].dropna().unique())
    training_stations = set(data["station"].dropna().unique())

    removed_stations = sorted(master_stations - training_stations)

    pd.DataFrame({"station": removed_stations}).to_csv(
        DIAG_DIR / "stations_in_master_but_not_training.csv",
        index=False
    )

    print("\nStations in master:", len(master_stations))
    print("Stations in training before final drop:", len(training_stations))
    print("Stations missing from training before final drop:")
    for s in removed_stations:
        print(" -", s)
else:
    print("\nWARNING: data/AB_master.csv.gz not found, cannot compare removed stations.")

# -----------------------------------
# Feature Columns - 3H ONLY
# -----------------------------------

feature_cols = [

    # AQHI state
    "AQHI",

    "AQHI_lag1",
    "AQHI_lag2",
    "AQHI_lag3",
    "AQHI_lag6",
    "AQHI_lag12",
    "AQHI_lag24",

    "AQHI_change_1h",
    "AQHI_change_3h",

    # Pollutants
    "PM25",
    "NO2",
    "O3",

    # Current weather
    "WS",
    "WD",
    "U",
    "V",
    "TEMP",
    "RH_model",

    # Time
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",

    # Spatial
    "lat_norm",
    "lon_norm",
    "dist_center",

    # Gap-fill flags
    "PM25_filled",
    "NO2_filled",
    "O3_filled",
    "TEMP_filled",
    "RH_was_missing",
    "WS_filled",
    "WD_filled",

    # 3-hour future meteorology proxy
    "WS_future_3h",
    "WD_future_3h",
    "TEMP_future_3h",
    "RH_future_3h",
    "U_future_3h",
    "V_future_3h"
]

target_col = "AQHI_future_3h"

required_cols = feature_cols + [target_col, "datetime", "station"]

missing_cols = [c for c in required_cols if c not in data.columns]

if missing_cols:
    raise ValueError(
        "Missing required columns: " + ", ".join(missing_cols)
    )

# -----------------------------------
# Final Drop
# -----------------------------------

rows_before_drop = len(data)
stations_before_drop = data["station"].nunique()

data = data.dropna(subset=required_cols).copy()

rows_after_drop = len(data)
stations_after_drop = data["station"].nunique()

print("\nRows before final model drop:", rows_before_drop)
print("Rows after final model drop :", rows_after_drop)
print("Rows retained %:", round(100 * rows_after_drop / rows_before_drop, 1))
print("Stations before final model drop:", stations_before_drop)
print("Stations after final model drop :", stations_after_drop)

station_counts_final = (
    data.groupby("station")
      .size()
      .reset_index(name="training_rows")
      .sort_values("training_rows")
)

station_counts_final.to_csv(
    DIAG_DIR / "station_rows_after_final_rf_3h_drop.csv",
    index=False
)

# -----------------------------------
# X / y
# -----------------------------------

X = data[feature_cols]
y = data[target_col]

print("\nFeature count:", len(feature_cols))
print(feature_cols)

# -----------------------------------
# Train 3H Model
# -----------------------------------

print("\n===================================")
print("Training: aqhi_3h")
print("Target:", target_col)
print("===================================\n")

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=10,
    max_features="sqrt",
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

high4_mask = y_test >= 4
high6_mask = y_test >= 6

high4_rmse = np.nan
high4_mae = np.nan
high6_rmse = np.nan
high6_mae = np.nan

if high4_mask.sum() > 0:
    high4_rmse = np.sqrt(
        mean_squared_error(
            y_test[high4_mask],
            pred[high4_mask]
        )
    )
    high4_mae = mean_absolute_error(
        y_test[high4_mask],
        pred[high4_mask]
    )

if high6_mask.sum() > 0:
    high6_rmse = np.sqrt(
        mean_squared_error(
            y_test[high6_mask],
            pred[high6_mask]
        )
    )
    high6_mae = mean_absolute_error(
        y_test[high6_mask],
        pred[high6_mask]
    )

print("\nResults")
print("RMSE:", round(rmse, 3))
print("MAE :", round(mae, 3))
print("R²  :", round(r2, 3))
print("OOB :", round(model.oob_score_, 3))

print("\nElevated AQHI Metrics")
print("AQHI >= 4 count:", int(high4_mask.sum()))
print("AQHI >= 4 RMSE :", round(high4_rmse, 3))
print("AQHI >= 4 MAE  :", round(high4_mae, 3))
print("AQHI >= 6 count:", int(high6_mask.sum()))
print("AQHI >= 6 RMSE :", round(high6_rmse, 3))
print("AQHI >= 6 MAE  :", round(high6_mae, 3))

# -----------------------------------
# Save model
# -----------------------------------

model_name = "aqhi_3h"

model_file = f"models/{model_name}_model.pkl"

joblib.dump(
    model,
    model_file
)

print("Saved:", model_file)

# -----------------------------------
# Feature importance
# -----------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

importance.to_csv(
    f"models/{model_name}_importance.csv"
)

print("\nTop 20 Features")
print(importance.head(20))

# -----------------------------------
# Metrics file
# -----------------------------------

with open(
    f"models/{model_name}_metrics.txt",
    "w"
) as f:

    f.write(f"Model: {model_name}\n")
    f.write(f"Rows: {len(X)}\n")
    f.write(f"Training Rows: {len(X_train)}\n")
    f.write(f"Testing Rows: {len(X_test)}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"R2: {r2}\n")
    f.write(f"OOB: {model.oob_score_}\n")
    f.write(f"AQHI_GE_4_Count: {int(high4_mask.sum())}\n")
    f.write(f"AQHI_GE_4_RMSE: {high4_rmse}\n")
    f.write(f"AQHI_GE_4_MAE: {high4_mae}\n")
    f.write(f"AQHI_GE_6_Count: {int(high6_mask.sum())}\n")
    f.write(f"AQHI_GE_6_RMSE: {high6_rmse}\n")
    f.write(f"AQHI_GE_6_MAE: {high6_mae}\n")

# -----------------------------------
# Summary
# -----------------------------------

summary = pd.DataFrame(
    {
        "RMSE": [rmse],
        "MAE": [mae],
        "R2": [r2],
        "OOB": [model.oob_score_],
        "AQHI_GE_4_Count": [int(high4_mask.sum())],
        "AQHI_GE_4_RMSE": [high4_rmse],
        "AQHI_GE_4_MAE": [high4_mae],
        "AQHI_GE_6_Count": [int(high6_mask.sum())],
        "AQHI_GE_6_RMSE": [high6_rmse],
        "AQHI_GE_6_MAE": [high6_mae]
    },
    index=[model_name]
)

summary.to_csv(
    "models/model_summary.csv"
)

print("\n===================================")
print("MODEL SUMMARY")
print("===================================\n")

print(summary)

print("\nFinished.")
