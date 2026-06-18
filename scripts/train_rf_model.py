
import pandas as pd
import numpy as np
import os
import joblib

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

# -----------------------------------
# Load Dataset
# -----------------------------------

print("Loading training dataset...")

data = pd.read_csv(DATA_FILE)

print("Rows:", len(data))
print("Columns:", len(data.columns))

data = data.dropna()

data["datetime"] = pd.to_datetime(data["datetime"])
data = data.sort_values("datetime").reset_index(drop=True)

print("Rows after NA removal:", len(data))

# -----------------------------------
# Feature Columns
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

    # Weather
    "WS",
    "WD",
    "U",
    "V",
    "TEMP",
    "RH",

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
    "RH_filled",
    "WS_filled",
    "WD_filled"
]

X = data[feature_cols]

print("\nFeature count:", len(feature_cols))
print(feature_cols)

# -----------------------------------
# Train Function
# -----------------------------------

def train_model(target, name):

    print("\n===================================")
    print("Training:", name)
    print("Target:", target)
    print("===================================\n")

    y = data[target]

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
        high4_rmse = np.sqrt(mean_squared_error(y_test[high4_mask], pred[high4_mask]))
        high4_mae = mean_absolute_error(y_test[high4_mask], pred[high4_mask])
    
    if high6_mask.sum() > 0:
        high6_rmse = np.sqrt(mean_squared_error(y_test[high6_mask], pred[high6_mask]))
        high6_mae = mean_absolute_error(y_test[high6_mask], pred[high6_mask])

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

    # -----------------------------
    # Save model
    # -----------------------------

    model_file = f"models/{name}_model.pkl"

    joblib.dump(
        model,
        model_file
    )

    print("Saved:", model_file)

    # -----------------------------
    # Feature importance
    # -----------------------------

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    importance.to_csv(
        f"models/{name}_importance.csv"
    )

    print("\nTop 20 Features")
    print(importance.head(20))

    # -----------------------------
    # Metrics file
    # -----------------------------

    with open(
        f"models/{name}_metrics.txt",
        "w"
    ) as f:

        f.write(f"Model: {name}\n")
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

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

# -----------------------------------
# Train Models
# -----------------------------------

results = {}

results["1h"] = train_model(
    "AQHI_future_1h",
    "aqhi_1h"
)

results["2h"] = train_model(
    "AQHI_future_2h",
    "aqhi_2h"
)

results["3h"] = train_model(
    "AQHI_future_3h",
    "aqhi_3h"
)

results["6h"] = train_model(
    "AQHI_future_6h",
    "aqhi_6h"
)

# -----------------------------------
# Summary
# -----------------------------------

summary = pd.DataFrame(results).T

summary.to_csv(
    "models/model_summary.csv"
)

print("\n===================================")
print("MODEL SUMMARY")
print("===================================\n")

print(summary)

print("\nFinished.")

