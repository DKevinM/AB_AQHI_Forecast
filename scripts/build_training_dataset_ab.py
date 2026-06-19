import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, asin, sqrt

DATA_IN = Path("data/AB_master.csv.gz")
DATA_OUT = Path("/tmp/training_dataset.csv.gz")
NEIGHBOR_OUT = Path("data/station_neighbors.csv")

FILL_COLS = ["PM25", "NO2", "O3", "WS", "TEMP", "RH"]
NO_SPATIAL_FILL = ["WD"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * R * asin(sqrt(a))


print("Loading AB master...")
data = pd.read_csv(DATA_IN)
data["datetime"] = pd.to_datetime(data["datetime"])
data = data.sort_values(["station", "datetime"])

# ------------------------------------------------
# Create fill flags before filling
# ------------------------------------------------
for col in FILL_COLS + NO_SPATIAL_FILL:
    data[f"{col}_filled"] = data[col].isna().astype(int)

# ------------------------------------------------
# Stage 1: short-gap interpolation within station
# ------------------------------------------------
print("Stage 1: short-gap interpolation...")

for col in FILL_COLS + NO_SPATIAL_FILL:
    data[col] = (
        data.groupby("station")[col]
        .transform(lambda x: x.interpolate(limit=3, limit_direction="both"))
    )

# ------------------------------------------------
# Build nearest-neighbour table
# ------------------------------------------------
print("Building station neighbour table...")

stations = (
    data[["station", "lat", "lon"]]
    .drop_duplicates()
    .dropna()
    .reset_index(drop=True)
)

neighbors = []

for _, s in stations.iterrows():
    dists = []

    for _, t in stations.iterrows():
        if s["station"] == t["station"]:
            continue

        d = haversine_km(
            s["lat"], s["lon"],
            t["lat"], t["lon"]
        )

        dists.append({
            "station": s["station"],
            "neighbor": t["station"],
            "distance_km": d
        })

    dists = sorted(dists, key=lambda x: x["distance_km"])[:2]
    neighbors.extend(dists)

neighbors = pd.DataFrame(neighbors)
neighbors.to_csv(NEIGHBOR_OUT, index=False)

print("Saved:", NEIGHBOR_OUT)



# ------------------------------------------------
# RH donor fill for stations with no RH
# ------------------------------------------------

print("\nRH coverage by station:")

rh_counts = (
    data.groupby("station")["RH"]
    .count()
    .sort_values()
)

print(rh_counts.head(20))

zero_rh_stations = rh_counts[rh_counts == 0].index.tolist()

print("\nStations with zero RH:")
print(zero_rh_stations)

valid_rh_stations = set(
    rh_counts[rh_counts > 0].index
)

for station in zero_rh_stations:

    station_neighbors = (
        neighbors[
            (neighbors["station"] == station) &
            (neighbors["neighbor"].isin(valid_rh_stations))
        ]
        .sort_values("distance_km")
    )

    if station_neighbors.empty:
        print(f"No RH donor found for {station}")
        continue

    donor = station_neighbors.iloc[0]["neighbor"]

    print(
        f"Filling RH for {station} "
        f"from {donor}"
    )

    donor_rh = (
        data[data["station"] == donor]
        [["datetime", "RH"]]
        .rename(columns={"RH": "RH_donor"})
    )

    mask = data["station"] == station

    tmp = (
        data.loc[mask]
        .merge(
            donor_rh,
            on="datetime",
            how="left"
        )
    )

    data.loc[mask, "RH"] = tmp["RH_donor"]

    data.loc[mask, "RH_filled"] = np.where(
        tmp["RH_donor"].notna(),
        1,
        data.loc[mask, "RH_filled"]
    )



# ------------------------------------------------
# Stage 2: spatial fill from two nearest stations
# ------------------------------------------------
print("Stage 2: spatial fill from nearest stations...")

for col in FILL_COLS:

    print(f"Spatial filling {col}...")

    wide = data.pivot_table(
        index="datetime",
        columns="station",
        values=col,
        aggfunc="mean"
    )

    for station in stations["station"]:

        if station not in wide.columns:
            continue

        station_neighbors = (
            neighbors[neighbors["station"] == station]
            .sort_values("distance_km")
        )

        if station_neighbors.empty:
            continue

        fill_series = None
        weight_sum = 0

        for _, n in station_neighbors.iterrows():

            nb = n["neighbor"]
            dist = n["distance_km"]

            if nb not in wide.columns:
                continue

            weight = 1 / max(dist, 1)

            if fill_series is None:
                fill_series = wide[nb] * weight
            else:
                fill_series = fill_series + wide[nb] * weight

            weight_sum += weight

        if fill_series is None or weight_sum == 0:
            continue

        fill_series = fill_series / weight_sum

        missing = wide[station].isna()
        wide.loc[missing, station] = fill_series.loc[missing]

    long_fill = (
        wide.reset_index()
        .melt(
            id_vars="datetime",
            var_name="station",
            value_name=f"{col}_spatial_filled"
        )
    )

    data = data.merge(
        long_fill,
        on=["datetime", "station"],
        how="left"
    )

    still_missing = data[col].isna()

    data.loc[still_missing, col] = data.loc[
        still_missing,
        f"{col}_spatial_filled"
    ]

    data = data.drop(columns=[f"{col}_spatial_filled"])




# ------------------------------------------------
# Wind vector features
# ------------------------------------------------
print("Building wind features...")

rad = np.deg2rad(data["WD"])

data["U"] = -data["WS"] * np.sin(rad)
data["V"] = -data["WS"] * np.cos(rad)

data["dlat"] = data["lat"] - data["lat"].mean()
data["dlon"] = data["lon"] - data["lon"].mean()

data["transport_index"] = data["U"] * data["dlat"] + data["V"] * data["dlon"]

data["lat_norm"] = (data["lat"] - data["lat"].mean()) / data["lat"].std()
data["lon_norm"] = (data["lon"] - data["lon"].mean()) / data["lon"].std()

data["dist_center"] = np.sqrt(
    (data["lat"] - data["lat"].mean()) ** 2 +
    (data["lon"] - data["lon"].mean()) ** 2
)

# ------------------------------------------------
# Time features
# ------------------------------------------------
data["doy"] = data["datetime"].dt.dayofyear
data["hour"] = data["datetime"].dt.hour

data["sin_doy"] = np.sin(2 * np.pi * data["doy"] / 365)
data["cos_doy"] = np.cos(2 * np.pi * data["doy"] / 365)

data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24)
data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24)

# ------------------------------------------------
# Rolling 3-hour pollutant averages
# ------------------------------------------------
print("Calculating 3-hour rolling averages...")

data = data.sort_values(["station", "datetime"])
data = data.set_index("datetime")

for col in ["PM25", "NO2", "O3"]:
    data[f"{col}_3hr"] = (
        data.groupby("station")[col]
        .rolling("3h", min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )

# ------------------------------------------------
# AQHI calculation
# ------------------------------------------------
print("Calculating AQHI...")

data["AQHI"] = (
    (1000 / 10.4) *
    (
        (np.exp(0.000537 * data["O3_3hr"]) - 1) +
        (np.exp(0.000871 * data["NO2_3hr"]) - 1) +
        (np.exp(0.000487 * data["PM25_3hr"]) - 1)
    )
)

data["AQHI"] = data["AQHI"].clip(lower=1, upper=11)
data["AQHI"] = data["AQHI"].round(1)

# ------------------------------------------------
# Future targets
# ------------------------------------------------
print("Creating future targets...")

for h in [1, 2, 3, 6]:
    data[f"AQHI_future_{h}h"] = (
        data.groupby("station")["AQHI"]
        .shift(-h)
    )

# ------------------------------------------------
# AQHI lags
# ------------------------------------------------
print("Creating AQHI lags...")

for lag in [1, 2, 3, 6, 12, 24]:
    data[f"AQHI_lag{lag}"] = (
        data.groupby("station")["AQHI"]
        .shift(lag)
    )

data["AQHI_change_1h"] = data["AQHI"] - data["AQHI_lag1"]
data["AQHI_change_3h"] = data["AQHI"] - data["AQHI_lag3"]



# ------------------------------------------------
# Future meteorology proxy features
# ------------------------------------------------
# These are observed future meteorology values used as a proxy
# for forecast meteorology during model development.
# Later, live forecasting will replace these with actual forecast met.

print("Creating future meteorology proxy features...")

future_met_cols = ["WS", "WD", "TEMP", "RH", "U", "V"]

for h in [1, 2, 3, 6]:
    for col in future_met_cols:
        data[f"{col}_future_{h}h"] = (
            data.groupby("station")[col]
            .shift(-h)
        )




data = data.reset_index()

# ------------------------------------------------
# Final cleanup
# ------------------------------------------------
required = [
    "AQHI",
    "AQHI_future_1h",
    "AQHI_future_2h",
    "AQHI_future_3h",
    "AQHI_future_6h",

    "AQHI_lag1",
    "AQHI_lag2",
    "AQHI_lag3",
    "AQHI_lag6",
    "AQHI_lag12",
    "AQHI_lag24",

    "AQHI_change_1h",
    "AQHI_change_3h",

    "PM25",
    "NO2",
    "O3",

    "WS",
    "WD",
    "TEMP",
    "RH",
    "U",
    "V",

    "WS_future_1h",
    "WD_future_1h",
    "TEMP_future_1h",
    "RH_future_1h",
    "U_future_1h",
    "V_future_1h",

    "WS_future_2h",
    "WD_future_2h",
    "TEMP_future_2h",
    "RH_future_2h",
    "U_future_2h",
    "V_future_2h",

    "WS_future_3h",
    "WD_future_3h",
    "TEMP_future_3h",
    "RH_future_3h",
    "U_future_3h",
    "V_future_3h",

    "WS_future_6h",
    "WD_future_6h",
    "TEMP_future_6h",
    "RH_future_6h",
    "U_future_6h",
    "V_future_6h",

    "lat_norm",
    "lon_norm",
    "dist_center",

    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy"
]

before = len(data)


print("\nStations before final drop:")
print(data["station"].nunique())

station_counts_before = (
    data.groupby("station")
    .size()
    .sort_values()
)

print(station_counts_before.head(20))


tmp = data.dropna(subset=required)

lost_stations = (
    set(data["station"].unique()) -
    set(tmp["station"].unique())
)

print("\nLost stations after final drop:")
print(sorted(lost_stations))

data = tmp


after = len(data)

print("Rows before final drop:", before)
print("Rows after final drop :", after)
print("Rows retained %       :", round(after / before * 100, 1))

# ------------------------------------------------
# Save
# ------------------------------------------------
data.to_csv(
    DATA_OUT,
    index=False,
    compression="gzip"
)

print("Saved:", DATA_OUT)
print("Final rows:", len(data))
print("Stations:", data["station"].nunique())
