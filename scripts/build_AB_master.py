# scripts/build_AB_master.py

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

FILES = {
    "PM25": "PM25_c.csv",
    "NO2":  "NO2_c.csv",
    "O3":   "O3_c.csv",
    "WS":   "WS_c.csv",
    "WD":   "WD_c.csv",
    "TEMP": "ET_c.csv",
    "RH":   "RH_c.csv"
}

STATIONS_FILE = DATA_DIR / "stations.csv"


def load_parameter_file(filepath, parameter, stations_meta):

    print(f"Loading {filepath}")

    df = pd.read_csv(
        filepath,
        dtype=str,
        low_memory=False
    )

    # standardize first column as datetime
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "datetime"})

    # long format
    long = df.melt(
        id_vars="datetime",
        var_name="station",
        value_name=parameter
    )

    long["station"] = long["station"].astype(str).str.strip()

    # clean values
    long[parameter] = (
        long[parameter]
        .replace(["S", "s", "", "NA", "N/A", "nan", "NaN", "null"], np.nan)
    )

    long[parameter] = pd.to_numeric(
        long[parameter],
        errors="coerce"
    )

    long["datetime"] = pd.to_datetime(
        long["datetime"],
        errors="coerce"
    )

    # merge station metadata
    long = long.merge(
        stations_meta,
        on="station",
        how="left"
    )

    # warn if station metadata missing
    missing_meta = long.loc[
        long["lat"].isna() | long["lon"].isna(),
        "station"
    ].dropna().unique()

    if len(missing_meta) > 0:
        print(f"WARNING: Missing lat/lon for {parameter}:")
        print(missing_meta)

    return long


# ------------------------------------
# Load station metadata
# ------------------------------------
stations_meta = pd.read_csv(STATIONS_FILE)

stations_meta.columns = [
    c.strip().lower() for c in stations_meta.columns
]

stations_meta = stations_meta.rename(columns={
    "latitude": "lat",
    "longitude": "lon",
    "name": "station",
    "stationname": "station"
})

stations_meta["station"] = stations_meta["station"].astype(str).str.strip()
stations_meta["lat"] = pd.to_numeric(stations_meta["lat"], errors="coerce")
stations_meta["lon"] = pd.to_numeric(stations_meta["lon"], errors="coerce")

stations_meta = stations_meta[["station", "lat", "lon"]].drop_duplicates()


# ------------------------------------
# Load all parameters
# ------------------------------------
dfs = {}

for param, fname in FILES.items():
    dfs[param] = load_parameter_file(
        DATA_DIR / fname,
        param,
        stations_meta
    )


# ------------------------------------
# Merge all parameters
# ------------------------------------
master = dfs["PM25"]

for param in ["NO2", "O3", "WS", "WD", "TEMP", "RH"]:

    master = master.merge(
        dfs[param],
        on=["datetime", "station", "lat", "lon"],
        how="outer"
    )


# ------------------------------------
# Sort and clean
# ------------------------------------
master = master.sort_values(["station", "datetime"])

master = master.dropna(subset=["datetime", "station"])

# remove rows with no AQHI pollutant data
master = master.dropna(
    subset=["PM25", "NO2", "O3"],
    how="all"
)


# ------------------------------------
# Save
# ------------------------------------
out = DATA_DIR / "AB_master.csv.gz"

master.to_csv(
    out,
    index=False,
    compression="gzip"
)


# ------------------------------------
# Diagnostics
# ------------------------------------
print(master.head())
print()
print(master.columns)
print()
print("Rows:", len(master))
print("Stations:", master["station"].nunique())
print()
print("Missing values:")
print(master.isna().sum())
print()
print("AQHI pollutant coverage by station:")
print(
    master.groupby("station")[["PM25", "NO2", "O3"]]
    .count()
    .sort_values("PM25", ascending=False)
    .head(20)
)
print()
print("Saved:", out)
