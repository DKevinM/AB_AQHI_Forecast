# buildABmaster.py

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


def load_parameter_file(filepath, parameter):

    print(f"Loading {filepath}")

    raw = pd.read_csv(
        filepath,
        header=None,
        dtype=str,
        low_memory=False
    )

    # ------------------------------------
    # Metadata rows
    # ------------------------------------
    lats = raw.iloc[0, 1:].values
    lons = raw.iloc[1, 1:].values
    stations = raw.iloc[2, 1:].values

    # ------------------------------------
    # Data section
    # ------------------------------------
    data = raw.iloc[3:].copy()

    cols = ["datetime"] + list(stations)
    data.columns = cols

    # long format
    long = data.melt(
        id_vars="datetime",
        var_name="station",
        value_name=parameter
    )

    # station metadata
    meta = pd.DataFrame({
        "station": stations,
        "lat": pd.to_numeric(lats, errors="coerce"),
        "lon": pd.to_numeric(lons, errors="coerce")
    })

    long = long.merge(
        meta,
        on="station",
        how="left"
    )

    # clean values
    long[parameter] = (
        long[parameter]
        .replace(["S", "", "NA"], np.nan)
    )

    long[parameter] = pd.to_numeric(
        long[parameter],
        errors="coerce"
    )

    long["datetime"] = pd.to_datetime(
        long["datetime"],
        errors="coerce"
    )

    return long


# ------------------------------------
# Load all parameters
# ------------------------------------
dfs = {}

for param, fname in FILES.items():

    dfs[param] = load_parameter_file(
        DATA_DIR / fname,
        param
    )

# ------------------------------------
# Merge all parameters
# ------------------------------------
master = dfs["PM25"]

for param in ["NO2","O3","WS","WD","TEMP","RH"]:

    master = master.merge(
        dfs[param],
        on=["datetime","station","lat","lon"],
        how="outer"
    )

# ------------------------------------
# Sort
# ------------------------------------
master = master.sort_values(
    ["station","datetime"]
)

# ------------------------------------
# Remove completely empty rows
# ------------------------------------
master = master.dropna(
    subset=["PM25","NO2","O3"],
    how="all"
)

# ------------------------------------
# Save
# ------------------------------------
out = "data/AB_master.csv.gz"

master.to_csv(
    out,
    index=False,
    compression="gzip"
)

print(master.head())
print()
print(master.columns)
print()
print("Rows:", len(master))
print("Stations:", master["station"].nunique())
print()
print("Saved:", out)
