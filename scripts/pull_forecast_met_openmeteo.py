import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# -----------------------------------
# CONFIG
# -----------------------------------

MASTER_FILE = Path("data/AB_master.csv.gz")
OUTDIR = Path("data/forecast_features")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUTDIR / "ab_station_forecast_met.csv.gz"

FORECAST_HOURS = 24

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
]


# -----------------------------------
# Helpers
# -----------------------------------

def wind_to_uv(speed, direction_deg):
    """
    Convert wind speed/direction to U/V components.
    Meteorological wind direction is direction wind comes from.
    """
    rad = np.deg2rad(direction_deg)

    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)

    return u, v


def get_station_list(master_file):
    print(f"Loading stations from: {master_file}")

    df = pd.read_csv(
        master_file,
        compression="gzip",
        usecols=["station", "lat", "lon"]
    )

    stations = (
        df.dropna(subset=["station", "lat", "lon"])
          .drop_duplicates(subset=["station"])
          .sort_values("station")
          .reset_index(drop=True)
    )

    print("Stations found:", len(stations))
    return stations


def fetch_openmeteo_forecast(station, lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_hours": FORECAST_HOURS,
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }

    r = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=60
    )

    r.raise_for_status()
    js = r.json()

    hourly = js.get("hourly", {})

    if not hourly or "time" not in hourly:
        raise ValueError(f"No hourly forecast returned for {station}")

    out = pd.DataFrame({
        "datetime": hourly["time"],
        "station": station,
        "lat": lat,
        "lon": lon,
        "TEMP_fcst": hourly.get("temperature_2m"),
        "RH_fcst": hourly.get("relative_humidity_2m"),
        "WS_fcst": hourly.get("wind_speed_10m"),
        "WD_fcst": hourly.get("wind_direction_10m"),
    })

    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)

    out["U_fcst"], out["V_fcst"] = wind_to_uv(
        out["WS_fcst"],
        out["WD_fcst"]
    )

    out["forecast_source"] = "open-meteo"
    out["pulled_at_utc"] = pd.Timestamp.utcnow()

    return out


# -----------------------------------
# Main
# -----------------------------------

def main():
    stations = get_station_list(MASTER_FILE)

    all_rows = []

    for i, row in stations.iterrows():
        station = row["station"]
        lat = row["lat"]
        lon = row["lon"]

        print(f"[{i + 1}/{len(stations)}] Fetching forecast: {station}")

        try:
            fcst = fetch_openmeteo_forecast(station, lat, lon)
            all_rows.append(fcst)

        except Exception as e:
            print(f"WARNING: failed for {station}: {e}")

        time.sleep(0.25)

    if not all_rows:
        raise RuntimeError("No forecast data pulled.")

    out = pd.concat(all_rows, ignore_index=True)

    out = out.sort_values(["station", "datetime"]).reset_index(drop=True)

    print("\nForecast rows:", len(out))
    print("Stations:", out["station"].nunique())
    print("Datetime min:", out["datetime"].min())
    print("Datetime max:", out["datetime"].max())

    missing = out.isna().sum()
    print("\nMissing values:")
    print(missing)

    out.to_csv(
        OUTFILE,
        index=False,
        compression="gzip"
    )

    print("\nSaved:", OUTFILE)


if __name__ == "__main__":
    main()
