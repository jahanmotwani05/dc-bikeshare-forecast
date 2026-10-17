import io
import zipfile

import pandas as pd
import requests

URL = "https://s3.amazonaws.com/capitalbikeshare-data/{ym}-capitalbikeshare-tripdata.zip"
MONTHS = pd.period_range("2025-10", "2026-09", freq="M")
COLS = ["started_at", "ended_at", "rideable_type", "member_casual", "start_station_name"]


def load_month(ym):
    r = requests.get(URL.format(ym=ym), timeout=120)
    if r.status_code != 200:
        print(f"  {ym}: not found ({r.status_code}), skipping")
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=COLS, parse_dates=["started_at", "ended_at"])
    print(f"  {ym}: {len(df):,} rides")
    return df


def main():
    daily, hourly, stations = [], [], []
    for p in MONTHS:
        df = load_month(p.strftime("%Y%m"))
        if df is None:
            continue
        df["minutes"] = (df.ended_at - df.started_at).dt.total_seconds() / 60
        df = df[(df.minutes > 1) & (df.minutes < 24 * 60)]
        df["date"] = df.started_at.dt.date
        df["hour"] = df.started_at.dt.hour

        d = df.groupby(["date", "member_casual"]).size().unstack(fill_value=0)
        d["total"] = d.sum(axis=1)
        d["avg_minutes"] = df.groupby("date").minutes.mean()
        daily.append(d)

        hourly.append(df.groupby(["date", "hour"]).size().rename("rides"))
        stations.append(df.groupby("start_station_name").size().rename("rides"))

    daily = pd.concat(daily).sort_index()
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    daily.to_csv("data/daily.csv")

    pd.concat(hourly).to_csv("data/hourly.csv")
    pd.concat(stations).groupby(level=0).sum().sort_values(ascending=False).to_csv("data/stations.csv")

    print(f"\nsaved {len(daily)} days, {int(daily.total.sum()):,} rides total")


if __name__ == "__main__":
    main()