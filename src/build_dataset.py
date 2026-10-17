import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar


def load_rides():
    raw = pd.read_csv("data/daily.csv", parse_dates=["date"])
    raw["minute_sum"] = raw["avg_minutes"] * raw["total"]
    d = raw.groupby("date")[["casual", "member", "total", "minute_sum"]].sum()
    d["avg_minutes"] = d["minute_sum"] / d["total"]
    d = d.drop(columns="minute_sum")
    d = d.loc["2025-10-01":"2026-09-30"]
    return d


def load_weather(start, end):
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": 38.9,
            "longitude": -77.04,
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,wind_speed_10m_max",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "wind_speed_unit": "mph",
            "timezone": "America/New_York",
        },
        timeout=60,
    )
    r.raise_for_status()
    w = pd.DataFrame(r.json()["daily"])
    w["date"] = pd.to_datetime(w["time"])
    w = w.drop(columns="time").set_index("date")
    w.columns = ["temp_max", "temp_min", "rain_in", "snow_in", "wind_max"]
    return w


def main():
    rides = load_rides()
    weather = load_weather(rides.index.min().date().isoformat(), rides.index.max().date().isoformat())
    df = rides.join(weather, how="left")

    holidays = USFederalHolidayCalendar().holidays(df.index.min(), df.index.max())
    df["weekday"] = df.index.dayofweek
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_holiday"] = df.index.isin(holidays).astype(int)
    df["month"] = df.index.month
    df["casual_share"] = df["casual"] / df["total"]

    df.to_csv("data/dataset.csv")
    print(df.head())
    print(f"\n{len(df)} days, missing values:\n{df.isna().sum()[df.isna().sum() > 0]}")


if __name__ == "__main__":
    main()