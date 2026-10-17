import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv("data/dataset.csv", parse_dates=["date"], index_col="date")
hourly = pd.read_csv("data/hourly.csv", parse_dates=["date"])
stations = pd.read_csv("data/stations.csv")
hourly = hourly.groupby(["date", "hour"], as_index=False)["rides"].sum()

# 1. rides over the year
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df.total, alpha=0.3, label="daily")
ax.plot(df.index, df.total.rolling(7, center=True).mean(), lw=2, label="7-day average")
ax.set_title("Capital Bikeshare daily rides, Oct 2025 to Sep 2026")
ax.set_ylabel("rides")
ax.legend()
fig.tight_layout()
fig.savefig("figures/01_rides_over_time.png")

# 2. members vs casual by weekday
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_day = df.groupby("weekday")[["member", "casual"]].mean()
fig, ax = plt.subplots(figsize=(7, 4))
by_day.plot(kind="bar", ax=ax)
ax.set_xticklabels(days, rotation=0)
ax.set_title("Average rides by day of week")
ax.set_xlabel("")
ax.set_ylabel("rides per day")
fig.tight_layout()
fig.savefig("figures/02_weekday_member_casual.png")

# 3. temperature vs rides
fig, ax = plt.subplots(figsize=(7, 4.5))
for flag, label in [(0, "weekday"), (1, "weekend/holiday")]:
    sub = df[(df.is_weekend | df.is_holiday) == flag]
    ax.scatter(sub.temp_max, sub.total, s=12, alpha=0.6, label=label)
ax.set_xlabel("daily high (°F)")
ax.set_ylabel("rides")
ax.set_title("Warmer days, more rides")
ax.legend()
fig.tight_layout()
fig.savefig("figures/03_temp_vs_rides.png")

# 4. rain effect
df["rain_bucket"] = pd.cut(df.rain_in, [-0.01, 0, 0.1, 0.5, 10], labels=["dry", "light", "moderate", "heavy"])
rain = df.groupby("rain_bucket", observed=True).total.mean()
fig, ax = plt.subplots(figsize=(6, 4))
rain.plot(kind="bar", ax=ax, color="steelblue")
ax.set_xticklabels(rain.index, rotation=0)
ax.set_title("Average rides by rainfall")
ax.set_xlabel("")
ax.set_ylabel("rides per day")
fig.tight_layout()
fig.savefig("figures/04_rain_effect.png")

# 5. hourly pattern, weekday vs weekend
hourly["weekend"] = hourly.date.dt.dayofweek >= 5
prof = hourly.groupby(["weekend", "hour"]).rides.mean().unstack(0)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(prof.index, prof[False], marker="o", label="weekday")
ax.plot(prof.index, prof[True], marker="o", label="weekend")
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("hour of day")
ax.set_ylabel("average rides")
ax.set_title("Commute peaks on weekdays, afternoon hump on weekends")
ax.legend()
fig.tight_layout()
fig.savefig("figures/05_hourly_profile.png")

# key numbers
print("busiest day:", df.total.idxmax().date(), int(df.total.max()))
print("quietest day:", df.total.idxmin().date(), int(df.total.min()))
print("avg rides/day by month:\n", df.groupby("month").total.mean().round(0))
print("\nweekday vs weekend casual share:", df.groupby("is_weekend").casual_share.mean().round(3).to_dict())
print("rain effect:\n", rain.round(0))
print("correlation temp_max vs rides:", round(df.temp_max.corr(df.total), 3))
print("\ntop 5 start stations:\n", stations.head(5).to_string(index=False))