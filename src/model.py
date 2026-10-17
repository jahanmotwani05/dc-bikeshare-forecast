import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv("data/dataset.csv", parse_dates=["date"], index_col="date")

# lag features: only things you'd actually know the night before
df["lag_1"] = df.total.shift(1)
df["lag_7"] = df.total.shift(7)
df["avg_last_7"] = df.total.shift(1).rolling(7).mean()
df["day_of_year_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365)
df["day_of_year_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365)
df = df.dropna()

features = [
    "temp_max", "temp_min", "rain_in", "snow_in", "wind_max",
    "weekday", "is_weekend", "is_holiday",
    "lag_1", "lag_7", "avg_last_7", "day_of_year_sin", "day_of_year_cos",
]

train = df.loc[:"2026-06-30"]
test = df.loc["2026-07-01":]
X_tr, y_tr = train[features], train.total
X_te, y_te = test[features], test.total


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


results = {}
preds = {}

# baseline 1: same day last week
preds["same day last week"] = test.lag_7
# baseline 2: average of the last 7 days
preds["last 7-day average"] = test.avg_last_7

# model 1: ridge regression
ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)
preds["ridge regression"] = pd.Series(ridge.predict(X_te), index=test.index)

# model 2: gradient boosting
gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0)
gbm.fit(X_tr, y_tr)
preds["gradient boosting"] = pd.Series(gbm.predict(X_te), index=test.index)

for name, p in preds.items():
    results[name] = {"MAE": round(mean_absolute_error(y_te, p)), "MAPE %": round(mape(y_te, p), 1)}

table = pd.DataFrame(results).T.sort_values("MAE")
print(table)
table.to_csv("figures/model_results.csv")

# plot: actual vs best model
best = table.index[0]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(test.index, y_te, label="actual", lw=2)
ax.plot(test.index, preds[best], label=f"predicted ({best})", lw=1.5)
ax.plot(test.index, preds["same day last week"], label="baseline: same day last week", lw=1, alpha=0.5)
ax.set_title("Test period: Jul to Sep 2026")
ax.set_ylabel("rides")
ax.legend()
fig.tight_layout()
fig.savefig("figures/06_forecast_vs_actual.png")

# biggest misses
err = (y_te - preds[best]).abs().sort_values(ascending=False)
print("\nbiggest misses:")
print(pd.DataFrame({"actual": y_te, "predicted": preds[best].round(), "rain_in": test.rain_in}).loc[err.index[:5]])

# which features mattered (ridge coefficients on standardized inputs)
std = X_tr.std()
coef = pd.Series(ridge.coef_ * std, index=features).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
coef.plot(kind="barh", ax=ax)
ax.set_title("Effect on rides of a 1-std change (ridge)")
fig.tight_layout()
fig.savefig("figures/07_feature_effects.png")