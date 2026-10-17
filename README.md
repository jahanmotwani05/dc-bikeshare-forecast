# DC Capital Bikeshare Ridership Forecast

In this project I looked at one year of Capital Bikeshare data in Washington DC. I wanted to understand what makes people ride more or less (weather, weekday vs weekend, season), and I also tried to predict how many rides there will be on a given day.

**Data used:**
- Capital Bikeshare public trip data, Oct 2025 to Sep 2026 (about 6 million rides)
- Daily weather for DC from Open-Meteo (free weather API)

## Why I built this
_(write 2-3 sentences here in your own words)_

## What I found

**1. Season matters the most.**
In January there were about 8,200 rides per day, but in September it was about 22,000 per day. The daily high temperature has a 0.82 correlation with the number of rides, so warmer days clearly mean more riding.

![rides over time](figures/01_rides_over_time.png)

**2. Members and casual riders use bikes differently.**
On weekdays, casual riders are 23.5% of all rides, but on weekends this goes up to 33.7%. On weekdays there are two big peaks, around 8am and 5pm, which is when people go to work and come back. On weekends the rides are spread through the afternoon.

![hourly profile](figures/05_hourly_profile.png)

**3. Heavy rain reduces rides, light rain doesn't really.**
On days with more than 0.5 inch of rain, there were about 13,900 rides on average, compared to about 16,500 on dry days. I was surprised that light rain days had slightly more rides than dry days. I think the reason is that it rains more in summer, and summer already has more riders.

![rain effect](figures/04_rain_effect.png)

**Some other facts:**
- Busiest day: Sep 19, 2026 with 27,667 rides
- Quietest day: Jan 25, 2026 with only 833 rides
- Most popular starting station: Columbus Circle / Union Station

## Prediction model

I tried to predict the total rides for each day using only information you would know the night before: rides from the past days, the day of the week, holidays, and weather.

I trained the models on Oct 2025 to Jun 2026 and tested them on Jul to Sep 2026. I compared two simple baselines with two machine learning models:

| Model | Average error (rides) | Error % |
|---|---|---|
| Ridge regression | 1,959 | 10.5% |
| Gradient boosting | 1,960 | 10.0% |
| Baseline: average of last 7 days | 2,244 | 12.6% |
| Baseline: same day last week | 2,875 | 15.9% |

The models reduced the error by about one third compared to the simple "same day last week" guess. Gradient boosting was not better than ridge regression, so here the simpler model is good enough.

![forecast vs actual](figures/06_forecast_vs_actual.png)

## Limitations
- I used the real weather instead of a weather forecast, so in real life the error would be a bit higher.
- I only used total rain per day. Rain at night and rain at 5pm affect riders very differently, and many of the biggest mistakes were on rainy days.
- I only have one year of data, so the model saw each season only once.
- Special events (concerts, marathons, the July 4 week) are not included.

## How to run
```
pip install -r requirements.txt
python src/download.py        # takes ~5 min, downloads 12 monthly files
python src/build_dataset.py
python src/explore.py
python src/model.py
```

## What I want to add next
- Use hourly weather so rain timing is included
- Predict rides for each station, not just the whole city
- Add more years of data