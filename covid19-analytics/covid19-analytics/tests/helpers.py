"""Synthetic JHU-style data for tests (no network needed)."""
import numpy as np
import pandas as pd


def bump(n_days, centers, width=15, height=1000):
    x = np.arange(n_days)
    return sum(height * np.exp(-((x - c) ** 2) / (2 * width**2)) for c in centers)


def jhu_wide(country_series: dict, start="2020-01-22", provinces: dict | None = None) -> pd.DataFrame:
    """country_series: name -> daily-new array. `provinces` splits a country in two rows."""
    provinces = provinces or {}
    rows = []
    for country, daily in country_series.items():
        cumulative = np.cumsum(daily).round().astype(int)
        dates = pd.date_range(start, periods=len(cumulative))
        labels = [d.strftime("%-m/%-d/%y") for d in dates]
        if country in provinces:
            half = cumulative // 2
            for prov, vals in zip(provinces[country], [half, cumulative - half]):
                rows.append({"Province/State": prov, "Country/Region": country, "Lat": 0.0, "Long": 0.0,
                             **dict(zip(labels, vals))})
        else:
            rows.append({"Province/State": np.nan, "Country/Region": country, "Lat": 0.0, "Long": 0.0,
                         **dict(zip(labels, cumulative))})
    return pd.DataFrame(rows)
