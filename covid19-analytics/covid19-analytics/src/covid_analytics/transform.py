"""Pure data-transformation functions (no I/O), so they are easy to unit test."""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

JHU_ID_COLS = ["Province/State", "Country/Region", "Lat", "Long"]


def jhu_wide_to_country_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """JHU wide table (one column per date, provinces split out) -> long, one row per country/day."""
    date_cols = [c for c in df.columns if c not in JHU_ID_COLS]
    by_country = df[["Country/Region"] + date_cols].groupby("Country/Region").sum().copy()
    long = by_country.reset_index().melt(
        id_vars="Country/Region", var_name="date", value_name=value_name
    )
    long["date"] = pd.to_datetime(long["date"], format="%m/%d/%y")
    long = long.rename(columns={"Country/Region": "country"})
    return long.sort_values(["country", "date"]).reset_index(drop=True)


def _daily_from_cumulative(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add daily-new, correction flag and 7-day average for a cumulative column.

    Cumulative series sometimes decrease when a country revises its numbers. Those
    negative daily values are clipped to zero and flagged, not silently kept.
    """
    raw = df.groupby("country")[col].diff()
    raw = raw.fillna(df[col])  # first observation: all cumulative so far is "new"
    new_col = f"new_{col}"
    df[f"{new_col}_correction"] = raw < 0
    df[new_col] = raw.clip(lower=0)
    df[f"{new_col}_7d"] = df.groupby("country")[new_col].transform(
        lambda s: s.rolling(7, min_periods=7).mean()
    )
    return df


def merge_and_derive(confirmed: pd.DataFrame, deaths: pd.DataFrame) -> pd.DataFrame:
    df = confirmed.merge(deaths, on=["country", "date"], validate="one_to_one")
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    df = _daily_from_cumulative(df, "confirmed")
    df = _daily_from_cumulative(df, "deaths")
    return df


def country_iso_map(lookup: pd.DataFrame) -> pd.Series:
    """JHU country name -> ISO3 code, from the JHU UID lookup table."""
    valid = lookup.dropna(subset=["iso3", "Country_Region"])
    return valid.groupby("Country_Region")["iso3"].first()


def add_per_million(df: pd.DataFrame, population: pd.Series) -> pd.DataFrame:
    """Add population-normalised columns. `population` is indexed by iso3."""
    df = df.copy()
    pop = df["iso3"].map(population)
    df["population"] = pop
    for col in ["confirmed", "deaths", "new_confirmed_7d", "new_deaths_7d"]:
        df[f"{col}_per_million"] = df[col] / pop * 1_000_000
    return df


def days_since_threshold(df: pd.DataFrame, col: str = "confirmed", threshold: int = 100) -> pd.DataFrame:
    """Align countries on a common clock: days since `col` first reached `threshold`."""
    first = (
        df[df[col] >= threshold].groupby("country")["date"].min().rename("t0")
    )
    out = df.merge(first, on="country", how="inner")
    out["days_since_t0"] = (out["date"] - out["t0"]).dt.days
    return out[out["days_since_t0"] >= 0].drop(columns="t0")


def detect_waves(values, min_distance_days: int = 45, prominence_frac: float = 0.10) -> np.ndarray:
    """Indices of epidemic waves = prominent peaks in a smoothed daily series."""
    arr = np.nan_to_num(np.asarray(values, dtype=float))
    if arr.size == 0 or arr.max() <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(arr, prominence=prominence_frac * arr.max(), distance=min_distance_days)
    return peaks


def summarize_countries(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per country: totals, normalised totals, CFR and number of waves."""
    last = daily.sort_values("date").groupby("country").tail(1).set_index("country")
    summary = last[
        ["iso3", "population", "confirmed", "deaths", "confirmed_per_million", "deaths_per_million"]
    ].copy()
    summary["cfr_pct"] = np.where(
        summary["confirmed"] >= 1000, 100 * summary["deaths"] / summary["confirmed"], np.nan
    )
    rows = {}
    for country, g in daily.groupby("country"):
        peaks = detect_waves(g["new_confirmed_7d"].to_numpy())
        rows[country] = {
            "n_waves": len(peaks),
            "peak_new_cases_7d": g["new_confirmed_7d"].max(),
        }
    return summary.join(pd.DataFrame.from_dict(rows, orient="index")).reset_index()


def data_quality_report(daily: pd.DataFrame) -> dict:
    return {
        "countries": int(daily["country"].nunique()),
        "first_date": str(daily["date"].min().date()),
        "last_date": str(daily["date"].max().date()),
        "negative_case_corrections": int(daily["new_confirmed_correction"].sum()),
        "negative_death_corrections": int(daily["new_deaths_correction"].sum()),
        "rows_missing_population": int(daily["population"].isna().sum()),
    }
