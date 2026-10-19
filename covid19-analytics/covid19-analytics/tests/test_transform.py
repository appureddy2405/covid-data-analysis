import numpy as np
import pandas as pd

from covid_analytics import analysis, transform as tf
from tests.helpers import bump, jhu_wide


def test_provinces_are_aggregated_into_countries():
    wide = jhu_wide({"A": np.full(10, 10.0)}, provinces={"A": ["north", "south"]})
    long = tf.jhu_wide_to_country_long(wide, "confirmed")
    assert len(long) == 10
    assert long["confirmed"].iloc[-1] == 100
    assert long["date"].is_monotonic_increasing


def test_daily_values_and_negative_corrections_are_flagged():
    cumulative = [0, 10, 20, 15, 30]  # revision on day 4 makes the series drop
    conf = pd.DataFrame({"country": "A", "date": pd.date_range("2020-03-01", periods=5), "confirmed": cumulative})
    dead = conf.rename(columns={"confirmed": "deaths"}).assign(deaths=0)
    df = tf.merge_and_derive(conf, dead)
    assert df["new_confirmed"].tolist() == [0, 10, 10, 0, 15]
    assert df["new_confirmed_correction"].tolist() == [False, False, False, True, False]


def test_seven_day_average_needs_full_window():
    conf = pd.DataFrame({"country": "A", "date": pd.date_range("2020-03-01", periods=10),
                         "confirmed": np.arange(10) * 7})
    dead = conf.rename(columns={"confirmed": "deaths"}).assign(deaths=0)
    df = tf.merge_and_derive(conf, dead)
    assert df["new_confirmed_7d"].iloc[:6].isna().all()
    assert df["new_confirmed_7d"].iloc[6] > 0


def test_days_since_threshold_starts_at_zero_per_country():
    dates = pd.date_range("2020-03-01", periods=6)
    df = pd.concat([
        pd.DataFrame({"country": "A", "date": dates, "confirmed": [0, 50, 100, 150, 200, 250]}),
        pd.DataFrame({"country": "B", "date": dates, "confirmed": [100, 110, 120, 130, 140, 150]}),
    ])
    out = tf.days_since_threshold(df, "confirmed", 100)
    assert out.groupby("country")["days_since_t0"].min().tolist() == [0, 0]
    assert out[out["country"] == "A"]["date"].min() == dates[2]


def test_detect_waves_finds_two_distinct_peaks():
    series = bump(500, centers=[100, 350])
    assert len(tf.detect_waves(series)) == 2


def test_detect_waves_ignores_flat_and_empty_series():
    assert len(tf.detect_waves(np.zeros(100))) == 0
    assert len(tf.detect_waves([])) == 0


def test_country_iso_map_drops_missing_codes():
    lookup = pd.DataFrame({"Country_Region": ["A", "A", "Diamond Princess"], "iso3": ["AAA", "AAA", np.nan]})
    m = tf.country_iso_map(lookup)
    assert m.to_dict() == {"A": "AAA"}


def test_benjamini_hochberg_is_monotone_and_bounded():
    adj = analysis.benjamini_hochberg([0.001, 0.04, 0.03, 0.5])
    assert (adj >= np.array([0.001, 0.04, 0.03, 0.5])).all()
    assert adj.max() <= 1.0
    order = np.argsort([0.001, 0.04, 0.03, 0.5])
    assert (np.diff(adj[order]) >= -1e-12).all()


def test_correlate_recovers_a_planted_relationship():
    rng = np.random.default_rng(0)
    n = 80
    x = rng.normal(size=n)
    summary = pd.DataFrame({"iso3": [f"C{i}" for i in range(n)], "y": 2 * x + rng.normal(scale=0.3, size=n)})
    indicators = pd.DataFrame({"iso3": summary["iso3"], "x": x, "noise": rng.normal(size=n)})
    out = analysis.correlate(summary, indicators, ["y"])
    top = out.iloc[0]
    assert top["indicator"] == "x" and top["rho"] > 0.9 and top["p_adjusted"] < 0.001
