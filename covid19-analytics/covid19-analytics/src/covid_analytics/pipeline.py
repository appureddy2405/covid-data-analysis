"""End-to-end run: raw files -> processed tables -> figures."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import analysis, plots, transform as tf
from .download import WB_INDICATORS, download_all


def load_indicators(raw_dir: Path) -> dict:
    return {name: pd.read_csv(raw_dir / f"wb_{name}.csv") for name in WB_INDICATORS}


def build_datasets(confirmed_wide, deaths_wide, lookup, indicators: dict):
    """Returns (daily, summary, indicator_table)."""
    confirmed = tf.jhu_wide_to_country_long(confirmed_wide, "confirmed")
    deaths = tf.jhu_wide_to_country_long(deaths_wide, "deaths")
    daily = tf.merge_and_derive(confirmed, deaths)

    daily["iso3"] = daily["country"].map(tf.country_iso_map(lookup))
    daily = daily.dropna(subset=["iso3"])  # drops cruise ships and other non-countries

    population = indicators["population"].set_index("iso3")["value"]
    daily = tf.add_per_million(daily, population)
    summary = tf.summarize_countries(daily)

    indicator_table = pd.concat(
        {name: df.set_index("iso3")["value"] for name, df in indicators.items() if name != "population"},
        axis=1,
    ).reset_index()
    return daily, summary, indicator_table


def run(raw_dir: Path, out_dir: Path, fig_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    daily, summary, indicator_table = build_datasets(
        pd.read_csv(raw_dir / "confirmed_global.csv"),
        pd.read_csv(raw_dir / "deaths_global.csv"),
        pd.read_csv(raw_dir / "uid_lookup.csv"),
        load_indicators(raw_dir),
    )

    corr = analysis.correlate(
        summary, indicator_table, ["confirmed_per_million", "deaths_per_million", "cfr_pct"]
    )
    since_t0 = tf.days_since_threshold(daily)

    daily.to_csv(out_dir / "country_daily.csv", index=False)
    summary.to_csv(out_dir / "country_summary.csv", index=False)
    corr.to_csv(out_dir / "correlations.csv", index=False)
    (out_dir / "data_quality.json").write_text(json.dumps(tf.data_quality_report(daily), indent=2))

    plots.global_timeline(daily, fig_dir / "global_timeline.png")
    plots.top_countries_since_t0(since_t0, summary, fig_dir / "top_countries_deaths_per_million.png")
    plots.correlation_bars(corr, fig_dir / "correlations_deaths_per_million.png")
    print(f"Done. Tables in {out_dir}, figures in {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--fig-dir", default="reports/figures")
    parser.add_argument("--download", action="store_true", help="download raw data first")
    args = parser.parse_args()
    if args.download:
        download_all(Path(args.raw_dir))
    run(Path(args.raw_dir), Path(args.out_dir), Path(args.fig_dir))
