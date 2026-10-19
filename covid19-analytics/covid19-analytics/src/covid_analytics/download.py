"""Download raw data: JHU CSSE case counts and World Bank indicators.

Note: the JHU CSSE repository stopped updating on 2023-03-10, so the data covers
the pandemic period from 2020-01-22 to 2023-03-09.
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

JHU_BASE = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/"
)
JHU_FILES = {
    "confirmed_global.csv": JHU_BASE + "csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
    "deaths_global.csv": JHU_BASE + "csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
    "uid_lookup.csv": JHU_BASE + "UID_ISO_FIPS_LookUp_Table.csv",
}

WB_INDICATORS = {
    "population": "SP.POP.TOTL",
    "gdp_per_capita_ppp": "NY.GDP.PCAP.PP.CD",
    "urban_pop_pct": "SP.URB.TOTL.IN.ZS",
    "rural_pop_pct": "SP.RUR.TOTL.ZS",
    "life_expectancy": "SP.DYN.LE00.IN",
    "health_expenditure_pct_gdp": "SH.XPD.CHEX.GD.ZS",
}


def _get(url: str, **kwargs) -> requests.Response:
    resp = requests.get(url, timeout=60, **kwargs)
    resp.raise_for_status()
    return resp


def fetch_world_bank(code: str) -> pd.DataFrame:
    """Most recent non-empty value of an indicator for every country (iso3, value)."""
    resp = _get(
        f"https://api.worldbank.org/v2/country/all/indicator/{code}",
        params={"format": "json", "per_page": 20000, "mrnev": 1},
    )
    _, rows = resp.json()
    return pd.DataFrame(
        [{"iso3": r["countryiso3code"], "value": r["value"]} for r in rows if r["countryiso3code"]]
    )


def download_all(raw_dir: Path, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in JHU_FILES.items():
        target = raw_dir / name
        if target.exists() and not force:
            print(f"cached: {name}")
            continue
        print(f"downloading {name}")
        target.write_bytes(_get(url).content)
    for name, code in WB_INDICATORS.items():
        target = raw_dir / f"wb_{name}.csv"
        if target.exists() and not force:
            print(f"cached: wb_{name}.csv")
            continue
        print(f"downloading World Bank {code}")
        fetch_world_bank(code).to_csv(target, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--force", action="store_true", help="re-download cached files")
    args = parser.parse_args()
    download_all(Path(args.raw_dir), args.force)
