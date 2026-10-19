import json

import numpy as np
import pandas as pd

from covid_analytics import pipeline
from covid_analytics.download import WB_INDICATORS
from tests.helpers import bump, jhu_wide


def _write_synthetic_raw(raw):
    raw.mkdir(parents=True)
    rng = np.random.default_rng(1)
    names = [f"Country{i}" for i in range(40)]
    cases = {n: bump(400, [rng.integers(80, 160), rng.integers(250, 330)], height=rng.integers(200, 3000)) for n in names}
    deaths = {n: v * 0.02 for n, v in cases.items()}
    jhu_wide(cases, provinces={"Country0": ["p1", "p2"]}).to_csv(raw / "confirmed_global.csv", index=False)
    jhu_wide(deaths, provinces={"Country0": ["p1", "p2"]}).to_csv(raw / "deaths_global.csv", index=False)
    lookup = pd.DataFrame({"Country_Region": names + ["Diamond Princess"], "iso3": [f"C{i:02d}" for i in range(40)] + [np.nan]})
    lookup.to_csv(raw / "uid_lookup.csv", index=False)
    iso = [f"C{i:02d}" for i in range(40)]
    for name in WB_INDICATORS:
        vals = rng.uniform(1e6, 1e8, 40) if name == "population" else rng.uniform(10, 90, 40)
        pd.DataFrame({"iso3": iso, "value": vals}).to_csv(raw / f"wb_{name}.csv", index=False)


def test_pipeline_end_to_end_on_synthetic_data(tmp_path):
    raw, out, figs = tmp_path / "raw", tmp_path / "out", tmp_path / "figs"
    _write_synthetic_raw(raw)
    pipeline.run(raw, out, figs)

    daily = pd.read_csv(out / "country_daily.csv")
    summary = pd.read_csv(out / "country_summary.csv")
    assert daily["country"].nunique() == 40
    assert (summary["n_waves"] >= 1).all()
    assert (daily["new_confirmed"] >= 0).all()
    quality = json.loads((out / "data_quality.json").read_text())
    assert quality["countries"] == 40 and quality["rows_missing_population"] == 0
    assert (out / "correlations.csv").exists()
    for fig in ["global_timeline.png", "top_countries_deaths_per_million.png", "correlations_deaths_per_million.png"]:
        assert (figs / fig).stat().st_size > 1000
