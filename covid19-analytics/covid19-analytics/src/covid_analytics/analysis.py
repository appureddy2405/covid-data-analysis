"""Association analysis between country outcomes and socioeconomic indicators."""
import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false-discovery-rate control)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p downwards
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


def correlate(summary: pd.DataFrame, indicators: pd.DataFrame, targets, min_n: int = 30) -> pd.DataFrame:
    """Spearman rank correlation of each target with each indicator (pairwise complete)."""
    data = summary.merge(indicators, on="iso3", how="left")
    rows = []
    for target in targets:
        for indicator in indicators.columns.drop("iso3"):
            pair = data[[target, indicator]].dropna()
            if len(pair) < min_n:
                continue
            rho, p = stats.spearmanr(pair[target], pair[indicator])
            rows.append({"target": target, "indicator": indicator, "rho": rho, "p_value": p, "n": len(pair)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adjusted"] = benjamini_hochberg(out["p_value"])
    return out.sort_values(["target", "rho"], ascending=[True, False]).reset_index(drop=True)
