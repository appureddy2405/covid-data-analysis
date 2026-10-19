"""Figures saved to reports/figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def global_timeline(daily, out: Path) -> None:
    g = daily.groupby("date")[["new_confirmed", "new_deaths"]].sum()
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(g.index, g["new_confirmed"].rolling(7).mean(), color="tab:blue")
    ax[0].set_ylabel("New cases (7-day avg)")
    ax[1].plot(g.index, g["new_deaths"].rolling(7).mean(), color="tab:red")
    ax[1].set_ylabel("New deaths (7-day avg)")
    ax[0].set_title("Global daily COVID-19 cases and deaths")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def top_countries_since_t0(since_t0, summary, out: Path, n: int = 10) -> None:
    top = summary.dropna(subset=["deaths_per_million"]).nlargest(n, "deaths_per_million")["country"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for country in top:
        g = since_t0[since_t0["country"] == country]
        ax.plot(g["days_since_t0"], g["deaths_per_million"], label=country)
    ax.set_xlabel("Days since 100th confirmed case")
    ax.set_ylabel("Cumulative deaths per million")
    ax.set_title(f"Top {n} countries by deaths per million")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def correlation_bars(corr, out: Path, target: str = "deaths_per_million") -> None:
    c = corr[corr["target"] == target].sort_values("rho")
    if c.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["tab:green" if p < 0.05 else "lightgray" for p in c["p_adjusted"]]
    ax.barh(c["indicator"], c["rho"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman rho (green = FDR-adjusted p < 0.05)")
    ax.set_title(f"{target} vs country indicators")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
