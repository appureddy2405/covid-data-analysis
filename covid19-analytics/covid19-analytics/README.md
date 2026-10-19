# COVID-19 Country Analytics Pipeline

A reproducible, tested pipeline that turns raw COVID-19 case data and World Bank indicators into
country-level metrics, wave detection and socioeconomic association analysis.

**Author:** Siva Kumar Reddy Kamuri

## Questions this project answers

1. How did cases and deaths evolve globally, and when did each country's waves occur?
2. After adjusting for population, which countries were hit hardest, and how do trajectories compare when
   aligned on the same clock (days since the 100th case)?
3. Are national outcomes associated with wealth, urbanisation, life expectancy or health spending,
   and which of those associations survive multiple-testing correction?

## Pipeline

```
JHU CSSE time series ─┐
JHU UID/ISO lookup ───┼─► download.py ─► transform.py ─► analysis.py ─► plots.py
World Bank API ───────┘     (cached)      (pure functions)  (Spearman + FDR)   (figures)
                                                 │
                                                 └─► data/processed/*.csv, data_quality.json
```

| Step | What happens |
|---|---|
| Aggregate | Province-level rows are summed to countries (China, Canada, Australia and others are split in the raw data). |
| Daily values | Cumulative counts become daily new cases/deaths. Negative values caused by data revisions are clipped to zero and **flagged**, with the total reported in `data_quality.json`. |
| Normalise | Per-million values use World Bank population, joined by ISO3 code (non-countries such as cruise ships are dropped). |
| Align | `days_since_t0` aligns countries on the day they reached 100 confirmed cases. |
| Waves | Prominent peaks of the 7-day average (minimum 45 days apart, at least 10% of the series maximum) are counted as waves. |
| Associate | Spearman rank correlations against World Bank indicators, with Benjamini–Hochberg FDR adjustment. |

## Run it

```bash
pip install -r requirements.txt
python -m covid_analytics.pipeline --download      # downloads raw data, then runs everything
# or, once data/raw is cached:
python -m covid_analytics.pipeline
pytest -q
```

Outputs: `data/processed/` (country_daily, country_summary, correlations, data_quality.json) and
`reports/figures/`.

## Data

- JHU CSSE COVID-19 repository (cases, deaths, UID lookup). **The repository stopped updating on
  2023-03-10**, so the analysis covers 2020-01-22 to 2023-03-09.
- World Bank indicators (most recent available value per country): population, GDP per capita (PPP),
  urban and rural population share, life expectancy, health expenditure (% of GDP).

## Limitations to keep in mind

- Confirmed cases depend heavily on testing capacity and reporting practices, so deaths per million is the
  more comparable outcome, and even that is affected by under-recording of deaths.
- Country-level correlations are ecological: they describe countries, not individuals, and don't show causation.
- World Bank indicators use the latest available year, not the year of each wave.

## Testing

Unit tests cover aggregation, correction flagging, rolling windows, wave detection, ISO mapping, FDR
adjustment and a planted-correlation check. An end-to-end test runs the full pipeline, including figure
output, on synthetic data in JHU format, so CI needs no network access.

## Findings

_Run the pipeline and summarise your own results here (top countries by deaths per million, which indicators
stayed significant after FDR adjustment, how many waves different regions saw)._

## Acknowledgements

Inspired by the COVID-19 analysis in [shsarv/Data-Analytics-Projects-in-python](https://github.com/shsarv/Data-Analytics-Projects-in-python)
(MIT licensed), used here with the author's permission. This repository is a separate implementation with a
different structure and added features: updated data window, testable pure functions, correction flagging,
wave detection, FDR-adjusted statistics and CI.
