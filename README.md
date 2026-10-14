# Machine-Learning Alpha Signal (S&P 500)

Cross-sectional prediction of next-period stock returns with linear models and gradient-boosted trees, built
around leakage control: pre-registered design, purged walk-forward splits, placebo and future-perturbation tests,
and a hold-out period locked until one final run. Work in progress; see [DESIGN.md](DESIGN.md).

## Data card

| | |
|---|---|
| **Universe** | Today's S&P 500 members (Wikipedia snapshot of 2026-10-14: 503 tickers; 502 with prices, VYLR has none) plus SPY as the market. |
| **Prices** | Daily open, high, low, close, adjusted close and volume from Yahoo Finance via `yfinance` (`auto_adjust=False`), 2012-01-03 to 2026-09-30, 3,707 trading days, 1,768,517 rows. Fetched 2026-10-14 23:05 UTC. |
| **Snapshot** | `data/raw/` is committed and never edited: `sp500_constituents_20261014.csv`, `prices_20261014T230531Z.parquet` (+ `.json` metadata). Every result is computed from it; `scripts/fetch_data.py` only refreshes it. |
| **Periods** | 2012 feeds lookback features only. Development 2013-01-01 to 2022-12-31 (all model choices). Hold-out 2023-01-01 to 2026-09-30, locked: data-quality checks only until the single final run. |
| **Cleaning** | `alpha.data.clean`, deterministic and logged: drops placeholder history before the current listing (SW before 2024-07, FERG before 2021-03, AMCR before 2019-06, HWM, VRT, GDDY, P), drops the remaining zero-volume days, and repairs one broken adjustment (DHR, Fortive spin-off 2016-07-05: +61% adjusted vs +3.6% actual). |
| **Quality after cleaning** | Coverage after each stock's first price: median 100%, minimum 99.56%. 0 stale-price runs (329 before cleaning). 41 one-day moves beyond ±40% (log), none with a split or adjustment signature after cleaning; they are kept, and include known events such as the 9 March 2020 oil crash (six energy stocks) and the PCG bankruptcy filing. Each is listed in the report for review. |
| **Liquidity** | Median daily dollar volume over the development period: median $143m per stock; 7 of 492 below $10m (lowest ERIE $4.7m). |
| **Known biases** | **Survivorship:** stocks that left the index since 2012 are missing. **Membership look-ahead:** 206 of 503 members (41%) joined after 2013 and are used before they joined, having been added because they grew. Both favour past winners and flatter momentum; no free point-in-time membership list exists, so every result carries this caveat. Yahoo history is revised over time, which is why the snapshot is frozen. |
| **Licence** | Yahoo Finance data is for personal research use; `yfinance` is community-maintained and not affiliated with Yahoo. |

Evidence: [reports/phase1_data.md](reports/phase1_data.md) (every cleaning action, every large move, liquidity),
[figures/phase1_coverage.png](figures/phase1_coverage.png).

## Results

| Phase | Result |
|---|---|
| [1. Data and universe](reports/phase1_data.md) | Snapshot frozen; three data faults found and fixed with a logged cleaning step (stale flags 329 → 0); biases stated. |

## Reproducibility

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python -m pytest
.venv/bin/python scripts/run_phase1.py
```

Dependencies pinned in `requirements.txt` (Python 3.13).
