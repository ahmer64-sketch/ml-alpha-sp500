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

## Split design

![Walk-forward splits](figures/phase3_splits.png)

```
 one fold (test year Y), on the trading calendar:

 2013-01-02 ............................................ | 11 days | Y-01-02 ........ Y-12-31
 [ inner training ] [5+5 gap] [ inner validation, 12 m ] [ purge  ] [ test year (out of sample) ]
 \________________ training window: final refit on all of it ___/  [+embargo]
```

- **Expanding window, yearly test blocks:** 8 development folds (test years 2015–2022), then 4 hold-out
  folds (2023 to Sep 2026) run once at the end, with the identical, frozen procedure.
- **Purge + embargo:** the label covers 5 days, so a training date is kept only if its label ends at least
  5 more trading days (the embargo) before the test year starts: 11 trading days separate the last
  training date from the first test date. The same gap separates inner training from inner validation.
- **Tuning:** hyperparameters are picked on the last 12 months of each training window, then the model is
  refitted on the whole window. Test years never inform any choice.
- **Hold-out stays clean:** development test labels end by 2022-12-30 (the last 6 trading days of 2022 are
  not tested), so no development result uses a 2023 price.
- Fold-by-fold dates and sample counts: [reports/phase3_splits.md](reports/phase3_splits.md).

## Results

| Phase | Result |
|---|---|
| [1. Data and universe](reports/phase1_data.md) | Snapshot frozen; three data faults found and fixed with a logged cleaning step (stale flags 329 → 0); biases stated. |
| [2. Features and label](reports/phase2_features.md) | 12 pre-registered features (momentum, reversal, 52-week high, volatility, beta, idiosyncratic volatility, volume trend, illiquidity, size proxy), rank-normalised per date; label = 5-day excess return from close t+1, trained as its rank. 1.63M eligible stock-days. Leakage tests pass: features unchanged when the future is removed (unit tests and three real dates), label depends only on closes t+1 and t+6, a planted look-ahead feature is caught. |
| [3. Split design](reports/phase3_splits.md) | 8 development + 4 hold-out walk-forward folds with purge and a 5-day embargo, checked on the real calendar and by unit tests (tight purge, disjoint test years, gap scales with the horizon). Development labels never touch 2023 prices. |
| [4. Models](reports/phase4_models.md) | Ridge, lasso and gradient-boosted trees (primary, fixed in advance) tuned per fold on inner validation; out-of-sample predictions saved for all 939,365 test stock-days 2015–2022 (`predictions/dev_predictions.parquet`), with no gaps. 96 tuning fits logged in `reports/trials.csv`. Validation ICs are small (0.00–0.04), as expected for weekly stock returns. |

## Reproducibility

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python -m pytest
.venv/bin/python scripts/run_phase1.py
.venv/bin/python scripts/run_phase2.py   # also writes data/processed/features_h5.parquet
.venv/bin/python scripts/run_phase3.py
.venv/bin/python scripts/run_phase4.py   # ~5 min
```

Dependencies pinned in `requirements.txt` (Python 3.13).
