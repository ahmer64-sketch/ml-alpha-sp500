# Design decisions, fixed before the results they affect

Recorded 14 Oct 2026, before any data was downloaded or any feature, label or model code existed.
Changing any of these after seeing the results it affects would be data snooping; every later
change is listed under "Changes" with its date and reason.

## Research split
- Data: daily prices and volume from 2012-01-01 (the first year only feeds lookback features).
- **Development period:** 2013-01-01 to 2022-12-31. All feature, model and hyperparameter choices
  are made by walk-forward validation inside this period.
- **Final hold-out (locked):** 2023-01-01 to 2026-09-30. Until Phase 6 it gets data-quality checks
  only, and the final, frozen pipeline runs on it exactly once.

## Universe
- Today's S&P 500 members (Wikipedia snapshot, dated), plus SPY as the market. This has
  survivorship bias (stocks removed from the index before today are missing), which will be stated
  in every result.

## Data cleaning (added 14 Oct 2026, after the Phase 1 quality checks, before any feature or label code)
- All analysis uses `alpha.data.clean` on the frozen snapshot: placeholder pre-listing history and
  zero-volume days dropped, broken adjustments repaired (rules in the docstring; actions logged in
  `reports/phase1_data.md`). These fixes came from data-quality checks, not from any return-based result.

## Timing, label and features (recorded 14 Oct 2026, before any feature, label or model code)
- **Timing:** features for date t use only data up to and including the close of t. Trading is
  assumed at the close of t+1 (one day for computing and acting), never at the close of t.
- **Label horizon h = 5 trading days.** Raw label: log return from close t+1 to close t+1+5, minus
  that day's cross-sectional mean (kept for R² and portfolio evaluation). **Training label:** its
  cross-sectional rank, scaled to [-0.5, 0.5]. h = 1 and h = 21 are robustness checks only (Phase 6).
- **Eligible stocks on date t:** price at t, at least 240 valid daily returns in the past 252 days,
  unadjusted close of at least $5.
- **Features** (all computed from cleaned adjusted closes, unadjusted closes and volume; SPY is the market):
  | Name | Definition | Family |
  |---|---|---|
  | mom_12_1 | log return t-252 to t-21 | momentum |
  | mom_6_1 | log return t-126 to t-21 | momentum |
  | rev_1m | log return t-21 to t | reversal |
  | rev_1w | log return t-5 to t | reversal |
  | high_52w | log(price / max price over the past 252 days) | momentum |
  | vol_1m | std of daily log returns, 21 days | volatility |
  | vol_3m | std of daily log returns, 63 days | volatility |
  | beta_1y | beta of daily returns on SPY, 252 days (min 126) | beta |
  | idio_vol_3m | std of residuals from beta_1y, 63 days | volatility |
  | volume_trend | log(mean dollar volume 5 days / mean dollar volume 63 days) | volume |
  | amihud_1m | log mean of abs(return) / dollar volume, 21 days | illiquidity |
  | size_proxy | log median dollar volume, 252 days | size |
- **Size:** free point-in-time shares outstanding are not available, so size is proxied by dollar
  volume (strongly correlated with market cap in large caps). Stated as a limitation.
- **Normalisation:** per date, each feature is ranked across eligible stocks and scaled to
  [-0.5, 0.5]; a missing value after ranking is set to 0 (the median). No information crosses dates.
- **No feature selection on returns:** all 12 features go into every model; univariate results are
  reported but are not used to drop or add features.
- **Leakage tests (must pass before Phase 3):** recomputing features and labels on data truncated after
  date t (or with all prices after t perturbed) leaves every feature at t unchanged; the label at t
  depends only on closes t+1 and t+1+h; a deliberately leaky feature is caught by the same test.

## Splits (recorded 14 Oct 2026, before any model code or model result)
- **Walk-forward, expanding window, yearly test blocks.** Development folds test on each calendar year
  2015 to 2022 (8 blocks); each trains on every eligible date from 2013-01-01 up to the gap before its
  test year. Models are retrained once a year and never see a date from their own test block.
- **Purge and embargo (h = 5):** a training date t is used only if its label window ends at least 5
  trading days (the embargo, equal to the horizon) before the first test date: index(t) + 1 + h + 5 <
  index(first test date). So 11 trading days separate the last training date from the test block.
  Features need no purge: they use only past data, which the leakage tests guarantee.
- **Hyperparameters:** chosen inside each training window, on an inner validation block = its last
  12 months, separated from the inner training part by the same purge and embargo. The model is then
  refitted on the whole (purged) training window with the chosen setting. Test blocks never inform
  hyperparameters.
- **Hold-out:** the identical procedure, frozen after Phase 6, runs once on 2023, 2024, 2025 and
  2026 (to 30 Sep, the last label ending by then) as yearly blocks, each trained on all earlier data.
- Every training date is used (daily, overlapping labels); IC and portfolio statistics are computed per
  test date. Standard errors in Phase 5 account for the 5-day label overlap.

## Changes after this date
- 14 Oct 2026: added the data-cleaning section after Phase 1 found placeholder histories and one
  broken adjustment in the raw data (decided from quality checks only, no returns examined).
