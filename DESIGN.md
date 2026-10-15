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
- **No label crosses into the hold-out:** a development test date is used only if its whole label
  window ends by 2022-12-31 (the last 6 trading days of 2022 are not tested); hold-out test dates
  need their label window to end by 2026-09-30.
- Every training date is used (daily, overlapping labels); IC and portfolio statistics are computed per
  test date. Standard errors in Phase 5 account for the 5-day label overlap.

## Models (recorded 14 Oct 2026, before any model code or model result)
- **Input:** the 12 rank-normalised features; **target:** the rank label (y_rank). One pooled model
  across all stocks and dates in the training window (no per-stock or per-sector models).
- **Candidates and grids** (every grid point x fold is logged to `reports/trials.csv`):
  | Model | Grid (chosen per fold by inner-validation IC) |
  |---|---|
  | ridge | alpha in {1, 1e3, 1e5, 1e6} |
  | lasso | alpha in {1e-6, 1e-5, 1e-4, 1e-3} |
  | gbt (sklearn HistGradientBoostingRegressor; LightGBM-equivalent, no libomp dependency) | learning_rate 0.05, 300 trees, l2 1.0, max_leaf_nodes in {7, 31} x min_samples_leaf in {200, 2000}, no early stopping, random_state 0 |
- **Tuning metric:** mean daily rank IC (Spearman correlation of prediction and label across stocks,
  per date) on the inner validation block.
- **Baselines (no fitting):** momentum = mom_12_1, reversal = -rev_1m, both as ranks.
- **Primary model, fixed now:** gbt. Ridge is the linear benchmark; lasso and baselines are reported.
  All are run on the hold-out and all are reported; the headline is gbt regardless of which looks best
  in development.
- **Output:** out-of-sample predictions for every development test date and every model, saved with the
  fold, the chosen hyperparameters and the training window.

## Evaluation (recorded 15 Oct 2026, before any test-year metric was computed)
Development test years 2015-2022 only; the hold-out uses the identical code once, after Phase 6.
- **Signals scored:** ridge, lasso, gbt (primary), momentum and reversal baselines (5 in total).
- **IC:** daily Spearman correlation of prediction and 5-day label across stocks. Reported: mean,
  t-stat with Newey-West standard errors (5 lags, for the overlapping labels), IC-IR = mean / sd
  of daily IC, and the share of years with a positive mean. Significance: Bonferroni over the 5 signals.
- **Out-of-sample R²** of the rank label against a zero forecast: 1 - sum((y - pred)²) / sum(y²),
  for the three fitted models (the baselines are not on the label's scale).
- **Deciles:** mean 5-day excess return per prediction decile, and top-minus-bottom spread.
- **Portfolio:** every 5th trading day the signal of date t is traded at the close of t+1: long the
  top decile, short the bottom decile, equal weights, $1 long and $1 short. Held for 5 days (simple
  returns, close t+1 to close t+6). Five versions starting on consecutive days are run and averaged
  (equivalent to a staggered book), so results do not depend on the start day. Weight drift within a
  week is ignored when computing turnover.
- **Costs:** one-way cost per stock = 3 bps + 30 bps / sqrt(ADV in $m), capped at 50 bps, where ADV is
  the 63-day median dollar volume known at t (about 5.5 bps at the median stock, 16 bps at $5m a day).
  Cost = sum over stocks of |weight change| x cost. Short borrow fee 0.5% a year on the short book.
- **Reported per portfolio:** annual return and volatility, Sharpe (gross and net) with a 95% interval
  (Lo 2002 standard error), maximum drawdown, turnover per rebalance, beta to SPY.
- **Signal decay:** IC of each prediction against 5-day returns starting 0, 5, 10 and 20 days later.

## Robustness checks (recorded 15 Oct 2026, before running any of them)
All on the development period. Pass/fail rules are fixed here; every result is reported, including failures.
1. **Placebo (shuffled labels):** in every training window the rank label is shuffled across stocks
   within each date; gbt and ridge are refitted with the hyperparameters Phase 4 chose for that fold and
   scored on the real test labels. 3 seeds. **Pass:** each run's mean test IC is within 2 Newey-West
   standard errors of 0.
2. **Future perturbation (whole pipeline):** prices of every stock after T = 2018-06-29 are multiplied by
   random noise; features, labels and the 2018 fold (tuning included) are recomputed. **Pass:** predictions
   for test dates up to T are identical to the unperturbed run (max difference < 1e-9), and predictions
   after T change (the check is sensitive).
3. **Sensitivity** (no pass/fail; reported in full):
   - label horizon h = 1 and h = 21 (embargo = h, rebalance every h days, staggered books), gbt and ridge;
   - each hyperparameter setting held fixed for all folds (no tuning), gbt and ridge;
   - costs x0, x0.5, x1, x2; portfolio of top/bottom 10% vs 20%;
   - universe restricted to stocks in the index since 2012 or earlier (less membership look-ahead).
4. **Importance:** for gbt (Phase 4 settings per fold), the drop in test IC when one feature family is
   shuffled across stocks within each date: momentum (mom_12_1, mom_6_1, high_52w), reversal (rev_1m,
   rev_1w), volatility (vol_1m, vol_3m, idio_vol_3m), beta (beta_1y), volume (volume_trend), size and
   liquidity (size_proxy, amihud_1m). Ridge coefficients per fold are also reported.
5. **Freeze:** after Phase 6 the pipeline is frozen at its commit; the hold-out uses the pre-registered
   settings (h = 5, Phase 4 grids and tuning, Phase 5 portfolio and costs), whatever the sensitivity runs show.

## Changes after this date
- 14 Oct 2026: added the data-cleaning section after Phase 1 found placeholder histories and one
  broken adjustment in the raw data (decided from quality checks only, no returns examined).
- 15 Oct 2026: after seeing the Phase 5 betas (+0.5 for the model books), added a **diagnostic**
  SPY-hedged version (ex-ante beta from beta_1y, known at t) and net returns by year. These are extra
  rows; no pre-registered number was changed or dropped. The hold-out will report both.
- 15 Oct 2026: after Phase 6, an interpretation section was added to its report, and the hold-out report
  will also show the members-since-2012 split (Phase 6 check 3, reused). No pass/fail rule changed; two
  placebo runs failed and are reported as failures.
- 15 Oct 2026: after the first hold-out run (2026-10-15 08:07 UTC), an interpretation section splitting
  the hold-out by the date each stock joined the index was added to its report. The script was rerun
  only to reproduce the same predictions; no pre-registered number changed.
