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

## Changes after this date
(none)
