"""Walk-forward splits with purge and embargo, as pre-registered in DESIGN.md.

Dates are positions on the trading calendar. A training date at index i has a label window ending at
index i + 1 + h; it is kept only if i + 1 + h + embargo < index of the first test date.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    name: str
    train: pd.DatetimeIndex        # purged training dates (whole window, used for the final refit)
    inner_train: pd.DatetimeIndex  # purged dates before the inner validation block
    val: pd.DatetimeIndex          # inner validation block (last 12 months of the training window)
    test: pd.DatetimeIndex


def purged_before(calendar, dates, first_test, h=5, embargo=5):
    """Dates whose label window ends at least `embargo` days before `first_test`."""
    pos = calendar.get_indexer(dates)
    j0 = calendar.searchsorted(first_test)
    return dates[pos + 1 + h + embargo < j0]


def walk_forward(calendar, test_years, train_start, h=5, embargo=5, last_date=None, val_months=12):
    """One Split per calendar year in `test_years`; expanding training window from `train_start`.

    `last_date`: the last close any test label may use (2022-12-31 for development folds).
    """
    calendar = pd.DatetimeIndex(calendar)
    last_date = pd.Timestamp(last_date) if last_date is not None else calendar[-1]
    # a test date needs its whole label window on or before last_date (development labels never touch
    # hold-out prices; hold-out labels never run past the end of the data)
    n_ok = calendar.searchsorted(last_date, side="right")
    labelled = calendar[: max(n_ok - (1 + h), 0)]
    splits = []
    for y in test_years:
        start, end = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        test = labelled[(labelled >= start) & (labelled <= end)]
        first = test[0]
        train = purged_before(calendar, calendar[(calendar >= pd.Timestamp(train_start)) & (calendar < first)],
                              first, h, embargo)
        val_start = train[-1] - pd.DateOffset(months=val_months) + pd.Timedelta(days=1)
        val = train[train >= val_start]
        inner = purged_before(calendar, train[train < val[0]], val[0], h, embargo)
        splits.append(Split(str(y), train, inner, val, test))
    return splits
