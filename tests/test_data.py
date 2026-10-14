import numpy as np
import pandas as pd

from alpha import data


def _long(prices, close=None):
    """Long frame from {ticker: adjusted closes} (NaN = no row); `close` overrides unadjusted closes."""
    dates = pd.bdate_range("2020-01-01", periods=len(next(iter(prices.values()))))
    rows = []
    for t, px in prices.items():
        cl = (close or {}).get(t, px)
        for d, p, c in zip(dates, px, cl):
            if np.isfinite(p):
                rows.append((d, t, c, c, c, c, p, 1000.0))
    return pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])


def test_large_move_flagged_and_split_recognised():
    n = 40
    crash = list(np.linspace(100, 110, n)); crash[20:] = [x * 0.5 for x in crash[20:]]
    # a 2:1 split: unadjusted close halves, adjusted price is smooth
    smooth = list(np.linspace(100, 110, n))
    split_close = smooth[:20] + [x / 2 for x in smooth[20:]]
    df = _long({"CRASH": crash, "SPLIT": smooth, "OK": list(np.linspace(50, 51, n))}, close={"SPLIT": split_close})
    _, issues, _, _ = data.quality_report(df)
    moves = issues[issues["check"] == "large move"]
    assert list(moves["ticker"]) == ["CRASH"]          # the split is adjusted away, the crash is real


def test_stale_and_missing_days_flagged():
    n = 30
    stale = list(np.linspace(10, 12, n)); stale[5:12] = [stale[5]] * 7
    gap = list(np.linspace(20, 22, n)); gap[15:18] = [np.nan] * 3
    df = _long({"STALE": stale, "GAP": gap, "OK": list(np.linspace(1, 2, n))})
    _, issues, cov, _ = data.quality_report(df)
    assert ((issues["ticker"] == "STALE") & (issues["check"] == "stale price")).sum() == 1
    assert ((issues["ticker"] == "GAP") & (issues["check"] == "missing days")).sum() == 1
    assert cov["GAP"] < 1 and cov["OK"] == 1


def test_late_listing_is_not_counted_as_missing():
    n = 30
    late = [np.nan] * 10 + list(np.linspace(5, 6, n - 10))     # IPO part-way through
    df = _long({"IPO": late, "OLD": list(np.linspace(1, 2, n)), "OLD2": list(np.linspace(3, 4, n))})
    _, issues, cov, first = data.quality_report(df)
    assert cov["IPO"] == 1.0 and first["IPO"] == pd.bdate_range("2020-01-01", periods=n)[10]
    assert not ((issues["ticker"] == "IPO") & (issues["check"] == "missing days")).any()


def test_clean_drops_placeholder_history_and_zero_volume_days():
    n = 200
    df = _long({"NEW": list(np.linspace(10, 12, n)), "OK": list(np.linspace(1, 2, n))})
    new = df["ticker"] == "NEW"
    idx = df.index[new]
    df.loc[idx[:25], "volume"] = 0.0          # placeholder history before the real listing
    df.loc[idx[150], "volume"] = 0.0          # one untraded day long after listing
    cleaned, log = data.clean(df)
    kept = cleaned[cleaned["ticker"] == "NEW"]
    assert len(kept) == n - 25 - 1 and kept["date"].min() == df.loc[idx[25], "date"]
    assert (cleaned["volume"] > 0).all()
    assert set(log["action"]) == {"placeholder history dropped", "zero-volume days dropped"}


def test_clean_repairs_broken_adjustment_but_keeps_splits():
    n = 40
    close = list(np.linspace(100, 104, n))
    adj = [c * 0.6 for c in close[:20]] + close[20:]      # adjustment factor jumps, the price does not
    smooth = list(np.linspace(100, 110, n))
    split_close = smooth[:20] + [x / 2 for x in smooth[20:]]
    df = _long({"BROKEN": adj, "SPLIT": smooth}, close={"BROKEN": close, "SPLIT": split_close})
    cleaned, log = data.clean(df)
    b = data.wide(cleaned, "adj_close")["BROKEN"]
    r = np.log(b).diff().abs()
    assert r.max() < 0.01                                  # the fake +67% day is gone
    assert np.isclose(b.iloc[-1], close[-1])               # later prices untouched
    assert list(log["ticker"]) == ["BROKEN"]               # the split is left alone
    s = data.wide(cleaned, "adj_close")["SPLIT"]
    assert np.allclose(s.to_numpy(), smooth)
