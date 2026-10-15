import numpy as np
import pandas as pd

from alpha import evaluate as E, robust as R
from alpha.features import FEATURES


def _x(n_dates=30, n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=n_dates), range(n)], names=["date", "ticker"])
    return pd.DataFrame(rng.uniform(-0.5, 0.5, (len(idx), len(FEATURES))), idx, FEATURES)


def test_shuffle_keeps_each_dates_values_but_breaks_the_link():
    x = _x()
    s = R.shuffle_within_date(x["mom_12_1"], 1)
    for d, g in s.groupby(level="date"):
        assert sorted(g) == sorted(x["mom_12_1"].xs(d, level="date"))
    assert abs(np.corrcoef(s, x["mom_12_1"])[0, 1]) < 0.1


def test_permute_family_moves_columns_jointly_and_leaves_others():
    x = _x()
    y = R.permute_family(x, ["rev_1m", "rev_1w"], 2)
    assert np.allclose(y.drop(columns=["rev_1m", "rev_1w"]), x.drop(columns=["rev_1m", "rev_1w"]))
    pairs_x = set(map(tuple, x[["rev_1m", "rev_1w"]].round(12).to_numpy()))
    pairs_y = set(map(tuple, y[["rev_1m", "rev_1w"]].round(12).to_numpy()))
    assert pairs_x == pairs_y and not np.allclose(y["rev_1m"], x["rev_1m"])


def test_families_cover_every_feature_once():
    cols = [c for v in R.FAMILIES.values() for c in v]
    assert sorted(cols) == sorted(FEATURES)


def test_forward_returns_stop_at_the_cutoff():
    dates = pd.bdate_range("2022-12-01", "2023-01-31")
    adj = pd.DataFrame({"A": np.linspace(10, 20, len(dates)), "B": np.linspace(5, 6, len(dates))}, dates)
    f = E.forward_returns(adj, 0, 5, "2022-12-31")
    ends = [dates[i + 6] for i in range(len(dates) - 6)]
    for i, e in enumerate(ends):
        assert np.isnan(f["A"].iloc[i]) == np.isnan(f["B"].iloc[i]) == (e > pd.Timestamp("2022-12-31"))
    assert np.isclose(f["A"].iloc[0], adj["A"].iloc[6] / adj["A"].iloc[1] - 1)
