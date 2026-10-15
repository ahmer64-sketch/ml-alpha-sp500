import numpy as np
import pandas as pd
import pytest

from alpha import models as M
from alpha.features import FEATURES
from alpha.splits import walk_forward

CAL = pd.bdate_range("2013-01-01", "2017-12-29")


def _panel(signal=0.3, seed=0, n=60):
    """Synthetic panel: the label depends on mom_12_1 (linear) and on vol_1m (non-linear, |x|)."""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([CAL, [f"S{i}" for i in range(n)]], names=["date", "ticker"])
    x = pd.DataFrame(rng.uniform(-0.5, 0.5, (len(idx), len(FEATURES))), idx, FEATURES)
    y = signal * x["mom_12_1"] + signal * (x["vol_1m"].abs() - 0.25) + rng.normal(0, 0.3, len(idx))
    x["y_raw"] = y
    x["y_rank"] = y.groupby(level="date").rank(pct=True) - 0.5
    return x


@pytest.fixture(scope="module")
def split():
    return walk_forward(CAL, [2017], "2013-01-01", 5, 5)[0]


def test_daily_ic_matches_spearman():
    p = _panel()
    d = p.loc[CAL[10]]
    ic = M.daily_ic(p["mom_12_1"], p["y_rank"])
    assert np.isclose(ic.loc[CAL[10]], d["mom_12_1"].corr(d["y_rank"], method="spearman"))


@pytest.mark.parametrize("name", ["ridge", "lasso", "gbt"])
def test_models_find_a_planted_signal(name, split):
    p = _panel()
    log = []
    pred, best = M.run_fold(p, split, name, log)
    assert pred.index.get_level_values("date").isin(split.test).all() and len(pred) == len(split.test) * 60
    ic = M.daily_ic(pred, p["y_rank"]).mean()
    assert ic > 0.25 and len(log) == len(M.GRIDS[name]) and best in M.GRIDS[name]


def test_trees_find_the_non_linear_part_linear_models_cannot(split):
    p = _panel()
    nl = p["vol_1m"].abs()
    lin, _ = M.run_fold(p, split, "ridge")
    gbt, _ = M.run_fold(p, split, "gbt")
    assert abs(M.daily_ic(lin, nl).mean()) < 0.1 < M.daily_ic(gbt, nl).mean()


def test_no_signal_gives_no_ic(split):
    p = _panel(signal=0.0)
    for name in ("ridge", "gbt"):
        pred, _ = M.run_fold(p, split, name)
        ic = M.daily_ic(pred, p["y_rank"])
        assert abs(ic.mean()) < 3 * ic.std() / np.sqrt(len(ic)) + 0.01


def test_test_labels_are_never_used_for_fitting(split):
    p = _panel()
    a, _ = M.run_fold(p, split, "gbt")
    q = p.copy()
    test_rows = q.index.get_level_values("date").isin(split.test)
    q.loc[test_rows, "y_rank"] = np.random.default_rng(9).permutation(q.loc[test_rows, "y_rank"].to_numpy())
    b, _ = M.run_fold(q, split, "gbt")
    assert np.allclose(a, b)                                  # scrambling test labels changes nothing


def test_baselines_are_signed_features(split):
    p = _panel()
    assert np.allclose(M.baseline(p, split, "reversal"), -p.loc[p.index.get_level_values("date").isin(split.test), "rev_1m"])
