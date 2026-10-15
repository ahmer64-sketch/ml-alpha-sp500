"""Models, grids and the walk-forward fit/predict loop, as pre-registered in DESIGN.md."""
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge

from alpha.features import FEATURES

GBT_BASE = dict(learning_rate=0.05, max_iter=300, l2_regularization=1.0, early_stopping=False, random_state=0)
GRIDS = {
    "ridge": [{"alpha": a} for a in (1.0, 1e3, 1e5, 1e6)],
    "lasso": [{"alpha": a} for a in (1e-6, 1e-5, 1e-4, 1e-3)],
    "gbt": [{"max_leaf_nodes": l, "min_samples_leaf": m} for l, m in itertools.product((7, 31), (200, 2000))],
}
BASELINES = {"momentum": ("mom_12_1", 1.0), "reversal": ("rev_1m", -1.0)}


def make(name, params):
    if name == "ridge":
        return Ridge(**params)
    if name == "lasso":
        return Lasso(max_iter=5000, **params)
    if name == "gbt":
        return HistGradientBoostingRegressor(**GBT_BASE, **params)
    raise ValueError(name)


def daily_ic(pred, y):
    """Per-date Spearman correlation of `pred` and `y` (Series on a (date, ticker) index)."""
    d = pd.DataFrame({"p": pred, "y": y}).dropna()
    r = d.groupby(level="date").rank()
    pc = r["p"] - r.groupby(level="date")["p"].transform("mean")
    yc = r["y"] - r.groupby(level="date")["y"].transform("mean")
    g = pd.DataFrame({"xy": pc * yc, "xx": pc ** 2, "yy": yc ** 2}).groupby(level="date").sum()
    return g["xy"] / np.sqrt(g["xx"] * g["yy"])


def _rows(panel, dates, labelled=True):
    x = panel[panel.index.get_level_values("date").isin(dates)]
    return x[x["y_rank"].notna()] if labelled else x


def run_fold(panel, split, name, log=None):
    """Tune on the inner split, refit on the whole training window, predict the test block.

    Returns (predictions Series on the test rows, chosen params). Appends one row per grid point to `log`.
    """
    inner, val = _rows(panel, split.inner_train), _rows(panel, split.val)
    train, test = _rows(panel, split.train), _rows(panel, split.test, labelled=False)
    best, best_ic = None, -np.inf
    for params in GRIDS[name]:
        t0 = time.perf_counter()
        m = make(name, params).fit(inner[FEATURES].to_numpy(), inner["y_rank"].to_numpy())
        ic = daily_ic(pd.Series(m.predict(val[FEATURES].to_numpy()), val.index), val["y_rank"])
        if log is not None:
            log.append({"fold": split.name, "model": name, "params": str(params), "val_ic_mean": ic.mean(),
                        "val_ic_sd": ic.std(), "val_days": len(ic), "inner_rows": len(inner),
                        "fit_seconds": round(time.perf_counter() - t0, 1)})
        if ic.mean() > best_ic:
            best, best_ic = params, ic.mean()
    m = make(name, best).fit(train[FEATURES].to_numpy(), train["y_rank"].to_numpy())
    assert train.index.get_level_values("date").max() < split.test[0]
    return pd.Series(m.predict(test[FEATURES].to_numpy()), test.index, name=name), best


def baseline(panel, split, name):
    col, sign = BASELINES[name]
    test = _rows(panel, split.test, labelled=False)
    return (sign * test[col]).rename(name)
