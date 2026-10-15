"""Signal and portfolio evaluation, as pre-registered in DESIGN.md (Evaluation section)."""
import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252 / 5


def nw_tstat(x, lags=5):
    """t-statistic of the mean with Newey-West (Bartlett) standard errors."""
    x = np.asarray(pd.Series(x).dropna(), float)
    n, e = len(x), x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(var / n)


def ic_summary(ic):
    years = ic.groupby(ic.index.year).mean()
    return {"ic_mean": ic.mean(), "ic_sd": ic.std(), "ic_ir": ic.mean() / ic.std(), "t_nw": nw_tstat(ic),
            "days": len(ic), "years_positive": f"{int((years > 0).sum())}/{len(years)}"}


def r2_zero(pred, y):
    d = pd.DataFrame({"p": pred, "y": y}).dropna()
    return 1 - ((d["y"] - d["p"]) ** 2).sum() / (d["y"] ** 2).sum()


def deciles(pred, y, n=10):
    """Mean of y per prediction decile (averaged over dates), decile 1 = lowest prediction."""
    d = pd.DataFrame({"p": pred, "y": y}).dropna()
    pct = d.groupby(level="date")["p"].rank(pct=True, method="first")
    d["q"] = np.ceil(pct * n).clip(1, n).astype(int)
    return d.groupby([d.index.get_level_values("date"), "q"])["y"].mean().groupby("q").mean()


def cost_bps(adv_musd, base=3.0, scale=30.0, cap=50.0):
    """One-way trading cost in basis points from average daily dollar volume in $m."""
    return np.minimum(base + scale / np.sqrt(adv_musd), cap)


def long_short(pred, fwd, cost, dates, frac=0.1, borrow=0.005, h=5, beta=None):
    """Equal-weight top-minus-bottom `frac` portfolio rebalanced on `dates`.

    pred: Series on (date, ticker). fwd: wide simple return held from close t+1 to close t+1+h, indexed by
    the signal date t. cost: wide one-way cost in bps, known at t. beta (optional): wide ex-ante beta known at t,
    giving the book's beta at formation (`beta_ex`). Returns one row per rebalance date.
    """
    rows, w_old = [], pd.Series(dtype=float)
    for t in dates:
        p = pred.xs(t, level="date").dropna()
        n = int(np.floor(len(p) * frac))
        order = p.rank(method="first")
        longs, shorts = order.index[order > len(p) - n], order.index[order <= n]
        w = pd.concat([pd.Series(1.0 / n, longs), pd.Series(-1.0 / n, shorts)])
        r = fwd.loc[t].reindex(w.index)
        missing = int(r.isna().sum())
        gross = float((w * r.fillna(0.0)).sum())
        dw = w.reindex(w.index.union(w_old.index), fill_value=0.0) - w_old.reindex(w.index.union(w_old.index),
                                                                                    fill_value=0.0)
        c = cost.loc[t].reindex(dw.index)
        c = c.fillna(c.max() if c.notna().any() else 50.0)
        tc = float((dw.abs() * c).sum()) / 1e4
        bf = borrow * h / 252
        rows.append({"date": t, "gross": gross, "cost": tc, "borrow": bf, "net": gross - tc - bf,
                     "turnover": float(dw.abs().sum()), "long_ret": float(r[longs].fillna(0).mean()),
                     "short_ret": float(r[shorts].fillna(0).mean()), "n": n, "missing": missing,
                     "beta_ex": float((w * beta.loc[t].reindex(w.index).fillna(1.0)).sum()) if beta is not None
                     else np.nan})
        w_old = w
    return pd.DataFrame(rows).set_index("date")


def perf(x, ppy=PERIODS_PER_YEAR):
    """Annualised mean, vol, Sharpe with Lo (2002) 95% interval, and max drawdown (cumulative sum)."""
    x = pd.Series(x).dropna()
    sr_p = x.mean() / x.std()
    se = np.sqrt((1 + 0.5 * sr_p ** 2) / len(x)) * np.sqrt(ppy)
    sr = sr_p * np.sqrt(ppy)
    cum = x.cumsum()
    return {"ann_return": x.mean() * ppy, "ann_vol": x.std() * np.sqrt(ppy), "sharpe": sr,
            "sharpe_lo": sr - 1.96 * se, "sharpe_hi": sr + 1.96 * se, "max_dd": float((cum - cum.cummax()).min())}


def forward_returns(adj, start_lag, horizon, last_end, log=False):
    """Return from close t+1+start_lag to close t+1+start_lag+horizon (wide, indexed by t).

    NaN when that window ends after `last_end`, so development results never use later prices.
    """
    cal = adj.index
    a, b = adj.shift(-(1 + start_lag)), adj.shift(-(1 + start_lag + horizon))
    r = np.log(b / a) if log else b / a - 1
    end = pd.Series(cal, cal).shift(-(1 + start_lag + horizon))
    ok = (end <= pd.Timestamp(last_end)).to_numpy()
    return r.where(np.broadcast_to(ok[:, None], r.shape))


def staggered(pred, fwd, cost, dates, h, beta, spy_fwd, frac=0.1, cost_mult=1.0):
    """Run h books started on consecutive days; return (summary dict averaged over books, all rows)."""
    rows, stats = [], []
    for k in range(h):
        r = long_short(pred, fwd, cost * cost_mult, dates[k::h], frac=frac, h=h, beta=beta)
        r["hedged"] = r["net"] - r["beta_ex"] * spy_fwd.reindex(r.index).fillna(0.0)
        r["offset"] = k
        rows.append(r)
        ppy = 252 / h
        n, hd, g = perf(r["net"], ppy), perf(r["hedged"], ppy), perf(r["gross"], ppy)
        stats.append({"gross_sharpe": g["sharpe"], "net_sharpe": n["sharpe"], "net_lo": n["sharpe_lo"],
                      "net_hi": n["sharpe_hi"], "net_ret": n["ann_return"], "hedged_sharpe": hd["sharpe"],
                      "hedged_lo": hd["sharpe_lo"], "hedged_hi": hd["sharpe_hi"],
                      "turnover": r["turnover"].iloc[1:].mean()})
    return pd.DataFrame(stats).mean().to_dict(), pd.concat(rows)
