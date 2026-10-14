"""Features and label, exactly as pre-registered in DESIGN.md.

Everything works on wide frames (dates x tickers) on the SPY trading calendar. A feature for date t
uses data up to and including the close of t; the label for t is the log return from close t+1 to
close t+1+h, so trading is assumed one day after the signal.
"""
import numpy as np
import pandas as pd

from alpha import data

MARKET = "SPY"
FEATURES = ["mom_12_1", "mom_6_1", "rev_1m", "rev_1w", "high_52w", "vol_1m", "vol_3m", "beta_1y",
            "idio_vol_3m", "volume_trend", "amihud_1m", "size_proxy"]


def panel(df):
    """Wide adjusted close, unadjusted close and dollar volume on the market's calendar."""
    adj = data.wide(df, "adj_close")
    cal = adj.index[adj[MARKET].notna()]
    close = data.wide(df, "close").reindex(index=cal, columns=adj.columns)
    vol = data.wide(df, "volume").reindex(index=cal, columns=adj.columns)
    return {"adj": adj.loc[cal], "close": close, "dvol": close * vol}


def _rolling_beta(r, m, window=252, min_obs=126):
    """Pairwise-complete rolling beta of each column of r on the market series m."""
    mask = r.notna()
    mm = mask.mul(m, axis=0).where(mask)               # market return on days the stock has one
    roll = lambda x: x.rolling(window, min_periods=min_obs).mean()
    cov = roll(r * mm) - roll(r) * roll(mm)
    var = roll(mm ** 2) - roll(mm) ** 2
    return cov / var


def raw_features(p):
    """Dict of raw (un-normalised) features plus the eligibility mask, for every stock and date."""
    adj, close, dvol = p["adj"], p["close"], p["dvol"]
    la = np.log(adj)
    r = la.diff()
    m = r[MARKET]
    beta = _rolling_beta(r, m)
    resid = r - beta.mul(m, axis=0)
    f = {
        "mom_12_1": la.shift(21) - la.shift(252),
        "mom_6_1": la.shift(21) - la.shift(126),
        "rev_1m": la - la.shift(21),
        "rev_1w": la - la.shift(5),
        "high_52w": la - la.rolling(252, min_periods=200).max(),
        "vol_1m": r.rolling(21, min_periods=17).std(),
        "vol_3m": r.rolling(63, min_periods=50).std(),
        "beta_1y": beta,
        "idio_vol_3m": resid.rolling(63, min_periods=50).std(),
        "volume_trend": np.log(dvol.rolling(5, min_periods=4).mean() / dvol.rolling(63, min_periods=50).mean()),
        "amihud_1m": np.log((r.abs() / dvol).rolling(21, min_periods=15).mean() * 1e6),
        "size_proxy": np.log(dvol.rolling(252, min_periods=126).median()),
    }
    eligible = adj.notna() & (r.rolling(252, min_periods=1).count() >= 240) & (close >= 5)
    eligible[MARKET] = False
    return f, eligible


def label(p, eligible, h=5):
    """(raw label, rank label): excess log return close t+1 -> t+1+h, and its cross-sectional rank."""
    la = np.log(p["adj"])
    fwd = (la.shift(-(1 + h)) - la.shift(-1)).where(eligible)
    raw = fwd.sub(fwd.mean(axis=1), axis=0)
    return raw, cs_rank(raw)


def cs_rank(x):
    """Per-date rank scaled to [-0.5, 0.5]; NaN stays NaN."""
    rk = x.rank(axis=1)
    return (rk - 1).div(x.count(axis=1) - 1, axis=0) - 0.5


def normalise(f, eligible):
    """Rank each feature across eligible stocks per date; missing after ranking -> 0 (the median)."""
    return {k: cs_rank(v.where(eligible)).where(eligible).fillna(0).where(eligible) for k, v in f.items()}


def build(df, h=5):
    """Long frame (date, ticker): normalised features, raw and rank label, for eligible stock-dates."""
    p = panel(df)
    f, eligible = raw_features(p)
    z = normalise(f, eligible)
    raw, rk = label(p, eligible, h)
    keep = eligible.stack()
    keep = keep[keep].index                            # eligible (date, ticker) pairs only
    out = pd.DataFrame({k: z[k].stack().reindex(keep) for k in FEATURES})
    out["y_raw"] = raw.stack().reindex(keep)
    out["y_rank"] = rk.stack().reindex(keep)
    out.index.names = ["date", "ticker"]
    return out.sort_index()
