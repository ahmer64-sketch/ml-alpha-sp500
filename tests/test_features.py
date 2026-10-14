"""Leakage tests: nothing at date t may depend on data after t (features) or outside t+1..t+1+h (label)."""
import numpy as np
import pandas as pd
import pytest

from alpha import features as F

N_DAYS, N_STOCKS, H = 420, 25, 5


def _market(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=N_DAYS)
    tickers = [f"S{i:02d}" for i in range(N_STOCKS)] + ["SPY"]
    r = rng.normal(0.0003, 0.015, (N_DAYS, len(tickers)))
    adj = pd.DataFrame(50 * np.exp(np.cumsum(r, axis=0)), dates, tickers)
    adj.iloc[:150, 3] = np.nan                                        # a late listing
    close = adj * 1.05
    vol = pd.DataFrame(rng.lognormal(13, 0.5, adj.shape), dates, tickers).where(adj.notna())
    return {"adj": adj, "close": close, "dvol": close * vol}


def _features(p):
    f, el = F.raw_features(p)
    return f, el, F.normalise(f, el)


def _perturb_after(p, t, seed=1):
    rng = np.random.default_rng(seed)
    q = {k: v.copy() for k, v in p.items()}
    shock = np.exp(rng.normal(0, 0.3, q["adj"].iloc[t + 1:].shape))
    for k in q:
        q[k].iloc[t + 1:] = q[k].iloc[t + 1:] * shock
    return q


def _truncate(p, t):
    return {k: v.iloc[: t + 1] for k, v in p.items()}


def _same(a, b):
    return np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


@pytest.mark.parametrize("t", [300, 360, N_DAYS - 2])
def test_features_unchanged_when_future_is_truncated_or_perturbed(t):
    p = _market()
    f, el, z = _features(p)
    for q in (_truncate(p, t), _perturb_after(p, t)):
        f2, el2, z2 = _features(q)
        assert el.iloc[t].equals(el2.iloc[t])
        for k in F.FEATURES:
            assert _same(f[k].iloc[t], f2[k].iloc[t]), k
            assert _same(z[k].iloc[t], z2[k].iloc[t]), k
    assert el.iloc[t].sum() >= N_STOCKS - 1                     # the test covers real values, not NaNs
    assert all(f[k].iloc[t].notna().sum() >= N_STOCKS - 1 for k in F.FEATURES)


def test_a_leaky_feature_is_caught():
    p = _market()
    t = 300
    leaky = lambda q: np.log(q["adj"]).shift(-1) - np.log(q["adj"])     # tomorrow's return
    assert not _same(leaky(p).iloc[t], leaky(_perturb_after(p, t)).iloc[t])


def test_label_uses_only_closes_t_plus_1_and_t_plus_1_plus_h():
    p = _market()
    t = 300
    _, el = F.raw_features(p)
    raw, rk = F.label(p, el, H)
    fwd = np.log(p["adj"].iloc[t + 1 + H] / p["adj"].iloc[t + 1])
    expect = fwd[el.iloc[t]] - fwd[el.iloc[t]].mean()
    assert _same(raw.iloc[t][el.iloc[t]], expect)
    for d in range(N_DAYS):
        q = {k: v.copy() for k, v in p.items()}
        q["adj"].iloc[d, 0] *= 1.3                # one stock (a common shock cancels in the demeaning)
        raw2, _ = F.label(q, el, H)
        changed = not _same(raw.iloc[t], raw2.iloc[t])
        assert changed == (d in (t + 1, t + 1 + H)), d


def test_label_is_missing_at_the_end_of_the_sample():
    p = _market()
    _, el = F.raw_features(p)
    raw, rk = F.label(p, el, H)
    assert raw.iloc[-(H + 1):].isna().all().all() and raw.iloc[-(H + 2)].notna().any()


def test_normalisation_is_per_date_and_in_range():
    p = _market()
    f, el, z = _features(p)
    t = 400
    q = {k: v.copy() for k, v in p.items()}
    q["adj"].iloc[t - 330] *= 2.0              # older than the longest lookback (63-day residuals on 252-day betas)
    _, _, z2 = _features(q)
    for k in F.FEATURES:
        v = z[k].iloc[t].dropna()
        assert v.between(-0.5, 0.5).all() and abs(v.median()) < 0.05
        assert _same(z[k].iloc[t], z2[k].iloc[t]), k


def test_rank_label_matches_raw_order():
    p = _market()
    _, el = F.raw_features(p)
    raw, rk = F.label(p, el, H)
    t = 300
    a, b = raw.iloc[t].dropna(), rk.iloc[t].dropna()
    assert (a.rank() == b.rank()).all() and b.min() == -0.5 and b.max() == 0.5


def test_market_is_never_eligible():
    _, el = F.raw_features(_market())
    assert not el["SPY"].any()


def test_build_keeps_only_eligible_rows_without_missing_features():
    p = _market()
    rows = []
    for k, w in {"adj_close": p["adj"], "close": p["close"], "volume": p["dvol"] / p["close"]}.items():
        rows.append(w.stack().rename(k))
    long = pd.concat(rows, axis=1).rename_axis(["date", "ticker"]).reset_index()
    long[["open", "high", "low"]] = np.nan
    out = F.build(long, H)
    _, el = F.raw_features(F.panel(long))
    assert len(out) == int(el.to_numpy().sum())
    assert out[F.FEATURES].notna().all().all()
    assert "SPY" not in out.index.get_level_values("ticker")
