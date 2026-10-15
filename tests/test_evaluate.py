import numpy as np
import pandas as pd

from alpha import evaluate as E


def test_newey_west_matches_plain_t_for_iid_and_shrinks_for_overlap():
    rng = np.random.default_rng(0)
    x = rng.normal(0.1, 1, 20_000)
    plain = x.mean() / (x.std() / np.sqrt(len(x)))
    assert abs(E.nw_tstat(x, 5) / plain - 1) < 0.05
    overlap = pd.Series(rng.normal(0.02, 1, 20_005)).rolling(5).mean().dropna()   # like 5-day labels
    naive = overlap.mean() / (overlap.std() / np.sqrt(len(overlap)))
    assert E.nw_tstat(overlap, 5) < 0.6 * naive


def _wide(seed=0, n_dates=60, n_stocks=50):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"S{i:02d}" for i in range(n_stocks)]
    fwd = pd.DataFrame(rng.normal(0, 0.03, (n_dates, n_stocks)), dates, tickers)
    return dates, tickers, fwd


def test_perfect_foresight_earns_and_noise_does_not():
    dates, tickers, fwd = _wide()
    cost = pd.DataFrame(0.0, dates, tickers)
    perfect = fwd.stack().rename_axis(["date", "ticker"])
    res = E.long_short(perfect, fwd, cost, dates[::5])
    assert (res["gross"] > 0).all() and (res["n"] == 5).all()
    noise = pd.Series(np.random.default_rng(1).normal(size=len(perfect)), perfect.index)
    res2 = E.long_short(noise, fwd, cost, dates[::5])
    assert abs(res2["gross"].mean()) < 3 * res2["gross"].std() / np.sqrt(len(res2))


def test_costs_follow_turnover_and_a_fixed_book_trades_once():
    dates, tickers, fwd = _wide()
    cost = pd.DataFrame(10.0, dates, tickers)                       # 10 bps everywhere
    fixed = pd.Series(np.tile(np.arange(50.0), len(dates)), fwd.stack().rename_axis(["date", "ticker"]).index)
    res = E.long_short(fixed, fwd, cost, dates[::5])
    assert np.isclose(res["turnover"].iloc[0], 2.0) and np.allclose(res["turnover"].iloc[1:], 0.0)
    assert np.isclose(res["cost"].iloc[0], 2.0 * 10 / 1e4) and np.allclose(res["cost"].iloc[1:], 0.0)
    assert np.allclose(res["net"], res["gross"] - res["cost"] - 0.005 * 5 / 252)
    flip = pd.Series(np.concatenate([np.arange(50.0) * (1 if i % 2 else -1) for i in range(len(dates))]),
                     fwd.stack().rename_axis(["date", "ticker"]).index)
    res2 = E.long_short(flip, fwd, cost, dates)                      # long and short swap every day
    assert np.allclose(res2["turnover"].iloc[1:], 4.0)


def test_cost_model_values():
    assert np.isclose(E.cost_bps(100.0), 6.0) and np.isclose(E.cost_bps(1e9), 3.0, atol=0.01)
    assert E.cost_bps(0.01) == 50.0


def test_r2_and_deciles():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=20), range(100)], names=["date", "ticker"])
    y = pd.Series(np.random.default_rng(2).uniform(-0.5, 0.5, len(idx)), idx)
    assert E.r2_zero(y * 0, y) == 0 and np.isclose(E.r2_zero(y, y), 1)
    assert E.r2_zero(y * 3, y) < 0                                   # overconfident forecasts are punished
    d = E.deciles(y, y)
    assert d.is_monotonic_increasing and len(d) == 10


def test_perf_on_known_series():
    x = pd.Series([0.01, -0.01] * 50 + [0.02])
    p = E.perf(x)
    assert p["max_dd"] == -0.01 and p["sharpe_lo"] < p["sharpe"] < p["sharpe_hi"]


def test_ex_ante_beta_of_the_book():
    dates, tickers, fwd = _wide()
    cost = pd.DataFrame(0.0, dates, tickers)
    beta = pd.DataFrame(np.tile(np.linspace(0.5, 1.5, 50), (len(dates), 1)), dates, tickers)
    pred = pd.Series(np.tile(np.arange(50.0), len(dates)), fwd.stack().rename_axis(["date", "ticker"]).index)
    res = E.long_short(pred, fwd, cost, dates[::5], beta=beta)
    hi, lo = np.linspace(0.5, 1.5, 50)[-5:].mean(), np.linspace(0.5, 1.5, 50)[:5].mean()
    assert np.allclose(res["beta_ex"], hi - lo)                       # long high-beta, short low-beta
