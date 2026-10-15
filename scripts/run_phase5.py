"""Phase 5: evaluation of the development out-of-sample predictions (2015-2022), net of costs.

Run: .venv/bin/python scripts/run_phase5.py   (needs predictions/dev_predictions.parquet from Phase 4)
Writes reports/phase5_evaluation.md, reports/phase5_portfolios.csv and figures/phase5_*.png.
No forward return used here ends after 2022-12-31 (the hold-out stays untouched).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from alpha import data, evaluate as E, features as F, models as M

ROOT = Path(__file__).resolve().parents[1]
DEV_END = pd.Timestamp("2022-12-31")
SIGNALS = ["gbt", "ridge", "lasso", "momentum", "reversal"]
FITTED = ["gbt", "ridge", "lasso"]
H = 5
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


pred = pd.read_parquet(ROOT / "predictions/dev_predictions.parquet")
raw, _ = data.load_prices(sorted((ROOT / "data/raw").glob("prices_*.parquet"))[-1])
df, _ = data.clean(raw)
p = F.panel(df)
cal = p["adj"].index


def forward(start_lag, horizon=H, log=False):
    """Return from close t+1+start_lag to close t+1+start_lag+horizon, NaN if it ends after DEV_END."""
    a, b = p["adj"].shift(-(1 + start_lag)), p["adj"].shift(-(1 + start_lag + horizon))
    r = np.log(b / a) if log else b / a - 1
    end = pd.Series(cal, cal).shift(-(1 + start_lag + horizon))
    return r[(end <= DEV_END).to_numpy()].reindex(cal)


fwd = forward(0)
spy_fwd = fwd[F.MARKET]
adv = p["dvol"].rolling(63, min_periods=20).median() / 1e6
cost = E.cost_bps(adv)
test_dates = pred.index.get_level_values("date").unique().sort_values()
beta_ex = F.raw_features(p)[0]["beta_1y"]

out("# Phase 5: evaluation (development test years 2015-2022)\n")
out(f"Out-of-sample predictions from Phase 4 for {len(test_dates):,} test dates "
    f"({test_dates[0]:%Y-%m-%d} to {test_dates[-1]:%Y-%m-%d}), {len(pred):,} stock-days. Metrics, portfolio rules and "
    "costs were pre-registered in [DESIGN.md](../DESIGN.md) before any of these numbers were computed. Hold-out "
    "(2023+) not used.\n")

# 1. IC
ics = {s: M.daily_ic(pred[s], pred["y_raw"]) for s in SIGNALS}
out("## 1. Information coefficient\n")
out("Daily Spearman correlation between the signal and the next 5-day excess return. t-statistics use Newey-West "
    "standard errors (5 lags, labels overlap); p-values are Bonferroni-adjusted for the 5 signals.\n")
out("| Signal | Mean IC | IC sd | IC-IR | t (NW) | p (Bonferroni) | Years positive | OOS R² (rank label) |")
out("|---|---|---|---|---|---|---|---|")
summ = {}
for s in SIGNALS:
    st = E.ic_summary(ics[s])
    pval = min(1.0, 2 * (1 - stats.norm.cdf(abs(st["t_nw"]))) * len(SIGNALS))
    r2 = E.r2_zero(pred[s], pred["y_rank"]) if s in FITTED else np.nan
    summ[s] = {**st, "p_bonf": pval, "r2": r2}
    out(f"| {s}{' (primary)' if s == 'gbt' else ''} | {st['ic_mean']:+.4f} | {st['ic_sd']:.3f} | {st['ic_ir']:+.3f} | "
        f"{st['t_nw']:+.2f} | {pval:.3f} | {st['years_positive']} | " + (f"{r2:+.5f}" if s in FITTED else "n/a") + " |")
best_fit = max(FITTED, key=lambda s: summ[s]["ic_mean"])
out("")
yr = pd.DataFrame({s: ics[s].groupby(ics[s].index.year).mean() for s in SIGNALS})
out("Mean IC by test year:\n")
out("| Year | " + " | ".join(SIGNALS) + " |")
out("|---|" + "---|" * len(SIGNALS))
for y, r in yr.iterrows():
    out(f"| {y} | " + " | ".join(f"{v:+.4f}" for v in r) + " |")
out("")

# 2. deciles
out("## 2. Decile returns\n")
dec = pd.DataFrame({s: E.deciles(pred[s], pred["y_raw"]) for s in SIGNALS}) * 1e4
out("Mean 5-day excess log return by prediction decile, in basis points (1 = lowest prediction):\n")
out("| Decile | " + " | ".join(SIGNALS) + " |")
out("|---|" + "---|" * len(SIGNALS))
for q, r in dec.iterrows():
    out(f"| {q} | " + " | ".join(f"{v:+.1f}" for v in r) + " |")
spread = dec.loc[10] - dec.loc[1]
mono = {s: stats.spearmanr(dec.index, dec[s])[0] for s in SIGNALS}
out("| **10 − 1** | " + " | ".join(f"**{spread[s]:+.1f}**" for s in SIGNALS) + " |")
out("| Rank corr. with decile | " + " | ".join(f"{mono[s]:+.2f}" for s in SIGNALS) + " |\n")

# 3. portfolios
out("## 3. Long-short portfolios, net of costs\n")
out("Top decile long, bottom decile short, equal weights, $1 each side, rebalanced every 5 days at the close after "
    "the signal (t+1), held 5 days. Costs: 3 bps + 30 bps/√(ADV in $m) one-way per stock (median stock ≈ "
    f"{float(np.nanmedian(cost.loc[test_dates].drop(columns=F.MARKET).to_numpy())):.1f} bps), short borrow 0.5% a "
    "year. Each figure is the average over 5 books started on consecutive days (range in brackets).\n")
books, rows = {}, []
for s in SIGNALS:
    res = []
    for k in range(H):
        r = E.long_short(pred[s], fwd, cost, test_dates[k::H], h=H, beta=beta_ex)
        r["hedged"] = r["net"] - r["beta_ex"] * spy_fwd.reindex(r.index).fillna(0.0)
        r["offset"], r["signal"] = k, s
        res.append(r)
    books[s] = pd.concat(res)
    for k, r in books[s].groupby("offset"):
        g, n = E.perf(r["gross"]), E.perf(r["net"])
        y = r["net"].dropna()
        beta = np.polyfit(spy_fwd.reindex(y.index).to_numpy(), y.to_numpy(), 1)[0]
        rows.append({"signal": s, "offset": k, "gross_sharpe": g["sharpe"], "net_sharpe": n["sharpe"],
                     "net_lo": n["sharpe_lo"], "net_hi": n["sharpe_hi"], "gross_ret": g["ann_return"],
                     "net_ret": n["ann_return"], "net_vol": n["ann_vol"], "max_dd": n["max_dd"],
                     "turnover": r["turnover"].iloc[1:].mean(), "cost_ann": r["cost"].mean() * E.PERIODS_PER_YEAR,
                     "beta": beta, "missing": int(r["missing"].sum()), "beta_ex": r["beta_ex"].mean(),
                     "hedged_ret": E.perf(r["hedged"])["ann_return"], "hedged_sharpe": E.perf(r["hedged"])["sharpe"],
                     "hedged_lo": E.perf(r["hedged"])["sharpe_lo"], "hedged_hi": E.perf(r["hedged"])["sharpe_hi"],
                     "hedged_beta": np.polyfit(spy_fwd.reindex(r.index).fillna(0).to_numpy(), r["hedged"].to_numpy(), 1)[0],
                     "long_ret": r["long_ret"].mean() * E.PERIODS_PER_YEAR,
                     "short_ret": r["short_ret"].mean() * E.PERIODS_PER_YEAR})
port = pd.DataFrame(rows)
port.to_csv(ROOT / "reports/phase5_portfolios.csv", index=False)
agg = port.groupby("signal").agg(["mean", "min", "max"]).reindex(SIGNALS)


def fmt(s, col, f="{:+.2f}", pct=False):
    m, lo, hi = agg.loc[s, (col, "mean")], agg.loc[s, (col, "min")], agg.loc[s, (col, "max")]
    if pct:
        return f"{m:+.1%} [{lo:+.1%}, {hi:+.1%}]"
    return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]"


out("| Signal | Gross return | Net return | Net vol | Gross Sharpe | **Net Sharpe** | Net Sharpe 95% CI (avg) | "
    "Max drawdown | Turnover / rebalance | Cost drag / yr | Beta to SPY |")
out("|---|---|---|---|---|---|---|---|---|---|---|")
for s in SIGNALS:
    out(f"| {s} | {fmt(s, 'gross_ret', pct=True)} | {fmt(s, 'net_ret', pct=True)} | {agg.loc[s, ('net_vol', 'mean')]:.1%} | "
        f"{fmt(s, 'gross_sharpe')} | **{fmt(s, 'net_sharpe')}** | [{agg.loc[s, ('net_lo', 'mean')]:+.2f}, "
        f"{agg.loc[s, ('net_hi', 'mean')]:+.2f}] | {agg.loc[s, ('max_dd', 'mean')]:.1%} | "
        f"{agg.loc[s, ('turnover', 'mean')]:.2f} | {agg.loc[s, ('cost_ann', 'mean')]:.1%} | "
        f"{agg.loc[s, ('beta', 'mean')]:+.2f} |")
out(f"\nTurnover is the sum of absolute weight changes per rebalance (4.0 = the whole book replaced). Drawdowns are "
    f"on cumulative (uncompounded) returns. Stock-weeks with a missing forward return (counted as 0): "
    f"{int(port['missing'].sum())}.\n")

spy_ann = spy_fwd.reindex(test_dates).mean() * E.PERIODS_PER_YEAR
out("### Market exposure (diagnostic added after seeing the betas above; not pre-registered)\n")
out("The books are dollar-neutral but not beta-neutral: they tend to buy higher-beta stocks than they short. "
    f"Over the test years SPY returned about {spy_ann:+.1%} a year (5-day simple returns, annualised), so part of "
    "the profit can be market exposure, not stock selection. Here each book is hedged with SPY using its beta at "
    "formation (the `beta_1y` feature, known at t; no future data), ignoring the small cost of trading SPY:\n")
out("| Signal | Ex-ante book beta | Realised beta before hedge | Net return hedged | **Net Sharpe hedged** | "
    "95% CI (avg) | Realised beta after hedge |")
out("|---|---|---|---|---|---|---|")
for s_ in SIGNALS:
    out(f"| {s_} | {agg.loc[s_, ('beta_ex', 'mean')]:+.2f} | {agg.loc[s_, ('beta', 'mean')]:+.2f} | "
        f"{fmt(s_, 'hedged_ret', pct=True)} | **{fmt(s_, 'hedged_sharpe')}** | [{agg.loc[s_, ('hedged_lo', 'mean')]:+.2f}, "
        f"{agg.loc[s_, ('hedged_hi', 'mean')]:+.2f}] | {agg.loc[s_, ('hedged_beta', 'mean')]:+.2f} |")
out("")

out("### Net return by year (staggered book, sum of 5-day net returns / 5)\n")
by_year = pd.DataFrame({s_: (books[s_]["net"] / H).groupby(books[s_].index.year).sum() for s_ in SIGNALS})
hedged_year = pd.DataFrame({s_: (books[s_]["hedged"] / H).groupby(books[s_].index.year).sum() for s_ in SIGNALS})
out("| Year | " + " | ".join(SIGNALS) + " | gbt, SPY-hedged |")
out("|---|" + "---|" * (len(SIGNALS) + 1))
for y_, r_ in by_year.iterrows():
    out(f"| {y_} | " + " | ".join(f"{v:+.1%}" for v in r_) + f" | {hedged_year.loc[y_, 'gbt']:+.1%} |")
g_years = by_year["gbt"]
top2 = g_years.sort_values(ascending=False).head(2)
out(f"\nFor gbt, {int((g_years > 0).sum())} of {len(g_years)} years are positive. The best two years "
    f"({', '.join(str(i) for i in top2.index)}) add {top2.sum():+.1%}; the other {len(g_years) - 2} together add "
    f"{g_years.sum() - top2.sum():+.1%} (total {g_years.sum():+.1%}). The result rests on a few years.\n")

# 4. decay
out("## 4. Signal decay\n")
out("Mean IC against 5-day excess returns that start k trading days after the usual entry (close t+1+k to t+6+k):\n")
lags = [0, 5, 10, 20]
dec_rows = {}
for k in lags:
    y = forward(k, log=True)
    y = y.sub(y.where(pred["y_raw"].unstack().reindex(columns=y.columns).notna()).mean(axis=1), axis=0)
    ys = y.stack().reindex(pred.index)
    dec_rows[k] = {s: M.daily_ic(pred[s], ys).mean() for s in SIGNALS}
dd = pd.DataFrame(dec_rows).T
out("| Start delay k (days) | " + " | ".join(SIGNALS) + " |")
out("|---|" + "---|" * len(SIGNALS))
for k, r in dd.iterrows():
    out(f"| {k} | " + " | ".join(f"{v:+.4f}" for v in r) + " |")
out("")

# verdict
g = summ["gbt"]
out("## 5. Verdict on the development period\n")
out(f"- **Primary model (gbt):** mean IC {g['ic_mean']:+.4f} (t = {g['t_nw']:+.2f}), "
    + ("significant" if g["p_bonf"] < 0.05 else "**not significant**")
    + f" after the Bonferroni adjustment for 5 signals (p = {g['p_bonf']:.2f}). Its deciles rise almost steadily "
    f"(rank correlation {mono['gbt']:+.2f}), but the out-of-sample R² is {g['r2']:+.4f}: the predictions' size "
    "is not reliable, only their order is.")
zero_in = all(agg.loc[s_, (lo, "mean")] < 0 < agg.loc[s_, (hi, "mean")] for s_ in SIGNALS
              for lo, hi in (("net_lo", "net_hi"), ("hedged_lo", "hedged_hi")))
out(f"- **After costs:** net Sharpe {agg.loc['gbt', ('net_sharpe', 'mean')]:+.2f}, "
    f"{agg.loc['gbt', ('hedged_sharpe', 'mean')]:+.2f} once its market beta is hedged; "
    + ("every signal's 95% interval, hedged or not, includes 0. " if zero_in else "some 95% intervals exclude 0. ")
    + f"Costs take {agg.loc['gbt', ('cost_ann', 'mean')]:.1%} a year from a gross {agg.loc['gbt', ('gross_ret', 'mean')]:.1%}. "
    f"Its turnover ({agg.loc['gbt', ('turnover', 'mean')]:.2f} per rebalance, out of a maximum 4) is close to the "
    f"pure reversal baseline's ({agg.loc['reversal', ('turnover', 'mean')]:.2f}), consistent with leaning on "
    "short-term reversal; Phase 6 importance checks this.")
out(f"- **Compared with the baselines:** gbt's IC is about the same as plain 1-month reversal "
    f"({summ['reversal']['ic_mean']:+.4f}) and its hedged Sharpe is not higher. The machine-learning models have "
    "not shown value beyond the simplest baseline.")
out("- **What this means:** in development there is weak evidence of an ordering signal and no evidence of a "
    "tradable edge after costs. The hold-out test (after Phase 6) is run once on the frozen pipeline regardless.\n")

# figures
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
colors = {"gbt": "#C44E52", "ridge": "#4C72B0", "lasso": "#8172B2", "momentum": "#55A868", "reversal": "#DD8452"}
for s in SIGNALS:
    b = books[s].sort_index()
    axes[0].plot(b.index, (b["net"] / H).cumsum(), label=s, color=colors[s], lw=2 if s == "gbt" else 1.2)
axes[0].axhline(0, color="k", lw=0.6)
axes[0].set_title("Cumulative net return, staggered long-short book"); axes[0].legend(fontsize=8)
for s in SIGNALS:
    axes[1].plot(dec.index, dec[s], marker="o", label=s, color=colors[s], lw=2 if s == "gbt" else 1.2)
axes[1].axhline(0, color="k", lw=0.6)
axes[1].set_xlabel("prediction decile"); axes[1].set_ylabel("bps / 5 days"); axes[1].set_title("Mean 5-day excess return by decile")
axes[1].set_xticks(range(1, 11))
fig.tight_layout(); fig.savefig(ROOT / "figures/phase5_portfolios.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 3.6))
yr.plot.bar(ax=ax, color=[colors[s] for s in SIGNALS], width=0.8)
ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("mean daily IC"); ax.set_title("Out-of-sample IC by test year")
ax.legend(fontsize=8, ncol=5)
fig.tight_layout(); fig.savefig(ROOT / "figures/phase5_ic_by_year.png", dpi=150); plt.close(fig)
out("![portfolios](../figures/phase5_portfolios.png)\n")
out("![IC by year](../figures/phase5_ic_by_year.png)\n")
(ROOT / "reports/phase5_evaluation.md").write_text("\n".join(lines) + "\n")
