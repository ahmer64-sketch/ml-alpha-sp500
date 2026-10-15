"""Phase 6: robustness checks (pre-registered in DESIGN.md), development period only.

Run: .venv/bin/python scripts/run_phase6.py   (~40 min; needs Phases 2 and 4 outputs)
Each section caches its results in data/processed/phase6/ (delete to recompute).
Writes reports/phase6_robustness.md and figures/phase6_*.png.
"""
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha import data, evaluate as E, features as F, models as M, robust as R
from alpha.features import FEATURES
from alpha.splits import walk_forward

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/processed/phase6"
CACHE.mkdir(parents=True, exist_ok=True)
DEV_END = "2022-12-31"
T_PERTURB = pd.Timestamp("2018-06-29")
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


def cached(name, fn):
    f = CACHE / f"{name}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    t0 = time.perf_counter()
    res = fn()
    print(f"[{name}] {time.perf_counter() - t0:.0f}s", flush=True)
    f.write_bytes(pickle.dumps(res))
    return res


# ---------------------------------------------------------------- setup
pred = pd.read_parquet(ROOT / "predictions/dev_predictions.parquet")
panel = pd.read_parquet(ROOT / "data/processed/features_h5.parquet")
trials = pd.read_csv(ROOT / "reports/trials.csv")
chosen = M.chosen_params(trials)
raw, _ = data.load_prices(sorted((ROOT / "data/raw").glob("prices_*.parquet"))[-1])
df, _ = data.clean(raw)
p = F.panel(df)
cal = p["adj"].index
folds = walk_forward(cal, range(2015, 2023), "2013-01-01", 5, 5, last_date=DEV_END)
cost = E.cost_bps(p["dvol"].rolling(63, min_periods=20).median() / 1e6)
beta = F.raw_features(p)[0]["beta_1y"]


def book_stats(pr, h=5, frac=0.1, cost_mult=1.0, fwd=None):
    fwd = E.forward_returns(p["adj"], 0, h, DEV_END) if fwd is None else fwd
    dates = pr.index.get_level_values("date").unique().sort_values()
    st, _ = E.staggered(pr, fwd, cost, dates, h, beta, fwd[F.MARKET], frac=frac, cost_mult=cost_mult)
    return st


def ic_row(pr, y):
    ic = M.daily_ic(pr, y)
    return {"ic_mean": ic.mean(), "t_nw": E.nw_tstat(ic, 5), "se": ic.mean() / E.nw_tstat(ic, 5)}


out("# Phase 6: robustness (development period 2015-2022)\n")
out("Checks and pass/fail rules were pre-registered in [DESIGN.md](../DESIGN.md) before any of them ran. "
    "Every result is shown, including failures. The hold-out is still untouched.\n")

# ---------------------------------------------------------------- 0. reproducibility + models for importance
def refit_gbt():
    models, preds = {}, []
    for s in folds:
        m, pr = M.fit_predict(panel, s, "gbt", chosen[(s.name, "gbt")])
        models[s.name] = m
        preds.append(pr)
    return models, pd.concat(preds)


gbt_models, gbt_refit = cached("refit_gbt", refit_gbt)
diff = float((gbt_refit.astype("float32") - pred["gbt"].reindex(gbt_refit.index)).abs().max())
out("## 0. Reproducibility\n")
out(f"Refitting gbt on every fold with the settings Phase 4 chose reproduces the saved predictions: max absolute "
    f"difference {diff:.1e} ({'pass' if diff < 1e-6 else 'FAIL'}; predictions are stored as float32). The refitted "
    "models are used for the importance test below.\n")

# ---------------------------------------------------------------- 1. placebo
def placebo():
    rows = []
    for seed in range(3):
        shuffled = panel.copy()
        shuffled["y_rank"] = R.shuffle_within_date(panel["y_rank"].dropna(), seed).reindex(panel.index)
        for name in ("gbt", "ridge"):
            prs = [M.fit_predict(shuffled, s, name, chosen[(s.name, name)])[1] for s in folds]
            pr = pd.concat(prs)
            rows.append({"seed": seed, "model": name, **ic_row(pr, pred["y_raw"]),
                         "sharpe_gross": book_stats(pr)["gross_sharpe"]})
    return pd.DataFrame(rows)


pl = cached("placebo", placebo)
pl["pass"] = pl["t_nw"].abs() < 2
out("## 1. Placebo: models trained on shuffled labels\n")
out("Within every training date the label is shuffled across stocks, so features carry no information about it. "
    "The same models and settings are refitted and scored on the real test labels. **Pass:** mean IC within 2 "
    "Newey-West standard errors of 0.\n")
out("| Seed | Model | Mean test IC | t (NW) | Gross Sharpe of its long-short book | Result |")
out("|---|---|---|---|---|---|")
for _, r in pl.iterrows():
    out(f"| {r['seed']} | {r['model']} | {r['ic_mean']:+.4f} | {r['t_nw']:+.2f} | {r['sharpe_gross']:+.2f} | "
        f"{'pass' if r['pass'] else 'FAIL'} |")
real = {m: M.daily_ic(pred[m], pred["y_raw"]).mean() for m in ("gbt", "ridge")}
out(f"\n{int(pl['pass'].sum())} of {len(pl)} placebo runs pass. For comparison, the real models' mean IC: gbt "
    f"{real['gbt']:+.4f}, ridge {real['ridge']:+.4f}.\n")

# ---------------------------------------------------------------- 2. future perturbation
def perturbation():
    rng = np.random.default_rng(7)
    q = raw.copy()
    after = q["date"] > T_PERTURB
    noise = np.exp(rng.normal(0, 0.05, after.sum()))
    for c in ("open", "high", "low", "close", "adj_close"):
        q.loc[after, c] = q.loc[after, c] * noise
    q.loc[after, "volume"] = q.loc[after, "volume"] * np.exp(rng.normal(0, 0.3, after.sum()))
    qc, _ = data.clean(q)
    panel_q = F.build(qc)
    s = [f for f in folds if f.name == "2018"][0]
    res = {}
    for name in ("gbt", "ridge"):
        a, pa = M.run_fold(panel, s, name)
        b, pb = M.run_fold(panel_q, s, name)
        res[name] = (a, b, pa, pb)
    feats_same = {}
    for k in FEATURES:
        x = panel.loc[(slice(None, T_PERTURB), slice(None)), k]
        y = panel_q.loc[(slice(None, T_PERTURB), slice(None)), k].reindex(x.index)
        feats_same[k] = float((x - y).abs().max())
    return res, feats_same


pt, feats_same = cached("perturbation", perturbation)
out("## 2. Future perturbation of the whole pipeline\n")
out(f"Every price after {T_PERTURB:%Y-%m-%d} is multiplied by random noise (5% sd) and volumes by noise (30% sd). "
    "Cleaning, features, labels and the whole 2018 fold (tuning included) are then rerun. **Pass:** predictions up "
    "to that date are unchanged, and later ones change.\n")
out("| Model | Hyperparameters chosen (original / perturbed) | Max diff, test dates ≤ cut-off | Max diff, after | Result |")
out("|---|---|---|---|---|")
pt_ok = True
for name, (a, b, pa, pb) in pt.items():
    d = a.index.get_level_values("date")
    before = float((a[d <= T_PERTURB] - b.reindex(a.index)[d <= T_PERTURB]).abs().max())
    after_ = float((a[d > T_PERTURB] - b.reindex(a.index)[d > T_PERTURB]).abs().max())
    ok = before < 1e-9 and after_ > 1e-6 and pa == pb
    pt_ok &= ok
    out(f"| {name} | `{pa}` / `{pb}` | {before:.1e} | {after_:.1e} | {'pass' if ok else 'FAIL'} |")
out(f"\nFeatures for every date up to the cut-off are also identical (max difference over all 12 features "
    f"{max(feats_same.values()):.1e}). **{'Pass' if pt_ok else 'FAIL'}.**\n")
out("**Known gap this test cannot see:** the cleaning rule that drops placeholder history before a listing "
    "(Phase 1) looks at each stock's whole history. For SW, FERG and AMCR it removes years of untraded prices that "
    "a trader at the time would also have seen as untraded, and none of the three was an index member then, so the "
    "effect is part of the membership look-ahead already stated, not a separate leak.\n")

# ---------------------------------------------------------------- 3. importance
def importance():
    rows = []
    for s in folds:
        m = gbt_models[s.name]
        test = panel[panel.index.get_level_values("date").isin(s.test)]
        base = M.daily_ic(pd.Series(m.predict(test[FEATURES].to_numpy()), test.index), test["y_raw"]).mean()
        for fam, cols in R.FAMILIES.items():
            drops = []
            for seed in range(2):
                xp = R.permute_family(test[FEATURES], cols, seed)
                ic = M.daily_ic(pd.Series(m.predict(xp.to_numpy()), test.index), test["y_raw"]).mean()
                drops.append(base - ic)
            rows.append({"fold": s.name, "family": fam, "base_ic": base, "drop": np.mean(drops)})
    coefs = []
    for s in folds:
        m, _ = M.fit_predict(panel, s, "ridge", chosen[(s.name, "ridge")])
        coefs.append(pd.Series(m.coef_, FEATURES, name=s.name))
    return pd.DataFrame(rows), pd.DataFrame(coefs)


imp, coefs = cached("importance", importance)
out("## 3. What the primary model relies on\n")
out("Permutation importance: the drop in each fold's test IC when one feature family is shuffled across stocks "
    "within each date (average of 2 shuffles). A large drop means the model's skill depends on that family; a "
    "negative drop means the family was hurting.\n")
tab = imp.pivot(index="fold", columns="family", values="drop")[list(R.FAMILIES)]
out("| Test year | Base IC | " + " | ".join(R.FAMILIES) + " |")
out("|---|---|" + "---|" * len(R.FAMILIES))
base = imp.groupby("fold")["base_ic"].first()
for f, r in tab.iterrows():
    out(f"| {f} | {base[f]:+.4f} | " + " | ".join(f"{v:+.4f}" for v in r) + " |")
mean_drop = tab.mean().sort_values(ascending=False)
out("| **Mean** | " + f"{base.mean():+.4f} | " + " | ".join(f"**{tab[c].mean():+.4f}**" for c in R.FAMILIES) + " |")
out(f"| Folds with a positive drop | | " + " | ".join(f"{int((tab[c] > 0).sum())}/8" for c in R.FAMILIES) + " |")
out(f"\nRanking by mean drop: " + ", ".join(f"{k} ({v:+.4f})" for k, v in mean_drop.items()) + ".\n")
out("Ridge coefficients per fold (x 1000; features are ranks in [-0.5, 0.5], so they are comparable):\n")
out("| Feature | " + " | ".join(coefs.index) + " | Same sign in all folds |")
out("|---|" + "---|" * (len(coefs) + 1))
for k in FEATURES:
    v = coefs[k]
    out(f"| {k} | " + " | ".join(f"{x * 1e3:+.2f}" for x in v) + f" | {'yes' if (v > 0).all() or (v < 0).all() else 'no'} |")
out("")

# ---------------------------------------------------------------- 4. sensitivity
out("## 4. Sensitivity\n")
base5 = {m: {**ic_row(pred[m], pred["y_raw"]), **book_stats(pred[m])} for m in ("gbt", "ridge")}


def horizons():
    res = {}
    for h in (1, 21):
        panel_h = F.build(df, h)
        folds_h = walk_forward(cal, range(2015, 2023), "2013-01-01", h, h, last_date=DEV_END)
        fwd_h = E.forward_returns(p["adj"], 0, h, DEV_END)
        for name in ("gbt", "ridge"):
            pr = pd.concat([M.run_fold(panel_h, s, name)[0] for s in folds_h])
            y = panel_h["y_raw"].reindex(pr.index)
            res[(h, name)] = {**ic_row(pr, y), **book_stats(pr, h=h, fwd=fwd_h), "dates": pr.index.get_level_values(0).nunique()}
    return res


hz = cached("horizons", horizons)
out("### 4a. Label horizon (models retrained and retuned for each horizon; embargo = h; rebalance every h days)\n")
out("| Horizon | Model | Mean IC | t (NW) | Gross Sharpe | Net Sharpe | Net Sharpe 95% CI | Hedged net Sharpe | Turnover / rebalance |")
out("|---|---|---|---|---|---|---|---|---|")
for h in (1, 5, 21):
    for name in ("gbt", "ridge"):
        r = base5[name] if h == 5 else hz[(h, name)]
        out(f"| {h} days{' (pre-registered)' if h == 5 else ''} | {name} | {r['ic_mean']:+.4f} | {r['t_nw']:+.2f} | "
            f"{r['gross_sharpe']:+.2f} | {r['net_sharpe']:+.2f} | [{r['net_lo']:+.2f}, {r['net_hi']:+.2f}] | "
            f"{r['hedged_sharpe']:+.2f} | {r['turnover']:.2f} |")
out("\nThe t-statistics use 5 Newey-West lags for every horizon; for h = 21 the labels overlap for 21 days, so its "
    "t-statistic is overstated.\n")


def fixed_grid():
    rows = []
    for name in ("gbt", "ridge"):
        for params in M.GRIDS[name]:
            pr = pd.concat([M.fit_predict(panel, s, name, params)[1] for s in folds])
            rows.append({"model": name, "params": str(params), **ic_row(pr, pred["y_raw"]), **book_stats(pr)})
    return pd.DataFrame(rows)


fg = cached("fixed_grid", fixed_grid)
out("### 4b. Hyperparameters held fixed for all folds (no tuning)\n")
out("| Model | Setting | Mean IC | t (NW) | Net Sharpe | Hedged net Sharpe |")
out("|---|---|---|---|---|---|")
for _, r in fg.iterrows():
    out(f"| {r['model']} | `{r['params']}` | {r['ic_mean']:+.4f} | {r['t_nw']:+.2f} | {r['net_sharpe']:+.2f} | "
        f"{r['hedged_sharpe']:+.2f} |")
for name in ("gbt", "ridge"):
    out(f"| {name} | **tuned per fold (Phase 4)** | {base5[name]['ic_mean']:+.4f} | {base5[name]['t_nw']:+.2f} | "
        f"{base5[name]['net_sharpe']:+.2f} | {base5[name]['hedged_sharpe']:+.2f} |")
best_fixed = fg.loc[fg[fg["model"] == "gbt"]["net_sharpe"].idxmax()]
out(f"\nTuning versus the best fixed gbt setting with hindsight ({best_fixed['params']}, net Sharpe "
    f"{best_fixed['net_sharpe']:+.2f}): choosing it after seeing these test results would be data snooping, which "
    "is why the tuned result is the one reported.\n")


def costs_and_universe():
    rows = []
    for mult in (0.0, 0.5, 1.0, 2.0):
        rows.append({"variant": f"costs x{mult:g}", **book_stats(pred["gbt"], cost_mult=mult)})
    rows.append({"variant": "top/bottom 20% (costs x1)", **book_stats(pred["gbt"], frac=0.2)})
    cons = pd.read_csv(sorted((ROOT / "data/raw").glob("sp500_constituents_*.csv"))[-1])
    old = set(cons.loc[pd.to_datetime(cons["added"], errors="coerce") <= "2012-12-31", "ticker"])
    keep = pred.index.get_level_values("ticker").isin(old)
    sub = pred[keep]
    u = {"variant": f"members since 2012 or earlier ({len(old)} stocks)", **ic_row(sub["gbt"], sub["y_raw"]),
         **book_stats(sub["gbt"])}
    u_rest = {"variant": "joined after 2012 (the rest)", **ic_row(pred[~keep]["gbt"], pred[~keep]["y_raw"]),
              **book_stats(pred[~keep]["gbt"])}
    return pd.DataFrame(rows), pd.DataFrame([u, u_rest])


cu, uni = cached("costs_universe", costs_and_universe)
out("### 4c. Costs and portfolio breadth (gbt)\n")
out("| Variant | Gross Sharpe | Net Sharpe | Net Sharpe 95% CI | Hedged net Sharpe | Turnover |")
out("|---|---|---|---|---|---|")
for _, r in cu.iterrows():
    out(f"| {r['variant']} | {r['gross_sharpe']:+.2f} | {r['net_sharpe']:+.2f} | [{r['net_lo']:+.2f}, {r['net_hi']:+.2f}] | "
        f"{r['hedged_sharpe']:+.2f} | {r['turnover']:.2f} |")
out("\n### 4d. Universe: membership look-ahead (gbt, same predictions, deciles formed within each group)\n")
out("| Stocks | Mean IC | t (NW) | Net Sharpe | Hedged net Sharpe |")
out("|---|---|---|---|---|")
out(f"| all (Phase 5) | {base5['gbt']['ic_mean']:+.4f} | {base5['gbt']['t_nw']:+.2f} | {base5['gbt']['net_sharpe']:+.2f} | "
    f"{base5['gbt']['hedged_sharpe']:+.2f} |")
for _, r in uni.iterrows():
    out(f"| {r['variant']} | {r['ic_mean']:+.4f} | {r['t_nw']:+.2f} | {r['net_sharpe']:+.2f} | {r['hedged_sharpe']:+.2f} |")
out("")

# ---------------------------------------------------------------- summary table
out("## 5. Summary\n")
n_pass_pl = int(pl["pass"].sum())
any_sig = [f"{h}-day {n}" for (h, n), r in hz.items() if r["net_lo"] > 0]
any_sig += [f"fixed {r['model']} {r['params']}" for _, r in fg.iterrows() if r["net_lo"] > 0]
any_sig += [r["variant"] for _, r in cu.iterrows() if r["net_lo"] > 0]
any_sig += [r["variant"] for _, r in uni.iterrows() if r["net_lo"] > 0]
out("| Check | Result |")
out("|---|---|")
out(f"| Reproducibility of Phase 4 predictions | {'pass' if diff < 1e-6 else 'FAIL'} |")
out(f"| Placebo (shuffled labels), 6 runs | {n_pass_pl}/6 pass |")
out(f"| Future perturbation of the whole pipeline | {'pass' if pt_ok else 'FAIL'} |")
out(f"| Most important feature family (gbt) | {mean_drop.index[0]} (mean IC drop {mean_drop.iloc[0]:+.4f}), then "
    f"{mean_drop.index[1]} ({mean_drop.iloc[1]:+.4f}) |")
u_old, u_new = uni.iloc[0], uni.iloc[1]
out(f"| Signal on stocks in the index since 2012 or earlier | IC {u_old['ic_mean']:+.4f} (t {u_old['t_nw']:+.2f}) vs "
    f"{u_new['ic_mean']:+.4f} (t {u_new['t_nw']:+.2f}) on later joiners |")
out(f"| Any variant with a net Sharpe 95% CI above 0 (excluding zero-cost runs) | "
    + (", ".join(v for v in any_sig if 'x0' != v[-2:]) or "none") + " |")
out("")

# ---------------------------------------------------------------- interpretation
pl_ic = pl["ic_mean"]
out("## 6. Interpretation (written after seeing the results above; pass/fail outcomes are unchanged)\n")
out(f"- **The placebo failures are not leakage.** Both failing runs are *negative* (leakage would push them up), "
    "and the future-perturbation test of the whole pipeline passes exactly. They show something else: a model fitted "
    "to shuffled labels is not a constant but a random mix of features, i.e. a random factor portfolio, and factor "
    "returns persist for months. Its IC therefore wanders further from 0 than 5-lag Newey-West errors allow. "
    f"Placebo ICs ranged from {pl_ic.min():+.4f} to {pl_ic.max():+.4f}; the largest in size "
    f"({pl_ic.abs().max():.4f}) is as large as the real gbt IC ({real['gbt']:+.4f}). The t-statistics in Phase 5 "
    "understate the uncertainty, and the real result is not clearly separated from what noise-fitted models produce.")
out(f"- **The signal sits in the stocks affected by membership look-ahead.** On stocks already in the index by 2012 "
    f"the IC is {u_old['ic_mean']:+.4f} (t {u_old['t_nw']:+.2f}); on stocks that joined later it is "
    f"{u_new['ic_mean']:+.4f} (t {u_new['t_nw']:+.2f}). The most important feature family is size and liquidity, "
    "and ridge puts a positive weight on illiquidity in every fold: the model favours smaller, less liquid stocks, "
    "and in a universe of today's members those include the ones that went on to grow into the index. Later "
    "joiners may also simply be more predictable (smaller, less covered), so the split cannot prove the cause, but "
    "it removes the evidence of an edge in the stocks where the data are free of this bias.")
out(f"- **Costs and horizon:** the gross Sharpe is about {cu.iloc[0]['gross_sharpe']:+.2f}; costs at the assumed level "
    "remove most of it and double costs make it negative. Daily trading loses money (gbt: the whole 95% interval is below 0); a 21-day "
    "horizon looks better but its interval still includes 0, and choosing it now would be data snooping.")
out("- **Conclusion for the hold-out:** the frozen, pre-registered pipeline (h = 5, gbt primary) runs once as planned. "
    "Given the above, the expectation is an IC near zero and no net edge; the hold-out also reports the "
    "members-since-2012 split.\n")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.bar(range(len(mean_drop)), mean_drop.values, color="#C44E52")
ax.errorbar(range(len(mean_drop)), mean_drop.values, yerr=tab[mean_drop.index].std().values / np.sqrt(8), fmt="none", color="k")
ax.set_xticks(range(len(mean_drop)), mean_drop.index, rotation=30, ha="right")
ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("drop in test IC when shuffled"); ax.set_title("gbt permutation importance by family (±1 s.e. over folds)")
ax = axes[1]
labels = [f"placebo {r['model']} s{r['seed']}" for _, r in pl.iterrows()] + ["real ridge", "real gbt"]
vals = list(pl["ic_mean"]) + [real["ridge"], real["gbt"]]
errs = list(1.96 * pl["se"].abs()) + [1.96 * abs(base5["ridge"]["se"]), 1.96 * abs(base5["gbt"]["se"])]
ax.barh(range(len(vals)), vals, xerr=errs, color=["#BBBBBB"] * len(pl) + ["#4C72B0", "#C44E52"])
ax.set_yticks(range(len(vals)), labels); ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("mean test IC (95% NW interval)"); ax.set_title("Placebo vs real")
fig.tight_layout(); fig.savefig(ROOT / "figures/phase6_robustness.png", dpi=150); plt.close(fig)
out("![robustness](../figures/phase6_robustness.png)")
(ROOT / "reports/phase6_robustness.md").write_text("\n".join(lines) + "\n")
