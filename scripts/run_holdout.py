"""Phase 7: the single hold-out run (2023 - Sep 2026) of the pipeline frozen after Phase 6.

Run: .venv/bin/python scripts/run_holdout.py   (~10 min; needs data/processed/features_h5.parquet)
Refuses to run unless src/alpha/ is identical to the frozen commit. The first run is recorded in
reports/holdout_run.json; later runs (e.g. a fresh-clone `make all`) only reproduce it.
Writes predictions/holdout_predictions.parquet, reports/phase7_holdout.md, figures/phase7_holdout.png.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from alpha import data, evaluate as E, features as F, models as M
from alpha.splits import walk_forward

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "1b30cb4"
SIGNALS = ["gbt", "ridge", "lasso", "momentum", "reversal"]
FITTED = ["gbt", "ridge", "lasso"]
H = 5
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


# ---------------------------------------------------------------- provenance guard
if subprocess.run(["git", "diff", "--quiet", FROZEN, "--", "src/alpha"], cwd=ROOT).returncode != 0:
    raise SystemExit(f"src/alpha differs from the frozen commit {FROZEN}; the hold-out must use the frozen pipeline.")
run_file = ROOT / "reports/holdout_run.json"
now = datetime.now(timezone.utc).isoformat(timespec="seconds")
record = json.loads(run_file.read_text()) if run_file.exists() else {"first_run_utc": now, "frozen_commit": FROZEN}
record["last_run_utc"] = now
run_file.write_text(json.dumps(record, indent=2) + "\n")

# ---------------------------------------------------------------- fit and predict
raw, _ = data.load_prices(sorted((ROOT / "data/raw").glob("prices_*.parquet"))[-1])
df, _ = data.clean(raw)
p = F.panel(df)
cal = p["adj"].index
panel = pd.read_parquet(ROOT / "data/processed/features_h5.parquet")
folds = walk_forward(cal, [2023, 2024, 2025, 2026], "2013-01-01", H, H, last_date=cal[-1])
preds, chosen = [], []
for s in folds:
    cols = {}
    for name in FITTED:
        cols[name], best = M.run_fold(panel, s, name)
        chosen.append({"fold": s.name, "model": name, "params": json.dumps(best)})
        print(s.name, name, best, flush=True)
    for name in M.BASELINES:
        cols[name] = M.baseline(panel, s, name)
    d = pd.DataFrame(cols)
    rows = panel[panel.index.get_level_values("date").isin(s.test)]
    d["y_raw"], d["y_rank"], d["fold"] = rows["y_raw"], rows["y_rank"], s.name
    preds.append(d)
pred = pd.concat(preds).sort_index()
pred.to_parquet(ROOT / "predictions/holdout_predictions.parquet")

# ---------------------------------------------------------------- evaluate (Phase 5 rules)
last = cal[-1]
fwd = E.forward_returns(p["adj"], 0, H, last)
spy_fwd = fwd[F.MARKET]
cost = E.cost_bps(p["dvol"].rolling(63, min_periods=20).median() / 1e6)
beta = F.raw_features(p)[0]["beta_1y"]
cons = pd.read_csv(sorted((ROOT / "data/raw").glob("sp500_constituents_*.csv"))[-1])
old = set(cons.loc[pd.to_datetime(cons["added"], errors="coerce") <= "2012-12-31", "ticker"])
dev = pd.read_parquet(ROOT / "predictions/dev_predictions.parquet")
dates = pred.index.get_level_values("date").unique().sort_values()


def ic_stats(pr, y):
    ic = M.daily_ic(pr, y)
    t = E.nw_tstat(ic, 5)
    return ic, {"ic_mean": ic.mean(), "t_nw": t, "p_bonf": min(1.0, 2 * (1 - stats.norm.cdf(abs(t))) * len(SIGNALS))}


def books(pr):
    d = pr.index.get_level_values("date").unique().sort_values()
    return E.staggered(pr, fwd, cost, d, H, beta, spy_fwd)


out("# Phase 7: hold-out (2023 - Sep 2026), single run of the frozen pipeline\n")
out(f"Pipeline frozen at commit `{FROZEN}` after Phase 6 (the script checks that `src/alpha/` is identical). "
    f"First run: {record['first_run_utc']} UTC; this run: {now} UTC. Same models, tuning, portfolio and costs as "
    "pre-registered; nothing was changed after seeing these numbers.\n")
out(f"{len(dates)} test dates ({dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}), {len(pred):,} stock-days, "
    f"{len(folds)} yearly folds (each trained on all earlier data).\n")
out("Hyperparameters chosen on each fold's inner validation: "
    + "; ".join(f"{c['fold']} {c['model']} `{c['params']}`" for c in chosen) + ".\n")

out("## 1. Signal quality\n")
out("| Signal | Hold-out mean IC | t (NW) | p (Bonferroni, 5) | Hold-out OOS R² | Development mean IC (Phase 5) |")
out("|---|---|---|---|---|---|")
ics, summ = {}, {}
for s in SIGNALS:
    ics[s], summ[s] = ic_stats(pred[s], pred["y_raw"])
    r2 = f"{E.r2_zero(pred[s], pred['y_rank']):+.5f}" if s in FITTED else "n/a"
    out(f"| {s}{' (primary)' if s == 'gbt' else ''} | {summ[s]['ic_mean']:+.4f} | {summ[s]['t_nw']:+.2f} | "
        f"{summ[s]['p_bonf']:.3f} | {r2} | {M.daily_ic(dev[s], dev['y_raw']).mean():+.4f} |")
yr = pd.DataFrame({s: ics[s].groupby(ics[s].index.year).mean() for s in SIGNALS})
out("\nMean IC by hold-out year:\n")
out("| Year | " + " | ".join(SIGNALS) + " |")
out("|---|" + "---|" * len(SIGNALS))
for y, r in yr.iterrows():
    out(f"| {y}{' (to Sep)' if y == 2026 else ''} | " + " | ".join(f"{v:+.4f}" for v in r) + " |")
dec = pd.DataFrame({s: E.deciles(pred[s], pred["y_raw"]) for s in SIGNALS}) * 1e4
out("\nTop-minus-bottom decile, mean 5-day excess return (bps): "
    + ", ".join(f"{s} {dec.loc[10, s] - dec.loc[1, s]:+.1f}" for s in SIGNALS) + ".\n")

out("## 2. Long-short portfolios net of costs (same rules as Phase 5; average of 5 staggered books)\n")
out("| Signal | Gross Sharpe | **Net Sharpe** | Net Sharpe 95% CI | Net return / yr | Hedged net Sharpe | Turnover |")
out("|---|---|---|---|---|---|---|")
bk = {}
for s in SIGNALS:
    st, rows = books(pred[s])
    bk[s] = (st, rows)
    out(f"| {s} | {st['gross_sharpe']:+.2f} | **{st['net_sharpe']:+.2f}** | [{st['net_lo']:+.2f}, {st['net_hi']:+.2f}] | "
        f"{st['net_ret']:+.1%} | {st['hedged_sharpe']:+.2f} | {st['turnover']:.2f} |")
out("")

out("## 3. Membership look-ahead split (as in Phase 6)\n")
out("| Stocks | gbt mean IC | t (NW) | Net Sharpe | Hedged net Sharpe |")
out("|---|---|---|---|---|")
keep = pred.index.get_level_values("ticker").isin(old)
for label, sub in ((f"members since 2012 or earlier", pred[keep]), ("joined after 2012", pred[~keep]), ("all", pred)):
    _, st_ic = ic_stats(sub["gbt"], sub["y_raw"])
    st, _ = books(sub["gbt"])
    out(f"| {label} | {st_ic['ic_mean']:+.4f} | {st_ic['t_nw']:+.2f} | {st['net_sharpe']:+.2f} | {st['hedged_sharpe']:+.2f} |")
out("")

g = summ["gbt"]
gs = bk["gbt"][0]
out("## 4. Verdict\n")
out(f"- **Primary model (gbt):** hold-out mean IC {g['ic_mean']:+.4f} (t {g['t_nw']:+.2f}, Bonferroni p "
    f"{g['p_bonf']:.2f}); net Sharpe {gs['net_sharpe']:+.2f} [{gs['net_lo']:+.2f}, {gs['net_hi']:+.2f}], "
    f"{gs['hedged_sharpe']:+.2f} hedged. "
    + ("The interval excludes 0." if gs["net_lo"] > 0 or gs["net_hi"] < 0 else "The interval includes 0."))
out(f"- **Against the baselines:** reversal IC {summ['reversal']['ic_mean']:+.4f}, momentum "
    f"{summ['momentum']['ic_mean']:+.4f}.")
out("- Phase 6 showed that t-statistics understate the uncertainty (placebo runs) and that the development signal "
    "sat in later index joiners; read these numbers with both in mind.\n")

out("## 5. Interpretation (added after the first run; the numbers above are unchanged)\n")
out("The universe is the index as of October 2026, so it includes stocks that joined *during* the hold-out, and "
    "they joined because they rose during it. Splitting the hold-out by when each stock joined:\n")
added = pd.to_datetime(cons.set_index("ticker")["added"], errors="coerce")
a = added.reindex(pred.index.get_level_values("ticker")).to_numpy()
groups = {"in the index before 2013": a <= np.datetime64("2012-12-31"),
          "joined 2013-2022": (a > np.datetime64("2012-12-31")) & (a <= np.datetime64("2022-12-31")),
          "joined during the hold-out (2023 on)": a > np.datetime64("2022-12-31")}
out("| Stocks | Count | Mean 5-day excess return | gbt IC | t (NW) | momentum IC |")
out("|---|---|---|---|---|---|")
gi = {}
for label, m in groups.items():
    sub = pred[m]
    _, sg = ic_stats(sub["gbt"], sub["y_raw"])
    _, sm = ic_stats(sub["momentum"], sub["y_raw"])
    gi[label] = sg
    out(f"| {label} | {sub.index.get_level_values('ticker').nunique()} | {sub['y_raw'].mean() * 1e4:+.1f} bps | "
        f"{sg['ic_mean']:+.4f} | {sg['t_nw']:+.2f} | {sm['ic_mean']:+.4f} |")
late = gi["joined during the hold-out (2023 on)"]
early = gi["in the index before 2013"]
out(f"\n- **The hold-out result is inflated by membership look-ahead.** The stocks that joined during the "
    f"hold-out outperform the rest by construction, and the model's IC on them is {late['ic_mean']:+.4f}, against "
    f"{early['ic_mean']:+.4f} (t {early['t_nw']:+.2f}) on stocks that were members throughout. On those long-time "
    "members the long-short book's market-hedged net Sharpe is negative (section 3).")
out("- **The market regime helped momentum-like signals:** plain 12-month momentum did about as well as the model "
    f"(IC {summ['momentum']['ic_mean']:+.4f}), and short-term reversal, which the model also relies on, did not work.")
out("- **The net Sharpe interval only just excludes 0** (lower end "
    f"{gs['net_lo']:+.2f}), and Phase 6's placebo runs showed these intervals are too narrow.")
out("- **Overall:** the hold-out does not show a tradable edge that survives the known biases. A clean test needs "
    "point-in-time index membership (e.g. CRSP), which this project could not obtain for free.\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = {"gbt": "#C44E52", "ridge": "#4C72B0", "lasso": "#8172B2", "momentum": "#55A868", "reversal": "#DD8452"}
for s in SIGNALS:
    r = bk[s][1].sort_index()
    axes[0].plot(r.index, (r["net"] / H).cumsum(), label=s, color=colors[s], lw=2 if s == "gbt" else 1.2)
axes[0].axhline(0, color="k", lw=0.6); axes[0].legend(fontsize=8)
axes[0].set_title("Hold-out: cumulative net return, staggered long-short book")
yr.plot.bar(ax=axes[1], color=[colors[s] for s in SIGNALS], width=0.8)
axes[1].axhline(0, color="k", lw=0.6); axes[1].set_title("Hold-out mean IC by year"); axes[1].legend(fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig(ROOT / "figures/phase7_holdout.png", dpi=150); plt.close(fig)
out("![hold-out](../figures/phase7_holdout.png)")
(ROOT / "reports/phase7_holdout.md").write_text("\n".join(lines) + "\n")
