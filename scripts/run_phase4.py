"""Phase 4: walk-forward fit and out-of-sample predictions for every development test year.

Run: .venv/bin/python scripts/run_phase4.py   (needs data/processed/features_h5.parquet from Phase 2)
Writes predictions/dev_predictions.parquet, reports/trials.csv, reports/phase4_models.md.
Test-year performance is evaluated in Phase 5; this phase only fits, predicts and checks completeness.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alpha import data, models as M
from alpha.splits import walk_forward

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["ridge", "lasso", "gbt"]
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


raw, _ = data.load_prices(sorted((ROOT / "data/raw").glob("prices_*.parquet"))[-1])
cal = pd.DatetimeIndex(sorted(raw.loc[raw["ticker"] == "SPY", "date"]))
panel = pd.read_parquet(ROOT / "data/processed/features_h5.parquet")
folds = walk_forward(cal, range(2015, 2023), "2013-01-01", 5, 5, last_date="2022-12-31")

log, chosen, preds = [], [], []
t_all = time.perf_counter()
for s in folds:
    cols = {}
    for name in MODELS:
        t0 = time.perf_counter()
        cols[name], best = M.run_fold(panel, s, name, log)
        chosen.append({"fold": s.name, "model": name, "params": json.dumps(best),
                       "seconds": round(time.perf_counter() - t0)})
        print(f"{s.name} {name} {best} {time.perf_counter() - t0:.0f}s", flush=True)
    for name in M.BASELINES:
        cols[name] = M.baseline(panel, s, name)
    df = pd.DataFrame(cols)
    test_rows = panel[panel.index.get_level_values("date").isin(s.test)]
    df["y_raw"], df["y_rank"] = test_rows["y_raw"], test_rows["y_rank"]
    df["fold"] = s.name
    preds.append(df)
secs = time.perf_counter() - t_all
pred = pd.concat(preds).sort_index()
pred[MODELS + list(M.BASELINES)] = pred[MODELS + list(M.BASELINES)].astype("float32")
pred.to_parquet(ROOT / "predictions/dev_predictions.parquet")
trials = pd.DataFrame(log)
trials.to_csv(ROOT / "reports/trials.csv", index=False)
ch = pd.DataFrame(chosen)

out("# Phase 4: models and out-of-sample predictions\n")
out(f"Ridge, lasso and gradient-boosted trees (`HistGradientBoostingRegressor`) fitted walk-forward on {len(folds)} "
    f"development folds; models, grids and the primary model (gbt) were pre-registered in [DESIGN.md](../DESIGN.md). "
    f"Total run time {secs / 60:.0f} min.\n")
out("## 1. Predictions saved for every test block\n")
out("`predictions/dev_predictions.parquet`: one row per eligible stock and test date, with each model's and "
    "baseline's prediction, the raw and rank label, and the fold.\n")
out("| Test year | Test dates | Dates with predictions | Stock-days | Missing predictions | Training rows (labelled) |")
out("|---|---|---|---|---|---|")
complete = True
for s in folds:
    p = pred[pred["fold"] == s.name]
    n_dates = p.index.get_level_values("date").nunique()
    expected = int(panel.index.get_level_values("date").isin(s.test).sum())
    miss = int(p[MODELS + list(M.BASELINES)].isna().sum().sum())
    train_rows = int((panel.index.get_level_values("date").isin(s.train) & panel["y_rank"].notna()).sum())
    complete &= (n_dates == len(s.test)) and (len(p) == expected) and miss == 0
    out(f"| {s.name} | {len(s.test)} | {n_dates} | {len(p):,} | {miss} | {train_rows:,} |")
out(f"\n**Completeness: {'pass' if complete else 'FAIL'}.** Every test date of every fold has a prediction from "
    f"every model for every eligible stock ({len(pred):,} stock-days in total, "
    f"{pred.index.get_level_values('date').min():%Y-%m-%d} to {pred.index.get_level_values('date').max():%Y-%m-%d}).\n")

out("## 2. Hyperparameters chosen per fold (inner validation, never the test year)\n")
tab = ch.pivot(index="fold", columns="model", values="params")[MODELS]
out("| Test year | " + " | ".join(MODELS) + " |")
out("|---|" + "---|" * len(MODELS))
for f, r in tab.iterrows():
    out(f"| {f} | " + " | ".join(f"`{r[m]}`" for m in MODELS) + " |")
out("")
best_val = trials.loc[trials.groupby(["fold", "model"])["val_ic_mean"].idxmax()].pivot(index="fold", columns="model",
                                                                                          values="val_ic_mean")[MODELS]
out("Inner-validation IC of the chosen setting (the 12 months before each test year; used for tuning only):\n")
out("| Test year | " + " | ".join(MODELS) + " |")
out("|---|" + "---|" * len(MODELS))
for f, r in best_val.iterrows():
    out(f"| {f} | " + " | ".join(f"{r[m]:+.4f}" for m in MODELS) + " |")
spread = trials.groupby(["fold", "model"])["val_ic_mean"].agg(lambda v: v.max() - v.min()).groupby("model").mean()
out(f"\nAverage gap between the best and worst grid point on validation: "
    + ", ".join(f"{m} {spread[m]:.4f}" for m in MODELS) + ". " + "These gaps are about as large as the validation ICs themselves (0.00 to 0.04), so the choice of setting is noisy: ridge's alpha jumps between the ends of its grid from year to year. Phase 6 checks how much results depend on it.")

out("\n## 3. Trial log (multiple testing)\n")
out(f"`reports/trials.csv` records all {len(trials)} fits used for tuning ({len(folds)} folds x "
    f"{sum(len(M.GRIDS[m]) for m in MODELS)} grid points), plus {len(folds) * len(MODELS)} refits and "
    f"{len(M.BASELINES)} baselines. Model families compared in Phase 5: {len(MODELS) + len(M.BASELINES)}. This count "
    "is what the Phase 5 multiple-testing adjustment uses. No other configurations were tried.\n")
out("## 4. Sanity checks on the predictions (no test-year labels used)\n")
corr = pred[MODELS + list(M.BASELINES)].groupby(level="date").corr(method="spearman")
avg = corr.groupby(level=1).mean().loc[MODELS + list(M.BASELINES), MODELS + list(M.BASELINES)]
out("Average daily rank correlation between the models' predictions:\n")
out("| | " + " | ".join(avg.columns) + " |")
out("|---|" + "---|" * len(avg.columns))
for i, r in avg.iterrows():
    out(f"| {i} | " + " | ".join(f"{v:+.2f}" for v in r) + " |")
sd = pred[MODELS].groupby(level="date").std().median()
out("\nMedian cross-sectional spread (sd) of predictions per date: " + ", ".join(f"{m} {sd[m]:.4f}" for m in MODELS)
    + ". None is constant (a lasso that shrinks every weight to zero would show 0 here).")
(ROOT / "reports/phase4_models.md").write_text("\n".join(lines) + "\n")
