"""Phase 4 figure from the trial log and saved predictions (no test-year labels used).

Run: .venv/bin/python scripts/make_phase4_figure.py   -> figures/phase4_models.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["ridge", "lasso", "gbt"]
SIGNALS = MODELS + ["momentum", "reversal"]
COLORS = {"ridge": "#4C72B0", "lasso": "#8172B2", "gbt": "#C44E52"}

trials = pd.read_csv(ROOT / "reports/trials.csv")
pred = pd.read_parquet(ROOT / "predictions/dev_predictions.parquet", columns=SIGNALS + ["fold"])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
# a) every grid point's inner-validation IC per fold, chosen one highlighted
ax = axes[0]
folds = sorted(trials["fold"].unique())
for i, m in enumerate(MODELS):
    t = trials[trials["model"] == m]
    for j, f in enumerate(folds):
        v = t[t["fold"] == f]["val_ic_mean"]
        x = j + (i - 1) * 0.25
        ax.scatter([x] * len(v), v, s=12, color=COLORS[m], alpha=0.35)
        ax.scatter([x], [v.max()], s=40, color=COLORS[m], edgecolor="k", label=m if j == 0 else None)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(len(folds)), folds); ax.set_xlabel("test year (validation = the 12 months before)")
ax.set_ylabel("mean daily IC on inner validation")
ax.set_title("Tuning: every grid point (faint) and the chosen one (outlined)")
ax.legend(fontsize=8)
# b) cross-sectional spread of predictions per test year
ax = axes[1]
sd = pred.groupby([pred.index.get_level_values("date"), "fold"])[MODELS].std().groupby("fold").median()
for m in MODELS:
    ax.plot(sd.index, sd[m], marker="o", color=COLORS[m], label=m)
ax.set_ylabel("median cross-sectional sd of predictions")
ax.set_xlabel("test year"); ax.set_title("Predictions are never constant"); ax.legend(fontsize=8)
# c) agreement between models
ax = axes[2]
corr = pred[SIGNALS].groupby(level="date").corr(method="spearman").groupby(level=1).mean().loc[SIGNALS, SIGNALS]
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(SIGNALS)), SIGNALS, rotation=45, ha="right"); ax.set_yticks(range(len(SIGNALS)), SIGNALS)
for i in range(len(SIGNALS)):
    for j in range(len(SIGNALS)):
        ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Average daily rank correlation\nbetween predictions")
fig.suptitle("Phase 4: walk-forward models, 2015-2022 test years (no test labels used in this figure)", y=1.0)
fig.tight_layout(); fig.savefig(ROOT / "figures/phase4_models.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("figures/phase4_models.png")
