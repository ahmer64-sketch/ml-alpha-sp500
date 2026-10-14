"""Phase 3: walk-forward split design, checked on the real calendar and panel.

Run: .venv/bin/python scripts/run_phase3.py   (needs data/processed/features_h5.parquet from Phase 2)
Writes reports/phase3_splits.md and figures/phase3_splits.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from alpha import data
from alpha.splits import walk_forward

ROOT = Path(__file__).resolve().parents[1]
H, EMB = 5, 5
lines = []


def out(s=""):
    print(s)
    lines.append(s)


raw, _ = data.load_prices(sorted((ROOT / "data/raw").glob("prices_*.parquet"))[-1])
cal = pd.DatetimeIndex(sorted(raw.loc[raw["ticker"] == "SPY", "date"]))
panel = pd.read_parquet(ROOT / "data/processed/features_h5.parquet", columns=["y_rank"])
n_per_date = panel.groupby(level="date").size()
dev = walk_forward(cal, range(2015, 2023), "2013-01-01", H, EMB, last_date="2022-12-31")
hold = walk_forward(cal, [2023, 2024, 2025, 2026], "2013-01-01", H, EMB, last_date=cal[-1])


def rows(d):
    return int(n_per_date.reindex(d).fillna(0).sum())


def check(s):
    """Real-calendar check: last training label ends > EMB days before the test block (also inner split)."""
    end = lambda d: cal.get_loc(d) + 1 + H
    gap = cal.get_loc(s.test[0]) - end(s.train[-1])
    inner_gap = cal.get_loc(s.val[0]) - end(s.inner_train[-1])
    return gap, inner_gap


out("# Phase 3: split design\n")
out(f"Walk-forward with an expanding training window, yearly test blocks, purge and embargo for the {H}-day "
    "label (rules pre-registered in [DESIGN.md](../DESIGN.md); code in `src/alpha/splits.py`).\n")
out("## Development folds\n")
out("| Test year | Training dates | Last training date | Inner validation | Test dates | First / last test date | "
    "Train / test stock-days | Untouched trading days between last training label and test (embargo = 5) |")
out("|---|---|---|---|---|---|---|---|")
ok = True
for s in dev + hold:
    gap, inner_gap = check(s)
    ok &= gap > EMB and inner_gap > EMB
    tag = " (hold-out, locked)" if int(s.name) >= 2023 else ""
    if s is hold[0]:
        out("\n## Hold-out folds (dates only; run once, after Phase 6)\n")
        out("| Test year | Training dates | Last training date | Inner validation | Test dates | First / last test date | "
            "Train / test stock-days | Untouched trading days between last training label and test (embargo = 5) |")
        out("|---|---|---|---|---|---|---|---|")
    out(f"| {s.name}{tag} | {len(s.train):,} ({s.train[0]:%Y-%m-%d} →) | {s.train[-1]:%Y-%m-%d} | "
        f"{s.val[0]:%Y-%m-%d} → {s.val[-1]:%Y-%m-%d} | {len(s.test)} | {s.test[0]:%Y-%m-%d} / {s.test[-1]:%Y-%m-%d} | "
        f"{rows(s.train):,} / {rows(s.test):,} | {gap - 1} (inner {inner_gap - 1}) |")
last_dev_label = cal[cal.get_loc(dev[-1].test[-1]) + 1 + H]
out(f"\n**Checks on the real calendar:** {'pass' if ok else 'FAIL'}. In every fold the last training label ends "
    f"more than {EMB} trading days (the embargo) before the first test date, and the same holds between the inner "
    f"training part and the validation block. The last development test label ends on {last_dev_label:%Y-%m-%d}, "
    "so no development result uses a 2023 price. Unit tests (`tests/test_splits.py`) also check that the purge is "
    "tight (one more day would break it), that test blocks are disjoint and cover every development date with a "
    "complete label, and that the gap scales with the horizon (h = 1 and 21 for Phase 6).\n")
out("**Why these choices:**")
out(f"- **Purge:** labels span {H} days, so a training date in the last {H} days before the test block would "
    "have a label overlapping test-period returns. Those dates are dropped.")
out(f"- **Embargo ({EMB} days, equal to the horizon):** extra margin for slow-moving effects (e.g. volatility "
    "regimes) that make returns just before and after the boundary related.")
out("- **Expanding window, yearly retraining:** matches how the model would be run in practice; each test year "
    "uses only data available on 1 January of that year (minus the gap).")
out("- **Inner validation = last 12 months of training:** hyperparameters are chosen without ever seeing the "
    "test year, so the test-year results are out-of-sample for the whole pipeline, not just the fitted weights.\n")

# diagram
fig, ax = plt.subplots(figsize=(11, 5.8))
folds = dev + hold
for i, s in enumerate(folds):
    y = len(folds) - i
    ho = int(s.name) >= 2023
    def bar(a, b, color, alpha=1.0, hatch=None):
        ax.barh(y, (b - a).days, left=a, height=0.6, color=color, alpha=alpha, hatch=hatch, edgecolor="white", lw=0)
    bar(s.inner_train[0], s.inner_train[-1], "#4C72B0", 0.5 if ho else 1)
    bar(s.inner_train[-1], s.val[0], "#BBBBBB")
    bar(s.val[0], s.val[-1], "#55A868", 0.5 if ho else 1)
    bar(s.train[-1], s.test[0], "#C44E52")
    bar(s.test[0], s.test[-1], "#DD8452", 0.5 if ho else 1, "//" if ho else None)
ax.set_yticks(range(len(folds), 0, -1), [s.name + (" hold-out" if int(s.name) >= 2023 else "") for s in folds])
ax.axvline(pd.Timestamp("2023-01-01"), color="k", lw=1, ls="--")
ax.text(pd.Timestamp("2023-02-01"), len(folds) + 0.55, "hold-out locked →", fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#4C72B0", label="inner training"), Patch(color="#55A868", label="inner validation (tuning)"),
                   Patch(color="#BBBBBB", label="purge + embargo (inner)"), Patch(color="#C44E52", label="purge + embargo (11 days)"),
                   Patch(color="#DD8452", label="test year")], loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=8, ncol=5, frameon=False)
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_title("Walk-forward splits: expanding training window, yearly test blocks (final fit uses training + validation)")
ax.set_xlim(pd.Timestamp("2012-10-01"), pd.Timestamp("2026-12-31"))
fig.tight_layout(); fig.savefig(ROOT / "figures/phase3_splits.png", dpi=150); plt.close(fig)
out("![splits](../figures/phase3_splits.png)")
(ROOT / "reports/phase3_splits.md").write_text("\n".join(lines) + "\n")
