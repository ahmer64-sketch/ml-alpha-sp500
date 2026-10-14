"""Phase 2: features and label on the real data, plus a real-data truncation (leakage) check.

Run: .venv/bin/python scripts/run_phase2.py
Writes data/processed/features_h5.parquet (not in git), reports/phase2_features.md,
figures/phase2_universe.png, figures/phase2_feature_corr.png. Statistics use the development period only.
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha import data, features as F

ROOT = Path(__file__).resolve().parents[1]
DEV = (pd.Timestamp("2013-01-01"), pd.Timestamp("2022-12-31"))
CHECK_DATES = ["2015-06-30", "2019-03-29", "2022-06-30"]
lines = []


def out(s=""):
    print(s)
    lines.append(s)


raw, meta = data.load_prices(sorted((ROOT / "data/raw").glob("prices_*.parquet"))[-1])
df, _ = data.clean(raw)
t0 = time.perf_counter()
p = F.panel(df)
f, eligible = F.raw_features(p)
z = F.normalise(f, eligible)
y_raw, y_rank = F.label(p, eligible, 5)
panel_long = F.build(df)
secs = time.perf_counter() - t0
panel_long.to_parquet(ROOT / "data/processed/features_h5.parquet")

out("# Phase 2: features and label\n")
out(f"Built {len(F.FEATURES)} features and the 5-day label for {len(panel_long):,} eligible stock-days "
    f"({panel_long.index.get_level_values('date').nunique():,} dates) in {secs:.0f} s. Definitions are pre-registered "
    "in [DESIGN.md](../DESIGN.md); code in `src/alpha/features.py`.\n")

# 1. leakage check on the real data
out("## 1. Leakage check on the real data\n")
out("Features are recomputed from data cut off at the close of date t and compared with the full-sample values "
    "at t (raw values, eligibility and normalised ranks). Any use of later data would show up as a difference.\n")
out("| Cut-off date t | Eligible stocks | Values compared | Max abs difference | Result |")
out("|---|---|---|---|---|")
all_ok = True
for d in CHECK_DATES:
    t = p["adj"].index.get_indexer([pd.Timestamp(d)], method="pad")[0]
    q = {k: v.iloc[: t + 1] for k, v in p.items()}
    f2, el2 = F.raw_features(q)
    z2 = F.normalise(f2, el2)
    diffs, n = [], 0
    same_el = el2.iloc[t].equals(eligible.iloc[t])
    for k in F.FEATURES:
        for a, b in ((f[k].iloc[t], f2[k].iloc[t]), (z[k].iloc[t], z2[k].iloc[t])):
            m = a.notna() | b.notna()
            both = a.notna() & b.notna()
            if (a.notna() != b.notna())[m].any():
                diffs.append(np.inf)
            diffs.append(float((a[both] - b[both]).abs().max()))
            n += int(both.sum())
    mx = max(diffs)
    ok = same_el and mx < 1e-9
    all_ok &= ok
    out(f"| {p['adj'].index[t]:%Y-%m-%d} | {int(eligible.iloc[t].sum())} | {n:,} | {mx:.1e} | {'pass' if ok else 'FAIL'} |")
out(f"\n{'All dates pass' if all_ok else 'A check FAILED'}: features at t do not change when everything after t is "
    "removed. The unit tests (`tests/test_features.py`) add: the same with prices after t randomly perturbed; the "
    "label at t changes only when the closes at t+1 or t+1+5 change; and a planted look-ahead feature is caught.\n")

# 2. universe size
dev = slice(*DEV)
n_el = eligible.sum(axis=1)
out("## 2. Eligible universe\n")
r_all = np.log(p["adj"]).diff()
short = (r_all.rolling(252, min_periods=1).count() < 240).loc[dev]
excl = (p["adj"].notna() & ~eligible).loc[dev].drop(columns=F.MARKET)
n_ex, n_short = int(excl.to_numpy().sum()), int((excl & short.drop(columns=F.MARKET)).to_numpy().sum())
out(f"Eligible stocks per date (development period): {n_el[dev].min()} to {n_el[dev].max()}, median "
    f"{n_el[dev].median():.0f}. It grows because the universe is today's members (Phase 1). Of {n_ex:,} stock-days "
    f"with a price but not eligible, {n_short / n_ex:.0%} have under 240 returns in the past year (recent listings) "
    f"and {1 - n_short / n_ex:.0%} only fail the $5 price floor.\n")

# 3. missing values before the median fill
out("## 3. Feature coverage (development period, eligible stock-days)\n")
out("| Feature | Missing before fill | Median | 5th / 95th percentile (raw) |")
out("|---|---|---|---|")
el_dev = eligible.loc[dev]
for k in F.FEATURES:
    v = f[k].loc[dev].where(el_dev).stack().dropna()
    miss = 1 - len(v) / el_dev.to_numpy().sum()
    out(f"| {k} | {miss:.2%} | {v.median():.4g} | {v.quantile(.05):.4g} / {v.quantile(.95):.4g} |")
out("")

# 4. feature correlations
dl = panel_long.loc[DEV[0]:DEV[1]]
corr = sum(g[F.FEATURES].corr(method="pearson") for _, g in dl.groupby(level="date")) / dl.index.get_level_values(0).nunique()
pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False)
out("## 4. Feature correlations\n")
out("Average cross-sectional correlation of the rank-normalised features. Most correlated pairs:\n")
out("| Pair | Correlation |")
out("|---|---|")
for (a, b), c in pairs.head(6).items():
    out(f"| {a} / {b} | {c:+.2f} |")
out("\nThe volatility features, and size with illiquidity, carry overlapping information. All are kept (no "
    "selection on results, as pre-registered); ridge and trees handle correlated inputs, and Phase 6 permutation "
    "importance is grouped by feature family for this reason.\n")
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(F.FEATURES)), F.FEATURES, rotation=60, ha="right"); ax.set_yticks(range(len(F.FEATURES)), F.FEATURES)
for i in range(len(F.FEATURES)):
    for j in range(len(F.FEATURES)):
        ax.text(j, i, f"{corr.iat[i, j]:.1f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, shrink=0.8); ax.set_title("Average cross-sectional feature correlation, 2013-2022")
fig.tight_layout(); fig.savefig(ROOT / "figures/phase2_feature_corr.png", dpi=150); plt.close(fig)
out("![correlations](../figures/phase2_feature_corr.png)\n")

# 5. label
yl = dl["y_raw"].dropna()
out("## 5. Label\n")
out(f"5-day excess log return from close t+1 to close t+6 (development period, {len(yl):,} values): sd "
    f"{yl.std():.2%}, 1st / 99th percentile {yl.quantile(.01):+.1%} / {yl.quantile(.99):+.1%}, "
    f"kurtosis {yl.kurt():.0f}. The fat tails are why models train on the cross-sectional rank instead "
    "(uniform on [-0.5, 0.5]); the raw value is kept for R² and portfolio returns. Labels overlap for 5 days, "
    "so Phase 3 purges and embargoes around every split.\n")

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.plot(n_el.index, n_el.values, color="#4C72B0")
ax.axvspan(pd.Timestamp("2023-01-01"), n_el.index.max(), color="grey", alpha=0.15, label="hold-out (locked)")
ax.set_ylabel("eligible stocks"); ax.set_title("Eligible universe per date"); ax.legend()
fig.tight_layout(); fig.savefig(ROOT / "figures/phase2_universe.png", dpi=150); plt.close(fig)
out("![universe](../figures/phase2_universe.png)")
(ROOT / "reports/phase2_features.md").write_text("\n".join(lines) + "\n")
