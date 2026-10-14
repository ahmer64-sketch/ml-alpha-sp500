"""Phase 1: data and universe report (the data card's evidence).

Run: .venv/bin/python scripts/run_phase1.py
Writes reports/phase1_data.md, figures/phase1_coverage.png. The hold-out period (2023+) gets
data-quality checks only.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha import data

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
DEV = (pd.Timestamp("2013-01-01"), pd.Timestamp("2022-12-31"))
HOLD = pd.Timestamp("2023-01-01")
lines = []


def out(s=""):
    print(s)
    lines.append(s)


cons_path = sorted(RAW.glob("sp500_constituents_*.csv"))[-1]
price_path = sorted(RAW.glob("prices_*.parquet"))[-1]
cons = pd.read_csv(cons_path)
raw, meta = data.load_prices(price_path)
_, raw_issues, _, _ = data.quality_report(raw)
df, clean_log = data.clean(raw)
calendar, issues, cov, first = data.quality_report(df)
stocks = [t for t in cov.index if t != "SPY"]

out("# Phase 1: data and universe\n")
out(f"- Constituents: `{cons_path.name}`, {len(cons)} tickers (today's S&P 500, from Wikipedia).")
out(f"- Prices: `{price_path.name}` (fetched {meta['fetched_at_utc'][:19]} UTC; {meta['source']}), "
    f"{meta['rows']:,} daily rows, {meta['start']} to {calendar.max():%Y-%m-%d}, {len(calendar):,} trading days.")
out(f"- Tickers with no data: {len(meta['missing'])}" + (f" ({', '.join(meta['missing'])})" if meta["missing"] else "") + ".")
out(f"- Development period {DEV[0]:%Y-%m-%d} to {DEV[1]:%Y-%m-%d}; hold-out from {HOLD:%Y-%m-%d} "
    "(locked: data-quality checks only).\n")

out("## 1. Universe and its biases\n")
out("| Sector | Stocks |")
out("|---|---|")
for s, n in cons["sector"].value_counts().items():
    out(f"| {s} | {n} |")
added = pd.to_datetime(cons["added"], errors="coerce")
late_add = (added > DEV[0]).sum()
late_list = int((first[stocks] > pd.Timestamp("2012-03-01")).sum())
out(f"\n**Survivorship and membership look-ahead.** The universe is today's members. Stocks that left the index "
    f"since 2012 (acquired, bankrupt or demoted) are missing, and **{late_add} of {len(cons)} current members "
    f"({late_add / len(cons):.0%}) joined the index after {DEV[0]:%Y}**: they are included before they were members, "
    "and they were added *because* they had grown. Both effects tilt the universe towards past winners, which "
    "flatters any signal that buys recent winners (momentum). A point-in-time membership list (e.g. CRSP) would fix "
    "this; it is not freely available, so the bias is stated in every result.\n")
out(f"**Late listings:** {late_list} stocks have no prices in early 2012 (IPOs or spin-offs since), e.g. "
    + ", ".join(f"{t} ({first[t]:%Y-%m})" for t in first[stocks].sort_values().index[-5:])
    + ". They enter the model only once they have enough history for their features.\n")

out("## 2. Data quality\n")
out("The raw snapshot is never edited. `data.clean` applies three deterministic fixes, each logged:\n")
out("| Ticker | Date | Action | Detail |")
out("|---|---|---|---|")
for _, r in clean_log.sort_values(["action", "ticker"]).iterrows():
    out(f"| {r['ticker']} | {r['date']:%Y-%m-%d} | {r['action']} | {r['detail']} |")
counts, raw_counts = issues["check"].value_counts(), raw_issues["check"].value_counts()
out("\nChecks on the cleaned data (raw snapshot in brackets):\n")
out("| Check | Result |")
out("|---|---|")
out(f"| Stocks with prices | {len(stocks)} (+ SPY) |")
out(f"| Coverage after each stock's first price (median / minimum) | {cov[stocks].median():.2%} / "
    f"{cov[stocks].min():.2%} ({cov[stocks].idxmin()}) |")
for name, check in [("Stocks with missing days after listing", "missing days"),
                    ("Stale prices (5+ unchanged closes)", "stale price"), ("One-day moves beyond ±40% (log)", "large move")]:
    out(f"| {name} | {counts.get(check, 0)} ({raw_counts.get(check, 0)}) |")
mv = issues[issues["check"] == "large move"].sort_values("date")
if len(mv):
    out("\nLarge one-day moves (kept: real events matter for the model; flagged so they can be checked):\n")
    out("| Date | Ticker | Detail | Period |")
    out("|---|---|---|---|")
    for _, r in mv.iterrows():
        period = "hold-out" if r["date"] >= HOLD else "development"
        out(f"| {r['date']:%Y-%m-%d} | {r['ticker']} | {r['detail']} | {period} |")
out("")

out("## 3. Liquidity (development period)\n")
dv = (data.wide(df, "close") * data.wide(df, "volume")).loc[DEV[0]:DEV[1]].reindex(columns=stocks)
med = dv.median() / 1e6
med = med.dropna()
low = med[med < 10].sort_values()
out(f"Median daily dollar volume per stock (cleaned data): lowest {med.min():,.1f}m ({med.idxmin()}), median "
    f"{med.median():,.0f}m, highest {med.max():,.0f}m ({med.idxmax()}). {len(low)} of {len(med)} stocks trade under "
    "$10m a day in this period" + (f" ({', '.join(f'{t} {v:.1f}m' for t, v in low.items())})" if len(low) else "")
    + ": smaller companies, or (FERG) mainly traded abroad, before they joined the index (the look-ahead above). Phase 5 therefore uses "
    "costs that rise for less liquid stocks rather than one flat figure.\n")

# figure: stocks with data per date
n_by_date = data.wide(df, "adj_close").loc[calendar, stocks].notna().sum(axis=1)
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(n_by_date.index, n_by_date.values, color="#4C72B0")
ax.axvspan(HOLD, n_by_date.index.max(), color="grey", alpha=0.15, label="hold-out (locked)")
ax.set_ylabel("stocks with a price"); ax.set_title("Universe size over time (today's members, so it grows)")
ax.legend()
fig.tight_layout(); fig.savefig(ROOT / "figures/phase1_coverage.png", dpi=150); plt.close(fig)
out("![coverage](../figures/phase1_coverage.png)")
(ROOT / "reports/phase1_data.md").write_text("\n".join(lines) + "\n")
