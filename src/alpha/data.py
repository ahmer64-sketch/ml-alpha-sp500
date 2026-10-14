"""Universe and price data.

Every analysis runs on saved snapshots in data/raw (constituents CSV, prices parquet + JSON
metadata), never on a live download, so results are reproducible even though Yahoo's history is
revised over time. yfinance is community-maintained and not affiliated with Yahoo.
"""
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_constituents(out_dir):
    """Today's S&P 500 members with GICS sector, saved with the fetch date."""
    import urllib.request

    req = urllib.request.Request(WIKI_SP500, headers={"User-Agent": "Mozilla/5.0 (research snapshot)"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    table = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
    df = pd.DataFrame({
        "ticker": table["Symbol"].str.replace(".", "-", regex=False),   # Yahoo uses BRK-B, not BRK.B
        "name": table["Security"],
        "sector": table["GICS Sector"],
        "added": table.get("Date added"),
    })
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = Path(out_dir) / f"sp500_constituents_{stamp}.csv"
    df.to_csv(path, index=False)
    return path


def fetch_prices(tickers, start, end, out_dir, batch=100):
    """Daily OHLCV + adjusted close for `tickers` (long format), saved as parquet + JSON metadata."""
    import yfinance as yf

    fetched_at = datetime.now(timezone.utc)
    frames, missing = [], []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        raw = yf.download(chunk, start=start, end=end, auto_adjust=False, progress=False, group_by="ticker",
                          threads=True)
        for t in chunk:
            if t not in raw.columns.get_level_values(0):
                missing.append(t)
                continue
            d = raw[t].dropna(how="all")
            if d.empty:
                missing.append(t)
                continue
            d = d.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                                  "Adj Close": "adj_close", "Volume": "volume"})
            frames.append(d.assign(ticker=t).rename_axis("date").reset_index())
    df = pd.concat(frames, ignore_index=True)[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    meta = {"fetched_at_utc": fetched_at.isoformat(), "start": start, "end": end, "requested": len(tickers),
            "missing": missing, "rows": len(df), "source": "Yahoo Finance via yfinance (auto_adjust=False)"}
    stamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    path = Path(out_dir) / f"prices_{stamp}.parquet"
    df.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def load_prices(path):
    path = Path(path)
    return pd.read_parquet(path), json.loads(path.with_suffix(".json").read_text())


def wide(df, field):
    return df.pivot(index="date", columns="ticker", values=field).sort_index()


def quality_report(df, calendar_min_share=0.5, jump=0.4, stale_run=5):
    """Return (calendar, issues DataFrame, coverage Series) for the long price frame."""
    adj = wide(df, "adj_close")
    calendar = adj.index[adj.notna().mean(axis=1) >= calendar_min_share]
    adj = adj.loc[calendar]
    first = adj.apply(lambda s: s.first_valid_index())
    # coverage counted from each stock's own first date (IPOs after 2012 are not "missing" before they list)
    cov = pd.Series({t: adj.loc[first[t]:, t].notna().mean() for t in adj.columns})
    issues = []
    lr = np.log(adj).diff()
    close = wide(df, "close").loc[calendar, adj.columns]
    ratio_jump = (close / adj).pct_change().abs()
    for t in adj.columns:
        for d in lr.index[lr[t].abs() > jump]:
            note = "; close/adj ratio also jumped (split or corporate action)" if ratio_jump.at[d, t] > 0.2 else ""
            issues.append((t, d, "large move", f"log return {lr.at[d, t]:+.1%}{note}"))
        same = (close[t].diff() == 0).astype(int)
        runs = same.groupby((same == 0).cumsum()).cumsum()
        for d in runs.index[runs == stale_run]:
            issues.append((t, d, "stale price", f"{stale_run}+ unchanged closes"))
        gaps = adj.loc[first[t]:, t].isna()
        if gaps.any():
            issues.append((t, gaps.idxmax(), "missing days", f"{int(gaps.sum())} days missing after first price"))
    return calendar, pd.DataFrame(issues, columns=["ticker", "date", "check", "detail"]), cov, first


def clean(df, window=60, max_zero_share=0.2, jump=0.4, ratio_jump=0.2):
    """Deterministic fixes on the raw snapshot; returns (clean long frame, log of actions).

    1. Placeholder history: rows up to the last zero-volume day of any stretch where more than
       `max_zero_share` of the trailing `window` days had zero volume are dropped (Yahoo back-fills some tickers with mostly untraded prices
       before their current listing, e.g. SW before 2024, FERG before 2021).
    2. Remaining zero-volume rows are dropped (no trade, so no price): they become missing days.
    3. Broken adjustment: a day with a large adjusted move, a jump in the close/adjusted ratio and a normal
       unadjusted move (e.g. DHR at the 2016 Fortive spin-off) gets the unadjusted return; all earlier
       adjusted prices are rescaled so later returns are unchanged.
    """
    out, log = [], []
    for t, d in df.sort_values("date").groupby("ticker", sort=False):
        d = d.reset_index(drop=True)
        zero = (d["volume"] <= 0).to_numpy()
        share = pd.Series(zero, dtype=float).rolling(window, min_periods=1).mean().to_numpy()
        long_end = np.flatnonzero(share > max_zero_share)
        if len(long_end):
            cut = np.flatnonzero(zero[: long_end[-1] + 1])[-1] + 1     # first day after the last untraded one
            log.append((t, d.at[cut - 1, "date"], "placeholder history dropped",
                        f"{cut} rows up to {d.at[cut - 1, 'date']:%Y-%m-%d}"))
            d = d.iloc[cut:].reset_index(drop=True)
        z = d["volume"] <= 0
        if z.any():
            log.append((t, d.loc[z, "date"].iloc[0], "zero-volume days dropped", f"{int(z.sum())} rows"))
            d = d[~z].reset_index(drop=True)
        r_adj = d["adj_close"] / d["adj_close"].shift()
        r_cl = d["close"] / d["close"].shift()
        ratio = (d["close"] / d["adj_close"]).pct_change().abs()
        bad = (np.log(r_adj).abs() > jump) & (ratio > ratio_jump) & (np.log(r_cl).abs() < jump)
        for i in np.flatnonzero(bad.to_numpy()):
            f = r_adj.iat[i] / r_cl.iat[i]
            d.loc[: i - 1, "adj_close"] *= f
            log.append((t, d.at[i, "date"], "adjustment repaired",
                        f"adjusted return {r_adj.iat[i] - 1:+.1%} replaced by unadjusted {r_cl.iat[i] - 1:+.1%}"))
        out.append(d)
    return pd.concat(out, ignore_index=True), pd.DataFrame(log, columns=["ticker", "date", "action", "detail"])
