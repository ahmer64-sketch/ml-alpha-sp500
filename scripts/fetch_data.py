"""Fetch the data snapshot (already committed in data/raw; only needed to refresh it).

Run: .venv/bin/python scripts/fetch_data.py
Writes a new dated constituents CSV and prices parquet + JSON; analyses use the newest snapshot.
"""
from pathlib import Path

import pandas as pd

from alpha import data

RAW = Path(__file__).resolve().parents[1] / "data/raw"
RAW.mkdir(parents=True, exist_ok=True)
cons = pd.read_csv(data.fetch_constituents(RAW))
print(data.fetch_prices(sorted(cons["ticker"]) + ["SPY"], "2012-01-01", "2026-10-01", RAW))
