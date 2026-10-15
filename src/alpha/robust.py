"""Robustness tools: placebo label shuffling and feature-family permutation importance."""
import numpy as np
import pandas as pd

FAMILIES = {
    "momentum": ["mom_12_1", "mom_6_1", "high_52w"],
    "reversal": ["rev_1m", "rev_1w"],
    "volatility": ["vol_1m", "vol_3m", "idio_vol_3m"],
    "beta": ["beta_1y"],
    "volume": ["volume_trend"],
    "size_liquidity": ["size_proxy", "amihud_1m"],
}


def shuffle_within_date(s, seed):
    """Permute a (date, ticker) Series across tickers within each date (keeps each date's values)."""
    rng = np.random.default_rng(seed)
    out = s.copy()
    codes = s.index.get_level_values("date")
    vals = out.to_numpy().copy()
    for idx in pd.Series(np.arange(len(s))).groupby(codes).indices.values():
        vals[idx] = vals[rng.permutation(idx)]
    out[:] = vals
    return out


def permute_family(x, cols, seed):
    """Shuffle the columns `cols` jointly across stocks within each date (rows keep their other columns)."""
    rng = np.random.default_rng(seed)
    out = x.copy()
    block = out[cols].to_numpy().copy()
    for idx in pd.Series(np.arange(len(x))).groupby(x.index.get_level_values("date")).indices.values():
        block[idx] = block[rng.permutation(idx)]
    out[cols] = block
    return out
