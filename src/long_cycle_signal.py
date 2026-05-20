"""Long-cycle (3-5yr) signal module for DH Dyn 2x3x [A] extension.

All signals are strictly backward-looking; no future leak.
lev in DH Dyn is in [0.0, 1.0] scale (multiplied by BASE_LEV=3 inside build_nav).
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Raw signal computations (all return z-score or centered rank in [-3, 3])
# ---------------------------------------------------------------------------

def compute_lt1(close: pd.Series, N: int) -> pd.Series:
    """LT1: Price-to-N-year-MA log-ratio, normalized by rolling std.
    Positive z = expensive (above MA) → contrarian down-weight.
    """
    ma_N  = close.rolling(N, min_periods=N).mean()
    log_r = np.log(close / ma_N)
    sigma = log_r.rolling(N, min_periods=N).std().replace(0.0, np.nan)
    z     = (log_r / sigma).clip(-3.0, 3.0).fillna(0.0)
    return z


def compute_lt2(close: pd.Series, N: int) -> pd.Series:
    """LT2: N-year momentum z-score vs 2N-day rolling distribution.
    Positive z = strong long-run momentum → contrarian down-weight.
    """
    mom_N = close / close.shift(N) - 1.0
    mu    = mom_N.rolling(2 * N, min_periods=N).mean()
    sigma = mom_N.rolling(2 * N, min_periods=N).std().replace(0.0, np.nan)
    z     = ((mom_N - mu) / sigma).clip(-3.0, 3.0).fillna(0.0)
    return z


def compute_lt3(close: pd.Series, N: int) -> pd.Series:
    """LT3: Expanding-window percentile rank of N-year CAGR, centered to [-1, +1].
    +1 = best N-yr CAGR ever → contrarian down-weight.
    """
    cagr_N   = (close / close.shift(N)) ** (TRADING_DAYS / N) - 1.0
    rank_t   = cagr_N.expanding(min_periods=N).rank(pct=True)
    centered = ((rank_t - 0.5) * 2.0).fillna(0.0)
    return centered


LT_SIGNALS = {'LT1': compute_lt1, 'LT2': compute_lt2, 'LT3': compute_lt3}


def build_lt_signal(close: pd.Series, signal_name: str, N: int) -> pd.Series:
    return LT_SIGNALS[signal_name](close, N)


# ---------------------------------------------------------------------------
# Conversion to multiplier (Mode A) or bias (Mode B)
# Both are contrarian: high signal → reduce exposure.
# ---------------------------------------------------------------------------

def signal_to_mult(signal: pd.Series, k_lt: float) -> pd.Series:
    """Mode A: contrarian multiplier in [0.5, 1.5]. Applied to raw_a2 before rebalancing.
    signal: z-score (LT1/LT2, range ±3) or centered rank (LT3, range ±1).
    """
    return (1.0 - k_lt * signal).clip(0.5, 1.5)


def signal_to_bias(signal: pd.Series, k_lt: float) -> pd.Series:
    """Mode B: additive bias to lev array (which is in [0, 1] scale).
    Max bias = ±(k_lt * 0.5) to keep within lev scale.
    """
    return (-k_lt * signal * 0.5).clip(-0.5, 0.5)


# ---------------------------------------------------------------------------
# Integration into DH Dyn pipeline
# ---------------------------------------------------------------------------

def apply_lt_mode_a(raw_a2: pd.Series, lt_mult: pd.Series) -> pd.Series:
    """Mode A: multiply raw_a2 by lt_mult before simulate_rebalance_A.
    Both must be indexed consistently. Re-clip to [0, 1].
    """
    return (raw_a2 * lt_mult).clip(0.0, 1.0).fillna(0.0)


def apply_lt_mode_b(lev_array: np.ndarray, lt_bias: pd.Series,
                    l_min: float = 0.0, l_max: float = 1.0) -> np.ndarray:
    """Mode B: add lt_bias to post-rebalance lev array, clip to [0, 1].
    Does not change wn/wg/wb — only scales the NASDAQ sleeve.
    """
    biased = lev_array + lt_bias.values
    return np.clip(biased, l_min, l_max)
