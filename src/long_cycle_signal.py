"""Long-cycle (3-8yr) signal module for DH Dyn 2x3x [A] extension.

All signals are strictly backward-looking; no future leak.
lev in DH Dyn is in [0.0, 1.0] scale (multiplied by BASE_LEV=3 inside build_nav).

Original signals: LT1, LT2, LT3
Extended signals: LT4 (LT2+LT3 composite), LT6 (vol-adjusted momentum), LT7 (multi-timeframe LT2)
New mode: C (Mode A + Mode B blend with alpha=0.5 split)
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


def compute_lt4(close: pd.Series, N: int) -> pd.Series:
    """LT4: Weighted composite of LT2 (60%) + LT3 (40%).
    Ensemble of z-score momentum and expanding percentile rank.
    No new free parameters beyond the fixed 0.6/0.4 weights.
    """
    s2 = compute_lt2(close, N)
    s3 = compute_lt3(close, N)
    return (0.6 * s2 + 0.4 * s3).clip(-3.0, 3.0)


def compute_lt6(close: pd.Series, N: int) -> pd.Series:
    """LT6: Vol-adjusted N-year momentum (Sharpe-like, AQR style).
    LT6 = cumulative_log_ret(N) / (daily_vol(N) * sqrt(N))
    Positive = strong risk-adjusted performance → contrarian down-weight.
    Normalized by expanding 95th-percentile of |raw| (strictly backward-looking).
    """
    log_ret  = np.log(close).diff()
    roll_ret = log_ret.rolling(N, min_periods=N).sum()
    roll_vol = log_ret.rolling(N, min_periods=N).std() * np.sqrt(N)
    raw = (roll_ret / roll_vol.replace(0.0, np.nan)).fillna(0.0)
    # Scale by expanding 95th-pct of |raw|, shifted 1 to avoid look-ahead
    scale = (raw.abs().expanding(min_periods=N).quantile(0.95)
             .shift(1).fillna(1.0).replace(0.0, 1.0))
    return (raw / scale).clip(-3.0, 3.0).fillna(0.0)


def compute_lt7(close: pd.Series, N: int = None,
                N_short: int = 750, N_long: int = 1250) -> pd.Series:
    """LT7: Multi-timeframe blend of LT2(N_short=750) + LT2(N_long=1250).
    N arg is ignored; Ns are fixed. Captures both ~3yr and ~5yr cycles.
    """
    s_short = compute_lt2(close, N_short)
    s_long  = compute_lt2(close, N_long)
    return ((s_short + s_long) / 2.0).clip(-3.0, 3.0)


LT_SIGNALS = {
    'LT1': compute_lt1,
    'LT2': compute_lt2,
    'LT3': compute_lt3,
    'LT4': compute_lt4,
    'LT6': compute_lt6,
    'LT7': compute_lt7,
}


def build_lt_signal(close: pd.Series, signal_name: str, N: int) -> pd.Series:
    if signal_name == 'LT7':
        return compute_lt7(close)  # fixed Ns internally
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


def signal_to_mult_c(signal: pd.Series, k_lt: float, alpha: float = 0.5) -> pd.Series:
    """Mode C helper: half-strength multiplier (for Mode A component)."""
    return signal_to_mult(signal, alpha * k_lt)


def signal_to_bias_c(signal: pd.Series, k_lt: float, alpha: float = 0.5) -> pd.Series:
    """Mode C helper: half-strength bias (for Mode B component)."""
    return signal_to_bias(signal, alpha * k_lt)
