import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
from multi_asset.walkforward import (
    yearly_windows,
    wfa_stats,
    block_bootstrap,
    paired_block_bootstrap,
)


def _nav_from_ret(ret):
    return (1.0 + ret).cumprod()


def _ret(mu, sigma, n, seed, start='2000-01-03'):
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start, periods=n, freq='B')
    return pd.Series(rng.normal(mu, sigma, n), index=idx)


def test_yearly_windows_one_per_calendar_year():
    idx = pd.date_range('2010-01-01', '2012-12-31', freq='B')
    wins = yearly_windows(idx)
    assert [w[0] for w in wins] == [2010, 2011, 2012]


def test_yearly_windows_skips_too_short_years():
    # only a handful of December days in 2010 → dropped (< min_days)
    idx = pd.date_range('2010-12-20', '2011-12-31', freq='B')
    years = [w[0] for w in yearly_windows(idx, min_days=60)]
    assert 2010 not in years and 2011 in years


def test_wfa_stats_positive_trend_passes_keys():
    ret = _ret(0.0005, 0.008, 252 * 8, seed=1)
    nav = _nav_from_ret(ret)
    st = wfa_stats(nav)
    for k in ['n_windows', 'wfe', 'ci95_lo_cagr', 'mean_window_cagr',
              'mean_window_sharpe', 'pass_ci', 'pass_wfe', 'passed']:
        assert k in st
    assert st['n_windows'] >= 7
    assert st['mean_window_cagr'] > 0


def test_wfa_wfe_robust_to_cash_only_years():
    # 4 risky years interleaved with 4 near-zero-vol (cash) years.
    # Cash years have degenerate (huge) Sharpe; WFE must NOT explode.
    rng = np.random.RandomState(11)
    parts = []
    for yr in range(8):
        idx = pd.date_range(f'{2000+yr}-01-03', periods=252, freq='B')
        if yr % 2 == 0:
            r = pd.Series(rng.normal(0.0005, 0.01, 252), index=idx)   # risky
        else:
            r = pd.Series(0.00008, index=idx)                          # ~cash
        parts.append(r)
    ret = pd.concat(parts)
    st = wfa_stats(_nav_from_ret(ret), min_vol=0.02)
    assert 0.0 < st['wfe'] < 5.0    # not the ~20-30 degenerate value


def test_wfa_ci95_lo_below_mean():
    ret = _ret(0.0004, 0.01, 252 * 10, seed=2)
    st = wfa_stats(_nav_from_ret(ret))
    assert st['ci95_lo_cagr'] <= st['mean_window_cagr']


def test_block_bootstrap_reproducible_and_detects_positive_drift():
    ret = _ret(0.0006, 0.009, 252 * 6, seed=3)
    a = block_bootstrap(ret, n_boot=500, block=60, seed=7)
    b = block_bootstrap(ret, n_boot=500, block=60, seed=7)
    assert a['p_pos'] == b['p_pos']            # reproducible
    assert a['cagr_lo'] == b['cagr_lo']
    assert a['p_pos'] > 0.9                     # clear positive drift
    assert a['cagr_lo'] <= a['cagr_median'] <= a['cagr_hi']


def test_paired_bootstrap_detects_better_series():
    base = _ret(0.0001, 0.008, 252 * 6, seed=4)
    better = base + 0.0008                       # strictly higher every day
    res = paired_block_bootstrap(better, base, n_boot=500, block=60, seed=9)
    assert res['p_a_gt_b'] > 0.95
    assert res['diff_median'] > 0
