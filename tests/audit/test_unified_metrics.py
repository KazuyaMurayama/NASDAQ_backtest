import pandas as pd
import numpy as np
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START


def test_split_constants():
    assert IS_END == pd.Timestamp('2021-05-07')
    assert OOS_START == pd.Timestamp('2021-05-08')


def test_cagr_on_known_series():
    # 年率10%, 252営業日×4年（FULL期間がIS/OOS両方を含むよう2019-2023に跨らせる）
    idx = pd.bdate_range('2019-01-01', periods=252*4)
    nav = pd.Series((1.10) ** (np.arange(252*4)/252), index=idx)
    m = compute_10metrics(nav, trades_per_year=27.0)
    assert abs(m['CAGR_FULL'] - 0.10) < 0.01
    assert m['MaxDD_FULL'] >= -1e-6          # 単調増加→DD≈0
    assert m['Trades_yr'] == 27.0
    assert 'Worst10Y_star' in m and 'P10_5Y' in m and 'Sharpe_OOS' in m
