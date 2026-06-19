"""
tests/audit/test_sweep_format.py
----------------------------------
_sweep_format.py v2.0 (新10列標準) のテスト

canonical imports（cfd_leverage_backtest 等）は不要。
_sweep_format.py は numpy のみ依存。
"""
import sys
import os
import numpy as np

# src/ を import パスに追加
_SRC = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, os.path.abspath(_SRC))

from _sweep_format import (
    MD_HEADER_1P, MD_HEADER_2P, MD_HEADER_STRAT, MD_HEADER_INTEGRATED,
    fmt_row_1p, fmt_row_2p, fmt_row_strat, fmt_row_integrated,
    _worst1d, _robustness_cell,
)


# ---------------------------------------------------------------------------
# サンプル戦略データ（CSV実数値 + Sharpe_FULL/Worst1D は推定値）
# ---------------------------------------------------------------------------

R_P09 = {
    'CAGR_IS':       0.188385,
    'CAGR_OOS':      0.175148,
    'Sharpe_FULL':   0.820,
    'MaxDD_FULL':   -0.351847,
    'Worst1D':      -0.123,
    'Worst1D_date': '2020-03-16',
    'Worst10Y_star': 0.114454,
    'Worst5Y':      -0.005854,
    'P10_5Y':        0.065612,
    'Trades_yr':     29.201306,
    'WFA_WFE':       1.016783,
    'WFA_CI95_lo':   0.179435,
    'IS_OOS_gap_pp': 1.3237,
    'CPCV_p10':      0.1794,
    't_p':           0.001,
    'Regime_min':   -0.006910,
}

R_B3A = {
    'CAGR_IS':       0.231035,
    'CAGR_OOS':      0.209760,
    'Sharpe_FULL':   0.830,
    'MaxDD_FULL':   -0.382040,
    'Worst1D':      -0.148,
    'Worst1D_date': '1987-10-19',
    'Worst10Y_star': 0.145328,
    'Worst5Y':       0.001022,
    'P10_5Y':        0.080827,
    'Trades_yr':     33.277242,
    'WFA_WFE':       0.987128,
    'WFA_CI95_lo':   0.225175,
    'IS_OOS_gap_pp': 2.5716,
    'CPCV_p10':      0.1601,
    't_p':           0.0000,
    'Regime_min':   -0.028829,
}

R_SCALE135 = {
    'CAGR_IS':       0.219471,
    'CAGR_OOS':      0.193503,
    'Sharpe_FULL':   0.810,
    'MaxDD_FULL':   -0.384597,
    'Worst1D':      -0.162,
    'Worst1D_date': '1987-10-19',
    'Worst10Y_star': 0.126963,
    'Worst5Y':      -0.007673,
    'P10_5Y':        0.074359,
    'Trades_yr':     29.201306,
    'WFA_WFE':       0.964659,
    'WFA_CI95_lo':   0.210581,
    'IS_OOS_gap_pp': 3.1389,
    'CPCV_p10':      np.nan,
    't_p':           np.nan,
    'Regime_min':   -0.011501,
}


# ---------------------------------------------------------------------------
# テスト: ヘッダの列数
# ---------------------------------------------------------------------------

def _count_cols(header_row: str) -> int:
    """Markdown テーブル行の列数を数える（先頭・末尾の | を除く）"""
    return len(header_row.strip().split('|')) - 2


def test_header_1p_col_count():
    """MD_HEADER_1P: ラベル列1 + メトリック10 = 11列"""
    assert _count_cols(MD_HEADER_1P[0]) == 11


def test_header_2p_col_count():
    """MD_HEADER_2P: ラベル列2 (N, k_lt) + メトリック10 = 12列"""
    assert _count_cols(MD_HEADER_2P[0]) == 12


def test_header_strat_col_count():
    """MD_HEADER_STRAT: ラベル列1 + メトリック10 = 11列"""
    assert _count_cols(MD_HEADER_STRAT[0]) == 11


def test_header_integrated_col_count():
    """MD_HEADER_INTEGRATED: ラベル列1 + メトリック10 = 11列"""
    assert _count_cols(MD_HEADER_INTEGRATED[0]) == 11


# ---------------------------------------------------------------------------
# テスト: ヘッダに新列名が含まれること
# ---------------------------------------------------------------------------

def test_header_contains_cagr_is():
    assert 'CAGR' in MD_HEADER_1P[0] and 'IS' in MD_HEADER_1P[0]

def test_header_contains_sharpe_full():
    assert 'Full' in MD_HEADER_1P[0] or 'Sharpe' in MD_HEADER_1P[0]

def test_header_contains_worst1d():
    assert '単日' in MD_HEADER_1P[0] or '最悪' in MD_HEADER_1P[0]

def test_header_contains_worst5y():
    assert '5Y' in MD_HEADER_1P[0] or 'Worst' in MD_HEADER_1P[0]

def test_header_no_isos_gap():
    """旧 IS-OOS gap 列ヘッダが独立列として残っていないこと"""
    assert 'IS-OOS<br>gap' not in MD_HEADER_1P[0]


# ---------------------------------------------------------------------------
# テスト: _worst1d フォーマッタ
# ---------------------------------------------------------------------------

def test_worst1d_normal():
    result = _worst1d(-0.123, '2020-03-16')
    assert '-12.3%' in result
    assert '2020-03-16' in result

def test_worst1d_nan():
    result = _worst1d(float('nan'), None)
    assert '—' in result


# ---------------------------------------------------------------------------
# テスト: _robustness_cell 判定
# ---------------------------------------------------------------------------

def test_robustness_pass():
    r = {'WFA_WFE': 1.0, 'WFA_CI95_lo': 0.20, 'IS_OOS_gap_pp': 1.3}
    cell = _robustness_cell(r)
    assert '✅' in cell

def test_robustness_fail_wfe():
    r = {'WFA_WFE': 0.3, 'WFA_CI95_lo': 0.20, 'IS_OOS_gap_pp': 1.3}
    cell = _robustness_cell(r)
    assert '❌' in cell

def test_robustness_fail_gap():
    r = {'WFA_WFE': 1.0, 'WFA_CI95_lo': 0.20, 'IS_OOS_gap_pp': 6.0}
    cell = _robustness_cell(r)
    assert '❌' in cell

def test_robustness_empty():
    assert _robustness_cell({}) == '—'

def test_robustness_partial():
    """t_p / CPCV 未算出なら (部分) が付く"""
    r = {'WFA_WFE': 1.0, 'WFA_CI95_lo': 0.20, 'IS_OOS_gap_pp': 2.5}
    cell = _robustness_cell(r)
    assert '(部分)' in cell


# ---------------------------------------------------------------------------
# テスト: fmt_row_* の列数
# ---------------------------------------------------------------------------

def test_fmt_row_1p_col_count():
    row = fmt_row_1p('test', R_P09)
    assert _count_cols(row) == 11

def test_fmt_row_2p_col_count():
    row = fmt_row_2p(750, 0.5, R_P09)
    assert _count_cols(row) == 12

def test_fmt_row_strat_col_count():
    row = fmt_row_strat('P09_C1', R_P09)
    assert _count_cols(row) == 11

def test_fmt_row_integrated_col_count():
    row = fmt_row_integrated('P09_C1', R_P09)
    assert _count_cols(row) == 11


# ---------------------------------------------------------------------------
# テスト: ◎/★ マーカが Sharpe_FULL ベースで付く
# ---------------------------------------------------------------------------

def test_star_marker_on_high_sharpe():
    r = dict(R_P09, Sharpe_FULL=0.900)
    row = fmt_row_strat('X', r, ref_lt2=0.800)
    assert '★' in row

def test_maru_marker_on_medium_sharpe():
    r = dict(R_P09, Sharpe_FULL=0.750)
    row = fmt_row_strat('X', r, ref_s2=0.700, ref_lt2=0.800)
    assert '◎' in row

def test_no_marker_on_low_sharpe():
    r = dict(R_P09, Sharpe_FULL=0.600)
    row = fmt_row_strat('X', r, ref_s2=0.700, ref_lt2=0.800)
    assert '◎' not in row and '★' not in row


# ---------------------------------------------------------------------------
# サンプル表レンダリング（3戦略）
# ---------------------------------------------------------------------------

def test_sample_table_render():
    """3戦略のサンプル表を標準出力に出力（pytest -s で確認）"""
    header, sep = MD_HEADER_STRAT
    rows = [
        header, sep,
        fmt_row_strat('P09_C1',        R_P09),
        fmt_row_strat('B3a_k365',       R_B3A),
        fmt_row_strat('scale1.35 LU2', R_SCALE135),
    ]
    table = '\n'.join(rows)
    print('\n\n=== サンプル表 ===\n')
    print(table)
    print('\n================\n')
    assert 'P09_C1' in table
    assert 'B3a_k365' in table
    assert 'scale1.35' in table
