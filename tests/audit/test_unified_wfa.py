"""
tests/audit/test_unified_wfa.py
TDD: unified_wfa.py の公開インターフェースを検証する。

調査済み事実 (2026-06-10):
  - per_window CSV 列名: CAGR, Sharpe, MaxDD, Vol, PosDay_pct, n_days,
                          strategy, window_id, start_date, end_date, short_flag
  - CAGR は小数形式 (0.417 = 41.7%)
  - WFE ≈ 1.131, CI95_lo ≈ 0.265 (小数)
  - OOS_START_REF = '2021-05-08'
"""
import os
import pandas as pd
import pytest
from src.audit.unified_wfa import summarize_wfa, WINDOW_LEN, OOS_START_REF

_G3_CSV = os.path.join(os.path.dirname(__file__), '..', '..', 'g3_wfa_e4_per_window.csv')


def test_window_constants():
    """公開定数が仕様値に一致する。"""
    assert WINDOW_LEN == 252
    assert OOS_START_REF == pd.Timestamp('2021-05-08')


def test_summary_matches_existing_g3_e4():
    """既存 g3_wfa_e4_per_window.csv を使い、既知の統計値との一致を確認。

    E4-RegimeKLT 行のみ使用 (summary CSV と同一条件)。
    既知値 (g3_wfa_e4_summary.csv より):
      WFE     = 1.130664  (許容 ±0.03)
      CI95_lo = 0.265093  (小数、許容 ±0.01)
    """
    per = pd.read_csv(_G3_CSV)

    # start_date / end_date を datetime に変換
    for col in per.columns:
        if 'date' in col.lower():
            per[col] = pd.to_datetime(per[col])

    # E4-RegimeKLT 行のみ抽出 (single strategy)
    e4 = per[per['strategy'] == 'E4-RegimeKLT'].copy()
    assert len(e4) > 0, "E4-RegimeKLT rows not found in per_window CSV"

    s = summarize_wfa(e4)

    # WFA_WFE ≈ 1.131 (許容 ±0.03)
    wfe = s.get('WFA_WFE')
    assert wfe is not None, "WFA_WFE not in summary dict"
    assert abs(wfe - 1.131) < 0.03, f"WFA_WFE={wfe:.4f}, expected ≈1.131"

    # WFA_CI95_lo ≈ 0.265 (小数、許容 ±0.01)
    ci = s.get('WFA_CI95_lo')
    assert ci is not None, "WFA_CI95_lo not in summary dict"
    assert abs(ci - 0.265) < 0.01, f"WFA_CI95_lo={ci:.4f}, expected ≈0.265"


def test_summarize_wfa_returns_required_keys():
    """summarize_wfa の戻り値が最低限のキーを含む。"""
    per = pd.read_csv(_G3_CSV)
    for col in per.columns:
        if 'date' in col.lower():
            per[col] = pd.to_datetime(per[col])
    e4 = per[per['strategy'] == 'E4-RegimeKLT'].copy()

    s = summarize_wfa(e4)
    required = {'WFA_CI95_lo', 'WFA_WFE'}
    missing = required - set(s.keys())
    assert not missing, f"Missing keys: {missing}"


def test_summarize_wfa_excludes_short_windows():
    """short_flag=True の行が除外されて計算されることを確認する。

    short_flag 全True にした場合 → 空 dict か n_windows=0 が返る。
    """
    per = pd.read_csv(_G3_CSV)
    for col in per.columns:
        if 'date' in col.lower():
            per[col] = pd.to_datetime(per[col])
    e4 = per[per['strategy'] == 'E4-RegimeKLT'].copy()

    # 全行を short_flag=True にしてフィルタが機能するか確認
    e4_all_short = e4.copy()
    e4_all_short['short_flag'] = True
    s_all_short = summarize_wfa(e4_all_short)
    # compute_summary_stats は valid=空の場合 {} を返す仕様
    assert s_all_short == {} or s_all_short.get('n_windows', 0) == 0
