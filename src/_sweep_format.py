"""
_sweep_format.py — Sweep MDテーブル共有フォーマッタ
=====================================================
EVALUATION_STANDARD §3.12 / §5.6 準拠 (v1.1, 2026-05-22)

統一指標セット（9指標）:
  1. CAGR_OOS    2. Sharpe  3. MaxDD  4. W10Y★
  5. P10▷        6. Gap     7. Tr     8. CI95_lo  9. WFE
"""
import numpy as np


# ---------------------------------------------------------------------------
# 値フォーマッタ
# ---------------------------------------------------------------------------

def _fp1(v):
    """百分率 1桁小数 例: +27.3%"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '  —  '
    return f'{v * 100:+5.1f}%'


def _ff2(v):
    """浮動小数 2桁小数 例: +0.88"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '  —  '
    return f'{v:+5.2f}'


def _gap_pp(v):
    """IS-OOS gap (pp表示) 例: +0.12pp"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '   —    '
    return f'{v * 100:+5.2f}pp'


def _tr(v):
    """Trades/yr 整数 例: 27"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ' — '
    return f'{int(round(v)):>3d}'


def _wfa(v):
    """WFA値 3桁 例: +0.041 (未計算時は —)"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '   —   '
    return f'{v:+6.3f}'


# ---------------------------------------------------------------------------
# 9-metric 標準テーブルヘッダ (§5.6)
# ---------------------------------------------------------------------------

# 1パラメータ sweep (例: b6 — N のみ)
# <br> で列ヘッダを2行に折り返し → 列幅を縮小 (§5.6)
MD_HEADER_1P = (
    '| Param | CAGR<br>_OOS | Sharpe | MaxDD | Worst<br>10Y★ | P10▷ | IS-OOS<br>gap | Tr | CI95<br>_lo | WFE |',
    '|:------|-------------:|-------:|------:|--------------:|-----:|--------------:|---:|-----------:|----:|',
)

# 2パラメータ sweep (例: b3/b4/b5/b7/b8 — N × k_lt)
MD_HEADER_2P = (
    '| N | k_lt | CAGR<br>_OOS | Sharpe | MaxDD | Worst<br>10Y★ | P10▷ | IS-OOS<br>gap | Tr | CI95<br>_lo | WFE |',
    '|--:|-----:|-------------:|-------:|------:|--------------:|-----:|--------------:|---:|-----------:|----:|',
)

# WFA 未計算注記（テーブル下に挿入）
MD_WFA_NOTE = (
    '*CI95_lo / WFE: `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。'
    '計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*'
)


# ---------------------------------------------------------------------------
# 汎用 data-row ビルダ
# ---------------------------------------------------------------------------

def fmt_row_2p(n, k_lt, r, ref_s2=0.770, ref_lt2=0.885):
    """N × k_lt 型の1行を返す (b3/b4/b5/b7/b8 共通)"""
    mark = ' ★' if r['Sharpe_OOS'] > ref_lt2 else (' ◎' if r['Sharpe_OOS'] > ref_s2 else '')
    return (
        f'| {n:>4d} | {k_lt:.1f} '
        f'| {_fp1(r["CAGR_OOS"])} '
        f'| {_ff2(r["Sharpe_OOS"])}{mark} '
        f'| {_fp1(r["MaxDD_FULL"])} '
        f'| {_fp1(r["Worst10Y_star"])} '
        f'| {_fp1(r["P10_5Y"])} '
        f'| {_gap_pp(r["IS_OOS_gap"])} '
        f'| {_tr(r.get("Trades_yr"))} '
        f'| {_wfa(r.get("WFA_CI95_lo"))} '
        f'| {_wfa(r.get("WFA_WFE"))} |'
    )


def fmt_row_1p(param_label, r, ref_s2=0.770, ref_lt2=0.885):
    """1パラメータ型の1行を返す (b6 等)"""
    mark = ' ★' if r['Sharpe_OOS'] > ref_lt2 else (' ◎' if r['Sharpe_OOS'] > ref_s2 else '')
    return (
        f'| {param_label} '
        f'| {_fp1(r["CAGR_OOS"])} '
        f'| {_ff2(r["Sharpe_OOS"])}{mark} '
        f'| {_fp1(r["MaxDD_FULL"])} '
        f'| {_fp1(r["Worst10Y_star"])} '
        f'| {_fp1(r["P10_5Y"])} '
        f'| {_gap_pp(r["IS_OOS_gap"])} '
        f'| {_tr(r.get("Trades_yr"))} '
        f'| {_wfa(r.get("WFA_CI95_lo"))} '
        f'| {_wfa(r.get("WFA_WFE"))} |'
    )
