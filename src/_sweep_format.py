"""
_sweep_format.py — Sweep MDテーブル共有フォーマッタ
=====================================================
EVALUATION_STANDARD §3.12 / §5.6 準拠 (v1.2, 2026-05-27)

統一指標セット（10指標）:
  1. CAGR_OOS    2. Sharpe  3. MaxDD  4. W10Y★
  5. P10▷        6. Gap     7. Tr     8. OvFit
  9. CI95_lo    10. WFE

OvFit（過学習リスクスコア）: |IS_OOS_gap| (pp) に基づく評価
  ✅ LOW : |gap| ≤ 2pp     ⚠ MED : 2pp < |gap| ≤ 5pp
  ❌ HIGH: |gap| > 5pp     —      : NaN

提供フォーマッタ:
  - MD_HEADER_1P  / fmt_row_1p   : 1パラメータ sweep（b6 等）
  - MD_HEADER_2P  / fmt_row_2p   : 2パラメータ sweep（b3/b4/b5/b7/b8 等）
  - MD_HEADER_STRAT / fmt_row_strat : 戦略横並び比較（e2_hybrid / STRATEGY_COMPARISON 等）

CAGR は CAGR_OOS の1列のみ。CAGR_IS / CAGR_FULL を MD ヘッダに含めると v1.1 違反。
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


def _ovfit(gap_decimal):
    """過学習スコア: |IS_OOS_gap| (decimal) から ✅/⚠/❌ ラベルを返す。

    gap は IS_CAGR - OOS_CAGR の decimal（例: -0.0181 = -1.81pp）。
    符号は問わず絶対値で評価。
    """
    if gap_decimal is None or (isinstance(gap_decimal, float) and np.isnan(gap_decimal)):
        return '  —  '
    abs_pp = abs(float(gap_decimal)) * 100.0
    if abs_pp <= 2.0:
        return '✅ LOW'
    if abs_pp <= 5.0:
        return '⚠ MED'
    return '❌ HIGH'


# ---------------------------------------------------------------------------
# 10-metric 標準テーブルヘッダ (§5.6 / OvFit は v1.2 で追加)
# ---------------------------------------------------------------------------

# 1パラメータ sweep (例: b6 — N のみ)
# <br> で列ヘッダを2〜3行に折り返し → 列幅を縮小 (§5.6)
MD_HEADER_1P = (
    '| Param | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | OvFit | CI95<br>_lo | WFE |',
    '|:------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|:-----:|-----------:|----:|',
)

# 2パラメータ sweep (例: b3/b4/b5/b7/b8 — N × k_lt)
MD_HEADER_2P = (
    '| N | k_lt | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | OvFit | CI95<br>_lo | WFE |',
    '|--:|-----:|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|:-----:|-----------:|----:|',
)

# 戦略横並び比較表（複数戦略を1行ずつ並べる: e2_hybrid / STRATEGY_COMPARISON 等）
# 列順・列幅は MD_HEADER_1P/2P と完全に一致させる（§3.12）
MD_HEADER_STRAT = (
    '| Strategy | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | OvFit | CI95<br>_lo | WFE |',
    '|:---------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|:-----:|-----------:|----:|',
)

# WFA 未計算注記（スイープ MDテーブル下に挿入）
# 進格条件・Sharpeマーカ凡例・★記号の区別を含む
MD_WFA_NOTE = (
    '*CI95_lo / WFE: `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。'
    '計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*  \n'
    '*進格条件: **CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）。'
    'Sharpe マーカ: ◎ = Sharpe_OOS > +0.770 / ★ = > +0.885。'
    '列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*'
)

# 戦略比較ファイル用 凡例1行（テーブル直下に挿入: e2_hybrid / B9_COMPARISON 等）
MD_METRIC_GLOSSARY = (
    '*◎ Sharpe_OOS > +0.770（S2ベースライン超過）/ ★ > +0.885（現行ベスト超過）。'
    '進格: CI95_lo > 0 かつ 0.5 ≤ WFE ≤ 2.0。'
    '列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカの ◎/★ とは別意味）。*'
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
        f'| {_ovfit(r.get("IS_OOS_gap"))} '
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
        f'| {_ovfit(r.get("IS_OOS_gap"))} '
        f'| {_wfa(r.get("WFA_CI95_lo"))} '
        f'| {_wfa(r.get("WFA_WFE"))} |'
    )


def fmt_row_strat(label, r, ref_s2=0.770, ref_lt2=0.885,
                  sharpe_ref_mark=None, maxdd_ref_mark=None):
    """戦略横並び比較表の1戦略1行を返す (MD_HEADER_STRAT と対で使う)。

    Parameters
    ----------
    label : str
        戦略名。Markdown 装飾（**...**）と §1.3 参考値マーカ（‡）は呼び出し側で付与する。
    r : dict
        必須キー: CAGR_OOS, Sharpe_OOS, MaxDD_FULL, Worst10Y_star,
                  P10_5Y, IS_OOS_gap, Trades_yr, WFA_CI95_lo, WFA_WFE
        OvFit 列は IS_OOS_gap から自動算出 (|gap| ≤ 2pp: LOW / ≤ 5pp: MED / > 5pp: HIGH)
    sharpe_ref_mark : str | None
        Sharpe 値の直後に付けるマーカ。§1.3 参考値戦略には '‡' を渡す。
    maxdd_ref_mark : str | None
        MaxDD 値の直後に付けるマーカ。§1.3 参考値戦略には '‡' を渡す。
    """
    mark = ' ★' if r['Sharpe_OOS'] > ref_lt2 else (' ◎' if r['Sharpe_OOS'] > ref_s2 else '')
    s_sfx = sharpe_ref_mark or ''
    m_sfx = maxdd_ref_mark  or ''
    return (
        f'| {label} '
        f'| {_fp1(r["CAGR_OOS"])} '
        f'| {_ff2(r["Sharpe_OOS"])}{mark}{s_sfx} '
        f'| {_fp1(r["MaxDD_FULL"])}{m_sfx} '
        f'| {_fp1(r["Worst10Y_star"])} '
        f'| {_fp1(r["P10_5Y"])} '
        f'| {_gap_pp(r["IS_OOS_gap"])} '
        f'| {_tr(r.get("Trades_yr"))} '
        f'| {_ovfit(r.get("IS_OOS_gap"))} '
        f'| {_wfa(r.get("WFA_CI95_lo"))} '
        f'| {_wfa(r.get("WFA_WFE"))} |'
    )
