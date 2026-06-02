"""
_sweep_format.py — Sweep MDテーブル共有フォーマッタ
=====================================================
EVALUATION_STANDARD §3.12 / §5.6 準拠 (v1.3, 2026-05-28)

統一指標セット（9指標）:
  1. CAGR_OOS    2. Sharpe  3. MaxDD  4. W10Y★
  5. P10▷        6. Gap     7. Tr     8. Overfit(WFE)
  9. CI95_lo

Overfit(WFE)（過学習リスクスコア）: WFE値に基づく評価
  ✅ LOW : 0.5 ≤ WFE ≤ 2.0（正常域）
  ⚠ MED : WFE > 2.0（OOS期間が過大に有利）
  ❌ HIGH: WFE < 0.5（IS期間優位 = 古典的過学習）
  —      : WFA未計算

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
    """WFA CI95_lo 3桁小数 例: +0.265 (未計算時は —)"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '   —   '
    return f'{v:+6.3f}'


def _ovfit_wfe(wfe):
    """過学習スコア (WFEベース): 判定ラベルと WFE値（小数1桁）を <br> 2行で返す。

    WFE = mean_CAGR_OOS / mean_CAGR_IS（WFA 全窓平均）
    ✅ LOW : 0.5 ≤ WFE ≤ 2.0（正常域）
    ⚠ MED : WFE > 2.0（OOS期間が過大に有利）
    ❌ HIGH: WFE < 0.5（IS期間優位 = 古典的過学習）
    """
    if wfe is None or (isinstance(wfe, float) and np.isnan(wfe)):
        return '—'
    wfe = float(wfe)
    if wfe < 0.5:
        label = '❌ HIGH'
    elif wfe <= 2.0:
        label = '✅ LOW'
    else:
        label = '⚠ MED'
    return f'{label}<br>({wfe:.1f})'


# ---------------------------------------------------------------------------
# 9-metric 標準テーブルヘッダ (§5.6 / Overfit(WFE) は v1.3 でOvFit+WFEを統合)
# ---------------------------------------------------------------------------

# 1パラメータ sweep (例: b6 — N のみ)
# v1.4: 列順変更 — IS-OOS gap CAGR を CAGR_OOS の右隣へ移動 / 4行折り返しで列幅を更に縮小
MD_HEADER_1P = (
    '| Param | CAGR<br>⓽<br>_<br>OOS | IS-OOS<br>gap<br>CAGR | Sharpe<br>ⓒ<br>_OOS | MaxDD<br>ⓒ | Worst<br>10Y★<br>⓽<br>CAGR | P10<br>⓽<br>5Y▷<br>CAGR | Trade<br>ⓞ<br>(回/<br>年) | Overfit<br>ⓞ<br>(WFE) | CI95<br>ⓡ<br>_lo |',
    '|:------|-------------:|-------------:|---------------:|------:|----------------------:|-------------------:|------------------:|:----------------:|-----------:|',
)

# 2パラメータ sweep (例: b3/b4/b5/b7/b8 — N × k_lt)
MD_HEADER_2P = (
    '| N | k_lt | CAGR<br>⓽<br>_<br>OOS | IS-OOS<br>gap<br>CAGR | Sharpe<br>ⓒ<br>_OOS | MaxDD<br>ⓒ | Worst<br>10Y★<br>⓽<br>CAGR | P10<br>⓽<br>5Y▷<br>CAGR | Trade<br>ⓞ<br>(回/<br>年) | Overfit<br>ⓞ<br>(WFE) | CI95<br>ⓡ<br>_lo |',
    '|--:|-----:|-------------:|-------------:|---------------:|------:|----------------------:|-------------------:|------------------:|:----------------:|-----------:|',
)

# 戦略横並び比較表（複数戦略を1行ずつ並べる: e2_hybrid / STRATEGY_COMPARISON 等）
# 列順・列幅は MD_HEADER_1P/2P と完全に一致させる（§3.12 v1.4）
MD_HEADER_STRAT = (
    '| Strategy | CAGR<br>⓽<br>_<br>OOS | IS-OOS<br>gap<br>CAGR | Sharpe<br>ⓒ<br>_OOS | MaxDD<br>ⓒ | Worst<br>10Y★<br>⓽<br>CAGR | P10<br>⓽<br>5Y▷<br>CAGR | Trade<br>ⓞ<br>(回/<br>年) | Overfit<br>ⓞ<br>(WFE) | CI95<br>ⓡ<br>_lo |',
    '|:---------|-------------:|-------------:|---------------:|------:|----------------------:|-------------------:|------------------:|:----------------:|-----------:|',
)

# WFA 未計算注記（スイープ MDテーブル下に挿入）
MD_WFA_NOTE = (
    '*CI95_lo / Overfit(WFE): `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。'
    '計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*  \n'
    '*進格条件: **CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）。'
    'Sharpe マーカ: ◎ = Sharpe_OOS > +0.770 / ★ = > +0.885。'
    '列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*'
)

# 戦略比較ファイル用 凡例1行（テーブル直下に挿入: e2_hybrid / B9_COMPARISON 等）
MD_METRIC_GLOSSARY = (
    '*◎ Sharpe_OOS > +0.770（S2ベースライン超過）/ ★ > +0.885（現行ベスト超過）。'
    '進格: CI95_lo > 0 かつ 0.5 ≤ WFE ≤ 2.0。'
    'Overfit(WFE): ✅LOW=WFE∈[0.5,2.0] / ⚠MED=WFE>2.0 / ❌HIGH=WFE<0.5。'
    '列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカの ◎/★ とは別意味）。*'
)


# ---------------------------------------------------------------------------
# 汎用 data-row ビルダ
# ---------------------------------------------------------------------------

def fmt_row_2p(n, k_lt, r, ref_s2=0.770, ref_lt2=0.885):
    """N × k_lt 型の1行を返す (b3/b4/b5/b7/b8 共通) — v1.4: IS-OOS gap CAGR を 2列目に"""
    mark = ' ★' if r['Sharpe_OOS'] > ref_lt2 else (' ◎' if r['Sharpe_OOS'] > ref_s2 else '')
    return (
        f'| {n:>4d} | {k_lt:.1f} '
        f'| {_fp1(r["CAGR_OOS"])} '
        f'| {_gap_pp(r["IS_OOS_gap"])} '
        f'| {_ff2(r["Sharpe_OOS"])}{mark} '
        f'| {_fp1(r["MaxDD_FULL"])} '
        f'| {_fp1(r["Worst10Y_star"])} '
        f'| {_fp1(r["P10_5Y"])} '
        f'| {_tr(r.get("Trades_yr"))} '
        f'| {_ovfit_wfe(r.get("WFA_WFE"))} '
        f'| {_wfa(r.get("WFA_CI95_lo"))} |'
    )


def fmt_row_1p(param_label, r, ref_s2=0.770, ref_lt2=0.885):
    """1パラメータ型の1行を返す (b6 等) — v1.4: IS-OOS gap CAGR を 2列目に"""
    mark = ' ★' if r['Sharpe_OOS'] > ref_lt2 else (' ◎' if r['Sharpe_OOS'] > ref_s2 else '')
    return (
        f'| {param_label} '
        f'| {_fp1(r["CAGR_OOS"])} '
        f'| {_gap_pp(r["IS_OOS_gap"])} '
        f'| {_ff2(r["Sharpe_OOS"])}{mark} '
        f'| {_fp1(r["MaxDD_FULL"])} '
        f'| {_fp1(r["Worst10Y_star"])} '
        f'| {_fp1(r["P10_5Y"])} '
        f'| {_tr(r.get("Trades_yr"))} '
        f'| {_ovfit_wfe(r.get("WFA_WFE"))} '
        f'| {_wfa(r.get("WFA_CI95_lo"))} |'
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
        Overfit(WFE) 列は WFA_WFE から自動算出
        (0.5 ≤ WFE ≤ 2.0: LOW / WFE > 2.0: MED / WFE < 0.5: HIGH)
    sharpe_ref_mark : str | None
        Sharpe 値の直後に付けるマーカ。§1.3 参考値戦略には '‡' を渡す。
    maxdd_ref_mark : str | None
        MaxDD 値の直後に付けるマーカ。§1.3 参考値戦略には '‡' を渡す。
    """
    mark = ' ★' if r['Sharpe_OOS'] > ref_lt2 else (' ◎' if r['Sharpe_OOS'] > ref_s2 else '')
    s_sfx = sharpe_ref_mark or ''
    m_sfx = maxdd_ref_mark  or ''
    # v1.4: IS-OOS gap CAGR を CAGR_OOS の右隣（2列目）に配置
    return (
        f'| {label} '
        f'| {_fp1(r["CAGR_OOS"])} '
        f'| {_gap_pp(r["IS_OOS_gap"])} '
        f'| {_ff2(r["Sharpe_OOS"])}{mark}{s_sfx} '
        f'| {_fp1(r["MaxDD_FULL"])}{m_sfx} '
        f'| {_fp1(r["Worst10Y_star"])} '
        f'| {_fp1(r["P10_5Y"])} '
        f'| {_tr(r.get("Trades_yr"))} '
        f'| {_ovfit_wfe(r.get("WFA_WFE"))} '
        f'| {_wfa(r.get("WFA_CI95_lo"))} |'
    )
