"""
gen_strategy_comparison_20260527.py
====================================
STRATEGY_PERFORMANCE_COMPARISON_2026-05-27.md 生成スクリプト (v1.7)

§3.12 v1.1 準拠:
- MD_HEADER_STRAT / fmt_row_strat を import（手書きヘッダ禁止）
- CAGR_IS / CAGR_FULL を MD テーブルヘッダに含めない
- Trades_yr / WFA_CI95_lo / WFA_WFE 必須

v1.7 変更点:
- Group A / Group B の二分割テーブル構造を廃止
- 全 21 戦略を 1 つの統合テーブルに集約（10列 × 21行）
"""

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from _sweep_format import MD_HEADER_STRAT, fmt_row_strat, MD_METRIC_GLOSSARY

OUT_MD = ROOT / 'STRATEGY_PERFORMANCE_COMPARISON_2026-05-27.md'


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _get_row(csv_name, **kw):
    """CSVから条件一致行を辞書で返す（最初の一致行）"""
    df = pd.read_csv(ROOT / csv_name)
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in kw.items():
        if isinstance(val, float):
            mask &= (df[col] - val).abs() < 1e-5
        elif isinstance(val, str):
            mask &= df[col].astype(str) == val
        else:
            mask &= (df[col] - float(val)).abs() < 1e-5
    rows = df[mask]
    if rows.empty:
        raise ValueError(f'No row in {csv_name} matching {kw}')
    return rows.iloc[0].to_dict()


def _wfa(csv_name, strat):
    """WFA サマリー CSV から (CI95_lo, WFE) を取得"""
    df = pd.read_csv(ROOT / csv_name)
    row = df[df['strategy'] == strat].iloc[0]
    return float(row['WFA_CI95_lo']), float(row['WFA_WFE'])


def mr(d, ci95=None, wfe=None, trades_override=None):
    """CSV行dict を fmt_row_strat 用 dict に変換"""
    return {
        'CAGR_OOS':      float(d.get('CAGR_OOS',      float('nan'))),
        'Sharpe_OOS':    float(d.get('Sharpe_OOS',    float('nan'))),
        'MaxDD_FULL':    float(d.get('MaxDD_FULL',    float('nan'))),
        'Worst10Y_star': float(d.get('Worst10Y_star', float('nan'))),
        'P10_5Y':        float(d.get('P10_5Y',        float('nan'))),
        'IS_OOS_gap':    float(d.get('IS_OOS_gap',    float('nan'))),
        'Trades_yr':     float(trades_override if trades_override is not None
                               else d.get('Trades_yr', float('nan'))),
        'WFA_CI95_lo':   ci95,
        'WFA_WFE':       wfe,
    }


# ---------------------------------------------------------------------------
# データ取得（21 戦略）
# ---------------------------------------------------------------------------

# ─── Block 1: [Active] 現行ベスト + Active候補（WFA完了, 4行） ───────────────

# 01: E4 Regime k_lt ◆（現行ベスト）
_e4_raw = _get_row('b4_klo_zero_results.csv',
                   k_lo=0.1, k_hi=0.8, vz_thr=0.7)
_e4_ci, _e4_wfe = _wfa('g3_wfa_e4_summary.csv', 'E4-RegimeKLT')
r_e4 = mr(_e4_raw, ci95=_e4_ci, wfe=_e4_wfe)

# 02: F10 ε=0.015
_f10_raw = _get_row('f10_epsilon_deadband_results.csv', eps=0.015)
_f10_ci, _f10_wfe = _wfa('g7_wfa_f10_summary.csv', 'F10-eps015')
r_f10 = mr(_f10_raw, ci95=_f10_ci, wfe=_f10_wfe)

# 03: F10+lmax5（Trades_yr は g8 mean_Trades_yr=52 を使用）
_f10l5_raw = _get_row('f10lmax5_fullmetrics.csv')  # 1行のみ
_f10l5_ci, _f10l5_wfe = _wfa('g8_wfa_lmax5_summary.csv', 'F10-eps015-lmax5')
_g8_df = pd.read_csv(ROOT / 'g8_wfa_lmax5_summary.csv')
_f10l5_tr = float(_g8_df[_g8_df['strategy'] == 'F10-eps015-lmax5']['mean_Trades_yr'].iloc[0])
r_f10l5 = mr(_f10l5_raw, ci95=_f10l5_ci, wfe=_f10l5_wfe, trades_override=_f10l5_tr)

# 04: D5 vz=0.65/lmax=5.0
_vz5_raw = _get_row('d5_vz_lmax_grid_results.csv',
                    vz_thr=0.65, l_max=5.0)
_vz5_ci, _vz5_wfe = _wfa('g10_wfa_vz065_lmax_row_summary.csv', 'vz065-lmax5')
r_vz5 = mr(_vz5_raw, ci95=_vz5_ci, wfe=_vz5_wfe)

# ─── Block 2: [B] B4 実験（1行） ─────────────────────────────────────────────

# 05: B4 k_lo=0
_b4_raw = _get_row('b4_klo_zero_results.csv',
                   k_lo=0.0, k_hi=0.7, vz_thr=0.7)
r_b4 = mr(_b4_raw)

# ─── Block 3: [A] A系実験（7行） ─────────────────────────────────────────────

# 06: A1 alpha=2
_a1_2 = _get_row('a1_soft_regime_klt_results.csv', alpha=2.0)
r_a1_2 = mr(_a1_2)

# 07: A1 alpha=3
_a1_3 = _get_row('a1_soft_regime_klt_results.csv', alpha=3.0)
r_a1_3 = mr(_a1_3)

# 08: A1 alpha=5
_a1_5 = _get_row('a1_soft_regime_klt_results.csv', alpha=5.0)
r_a1_5 = mr(_a1_5)

# 09: A1 alpha=8
_a1_8 = _get_row('a1_soft_regime_klt_results.csv', alpha=8.0)
r_a1_8 = mr(_a1_8)

# 10: A2 dyn_lmax
_a2 = _get_row('a2_dyn_lmax_results.csv', lmax_base=6.0, vol_sens=2.0)
r_a2 = mr(_a2)

# 11: A2B rolling_vref
_a2b = _get_row('a2b_dyn_lmax_rolling_vref_results.csv', lmax_base=6.0, vol_sens=2.0)
r_a2b = mr(_a2b)

# 12: A3 regime_asset_tilt
_a3 = _get_row('a3_regime_asset_tilt_results.csv', vov_thr=1.3, alpha_max=0.2)
r_a3 = mr(_a3)

# ─── Block 4: [C] C系実験（2行） ─────────────────────────────────────────────

# 13: C2 adaptive_deadband
_c2 = _get_row('c2_adaptive_deadband_results.csv', eps_0=0.020, mode='adaptive')
r_c2 = mr(_c2)

# 14: C3 yang_zhang
_c3 = _get_row('c3_yang_zhang_results.csv', yz_n=10, vz_thr=0.7)
r_c3 = mr(_c3)

# ─── Block 5: [D5] D5 グリッド（7行） ────────────────────────────────────────

# 15: D5 vz=0.60/lmax=4.5（WFA 未実施）
_d5_60_45 = _get_row('d5_vz_lmax_grid_results.csv', vz_thr=0.60, l_max=4.5)
r_d5_60_45 = mr(_d5_60_45)

# 16: D5 vz=0.60/lmax=5.0（WFA 未実施）
_d5_60_50 = _get_row('d5_vz_lmax_grid_results.csv', vz_thr=0.60, l_max=5.0)
r_d5_60_50 = mr(_d5_60_50)

# 17: D5 vz=0.65/lmax=4.5（WFA 未実施）
_d5_65_45 = _get_row('d5_vz_lmax_grid_results.csv', vz_thr=0.65, l_max=4.5)
r_d5_65_45 = mr(_d5_65_45)

# 18: D5 vz=0.65/lmax=5.5 (G10 WFA)
_d5_65_55_raw = _get_row('d5_vz_lmax_grid_results.csv', vz_thr=0.65, l_max=5.5)
_d5_55_ci, _d5_55_wfe = _wfa('g10_wfa_vz065_lmax_row_summary.csv', 'vz065-lmax5p5')
r_d5_65_55 = mr(_d5_65_55_raw, ci95=_d5_55_ci, wfe=_d5_55_wfe)

# 19: D5 vz=0.65/lmax=6.0 (G10 WFA)
_d5_65_60_raw = _get_row('d5_vz_lmax_grid_results.csv', vz_thr=0.65, l_max=6.0)
_d5_60_ci, _d5_60_wfe = _wfa('g10_wfa_vz065_lmax_row_summary.csv', 'vz065-lmax6')
r_d5_65_60 = mr(_d5_65_60_raw, ci95=_d5_60_ci, wfe=_d5_60_wfe)

# 20: D5 vz=0.65/lmax=7.0 (G10 WFA)
_d5_65_70_raw = _get_row('d5_vz_lmax_grid_results.csv', vz_thr=0.65, l_max=7.0)
_d5_70_ci, _d5_70_wfe = _wfa('g10_wfa_vz065_lmax_row_summary.csv', 'vz065-lmax7')
r_d5_65_70 = mr(_d5_65_70_raw, ci95=_d5_70_ci, wfe=_d5_70_wfe)

# 21: D5 vz=0.70/lmax=5.0（WFA 未実施）
_d5_70_50 = _get_row('d5_vz_lmax_grid_results.csv', vz_thr=0.70, l_max=5.0)
r_d5_70_50 = mr(_d5_70_50)


# ---------------------------------------------------------------------------
# MD 生成
# ---------------------------------------------------------------------------

HEADER_1, HEADER_2 = MD_HEADER_STRAT

lines = []
W = lines.append  # alias

W('# 戦略パフォーマンス比較表 v1.7 — 2026-05-27 統合21戦略比較')
W('')
W('作成日: 2026-05-27')
W('最終更新日: 2026-05-27')
W('EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**')
W('生成スクリプト: `src/gen_strategy_comparison_20260527.py`')
W('')
W('> ### ◆ 現行ベスト戦略')
W('> **E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7)**')
W('> CAGR_OOS **+33.53%** | Sharpe **+0.891** | MaxDD **−60.0%** | Trades/yr **27** | G3 WFA PASS ✓')
W('')
W('---')
W('')

# ── §1 比較前提 ───────────────────────────────────────────────────────────────
W('## 📋 §1 比較前提')
W('')
W('| 項目 | 定義 |')
W('|------|------|')
W('| **IS** | 1974-01-02 〜 2021-05-07（47.3年） |')
W('| **OOS** | 2021-05-08 〜 2026-03-26（4.9年） |')
W('| **FULL** | 1974-01-02 〜 2026-03-26（52.26年） |')
W('| **コスト** | Scenario D（`src/product_costs.py` 2026-05-12 基準） |')
W('| **DELAY** | 2営業日（look-ahead bias 対策） |')
W('| **Sharpe Rf** | 0 |')
W('| **CURRENT_BEST** | E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7)（◆, G3 WFA PASS 確定） |')
W('| **WFA** | G10: 49窓（252日 calendar-year-anchored non-overlapping）|')
W('')
W('| 凡例 | 意味 |')
W('|------|------|')
W('| ◆ | 現行ベスト戦略 |')
W('| ✅ | Shortlisted（WFA PASS・ベスト昇格候補） |')
W('| ⚠ | IS-OOS gap 警戒（gap ≤ −4.0pp） |')
W('| ‡ | 参照行（採用候補外、比較基準として掲載） |')
W('| [B]/[A]/[C]/[D5] | 実験系統ラベル |')
W('| WARN | 改善効果なし・実験目的のみ |')
W('| FAIL | 指標悪化・不採用確定 |')
W('')
W('---')
W('')

# ── §2 統合 21 戦略比較表 ───────────────────────────────────────────────────
W('## 📊 §2 全戦略 統合比較表（21戦略 × 10指標）')
W('')
W('> **単位**: CAGR_OOS / Worst10Y★ / P10▷ / MaxDD = %、IS-OOS gap = pp、Tr = 回/年')
W('> ★ = Sharpe_OOS > +0.885 / ◎ = > +0.770（S2ベースライン）')
W('> WFA 列が `—` の行は WFA 未実施（合計 14/21 行が未実施 / 7行が PASS 済み）')
W('')
W(HEADER_1)
W(HEADER_2)

# Block 1: [Active] 4行
W(fmt_row_strat('**E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7) ◆**', r_e4))
W(fmt_row_strat('**[Active候補] F10 ε=0.015 ✅⚠**', r_f10))
W(fmt_row_strat('**[Active候補] F10+lmax5 ✅⚠**', r_f10l5))
W(fmt_row_strat('**[Active候補] D5 vz=0.65/lmax=5.0 ✅⚠**', r_vz5))

# Block 2: [B] 1行
W(fmt_row_strat('[B] B4 k_lo=0/k_hi=0.7/vz=0.7', r_b4))

# Block 3: [A] 7行
W(fmt_row_strat('[A] A1 α=2 (soft regime)', r_a1_2))
W(fmt_row_strat('[A] A1 α=3 (soft regime)', r_a1_3))
W(fmt_row_strat('[A] A1 α=5 (soft regime)', r_a1_5))
W(fmt_row_strat('[A] A1 α=8 (soft regime)', r_a1_8))
W(fmt_row_strat('[A] A2 lmax_base=6/vol_sens=2', r_a2))
W(fmt_row_strat('[A] A2B rolling VOL_REF', r_a2b))
W(fmt_row_strat('[A] A3 VoV dual gate (vov=1.3/α=0.2)', r_a3))

# Block 4: [C] 2行
W(fmt_row_strat('[C] C2 adaptive deadband (ε₀=0.020)', r_c2))
W(fmt_row_strat('[C] C3 Yang-Zhang (yz_n=10/vz=0.7)', r_c3))

# Block 5: [D5] 7行
W(fmt_row_strat('[D5] vz=0.60/lmax=4.5', r_d5_60_45))
W(fmt_row_strat('[D5] vz=0.60/lmax=5.0', r_d5_60_50))
W(fmt_row_strat('[D5] vz=0.65/lmax=4.5', r_d5_65_45))
W(fmt_row_strat('[D5] vz=0.65/lmax=5.5 ✅', r_d5_65_55))
W(fmt_row_strat('[D5] vz=0.65/lmax=6.0 ✅', r_d5_65_60))
W(fmt_row_strat('[D5] vz=0.65/lmax=7.0 ✅⚠', r_d5_65_70))
W(fmt_row_strat('[D5] vz=0.70/lmax=5.0', r_d5_70_50))

W('')
W(MD_METRIC_GLOSSARY)
W('')
W('**WFA 完了行**: 7/21（E4 ◆ + Active候補3行 + D5 vz=0.65 系3行）')
W('**WFA 未実施行**: 14/21（B4, A1×4, A2/A2B/A3, C2/C3, D5 vz=0.60/0.70 系4行）')
W('')
W('---')
W('')

# ── §3 実験別判定サマリー ──────────────────────────────────────────────────
W('## 🔬 §3 実験別判定サマリー')
W('')
W('| 実験ID | 仮説 | 判定 | 結論 |')
W('|--------|------|:----:|------|')
W('| **B4** k_lo=0 | k_lo=0 でも E4 に近い性能が出るか | FAIL | Sharpe=+0.887 < E4 +0.891。k_lo=0.1 が最適。変更不要。 |')
W('| **A1** α=2 | soft sigmoid (緩慢遷移) で MaxDD 改善 | WARN | MaxDD=-61.7% 改善不十分。Sharpe=+0.886 < E4。 |')
W('| **A1** α=3 | soft sigmoid 中間値 | WARN | Sharpe=+0.890 ≈ E4。MaxDD=-62.9% 微改善。差異微小。 |')
W('| **A1** α=5 | soft sigmoid 中間値 | WARN | Sharpe=+0.898 向上だが gap=-2.06pp 拡大。 |')
W('| **A1** α=8 | soft sigmoid (急遷移) で Sharpe 向上 | WARN | Sharpe=+0.904 向上だが MaxDD=-67.1%⚠ 悪化。 |')
W('| **A2** base=6/sens=2 | vol 感応 l_max で MaxDD 改善 | WARN | MaxDD=-58.7% 改善。CAGR=+32.6% 低下。E4 非優越。 |')
W('| **A2B** rolling VOL_REF | 10年ローリング VOL_REF で中立化 | FAIL | vol_ref_t=0.161（mean < 常時閾値）→ l_max 常時削減バイアス。CAGR=+31.7% 大幅低下。 |')
W('| **A3** vov=1.3/α=0.2 | VoV z-score で Volcker 期を回避 | FAIL | MaxDD=-60.01% 変化なし。1980-82 Volcker 期は vol 低下期→ VoV 0 日発火。構造的限界。 |')
W('| **C2** ε₀=0.020/adaptive | 適応 deadband で IS-OOS gap 縮小 | WARN | F10 と同性能。gap=-4.28pp⚠ 改善なし。Trades=51 高コスト。 |')
W('| **C3** yz_n=10/vz=0.7 | Yang-Zhang 推定量で MaxDD 改善 | WARN | MaxDD=-58.4% 改善・Worst10Y=+19.37% 向上。ただし Sharpe=+0.888 < E4。WFA 未実施。 |')
W('| **D5** vz=0.65/lmax=5.5 | vz_thr=0.65 × l_max チューニング | **PASS** | G10 PASS (CI95_lo=+25.7%, WFE=+1.257)。CAGR=+34.6%, Sharpe=+0.945。gap=-4.14pp⚠ |')
W('| **D5** vz=0.65/lmax=6.0 | vz_thr=0.65 × l_max チューニング | **PASS** | G10 PASS (CI95_lo=+26.3%, WFE=+1.244)。CAGR=+35.3%, Sharpe=+0.939。gap=-4.20pp⚠ |')
W('| **D5** vz=0.65/lmax=7.0 | vz_thr=0.65 × l_max チューニング | **PASS⚠** | G10 PASS (CI95_lo=+26.9%, WFE=+1.230)。CAGR=+37.0%（最高）。gap=-5.22pp⚠⚠ 過学習警戒。 |')
W('| **D5** vz=0.60/lmax=4.5 | MaxDD 最小化フロンティア探索 | TBD | MaxDD=-49.2%（最良）。Sharpe=+0.885 < E4。gap=-0.15pp 最良。WFA 未実施。 |')
W('| **D5** vz=0.60/lmax=5.0 | vz=0.60 系 l_max スイープ | TBD | MaxDD=-51.1%。Sharpe=+0.900。WFA 未実施。 |')
W('| **D5** vz=0.65/lmax=4.5 | vz=0.65 系 l_max 下限探索 | TBD | MaxDD=-49.9%（vz=0.65 系最良）。Sharpe=+0.931。WFA 未実施。 |')
W('| **D5** vz=0.70/lmax=5.0 | vz_thr=0.70 (E4) × lmax=5.0 | TBD | MaxDD=-54.1%。Sharpe=+0.896。WFA 未実施。 |')
W('')
W('---')
W('')

# ── §4 G10 WFA サマリー ──────────────────────────────────────────────────────
W('## 📈 §4 G10 WFA サマリー（vz=0.65 ロー 全5戦略）')
W('')
W('> 実施: 2026-05-27 | ソース: `g10_wfa_vz065_lmax_row_summary.csv`')
W('> 49窓 WFA（252日 calendar-year-anchored, non-overlapping）')
W('> 進格条件 α: CI95_lo > 0 / 条件 β: 0.5 ≤ WFE ≤ 2.0')
W('')
W('| Strategy | 判定 | CI95_lo | WFE | 備考 |')
W('|----------|:----:|--------:|----:|------|')

g10df = pd.read_csv(ROOT / 'g10_wfa_vz065_lmax_row_summary.csv')
for _, row in g10df.iterrows():
    v = row['verdict']
    ci = row['WFA_CI95_lo']
    wfe = row['WFA_WFE']
    n = row['strategy']
    note = ''
    if n == 'REF-E4':
        note = '参照行（E4 現行ベスト）'
    elif n == 'vz065-lmax5':
        note = 'Active 候補。Sharpe 最高（WFA内）。'
    elif n == 'vz065-lmax5p5':
        note = 'CAGR=+34.6%, Sharpe=+0.945。'
    elif n == 'vz065-lmax6':
        note = 'CAGR=+35.3%, Sharpe=+0.939。CI95_lo=+26.3%（E4 比較: +26.5%）。'
    elif n == 'vz065-lmax7':
        note = 'CAGR=+37.0%（最高）。gap=-5.22pp⚠⚠ 過学習リスク。'
    mark = '✅' if v == 'PASS' else '❌'
    W(f'| {n} | {mark} {v} | +{ci * 100:.2f}% | +{wfe:.3f} | {note} |')

W('')
W('**全5戦略 WFA PASS**: vz=0.65 レジームは統計的に頑健。')
W('**WFE 傾向**: l_max が低いほど WFE 高（vz065-lmax5: 1.272 → 最高の OOS 効率性）。')
W('**CI95_lo 傾向**: l_max が高いほど CI95_lo 高だが gap も拡大 → 過学習トレードオフ。')
W('')
W('---')
W('')

# ── §5 採用判断サマリー ──────────────────────────────────────────────────────
W('## 🏆 §5 採用判断サマリー')
W('')
W('### 今回実験の総括（最高値レコード）')
W('')
W('| 指標 | 優勝戦略 | 値 |')
W('|------|---------|---|')
W('| CAGR_OOS 最高 | D5 vz=0.65/lmax=7.0⚠ | +37.0% |')
W('| Sharpe_OOS 最高 | D5 vz=0.65/lmax=5.0 (Active候補) | +0.949 |')
W('| MaxDD 最良 | D5 vz=0.60/lmax=4.5 | −49.2% |')
W('| Worst10Y★ 最高 | C3 yz_n=10 | +19.37% |')
W('| P10_5Y 最高 | F10+lmax5 (Active候補) | +12.8% |')
W('| IS-OOS gap 最小 | D5 vz=0.60/lmax=4.5 | −0.15pp |')
W('| WFA CI95_lo 最高 | F10 ε=0.015 (Active候補) | +27.9% |')
W('| Trades/yr 最少 | 全 E4/D5 系 | 27 |')
W('')
W('### 実験別採否')
W('')
W('| 実験 | 採否 | 理由 |')
W('|------|:----:|------|')
W('| **B4** k_lo=0 | ❌ 棄却 | k_lo=0.1 の E4 が優越。変更不要。 |')
W('| **A1** soft regime (α=2/3/5/8) | ❌ 棄却 | MaxDD 悪化 or Sharpe 改善微小。過学習リスク高。 |')
W('| **A2** dyn lmax | ❌ 棄却 | CAGR 低下。vol 感応 l_max は CAGR/Sharpe を下げる。 |')
W('| **A2B** rolling VOL_REF | ❌ 棄却 | ローリング中央値 < 固定 VOL_REF → 常時 l_max 削減バイアス。根本欠陥。 |')
W('| **A3** VoV dual gate | ❌ 棄却 | 1980-82 Volcker 期（最大 MaxDD）に無効。構造的限界確認。 |')
W('| **C2** adaptive deadband | ❌ 不採用 | F10 と同性能。gap 改善なし。F10 ε=0.015 固定の方がシンプル。 |')
W('| **C3** Yang-Zhang | 🔶 保留 | MaxDD/Worst10Y 改善。Sharpe やや低下。WFA 未実施→次フェーズ候補。 |')
W('| **D5** vz=0.65 × lmax | ✅ 確認 | vz_thr=0.65 が Pareto 最適（全 l_max でベスト Sharpe）。D5 vz=0.65/lmax=5.0 が Active 候補。 |')
W('')
W('### ◆ 現行ベスト維持 + 採用候補')
W('')
W('> **◆ 現行ベスト: E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7)** — Active 変更なし')
W('>')
W('> **Active 候補（ユーザー判断待ち, WFA PASS済み）**:')
W('> 1. **D5 vz=0.65/lmax=5.0**: Sharpe=+0.949（最高）, MaxDD=-51.8%（最良）, Trades=27（低コスト）')
W('>    G10 PASS (CI95_lo=+24.8%, WFE=+1.272)。gap=-3.91pp⚠ が唯一の懸念。')
W('> 2. **F10 ε=0.015**: CAGR=+36.8%（高リターン）。G7 PASS。Trades=52 高コスト・gap=-4.31pp⚠。')
W('> 3. **F10+lmax5**: P10_5Y=+12.8%（最高）, MaxDD改善。G8 PASS。Trades=52 高コスト。')
W('>')
W('> **次善候補（WFA PASS済み）**:')
W('> - **D5 vz=0.65/lmax=5.5**: CAGR=+34.6%, Sharpe=+0.945。G10 PASS。')
W('> - **D5 vz=0.65/lmax=6.0**: CAGR=+35.3%, Sharpe=+0.939。G10 PASS。')
W('>')
W('> **WFA 候補（要 WFA 実施）**:')
W('> - **C3 Yang-Zhang**: Worst10Y=+19.37%（最高）。要 WFA。')
W('> - **D5 vz=0.65/lmax=4.5**: MaxDD=-49.9%（vz=0.65 系最良）。要 WFA。')
W('')
W('---')
W('')

# ── §6 一次根拠ファイル ──────────────────────────────────────────────────────
W('## 📁 §6 一次根拠ファイル')
W('')
W('| ファイル | 実験 | 役割 |')
W('|----------|------|------|')
W('| `b4_klo_zero_results.csv` | B4 / E4 | k_lo=0 実験 9指標（E4 REF含む） |')
W('| `a1_soft_regime_klt_results.csv` | A1 | soft sigmoid 実験 9指標（α=2/3/5/8/100） |')
W('| `a2_dyn_lmax_results.csv` | A2 | vol 感応 l_max 実験 9指標 |')
W('| `a2b_dyn_lmax_rolling_vref_results.csv` | A2B | rolling VOL_REF 実験 9指標 |')
W('| `a3_regime_asset_tilt_results.csv` | A3 | VoV+vz dual gate 実験 9指標 |')
W('| `c2_adaptive_deadband_results.csv` | C2 | adaptive deadband 実験 9指標 |')
W('| `c3_yang_zhang_results.csv` | C3 | Yang-Zhang 実験 9指標 |')
W('| `d5_vz_lmax_grid_results.csv` | D5 | vz_thr×l_max グリッド 9指標（20 config） |')
W('| `g10_wfa_vz065_lmax_row_summary.csv` | G10 | vz=0.65 ロー WFA サマリー（5戦略） |')
W('| `g10_wfa_vz065_lmax_row_per_window.csv` | G10 | vz=0.65 ロー WFA 窓別詳細 |')
W('| `f10lmax5_fullmetrics.csv` | F10+lmax5 | F10+lmax5 フル 9指標 |')
W('| `g8_wfa_lmax5_summary.csv` | G8 | F10+lmax5 / E4-lmax5 WFA |')
W('| `f10_epsilon_deadband_results.csv` | F10 | F10 ε sweep 9指標 |')
W('| `g7_wfa_f10_summary.csv` | G7 | F10 ε=0.015 WFA |')
W('| `g3_wfa_e4_summary.csv` | G3 | E4 ◆ WFA |')
W('| `EVALUATION_STANDARD.md` | — | 評価基準 v1.1 |')
W('| `CURRENT_BEST_STRATEGY.md` | — | ベスト戦略単一の真実 |')
W('')
W('---')
W('')

# ── §7 改訂履歴 ──────────────────────────────────────────────────────────────
W('## 📝 §7 改訂履歴')
W('')
W('| 版 | 日付 | 変更内容 |')
W('|----|------|---------|')
W('| **v1.0** | 2026-05-27 | 初版。Group A × 4 + Group B × 13 計 17 戦略の §3.12 準拠 9指標比較表（2 テーブル分割）。 |')
W('| **v1.7** | 2026-05-27 | Group A/B 分割を廃止。**1 テーブル × 21 戦略** に統合。A1 α=3/5、D5 vz=0.60/lmax=5.0、vz=0.65/lmax=4.5、vz=0.70/lmax=5.0 の 4 行を追加。実験別判定サマリーを 1 テーブルに統合。 |')
W('')
W('---')
W('')
W('*管理者: Kazuya Murayama*')
W('*準拠: `EVALUATION_STANDARD.md v1.1` / `src/_sweep_format.py MD_HEADER_STRAT`*')

# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------
md_text = '\n'.join(lines) + '\n'
OUT_MD.write_text(md_text, encoding='utf-8')
print(f'Written: {OUT_MD}')
print(f'Lines: {len(lines)}')
