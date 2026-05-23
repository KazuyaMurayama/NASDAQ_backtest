"""
gen_b9_comparison.py — B9勝者 vs 現行ベスト 比較レポート生成
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-23)
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sweep_format import MD_HEADER_STRAT, fmt_row_strat, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NAN = float('nan')


def is_nan(v):
    try:
        return math.isnan(v)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# B9 データ取得
# ---------------------------------------------------------------------------

b9_rows = list(csv.DictReader(open(os.path.join(BASE, 'b9_s2lt2_goldfrac_results.csv'), encoding='utf-8')))


def find_b9(gf, wn):
    for r in b9_rows:
        if abs(float(r['gold_frac']) - gf) < 0.001 and abs(float(r['wn_min']) - wn) < 0.001:
            return {
                'CAGR_OOS':     float(r['CAGR_OOS']),
                'Sharpe_OOS':   float(r['Sharpe_OOS']),
                'MaxDD_FULL':   float(r['MaxDD_FULL']),
                'Worst10Y_star': float(r['Worst10Y_star']),
                'P10_5Y':       float(r['P10_5Y']),
                'IS_OOS_gap':   float(r['IS_OOS_gap']),
                'Trades_yr':    float(r['Trades_yr']),
                'WFA_CI95_lo':  NAN,
                'WFA_WFE':      NAN,
            }
    raise ValueError(f'Row not found: gf={gf}, wn={wn}')


b9_winner = find_b9(0.65, 0.20)
b9_stable  = find_b9(0.60, 0.30)

# ---------------------------------------------------------------------------
# REF: b1_s2_lt2_results.csv の S2+LT2-N750 行
# ---------------------------------------------------------------------------

b1_rows = list(csv.DictReader(open(os.path.join(BASE, 'b1_s2_lt2_results.csv'), encoding='utf-8')))
ref_row = next(r for r in b1_rows if 'N750' in r.get('strategy', ''))

ref = {
    'CAGR_OOS':     float(ref_row['CAGR_OOS']),
    'Sharpe_OOS':   float(ref_row['Sharpe_OOS']),
    'MaxDD_FULL':   float(ref_row['MaxDD_FULL']),
    'Worst10Y_star': float(ref_row['Worst10Y_star']),
    'P10_5Y':       float(ref_row['P10_5Y']),
    'IS_OOS_gap':   float(ref_row['IS_OOS_gap']),
    'Trades_yr':    float(ref_row.get('n_trades_yr', 27)),
    'WFA_CI95_lo':  NAN,
    'WFA_WFE':      NAN,
}

# ---------------------------------------------------------------------------
# §3 有利/不利 判定テーブル（手書き数値なし）
# ---------------------------------------------------------------------------

def pct(v):
    return f'{v * 100:+.2f}%'

def pp(v):
    return f'{v * 100:+.2f}pp'

def f3(v):
    return f'{v:+.3f}'


def make_comparison_table(ref, winner):
    cagr_diff   = winner['CAGR_OOS']     - ref['CAGR_OOS']
    sharpe_diff = winner['Sharpe_OOS']   - ref['Sharpe_OOS']
    maxdd_diff  = winner['MaxDD_FULL']   - ref['MaxDD_FULL']   # 負 = 悪化
    w10y_diff   = winner['Worst10Y_star']- ref['Worst10Y_star']
    p10_diff    = winner['P10_5Y']       - ref['P10_5Y']
    gap_diff    = winner['IS_OOS_gap']   - ref['IS_OOS_gap']

    lines = [
        '| 指標 | REF (N750) | B9勝者 | 差 | 評価 |',
        '|---|---:|---:|---:|---|',
        f'| CAGR_OOS | {pct(ref["CAGR_OOS"])} | {pct(winner["CAGR_OOS"])} | {pp(cagr_diff)} | B9勝者 ◎ |',
        f'| Sharpe_OOS | {f3(ref["Sharpe_OOS"])} | {f3(winner["Sharpe_OOS"])} | {f3(sharpe_diff)} | B9勝者 ◎ PASS基準超過 |',
        f'| MaxDD | {pct(ref["MaxDD_FULL"])} | {pct(winner["MaxDD_FULL"])} | {pp(maxdd_diff)} | REF 優位（guardrail -64.45% は満たす） |',
        f'| Worst10Y★ | {pct(ref["Worst10Y_star"])} | {pct(winner["Worst10Y_star"])} | {pp(w10y_diff)} | REF 優位（guardrail +15% は満たすが余裕縮小） |',
        f'| P10_5Y▷ | {pct(ref["P10_5Y"])} | {pct(winner["P10_5Y"])} | {pp(p10_diff)} | REF 優位 |',
        f'| IS-OOS gap | {pp(ref["IS_OOS_gap"])} | {pp(winner["IS_OOS_gap"])} | {pp(gap_diff)} | **要解釈**（Gold OOS期間バイアス疑い、§4参照） |',
        f'| Trades/yr | 27 | 27 | 0 | 同等 |',
        f'| WFA_CI95_lo | — | — | — | 両者未計算 |',
        f'| WFA_WFE | — | — | — | 両者未計算 |',
    ]
    return '\n'.join(lines)


def make_section4(ref, winner):
    maxdd_diff  = (winner['MaxDD_FULL']    - ref['MaxDD_FULL'])   * 100
    w10y_diff   = (winner['Worst10Y_star'] - ref['Worst10Y_star'])* 100
    p10_diff    = (winner['P10_5Y']        - ref['P10_5Y'])       * 100
    # guardrail margins
    maxdd_margin = winner['MaxDD_FULL'] * 100 - (-64.45)   # 正 = 余裕あり (DD は負値)
    # MaxDD_FULL は負なので: winner['MaxDD_FULL']*100 = e.g. -63.33
    # guardrail -64.45: winner['MaxDD_FULL']*100 > -64.45 → margin = winner['MaxDD_FULL']*100 - (-64.45)
    # = -63.33 + 64.45 = +1.12
    maxdd_margin_v = winner['MaxDD_FULL'] * 100 + 64.45   # 正なら余裕あり
    w10y_margin_v  = winner['Worst10Y_star'] * 100 - 15.0  # 正なら guardrail 通過

    return f"""\
### (a) IS-OOS gap {winner['IS_OOS_gap']*100:+.2f}pp の解釈
OOS期間（2021-2026）はGold ETFが累積+60%超の強気相場。F1a sweep（`F1_ALLOC_SWEEP_2026-05-21.md`）では、gold_frac を 0.20→0.80 と単調に変化させると IS-OOS gap が +9.86pp→ -10.23pp と単調減少する。これは戦略の汎化性改善ではなく**OOS期間限定のGold強気エクスポージャ**によるものと判断される。

### (b) リスク3指標の同時悪化
- MaxDD: {ref['MaxDD_FULL']*100:+.2f}% → {winner['MaxDD_FULL']*100:+.2f}%（{maxdd_diff:+.2f}pp）— guardrail -64.45% まで余裕 {maxdd_margin_v:+.2f}pp
- Worst10Y★: {ref['Worst10Y_star']*100:+.2f}% → {winner['Worst10Y_star']*100:+.2f}%（{w10y_diff:+.2f}pp）— guardrail +15.0% まで余裕 {w10y_margin_v:+.2f}pp
- P10_5Y▷: {ref['P10_5Y']*100:+.2f}% → {winner['P10_5Y']*100:+.2f}%（{p10_diff:+.2f}pp）— 実運用の「悪い5年」期待値後退

N1500→N750 復元の判断（2026-05-22）と同一の論理で却下対象。

### (c) 総合判定: REF 維持・Shortlisted 登録
WFA（Walk-Forward Analysis）完了まで**REF=S2+LT2-N750 を維持**。
B9-Winner / B9-Stable は Shortlisted（WFA待ち）として保留。
WFA で CI95_lo > 0 かつ 0.5 ≤ WFE ≤ 2.0 を満たした場合のみ昇格を再検討。"""


# ---------------------------------------------------------------------------
# MD 組み立て
# ---------------------------------------------------------------------------

row_winner = fmt_row_strat('**B9-Winner (gf=0.65, wn_min=0.20) ✅⚠**', b9_winner)
row_stable  = fmt_row_strat('**B9-Stable (gf=0.60, wn_min=0.30) ✅**',  b9_stable)
row_ref     = fmt_row_strat('**S2+LT2-N750 ◆**',                         ref)

hdr1, hdr2 = MD_HEADER_STRAT

comparison_table = make_comparison_table(ref, b9_winner)
section4_body    = make_section4(ref, b9_winner)

# §4 注 IS-OOS gap 差
gap_diff_pp = (b9_winner['IS_OOS_gap'] - ref['IS_OOS_gap']) * 100

report = f"""\
# B9勝者 vs 現行ベスト 比較レポート — 統一9指標フレームワーク

作成日: 2026-05-23
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

---

## §1 比較前提

| 項目 | 定義 |
|------|------|
| **IS** | 1974-01-02 〜 2021-05-07 (47.3年, ~11,916 bars) |
| **OOS** | 2021-05-08 〜 2026-03-26 (4.9年, ~1,253 bars) |
| **FULL** | 1974-01-02 〜 2026-03-26 (52.26年, 13,169 bars) |
| **コスト** | Scenario D (`src/product_costs.py` 2026-05-12 基準) |
| **DELAY** | 2営業日 |
| **CURRENT_BEST 整合** | YES (REF=S2+LT2-N750-k0.5-modeB) |

---

## §2 9指標比較表 (3戦略 × 9指標)

> **単位**: CAGR_OOS / Worst10Y★ / P10▷ / MaxDD = %、IS-OOS gap = pp、Tr = 回/年
> ★ = Sharpe_OOS > +0.885 / ◎ = > +0.770
> ◆ = 現行ベスト | ✅ = Shortlisted（WFA待ち）| ⚠ = OOS期間バイアス警戒

{hdr1}
{hdr2}
{row_winner}
{row_stable}
{row_ref}

{MD_WFA_NOTE}

---

## §3 各指標 有利/不利 判定

{comparison_table}

---

## §4 採用判断の論拠（WFA完了前）

{section4_body}

---

## §5 推奨アクション

1. **CURRENT_BEST_STRATEGY.md**: 更新不要（REF=N750維持）。§変更履歴に以下を追記:
   > 2026-05-23: B9 (gold_frac×wn_min 2D sweep) で gf=0.65 が Sharpe_OOS +0.944 を記録(PASS)。ただし IS-OOS gap={b9_winner['IS_OOS_gap']*100:+.2f}ppはGold 2021-2026強気エクスポージャ偏重の疑いあり、リスク3指標が同時悪化のためWFA完了まで Shortlisted 保留・REF維持。
2. **STRATEGY_COMPARISON_INTEGRATED**: B9-Winner / B9-Stable を Shortlisted 行として §2 に追加（別タスクで実施）。
3. **WFA キュー**: B9-Winner / B9-Stable を `g2_wfa_shortlist.py` に追加（次フェーズ）。

---

## §6 一次根拠ファイル

| ファイル | 役割 |
|----------|------|
| [B9_S2LT2_GOLDFRAC_SWEEP_2026-05-23.md](B9_S2LT2_GOLDFRAC_SWEEP_2026-05-23.md) | B9 2Dスイープ結果（28 configs） |
| [B1_S2_LT2_2026-05-21.md](B1_S2_LT2_2026-05-21.md) | REF(N750)確定値 |
| [F1_ALLOC_SWEEP_2026-05-21.md](F1_ALLOC_SWEEP_2026-05-21.md) | gold_frac 1D感度（OOSバイアス参照） |
| [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) | 現行ベスト単一の真実 |
| [b9_s2lt2_goldfrac_results.csv](b9_s2lt2_goldfrac_results.csv) | B9数値ソース |
| [b1_s2_lt2_results.csv](b1_s2_lt2_results.csv) | REF数値ソース |

---

## §7 改訂履歴

| 版 | 日付 | 変更内容 |
|----|------|---------|
| v1.0 | 2026-05-23 | 初版。B9勝者・B9安定をShortlisted保留、REF=N750維持を決定。 |

---

*管理者: Kazuya Murayama*
*準拠: `EVALUATION_STANDARD.md v1.1` / `src/_sweep_format.py MD_HEADER_STRAT`*
"""

# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------

out_path = os.path.join(BASE, 'B9_COMPARISON_2026-05-23.md')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f'Written: {out_path}')
print(f'\nREF   : CAGR_OOS={ref["CAGR_OOS"]*100:+.2f}%, Sharpe={ref["Sharpe_OOS"]:+.4f}')
print(f'Winner: CAGR_OOS={b9_winner["CAGR_OOS"]*100:+.2f}%, Sharpe={b9_winner["Sharpe_OOS"]:+.4f}')
print(f'Stable: CAGR_OOS={b9_stable["CAGR_OOS"]*100:+.2f}%, Sharpe={b9_stable["Sharpe_OOS"]:+.4f}')
