"""
D5: vz_thr × l_max 2D グリッド（vz065+lmax5 周辺フロンティア探索）
===================================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-27)

目的: 既存ベスト候補 vz065+lmax5 (Sharpe=+0.947, MaxDD=-51.8%, G9 PASS) の
  周辺フロンティアを可視化し、Sharpe-MaxDD トレードオフの最適点を確定する。

E4 ベース完全維持（k_lo=0.1, k_hi=0.8, LT2-N750）。
変更点: vz_thr と l_max を 2D グリッドで動かす。

グリッド:
  VZ_THR_GRID = [0.60, 0.65, 0.70, 0.75]     # 4点
  LMAX_GRID   = [4.5, 5.0, 5.5, 6.0, 7.0]    # 5点
  合計: 4×5 = 20 configs（うち vz=0.70/lmax=7.0 が REF=E4）

知見背景:
  - A2/A2B/A3 で「ボラ系ゲートは1980-82ボルカー期のMaxDDを検出不可」と確定
  - l_max 引下げは CFDレバ上限を直接下げるので 1980-82 を直接緩和できる
  - vz_thr 引下げ（0.65-0.60）はレジーム切替を早め、高ボラ期の rebalance 頻度を上げる
  - vz065+lmax5 単点が G9 WFA PASS（CI95_lo=+24.82%）済 → 周辺も PASS 期待

出力:
  - d5_vz_lmax_grid_results.csv
  - D5_VZ_LMAX_GRID_2026-05-27.md
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr, build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_a2_signal, simulate_rebalance_A,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy, calc_7metrics,
    CFD_SPREAD_LOW, IS_START, IS_END, OOS_START,
)
from compute_cfd_worst10y import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import build_lt_signal, apply_lt_mode_b
from a2_dyn_lmax import (
    compute_L_s2_dyn_lmax, signal_to_bias_dynamic,
    compute_p10_5y, calc_all_metrics,
    S2_BASE, K_LO, K_HI, K_MID,
    REF_CAGR_OOS, REF_SHARPE_OOS, REF_MAXDD,
    PASS_SHARPE_DELTA, PASS_CAGR_OOS, PASS_GAP, PASS_MAXDD, PASS_WORST10Y,
)
from _sweep_format import MD_HEADER_1P, fmt_row_1p, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# D5 グリッド
# ---------------------------------------------------------------------------
VZ_THR_GRID = [0.60, 0.65, 0.70, 0.75]
LMAX_GRID   = [4.5, 5.0, 5.5, 6.0, 7.0]
LT_N = 750  # LT2-N750 (E4採用値)

# E4固定値
K_LO_E4 = 0.1; K_HI_E4 = 0.8; K_MID_E4 = 0.50

# REFはvz=0.70, l_max=7.0 (= E4ベスト)
REF_VZ_THR = 0.70; REF_LMAX = 7.0


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('D5: vz_thr × l_max 2D グリッド（vz065+lmax5 周辺フロンティア）')
    print('=' * 70)
    total = len(VZ_THR_GRID) * len(LMAX_GRID)
    print(f'グリッド: VZ_THR={VZ_THR_GRID} × LMAX={LMAX_GRID} ({total} configs)')
    print(f'E4固定値: K_LO={K_LO_E4}, K_HI={K_HI_E4}, LT2-N{LT_N}')

    # --- データロード ---
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({n:,} days)')

    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                          bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_trades_yr = n_tr / n_years

    lt_sig_raw = build_lt_signal(close, 'LT2', LT_N)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values

    print('Assets and signals built. Starting sweep...')

    results = []
    idx = 0
    m_ref = None

    for vz_thr_loc in VZ_THR_GRID:
        # vz_thr ごとにレジームとlev_modを再計算
        regime_hi = vz_arr > +vz_thr_loc
        regime_lo = vz_arr < -vz_thr_loc
        k_dyn     = np.where(regime_hi, K_HI_E4, np.where(regime_lo, K_LO_E4, K_MID_E4))
        lt_bias   = pd.Series(signal_to_bias_dynamic(lt_sig_arr, k_dyn),
                              index=lt_sig_raw.index)
        lev_mod   = apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)

        for l_max_loc in LMAX_GRID:
            idx += 1
            l_max_series = pd.Series(l_max_loc, index=ret.index)
            L_s2 = compute_L_s2_dyn_lmax(ret, vz, l_max_series, **S2_BASE)

            nav = build_nav_strategy(
                close, lev_mod, wn_A, wg_A, wb_A, dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
            )
            m = calc_all_metrics(nav, dates, n_trades_yr)
            m.update({'vz_thr': vz_thr_loc, 'l_max': l_max_loc,
                      'Trades_yr': n_trades_yr,
                      'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
            results.append(m)

            # REF（E4ベスト）を記録
            if abs(vz_thr_loc - REF_VZ_THR) < 1e-6 and abs(l_max_loc - REF_LMAX) < 1e-6:
                m_ref = m
                diff_ref = (m['CAGR_OOS'] - REF_CAGR_OOS) * 100
                print(f'[REF vz={REF_VZ_THR}/lmax={REF_LMAX}] '
                      f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
                      f'Sharpe={m["Sharpe_OOS"]:+.3f}  MaxDD={m["MaxDD_FULL"]*100:+.2f}%')
                print(f'[SANITY] REF CAGR_OOS diff={diff_ref:+.2f}pp → '
                      f'{"OK" if abs(diff_ref) <= 0.30 else "WARN (>0.30pp)"}')

            print(f'  [{idx:>2d}/{total}] vz={vz_thr_loc:.2f} lmax={l_max_loc:.1f}: '
                  f'CAGR={m["CAGR_OOS"]*100:+.2f}%  Sharpe={m["Sharpe_OOS"]:+.3f}  '
                  f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%')

    print(f'\n全 {total} configs 完了。')

    if m_ref is None:
        m_ref = results[0]

    # --- 判定 ---
    def passes_all(r):
        return (r['Sharpe_OOS'] - REF_SHARPE_OOS >= PASS_SHARPE_DELTA and
                r['CAGR_OOS'] >= PASS_CAGR_OOS and
                r['IS_OOS_gap'] <= PASS_GAP and
                r['MaxDD_FULL'] > PASS_MAXDD and
                r['Worst10Y_star'] >= PASS_WORST10Y)

    non_ref = [r for r in results if not (abs(r['vz_thr'] - REF_VZ_THR) < 1e-6 and
                                          abs(r['l_max'] - REF_LMAX) < 1e-6)]
    pass_list   = [r for r in non_ref if passes_all(r)]
    best_sharpe = max(results, key=lambda r: r['Sharpe_OOS'])
    best_maxdd  = max(results, key=lambda r: r['MaxDD_FULL'])
    improved_dd = [r for r in non_ref if r['MaxDD_FULL'] > m_ref['MaxDD_FULL']]

    print(f'PASS configs: {len(pass_list)} / {len(non_ref)}')
    print(f'MaxDD改善 configs: {len(improved_dd)} / {len(non_ref)}')
    print(f'Best Sharpe: vz={best_sharpe["vz_thr"]:.2f} lmax={best_sharpe["l_max"]:.1f} '
          f'→ Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%')
    print(f'Best MaxDD: vz={best_maxdd["vz_thr"]:.2f} lmax={best_maxdd["l_max"]:.1f} '
          f'→ MaxDD={best_maxdd["MaxDD_FULL"]*100:+.2f}%  CAGR={best_maxdd["CAGR_OOS"]*100:+.2f}%')

    verdict = ('PASS' if pass_list else
               'WARN' if best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')
    print(f'総合判定: {verdict}')

    # --- CSV出力 ---
    pd.DataFrame([{
        'vz_thr': r['vz_thr'], 'l_max': r['l_max'],
        'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'], 'Trades_yr': r['Trades_yr'],
    } for r in results]).to_csv(
        os.path.join(BASE, 'd5_vz_lmax_grid_results.csv'),
        index=False, float_format='%.6f')

    # --- MD出力 (MD_HEADER_1P 使用、複合ラベル) ---
    hdr1, hdr2 = MD_HEADER_1P
    ref_row = fmt_row_1p(f'vz={REF_VZ_THR:.2f}/lmax={REF_LMAX:.1f} (REF=E4)', m_ref)

    # Sharpe降順でソート（REFを先頭に除いてから）
    top_results = sorted(results, key=lambda r: r['Sharpe_OOS'], reverse=True)
    rows_md = '\n'.join(
        fmt_row_1p(f'vz={r["vz_thr"]:.2f}/lmax={r["l_max"]:.1f}', r)
        for r in top_results
    )

    dd_delta   = (best_maxdd['MaxDD_FULL'] - m_ref['MaxDD_FULL']) * 100
    cagr_delta = (best_maxdd['CAGR_OOS']   - m_ref['CAGR_OOS'])   * 100

    report = f"""\
# D5: vz_thr × l_max 2D グリッド — vz065+lmax5 周辺フロンティア探索

作成日: 2026-05-27
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

### 背景
A2/A2B/A3 の実験で「ボラ系ゲート（VoV/vz条件）は 1980-82 ボルカー期の MaxDD
（-60.01%、441日中stress発動=0日）を構造的に検出不可能」と確定した。
l_max 引下げは CFD レバ上限を直接下げるため、ボラ条件に依存せず 1980-82 期の
NAV 破壊を緩和できる唯一のボラ系アプローチ。本 D5 は vz_thr と l_max の
2次元グリッドで Sharpe-MaxDD トレードオフフロンティアを可視化する。

### 参照: 既知ベスト候補
既存実験 vz065+lmax5（Sharpe=+0.947, MaxDD=-51.8%, Trades/yr=27）は
G9 WFA PASS (CI95_lo=+24.82%)。本 D5 の vz=0.65/lmax=5.0 で同値を再現するか確認する。

| 項目 | 定義 |
|------|------|
| **E4固定値** | k_lo={K_LO_E4}, k_hi={K_HI_E4}, LT2-N{LT_N}（変更なし） |
| **VZ_THR グリッド** | {VZ_THR_GRID}（vz閾値、レジーム切替感度） |
| **LMAX グリッド** | {LMAX_GRID}（CFDレバ上限） |
| **合計 configs** | {total}（含む REF: vz=0.70/lmax=7.0=E4） |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**サニティ**: REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (E4={REF_CAGR_OOS*100:+.2f}%)

---

## §2 9指標テーブル（Sharpe 降順）

{hdr1}
{hdr2}
{rows_md}

{MD_WFA_NOTE}

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ REF+{PASS_SHARPE_DELTA:.3f} (≥{REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}) |
| (ii) CAGR_OOS | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap | ≤ {PASS_GAP*100:.1f}pp |
| (iv) MaxDD | > {PASS_MAXDD*100:.1f}% |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% |

- **PASS configs**: {len(pass_list)} / {len(non_ref)}
- **MaxDD改善 configs**: {len(improved_dd)} / {len(non_ref)}
- **最高 Sharpe**: vz={best_sharpe["vz_thr"]:.2f}, lmax={best_sharpe["l_max"]:.1f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}
- **最良 MaxDD**: vz={best_maxdd["vz_thr"]:.2f}, lmax={best_maxdd["l_max"]:.1f} → MaxDD={best_maxdd["MaxDD_FULL"]*100:+.2f}% (E4比 {dd_delta:+.2f}pp), CAGR_OOS={best_maxdd["CAGR_OOS"]*100:+.2f}% (E4比 {cagr_delta:+.2f}pp)
- **総合判定: {verdict}**

---

## §4 考察

- vz_thr 引下げ（0.70→0.65→0.60）: レジーム切替が早まり、高ボラ期の rebalance 頻度増加
- l_max 引下げ（7.0→5.0→4.5）: CFD レバ上限を直接下げ、1980-82 期の NAV 破壊を緩和
- 両軸の交互作用: vz_thr 引下げと l_max 引下げの組み合わせは独立効果の和ではない
  （vz_thr が低いと高ボラ期に k_dyn の変化が早まり、lt_bias も変わる）
- vz065+lmax5 の再現性: 本 D5 の vz=0.65/lmax=5.0 が G9 PASS の +0.947 を再現するか確認

---

*生成スクリプト: `src/d5_vz_lmax_grid.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `INTEGRATION_DEBATE_2026-05-26.md`, `EVALUATION_STANDARD.md`*
"""
    md_path = os.path.join(BASE, 'D5_VZ_LMAX_GRID_2026-05-27.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "d5_vz_lmax_grid_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
