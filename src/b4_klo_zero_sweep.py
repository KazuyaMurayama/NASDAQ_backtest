"""
B4: k_lo=0.0 Sweep
================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

概念: E4（k_lo=0.1）の弱気時レバレッジ乗数を 0.0 にして
  弱気時（vz < -vz_thr）に LT2 バイアスを完全クローズすることで、
  スワップコストを削減し、MaxDD と Worst10Y★ を改善することを狙う。

  vz > +vz_threshold  → k_lt = k_hi（高ボラ: LT2バイアスを強め、防御的）
  vz < -vz_threshold  → k_lt = k_lo（低ボラ: LT2バイアスを弱め/完全クローズ）
  otherwise           → k_lt = k_mid = 0.50（REF 固定値）

  lt_bias_t = (-k_lt_t * lt_sig_t * 0.5).clip(-0.5, 0.5)   [signal_to_bias 公式]

グリッド:
  K_LO_GRID       = [0.0, 0.05, 0.1]   (0.0を含む: 完全クローズ)
  K_HI_GRID       = [0.7, 0.8]          (E4採用近傍)
  VZ_THR_GRID     = [0.5, 0.7]          (E4採用近傍)
  REF: k_lo=k_hi=k_mid=0.5, vz_thr=∞（常に固定 k=0.5）

  合計: 3×2×2=12 configs + REF

出力:
  - b4_klo_zero_results.csv
  - B4_KLO_ZERO_SWEEP_2026-05-27.md
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
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import build_lt_signal, apply_lt_mode_b
from _sweep_format import MD_HEADER_1P, fmt_row_1p, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# グリッド
# ---------------------------------------------------------------------------
K_LO_GRID   = [0.0, 0.05, 0.1]   # 0.0を含む（完全クローズ）
K_HI_GRID   = [0.7, 0.8]          # E4採用近傍
VZ_THR_GRID = [0.5, 0.7]          # E4採用近傍
K_MID       = 0.50   # 中間（閾値外）は常に k=0.5

# E4ベスト（k_lo=0.1, k_hi=0.8, vz_thr=0.7）を REF として記録
REF_CAGR_OOS   = 0.3353
REF_SHARPE_OOS = 0.891
REF_MAXDD      = -0.6001

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)

PASS_SHARPE_DELTA = 0.020
PASS_CAGR_OOS     = 0.2916
PASS_GAP          = 0.030
PASS_MAXDD        = -0.6445
PASS_WORST10Y     = 0.150


def signal_to_bias_dynamic(lt_sig_arr: np.ndarray, k_arr: np.ndarray) -> np.ndarray:
    """element-wise lt_bias = (-k * sig * 0.5).clip(-0.5, 0.5)"""
    return np.clip(-k_arr * lt_sig_arr * 0.5, -0.5, 0.5)


def compute_p10_5y(nav, td=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    return float(((s / s.shift(td * 5)) ** 0.2 - 1).dropna().quantile(0.10))


def calc_all_metrics(nav, dates, trades_yr):
    m = calc_7metrics(nav, dates, trades_per_year=trades_yr)
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, 10)
    return {**m,
            'Worst10Y_star': float(r10.min()) if len(r10) > 0 else float('nan'),
            'P10_5Y':        compute_p10_5y(nav.values),
            'IS_OOS_gap':    m['CAGR_IS'] - m['CAGR_OOS']}


def passes_all(r):
    return (r['Sharpe_OOS'] - REF_SHARPE_OOS >= PASS_SHARPE_DELTA and
            r['CAGR_OOS'] >= PASS_CAGR_OOS and
            r['IS_OOS_gap'] <= PASS_GAP and
            r['MaxDD_FULL'] > PASS_MAXDD and
            r['Worst10Y_star'] >= PASS_WORST10Y)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('B4: k_lo=0.0 Sweep — 弱気完全クローズによるスワップコスト削減検証')
    print('=' * 70)
    total = len(K_LO_GRID) * len(K_HI_GRID) * len(VZ_THR_GRID)
    print(f'グリッド: k_lo={K_LO_GRID} × k_hi={K_HI_GRID} × vz_thr={VZ_THR_GRID}  ({total} configs + REF)')

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
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_trades_yr = n_tr / n_years
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)

    lt_sig_raw = build_lt_signal(close, 'LT2', 750)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    lev_arr    = lev_raw

    print('Assets and signals built. Starting sweep...')

    # REF (k=0.5 固定)
    k_ref_arr   = np.full(n, 0.5)
    lt_bias_ref = pd.Series(signal_to_bias_dynamic(lt_sig_arr, k_ref_arr), index=lt_sig_raw.index)
    lev_mod_ref = apply_lt_mode_b(lev_arr, lt_bias_ref, l_min=0.0, l_max=1.0)
    nav_ref = build_nav_strategy(
        close, lev_mod_ref, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_ref, dates, n_trades_yr)
    m_ref.update({'k_lo': K_MID, 'k_hi': K_MID, 'vz_thr': 999.0,
                  'Trades_yr': n_trades_yr,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
    print(f'[REF k=0.5] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  Sharpe={m_ref["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%')
    diff_ref = (m_ref['CAGR_OOS'] - 0.3116) * 100
    print(f'[SANITY] diff={diff_ref:+.2f}pp → {"OK" if abs(diff_ref) <= 0.15 else "WARN"}')

    # E4ベスト（k_lo=0.1, k_hi=0.8, vz_thr=0.7）を比較 REF として計算
    print('\n[E4 REF: k_lo=0.1, k_hi=0.8, vz_thr=0.7] 計算中...')
    vz_thr_e4 = 0.7
    regime_hi_e4  = vz_arr > +vz_thr_e4
    regime_lo_e4  = vz_arr < -vz_thr_e4
    k_dyn_e4 = np.where(regime_hi_e4, 0.8, np.where(regime_lo_e4, 0.1, K_MID))
    lt_bias_e4 = pd.Series(signal_to_bias_dynamic(lt_sig_arr, k_dyn_e4), index=lt_sig_raw.index)
    lev_mod_e4 = apply_lt_mode_b(lev_arr, lt_bias_e4, l_min=0.0, l_max=1.0)
    nav_e4 = build_nav_strategy(
        close, lev_mod_e4, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_e4 = calc_all_metrics(nav_e4, dates, n_trades_yr)
    m_e4.update({'k_lo': 0.1, 'k_hi': 0.8, 'vz_thr': vz_thr_e4,
                 'Trades_yr': n_trades_yr,
                 'WFA_CI95_lo': 0.2651, 'WFA_WFE': 1.131})  # G3 WFA 確定値
    print(f'[E4 REF] CAGR_OOS={m_e4["CAGR_OOS"]*100:+.2f}%  Sharpe={m_e4["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={m_e4["MaxDD_FULL"]*100:+.2f}%  Worst10Y★={m_e4["Worst10Y_star"]*100:+.2f}%')

    results = [m_ref]  # results[0] = REF(k=0.5固定)
    e4_result = m_e4   # 別途保持（MDのREF行として出力）
    idx = 0

    for vz_thr in VZ_THR_GRID:
        regime_hi  = vz_arr > +vz_thr
        regime_lo  = vz_arr < -vz_thr

        for k_lo in K_LO_GRID:
            for k_hi in K_HI_GRID:
                idx += 1
                k_dyn = np.where(regime_hi, k_hi, np.where(regime_lo, k_lo, K_MID))
                lt_bias_dyn = pd.Series(
                    signal_to_bias_dynamic(lt_sig_arr, k_dyn),
                    index=lt_sig_raw.index,
                )
                lev_mod_dyn = apply_lt_mode_b(lev_arr, lt_bias_dyn, l_min=0.0, l_max=1.0)
                nav_dyn = build_nav_strategy(
                    close, lev_mod_dyn, wn_A, wg_A, wb_A, dates,
                    gold_2x, bond_3x, sofr,
                    nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
                )
                m = calc_all_metrics(nav_dyn, dates, n_trades_yr)
                m.update({'k_lo': k_lo, 'k_hi': k_hi, 'vz_thr': vz_thr,
                          'Trades_yr': n_trades_yr,
                          'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
                results.append(m)
                print(f'  [{idx:>2d}/{total}] vz_thr={vz_thr:.1f} k_lo={k_lo:.2f} k_hi={k_hi:.1f}: '
                      f'CAGR={m["CAGR_OOS"]*100:+.2f}%  Sharpe={m["Sharpe_OOS"]:+.3f}  '
                      f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  W10Y★={m["Worst10Y_star"]*100:+.2f}%')

    print(f'\n全 {len(results)-1} configs 完了。')

    pass_list   = [r for r in results[1:] if passes_all(r)]
    best_sharpe = max(results[1:], key=lambda r: r['Sharpe_OOS'])
    verdict     = ('PASS' if pass_list else
                   'WARN' if best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')
    print(f'PASS configs: {len(pass_list)}')
    print(f'Best Sharpe: vz_thr={best_sharpe["vz_thr"]:.1f} k_lo={best_sharpe["k_lo"]:.2f} '
          f'k_hi={best_sharpe["k_hi"]:.1f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%  IS-OOS={best_sharpe["IS_OOS_gap"]*100:+.2f}pp  '
          f'Worst10Y★={best_sharpe["Worst10Y_star"]*100:+.2f}%')
    print(f'総合判定: {verdict}')

    # E4 REF との比較
    print(f'\n[E4 REF 比較]')
    print(f'  Sharpe:    E4={m_e4["Sharpe_OOS"]:+.3f}  Best={best_sharpe["Sharpe_OOS"]:+.3f}  '
          f'Δ={best_sharpe["Sharpe_OOS"]-m_e4["Sharpe_OOS"]:+.3f}')
    print(f'  CAGR_OOS:  E4={m_e4["CAGR_OOS"]*100:+.2f}%  Best={best_sharpe["CAGR_OOS"]*100:+.2f}%  '
          f'Δ={( best_sharpe["CAGR_OOS"]-m_e4["CAGR_OOS"])*100:+.2f}pp')
    print(f'  MaxDD:     E4={m_e4["MaxDD_FULL"]*100:+.2f}%  Best={best_sharpe["MaxDD_FULL"]*100:+.2f}%  '
          f'Δ={( best_sharpe["MaxDD_FULL"]-m_e4["MaxDD_FULL"])*100:+.2f}pp')
    print(f'  Worst10Y★: E4={m_e4["Worst10Y_star"]*100:+.2f}%  Best={best_sharpe["Worst10Y_star"]*100:+.2f}%  '
          f'Δ={( best_sharpe["Worst10Y_star"]-m_e4["Worst10Y_star"])*100:+.2f}pp')

    # CSV（全 results + E4 REF）
    all_rows = []
    for r in results:
        all_rows.append({
            'k_lo': r['k_lo'], 'k_hi': r['k_hi'], 'vz_thr': r['vz_thr'],
            'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
            'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
            'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
            'IS_OOS_gap': r['IS_OOS_gap'], 'Trades_yr': r['Trades_yr'],
        })
    # E4 REF も CSV に含める
    all_rows.append({
        'k_lo': e4_result['k_lo'], 'k_hi': e4_result['k_hi'], 'vz_thr': e4_result['vz_thr'],
        'CAGR_IS': e4_result['CAGR_IS'], 'CAGR_OOS': e4_result['CAGR_OOS'],
        'Sharpe_OOS': e4_result['Sharpe_OOS'], 'MaxDD_FULL': e4_result['MaxDD_FULL'],
        'Worst10Y_star': e4_result['Worst10Y_star'], 'P10_5Y': e4_result['P10_5Y'],
        'IS_OOS_gap': e4_result['IS_OOS_gap'], 'Trades_yr': e4_result['Trades_yr'],
    })
    pd.DataFrame(all_rows).to_csv(
        os.path.join(BASE, 'b4_klo_zero_results.csv'),
        index=False, float_format='%.6f'
    )

    # MD: Sharpe 降順で全 configs 表示（12件 + E4 REF + REF k=0.5）
    hdr1, hdr2 = MD_HEADER_1P
    top_results = sorted(results[1:], key=lambda r: r['Sharpe_OOS'], reverse=True)
    rows_md = '\n'.join(
        fmt_row_1p(f'lo={r["k_lo"]:.2f}/hi={r["k_hi"]:.1f}/vz={r["vz_thr"]:.1f}', r)
        for r in top_results
    )
    ref_row     = fmt_row_1p('k=0.5 固定 (REF)', m_ref)
    e4_ref_row  = fmt_row_1p('**E4 lo=0.1/hi=0.8/vz=0.7 (REF Active)**', e4_result)

    diff_ref_val = (m_ref['CAGR_OOS'] - 0.3116) * 100

    report = f"""\
# B4: k_lo=0.0 Sweep — 弱気完全クローズ検証

作成日: 2026-05-27
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

B4: 弱気完全クローズ（k_lo=0.0）によるスワップコスト削減の検証

| 項目 | 定義 |
|------|------|
| **k_lo グリッド** | {K_LO_GRID}（0.0=完全クローズを含む） |
| **k_hi グリッド** | {K_HI_GRID}（E4採用近傍） |
| **vz_thr グリッド** | {VZ_THR_GRID}（E4採用近傍） |
| **k_mid** | {K_MID}（閾値外の中間域: 固定） |
| **合計 configs** | {total} + REF |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**レジーム割り当て**:
```
vz > +vz_thr  → k = k_hi  (高ボラ: 強い防御バイアス)
vz < -vz_thr  → k = k_lo  (低ボラ/弱気: 弱い/ゼロバイアス)
otherwise     → k = 0.50  (中間域: 固定)
```
`lt_bias = (-k × lt_sig × 0.5).clip(-0.5, 0.5)`

**比較 REF**: E4ベスト（k_lo=0.1, k_hi=0.8, vz_thr=0.7）
  CAGR_OOS=+33.53%, Sharpe_OOS=+0.891, MaxDD=-60.01%, Worst10Y★=+18.67%
  G3 WFA PASS (CI95_lo=+26.51%, WFE=+1.131)

**サニティ**: REF(k=0.5) CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (diff {diff_ref_val:+.2f}pp)

---

## §2 9指標テーブル（Sharpe 降順 + E4 REF + k=0.5 REF）

> 列 `Param` = lo=k_lo / hi=k_hi / vz=vz_thr

{hdr1}
{hdr2}
{e4_ref_row}
{rows_md}
{ref_row}

{MD_WFA_NOTE}

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ E4 REF +0.020 (≥{REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}) |
| (ii) CAGR_OOS | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap | ≤ {PASS_GAP*100:.1f}pp |
| (iv) MaxDD | > {PASS_MAXDD*100:.1f}% |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% |

- **PASS configs**: {len(pass_list)} / {total}
- **最高 Sharpe**: vz_thr={best_sharpe["vz_thr"]:.1f}, k_lo={best_sharpe["k_lo"]:.2f}, k_hi={best_sharpe["k_hi"]:.1f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}
- **総合判定: {verdict}**

---

## §4 E4 REF との比較

| 指標 | E4 REF | B4 Best | Δ | 改善/悪化 |
|------|--------|---------|---|-----------|
| Sharpe_OOS | {m_e4["Sharpe_OOS"]:+.3f} | {best_sharpe["Sharpe_OOS"]:+.3f} | {best_sharpe["Sharpe_OOS"]-m_e4["Sharpe_OOS"]:+.3f} | {"改善" if best_sharpe["Sharpe_OOS"] > m_e4["Sharpe_OOS"] else "悪化"} |
| CAGR_OOS | {m_e4["CAGR_OOS"]*100:+.2f}% | {best_sharpe["CAGR_OOS"]*100:+.2f}% | {(best_sharpe["CAGR_OOS"]-m_e4["CAGR_OOS"])*100:+.2f}pp | {"改善" if best_sharpe["CAGR_OOS"] > m_e4["CAGR_OOS"] else "悪化"} |
| MaxDD | {m_e4["MaxDD_FULL"]*100:+.2f}% | {best_sharpe["MaxDD_FULL"]*100:+.2f}% | {(best_sharpe["MaxDD_FULL"]-m_e4["MaxDD_FULL"])*100:+.2f}pp | {"改善" if best_sharpe["MaxDD_FULL"] > m_e4["MaxDD_FULL"] else "悪化"} |
| Worst10Y★ | {m_e4["Worst10Y_star"]*100:+.2f}% | {best_sharpe["Worst10Y_star"]*100:+.2f}% | {(best_sharpe["Worst10Y_star"]-m_e4["Worst10Y_star"])*100:+.2f}pp | {"改善" if best_sharpe["Worst10Y_star"] > m_e4["Worst10Y_star"] else "悪化"} |
| Trades/yr | {int(round(m_e4.get("Trades_yr",0)))} | {int(round(best_sharpe.get("Trades_yr",0)))} | — | — |
| CI95_lo | {m_e4.get("WFA_CI95_lo",float("nan"))*100:+.2f}% | — (WFA未) | — | — |

---

## §5 考察

k_lo=0.0（弱気完全クローズ）の効果検証:
- 弱気時（vz < -vz_thr）に LT2 バイアスを完全にゼロにすることでスワップコスト削減を狙う
- E4（k_lo=0.1）との差分が MaxDD・Worst10Y★ に現れるかを確認
- PASS した場合は WFA による過学習チェックが必要

---

*生成スクリプト: `src/b4_klo_zero_sweep.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/e4_regime_klt.py`*
"""
    md_path = os.path.join(BASE, 'B4_KLO_ZERO_SWEEP_2026-05-27.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "b4_klo_zero_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
