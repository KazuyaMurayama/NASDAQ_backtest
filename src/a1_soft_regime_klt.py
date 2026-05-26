"""
A1: Soft-Regime k_lt (sigmoid連続化)
=====================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-27)

概念: VZ（ボラティリティ z-score）に応じて k_lt を連続 sigmoid 関数で変化させる。
  k(vz) = K_LO + (K_HI - K_LO) * sigmoid(alpha * vz)
  where sigmoid(x) = 1 / (1 + exp(-x))

  E4 の離散3点判定（k_lo/k_mid/k_hi）と異なり、境界でのホイップソー損失を排除。
  IS-OOS gap 縮小・Sharpe 改善を狙う。

  lt_bias_t = (-k_lt_t * lt_sig_t * 0.5).clip(-0.5, 0.5)   [signal_to_bias 公式]

グリッド:
  K_LO = 0.1, K_HI = 0.8   （E4 採用値固定）
  ALPHA_GRID = [2.0, 3.0, 5.0, 8.0]   （sigmoid 勾配）
  REF: alpha=100 ≈ E4 離散版 (k_lo=0.1, k_hi=0.8, vz_thr=0.7) の近似

出力:
  - a1_soft_regime_klt_results.csv
  - A1_SOFT_REGIME_KLT_2026-05-27.md

参照: E4採用 config = k_lo=0.1, k_hi=0.8, vz_thr=0.7
  CAGR_OOS=+33.53%, Sharpe_OOS=+0.891, MaxDD=-60.01%, Worst10Y★=+18.67%, IS-OOS gap=-1.81pp
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
# グリッド（E4採用値固定 + alpha スイープ）
# ---------------------------------------------------------------------------
K_LO = 0.1   # E4採用値固定（低ボラ端）
K_HI = 0.8   # E4採用値固定（高ボラ端）
VZ_THR = 0.7  # 参考用（sigmoid では使わない）
ALPHA_GRID = [2.0, 3.0, 5.0, 8.0]   # sigmoid 勾配

# alpha=100 を REF として追加（→ E4 離散版の近似）
ALPHA_REF = 100.0

REF_CAGR_OOS   = 0.3353   # E4ベスト (k_lo=0.1, k_hi=0.8, vz_thr=0.7)
REF_SHARPE_OOS = 0.891
REF_MAXDD      = -0.6001

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)

PASS_SHARPE_DELTA = 0.020
PASS_CAGR_OOS     = 0.2916
PASS_GAP          = 0.030
PASS_MAXDD        = -0.6445
PASS_WORST10Y     = 0.150


def sigmoid_k(vz_arr: np.ndarray, alpha: float) -> np.ndarray:
    """Sigmoid 連続 k_lt: k(vz) = K_LO + (K_HI - K_LO) * sigmoid(alpha * vz)
    オーバーフロー対策: alpha * vz を clip(-500, 500) してから exp を計算
    """
    x = np.clip(alpha * vz_arr, -500.0, 500.0)
    return K_LO + (K_HI - K_LO) / (1.0 + np.exp(-x))


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


def run_one_alpha(alpha, lt_sig_arr, vz_arr, lev_arr, lev_raw,
                  close, wn_A, wg_A, wb_A, dates,
                  gold_2x, bond_3x, sofr, L_s2, lt_sig_raw, n_trades_yr):
    """alpha 1点分のバックテストを実行してメトリクス dict を返す"""
    # Sigmoid 連続 k_lt
    k_dyn = sigmoid_k(vz_arr, alpha)
    lt_bias_dyn = pd.Series(
        signal_to_bias_dynamic(lt_sig_arr, k_dyn),
        index=lt_sig_raw.index,
    )
    lev_mod_dyn = apply_lt_mode_b(lev_raw, lt_bias_dyn, l_min=0.0, l_max=1.0)
    nav_dyn = build_nav_strategy(
        close, lev_mod_dyn, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m = calc_all_metrics(nav_dyn, dates, n_trades_yr)
    m.update({'alpha': alpha,
              'Trades_yr': n_trades_yr,
              'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
    return m


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('A1: Soft-Regime k_lt (sigmoid連続化) スイープ')
    print('=' * 70)
    total = len(ALPHA_GRID)
    print(f'グリッド: K_LO={K_LO}, K_HI={K_HI}, ALPHA_GRID={ALPHA_GRID}  ({total} configs + REF)')
    print(f'REF: alpha={ALPHA_REF} ≈ E4離散版 (k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR})')

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

    print('Assets and signals built. Starting sweep...')

    # ---------------------------------------------------------------------------
    # REF: alpha=100 ≈ E4 離散版近似
    # ---------------------------------------------------------------------------
    m_ref = run_one_alpha(
        ALPHA_REF, lt_sig_arr, vz_arr, lev_arr=lev_raw, lev_raw=lev_raw,
        close=close, wn_A=wn_A, wg_A=wg_A, wb_A=wb_A, dates=dates,
        gold_2x=gold_2x, bond_3x=bond_3x, sofr=sofr, L_s2=L_s2,
        lt_sig_raw=lt_sig_raw, n_trades_yr=n_trades_yr,
    )
    print(f'[REF alpha={ALPHA_REF:.0f}] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  '
          f'Sharpe={m_ref["Sharpe_OOS"]:+.3f}  MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%')
    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    sanity_ok = abs(diff_ref) <= 0.2
    print(f'[SANITY] diff={diff_ref:+.2f}pp → {"OK" if sanity_ok else "WARN (>0.2pp)"}')

    results = []
    for idx, alpha in enumerate(ALPHA_GRID, 1):
        m = run_one_alpha(
            alpha, lt_sig_arr, vz_arr, lev_arr=lev_raw, lev_raw=lev_raw,
            close=close, wn_A=wn_A, wg_A=wg_A, wb_A=wb_A, dates=dates,
            gold_2x=gold_2x, bond_3x=bond_3x, sofr=sofr, L_s2=L_s2,
            lt_sig_raw=lt_sig_raw, n_trades_yr=n_trades_yr,
        )
        results.append(m)
        print(f'  [{idx:>2d}/{total}] alpha={alpha:.1f}: '
              f'CAGR={m["CAGR_OOS"]*100:+.2f}%  Sharpe={m["Sharpe_OOS"]:+.3f}  '
              f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  IS-OOS={m["IS_OOS_gap"]*100:+.2f}pp')

    print(f'\n全 {len(results)} configs 完了。')

    pass_list   = [r for r in results if passes_all(r)]
    best_sharpe = max(results, key=lambda r: r['Sharpe_OOS'])
    verdict     = ('PASS' if pass_list else
                   'WARN' if best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')
    print(f'PASS configs: {len(pass_list)}')
    print(f'Best Sharpe: alpha={best_sharpe["alpha"]:.1f} → '
          f'Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  '
          f'CAGR_OOS={best_sharpe["CAGR_OOS"]*100:+.2f}%  '
          f'MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%  '
          f'IS-OOS={best_sharpe["IS_OOS_gap"]*100:+.2f}pp')
    print(f'E4 REF (alpha=100): Sharpe={m_ref["Sharpe_OOS"]:+.3f}  '
          f'CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%')
    print(f'総合判定: {verdict}')

    # ---------------------------------------------------------------------------
    # CSV（REF + 全 alpha configs）
    # ---------------------------------------------------------------------------
    all_rows = [m_ref] + results
    pd.DataFrame([{
        'alpha': r['alpha'],
        'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'], 'Trades_yr': r['Trades_yr'],
    } for r in all_rows]).to_csv(
        os.path.join(BASE, 'a1_soft_regime_klt_results.csv'),
        index=False, float_format='%.6f'
    )

    # ---------------------------------------------------------------------------
    # MD（§3.12 準拠 / MD_HEADER_1P 使用）
    # ---------------------------------------------------------------------------
    hdr1, hdr2 = MD_HEADER_1P
    rows_md = '\n'.join(
        fmt_row_1p(f'α={r["alpha"]:.1f}', r)
        for r in sorted(results, key=lambda r: r['Sharpe_OOS'], reverse=True)
    )
    ref_row = fmt_row_1p(f'α={ALPHA_REF:.0f} (REF≈E4)', m_ref)

    report = f"""\
# A1: Soft-Regime k_lt (sigmoid連続化) スイープ

作成日: 2026-05-27
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

A1: Soft-Regime k_lt (sigmoid連続化) - 境界ジャンプ除去によるSharpe改善

| 項目 | 定義 |
|------|------|
| **K_LO** | {K_LO}（E4採用値固定: 低ボラ端） |
| **K_HI** | {K_HI}（E4採用値固定: 高ボラ端） |
| **ALPHA_GRID** | {ALPHA_GRID}（sigmoid 勾配） |
| **REF** | α={ALPHA_REF:.0f} ≈ E4離散版 (k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}) |
| **合計 configs** | {total} + REF |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**sigmoid 連続 k_lt 計算式**:
```
k(vz) = K_LO + (K_HI - K_LO) * sigmoid(alpha * vz)
sigmoid(x) = 1 / (1 + exp(-x))
```
`lt_bias = (-k × lt_sig × 0.5).clip(-0.5, 0.5)`

**alpha → ∞ の極限**: sigmoid が step 関数に収束 → E4 離散版（vz_thr=0 相当）に近似
**alpha → 0 の極限**: k が一定 = (K_LO + K_HI) / 2 = {(K_LO + K_HI) / 2:.2f}（レジーム無効化）

**サニティ**: REF(α={ALPHA_REF:.0f}) CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (E4比 {diff_ref:+.2f}pp) → {"OK" if sanity_ok else "WARN"}

---

## §2 9指標テーブル（Sharpe 降順 + REF）

{hdr1}
{hdr2}
{ref_row}
{rows_md}

{MD_WFA_NOTE}

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ E4_REF+0.020 (≥{REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}) |
| (ii) CAGR_OOS | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap | ≤ {PASS_GAP*100:.1f}pp |
| (iv) MaxDD | > {PASS_MAXDD*100:.1f}% |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% |

- **PASS configs**: {len(pass_list)} / {total}
- **最高 Sharpe**: α={best_sharpe["alpha"]:.1f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}, CAGR_OOS={best_sharpe["CAGR_OOS"]*100:+.2f}%
- **E4 REF (α={ALPHA_REF:.0f})**: Sharpe={m_ref["Sharpe_OOS"]:+.3f}, CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%
- **総合判定: {verdict}**

---

## §4 考察

sigmoid 連続化により境界でのホイップソー損失を排除:
- alpha が小さい（2〜3）: 緩やかな連続遷移 → 過剰反応を抑制するが、regime 信号の強度も低下
- alpha が大きい（8〜100）: ほぼ離散判定に近づく → E4 との差異が縮小
- IS-OOS gap の変化に注目: 連続化でギャップ縮小なら「境界ホイップソーが gap 原因」と示唆
- PASS なら WFA (G3) 実施を推奨

---

*生成スクリプト: `src/a1_soft_regime_klt.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/e4_regime_klt.py`*
"""
    md_path = os.path.join(BASE, 'A1_SOFT_REGIME_KLT_2026-05-27.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "a1_soft_regime_klt_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
