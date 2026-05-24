"""
E3: VZ Gate_min パラメータスイープ
====================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

S2_VZGated の VZ ゲート下限 gate_min を [0.2, 0.3, 0.4, 0.5(REF), 0.6, 0.7, 0.8] でスイープ。
目的: MaxDD / Sharpe トレードオフを探索し、REF 超過の設定を発見する。

出力:
  - e3_vzgate_results.csv
  - E3_VZGATE_SWEEP_2026-05-24.md
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
from long_cycle_signal import build_lt_signal, signal_to_bias, apply_lt_mode_b
from _sweep_format import MD_HEADER_1P, fmt_row_1p, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# グリッド & 定数
# ---------------------------------------------------------------------------
GATE_MIN_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
REF_GATE_MIN  = 0.5

S2_BASE = dict(target_vol=0.8, k_vz=0.3, n=20, l_min=1.0, l_max=7.0, step=0.5)

REF_CAGR_OOS   = 0.3116
REF_SHARPE_OOS = 0.858
REF_MAXDD      = -0.5945

PASS_SHARPE_DELTA = 0.020
PASS_CAGR_OOS     = 0.2916
PASS_GAP          = 0.030
PASS_MAXDD        = -0.6445
PASS_WORST10Y     = 0.150


def compute_p10_5y(nav, td=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(td * 5)) ** 0.2 - 1
    return float(r.dropna().quantile(0.10))


def calc_all_metrics(nav, dates, trades_yr):
    m = calc_7metrics(nav, dates, trades_per_year=trades_yr)
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, 10)
    worst10y = float(r10.min()) if len(r10) > 0 else float('nan')
    return {**m,
            'Worst10Y_star': worst10y,
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
    print('E3: VZ gate_min スイープ')
    print('=' * 70)

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
    print('Shared assets done.')

    raw_a2, vz = build_a2_signal(close, ret)
    _, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_trades_yr = n_tr / n_years

    lt_sig  = build_lt_signal(close, 'LT2', 750)
    lt_bias = signal_to_bias(lt_sig, 0.5)
    print('Signals built.')

    results = []
    for gm in GATE_MIN_GRID:
        is_ref = abs(gm - REF_GATE_MIN) < 1e-9
        L_s2 = compute_L_s2_vz_gated(ret, vz, gate_min=gm, **S2_BASE)
        lev_raw, wn, wg, wb, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
        lev_mod = apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)
        nav = build_nav_strategy(
            close, lev_mod, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, n_trades_yr)
        m['gate_min'] = gm
        m['Trades_yr'] = n_trades_yr
        m['WFA_CI95_lo'] = float('nan')
        m['WFA_WFE']     = float('nan')
        results.append(m)
        ref_tag = ' [REF]' if is_ref else ''
        print(f'  gate_min={gm:.1f}{ref_tag}: CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
              f'Sharpe={m["Sharpe_OOS"]:+.3f}  MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
              f'Worst10Y★={m["Worst10Y_star"]*100:+.2f}%  IS-OOS={m["IS_OOS_gap"]*100:+.2f}pp')

    # Sanity
    r_ref = next(r for r in results if abs(r['gate_min'] - REF_GATE_MIN) < 1e-9)
    diff = (r_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    sanity_ok = abs(diff) <= 0.20
    print(f'\n[SANITY] REF CAGR_OOS={r_ref["CAGR_OOS"]*100:+.2f}% (ref +31.16%, diff {diff:+.2f}pp) '
          f'→ {"OK" if sanity_ok else "WARN"}')

    pass_list = [r for r in results
                 if passes_all(r) and abs(r['gate_min'] - REF_GATE_MIN) > 1e-9]
    best = max(results, key=lambda r: r['Sharpe_OOS'])
    verdict = ('PASS' if pass_list else
               'WARN' if best['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')
    print(f'Best gate_min={best["gate_min"]:.1f}  Sharpe={best["Sharpe_OOS"]:+.3f}')
    print(f'PASS configs: {len(pass_list)}  →  総合判定: {verdict}')

    # CSV
    df_out = pd.DataFrame([{
        'gate_min': r['gate_min'], 'CAGR_IS': r['CAGR_IS'],
        'CAGR_OOS': r['CAGR_OOS'], 'Sharpe_OOS': r['Sharpe_OOS'],
        'MaxDD_FULL': r['MaxDD_FULL'], 'Worst10Y_star': r['Worst10Y_star'],
        'P10_5Y': r['P10_5Y'], 'IS_OOS_gap': r['IS_OOS_gap'],
        'Trades_yr': r['Trades_yr'],
    } for r in results])
    df_out.to_csv(os.path.join(BASE, 'e3_vzgate_results.csv'), index=False, float_format='%.6f')

    # MD レポート
    hdr1, hdr2 = MD_HEADER_1P
    rows_md = '\n'.join(
        fmt_row_1p(f'{r["gate_min"]:.1f}{"  ← REF" if abs(r["gate_min"]-REF_GATE_MIN)<1e-9 else ""}', r)
        for r in results
    )
    report = f"""\
# E3: VZ gate_min スイープ

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

| 項目 | 定義 |
|------|------|
| **sweep パラメータ** | `gate_min` ∈ {GATE_MIN_GRID}（REF=0.5） |
| **S2 固定パラメータ** | target_vol=0.8, k_vz=0.3, n=20, l_min=1.0, l_max=7.0 |
| **LT2 固定** | N=750, k=0.5, modeB |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |
| **目的** | VZ ゲート強度調整による MaxDD/Sharpe 改善探索 |

**ゲート動作**: `vz_gate = clip(1 - 0.3 × max(vz, 0), gate_min, 1.0)`
- `gate_min` 大 → ゲートの効き弱（高ボラ時も高レバ維持）→ リターン高め・MaxDD 深め
- `gate_min` 小 → ゲートの効き強（高ボラ時にレバ大幅削減）→ MaxDD 浅め・CAGR 犠牲

**サニティ**: REF (gate_min=0.5) CAGR_OOS={r_ref["CAGR_OOS"]*100:+.2f}% (ref +31.16%, diff {diff:+.2f}pp) → {"OK" if sanity_ok else "WARN"}

---

## §2 9指標テーブル（gate_min 7通り）

{hdr1}
{hdr2}
{rows_md}

{MD_WFA_NOTE}

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ REF+0.020 (≥{REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}) |
| (ii) CAGR_OOS | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap | ≤ {PASS_GAP*100:.1f}pp |
| (iv) MaxDD | > {PASS_MAXDD*100:.1f}% |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% |

- **PASS configs**: {len(pass_list)} / {len(GATE_MIN_GRID)-1}（REF除く）
- **最高 Sharpe**: gate_min={best["gate_min"]:.1f} → Sharpe={best["Sharpe_OOS"]:+.3f}, CAGR_OOS={best["CAGR_OOS"]*100:+.2f}%, MaxDD={best["MaxDD_FULL"]*100:+.2f}%
- **総合判定: {verdict}**

---

## §4 考察

gate_min を {REF_GATE_MIN} から変化させた際の影響:
- **gate_min < {REF_GATE_MIN}**: 高ボラ時のレバ抑制が強くなり MaxDD が改善する可能性。ただし CAGR 低下のリスク。
- **gate_min > {REF_GATE_MIN}**: ゲートが緩まり高ボラ時もレバを維持。CAGR 向上も MaxDD 深化の可能性。
- REF の gate_min=0.5 が既に最適付近であれば、変更による改善余地は限定的。

---

*生成スクリプト: `src/e3_vzgate_sweep.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`*
"""
    md_path = os.path.join(BASE, 'E3_VZGATE_SWEEP_2026-05-24.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "e3_vzgate_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
