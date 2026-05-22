"""
B5: S2_VZGated + LT7 Sweep (k_lt のみ; N は LT7 内部固定)
===========================================================
# Evaluation Standard: v1.1
# Cost Scenario: D
# IS: 1974-01-02〜2021-05-07 / OOS: 2021-05-08〜現在

目的: LT7 信号（N_short=750, N_long=1250 内部固定）を S2_VZGated の
     lev シグナルに modeB で適用し、k_lt の最適値を探索する。

グリッド: k_lt ∈ {0.3, 0.5, 0.7} = 3 runs
  LT7 は N パラメータを無視（内部で N_short=750, N_long=1250 を使用）。

比較ベースライン:
  S2_VZGated 単体: CAGR_OOS +27.51%, Sharpe_OOS +0.770
  S2+LT2-N750-k0.5: CAGR_OOS +31.16%, Sharpe_OOS +0.858 (B1 実績)

出力:
  - b5_s2_lt7_sweep_results.csv  (プロジェクトルート)
  - B5_S2_LT7_SWEEP_2026-05-22.md (プロジェクトルート)
"""

import sys
import os
import types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr,
    build_bond_1x_nav_corrected,
    build_gold_2x,
    build_bond_3x,
    build_a2_signal,
    simulate_rebalance_A,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy,
    calc_7metrics,
    CFD_SPREAD_LOW,
    IS_START, IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import (
    prepare_gold_local,
    nav_to_annual,
    rolling_nY_cagr,
)
from long_cycle_signal import build_lt_signal, signal_to_bias, apply_lt_mode_b
from _sweep_format import MD_HEADER_1P, fmt_row_1p, MD_WFA_NOTE

# ---------------------------------------------------------------------------
REF_S2_SHARPE    = 0.770
REF_LT2_SHARPE   = 0.858

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KLT_SWEEP = [0.3, 0.5, 0.7]

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)
LT_SIGNAL = 'LT7'
LT7_N_DUMMY = 750  # LT7 内部固定 N_short=750 に揃えるが、実際は無視される


def compute_p10_5y(nav, trading_days=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(trading_days * 5)) ** 0.2 - 1
    return float(r.dropna().quantile(0.10))


def compute_worst5y(nav, trading_days=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(trading_days * 5)) ** 0.2 - 1
    return float(r.dropna().min())


def calc_all_metrics(nav, dates, trades_per_year):
    m = calc_7metrics(nav, dates, trades_per_year=trades_per_year)
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, 10)
    worst10y_star = float(r10.min()) if len(r10) > 0 else np.nan
    return {
        **m,
        'Worst10Y_star': worst10y_star,
        'P10_5Y':        compute_p10_5y(nav.values),
        'Worst5Y':       compute_worst5y(nav.values),
        'IS_OOS_gap':    m['CAGR_IS'] - m['CAGR_OOS'],
    }


def _fp(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v * 100:+.{d}f}%'


def _ff(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:+.{d}f}'


def generate_report(results):
    beat_s2  = sum(1 for r in results if r['Sharpe_OOS'] > REF_S2_SHARPE)
    beat_lt2 = sum(1 for r in results if r['Sharpe_OOS'] > REF_LT2_SHARPE)
    best = max(results, key=lambda x: x['Sharpe_OOS'])
    lines = [
        f'# B5: S2_VZGated + {LT_SIGNAL} Sweep (k_lt)',
        '',
        '作成日: 2026-05-22',
        '最終更新日: 2026-05-22',
        '',
        f'**初回 Scenario D 正式評価**。{LT_SIGNAL}-modeB を S2_VZGated lev に適用。',
        f'LT7 は N パラメータを無視（内部固定 N_short=750, N_long=1250）。',
        '',
        f'- **IS**: {IS_START} 〜 {IS_END} / **OOS**: {OOS_START} 〜',
        f'- **S2 単体 Sharpe_OOS**: +0.770',
        f'- **S2+LT2-N750-k0.5 Sharpe_OOS**: +0.858 (B1 実績)',
        '',
        '## 結果テーブル（k_lt 昇順）',
        '',
        MD_HEADER_1P[0],
        MD_HEADER_1P[1],
    ]
    for r in sorted(results, key=lambda x: x['k_lt']):
        lines.append(fmt_row_1p(f"k={r['k_lt']:.1f}", r, REF_S2_SHARPE, REF_LT2_SHARPE))
    lines += [
        '',
        '◎ = S2 単体 (Sharpe +0.770) を上回る  ★ = S2+LT2 (Sharpe +0.858) を上回る',
        '',
        MD_WFA_NOTE,
        '',
        '## サマリ',
        '',
        f'- S2 単体を上回る構成: **{beat_s2}/{len(results)}**',
        f'- S2+LT2 実績を上回る構成: **{beat_lt2}/{len(results)}**',
        f'- 最高 Sharpe_OOS: **{best["Sharpe_OOS"]:+.3f}** (k_lt={best["k_lt"]:.1f})',
        '',
        '## 再現コマンド',
        '',
        '```',
        f'python -X utf8 src/b5_s2_lt7_sweep.py',
        '```',
        '',
        '*参照: `B1_S2_LT2_2026-05-21.md`, `CURRENT_BEST_STRATEGY.md`*',
    ]
    return '\n'.join(lines)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print(f'B5: S2_VZGated + {LT_SIGNAL} Sweep — {len(KLT_SWEEP)} runs')
    print('実行日: 2026-05-22')
    print('=' * 70)

    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({n:,} days)')

    print('\nBuilding shared assets...')
    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    print('Building DH Dyn signal...')
    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_years = n / TRADING_DAYS
    print(f'  DH Dyn: {n_tr} trades, {n_tr / n_years:.1f}/yr')

    print('\nBuilding S2 CFD leverage series (fixed)...')
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)

    print(f'\nBuilding {LT_SIGNAL} signal (N param ignored internally)...')
    lt_sig = build_lt_signal(close, LT_SIGNAL, LT7_N_DUMMY)
    print(f'  LT7 signal built: [{lt_sig.min():.3f}, {lt_sig.max():.3f}]')

    print(f'\nSweeping k_lt ∈ {KLT_SWEEP} ...')
    results = []

    for k_lt in KLT_SWEEP:
        print(f'  k_lt={k_lt:.1f} ...', end=' ', flush=True)
        lt_bias = signal_to_bias(lt_sig, k_lt)
        lev_mod = apply_lt_mode_b(lev_A, lt_bias, l_min=0.0, l_max=1.0)

        nav = build_nav_strategy(
            close, lev_mod, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD',
            cfd_leverage=L_s2.values,
            cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, n_tr / n_years)
        row = {
            'k_lt':          k_lt,
            'CAGR_IS':       m['CAGR_IS'],
            'CAGR_OOS':      m['CAGR_OOS'],
            'Sharpe_OOS':    m['Sharpe_OOS'],
            'MaxDD_FULL':    m['MaxDD_FULL'],
            'Worst10Y_star': m['Worst10Y_star'],
            'P10_5Y':        m['P10_5Y'],
            'Worst5Y':       m['Worst5Y'],
            'IS_OOS_gap':    m['IS_OOS_gap'],
            'Trades_yr':     m['Trades_yr'],
            'WFA_CI95_lo':   np.nan,
            'WFA_WFE':       np.nan,
        }
        results.append(row)
        beat = '★' if m['Sharpe_OOS'] > REF_LT2_SHARPE else ('◎' if m['Sharpe_OOS'] > REF_S2_SHARPE else ' ')
        print(f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m["Sharpe_OOS"]:+.3f} {beat}')

    best = max(results, key=lambda x: x['Sharpe_OOS'])
    beat_s2  = sum(1 for r in results if r['Sharpe_OOS'] > REF_S2_SHARPE)
    beat_lt2 = sum(1 for r in results if r['Sharpe_OOS'] > REF_LT2_SHARPE)
    print()
    print('=' * 100)
    print(f'B5 {LT_SIGNAL} Results ({len(results)} runs)')
    print('=' * 100)
    for r in results:
        beat = '★' if r['Sharpe_OOS'] > REF_LT2_SHARPE else ('◎' if r['Sharpe_OOS'] > REF_S2_SHARPE else ' ')
        print(
            f'k_lt={r["k_lt"]:.1f}'
            f'  CAGR_OOS={r["CAGR_OOS"]*100:>+8.2f}%'
            f'  Sharpe_OOS={r["Sharpe_OOS"]:>+8.3f} {beat}'
            f'  MaxDD={r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  IS-OOS gap={r["IS_OOS_gap"]*100:>+8.2f} pp'
        )
    print('=' * 100)
    print(f'\nS2 単体: Sharpe_OOS +0.770  S2+LT2-N750-k0.5: +0.858')
    print(f'{LT_SIGNAL} beat S2: {beat_s2}/{len(results)}  beat LT2: {beat_lt2}/{len(results)}')
    print(f'Best: k_lt={best["k_lt"]:.1f} '
          f'→ CAGR_OOS {best["CAGR_OOS"]*100:+.2f}%, Sharpe_OOS {best["Sharpe_OOS"]:+.3f}')

    csv_path = os.path.join(BASE, 'b5_s2_lt7_sweep_results.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    md = generate_report(results)
    md_path = os.path.join(BASE, 'B5_S2_LT7_SWEEP_2026-05-22.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
