"""Option C: NASDAQ高比率 + 軽量Gold/Bond防御 スイープ (2026-05-17)
========================================================================
S2 CFD を主体に、wn上限capしてGold/Bond を防御として組み込み、
CAGR≥25% AND Worst5Y≥-3% を両立する組合せを探索。

採用基準:
- CAGR_IS/OOS ≥ 25%
- Sharpe_IS/OOS ≥ 0.70
- Worst5Y ≥ -3.0%

グリッド:
- wn_max ∈ {0.80, 0.85, 0.90, 0.95}    (4)
- wg_frac ∈ {0.40, 0.60, 0.80, 1.00}   (4)
- gold_kind ∈ {tocom_3x, cfd_5x}        (2)
- 合計: 32コンボ
"""

import sys, os, types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr, build_bond_1x_nav_corrected, build_bond_3x,
    build_a2_signal, simulate_rebalance_A,
    TRADING_DAYS, THRESHOLD,
)
from test_portfolio_diversification import prepare_gold_data
from cfd_leverage_backtest import (
    build_nav_strategy, calc_7metrics, CFD_SPREAD_LOW,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from sleeves_extended import build_gold_tocom, build_gold_cfd

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')
OUT_MD    = os.path.join(BASE, 'C_NASDAQ_HEAVY_SWEEP_2026-05-17.md')
OUT_CSV   = os.path.join(BASE, 'C_NASDAQ_HEAVY_SWEEP_2026-05-17.csv')

WN_MAX_GRID   = [0.80, 0.85, 0.90, 0.95]
WG_FRAC_GRID  = [0.40, 0.60, 0.80, 1.00]
GOLD_KINDS    = ['tocom_3x', 'cfd_5x']

S2_PARAMS = dict(target_vol=0.80, k_vz=0.30, gate_min=0.50)

CAGR_IS_MIN    = 0.25
CAGR_OOS_MIN   = 0.25
SHARPE_IS_MIN  = 0.70
SHARPE_OOS_MIN = 0.70
WORST5Y_MIN    = -0.03
GAP_MAX        = 8.0


def _fp(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v*100:+.{d}f}%'


def _ff(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:.{d}f}'


def main():
    print('=' * 70)
    print('Option C: NASDAQ高比率 + 軽量Gold/Bond防御 スイープ (32コンボ)')
    print('=' * 70)

    df      = load_data(DATA_PATH)
    close   = df['Close']
    dates   = df['Date']
    returns = close.pct_change().fillna(0)
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df):,} days)')

    print('Loading SOFR...')
    sofr = load_sofr(dates)

    print('Building bond sleeves...')
    bond_1x        = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x_nodrag = build_bond_3x(bond_1x, sofr, True)

    print('Building gold sleeves...')
    gold_1x     = prepare_gold_data(dates)
    gold_1x_arr = np.asarray(gold_1x, dtype=float)
    gold_navs = {
        'tocom_3x': build_gold_tocom(gold_1x_arr, 3.0, sofr, roll_cost_annual=0.02),
        'cfd_5x':   build_gold_cfd(gold_1x_arr, 5.0, sofr, spread_annual=0.012),
    }

    print('Building DH Dyn [A] signal + S2 leverage...')
    raw_a2, vz, _ = build_a2_signal(close, returns, return_components=True)
    lev_A, wn_A, _, _, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    L_s2 = compute_L_s2_vz_gated(returns, vz, **S2_PARAMS)
    L_s2_arr = np.asarray(L_s2)

    records = []
    total = len(WN_MAX_GRID) * len(WG_FRAC_GRID) * len(GOLD_KINDS)
    done = 0

    print(f'\n合計 {total} コンボ実行:')
    print('-' * 70)

    for wn_max in WN_MAX_GRID:
        for wg_frac in WG_FRAC_GRID:
            for gold_kind in GOLD_KINDS:
                done += 1
                print(f'  [{done:2d}/{total}] wn_max={wn_max:.2f} wg_frac={wg_frac:.2f} {gold_kind:<10}', end=' ')

                wn = np.clip(np.asarray(wn_A, dtype=float), 0.0, wn_max)
                rest = 1.0 - wn
                wg = rest * wg_frac
                wb = rest * (1.0 - wg_frac)

                nav = build_nav_strategy(
                    close, lev_A, wn, wg, wb, dates,
                    gold_navs[gold_kind], bond_3x_nodrag,
                    sofr, 'CFD', L_s2_arr, CFD_SPREAD_LOW,
                )
                m = calc_7metrics(nav, dates)

                cagr_is  = m.get('CAGR_IS',  float('nan'))
                cagr_oos = m.get('CAGR_OOS', float('nan'))
                gap = abs(cagr_is - cagr_oos) * 100 if not (np.isnan(cagr_is) or np.isnan(cagr_oos)) else float('nan')

                rec = {
                    'wn_max':       wn_max,
                    'wg_frac':      wg_frac,
                    'gold_kind':    gold_kind,
                    'CAGR_FULL':    m.get('CAGR_FULL',   float('nan')),
                    'CAGR_IS':      cagr_is,
                    'CAGR_OOS':     cagr_oos,
                    'Sharpe_FULL':  m.get('Sharpe_FULL', float('nan')),
                    'Sharpe_IS':    m.get('Sharpe_IS',   float('nan')),
                    'Sharpe_OOS':   m.get('Sharpe_OOS',  float('nan')),
                    'MaxDD_FULL':   m.get('MaxDD_FULL',  float('nan')),
                    'Worst5Y':      m.get('Worst5Y',     float('nan')),
                    'IS_OOS_Gap':   gap,
                }

                p_cagr_is  = rec['CAGR_IS']    >= CAGR_IS_MIN
                p_cagr_oos = rec['CAGR_OOS']   >= CAGR_OOS_MIN
                p_sh_is    = rec['Sharpe_IS']  >= SHARPE_IS_MIN
                p_sh_oos   = rec['Sharpe_OOS'] >= SHARPE_OOS_MIN
                p_w5       = rec['Worst5Y']    >= WORST5Y_MIN
                rec['pass_all']  = all([p_cagr_is, p_cagr_oos, p_sh_is, p_sh_oos, p_w5])
                rec['pass_core'] = p_sh_oos and p_w5

                status = '✅' if rec['pass_all'] else ('⚡' if rec['pass_core'] else '❌')
                print(f'CAGR_OOS={cagr_oos*100:+.1f}% Sh_OOS={rec["Sharpe_OOS"]:.3f} W5Y={rec["Worst5Y"]*100:+.2f}% {status}')
                records.append(rec)

    df_results = pd.DataFrame(records)
    df_sorted  = df_results.sort_values('Sharpe_OOS', ascending=False).reset_index(drop=True)
    passing    = df_sorted[df_sorted.pass_all].copy()

    print('\n' + '=' * 70)
    print(f'採用基準パス (全5条件): {len(passing)} / {len(df_sorted)}')

    df_sorted.to_csv(OUT_CSV, index=False, float_format='%.6f')
    print(f'CSV: {OUT_CSV}')
    print('Done.')


if __name__ == '__main__':
    main()
