"""Option C2: NASDAQ cap 実効化 + Gold/Bond防御強化スイープ (2026-05-18)
========================================================================
C sweep で wn_max ∈ {0.80..0.95} が dead axis と判明 (wn_A は実データで 0.80 未満)。
wn_max を 0.40–0.70 に下げて cap を実際に効かせ、Gold/Bond 防御を厚くして
Worst5Y ≥ -3% を達成する組合せを探索する。

採用基準:
- CAGR_IS/OOS ≥ 25%
- Sharpe_IS/OOS ≥ 0.70
- Worst5Y ≥ -3.0%

グリッド:
- wn_max ∈ {0.40, 0.50, 0.60, 0.70}      (4)  ← cap が実際に binding になる範囲
- wg_frac ∈ {0.30, 0.40, 0.50, 0.60, 0.80, 1.00}  (6)
- bond_drag ∈ {False, True}              (2)
- gold: TOCOM 3x のみ (CFD 5x は W5Y を悪化させることがCで確認済み)
- 合計: 4 × 6 × 2 = 48 コンボ
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
    TRADING_DAYS, THRESHOLD, SWAP_SPREAD,
)
from test_portfolio_diversification import prepare_gold_data
from cfd_leverage_backtest import (
    build_nav_strategy, calc_7metrics, CFD_SPREAD_LOW,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from sleeves_extended import build_gold_tocom, build_bond_3x_with_drag

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')
OUT_MD    = os.path.join(BASE, 'C2_NASDAQ_CAP_SWEEP_2026-05-18.md')
OUT_CSV   = os.path.join(BASE, 'C2_NASDAQ_CAP_SWEEP_2026-05-18.csv')

WN_MAX_GRID  = [0.40, 0.50, 0.60, 0.70]
WG_FRAC_GRID = [0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
BOND_DRAG    = [False, True]

S2_PARAMS = dict(target_vol=0.80, k_vz=0.30, gate_min=0.50)

CAGR_IS_MIN    = 0.25
CAGR_OOS_MIN   = 0.25
SHARPE_IS_MIN  = 0.70
SHARPE_OOS_MIN = 0.70
WORST5Y_MIN    = -0.03


def main():
    print('=' * 70)
    print('Option C2: NASDAQ cap 実効化スイープ (48コンボ)')
    print('=' * 70)

    df      = load_data(DATA_PATH)
    close   = df['Close']
    dates   = df['Date']
    returns = close.pct_change().fillna(0)
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df):,} days)')

    print('Loading SOFR...')
    sofr = load_sofr(dates)

    print('Building bond sleeves...')
    bond_1x          = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x_nodrag   = build_bond_3x(bond_1x, sofr, True)
    bond_3x_drag_arr = build_bond_3x_with_drag(np.asarray(bond_1x, dtype=float), sofr, SWAP_SPREAD)
    bond_navs = {False: bond_3x_nodrag, True: bond_3x_drag_arr}

    print('Building gold sleeve (TOCOM 3x)...')
    gold_1x     = prepare_gold_data(dates)
    gold_1x_arr = np.asarray(gold_1x, dtype=float)
    gold_nav    = build_gold_tocom(gold_1x_arr, 3.0, sofr, roll_cost_annual=0.02)

    print('Building DH Dyn [A] signal + S2 leverage...')
    raw_a2, vz, _ = build_a2_signal(close, returns, return_components=True)
    lev_A, wn_A, _, _, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    wn_A_arr = np.asarray(wn_A, dtype=float)

    wn_A_max = wn_A_arr.max()
    print(f'  wn_A max observed: {wn_A_max:.4f}')

    L_s2     = compute_L_s2_vz_gated(returns, vz, **S2_PARAMS)
    L_s2_arr = np.asarray(L_s2)

    records = []
    total = len(WN_MAX_GRID) * len(WG_FRAC_GRID) * len(BOND_DRAG)
    done = 0

    print(f'\n合計 {total} コンボ実行:')
    print('-' * 70)

    for wn_max in WN_MAX_GRID:
        for wg_frac in WG_FRAC_GRID:
            for bond_drag in BOND_DRAG:
                done += 1
                drag_label = 'drag' if bond_drag else 'nodr'
                print(f'  [{done:2d}/{total}] wn_max={wn_max:.2f} wg_frac={wg_frac:.2f} {drag_label}', end=' ')

                wn = np.clip(wn_A_arr, 0.0, wn_max)
                bind_frac = (wn_A_arr >= wn_max).mean()

                rest = 1.0 - wn
                wg = rest * wg_frac
                wb = rest * (1.0 - wg_frac)

                nav = build_nav_strategy(
                    close, lev_A, wn, wg, wb, dates,
                    gold_nav, bond_navs[bond_drag],
                    sofr, 'CFD', L_s2_arr, CFD_SPREAD_LOW,
                )
                m = calc_7metrics(nav, dates)

                cagr_is  = m.get('CAGR_IS',  float('nan'))
                cagr_oos = m.get('CAGR_OOS', float('nan'))
                gap = abs(cagr_is - cagr_oos) * 100 if not (np.isnan(cagr_is) or np.isnan(cagr_oos)) else float('nan')

                rec = {
                    'wn_max':       wn_max,
                    'wg_frac':      wg_frac,
                    'bond_drag':    bond_drag,
                    'wn_bind_frac': bind_frac,
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
                print(f'CAGR_OOS={cagr_oos*100:+.1f}% Sh_OOS={rec["Sharpe_OOS"]:.3f} '
                      f'W5Y={rec["Worst5Y"]*100:+.2f}% bind={bind_frac:.2f} {status}')
                records.append(rec)

    df_results = pd.DataFrame(records)
    df_sorted  = df_results.sort_values('Sharpe_OOS', ascending=False).reset_index(drop=True)
    passing    = df_sorted[df_sorted.pass_all].copy()
    core_pass  = df_sorted[df_sorted.pass_core].copy()

    print('\n' + '=' * 70)
    print(f'採用基準パス (全5条件): {len(passing)} / {len(df_sorted)}')
    print(f'コア基準パス (Sh+W5Y):  {len(core_pass)} / {len(df_sorted)}')

    if len(passing) > 0:
        print('\n=== ✅ 全基準パス コンボ ===')
        for _, r in passing.iterrows():
            drag = 'drag' if r['bond_drag'] else 'nodr'
            print(f"  wn_max={r['wn_max']:.2f} wg={r['wg_frac']:.2f} {drag}: "
                  f"CAGR_IS={r['CAGR_IS']*100:+.2f}% CAGR_OOS={r['CAGR_OOS']*100:+.2f}% "
                  f"Sh_IS={r['Sharpe_IS']:.3f} Sh_OOS={r['Sharpe_OOS']:.3f} "
                  f"W5Y={r['Worst5Y']*100:+.2f}%")
    else:
        print('\n=== ❌ 全基準パスなし — W5Y ベスト Top5 ===')
        top5 = df_sorted.nlargest(5, 'Worst5Y')
        for _, r in top5.iterrows():
            drag = 'drag' if r['bond_drag'] else 'nodr'
            print(f"  wn_max={r['wn_max']:.2f} wg={r['wg_frac']:.2f} {drag}: "
                  f"CAGR_OOS={r['CAGR_OOS']*100:+.2f}% Sh_OOS={r['Sharpe_OOS']:.3f} "
                  f"W5Y={r['Worst5Y']*100:+.2f}%")

    df_sorted.to_csv(OUT_CSV, index=False, float_format='%.6f')
    print(f'\nCSV: {OUT_CSV}')
    print('Done.')


if __name__ == '__main__':
    main()
