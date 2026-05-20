"""72-config sweep of long-cycle signal extensions to DH Dyn 2x3x [A] Scenario D.

Signals: LT1 (valuation/MA ratio), LT2 (momentum z-score), LT3 (percentile rank)
Lookbacks: N in {750, 1000, 1250} trading days (~3/4/5 years)
Sensitivities: k_lt in {0.1, 0.2, 0.3, 0.5}
Modes: A (modify raw_a2), B (bias post-rebalance lev)
Total: 3 x 3 x 4 x 2 = 72 configs

Baseline: DH Dyn 2x3x [A] Scenario D — CAGR_FULL=22.50%, Sharpe_OOS=0.993, MaxDD=-45.08%
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

import itertools
import numpy as np
import pandas as pd

from corrected_strategy_backtest import (
    load_data, load_sofr,
    build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_a2_signal, simulate_rebalance_A, build_nav,
    DATA_PATH, DATA_DIR, TRADING_DAYS, THRESHOLD,
)
from compute_dha_worst10y_only import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import (
    build_lt_signal, signal_to_mult, signal_to_bias,
    apply_lt_mode_a, apply_lt_mode_b,
)

FULL_START = '1974-01-02'
IS_END     = '2021-05-07'
OOS_START  = '2021-05-08'

BASELINE = dict(
    CAGR_IS=0.2336, CAGR_OOS=0.1488, CAGR_FULL=0.2250,
    Sharpe_OOS=0.6460, MaxDD_FULL=-0.4508,
    W5Y_FULL=0.0087, W10Y_STAR=0.1430,
)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def calc_period_metrics(nav: pd.Series, dates: pd.Series, start: str, end: str) -> dict:
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    idx  = dates[mask].index
    if len(idx) < 100:
        return dict(CAGR=np.nan, Sharpe=np.nan, MaxDD=np.nan, W5Y=np.nan)
    ns  = nav.loc[idx[0]:idx[-1]]
    ns  = ns / ns.iloc[0]
    r   = ns.pct_change().fillna(0)
    yrs = len(ns) / TRADING_DAYS
    cagr = float(ns.iloc[-1]) ** (1 / yrs) - 1
    sh   = (r.mean() * TRADING_DAYS) / (r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan
    mdd  = float((ns / ns.cummax() - 1).min())
    if len(ns) >= TRADING_DAYS * 5:
        w5 = float(((ns / ns.shift(TRADING_DAYS * 5)) ** 0.2 - 1).min())
    else:
        w5 = np.nan
    return dict(CAGR=cagr, Sharpe=sh, MaxDD=mdd, W5Y=w5)


def calc_worst10y_star(nav: pd.Series, dates: pd.Series) -> float:
    ann = nav_to_annual(nav, dates)
    if len(ann) < 10:
        return np.nan
    r10 = rolling_nY_cagr(ann, n=10)
    return float(r10.min())


# ---------------------------------------------------------------------------
# Single config runner
# ---------------------------------------------------------------------------

def run_config(close, ret, dates, sofr, gold_2x, bond_3x,
               signal_name: str, N: int, k_lt: float, mode: str) -> dict:
    raw_a2, vz = build_a2_signal(close, ret)
    lt_sig     = build_lt_signal(close, signal_name, N)

    if mode == 'A':
        lt_mult  = signal_to_mult(lt_sig, k_lt)
        raw_mod  = apply_lt_mode_a(raw_a2, lt_mult)
        lev, wn, wg, wb, n_tr = simulate_rebalance_A(raw_mod, vz, THRESHOLD)
    else:  # Mode B
        lev_base, wn, wg, wb, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
        lt_bias = signal_to_bias(lt_sig, k_lt)
        lev     = apply_lt_mode_b(lev_base, lt_bias, l_min=0.0, l_max=1.0)

    nav = build_nav(close, lev, wn, wg, wb, dates,
                    gold_2x, bond_3x, sofr_daily=sofr, apply_tqqq_sofr=True)

    m_full = calc_period_metrics(nav, dates, FULL_START, str(dates.iloc[-1].date()))
    m_is   = calc_period_metrics(nav, dates, FULL_START, IS_END)
    m_oos  = calc_period_metrics(nav, dates, OOS_START,  str(dates.iloc[-1].date()))
    w10s   = calc_worst10y_star(nav, dates)

    return dict(
        config     = f'{signal_name}-N{N}-k{k_lt}-mode{mode}',
        signal     = signal_name,
        N          = N,
        k_lt       = k_lt,
        mode       = mode,
        n_trades   = n_tr,
        CAGR_IS    = m_is['CAGR'],
        CAGR_OOS   = m_oos['CAGR'],
        CAGR_FULL  = m_full['CAGR'],
        Sharpe_OOS = m_oos['Sharpe'],
        MaxDD_FULL = m_full['MaxDD'],
        W5Y_FULL   = m_full['W5Y'],
        W10Y_STAR  = w10s,
        IS_OOS_GAP = m_is['CAGR'] - m_oos['CAGR'] if not np.isnan(m_is['CAGR']) else np.nan,
    )


# ---------------------------------------------------------------------------
# Acceptance filter (Opus §6 criteria)
# ---------------------------------------------------------------------------

def is_accepted(row: dict) -> bool:
    """Return True if config passes all overfitting/quality gates.

    n_trades is TOTAL count over the full period (~52.26 years).
    Baseline: 1417 total = 27.1/yr. Gate: 10-60/yr → 522-3136 total.
    """
    b = BASELINE
    YEARS = 52.26
    ok = (
        row['CAGR_OOS']    >= b['CAGR_OOS'] + 0.005                      # C1: +0.5pp OOS
        and abs(row['IS_OOS_GAP']) <= 0.10                                 # C2: IS-OOS gap ≤ 10pp
        and row['Sharpe_OOS']  >= b['Sharpe_OOS'] - 0.02                  # C3: Sharpe ≥ baseline
        and row['MaxDD_FULL']  >= -0.55                                    # C4: MaxDD not worse than -55%
        and (np.isnan(row['W5Y_FULL']) or row['W5Y_FULL'] >= 0.0)         # C5: no negative 5Y
        and (np.isnan(row['W10Y_STAR']) or row['W10Y_STAR'] >= b['W10Y_STAR'] - 0.02)  # C6
        and 10 * YEARS <= row['n_trades'] <= 60 * YEARS                   # C7: 10-60 trades/yr
    )
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 70)
    print('Long-Cycle Signal Sweep  (72 configs, DH Dyn 2x3x [A] Scenario D)')
    print('=' * 70)

    df      = load_data(DATA_PATH)
    close   = df['Close']
    ret     = close.pct_change().fillna(0)
    dates   = df['Date']
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(dates)} days)')

    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print('Assets ready. Starting sweep...\n')

    signals = ['LT1', 'LT2', 'LT3']
    Ns      = [750, 1000, 1250]
    ks      = [0.1, 0.2, 0.3, 0.5]
    modes   = ['A', 'B']

    rows = []
    total = len(signals) * len(Ns) * len(ks) * len(modes)
    for idx, (sig, N, k, mode) in enumerate(itertools.product(signals, Ns, ks, modes), 1):
        print(f'[{idx:2d}/{total}] {sig}-N{N}-k{k}-mode{mode}', end='... ', flush=True)
        row = run_config(close, ret, dates, sofr, gold_2x, bond_3x, sig, N, k, mode)
        rows.append(row)
        print(f"CAGR_OOS={row['CAGR_OOS']*100:+.2f}%  Sharpe={row['Sharpe_OOS']:.3f}  "
              f"MaxDD={row['MaxDD_FULL']*100:.1f}%  W10Y={row['W10Y_STAR']*100:+.2f}%"
              if not np.isnan(row['Sharpe_OOS']) else 'NaN')

    out_df  = pd.DataFrame(rows)
    out_dir = os.path.dirname(DATA_PATH)
    out_csv = os.path.join(out_dir, 'lt_sweep_results.csv')
    out_df.to_csv(out_csv, index=False, float_format='%.4f')
    print(f'\nSaved: {out_csv}')

    print('\n' + '=' * 70)
    print('BASELINE (DH Dyn 2x3x [A] Scenario D):')
    b = BASELINE
    print(f"  CAGR_IS={b['CAGR_IS']*100:.2f}%  CAGR_OOS={b['CAGR_OOS']*100:.2f}%  "
          f"CAGR_FULL={b['CAGR_FULL']*100:.2f}%  Sharpe_OOS={b['Sharpe_OOS']:.4f}  "
          f"MaxDD={b['MaxDD_FULL']*100:.2f}%  W5Y={b['W5Y_FULL']*100:.2f}%")

    print('\nTop 10 by CAGR_OOS:')
    top_oos = out_df.nlargest(10, 'CAGR_OOS')
    print(top_oos[['config', 'CAGR_IS', 'CAGR_OOS', 'CAGR_FULL',
                    'Sharpe_OOS', 'MaxDD_FULL', 'W5Y_FULL', 'W10Y_STAR', 'IS_OOS_GAP']].to_string(index=False))

    print('\nTop 10 by Sharpe_OOS:')
    top_sh = out_df.nlargest(10, 'Sharpe_OOS')
    print(top_sh[['config', 'CAGR_IS', 'CAGR_OOS', 'CAGR_FULL',
                   'Sharpe_OOS', 'MaxDD_FULL', 'W5Y_FULL', 'W10Y_STAR', 'IS_OOS_GAP']].to_string(index=False))

    accepted = out_df[out_df.apply(is_accepted, axis=1)]
    print(f'\n=== ACCEPTED (all gates passed): {len(accepted)} / {total} configs ===')
    if len(accepted) > 0:
        best = accepted.sort_values(
            ['Sharpe_OOS', 'W10Y_STAR', 'IS_OOS_GAP', 'CAGR_OOS'],
            ascending=[False, False, True, False]
        ).iloc[0]
        print('\nBEST candidate:')
        for k2, v in best.items():
            print(f'  {k2}: {v:.4f}' if isinstance(v, float) else f'  {k2}: {v}')
    else:
        print('No config passed all gates. Long-cycle signal does not improve DH Dyn [A] '
              'under the specified acceptance criteria.')

    print('\n' + '=' * 70)
    return out_df


if __name__ == '__main__':
    main()
