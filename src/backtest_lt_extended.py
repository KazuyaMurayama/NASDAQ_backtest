"""Extended LT signal sweep: 56 new configs on top of the initial 72-config sweep.

New elements vs. Phase-1:
  Signals  : LT4 (LT2+LT3 composite), LT6 (vol-adj momentum), LT7 (multi-TF blend)
  Lookbacks: N = 500, 600, 1500  (short-cycle + long-cycle additions)
  k_lt     : 0.7, 1.0  (stronger signal strength exploration)
  Mode C   : A+B blend (alpha=0.5 pilot, LT2 only)

Config breakdown (56 total):
  LT2-modeB extended    : 18  (N=500/600/1500 × k=0.3-1.0 + N=750/1000/1250 × k=0.7/1.0)
  LT4-modeB             : 12  (N=500/750/1000/1250 × k=0.5/0.7/1.0)
  LT6-modeB             : 16  (N=500/750/1000/1250 × k=0.3/0.5/0.7/1.0)
  LT7-modeB             :  4  (fixed N=750/1250 internally × k=0.3/0.5/0.7/1.0)
  LT2-modeC pilot       :  6  (N=750/1000 × k=0.5/0.7/1.0)

Baseline: DH Dyn 2x3x [A] Scenario D — CAGR_FULL=22.50%, Sharpe_OOS=0.6460, MaxDD=-45.08%
Phase-1 best: LT2-N750-k0.5-modeB — CAGR_OOS=18.87%, Sharpe_OOS=0.777, IS-OOS=3.25pp
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

from corrected_strategy_backtest import (
    load_data, load_sofr,
    build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_a2_signal, simulate_rebalance_A, build_nav,
    DATA_PATH, DATA_DIR, TRADING_DAYS, THRESHOLD,
)
from compute_dha_worst10y_only import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import (
    build_lt_signal, signal_to_mult, signal_to_bias,
    signal_to_mult_c, signal_to_bias_c,
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

PHASE1_BEST = dict(
    config='LT2-N750-k0.5-modeB',
    CAGR_OOS=0.1887, Sharpe_OOS=0.7766, MaxDD_FULL=-0.4476,
    IS_OOS_GAP=0.0325,
)

# Already tested in Phase-1 (skip to avoid duplication)
PHASE1_TESTED = set()
for sig in ['LT1', 'LT2', 'LT3']:
    for N in [750, 1000, 1250]:
        for k in [0.1, 0.2, 0.3, 0.5]:
            for mode in ['A', 'B']:
                PHASE1_TESTED.add(f'{sig}-N{N}-k{k:.1f}-mode{mode}')


# ---------------------------------------------------------------------------
# Build sweep configs (56 total)
# ---------------------------------------------------------------------------

def build_sweep_configs() -> list:
    configs = []

    # 1. LT2-modeB extended (N=500/600/1500 + new k for existing N)
    lt2_b_ext = (
        [(500, k) for k in [0.3, 0.5, 0.7, 1.0]] +
        [(600, k) for k in [0.3, 0.5, 0.7, 1.0]] +
        [(750, k) for k in [0.7, 1.0]] +
        [(1000, k) for k in [0.7, 1.0]] +
        [(1250, k) for k in [0.7, 1.0]] +
        [(1500, k) for k in [0.3, 0.5, 0.7, 1.0]]
    )
    for N, k in lt2_b_ext:
        configs.append(('LT2', N, k, 'B'))

    # 2. LT4-modeB: composite LT2+LT3 ensemble
    for N in [500, 750, 1000, 1250]:
        for k in [0.5, 0.7, 1.0]:
            configs.append(('LT4', N, k, 'B'))

    # 3. LT6-modeB: vol-adjusted momentum
    for N in [500, 750, 1000, 1250]:
        for k in [0.3, 0.5, 0.7, 1.0]:
            configs.append(('LT6', N, k, 'B'))

    # 4. LT7-modeB: multi-timeframe blend (N arg ignored internally)
    for k in [0.3, 0.5, 0.7, 1.0]:
        configs.append(('LT7', 0, k, 'B'))  # N=0 is placeholder

    # 5. LT2-modeC pilot (A+B blend)
    for N in [750, 1000]:
        for k in [0.5, 0.7, 1.0]:
            configs.append(('LT2', N, k, 'C'))

    return configs


# ---------------------------------------------------------------------------
# Metrics helpers (identical to Phase-1)
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

    # LT7 ignores N (fixed internally)
    lt_sig = build_lt_signal(close, signal_name, N if signal_name != 'LT7' else None)

    if mode == 'A':
        lt_mult = signal_to_mult(lt_sig, k_lt)
        raw_mod = apply_lt_mode_a(raw_a2, lt_mult)
        lev, wn, wg, wb, n_tr = simulate_rebalance_A(raw_mod, vz, THRESHOLD)
    elif mode == 'B':
        lev_base, wn, wg, wb, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
        lt_bias = signal_to_bias(lt_sig, k_lt)
        lev     = apply_lt_mode_b(lev_base, lt_bias, l_min=0.0, l_max=1.0)
    elif mode == 'C':
        # Mode C: apply half-strength A first, then half-strength B after rebalance
        lt_mult_c = signal_to_mult_c(lt_sig, k_lt, alpha=0.5)
        raw_mod   = apply_lt_mode_a(raw_a2, lt_mult_c)
        lev_base, wn, wg, wb, n_tr = simulate_rebalance_A(raw_mod, vz, THRESHOLD)
        lt_bias_c = signal_to_bias_c(lt_sig, k_lt, alpha=0.5)
        lev       = apply_lt_mode_b(lev_base, lt_bias_c, l_min=0.0, l_max=1.0)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    nav = build_nav(close, lev, wn, wg, wb, dates,
                    gold_2x, bond_3x, sofr_daily=sofr, apply_tqqq_sofr=True)

    m_full = calc_period_metrics(nav, dates, FULL_START, str(dates.iloc[-1].date()))
    m_is   = calc_period_metrics(nav, dates, FULL_START, IS_END)
    m_oos  = calc_period_metrics(nav, dates, OOS_START,  str(dates.iloc[-1].date()))
    w10s   = calc_worst10y_star(nav, dates)

    # Config label: LT7 shows fixed Ns
    if signal_name == 'LT7':
        config_label = f'LT7-N750x1250-k{k_lt}-mode{mode}'
        N_disp = 'fixed'
    else:
        config_label = f'{signal_name}-N{N}-k{k_lt}-mode{mode}'
        N_disp = N

    return dict(
        config     = config_label,
        signal     = signal_name,
        N          = N_disp,
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
# Acceptance filter (same gates as Phase-1)
# ---------------------------------------------------------------------------

def is_accepted(row: dict) -> bool:
    b = BASELINE
    YEARS = 52.26
    ok = (
        row['CAGR_OOS']    >= b['CAGR_OOS'] + 0.005
        and abs(row['IS_OOS_GAP']) <= 0.10
        and row['Sharpe_OOS']  >= b['Sharpe_OOS'] - 0.02
        and row['MaxDD_FULL']  >= -0.55
        and (np.isnan(row['W5Y_FULL']) or row['W5Y_FULL'] >= 0.0)
        and (np.isnan(row['W10Y_STAR']) or row['W10Y_STAR'] >= b['W10Y_STAR'] - 0.02)
        and 10 * YEARS <= row['n_trades'] <= 60 * YEARS
    )
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 72)
    print('LT Extended Sweep  (56 new configs: LT4/LT6/LT7 + N500/600/1500 + k0.7/1.0 + ModeC)')
    print('=' * 72)

    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(dates)} days)')

    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print('Assets ready. Starting sweep...\n')

    configs = build_sweep_configs()
    print(f'Total new configs: {len(configs)}')
    b = BASELINE

    rows = []
    for idx, (sig, N, k, mode) in enumerate(configs, 1):
        label = f'LT7-N750x1250-k{k}-mode{mode}' if sig == 'LT7' else f'{sig}-N{N}-k{k}-mode{mode}'
        print(f'[{idx:2d}/{len(configs)}] {label}', end='... ', flush=True)
        try:
            row = run_config(close, ret, dates, sofr, gold_2x, bond_3x, sig, N, k, mode)
            rows.append(row)
            if not np.isnan(row['Sharpe_OOS']):
                print(f"CAGR_OOS={row['CAGR_OOS']*100:+.2f}%  "
                      f"Sharpe={row['Sharpe_OOS']:.3f}  "
                      f"MaxDD={row['MaxDD_FULL']*100:.1f}%  "
                      f"W10Y={row['W10Y_STAR']*100:+.2f}%  "
                      f"gap={row['IS_OOS_GAP']*100:+.2f}pp")
            else:
                print('NaN')
        except Exception as e:
            print(f'ERROR: {e}')

    out_df  = pd.DataFrame(rows)
    out_dir = os.path.dirname(DATA_PATH)
    out_csv = os.path.join(out_dir, 'lt_extended_results.csv')
    out_df.to_csv(out_csv, index=False, float_format='%.4f')
    print(f'\nSaved: {out_csv}')

    # --- Summary ---
    print('\n' + '=' * 72)
    print('BASELINE (DH Dyn 2x3x [A] Scenario D):')
    print(f"  CAGR_OOS={b['CAGR_OOS']*100:.2f}%  Sharpe_OOS={b['Sharpe_OOS']:.4f}  "
          f"MaxDD={b['MaxDD_FULL']*100:.2f}%")
    print(f"Phase-1 Best: {PHASE1_BEST['config']}  "
          f"CAGR_OOS={PHASE1_BEST['CAGR_OOS']*100:.2f}%  "
          f"Sharpe={PHASE1_BEST['Sharpe_OOS']:.4f}")

    print('\nTop 10 by CAGR_OOS (this sweep):')
    cols = ['config', 'CAGR_IS', 'CAGR_OOS', 'CAGR_FULL',
            'Sharpe_OOS', 'MaxDD_FULL', 'W5Y_FULL', 'W10Y_STAR', 'IS_OOS_GAP']
    top_oos = out_df.nlargest(10, 'CAGR_OOS')
    print(top_oos[cols].to_string(index=False))

    print('\nTop 10 by Sharpe_OOS (this sweep):')
    top_sh = out_df.nlargest(10, 'Sharpe_OOS')
    print(top_sh[cols].to_string(index=False))

    accepted = out_df[out_df.apply(is_accepted, axis=1)]
    print(f'\n=== ACCEPTED (all gates C1-C7): {len(accepted)} / {len(out_df)} new configs ===')
    if len(accepted) > 0:
        best = accepted.sort_values(
            ['Sharpe_OOS', 'W10Y_STAR', 'IS_OOS_GAP', 'CAGR_OOS'],
            ascending=[False, False, True, False]
        ).iloc[0]
        print('\nBEST new candidate:')
        for k2, v in best.items():
            print(f'  {k2}: {v:.4f}' if isinstance(v, float) else f'  {k2}: {v}')

        # Compare vs Phase-1 best
        if best['Sharpe_OOS'] > PHASE1_BEST['Sharpe_OOS']:
            print(f"\n  *** NEW OVERALL BEST: Sharpe {best['Sharpe_OOS']:.4f} > "
                  f"Phase-1 {PHASE1_BEST['Sharpe_OOS']:.4f} ***")
        if best['CAGR_OOS'] > PHASE1_BEST['CAGR_OOS']:
            print(f"  *** CAGR_OOS {best['CAGR_OOS']*100:.2f}% > "
                  f"Phase-1 {PHASE1_BEST['CAGR_OOS']*100:.2f}% ***")
    else:
        print('No new config passed all gates.')

    # Combined summary (Phase-1 + Extended)
    phase1_csv = os.path.join(out_dir, 'lt_sweep_results.csv')
    if os.path.exists(phase1_csv):
        p1 = pd.read_csv(phase1_csv)
        combined = pd.concat([p1, out_df], ignore_index=True)
        combined_csv = os.path.join(out_dir, 'lt_combined_results.csv')
        combined.to_csv(combined_csv, index=False, float_format='%.4f')
        combined_accepted = combined[combined.apply(is_accepted, axis=1)]
        print(f'\n=== COMBINED (Phase-1 + Extended): {len(combined_accepted)} / {len(combined)} total ===')
        if len(combined_accepted) > 0:
            print('Top-5 combined (by Sharpe_OOS):')
            top5 = combined_accepted.sort_values('Sharpe_OOS', ascending=False).head(5)
            print(top5[cols].to_string(index=False))
        print(f'\nSaved: {combined_csv}')

    print('\n' + '=' * 72)
    return out_df


if __name__ == '__main__':
    main()
