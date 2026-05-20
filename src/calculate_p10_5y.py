"""
P10 5Y (10th-percentile 5-Year CAGR) computation for all 10 strategies.

Rolling daily window: (nav / nav.shift(252*5))**0.2 - 1
Full period samples: ~11,909 (1974-01-02 to 2026-03-26)

Strategies:
  1. S2_VZGated (tv=0.8, k=0.3, gate=0.5)
  2. P2 best (vol-target, tv=0.8)
  3. S4_RelVol (l_base=7, k_rel=2.0)
  4. CFD 7x [fixed]
  5. DH Dyn 2x3x [A] Scenario D
  6. BH 1x (NASDAQ)
  7. P02_Dyn×CPI [mult] (bond_gate=DynCorr, nas_gate=CPI)
  8. P05_HY×CPI [mult]  (nas_gate=HY×CPI)
  9. P01_Dyn×HY [mult]  (bond_gate=DynCorr, nas_gate=HY)
 10. DH Dyn 2x3x [A+LT2] (LT2-N750-k0.5-modeB)

Output: p10_5y_results.csv
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
    build_nav,
    DATA_PATH, DATA_DIR, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy,
    CFD_SPREAD_LOW,
)
from dynamic_leverage_strategies import (
    compute_L_vol_target,
    compute_L_s2_vz_gated,
    compute_L_s4_relvol,
)
from compute_cfd_worst10y import prepare_gold_local
from long_cycle_signal import (
    build_lt_signal, signal_to_bias, apply_lt_mode_b,
)
from sleeves_extended import build_gold_tocom
from test_portfolio_diversification import prepare_gold_data


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNALS_PATH = os.path.join(BASE, 'data', 'timing_signals_raw.csv')


# ---------------------------------------------------------------------------
# P10 5Y helper
# ---------------------------------------------------------------------------

def compute_p10_5y(nav: np.ndarray, trading_days: int = 252) -> float:
    """10th percentile of daily rolling 5Y CAGR distribution."""
    nav_s = pd.Series(nav)
    rolling5 = (nav_s / nav_s.shift(trading_days * 5)) ** 0.2 - 1
    rolling5 = rolling5.dropna()
    return float(rolling5.quantile(0.10))


def compute_worst5y(nav: np.ndarray, trading_days: int = 252) -> float:
    nav_s = pd.Series(nav)
    rolling5 = (nav_s / nav_s.shift(trading_days * 5)) ** 0.2 - 1
    return float(rolling5.min())


def fmt(v):
    return f'{v*100:+.2f}%' if v is not None and not np.isnan(v) else 'N/A'


# ---------------------------------------------------------------------------
# Gate signal builders for P01/P02/P05 (from p4_overfitting_check.py)
# ---------------------------------------------------------------------------

def build_hy_gate(hy, z_thresh=1.0, slope=0.5):
    mu = hy.rolling(252, min_periods=126).mean()
    sd = hy.rolling(252, min_periods=126).std().clip(lower=0.01)
    z = (hy - mu) / sd
    g = (1.0 - np.maximum(0.0, z - z_thresh) * slope).clip(0.2, 1.0)
    return g.fillna(1.0)


def build_cpi_gate(cpi_yoy, cpi_accel, cpi_thresh=5.0, reduce_factor=0.3):
    infl_regime = ((cpi_yoy - cpi_thresh) / 5.0).clip(0.0, 1.0)
    accel_norm = (cpi_accel / 2.0).clip(0.0, 1.0)
    g = (1.0 - reduce_factor * np.maximum(infl_regime, accel_norm)).clip(
        1.0 - reduce_factor, 1.0
    )
    return g.fillna(1.0)


def build_corr_gate(close, bond_3x, gold_2x, window=60, min_gate=0.2):
    ret = pd.Series(close.pct_change().fillna(0).values, index=close.index)
    bond_ret = pd.Series(bond_3x, index=close.index).pct_change().fillna(0)
    gold_ret = pd.Series(gold_2x, index=close.index).pct_change().fillna(0)
    rho_nb = ret.rolling(window).corr(bond_ret)
    rho_ng = ret.rolling(window).corr(gold_ret)
    hedge_health = (-rho_nb).clip(lower=0.0) + (-rho_ng).clip(lower=0.0)
    g = hedge_health.clip(lower=min_gate, upper=1.0)
    return g.fillna(1.0)


def apply_gates(wn_A_arr, nas_gate=None, bond_gate=None):
    ones = np.ones(len(wn_A_arr))
    g_nas = np.where(np.isnan(nas_gate), 1.0, nas_gate) if nas_gate is not None else ones
    g_bond = np.where(np.isnan(bond_gate), 1.0, bond_gate) if bond_gate is not None else ones
    wn = np.clip(wn_A_arr * g_nas, 0.0, 1.0)
    rest = 1.0 - wn
    wg = rest * 0.5
    wb = rest * 0.5 * g_bond
    return wn, wg, wb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('P10 5Y (10th-percentile 5-Year CAGR) Calculation')
    print('All 10 Strategies - DH Dyn 2x3x [A] Comparison Table')
    print('=' * 70)

    # ---- Load base data ----
    df = load_data(DATA_PATH)
    close = df['Close']
    ret = close.pct_change().fillna(0)
    dates = df['Date']
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(dates)} days)')

    # ---- Shared assets for main-branch strategies (Scenario D) ----
    sofr = load_sofr(dates)
    gold_1x_local = prepare_gold_local(dates)
    gold_2x_sd = build_gold_2x(gold_1x_local, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x_sd = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    # ---- DH Dyn signal ----
    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    print(f'DH Dyn signal: {n_tr} trades, {n_tr/52.26:.1f}/yr')

    results = []

    # ==========================================================
    # Strategies 1-4: CFD variants (Scenario D gold/bond)
    # ==========================================================
    print('\n--- Strategy 1: S2_VZGated ---')
    L_s2 = compute_L_s2_vz_gated(ret, vz, target_vol=0.8, k_vz=0.3, gate_min=0.5,
                                   n=20, l_min=1.0, l_max=7.0, step=0.5)
    nav_s2 = build_nav_strategy(close, lev_A, wn_A, wg_A, wb_A, dates,
                                 gold_2x_sd, bond_3x_sd, sofr,
                                 nas_mode='CFD', cfd_leverage=L_s2.values,
                                 cfd_spread=CFD_SPREAD_LOW)
    p10 = compute_p10_5y(nav_s2.values)
    w5 = compute_worst5y(nav_s2.values)
    print(f'  P10_5Y={fmt(p10)}  Worst5Y={fmt(w5)}')
    results.append({'rank': 1, 'strategy': 'S2_VZGated (tv=0.8, k=0.3, gate=0.5)',
                    'P10_5Y': p10, 'Worst5Y_check': w5})

    print('--- Strategy 2: P2 best (vol-target tv=0.8) ---')
    L_p2 = compute_L_vol_target(ret, target_vol=0.8, n=20, l_min=1.0, l_max=7.0, step=0.5)
    nav_p2 = build_nav_strategy(close, lev_A, wn_A, wg_A, wb_A, dates,
                                 gold_2x_sd, bond_3x_sd, sofr,
                                 nas_mode='CFD', cfd_leverage=L_p2.values,
                                 cfd_spread=CFD_SPREAD_LOW)
    p10 = compute_p10_5y(nav_p2.values)
    w5 = compute_worst5y(nav_p2.values)
    print(f'  P10_5Y={fmt(p10)}  Worst5Y={fmt(w5)}')
    results.append({'rank': 2, 'strategy': 'P2 best (vol-target, tv=0.8)',
                    'P10_5Y': p10, 'Worst5Y_check': w5})

    print('--- Strategy 3: S4_RelVol (l_base=7, k_rel=2.0) ---')
    L_s4 = compute_L_s4_relvol(ret, vz, l_base=7.0, k_rel=2.0,
                                  l_min=1.0, step=0.5)
    nav_s4 = build_nav_strategy(close, lev_A, wn_A, wg_A, wb_A, dates,
                                 gold_2x_sd, bond_3x_sd, sofr,
                                 nas_mode='CFD', cfd_leverage=L_s4.values,
                                 cfd_spread=CFD_SPREAD_LOW)
    p10 = compute_p10_5y(nav_s4.values)
    w5 = compute_worst5y(nav_s4.values)
    print(f'  P10_5Y={fmt(p10)}  Worst5Y={fmt(w5)}')
    results.append({'rank': 3, 'strategy': 'S4_RelVol (l_base=7, k_rel=2.0)',
                    'P10_5Y': p10, 'Worst5Y_check': w5})

    print('--- Strategy 4: CFD 7x [fixed] ---')
    nav_7x = build_nav_strategy(close, lev_A, wn_A, wg_A, wb_A, dates,
                                 gold_2x_sd, bond_3x_sd, sofr,
                                 nas_mode='CFD', cfd_leverage=7.0,
                                 cfd_spread=CFD_SPREAD_LOW)
    p10 = compute_p10_5y(nav_7x.values)
    w5 = compute_worst5y(nav_7x.values)
    print(f'  P10_5Y={fmt(p10)}  Worst5Y={fmt(w5)}')
    results.append({'rank': 4, 'strategy': 'CFD 7x (DH Dyn+7x fixed)',
                    'P10_5Y': p10, 'Worst5Y_check': w5})

    # ==========================================================
    # Strategy 5: DH Dyn 2x3x [A] Scenario D (TQQQ/Gold/Bond)
    # ==========================================================
    print('--- Strategy 5: DH Dyn 2x3x [A] Scenario D ---')
    nav_dha = build_nav(close, lev_A, wn_A, wg_A, wb_A, dates,
                        gold_2x_sd, bond_3x_sd, sofr_daily=sofr, apply_tqqq_sofr=True)
    p10 = compute_p10_5y(nav_dha.values)
    w5 = compute_worst5y(nav_dha.values)
    print(f'  P10_5Y={fmt(p10)}  Worst5Y={fmt(w5)}')
    results.append({'rank': 5, 'strategy': 'DH Dyn 2x3x [A] Scenario D (th=0.15)',
                    'P10_5Y': p10, 'Worst5Y_check': w5})

    # ==========================================================
    # Strategy 6: BH 1x
    # ==========================================================
    print('--- Strategy 6: BH 1x (NASDAQ) ---')
    nav_bh = (1 + ret).cumprod()
    p10 = compute_p10_5y(nav_bh.values)
    w5 = compute_worst5y(nav_bh.values)
    print(f'  P10_5Y={fmt(p10)}  Worst5Y={fmt(w5)}')
    results.append({'rank': 6, 'strategy': 'BH 1x (NASDAQ)',
                    'P10_5Y': p10, 'Worst5Y_check': w5})

    # ==========================================================
    # Strategies 7-9: P01/P02/P05 with gate signals
    # Cost model: gold_tocom + bond_3x (no SOFR on bond) - same as original branch
    # ==========================================================
    print('\n--- Loading timing signals for P01/P02/P05 ---')
    sig_raw = pd.read_csv(SIGNALS_PATH, index_col=0, parse_dates=True)
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    sig = sig_raw.reindex(dates_dt)
    hy_s = pd.Series(sig['hy_spread'].values, index=close.index).fillna(method='ffill').fillna(4.0)
    cpi_yoy = pd.Series(sig['cpi_yoy'].fillna(0.0).values, index=close.index)
    cpi_acc = pd.Series(sig['cpi_accel'].fillna(0.0).values, index=close.index)
    print(f'  HY spread: {hy_s.isna().sum()} NaN; CPI yoy: {cpi_yoy.isna().sum()} NaN')

    # Build gold_tocom and bond_3x (no SOFR on bond) for P-series cost model
    gold_1x_pf = prepare_gold_data(dates_dt)
    gold_2x_p = build_gold_tocom(gold_1x_pf, 2.0, sofr)
    bond_3x_p = build_bond_3x(bond_1x, sofr, apply_sofr=False)

    # Shared DH Dyn A2 signal for P-series (same threshold)
    raw_a2_p, vz_p = build_a2_signal(close, close.pct_change())
    lev_p, wn_p, wg_p, wb_p, _ = simulate_rebalance_A(raw_a2_p, vz_p, THRESHOLD)

    print('--- Strategy 7: P02_Dyn×CPI [mult] ---')
    g_corr = build_corr_gate(close, bond_3x_p, gold_2x_p, window=60, min_gate=0.2)
    g_cpi = build_cpi_gate(cpi_yoy, cpi_acc, cpi_thresh=5.0, reduce_factor=0.3)
    wn7, wg7, wb7 = apply_gates(wn_p, nas_gate=g_cpi.values, bond_gate=g_corr.values)
    nav7 = build_nav_strategy(close, lev_p, wn7, wg7, wb7, dates,
                               gold_2x_p, bond_3x_p, sofr,
                               nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002)
    p10 = compute_p10_5y(nav7.values)
    w5 = compute_worst5y(nav7.values)
    print(f'  P10_5Y={fmt(p10)}  Worst5Y={fmt(w5)} (ref: +0.49%)')
    results.append({'rank': 7, 'strategy': 'P02_Dyn×CPI [mult]',
                    'P10_5Y': p10, 'Worst5Y_check': w5})

    print('--- Strategy 8: P05_HY×CPI [mult] ---')
    g_hy = build_hy_gate(hy_s, z_thresh=1.0, slope=0.5)
    g_cpi = build_cpi_gate(cpi_yoy, cpi_acc, cpi_thresh=5.0, reduce_factor=0.3)
    g_nas = np.clip(g_hy.values * g_cpi.values, 0.2, 1.0)
    wn8, wg8, wb8 = apply_gates(wn_p, nas_gate=g_nas, bond_gate=None)
    nav8 = build_nav_strategy(close, lev_p, wn8, wg8, wb8, dates,
                               gold_2x_p, bond_3x_p, sofr,
                               nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002)
    p10 = compute_p10_5y(nav8.values)
    w5 = compute_worst5y(nav8.values)
    print(f'  P10_5Y={fmt(p10)}  Worst5Y={fmt(w5)} (ref: +6.04%)')
    results.append({'rank': 8, 'strategy': 'P05_HY×CPI [mult]',
                    'P10_5Y': p10, 'Worst5Y_check': w5})

    print('--- Strategy 9: P01_Dyn×HY [mult] ---')
    g_corr = build_corr_gate(close, bond_3x_p, gold_2x_p, window=60, min_gate=0.2)
    g_hy = build_hy_gate(hy_s, z_thresh=1.0, slope=0.5)
    wn9, wg9, wb9 = apply_gates(wn_p, nas_gate=g_hy.values, bond_gate=g_corr.values)
    nav9 = build_nav_strategy(close, lev_p, wn9, wg9, wb9, dates,
                               gold_2x_p, bond_3x_p, sofr,
                               nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002)
    p10 = compute_p10_5y(nav9.values)
    w5 = compute_worst5y(nav9.values)
    print(f'  P10_5Y={fmt(p10)}  Worst5Y={fmt(w5)} (ref: -0.25%)')
    results.append({'rank': 9, 'strategy': 'P01_Dyn×HY [mult]',
                    'P10_5Y': p10, 'Worst5Y_check': w5})

    # ==========================================================
    # Strategy 10: DH Dyn 2x3x [A+LT2] (LT2-N750-k0.5-modeB)
    # ==========================================================
    print('--- Strategy 10: DH Dyn 2x3x [A+LT2] (LT2-N750-k0.5-modeB) ---')
    lt_sig = build_lt_signal(close, 'LT2', N=750)
    lt_bias = signal_to_bias(lt_sig, k_lt=0.5)
    lev_base10, wn10, wg10, wb10, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    lev10 = apply_lt_mode_b(lev_base10, lt_bias, l_min=0.0, l_max=1.0)
    nav10 = build_nav(close, lev10, wn10, wg10, wb10, dates,
                      gold_2x_sd, bond_3x_sd, sofr_daily=sofr, apply_tqqq_sofr=True)
    p10 = compute_p10_5y(nav10.values)
    w5 = compute_worst5y(nav10.values)
    print(f'  P10_5Y={fmt(p10)}  Worst5Y={fmt(w5)}')
    results.append({'rank': 10, 'strategy': 'DH Dyn 2x3x [A+LT2] (LT2-N750-k0.5-modeB)',
                    'P10_5Y': p10, 'Worst5Y_check': w5})

    # ==========================================================
    # Save and print
    # ==========================================================
    df_out = pd.DataFrame(results)
    out_csv = os.path.join(BASE, 'p10_5y_results.csv')
    df_out.to_csv(out_csv, index=False, float_format='%.4f')
    print(f'\nSaved: {out_csv}')

    print('\n' + '=' * 70)
    print('P10 5Y Results Summary')
    print('=' * 70)
    print(f'{"#":<4} {"Strategy":<45} {"P10_5Y":>9} {"Worst5Y(check)":>14}')
    print('-' * 75)
    for r in results:
        print(f'{r["rank"]:<4} {r["strategy"]:<45} {fmt(r["P10_5Y"]):>9} {fmt(r["Worst5Y_check"]):>14}')

    print(f'\nRolling window: 252×5 = {252*5} trading days')
    print(f'Total samples:  ~{len(close) - 252*5} (after warmup)')
    print(f'P10_5Y = 10th percentile of rolling 5Y CAGR distribution')


if __name__ == '__main__':
    main()
