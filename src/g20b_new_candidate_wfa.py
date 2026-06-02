"""
G20B: NEW CANDIDATE (vz=0.65+lmax=7+F10ε=0.015) の WFA 50窓厳密検証
=================================================================
g14_wfa_sbi_cfd.py 同等の窓設計で NEW CANDIDATE を評価。
CI95_lo / WFE / t-stat を出して合否判定。

判定基準 (EVALUATION_STANDARD v1.1):
  α: WFA_CI95_lo > 0 AND t_pvalue < 0.05
  β: WFA_WFE ∈ [0.5, 2.0]
  総合: α+β → PASS / α のみ → WARN / α FAIL → FAIL

同時に vz=0.625 (g20a で更に良値だった) も評価して比較。
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import (
    load_shared_assets, BASE,
    generate_windows, compute_window_metrics, compute_summary_stats,
    evaluate_criteria,
    K_LO, K_HI, K_MID, THRESHOLD,
    _make_nav, SBI_CFD_SPREAD,
)
from g18_daily_trade_cost_wfa import (
    build_cfd_nav_with_cost,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from long_cycle_signal import apply_lt_mode_b
from g19a_f10_eps_extended import build_f10_wn_for_eps
from corrected_strategy_backtest import TRADING_DAYS

S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)
LMAX = 7.0
EPS = 0.015
SPREAD_RT = 0.00050


def build_lev_mod_for_vz(a, vz_thr):
    vz_arr = a['vz_arr']
    lt_sig_arr = a['lt_sig_arr']
    k_dyn = np.where(vz_arr > vz_thr, K_HI,
            np.where(vz_arr < -vz_thr, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    return apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)


def run_wfa_for_vz(a, dates, close, vz_thr, windows, label):
    raw_a2_vals = a['raw_a2'].values
    vz_vals = a['vz_arr']
    bull_mask = raw_a2_vals > THRESHOLD

    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': LMAX})
    lev_mod = build_lev_mod_for_vz(a, vz_thr)
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2_vals, vz_vals, a['wn_A'], a['wb_A'],
        np.asarray(a['lev_raw']), bull_mask, EPS,
    )

    # NAV with daily trade cost
    spread_ow = SPREAD_RT / 2.0
    nav_adj, _ = build_cfd_nav_with_cost(
        close, lev_mod, wn_f10, a['wg_A'], wb_f10, dates,
        a['gold_2x'], a['bond_3x'], a['sofr'], L_s2.values,
        spread_ow,
    )

    # Per-window metrics
    L_arr = np.asarray(L_s2.values)
    rows = []
    for w in windows:
        m = compute_window_metrics(nav_adj, w, wn=wn_f10, wb=wb_f10, lev_arr=L_arr)
        m.update(window_id=w['window_id'], short_flag=w.get('short_flag', False),
                 start_date=w['start_date'], end_date=w['end_date'])
        rows.append(m)
    per = pd.DataFrame(rows)
    summary = compute_summary_stats(per)
    verdict, crits = evaluate_criteria(summary)
    summary['verdict'] = verdict
    summary['strategy'] = label
    return summary, per


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G20B: NEW CANDIDATE WFA 50窓厳密検証')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    close = a['close']
    windows = generate_windows(dates)
    print(f'  Windows: {len(windows)}')

    candidates = [
        (0.625, 'NEW-CAND-vz0.625'),
        (0.650, 'NEW-CAND-vz0.650 (★ originally proposed)'),
        (0.700, 'F10-vz0.700 (REF, v6.1 推奨)'),
    ]

    summaries = []
    per_data = {}
    for vz_thr, label in candidates:
        print(f'\n--- {label} (vz_thr={vz_thr}) ---')
        s, per = run_wfa_for_vz(a, dates, close, vz_thr, windows, label)
        summaries.append(s)
        per_data[label] = per
        print(f'  n_windows: {s["n_windows"]}')
        print(f'  mean_CAGR: {s["mean_CAGR"]*100:+.2f}% (IS: {s["mean_CAGR_IS"]*100:+.2f}%, postIS: {s["mean_CAGR_postIS"]*100:+.2f}%)')
        print(f'  CI95_lo: {s["WFA_CI95_lo"]:+.3f}')
        print(f'  CI95_hi: {s["WFA_CI95_hi"]:+.3f}')
        print(f'  t-stat: {s["t_stat"]:.3f}, p-value: {s["t_pvalue"]:.6f}')
        print(f'  WFE: {s["WFA_WFE"]:.3f}')
        print(f'  mean_Sharpe: {s["mean_Sharpe"]:.3f}')
        print(f'  Verdict: {s["verdict"]}')

    df = pd.DataFrame(summaries)
    csv_out = os.path.join(BASE, 'g20b_new_candidate_wfa_summary.csv')
    df.to_csv(csv_out, index=False)
    print(f'\n→ Summary CSV: {csv_out}')

    # Per window CSV
    for label, per in per_data.items():
        per['strategy'] = label
    per_all = pd.concat(per_data.values(), ignore_index=True)
    per_csv = os.path.join(BASE, 'g20b_new_candidate_per_window.csv')
    per_all.to_csv(per_csv, index=False)
    print(f'→ Per-window CSV: {per_csv}')

    print('\n[判定]')
    new_cand_065 = df[df['strategy'].str.contains('vz0.650')].iloc[0]
    new_cand_625 = df[df['strategy'].str.contains('vz0.625')].iloc[0]
    f10_ref = df[df['strategy'].str.contains('vz0.700')].iloc[0]
    print(f'  vz=0.65 verdict: {new_cand_065["verdict"]} (CI95_lo={new_cand_065["WFA_CI95_lo"]:+.3f}, WFE={new_cand_065["WFA_WFE"]:.3f})')
    print(f'  vz=0.625 verdict: {new_cand_625["verdict"]} (CI95_lo={new_cand_625["WFA_CI95_lo"]:+.3f}, WFE={new_cand_625["WFA_WFE"]:.3f})')
    print(f'  F10 (vz=0.70) verdict: {f10_ref["verdict"]} (CI95_lo={f10_ref["WFA_CI95_lo"]:+.3f}, WFE={f10_ref["WFA_WFE"]:.3f})')


if __name__ == '__main__':
    main()
