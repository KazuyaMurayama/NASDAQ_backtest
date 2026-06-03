"""G23F: DH-W1 閾値 (enter, exit) sensitivity sweep
=================================================================
W1 の hysteresis 閾値 (enter_thr, exit_thr) を grid 探索し、
OOS CAGR / IS CAGR / gap の robust 性を確認。

採用値: (enter=0.7, exit=0.3)
Grid: enter ∈ {0.5, 0.6, 0.7, 0.8, 0.9} × exit ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
ただし exit < enter のみ有効 (15 組合せ)

出力:
  - g23f_dh_w1_threshold_sweep.csv (grid 結果)
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from g14_wfa_sbi_cfd import load_shared_assets, BASE
from g18_daily_trade_cost_wfa import (
    build_dh_nav_with_cost, metrics_from_nav, apply_tax_etf_decimal,
)
from g23a_dh_refinement_variants import hold_mask_W1, DH_PER_UNIT


def build_w1_with_thr(a, enter_thr, exit_thr):
    mask = hold_mask_W1(a, enter_thr=enter_thr, exit_thr=exit_thr)
    wn = np.asarray(a['wn_A']) * mask
    wg = np.asarray(a['wg_A']) * mask
    wb = np.asarray(a['wb_A']) * mask
    lev_raw = np.asarray(a['lev_raw']) * mask
    nav, _ = build_dh_nav_with_cost(
        a['close'], lev_raw, wn, wg, wb,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    m = metrics_from_nav(nav, a['dates'], a['ret'])
    yr_aft = m['yearly'].apply(apply_tax_etf_decimal)
    is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1977 <= y <= 2020]]
    oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
    def _geo(x):
        if len(x) == 0: return float('nan')
        c = float(np.prod(1.0 + x.values))
        return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
    cagr_is, cagr_oos = _geo(is_subset), _geo(oos_subset)
    return dict(
        enter_thr=enter_thr, exit_thr=exit_thr,
        hold_ratio_pct=float(mask.mean())*100,
        CAGR_OOS_pct=cagr_oos*100,
        CAGR_IS_pct=cagr_is*100,
        gap_pp=(cagr_is - cagr_oos)*100,
        MaxDD_pct=m['MaxDD_FULL']*100,
        Worst10Y_pct=m['Worst10Y_star']*100,
        P10_5Y_pct=m['P10_5Y']*100,
        Sharpe_OOS=m['Sharpe_OOS'],
    )


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G23F: DH-W1 閾値 (enter, exit) sensitivity sweep')
    print('=' * 80)
    a = load_shared_assets()

    enter_grid = [0.5, 0.6, 0.7, 0.8, 0.9]
    exit_grid  = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []
    print(f'\nGrid: enter ∈ {enter_grid} × exit ∈ {exit_grid}')
    print(f'Constraint: exit < enter (15 valid combos)\n')

    for en in enter_grid:
        for ex in exit_grid:
            if ex >= en:
                continue  # invalid
            r = build_w1_with_thr(a, en, ex)
            rows.append(r)
            flag = ' ★採用' if (en == 0.7 and ex == 0.3) else ''
            print(f'  enter={en:.1f} exit={ex:.1f}: HOLD={r["hold_ratio_pct"]:5.1f}%  '
                  f'OOS={r["CAGR_OOS_pct"]:+6.2f}%  IS={r["CAGR_IS_pct"]:+6.2f}%  '
                  f'gap={r["gap_pp"]:+5.2f}pp  MaxDD={r["MaxDD_pct"]:+6.2f}%  '
                  f'W10Y={r["Worst10Y_pct"]:+5.2f}%{flag}')

    df = pd.DataFrame(rows)
    csv = os.path.join(BASE, 'g23f_dh_w1_threshold_sweep.csv')
    df.to_csv(csv, index=False, float_format='%.4f')
    print(f'\n→ Sweep CSV: {csv}')

    # ranking by OOS CAGR
    df2 = df.sort_values('CAGR_OOS_pct', ascending=False)
    print('\n[OOS CAGR 上位 5]')
    print(df2[['enter_thr', 'exit_thr', 'CAGR_OOS_pct', 'CAGR_IS_pct', 'gap_pp',
                'MaxDD_pct', 'Worst10Y_pct']].head().to_string(index=False))

    # 採用値の rank
    adopted_idx = df.index[(df['enter_thr']==0.7) & (df['exit_thr']==0.3)].tolist()
    if adopted_idx:
        sorted_oos = df['CAGR_OOS_pct'].sort_values(ascending=False).reset_index()
        rank_oos = sorted_oos[sorted_oos['index']==adopted_idx[0]].index[0] + 1
        print(f'\n[採用値 (0.7, 0.3) ranking]')
        print(f'  OOS CAGR rank: {rank_oos} / {len(df)} (中央値 {int(np.ceil(len(df)/2))} 位以上が望ましい)')


if __name__ == '__main__':
    main()
