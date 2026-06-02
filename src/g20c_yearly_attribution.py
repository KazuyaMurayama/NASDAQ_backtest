"""
G20C: NEW CANDIDATE vs F10 年次寄与分解
=================================================================
QC Agent 3 指摘:
  OOS CAGR +2.05pp 改善が 2022 単年由来か全年均等か?
  → 単年集中なら lucky regime fit、均等なら structural

NEW CANDIDATE (vz=0.65+lmax=7+ε=0.015) vs F10 (vz=0.70) の
年次税後リターン差を全期間で出し、特に OOS の年次 attribution を可視化。
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
    K_LO, K_HI, K_MID, THRESHOLD,
)
from g18_daily_trade_cost_wfa import (
    build_cfd_nav_with_cost, metrics_from_nav,
    apply_tax_cfd_decimal,
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


def get_yearly_returns(a, dates, close, vz_thr):
    raw_a2_vals = a['raw_a2'].values
    vz_vals = a['vz_arr']
    bull_mask = raw_a2_vals > THRESHOLD

    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': LMAX})
    lev_mod = build_lev_mod_for_vz(a, vz_thr)
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2_vals, vz_vals, a['wn_A'], a['wb_A'],
        np.asarray(a['lev_raw']), bull_mask, EPS,
    )

    spread_ow = SPREAD_RT / 2.0
    nav_adj, _ = build_cfd_nav_with_cost(
        close, lev_mod, wn_f10, a['wg_A'], wb_f10, dates,
        a['gold_2x'], a['bond_3x'], a['sofr'], L_s2.values,
        spread_ow,
    )
    m = metrics_from_nav(nav_adj, dates, a['ret'])
    yr_pre = m['yearly']
    yr_aft = yr_pre.apply(apply_tax_cfd_decimal)
    return yr_aft, yr_pre


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G20C: NEW CANDIDATE vs F10 年次寄与分解')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    close = a['close']

    print('\n[計算中] vz=0.65 (NEW CANDIDATE)...')
    yr_065_aft, yr_065_pre = get_yearly_returns(a, dates, close, 0.65)
    print('[計算中] vz=0.70 (F10 REF)...')
    yr_070_aft, yr_070_pre = get_yearly_returns(a, dates, close, 0.70)

    # 年次差分 = NEW (vz=0.65) - F10 (vz=0.70)
    diff_aft = (yr_065_aft - yr_070_aft).dropna()

    df = pd.DataFrame({
        'NEW_065_pre': yr_065_pre,
        'F10_070_pre': yr_070_pre,
        'NEW_065_aft': yr_065_aft,
        'F10_070_aft': yr_070_aft,
        'diff_aft_pct': diff_aft * 100,
    }).dropna()
    df.index.name = 'year'

    csv_out = os.path.join(BASE, 'g20c_yearly_attribution.csv')
    df.to_csv(csv_out)
    print(f'\n→ CSV: {csv_out}')

    # OOS 期間集中
    print('\n[OOS (2021-2026) 年次差分]')
    oos = df.loc[df.index >= 2021]
    print(f'{"year":>5s} {"NEW_aft":>9s} {"F10_aft":>9s} {"diff":>9s}')
    for y, row in oos.iterrows():
        print(f'{int(y):>5d} {row["NEW_065_aft"]*100:>+7.2f}% {row["F10_070_aft"]*100:>+7.2f}% '
              f'{row["diff_aft_pct"]:>+7.2f}pp')
    sum_oos = oos['diff_aft_pct'].sum()
    mean_oos = oos['diff_aft_pct'].mean()
    max_year = oos['diff_aft_pct'].abs().idxmax()
    max_val = oos.loc[max_year, 'diff_aft_pct']
    print(f'\n  OOS 6 年 diff 合計: {sum_oos:+.2f}pp / 年平均: {mean_oos:+.2f}pp')
    print(f'  最大絶対 diff 年: {int(max_year)} ({max_val:+.2f}pp)')

    # 単年集中度: max year が合計の何 % を占めるか
    if abs(sum_oos) > 0.01:
        concentration = abs(max_val) / sum([abs(v) for v in oos['diff_aft_pct'].values]) * 100
        print(f'  単年集中度: 最大差年が |diff| 合計の {concentration:.1f}% を占める')

    # IS 期間 (1977-2020)
    is_df = df.loc[(df.index >= 1977) & (df.index <= 2020)]
    is_mean_diff = is_df['diff_aft_pct'].mean()
    print(f'\n[IS (1977-2020) 年次平均 diff]: {is_mean_diff:+.3f}pp/年')

    # 年代別 attribution
    print('\n[年代別 diff sum]')
    for decade_start in [1977, 1980, 1990, 2000, 2010, 2020]:
        decade_end = decade_start + 9
        sub = df.loc[(df.index >= decade_start) & (df.index <= decade_end)]
        if len(sub) > 0:
            s = sub['diff_aft_pct'].sum()
            print(f'  {decade_start}s ({len(sub)} 年): {s:+.2f}pp 合計, 平均 {sub["diff_aft_pct"].mean():+.3f}pp/年')

    # 判定
    print('\n[判定]')
    oos_diff_concentration_2022 = abs(oos.loc[2022, 'diff_aft_pct']) if 2022 in oos.index else 0
    total_abs_oos = sum([abs(v) for v in oos['diff_aft_pct'].values])
    if total_abs_oos > 0:
        pct_2022 = oos_diff_concentration_2022 / total_abs_oos * 100
        if pct_2022 > 60:
            print(f'  ❌ 2022 単年が {pct_2022:.1f}% を占有 → lucky single event')
        elif pct_2022 > 40:
            print(f'  ⚠ 2022 単年が {pct_2022:.1f}% を占有 → 中程度の集中、要再検')
        else:
            print(f'  ✅ 2022 単年は {pct_2022:.1f}% で複数年に分散 → structural advantage 支持')


if __name__ == '__main__':
    main()
