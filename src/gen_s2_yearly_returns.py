"""
S2_VZGated Yearly Returns Report (2026-05-17)
=============================================
S2_VZGated / CFD 3x固定 / CFD 7x固定 / DH Dyn 2x3x [A] / BH 1x の
年次リターン表を生成。FULL / IS / OOS 3期間の統計を含む。
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
    load_sofr,
    build_bond_1x_nav_corrected,
    build_gold_2x,
    build_bond_3x,
    build_a2_signal,
    simulate_rebalance_A,
    TRADING_DAYS, THRESHOLD,
)
from test_portfolio_diversification import prepare_gold_data
from cfd_leverage_backtest import (
    build_cfd_nas_sleeve,
    build_nav_strategy,
    calc_7metrics,
    CFD_SPREAD_LOW,
    FULL_START, FULL_END, IS_START, IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

# S2_VZGated 採用パラメータ (ENH_LEVERAGE_BACKTEST_2026-05-16.md 確定値)
S2_TARGET_VOL = 0.8
S2_K_VZ       = 0.3
S2_GATE_MIN   = 0.5


def build_bh1x_nav(close: pd.Series) -> pd.Series:
    r = close.pct_change().fillna(0)
    nav = (1 + r).cumprod()
    nav.attrs['blowup_days'] = 0
    return nav


def build_cfd_fixed_nav(close: pd.Series, dates: pd.Series,
                          sofr: np.ndarray, gold_2x, bond_3x, leverage: float) -> pd.Series:
    """CFD 固定倍率: DH Dyn(A) ポートフォリオ + NAS スリーブ固定CFDレバ
    （ENH_LEVERAGE_BACKTEST_2026-05-15.md と同じ定義）"""
    raw_a2, vz = build_a2_signal(close, close.pct_change())
    lev_A, wn_A, wg_A, wb_A, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    return build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=float(leverage), cfd_spread=CFD_SPREAD_LOW
    )


def build_s2_vzgated_nav(close: pd.Series, dates: pd.Series,
                          sofr: np.ndarray, gold_2x, bond_3x) -> pd.Series:
    """S2_VZGated: DH Dynポートフォリオ + S2動的CFDレバレッジ"""
    returns = close.pct_change()
    raw_a2, vz = build_a2_signal(close, returns)
    lev_A, wn_A, wg_A, wb_A, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    L_s2 = compute_L_s2_vz_gated(
        returns, vz,
        target_vol=S2_TARGET_VOL, k_vz=S2_K_VZ, gate_min=S2_GATE_MIN,
        n=20, l_min=1.0, l_max=7.0, step=0.5
    )
    return build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW
    )


def build_dh_dyn_2x3x_A_nav(close: pd.Series, dates: pd.Series,
                              sofr: np.ndarray, gold_2x, bond_3x) -> pd.Series:
    """DH Dyn 2x3x [A]: 現行ベスト戦略 (TQQQ mode, threshold=0.15)"""
    raw_a2, vz = build_a2_signal(close, close.pct_change())
    lev_A, wn_A, wg_A, wb_A, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    return build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='TQQQ'
    )


def nav_to_annual_returns(nav: pd.Series, dates: pd.Series) -> pd.Series:
    """日次NAV → 年次リターン (%)"""
    df = pd.DataFrame({'nav': nav.values, 'dt': dates.values})
    df['year'] = pd.to_datetime(df['dt']).dt.year
    last_val = df.groupby('year')['nav'].last()
    ret = (last_val / last_val.shift(1) - 1).dropna()
    first_year = last_val.index[0]
    ret[first_year] = last_val[first_year] / nav.iloc[0] - 1
    return (ret * 100).round(1).sort_index()


def calc_stats_extended(nav: pd.Series, dates: pd.Series) -> dict:
    """FULL / IS / OOS 統計 + Worst5Y, 年次サマリー"""
    m = calc_7metrics(nav, dates)
    ar = nav_to_annual_returns(nav, dates)
    return {
        'CAGR_FULL':   m.get('CAGR_FULL',   np.nan) * 100,
        'CAGR_IS':     m.get('CAGR_IS',     np.nan) * 100,
        'CAGR_OOS':    m.get('CAGR_OOS',    np.nan) * 100,
        'Sharpe_FULL': m.get('Sharpe_FULL', np.nan),
        'Sharpe_IS':   m.get('Sharpe_IS',   np.nan),
        'Sharpe_OOS':  m.get('Sharpe_OOS',  np.nan),
        'MaxDD_FULL':  m.get('MaxDD_FULL',  np.nan) * 100,
        'Worst5Y':     m.get('Worst5Y',     np.nan) * 100,
        'Median':      float(ar.median()),
        'Max':         float(ar.max()),
        'Min':         float(ar.min()),
        'Plus':        int((ar > 0).sum()),
        'Minus':       int((ar <= 0).sum()),
    }


def generate_md(all_annual: dict, all_stats: dict, data_info: dict) -> str:
    strat_names = list(all_annual.keys())
    all_years   = sorted(set().union(*[set(s.index) for s in all_annual.values()]))

    short = {
        'S2_VZGated':      'S2_VZG',
        'CFD 3x [固定]':   'CFD 3x',
        'CFD 7x [固定]':   'CFD 7x',
        'DH Dyn 2x3x [A]': 'DH 2x3x',
        'BH 1x':           'BH 1x',
    }

    lines = []
    lines.append('# S2_VZGated 年次リターン表 (S2/CFD 3x/7x + DH 2x3x [A] + BH 1x)')
    lines.append('')
    lines.append('作成日: 2026-05-17')
    lines.append('最終更新日: 2026-05-17')
    lines.append('')
    lines.append(f'**生成日**: 2026-05-17')
    lines.append(f'**データ期間**: {data_info["start"]} 〜 {data_info["end"]}')
    lines.append(f'**IS期間**: {IS_START} 〜 {IS_END}')
    lines.append(f'**OOS期間**: {OOS_START} 〜 {data_info["end"]}')
    lines.append(f'**参照**: [ENH_LEVERAGE_BACKTEST_2026-05-16.md](ENH_LEVERAGE_BACKTEST_2026-05-16.md) / [CFD_DYNAMIC_LEVERAGE_GUIDE.md](CFD_DYNAMIC_LEVERAGE_GUIDE.md)')
    lines.append('')
    lines.append('---')
    lines.append('')

    # --- 補正条件 ---
    lines.append('## 補正条件')
    lines.append('')
    lines.append('| 戦略 | コスト条件 |')
    lines.append('|------|-----------|')
    lines.append('| S2_VZGated | CFD (L-1)×SOFR + (L-1)×0.20%スプレッド + DH Dynポートフォリオ (Gold2x 20%, Bond3x 20%) |')
    lines.append('| CFD 3x/7x [固定] | DH Dyn(A) ポートフォリオ + NASスリーブ固定CFDレバ ((L-1)×SOFR + (L-1)×0.20%スプレッド) |')
    lines.append('| DH Dyn 2x3x [A] | TQQQ mode: 3×SOFR + 2×swap(0.50%) + TER(0.86%) + DH Dynポートフォリオ |')
    lines.append('| BH 1x | 補正なし (ベンチマーク) |')
    lines.append('| SOFR proxy | DTB3 (FRED 3M T-bill, 平均 4.37%/年) |')
    lines.append('| S2パラメータ | target_vol=0.8, k_vz=0.3, gate_min=0.5, l_max=7 |')
    lines.append('| DH Dyn閾値 | Approach A, threshold=0.15, DELAY=2 |')
    lines.append('')
    lines.append('---')
    lines.append('')

    # --- 統計サマリー (FULL) ---
    hdr = '| 統計量 | ' + ' | '.join(short[n] for n in strat_names) + ' |'
    sep = '|' + '--------|' * (len(strat_names) + 1)

    lines.append('## 統計サマリー (FULL期間)')
    lines.append('')
    lines.append(hdr)
    lines.append(sep)

    full_rows = [
        ('**CAGR (FULL)**', lambda s: f'{s["CAGR_FULL"]:+.2f}%'),
        ('Sharpe (FULL)',    lambda s: f'{s["Sharpe_FULL"]:.3f}'),
        ('MaxDD (FULL)',     lambda s: f'{s["MaxDD_FULL"]:.1f}%'),
        ('Worst5Y CAGR',    lambda s: f'{s["Worst5Y"]:+.2f}%'),
        ('中央値',           lambda s: f'{s["Median"]:+.1f}%'),
        ('最大',             lambda s: f'{s["Max"]:+.1f}%'),
        ('最小',             lambda s: f'{s["Min"]:+.1f}%'),
        ('プラス年数',       lambda s: str(s['Plus'])),
        ('マイナス年数',     lambda s: str(s['Minus'])),
    ]
    for label, fmt in full_rows:
        row = f'| {label} | ' + ' | '.join(fmt(all_stats[n]) for n in strat_names) + ' |'
        lines.append(row)
    lines.append('')

    # --- IS統計 ---
    lines.append(f'## IS統計 ({IS_START} 〜 {IS_END})')
    lines.append('')
    lines.append(hdr)
    lines.append(sep)
    is_rows = [
        ('**CAGR (IS)**',  lambda s: f'{s["CAGR_IS"]:+.2f}%'),
        ('Sharpe (IS)',    lambda s: f'{s["Sharpe_IS"]:.3f}'),
    ]
    for label, fmt in is_rows:
        row = f'| {label} | ' + ' | '.join(fmt(all_stats[n]) for n in strat_names) + ' |'
        lines.append(row)
    lines.append('')

    # --- OOS統計 ---
    lines.append(f'## OOS統計 ({OOS_START} 〜 {data_info["end"]})')
    lines.append('')
    lines.append(hdr)
    lines.append(sep)
    oos_rows = [
        ('**CAGR (OOS)**',  lambda s: f'{s["CAGR_OOS"]:+.2f}%'),
        ('Sharpe (OOS)',    lambda s: f'{s["Sharpe_OOS"]:.3f}'),
    ]
    for label, fmt in oos_rows:
        row = f'| {label} | ' + ' | '.join(fmt(all_stats[n]) for n in strat_names) + ' |'
        lines.append(row)
    lines.append('')
    lines.append('---')
    lines.append('')

    # --- 年次リターン表 ---
    lines.append('## 年次リターン表 (1974-2026) [単位: %]')
    lines.append('')
    lines.append('> `[OOS]` = OOS期間 (2021年以降)')
    lines.append('')
    yr_hdr = '| 年 | ' + ' | '.join(short[n] for n in strat_names) + ' |'
    yr_sep = '|----|' + ':---:|' * len(strat_names)
    lines.append(yr_hdr)
    lines.append(yr_sep)

    for yr in all_years:
        yr_str = f'{yr} [OOS]' if yr >= 2021 else str(yr)
        cells = []
        for n in strat_names:
            v = all_annual[n].get(yr, np.nan)
            cells.append('—' if np.isnan(v) else f'{v:+.1f}')
        lines.append(f'| {yr_str} | ' + ' | '.join(cells) + ' |')

    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/gen_s2_yearly_returns.py`*')
    lines.append(f'*データ期間: {data_info["start"]} 〜 {data_info["end"]}*')

    return '\n'.join(lines)


def main():
    print('=' * 70)
    print('S2 Yearly Returns -- S2_VZGated + CFD 3x/7x + DH 2x3x [A] + BH 1x')
    print('=' * 70)

    df    = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df):,} days)')

    print('Loading SOFR...')
    sofr = load_sofr(dates)

    print('Building bond 3x + gold 2x...')
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, True)
    gold_1x = prepare_gold_data(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)

    print('Building NAVs...')
    navs = {}
    print('  S2_VZGated...')
    navs['S2_VZGated']      = build_s2_vzgated_nav(close, dates, sofr, gold_2x, bond_3x)
    print('  CFD 3x [固定]...')
    navs['CFD 3x [固定]']   = build_cfd_fixed_nav(close, dates, sofr, gold_2x, bond_3x, 3.0)
    print('  CFD 7x [固定]...')
    navs['CFD 7x [固定]']   = build_cfd_fixed_nav(close, dates, sofr, gold_2x, bond_3x, 7.0)
    print('  DH Dyn 2x3x [A]...')
    navs['DH Dyn 2x3x [A]'] = build_dh_dyn_2x3x_A_nav(close, dates, sofr, gold_2x, bond_3x)
    print('  BH 1x...')
    navs['BH 1x']           = build_bh1x_nav(close)

    print('\nComputing annual returns & stats...')
    all_annual = {}
    all_stats  = {}
    for name, nav in navs.items():
        ar = nav_to_annual_returns(nav, dates)
        all_annual[name] = ar
        all_stats[name]  = calc_stats_extended(nav, dates)
        s = all_stats[name]
        print(f'  {name}: CAGR_FULL={s["CAGR_FULL"]:+.2f}%  Sharpe_OOS={s["Sharpe_OOS"]:.3f}  Worst5Y={s["Worst5Y"]:.2f}%')

    # Sanity check
    print('\n--- Sanity Check ---')
    # 旧 ENH_LEVERAGE_BACKTEST_2026-05-15.md と一致するべき値
    expected = {
        'CFD 3x [固定]':    (23.20, 0.6),
        'CFD 7x [固定]':    (41.36, 0.6),
        'DH Dyn 2x3x [A]': (22.50, 0.6),
        'BH 1x':            (10.98, 0.6),
    }
    for name, (exp, tol) in expected.items():
        act = all_stats[name]['CAGR_FULL']
        ok  = abs(act - exp) < tol
        print(f'  {name}: {act:+.2f}% (expect ~{exp:+.2f}%) {"✅" if ok else "⚠️ WARN"}')

    # Print abbreviated table
    print('\n--- Annual Returns (first 5 + last 5 years) ---')
    sample_years = sorted(all_annual['BH 1x'].index)
    show_years   = sample_years[:5] + sample_years[-5:]
    header = f"{'Year':<10}" + ''.join(f'{n[:12]:>14}' for n in navs.keys())
    print(header)
    print('-' * (10 + 14 * len(navs)))
    for yr in show_years:
        yr_str = f'{yr}[OOS]' if yr >= 2021 else str(yr)
        cells  = ''.join(
            f'{all_annual[n].get(yr, float("nan")):>+13.1f}%'
            if not np.isnan(all_annual[n].get(yr, float('nan')))
            else f'{"—":>14}'
            for n in navs.keys()
        )
        print(f'{yr_str:<10}' + cells)

    print('\nGenerating report...')
    data_info = {'start': str(dates.iloc[0].date()), 'end': str(dates.iloc[-1].date())}
    md  = generate_md(all_annual, all_stats, data_info)
    out = os.path.join(BASE, 'CFD_S2_YEARLY_RETURNS_2026-05-17.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
