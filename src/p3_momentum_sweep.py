"""
P3: Momentum Leverage Sweep
============================
# Evaluation Standard: v1.0
# Cost Scenario: D
# IS: 1974-01-02〜2021-05-07 / OOS: 2021-05-08〜現在

目的: P3_Momentum（モメンタム z-score → sigmoid → レバ）を
     Scenario D 条件で正式評価する。

グリッド: m × k = 4×4 = 16 runs
  m ∈ {10, 20, 60, 120}  (モメンタム窓)
  k ∈ {0.5, 1.0, 1.5, 2.0}  (sigmoid 傾き)

比較ベースライン: S2_VZGated (CAGR_OOS +27.51%, Sharpe_OOS +0.770)

出力:
  - コンソール結果テーブル
  - p3_momentum_sweep_results.csv  (プロジェクトルート)
  - P3_MOMENTUM_SWEEP_2026-05-22.md (プロジェクトルート)
"""

import sys
import os
import types
import itertools

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
from dynamic_leverage_strategies import compute_L_momentum
from compute_cfd_worst10y import (
    prepare_gold_local,
    nav_to_annual,
    rolling_nY_cagr,
)
from _sweep_format import MD_WFA_NOTE

# ---------------------------------------------------------------------------
REF_S2_CAGR_OOS = 0.2751
REF_S2_SHARPE   = 0.770

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

M_SWEEP = [10, 20, 60, 120]
K_SWEEP = [0.5, 1.0, 1.5, 2.0]
P3_FIXED = dict(l_min=1.0, l_max=7.0, step=0.5)


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
    beat_count = sum(1 for r in results if r['Sharpe_OOS'] > REF_S2_SHARPE)
    best = max(results, key=lambda x: x['Sharpe_OOS'])
    lines = [
        '# P3: Momentum Leverage Sweep',
        '',
        '作成日: 2026-05-22',
        '最終更新日: 2026-05-22',
        '',
        '**初回 Scenario D 正式評価**。モメンタム z-score を sigmoid でレバに変換する P3 戦略。',
        '',
        f'- **IS**: {IS_START} 〜 {IS_END} / **OOS**: {OOS_START} 〜',
        f'- **S2 ベースライン**: CAGR_OOS +27.51%, Sharpe_OOS +0.770',
        '',
        '## 結果テーブル（Sharpe_OOS 降順）',
        '',
        '| m | k | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD_FULL | Worst10Y★ | P10_5Y | Worst5Y | IS-OOS gap | Tr |',
        '|--:|--:|--------:|---------:|-----------:|-----------:|----------:|-------:|--------:|-----------:|---:|',
    ]
    for r in sorted(results, key=lambda x: x['Sharpe_OOS'], reverse=True):
        gap_pp = (r['CAGR_IS'] - r['CAGR_OOS']) * 100
        beat = ' ◎' if r['Sharpe_OOS'] > REF_S2_SHARPE else ''
        lines.append(
            f'| {r["m"]} '
            f'| {r["k"]:.1f} '
            f'| {_fp(r["CAGR_IS"])} '
            f'| {_fp(r["CAGR_OOS"])} '
            f'| {_ff(r["Sharpe_OOS"])}{beat} '
            f'| {_fp(r["MaxDD_FULL"])} '
            f'| {_fp(r["Worst10Y_star"])} '
            f'| {_fp(r["P10_5Y"])} '
            f'| {_fp(r["Worst5Y"])} '
            f'| {gap_pp:+.2f} pp '
            f'| {int(round(r.get("Trades_yr", 27)))} |'
        )
    lines += [
        '',
        f'◎ = S2 ベースライン (Sharpe +0.770) を上回る',
        '',
        MD_WFA_NOTE,
        '',
        '## サマリ',
        '',
        f'- 全 {len(results)} runs 中、S2 ベースラインを上回る: **{beat_count} 件**',
        f'- 最高 Sharpe_OOS: **{best["Sharpe_OOS"]:+.3f}** (m={best["m"]}, k={best["k"]:.1f})',
        '',
        '## 判定',
        '',
    ]
    if beat_count > 0:
        lines.append(
            f'✅ **P3 は S2 ベースラインを {beat_count}/{len(results)} 構成で上回る。'
            '上位構成を STRATEGY_REGISTRY へ Shortlisted として追記すること。**'
        )
    else:
        lines.append(
            '❌ **P3 は全構成で S2 ベースラインを下回る。'
            'STRATEGY_REGISTRY へ Rejected として追記すること（理由: モメンタムのみのレバ変調では改善なし）。**'
        )
    lines += [
        '',
        '## 再現コマンド',
        '',
        '```',
        'python -X utf8 src/p3_momentum_sweep.py',
        '```',
        '',
        '*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `STRATEGY_REGISTRY.md`*',
    ]
    return '\n'.join(lines)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    total_runs = len(M_SWEEP) * len(K_SWEEP)
    print('=' * 70)
    print(f'P3: Momentum Leverage Sweep — {total_runs} runs')
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

    print(f'\nRunning P3 grid ({total_runs} runs)...')
    results = []
    run_idx = 0

    for m_mom, k_mom in itertools.product(M_SWEEP, K_SWEEP):
        run_idx += 1
        print(f'  [{run_idx:2d}/{total_runs}] m={m_mom:3d} k={k_mom:.1f} ...',
              end=' ', flush=True)

        L_p3 = compute_L_momentum(
            close,
            m=m_mom,
            k=k_mom,
            **P3_FIXED,
        )
        nav = build_nav_strategy(
            close, lev_A, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD',
            cfd_leverage=L_p3.values,
            cfd_spread=CFD_SPREAD_LOW,
        )
        m_met = calc_all_metrics(nav, dates, n_tr / n_years)
        row = {
            'm':             m_mom,
            'k':             k_mom,
            'CAGR_IS':       m_met['CAGR_IS'],
            'CAGR_OOS':      m_met['CAGR_OOS'],
            'Sharpe_OOS':    m_met['Sharpe_OOS'],
            'MaxDD_FULL':    m_met['MaxDD_FULL'],
            'Worst10Y_star': m_met['Worst10Y_star'],
            'P10_5Y':        m_met['P10_5Y'],
            'Worst5Y':       m_met['Worst5Y'],
            'IS_OOS_gap':    m_met['IS_OOS_gap'],
            'Trades_yr':     m_met['Trades_yr'],
            'WFA_CI95_lo':   np.nan,
            'WFA_WFE':       np.nan,
        }
        results.append(row)
        beat = '◎' if m_met['Sharpe_OOS'] > REF_S2_SHARPE else ' '
        print(f'CAGR_OOS={m_met["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m_met["Sharpe_OOS"]:+.3f} {beat}')

    beat_count = sum(1 for r in results if r['Sharpe_OOS'] > REF_S2_SHARPE)
    best = max(results, key=lambda x: x['Sharpe_OOS'])
    print()
    print('=' * 110)
    print(f'P3 Results ({total_runs} runs) — sorted by Sharpe_OOS desc')
    print('=' * 110)
    hdr = (f'{"m":>4}  {"k":>4}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"MaxDD":>9}  {"Worst10Y★":>10}  {"IS-OOS gap":>11}')
    print(hdr)
    print('-' * 110)
    for r in sorted(results, key=lambda x: x['Sharpe_OOS'], reverse=True):
        beat = '◎' if r['Sharpe_OOS'] > REF_S2_SHARPE else ' '
        print(
            f'{r["m"]:>4d}'
            f'  {r["k"]:>4.1f}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f} {beat}'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {r["Worst10Y_star"]*100:>+9.2f}%'
            f'  {(r["IS_OOS_gap"]*100):>+10.2f} pp'
        )
    print('=' * 110)
    print(f'\nS2 baseline: CAGR_OOS +27.51%, Sharpe_OOS +0.770')
    print(f'P3 beat S2: {beat_count}/{total_runs}')
    print(f'Best: m={best["m"]}, k={best["k"]:.1f} '
          f'→ CAGR_OOS {best["CAGR_OOS"]*100:+.2f}%, Sharpe_OOS {best["Sharpe_OOS"]:+.3f}')

    csv_path = os.path.join(BASE, 'p3_momentum_sweep_results.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    md = generate_report(results)
    md_path = os.path.join(BASE, 'P3_MOMENTUM_SWEEP_2026-05-22.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
