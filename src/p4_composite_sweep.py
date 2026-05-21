"""
P4: Composite Leverage Sweep (本命) — SOFR×Vol×Momentum 三因子
================================================================
# Evaluation Standard: v1.0
# Cost Scenario: D
# IS: 1974-01-02〜2021-05-07 / OOS: 2021-05-08〜現在

目的: P4_Composite（dynamic_leverage_strategies.py で「本命」と明記）を
     初めて Scenario D 条件で正式評価する。
     3因子（SOFR×ボラ×モメンタム）の乗算スコアでレバ決定。

設計思想:
  score = f_sofr × f_vol × f_mom  ∈ [0,1]
  L = l_min + (l_max - l_min) × score
  「低金利×低ボラ×上昇トレンド」が同時成立するときのみフルレバ。
  1因子でも悪条件なら乗算の論理 AND 効果でデレバ。

グリッド: sofr_high × target_vol × m × k = 3×2×2×2 = 24 runs
  sofr_high ∈ {0.05, 0.08, 0.10}
  target_vol ∈ {0.20, 0.60}
  m (momentum window) ∈ {20, 60}
  k (momentum sigmoid steepness) ∈ {0.5, 1.0}

比較ベースライン: S2_VZGated (CAGR_OOS +27.51%, Sharpe_OOS +0.770)

出力:
  - コンソール結果テーブル
  - p4_composite_sweep_results.csv  (プロジェクトルート)
  - P4_COMPOSITE_SWEEP_2026-05-21.md (プロジェクトルート)
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
from dynamic_leverage_strategies import compute_L_composite
from compute_cfd_worst10y import (
    prepare_gold_local,
    nav_to_annual,
    rolling_nY_cagr,
)

# ---------------------------------------------------------------------------
# Baseline comparison (no sanity reference for new strategy; soft warn only)
REF_S2_CAGR_OOS  = 0.2751  # S2_VZGated baseline CAGR_OOS
REF_S2_SHARPE    = 0.770   # S2_VZGated baseline Sharpe_OOS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sweep grid
SOFR_HIGH_SWEEP = [0.05, 0.08, 0.10]
TV_SWEEP        = [0.20, 0.60]
M_SWEEP         = [20, 60]
K_SWEEP         = [0.5, 1.0]

# Fixed params (common to all runs)
P4_FIXED = dict(l_min=1.0, l_max=7.0, step=0.5)


def compute_p10_5y(nav, trading_days=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(trading_days * 5)) ** 0.2 - 1
    return float(r.dropna().quantile(0.10))


def compute_worst5y(nav, trading_days=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(trading_days * 5)) ** 0.2 - 1
    return float(r.dropna().min())


def _fp(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v * 100:+.{d}f}%'


def _ff(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:+.{d}f}'


def generate_report(results):
    lines = [
        '# P4: Composite Leverage Sweep (本命) — SOFR×Vol×Momentum',
        '',
        '作成日: 2026-05-21',
        '最終更新日: 2026-05-21',
        '',
        '**初回 Scenario D 正式評価**。P4 は dynamic_leverage_strategies.py で「本命」と明記されている',
        '3因子乗算型レバレッジ戦略。S2_VZGated との比較で有効性を判定する。',
        '',
        f'- **IS**: {IS_START} 〜 {IS_END} / **OOS**: {OOS_START} 〜',
        f'- **S2 ベースライン**: CAGR_OOS +27.51%, Sharpe_OOS +0.770',
        '',
        '## 結果テーブル',
        '',
        '| sofr_high | tv | m | k | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD_FULL | Worst10Y★ | P10_5Y | Worst5Y | IS-OOS gap |',
        '|----------:|---:|--:|--:|--------:|---------:|-----------:|-----------:|----------:|-------:|--------:|-----------:|',
    ]
    # Sort by Sharpe_OOS descending
    for r in sorted(results, key=lambda x: x['Sharpe_OOS'], reverse=True):
        gap_pp = (r['CAGR_IS'] - r['CAGR_OOS']) * 100
        beat = ' ◎' if r['Sharpe_OOS'] > REF_S2_SHARPE else ''
        lines.append(
            f'| {r["sofr_high"]:.2f} '
            f'| {r["target_vol"]:.2f} '
            f'| {r["m"]} '
            f'| {r["k"]:.1f} '
            f'| {_fp(r["CAGR_IS"])} '
            f'| {_fp(r["CAGR_OOS"])} '
            f'| {_ff(r["Sharpe_OOS"])}{beat} '
            f'| {_fp(r["MaxDD_FULL"])} '
            f'| {_fp(r["Worst10Y_star"])} '
            f'| {_fp(r["P10_5Y"])} '
            f'| {_fp(r["Worst5Y"])} '
            f'| {gap_pp:+.2f} pp |'
        )

    best = max(results, key=lambda x: x['Sharpe_OOS'])
    beat_count = sum(1 for r in results if r['Sharpe_OOS'] > REF_S2_SHARPE)

    lines += [
        '',
        f'◎ = S2 ベースライン (Sharpe +0.770) を上回る',
        '',
        '## サマリ',
        '',
        f'- 全 {len(results)} runs 中、S2 ベースラインを上回る構成: **{beat_count} 件**',
        f'- 最高 Sharpe_OOS: **{best["Sharpe_OOS"]:+.3f}** '
        f'(sofr_high={best["sofr_high"]:.2f}, tv={best["target_vol"]:.2f}, '
        f'm={best["m"]}, k={best["k"]:.1f})',
        f'- 最高 CAGR_OOS: **{_fp(max(results, key=lambda x: x["CAGR_OOS"])["CAGR_OOS"])}**',
        '',
        '## 判定',
        '',
    ]
    if beat_count > 0:
        lines.append(
            f'✅ **P4 は S2 ベースラインを {beat_count}/{len(results)} 構成で上回る。'
            '上位構成を STRATEGY_REGISTRY へ Shortlisted として追記すること。**'
        )
    else:
        lines.append(
            '❌ **P4 は全構成で S2 ベースラインを下回る。'
            'STRATEGY_REGISTRY へ Rejected として追記すること（理由: 三因子乗算でベース戦略を改善せず）。**'
        )

    lines += [
        '',
        '## 再現コマンド',
        '',
        '```',
        'python -X utf8 src/p4_composite_sweep.py',
        '```',
        '',
        '*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `STRATEGY_REGISTRY.md`*',
    ]
    return '\n'.join(lines)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    total_runs = len(SOFR_HIGH_SWEEP) * len(TV_SWEEP) * len(M_SWEEP) * len(K_SWEEP)
    print('=' * 70)
    print(f'P4: Composite Leverage Sweep — {total_runs} runs')
    print('実行日: 2026-05-21')
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

    print(f'\nRunning P4 grid ({total_runs} runs)...')
    results = []
    run_idx = 0

    for sofr_high, tv, m_mom, k_mom in itertools.product(
            SOFR_HIGH_SWEEP, TV_SWEEP, M_SWEEP, K_SWEEP):
        run_idx += 1
        print(f'  [{run_idx:2d}/{total_runs}] sofr_high={sofr_high:.2f} tv={tv:.2f} m={m_mom:2d} k={k_mom:.1f} ...',
              end=' ', flush=True)

        L_p4 = compute_L_composite(
            close, ret, sofr,
            sofr_high=sofr_high,
            target_vol=tv,
            m=m_mom,
            k=k_mom,
            **P4_FIXED,
        )
        nav = build_nav_strategy(
            close, lev_A, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD',
            cfd_leverage=L_p4.values,
            cfd_spread=CFD_SPREAD_LOW,
        )
        m_met = calc_7metrics(nav, dates, trades_per_year=n_tr / n_years)
        ann = nav_to_annual(nav, dates)
        r10 = rolling_nY_cagr(ann, 10)
        worst10y_star = float(r10.min()) if len(r10) > 0 else np.nan
        p10_5y  = compute_p10_5y(nav.values)
        worst5y = compute_worst5y(nav.values)

        row = {
            'sofr_high':     sofr_high,
            'target_vol':    tv,
            'm':             m_mom,
            'k':             k_mom,
            'CAGR_IS':       m_met['CAGR_IS'],
            'CAGR_OOS':      m_met['CAGR_OOS'],
            'Sharpe_OOS':    m_met['Sharpe_OOS'],
            'MaxDD_FULL':    m_met['MaxDD_FULL'],
            'Worst10Y_star': worst10y_star,
            'P10_5Y':        p10_5y,
            'Worst5Y':       worst5y,
            'IS_OOS_gap':    m_met['CAGR_IS'] - m_met['CAGR_OOS'],
        }
        results.append(row)
        beat = '◎' if m_met['Sharpe_OOS'] > REF_S2_SHARPE else ' '
        print(f'CAGR_OOS={m_met["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m_met["Sharpe_OOS"]:+.3f} {beat}')

    # Console summary
    best = max(results, key=lambda x: x['Sharpe_OOS'])
    beat_count = sum(1 for r in results if r['Sharpe_OOS'] > REF_S2_SHARPE)
    print()
    print('=' * 110)
    print(f'P4 Results ({total_runs} runs) — sorted by Sharpe_OOS desc')
    print('=' * 110)
    hdr = (f'{"sofr_h":>6}  {"tv":>4}  {"m":>3}  {"k":>3}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  '
           f'{"Sharpe_OOS":>11}  {"MaxDD":>9}  {"Worst10Y★":>10}  {"IS-OOS gap":>11}')
    print(hdr)
    print('-' * 110)
    for r in sorted(results, key=lambda x: x['Sharpe_OOS'], reverse=True):
        beat = '◎' if r['Sharpe_OOS'] > REF_S2_SHARPE else ' '
        print(
            f'{r["sofr_high"]:>6.2f}'
            f'  {r["target_vol"]:>4.2f}'
            f'  {r["m"]:>3d}'
            f'  {r["k"]:>3.1f}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f} {beat}'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {r["Worst10Y_star"]*100:>+9.2f}%'
            f'  {(r["IS_OOS_gap"]*100):>+10.2f} pp'
        )
    print('=' * 110)
    print(f'\nS2 baseline: CAGR_OOS +27.51%, Sharpe_OOS +0.770')
    print(f'P4 beat S2: {beat_count}/{total_runs} configs')
    print(f'Best: sofr_high={best["sofr_high"]:.2f}, tv={best["target_vol"]:.2f}, '
          f'm={best["m"]}, k={best["k"]:.1f}')
    print(f'  → CAGR_OOS {best["CAGR_OOS"]*100:+.2f}%, Sharpe_OOS {best["Sharpe_OOS"]:+.3f}')

    csv_path = os.path.join(BASE, 'p4_composite_sweep_results.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    md = generate_report(results)
    md_path = os.path.join(BASE, 'P4_COMPOSITE_SWEEP_2026-05-21.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
