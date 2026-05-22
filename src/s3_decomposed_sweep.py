"""
S3: Decomposed A2 Leverage Sweep
==================================
# Evaluation Standard: v1.0
# Cost Scenario: D
# IS: 1974-01-02〜2021-05-07 / OOS: 2021-05-08〜現在

目的: S3_Decomposed（A2 原子因子 → L_t 直接再配線）を
     Scenario D 条件で正式評価する。
     守備系（dd × vm）^ beta × 攻撃系（slope × mom）× ボラ調整（vt）

グリッド: beta_defense ∈ {0.5, 0.75, 1.0, 1.5, 2.0} = 5 runs
  beta_defense < 1: 部分回復でも早めにレバ上昇（積極型）
  beta_defense > 1: 完全回復確認後にレバ上昇（保守型）
  beta_defense = 1: 線形（基準）

比較ベースライン: S2_VZGated (CAGR_OOS +27.51%, Sharpe_OOS +0.770)

出力:
  - コンソール結果テーブル
  - s3_decomposed_sweep_results.csv  (プロジェクトルート)
  - S3_DECOMPOSED_SWEEP_2026-05-22.md (プロジェクトルート)
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

from backtest_engine import load_data, calc_dd_signal
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
from dynamic_leverage_strategies import compute_L_s3_decomposed
from compute_cfd_worst10y import (
    prepare_gold_local,
    nav_to_annual,
    rolling_nY_cagr,
)
from opt_lev2x3x import calc_asym_ewma
from test_delay_robust import calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from _sweep_format import MD_WFA_NOTE

# ---------------------------------------------------------------------------
REF_S2_CAGR_OOS = 0.2751
REF_S2_SHARPE   = 0.770

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BETA_SWEEP = [0.5, 0.75, 1.0, 1.5, 2.0]
S3_FIXED   = dict(l_min=1.0, l_max=7.0, step=0.5)


def build_a2_components(close, returns):
    """A2 原子因子を個別に返す（build_a2_signal の内部ロジックを再現）。"""
    dd  = calc_dd_signal(close, 0.82, 0.92)
    av  = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean()
    ratio = close / ma150
    ttv = (0.10 + 0.20 * (ratio - 0.85) / 0.30).clip(0.10, 0.30).fillna(0.20)
    vt  = (ttv / av).clip(0, 1.0)
    ma200 = close.rolling(200).mean()
    sl = ma200.pct_change()
    sm = sl.rolling(60).mean()
    ss = sl.rolling(60).std().replace(0, 0.0001)
    slope = (0.9 + 0.35 * (sl - sm) / ss).clip(0.3, 1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp  = calc_vix_proxy(returns)
    vz  = (vp - vp.rolling(252).mean()) / vp.rolling(252).std().replace(0, 0.001)
    vm  = (1.0 - 0.25 * vz).clip(0.5, 1.15)
    return {'dd': dd, 'vt': vt, 'slope': slope, 'mom': mom, 'vm': vm.fillna(1.0)}, vz.fillna(0)


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
        '# S3: Decomposed A2 Leverage Sweep',
        '',
        '作成日: 2026-05-22',
        '最終更新日: 2026-05-22',
        '',
        '**初回 Scenario D 正式評価**。A2 原子因子（dd, vt, slope, mom, vm）を直接 L_t に再配線する S3 戦略。',
        '',
        f'- **IS**: {IS_START} 〜 {IS_END} / **OOS**: {OOS_START} 〜',
        f'- **S2 ベースライン**: CAGR_OOS +27.51%, Sharpe_OOS +0.770',
        '',
        '## 結果テーブル（beta_defense 昇順）',
        '',
        '| beta_defense | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD_FULL | Worst10Y★ | P10_5Y | Worst5Y | IS-OOS gap | Tr |',
        '|-------------:|--------:|---------:|-----------:|-----------:|----------:|-------:|--------:|-----------:|---:|',
    ]
    for r in sorted(results, key=lambda x: x['beta_defense']):
        gap_pp = (r['CAGR_IS'] - r['CAGR_OOS']) * 100
        beat = ' ◎' if r['Sharpe_OOS'] > REF_S2_SHARPE else ''
        tag = ' ← 基準' if abs(r['beta_defense'] - 1.0) < 1e-9 else ''
        lines.append(
            f'| {r["beta_defense"]:.2f}{tag}{beat} '
            f'| {_fp(r["CAGR_IS"])} '
            f'| {_fp(r["CAGR_OOS"])} '
            f'| {_ff(r["Sharpe_OOS"])} '
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
        f'- 最高 Sharpe_OOS: **{best["Sharpe_OOS"]:+.3f}** (beta_defense={best["beta_defense"]:.2f})',
        '',
        '## 判定',
        '',
    ]
    if beat_count > 0:
        lines.append(
            f'✅ **S3 は S2 ベースラインを {beat_count}/{len(results)} 構成で上回る。'
            '上位構成を STRATEGY_REGISTRY へ Shortlisted として追記すること。**'
        )
    else:
        lines.append(
            '❌ **S3 は全構成で S2 ベースラインを下回る。'
            'STRATEGY_REGISTRY へ Rejected として追記すること（理由: A2 因子直接再配線では改善なし）。**'
        )
    lines += [
        '',
        '## 再現コマンド',
        '',
        '```',
        'python -X utf8 src/s3_decomposed_sweep.py',
        '```',
        '',
        '*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `STRATEGY_REGISTRY.md`*',
    ]
    return '\n'.join(lines)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print(f'S3: Decomposed A2 Leverage Sweep — {len(BETA_SWEEP)} runs')
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
    raw_a2, vz_base = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz_base, THRESHOLD)
    n_years = n / TRADING_DAYS
    print(f'  DH Dyn: {n_tr} trades, {n_tr / n_years:.1f}/yr')

    print('Building A2 components for S3...')
    components, vz_s3 = build_a2_components(close, ret)
    print(f'  Components: dd[{components["dd"].min():.3f},{components["dd"].max():.3f}] '
          f'vt[{components["vt"].min():.3f},{components["vt"].max():.3f}]')

    print(f'\nSweeping beta_defense ∈ {BETA_SWEEP} ...')
    results = []

    for beta in BETA_SWEEP:
        print(f'  beta_defense={beta:.2f} ...', end=' ', flush=True)
        L_s3 = compute_L_s3_decomposed(
            components,
            beta_defense=beta,
            **S3_FIXED,
        )
        nav = build_nav_strategy(
            close, lev_A, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD',
            cfd_leverage=L_s3.values,
            cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, n_tr / n_years)
        row = {
            'beta_defense':  beta,
            'CAGR_IS':       m['CAGR_IS'],
            'CAGR_OOS':      m['CAGR_OOS'],
            'Sharpe_OOS':    m['Sharpe_OOS'],
            'MaxDD_FULL':    m['MaxDD_FULL'],
            'Worst10Y_star': m['Worst10Y_star'],
            'P10_5Y':        m['P10_5Y'],
            'Worst5Y':       m['Worst5Y'],
            'IS_OOS_gap':    m['IS_OOS_gap'],
            'Trades_yr':     m['Trades_yr'],
            'WFA_CI95_lo':   np.nan,
            'WFA_WFE':       np.nan,
        }
        results.append(row)
        beat = '◎' if m['Sharpe_OOS'] > REF_S2_SHARPE else ' '
        print(f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m["Sharpe_OOS"]:+.3f} {beat}')

    beat_count = sum(1 for r in results if r['Sharpe_OOS'] > REF_S2_SHARPE)
    best = max(results, key=lambda x: x['Sharpe_OOS'])
    print()
    print('=' * 110)
    print(f'S3 Results ({len(results)} runs) — beta_defense sweep')
    print('=' * 110)
    hdr = (f'{"beta":>5}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"MaxDD":>9}  {"Worst10Y★":>10}  {"P10_5Y":>8}  {"IS-OOS gap":>11}')
    print(hdr)
    print('-' * 110)
    for r in results:
        tag = ' ← BASE' if abs(r['beta_defense'] - 1.0) < 1e-9 else '       '
        beat = '◎' if r['Sharpe_OOS'] > REF_S2_SHARPE else ' '
        print(
            f'{r["beta_defense"]:>5.2f}{tag}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f} {beat}'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {r["Worst10Y_star"]*100:>+9.2f}%'
            f'  {r["P10_5Y"]*100:>+7.2f}%'
            f'  {(r["IS_OOS_gap"]*100):>+10.2f} pp'
        )
    print('=' * 110)
    print(f'\nS2 baseline: CAGR_OOS +27.51%, Sharpe_OOS +0.770')
    print(f'S3 beat S2: {beat_count}/{len(results)}')
    print(f'Best: beta_defense={best["beta_defense"]:.2f} '
          f'→ CAGR_OOS {best["CAGR_OOS"]*100:+.2f}%, Sharpe_OOS {best["Sharpe_OOS"]:+.3f}')

    csv_path = os.path.join(BASE, 's3_decomposed_sweep_results.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    md = generate_report(results)
    md_path = os.path.join(BASE, 'S3_DECOMPOSED_SWEEP_2026-05-22.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
