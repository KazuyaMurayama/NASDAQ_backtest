"""
A2: target_vol Robustness Sweep — S2_VZGated
=============================================
# Evaluation Standard: v1.0
# Cost Scenario: D
# IS: 1974-01-02〜2021-05-07 / OOS: 2021-05-08〜現在

目的: ボラターゲット target_vol ∈ {0.10, 0.20, 0.40, 0.60, 0.80, 1.00} で
     S2_VZGated を再実行し、tv=0.80 の感度と有効レンジを確認する。

設計注意 (dynamic_leverage_strategies.py より):
  NASDAQ 実現ボラ中央値≈13.6%/年に対し、tv >= 0.40 では
  target_vol/sigma の中央値 >= 3 → l_max=7 に 99% 以上クリップ。
  tv=0.10〜0.20 では実際に vol-targeting が機能する（sigma 中央値付近）。

固定パラメータ: n=20, k_vz=0.3, gate_min=0.5, l_min=1.0, l_max=7.0, step=0.5
可変: target_vol

出力:
  - コンソール結果テーブル
  - a2_tv_sweep_results.csv  (プロジェクトルート)
  - A2_TV_SWEEP_2026-05-21.md (プロジェクトルート)
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
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy,
    calc_7metrics,
    CFD_SPREAD_LOW,
    IS_START, IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import (
    prepare_gold_local,
    nav_to_annual,
    rolling_nY_cagr,
)

# ---------------------------------------------------------------------------
REF_CAGR_OOS_TV80 = 0.2751  # A1/A6 baseline: tv=0.80, n=20, k_vz=0.30, gate_min=0.50

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

S2_FIXED = dict(n=20, k_vz=0.3, gate_min=0.5, l_min=1.0, l_max=7.0, step=0.5)
TV_SWEEP  = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]


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


def generate_report(results, sanity_ok, sanity_diff_pp):
    lines = [
        '# A2: target_vol Robustness Sweep — S2_VZGated',
        '',
        '作成日: 2026-05-21',
        '最終更新日: 2026-05-21',
        '',
        '**目的**: NASDAQ 実現ボラ中央値≈13.6% に対する target_vol の感度確認。',
        'tv >= 0.40 では l_max=7 に飽和するため、低 tv 帯での挙動を定量化する。',
        '',
        f'- **固定パラメータ**: n=20, k_vz=0.3, gate_min=0.5, l_min=1.0, l_max=7.0, step=0.5',
        f'- **IS**: {IS_START} 〜 {IS_END} / **OOS**: {OOS_START} 〜',
        '',
        '## 結果テーブル',
        '',
        '| target_vol | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD_FULL | Worst10Y★ | P10_5Y | Worst5Y | IS-OOS gap |',
        '|----------:|--------:|---------:|-----------:|-----------:|----------:|-------:|--------:|-----------:|',
    ]
    for r in results:
        tag = ' ← 現行ベスト' if abs(r['target_vol'] - 0.80) < 1e-9 else ''
        gap_pp = (r['CAGR_IS'] - r['CAGR_OOS']) * 100
        lines.append(
            f'| {r["target_vol"]:.2f}{tag} '
            f'| {_fp(r["CAGR_IS"])} '
            f'| {_fp(r["CAGR_OOS"])} '
            f'| {_ff(r["Sharpe_OOS"])} '
            f'| {_fp(r["MaxDD_FULL"])} '
            f'| {_fp(r["Worst10Y_star"])} '
            f'| {_fp(r["P10_5Y"])} '
            f'| {_fp(r["Worst5Y"])} '
            f'| {gap_pp:+.2f} pp |'
        )

    sanity_tag = '✅ 一致（±0.10 pp）' if sanity_ok else f'⚠️ 乖離 {sanity_diff_pp:+.2f} pp'
    lines += [
        '',
        '## サニティチェック',
        '',
        f'- tv=0.80 の CAGR_OOS 対 A1/A6 参照値 (+27.51%): {sanity_tag}',
        '',
        '## 観察',
        '',
        '- tv=0.10〜0.20: 実際に vol-targeting が機能する低 tv 領域（sigma 中央値付近）。',
        '- tv >= 0.40: target_vol/sigma の中央値 >= 3 → l_max=7 に飽和。tv 感度は実質ゼロ。',
        '',
        '## 再現コマンド',
        '',
        '```',
        'python -X utf8 src/a2_tv_sweep.py',
        '```',
        '',
        '*参照: `A1_NVOL_SWEEP_2026-05-21.md`, `CURRENT_BEST_STRATEGY.md`*',
    ]
    return '\n'.join(lines)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('A2: target_vol Robustness Sweep — S2_VZGated')
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

    print(f'\nSweeping target_vol ∈ {TV_SWEEP} ...')
    results = []

    for tv in TV_SWEEP:
        print(f'  target_vol={tv:.2f} ...', end=' ', flush=True)
        L_s2 = compute_L_s2_vz_gated(
            ret, vz,
            target_vol=tv,
            **S2_FIXED,
        )
        nav = build_nav_strategy(
            close, lev_A, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD',
            cfd_leverage=L_s2.values,
            cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_7metrics(nav, dates, trades_per_year=n_tr / n_years)
        ann = nav_to_annual(nav, dates)
        r10 = rolling_nY_cagr(ann, 10)
        worst10y_star = float(r10.min()) if len(r10) > 0 else np.nan
        p10_5y  = compute_p10_5y(nav.values)
        worst5y = compute_worst5y(nav.values)
        row = {
            'target_vol':    tv,
            'CAGR_IS':       m['CAGR_IS'],
            'CAGR_OOS':      m['CAGR_OOS'],
            'Sharpe_OOS':    m['Sharpe_OOS'],
            'MaxDD_FULL':    m['MaxDD_FULL'],
            'Worst10Y_star': worst10y_star,
            'P10_5Y':        p10_5y,
            'Worst5Y':       worst5y,
            'IS_OOS_gap':    m['CAGR_IS'] - m['CAGR_OOS'],
        }
        results.append(row)
        print(f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m["Sharpe_OOS"]:+.3f}')

    # Sanity check
    row80 = next(r for r in results if abs(r['target_vol'] - 0.80) < 1e-9)
    sanity_diff_pp = (row80['CAGR_OOS'] - REF_CAGR_OOS_TV80) * 100
    sanity_ok = abs(sanity_diff_pp) <= 0.10
    print(f'\n[SANITY] tv=0.80 CAGR_OOS = {row80["CAGR_OOS"]*100:+.2f}% '
          f'(ref +27.51%, diff {sanity_diff_pp:+.2f} pp) → {"OK" if sanity_ok else "WARN"}')

    # Console table
    print()
    print('=' * 110)
    hdr = (f'{"tv":>6}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"MaxDD":>9}  {"Worst10Y★":>10}  {"P10_5Y":>8}  {"Worst5Y":>8}  {"IS-OOS gap":>11}')
    print(hdr)
    print('-' * 110)
    for r in results:
        tag = ' ← BASE' if abs(r['target_vol'] - 0.80) < 1e-9 else '       '
        print(
            f'{r["target_vol"]:>6.2f}{tag}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f}'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {r["Worst10Y_star"]*100:>+9.2f}%'
            f'  {r["P10_5Y"]*100:>+7.2f}%'
            f'  {r["Worst5Y"]*100:>+7.2f}%'
            f'  {(r["IS_OOS_gap"]*100):>+10.2f} pp'
        )
    print('=' * 110)

    csv_path = os.path.join(BASE, 'a2_tv_sweep_results.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    md = generate_report(results, sanity_ok, sanity_diff_pp)
    md_path = os.path.join(BASE, 'A2_TV_SWEEP_2026-05-21.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
