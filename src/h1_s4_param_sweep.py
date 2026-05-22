"""
H1: S4 RelVol-Gated Multi-Param Sweep
======================================
# Evaluation Standard: v1.0
# Cost Scenario: D
# IS: 1974-01-02〜2021-05-07 / OOS: 2021-05-08〜現在

目的: S4_RelVol（相対ボラゲート + VIXゲート 2段構成）を
     Scenario D 条件で初めて正式評価する。
     P2/S2 の「target_vol 死パラメータ問題」を回避する設計。

グリッド: l_base × k_rel × rel_threshold × HL_PAIRS = 2×3×3×2 = 36 runs
  l_base          ∈ {6, 7}
  k_rel           ∈ {1.5, 2.0, 3.0}
  rel_threshold   ∈ {1.0, 1.2, 1.4}
  HL_PAIRS (short_hl, long_hl) ∈ {(10, 60), (20, 120)}

固定パラメータ: k_vz=0.3, gate_min=0.5, l_min=1.0, step=0.5

比較ベースライン: S2_VZGated (CAGR_OOS +27.51%, Sharpe_OOS +0.770)
上位目標: S2+LT2 (Sharpe_OOS +0.858)

出力:
  - h1_s4_param_sweep_results.csv  (プロジェクトルート)
  - H1_S4_PARAM_SWEEP_2026-05-22.md (プロジェクトルート)
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
from dynamic_leverage_strategies import compute_L_s4_relvol
from compute_cfd_worst10y import (
    prepare_gold_local,
    nav_to_annual,
    rolling_nY_cagr,
)
from _sweep_format import MD_WFA_NOTE

# ---------------------------------------------------------------------------
REF_S2_SHARPE   = 0.770
REF_LT2_SHARPE  = 0.858

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

L_BASE_SWEEP      = [6, 7]
K_REL_SWEEP       = [1.5, 2.0, 3.0]
REL_THRESH_SWEEP  = [1.0, 1.2, 1.4]
HL_PAIRS          = [(10, 60), (20, 120)]  # (short_hl, long_hl)

S4_FIXED = dict(k_vz=0.3, gate_min=0.5, l_min=1.0, step=0.5)


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
    beat_s2  = sum(1 for r in results if r['Sharpe_OOS'] > REF_S2_SHARPE)
    beat_lt2 = sum(1 for r in results if r['Sharpe_OOS'] > REF_LT2_SHARPE)
    best = max(results, key=lambda x: x['Sharpe_OOS'])
    lines = [
        '# H1: S4 RelVol-Gated Multi-Param Sweep',
        '',
        '作成日: 2026-05-22',
        '最終更新日: 2026-05-22',
        '',
        '**初回 Scenario D 正式評価**。S4 は相対ボラ（短期/長期 EWMA 比）+ VIX ゲートの 2 段構成。',
        'P2/S2 の target_vol 死パラメータ問題を回避するため相対化されたレバ変調を採用。',
        '',
        f'- **IS**: {IS_START} 〜 {IS_END} / **OOS**: {OOS_START} 〜',
        f'- **S2 ベースライン**: CAGR_OOS +27.51%, Sharpe_OOS +0.770',
        f'- **上位目標 (S2+LT2)**: Sharpe_OOS +0.858',
        f'- **固定**: k_vz=0.3, gate_min=0.5, l_min=1.0, step=0.5',
        '',
        '## 結果テーブル（Sharpe_OOS 降順, 上位 20 件）',
        '',
        '| l_base | k_rel | rel_th | HL | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD_FULL | Worst10Y★ | P10_5Y | Worst5Y | IS-OOS gap | Tr |',
        '|-------:|------:|-------:|---:|--------:|---------:|-----------:|-----------:|----------:|-------:|--------:|-----------:|---:|',
    ]
    top20 = sorted(results, key=lambda x: x['Sharpe_OOS'], reverse=True)[:20]
    for r in top20:
        gap_pp = (r['CAGR_IS'] - r['CAGR_OOS']) * 100
        mark = ' ★' if r['Sharpe_OOS'] > REF_LT2_SHARPE else (' ◎' if r['Sharpe_OOS'] > REF_S2_SHARPE else '')
        hl_str = f'{r["short_hl"]}/{r["long_hl"]}'
        lines.append(
            f'| {r["l_base"]} '
            f'| {r["k_rel"]:.1f} '
            f'| {r["rel_threshold"]:.1f} '
            f'| {hl_str} '
            f'| {_fp(r["CAGR_IS"])} '
            f'| {_fp(r["CAGR_OOS"])} '
            f'| {_ff(r["Sharpe_OOS"])}{mark} '
            f'| {_fp(r["MaxDD_FULL"])} '
            f'| {_fp(r["Worst10Y_star"])} '
            f'| {_fp(r["P10_5Y"])} '
            f'| {_fp(r["Worst5Y"])} '
            f'| {gap_pp:+.2f} pp '
            f'| {int(round(r.get("Trades_yr", 27)))} |'
        )
    lines += [
        '',
        '◎ = S2 単体 (Sharpe +0.770) を上回る  ★ = S2+LT2 (Sharpe +0.858) を上回る',
        '',
        MD_WFA_NOTE,
        '',
        '## サマリ',
        '',
        f'- 全 {len(results)} runs 中、S2 ベースラインを上回る: **{beat_s2} 件**',
        f'- S2+LT2 実績を上回る: **{beat_lt2} 件**',
        f'- 最高 Sharpe_OOS: **{best["Sharpe_OOS"]:+.3f}** '
        f'(l_base={best["l_base"]}, k_rel={best["k_rel"]:.1f}, '
        f'rel_th={best["rel_threshold"]:.1f}, HL={best["short_hl"]}/{best["long_hl"]})',
        '',
        '## 判定',
        '',
    ]
    if beat_lt2 > 0:
        lines.append(
            f'✅ **S4 は S2+LT2 ベースラインを {beat_lt2}/{len(results)} 構成で上回る。'
            '上位構成を STRATEGY_REGISTRY へ Shortlisted として追記すること。**'
        )
    elif beat_s2 > 0:
        lines.append(
            f'✅ **S4 は S2 単体を {beat_s2}/{len(results)} 構成で上回るが、S2+LT2 は超えない。'
            'S2+LT2 との組み合わせ検証を検討すること。**'
        )
    else:
        lines.append(
            '❌ **S4 は全構成で S2 ベースラインを下回る。'
            'STRATEGY_REGISTRY へ Rejected として追記すること（理由: 相対ボラゲートでは改善なし）。**'
        )
    lines += [
        '',
        '## 再現コマンド',
        '',
        '```',
        'python -X utf8 src/h1_s4_param_sweep.py',
        '```',
        '',
        '*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `STRATEGY_REGISTRY.md`*',
    ]
    return '\n'.join(lines)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    total_runs = len(L_BASE_SWEEP) * len(K_REL_SWEEP) * len(REL_THRESH_SWEEP) * len(HL_PAIRS)
    print('=' * 70)
    print(f'H1: S4 RelVol-Gated Multi-Param Sweep — {total_runs} runs')
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

    print(f'\nRunning S4 grid ({total_runs} runs)...')
    results = []
    run_idx = 0

    for l_base, k_rel, rel_th, (short_hl, long_hl) in itertools.product(
            L_BASE_SWEEP, K_REL_SWEEP, REL_THRESH_SWEEP, HL_PAIRS):
        run_idx += 1
        print(f'  [{run_idx:2d}/{total_runs}] l_base={l_base} k_rel={k_rel:.1f} '
              f'rel_th={rel_th:.1f} HL={short_hl}/{long_hl} ...',
              end=' ', flush=True)

        L_s4 = compute_L_s4_relvol(
            ret, vz,
            l_base=l_base,
            k_rel=k_rel,
            rel_threshold=rel_th,
            short_hl=short_hl,
            long_hl=long_hl,
            **S4_FIXED,
        )
        nav = build_nav_strategy(
            close, lev_A, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD',
            cfd_leverage=L_s4.values,
            cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, n_tr / n_years)
        row = {
            'l_base':        l_base,
            'k_rel':         k_rel,
            'rel_threshold': rel_th,
            'short_hl':      short_hl,
            'long_hl':       long_hl,
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
        beat = '★' if m['Sharpe_OOS'] > REF_LT2_SHARPE else ('◎' if m['Sharpe_OOS'] > REF_S2_SHARPE else ' ')
        print(f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m["Sharpe_OOS"]:+.3f} {beat}')

    beat_s2  = sum(1 for r in results if r['Sharpe_OOS'] > REF_S2_SHARPE)
    beat_lt2 = sum(1 for r in results if r['Sharpe_OOS'] > REF_LT2_SHARPE)
    best = max(results, key=lambda x: x['Sharpe_OOS'])

    print()
    print('=' * 120)
    print(f'H1 S4 Results ({total_runs} runs) — Top 10 by Sharpe_OOS')
    print('=' * 120)
    hdr = (f'{"lbase":>5}  {"krel":>5}  {"rth":>4}  {"HL":>7}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  '
           f'{"Sharpe_OOS":>11}  {"MaxDD":>9}  {"Worst10Y★":>10}  {"IS-OOS gap":>11}')
    print(hdr)
    print('-' * 120)
    for r in sorted(results, key=lambda x: x['Sharpe_OOS'], reverse=True)[:10]:
        beat = '★' if r['Sharpe_OOS'] > REF_LT2_SHARPE else ('◎' if r['Sharpe_OOS'] > REF_S2_SHARPE else ' ')
        print(
            f'{r["l_base"]:>5d}'
            f'  {r["k_rel"]:>5.1f}'
            f'  {r["rel_threshold"]:>4.1f}'
            f'  {r["short_hl"]:>3d}/{r["long_hl"]:<3d}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f} {beat}'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {r["Worst10Y_star"]*100:>+9.2f}%'
            f'  {(r["IS_OOS_gap"]*100):>+10.2f} pp'
        )
    print('=' * 120)
    print(f'\nS2 baseline: Sharpe_OOS +0.770  S2+LT2: +0.858')
    print(f'S4 beat S2: {beat_s2}/{total_runs}  beat LT2: {beat_lt2}/{total_runs}')
    print(f'Best: l_base={best["l_base"]}, k_rel={best["k_rel"]:.1f}, '
          f'rel_th={best["rel_threshold"]:.1f}, HL={best["short_hl"]}/{best["long_hl"]}')
    print(f'  → CAGR_OOS {best["CAGR_OOS"]*100:+.2f}%, Sharpe_OOS {best["Sharpe_OOS"]:+.3f}')

    csv_path = os.path.join(BASE, 'h1_s4_param_sweep_results.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    md = generate_report(results)
    md_path = os.path.join(BASE, 'H1_S4_PARAM_SWEEP_2026-05-22.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
