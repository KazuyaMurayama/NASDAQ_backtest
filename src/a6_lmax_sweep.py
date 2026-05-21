"""
A6: l_max Robustness Sweep — S2_VZGated
=========================================
目的: l_max ∈ {5, 6, 7, 8} でS2_VZGatedを再実行し、l_max=7選択が
     インサンプル最適化バイアスでないことを定量確認する。

固定パラメータ: target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5
可変: l_max

出力:
  - コンソール結果テーブル
  - a6_lmax_sweep_results.csv  (プロジェクトルート)
  - A6_LMAX_SWEEP_2026-05-21.md (プロジェクトルート)
"""

import sys
import os
import types
import datetime

# multitasking スタブ (yfinance 依存回避) -- sys.path 操作より前に置く
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
    IS_START, IS_END, OOS_START, FULL_START, FULL_END,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import (
    prepare_gold_local,
    nav_to_annual,
    rolling_nY_cagr,
)

# ---------------------------------------------------------------------------
# CURRENT_BEST_STRATEGY.md の参照値 (サニティチェック用)
# ---------------------------------------------------------------------------
REF_CAGR_OOS_LMAX7 = 0.2757  # CURRENT_BEST_STRATEGY.md (2026-05-21)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 固定パラメータ
S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)
L_MAX_SWEEP = [5.0, 6.0, 7.0, 8.0]


# ---------------------------------------------------------------------------
# ヘルパー: P10_5Y / Worst5Y (日次ローリング 252×5 窓)
# ---------------------------------------------------------------------------

def compute_p10_5y(nav, trading_days=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(trading_days * 5)) ** 0.2 - 1
    return float(r.dropna().quantile(0.10))


def compute_worst5y(nav, trading_days=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(trading_days * 5)) ** 0.2 - 1
    return float(r.dropna().min())


# ---------------------------------------------------------------------------
# フォーマッタ
# ---------------------------------------------------------------------------

def _fp(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v * 100:+.{d}f}%'


def _ff(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:+.{d}f}'


# ---------------------------------------------------------------------------
# Markdown レポート生成
# ---------------------------------------------------------------------------

def generate_report(results: list, sanity_ok: bool, sanity_diff_pp: float) -> str:
    lines = []
    lines.append('# A6: l_max Robustness Sweep — S2_VZGated')
    lines.append('')
    lines.append(f'**実行日**: 2026-05-21')
    lines.append(f'**目的**: l_max ∈ {{5, 6, 7, 8}} の sweep で l_max=7 採用が'
                 'インサンプル最適化バイアスでないことを定量確認する')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 1. セットアップ')
    lines.append('')
    lines.append('- **戦略**: S2_VZGated（Vol-Zone ゲート型 CFD レバレッジ）')
    lines.append('- **固定パラメータ**: target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5')
    lines.append('- **可変パラメータ**: l_max ∈ {5, 6, 7, 8}')
    lines.append('- **データ期間**: 1974-01-02 〜 2026-03-26（52.26 年）')
    lines.append(f'- **IS 期間**: {IS_START} 〜 {IS_END}')
    lines.append(f'- **OOS 期間**: {OOS_START} 〜')
    lines.append('- **CFD スプレッド**: CFD_SPREAD_LOW (0.20%/yr)')
    lines.append('- **Worst10Y★**: カレンダー年ベース 10 年ローリング CAGR の最小値')
    lines.append('- **P10_5Y**: 日次ローリング 252×5 窓 5 年 CAGR の 10 パーセンタイル')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 2. 結果テーブル')
    lines.append('')

    hdr = ('| l_max | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD_FULL '
           '| Worst10Y★ | P10_5Y | Worst5Y | IS−OOS gap |')
    sep = ('|------:|--------:|---------:|-----------:|-----------:'
           '|----------:|-------:|--------:|-----------:|')
    lines.append(hdr)
    lines.append(sep)

    for r in results:
        lmax_str = f'{r["l_max"]:.1f}'
        tag = ' ← 現行ベスト' if abs(r['l_max'] - 7.0) < 1e-9 else ''
        gap_pp = (r['CAGR_IS'] - r['CAGR_OOS']) * 100
        lines.append(
            f'| {lmax_str}{tag} '
            f'| {_fp(r["CAGR_IS"])} '
            f'| {_fp(r["CAGR_OOS"])} '
            f'| {_ff(r["Sharpe_OOS"])} '
            f'| {_fp(r["MaxDD_FULL"])} '
            f'| {_fp(r["Worst10Y_star"])} '
            f'| {_fp(r["P10_5Y"])} '
            f'| {_fp(r["Worst5Y"])} '
            f'| {gap_pp:+.2f} pp |'
        )
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 3. サニティチェック')
    lines.append('')
    row7 = next((r for r in results if abs(r['l_max'] - 7.0) < 1e-9), None)
    if row7:
        sanity_tag = '✅ 一致（±0.50 pp 以内）' if sanity_ok else f'⚠️ 乖離 {sanity_diff_pp:+.2f} pp — 再現性を確認'
        lines.append(f'- l_max=7.0 の CAGR_OOS: **{row7["CAGR_OOS"]*100:+.2f}%**')
        lines.append(f'- CURRENT_BEST_STRATEGY.md 記載値: **+27.57%**')
        lines.append(f'- 差分: **{sanity_diff_pp:+.2f} pp** → {sanity_tag}')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 4. 観察')
    lines.append('')

    # 単調性チェック
    oos_vals = [r['CAGR_OOS'] for r in results]
    sharpe_vals = [r['Sharpe_OOS'] for r in results]
    is_oos_mono = all(oos_vals[i] <= oos_vals[i+1] for i in range(len(oos_vals)-1))
    is_oos_plateau = (max(oos_vals) - min(oos_vals)) < 0.02  # 2pp 以内なら平坦

    if is_oos_plateau:
        lines.append('- CAGR_OOS は l_max={5,6,7,8} で **2pp 以内に収束**（プラトー）。'
                     'l_max の選択感度は低い。')
    elif is_oos_mono:
        lines.append('- CAGR_OOS は l_max 増加につれ**単調増加**。l_max=8.0 が数値上は優位。')
    else:
        best_lmax = results[oos_vals.index(max(oos_vals))]['l_max']
        lines.append(f'- CAGR_OOS は l_max={best_lmax:.1f} でピーク（非単調）。')

    # Sharpe_OOS
    best_sharpe_lmax = results[sharpe_vals.index(max(sharpe_vals))]['l_max']
    lines.append(f'- Sharpe_OOS の最高値は l_max={best_sharpe_lmax:.1f}。')

    # IS-OOS gap
    gaps = [(r['l_max'], (r['CAGR_IS'] - r['CAGR_OOS']) * 100) for r in results]
    gap7 = next(g for l, g in gaps if abs(l - 7.0) < 1e-9)
    max_gap = max(g for _, g in gaps)
    lines.append(f'- IS−OOS gap (l_max=7): {gap7:+.2f} pp（全体最大 {max_gap:+.2f} pp）。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 5. 判定（インサンプル最適化バイアスの定量評価）')
    lines.append('')
    lines.append('### 判定基準（事前定義）')
    lines.append('')
    lines.append('| 基準 | 条件 | 結果 |')
    lines.append('|------|------|------|')

    # 基準 (i): l_max=7 の CAGR_OOS が最大、または最大値との差が 1.0pp 以内
    row7_oos = row7['CAGR_OOS'] if row7 else 0
    max_oos = max(oos_vals)
    diff_from_best = (max_oos - row7_oos) * 100
    crit1 = diff_from_best <= 1.0
    lines.append(f'| (i) CAGR_OOS: l_max=7 と最高値の差 ≤ 1.0 pp | diff = {diff_from_best:+.2f} pp | {"✅ PASS" if crit1 else "⚠️ WARN"} |')

    # 基準 (ii): Sharpe_OOS で隣接（6,8）との差が 0.10 以内
    sharpe7 = next(r['Sharpe_OOS'] for r in results if abs(r['l_max'] - 7.0) < 1e-9)
    sharpe6 = next((r['Sharpe_OOS'] for r in results if abs(r['l_max'] - 6.0) < 1e-9), None)
    sharpe8 = next((r['Sharpe_OOS'] for r in results if abs(r['l_max'] - 8.0) < 1e-9), None)
    neighbors = [v for v in [sharpe6, sharpe8] if v is not None and not np.isnan(v)]
    max_sharpe_diff = max(abs(sharpe7 - v) for v in neighbors) if neighbors else 0.0
    crit2 = max_sharpe_diff <= 0.10
    lines.append(f'| (ii) Sharpe_OOS: 隣接 l_max との差 ≤ 0.10 | max diff = {max_sharpe_diff:.3f} | {"✅ PASS" if crit2 else "⚠️ WARN"} |')

    # 基準 (iii): IS-OOS gap が l_max=7 で隣接より大幅に大きくない（2pp 超過しない）
    gap6 = next((g for l, g in gaps if abs(l - 6.0) < 1e-9), None)
    gap8 = next((g for l, g in gaps if abs(l - 8.0) < 1e-9), None)
    neighbor_gaps = [g for g in [gap6, gap8] if g is not None]
    max_gap_diff = max(gap7 - g for g in neighbor_gaps) if neighbor_gaps else 0.0
    crit3 = max_gap_diff <= 2.0
    lines.append(f'| (iii) IS−OOS gap (l_max=7) が隣接より ≤ 2.0 pp 大きい | max excess = {max_gap_diff:.2f} pp | {"✅ PASS" if crit3 else "⚠️ WARN"} |')
    lines.append('')

    all_pass = crit1 and crit2 and crit3
    verdict = 'PASS' if all_pass else ('WARN' if (crit1 or crit2) else 'FAIL')
    lines.append(f'**総合判定: {verdict}**')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 6. 結論')
    lines.append('')
    if verdict == 'PASS':
        lines.append('l_max=7.0 の採用は **OOS・Sharpe・IS-OOS gap の 3 基準すべてで合格**。'
                     'インサンプル最適化バイアスの証拠なし。現行ベスト戦略を維持する。')
    elif verdict == 'WARN':
        lines.append('l_max=7.0 の採用は一部基準で境界的。'
                     '上記 WARN 項目を確認し、l_max=6 または l_max=8 への変更を検討する価値がある。')
    else:
        lines.append('l_max=7.0 は複数基準で不合格。l_max 選択バイアスの可能性が高い。'
                     'A1（vol窓 n=40）実装後に改めてレビューを推奨。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 7. 再現コマンド')
    lines.append('')
    lines.append('```')
    lines.append('python -X utf8 src/a6_lmax_sweep.py')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/a6_lmax_sweep.py`*')
    lines.append('*関連ファイル: `CURRENT_BEST_STRATEGY.md`, `src/dynamic_leverage_strategies.py`*')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('A6: l_max Robustness Sweep — S2_VZGated')
    print(f'実行日: 2026-05-21')
    print('=' * 70)

    # S1: データロード
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} '
          f'({n:,} days, {n / TRADING_DAYS:.2f} years)')

    # S2: 共有資産（1回のみ生成）
    print('\nBuilding shared assets...')
    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print('  Shared assets done.')

    # S3: DH Dyn シグナル（1回のみ生成）
    print('Building DH Dyn signal...')
    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_years = n / TRADING_DAYS
    print(f'  DH Dyn signal: {n_tr} trades, {n_tr / n_years:.1f}/yr')

    # S4: l_max sweep
    print(f'\nSweeping l_max ∈ {L_MAX_SWEEP} ...')
    results = []

    for l_max in L_MAX_SWEEP:
        print(f'  l_max={l_max:.1f} ...', end=' ', flush=True)

        # レバレッジ系列を計算
        L_s2 = compute_L_s2_vz_gated(
            ret, vz,
            target_vol=S2_FIXED['target_vol'],
            k_vz=S2_FIXED['k_vz'],
            gate_min=S2_FIXED['gate_min'],
            n=S2_FIXED['n'],
            l_min=S2_FIXED['l_min'],
            l_max=l_max,
            step=S2_FIXED['step'],
        )

        # NAV 構築
        nav = build_nav_strategy(
            close, lev_A, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD',
            cfd_leverage=L_s2.values,
            cfd_spread=CFD_SPREAD_LOW,
        )

        # 標準 7 指標
        m = calc_7metrics(nav, dates, trades_per_year=n_tr / n_years)

        # Worst10Y★ (カレンダー年ベース)
        ann = nav_to_annual(nav, dates)
        r10 = rolling_nY_cagr(ann, 10)
        worst10y_star = float(r10.min()) if len(r10) > 0 else np.nan

        # P10_5Y / Worst5Y (日次ローリング)
        p10_5y  = compute_p10_5y(nav.values)
        worst5y = compute_worst5y(nav.values)

        is_oos_gap = m['CAGR_IS'] - m['CAGR_OOS']

        row = {
            'l_max':         l_max,
            'CAGR_IS':       m['CAGR_IS'],
            'CAGR_OOS':      m['CAGR_OOS'],
            'Sharpe_OOS':    m['Sharpe_OOS'],
            'MaxDD_FULL':    m['MaxDD_FULL'],
            'Worst10Y_star': worst10y_star,
            'P10_5Y':        p10_5y,
            'Worst5Y':       worst5y,
            'IS_OOS_gap':    is_oos_gap,
        }
        results.append(row)
        print(f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m["Sharpe_OOS"]:+.3f}')

    # S5: サニティチェック
    print()
    print('--- Sanity Check ---')
    row7 = next(r for r in results if abs(r['l_max'] - 7.0) < 1e-9)
    sanity_diff_pp = (row7['CAGR_OOS'] - REF_CAGR_OOS_LMAX7) * 100
    sanity_ok = abs(sanity_diff_pp) <= 0.50
    print(f'[SANITY] l_max=7.0 CAGR_OOS = {row7["CAGR_OOS"]*100:+.2f}%  '
          f'(ref +27.57%, diff {sanity_diff_pp:+.2f} pp)')
    if sanity_ok:
        print('  [OK] within 0.50 pp tolerance.')
    else:
        print('  [WARN] diff > 0.50 pp — 再現性に乖離あり。コードを見直すこと。')

    # S6: コンソール結果テーブル
    print()
    print('=' * 100)
    print('A6: l_max Robustness Sweep — S2_VZGated Results')
    print('=' * 100)
    hdr = (f'{"l_max":>6}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"MaxDD":>9}  {"Worst10Y★":>10}  {"P10_5Y":>8}  {"Worst5Y":>8}  {"IS-OOS gap":>11}')
    print(hdr)
    print('-' * 100)
    for r in results:
        tag = ' ← BEST' if abs(r['l_max'] - 7.0) < 1e-9 else '       '
        gap_pp = r['IS_OOS_gap'] * 100
        print(
            f'{r["l_max"]:>6.1f}{tag}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f}'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {r["Worst10Y_star"]*100:>+9.2f}%'
            f'  {r["P10_5Y"]*100:>+7.2f}%'
            f'  {r["Worst5Y"]*100:>+7.2f}%'
            f'  {gap_pp:>+10.2f} pp'
        )
    print('=' * 100)

    # S7: CSV 保存
    df_out = pd.DataFrame(results)
    csv_path = os.path.join(BASE, 'a6_lmax_sweep_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    # S8: Markdown レポート生成・保存
    md = generate_report(results, sanity_ok, sanity_diff_pp)
    md_path = os.path.join(BASE, 'A6_LMAX_SWEEP_2026-05-21.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
