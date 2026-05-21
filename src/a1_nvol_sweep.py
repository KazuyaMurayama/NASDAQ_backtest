"""
A1: n_vol Robustness Sweep — S2_VZGated
=========================================
目的: vol 計算窓 n ∈ {10, 20, 30, 40, 60} で S2_VZGated を再実行し、
     n=20 の選択感度と長期窓による過剰反応平滑化効果を定量確認する。

固定パラメータ: target_vol=0.8, k_vz=0.3, gate_min=0.5, l_min=1.0, l_max=7.0, step=0.5
可変: n (vol 計算窓)

出力:
  - コンソール結果テーブル
  - a1_nvol_sweep_results.csv  (プロジェクトルート)
  - A1_NVOL_SWEEP_2026-05-21.md (プロジェクトルート)
"""

import sys
import os
import types

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
    IS_START, IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import (
    prepare_gold_local,
    nav_to_annual,
    rolling_nY_cagr,
)

# ---------------------------------------------------------------------------
# A6 結果の参照値 (サニティチェック用: n=20 の CAGR_OOS)
# ---------------------------------------------------------------------------
REF_CAGR_OOS_N20 = 0.2751  # A6 sweep n=20 行の実測値 (2026-05-21)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A1 の固定パラメータ (n を除く)
S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, l_min=1.0, l_max=7.0, step=0.5)
N_SWEEP  = [10, 20, 30, 40, 60]   # int


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

def generate_report(results: list, sanity_ok: bool, sanity_diff_pp: float,
                    sanity_warn_forced: bool) -> str:
    lines = []
    lines.append('# A1: n_vol Robustness Sweep — S2_VZGated')
    lines.append('')
    lines.append('**実行日**: 2026-05-21')
    lines.append('**目的**: vol 計算窓 n ∈ {10, 20, 30, 40, 60} の sweep で n=20 の'
                 '選択感度と長期窓による過剰反応平滑化効果を定量確認する')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 1. セットアップ')
    lines.append('')
    lines.append('- **戦略**: S2_VZGated（Vol-Zone ゲート型 CFD レバレッジ）')
    lines.append('- **固定パラメータ**: target_vol=0.8, k_vz=0.3, gate_min=0.5, l_min=1.0, l_max=7.0, step=0.5')
    lines.append('- **可変パラメータ**: n（vol 計算窓） ∈ {10, 20, 30, 40, 60}')
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

    hdr = ('| n_vol | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD_FULL '
           '| Worst10Y★ | P10_5Y | Worst5Y | IS−OOS gap |')
    sep = ('|------:|--------:|---------:|-----------:|-----------:'
           '|----------:|-------:|--------:|-----------:|')
    lines.append(hdr)
    lines.append(sep)

    for r in results:
        tag = ' ← 現行ベスト' if r['n_vol'] == 20 else ''
        gap_pp = (r['CAGR_IS'] - r['CAGR_OOS']) * 100
        lines.append(
            f'| {r["n_vol"]}{tag} '
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
    row20 = next((r for r in results if r['n_vol'] == 20), None)
    if row20:
        sanity_tag = '✅ 一致（±0.10 pp 以内）' if sanity_ok else f'⚠️ 乖離 {sanity_diff_pp:+.2f} pp'
        lines.append(f'- n=20 の CAGR_OOS: **{row20["CAGR_OOS"]*100:+.2f}%**')
        lines.append(f'- A6 sweep 参照値 (n=20): **+27.51%**')
        lines.append(f'- CURRENT_BEST_STRATEGY.md 記載値: **+27.57%**（系統差 −0.06 pp は A6 確認済み）')
        lines.append(f'- 差分（対 A6 参照値）: **{sanity_diff_pp:+.2f} pp** → {sanity_tag}')
    if sanity_warn_forced:
        lines.append('- ⚠️ サニティ不一致により総合判定を強制 WARN 降格')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 4. 観察')
    lines.append('')

    # 当初仮説 1: 長期窓 Sharpe 改善
    sharpe_vals = {r['n_vol']: r['Sharpe_OOS'] for r in results}
    sharpe_long_vals = [sharpe_vals[n] for n in [40, 60] if n in sharpe_vals]
    sharpe_short_vals = [sharpe_vals[n] for n in [10, 20] if n in sharpe_vals]
    if sharpe_long_vals and sharpe_short_vals:
        delta_sharpe = np.mean(sharpe_long_vals) - np.mean(sharpe_short_vals)
        if delta_sharpe >= 0.05:
            sharpe_hyp = f'✅ 仮説支持（長期窓平均 Sharpe が {delta_sharpe:+.3f} 高い）'
        elif delta_sharpe <= -0.05:
            sharpe_hyp = f'❌ 仮説不成立（短期窓優位、差 {delta_sharpe:+.3f}）'
        else:
            sharpe_hyp = f'△ 中立（差 {delta_sharpe:+.3f}）'
        lines.append(f'- **長期窓 Sharpe 改善仮説**: {sharpe_hyp}')

    # 当初仮説 2: 長期窓 Worst5Y 改善
    w5_vals = {r['n_vol']: r['Worst5Y'] for r in results}
    w5_long = [w5_vals[n] for n in [40, 60] if n in w5_vals]
    w5_short = [w5_vals[n] for n in [10, 20] if n in w5_vals]
    if w5_long and w5_short:
        delta_w5 = (np.mean(w5_long) - np.mean(w5_short)) * 100
        w5_hyp = f'改善（+{delta_w5:.2f} pp）' if delta_w5 >= 1.0 else f'変化なし（{delta_w5:+.2f} pp）'
        lines.append(f'- **長期窓 Worst5Y 改善仮説**: {w5_hyp}')

    # n=20 中心性
    oos_vals = [r['CAGR_OOS'] for r in results]
    max_sharpe = max(r['Sharpe_OOS'] for r in results)
    max_oos = max(oos_vals)
    row20_sharpe = sharpe_vals.get(20, np.nan)
    row20_oos = row20['CAGR_OOS'] if row20 else np.nan
    if abs(row20_sharpe - max_sharpe) < 1e-9 and abs(row20_oos - max_oos) < 1e-9:
        center_label = 'n=20 は Sharpe_OOS・CAGR_OOS 双方の局所最適'
    elif abs(row20_sharpe - max_sharpe) < 1e-9:
        best_oos_n = results[oos_vals.index(max_oos)]['n_vol']
        center_label = f'n=20 は Sharpe_OOS 最高だが CAGR_OOS は n={best_oos_n} が最高'
    elif abs(row20_oos - max_oos) < 1e-9:
        best_sh_n = results[[r['Sharpe_OOS'] for r in results].index(max_sharpe)]['n_vol']
        center_label = f'n=20 は CAGR_OOS 最高だが Sharpe_OOS は n={best_sh_n} が最高'
    else:
        best_sh_n = results[[r['Sharpe_OOS'] for r in results].index(max_sharpe)]['n_vol']
        best_oos_n = results[oos_vals.index(max_oos)]['n_vol']
        center_label = f'n=20 は最適点ではない（Sharpe: n={best_sh_n}, CAGR_OOS: n={best_oos_n} が最高 → 要再評価）'
    lines.append(f'- **n=20 中心性**: {center_label}')

    # IS-OOS gap の単調性
    gaps = [(r['n_vol'], (r['CAGR_IS'] - r['CAGR_OOS']) * 100) for r in results]
    gaps_sorted = sorted(gaps, key=lambda x: x[0])
    gap_diffs = [gaps_sorted[i+1][1] - gaps_sorted[i][1] for i in range(len(gaps_sorted)-1)]
    increasing_pairs = sum(1 for d in gap_diffs if d > 0)
    if increasing_pairs == 0:
        lines.append('- IS−OOS gap は n 増加に対して**単調減少**（過剰平滑化なし）。')
    elif increasing_pairs <= 1:
        lines.append('- IS−OOS gap は概ね n 増加に対して安定（1 ペアのみ上昇）。')
    else:
        lines.append(f'- IS−OOS gap は n 増加で {increasing_pairs}/4 ペアが増加（過剰平滑化の兆候）。')

    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 5. 判定（n=20 の選択ロバストネス）')
    lines.append('')
    lines.append('### 判定基準（事前定義）')
    lines.append('')
    lines.append('| 基準 | 条件 | 結果 |')
    lines.append('|------|------|------|')

    # 基準 (i): n=20 の Sharpe_OOS が最高または最高値との差 ≤ 0.020
    sharpe20 = sharpe_vals.get(20, np.nan)
    diff_sharpe_from_best = max_sharpe - sharpe20
    crit1 = diff_sharpe_from_best <= 0.020
    lines.append(f'| (i) Sharpe_OOS: n=20 と最高値の差 ≤ 0.020 | diff = {diff_sharpe_from_best:.3f} | {"✅ PASS" if crit1 else "⚠️ WARN"} |')

    # 基準 (ii): n=40 の CAGR_OOS が n=20 ±1.5 pp 以内
    cagr20 = row20['CAGR_OOS'] if row20 else np.nan
    row40 = next((r for r in results if r['n_vol'] == 40), None)
    cagr40 = row40['CAGR_OOS'] if row40 else np.nan
    diff_cagr_40_20 = abs(cagr40 - cagr20) * 100 if not np.isnan(cagr40) and not np.isnan(cagr20) else np.nan
    crit2 = diff_cagr_40_20 <= 1.5 if not np.isnan(diff_cagr_40_20) else False
    lines.append(f'| (ii) n=40 CAGR_OOS が n=20 ±1.5 pp 以内 | diff = {diff_cagr_40_20:.2f} pp | {"✅ PASS" if crit2 else "⚠️ WARN"} |')

    # 基準 (iii): IS-OOS gap の非増大（増加ペア ≤ 1/4）
    crit3 = increasing_pairs <= 1
    lines.append(f'| (iii) IS−OOS gap が n 増加で大幅増大しない（増加ペア ≤ 1/4） | 増加ペア = {increasing_pairs}/4 | {"✅ PASS" if crit3 else "⚠️ WARN"} |')
    lines.append('')

    all_pass = crit1 and crit2 and crit3
    verdict = 'PASS' if (all_pass and not sanity_warn_forced) else ('WARN' if (crit1 or crit2) else 'FAIL')
    if sanity_warn_forced:
        verdict = 'WARN'

    lines.append(f'**総合判定: {verdict}**')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 6. 結論')
    lines.append('')
    if verdict == 'PASS':
        lines.append('vol 窓 **n=20 はロバストな選択**。n=40/60 への変更による Sharpe 改善は確認されず、'
                     '現行設定を維持する。A6 と合わせて S2_VZGated の n=20, l_max=7 パラメータが'
                     'インサンプル最適化バイアスなしと確認された。')
    elif verdict == 'WARN':
        lines.append('n=40 で Sharpe 改善の兆候あり、または基準の一部で境界的な結果。'
                     'A1 単独では確定せず、k_vz と n の交差 sweep（A2 候補）などの追加検証を推奨。')
    else:
        best_n = results[[r['Sharpe_OOS'] for r in results].index(max_sharpe)]['n_vol']
        lines.append(f'n=20 はロバストでない。最高 Sharpe を示した n={best_n} を新ベスト候補として'
                     '追加検証を実施すること。CURRENT_BEST_STRATEGY.md の更新前に OOS 検証が必要。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 7. 再現コマンド')
    lines.append('')
    lines.append('```')
    lines.append('python -X utf8 src/a1_nvol_sweep.py')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/a1_nvol_sweep.py`*')
    lines.append('*参照: `A6_LMAX_SWEEP_2026-05-21.md`, `CURRENT_BEST_STRATEGY.md`*')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('A1: n_vol Robustness Sweep — S2_VZGated')
    print('実行日: 2026-05-21')
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

    # S4: n sweep
    print(f'\nSweeping n_vol ∈ {N_SWEEP} ...')
    results = []

    for n_vol in N_SWEEP:
        print(f'  n_vol={n_vol:2d} ...', end=' ', flush=True)

        # レバレッジ系列を計算 (n はint で渡す)
        L_s2 = compute_L_s2_vz_gated(
            ret, vz,
            target_vol=S2_FIXED['target_vol'],
            k_vz=S2_FIXED['k_vz'],
            gate_min=S2_FIXED['gate_min'],
            n=n_vol,
            l_min=S2_FIXED['l_min'],
            l_max=S2_FIXED['l_max'],
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

        row = {
            'n_vol':         n_vol,
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

    # S5: サニティチェック
    print()
    print('--- Sanity Check ---')
    row20 = next(r for r in results if r['n_vol'] == 20)
    sanity_diff_pp = (row20['CAGR_OOS'] - REF_CAGR_OOS_N20) * 100
    sanity_ok = abs(sanity_diff_pp) <= 0.10
    sanity_warn_forced = not sanity_ok
    print(f'[SANITY] n=20 CAGR_OOS = {row20["CAGR_OOS"]*100:+.2f}%  '
          f'(ref +27.51%, diff {sanity_diff_pp:+.2f} pp)')
    if sanity_ok:
        print('  [OK] within 0.10 pp tolerance.')
    else:
        print('  [WARN] diff > 0.10 pp — DH Dynシグナル/データ更新を確認。総合判定を強制 WARN 降格。')

    # S6: コンソール結果テーブル
    print()
    print('=' * 100)
    print('A1: n_vol Robustness Sweep — S2_VZGated Results')
    print('=' * 100)
    hdr = (f'{"n_vol":>5}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"MaxDD":>9}  {"Worst10Y★":>10}  {"P10_5Y":>8}  {"Worst5Y":>8}  {"IS-OOS gap":>11}')
    print(hdr)
    print('-' * 100)
    for r in results:
        tag = ' ← BASE' if r['n_vol'] == 20 else '       '
        gap_pp = r['IS_OOS_gap'] * 100
        print(
            f'{r["n_vol"]:>5d}{tag}'
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
    csv_path = os.path.join(BASE, 'a1_nvol_sweep_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    # S8: Markdown レポート生成・保存
    md = generate_report(results, sanity_ok, sanity_diff_pp, sanity_warn_forced)
    md_path = os.path.join(BASE, 'A1_NVOL_SWEEP_2026-05-21.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
