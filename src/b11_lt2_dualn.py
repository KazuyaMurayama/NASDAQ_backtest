"""
B11: LT2 Dual-N 加重平均スイープ
==================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-23)

lt_bias_combo = w * bias_N750 + (1-w) * bias_N1500
グリッド:
  w     ∈ {0.00, 0.25, 0.50, 0.75, 1.00}
  k_lt  ∈ {0.5, 0.7}
  実行行: 8行（REF×2 + Combo×6）

REF1 (w=1.00, k=0.5) = N750 現行ベスト: Sharpe_OOS +0.858, CAGR_OOS +31.16%
REF2 (w=0.00, k=0.5) = N1500: Sharpe_OOS +0.885, CAGR_OOS +30.84%

出力:
  - b11_lt2_dualn_results.csv
  - B11_LT2_DUALN_2026-05-23.md
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
from long_cycle_signal import build_lt_signal, signal_to_bias, apply_lt_mode_b
from _sweep_format import (
    MD_HEADER_2P,
    MD_WFA_NOTE,
    _fp1, _ff2, _gap_pp, _tr, _wfa,
)

# ---------------------------------------------------------------------------
# 参照値
# ---------------------------------------------------------------------------
REF_N750_SHARPE_OOS  = 0.858
REF_N750_CAGR_OOS    = 0.3116
REF_N1500_SHARPE_OOS = 0.885
REF_N1500_CAGR_OOS   = 0.3084

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 固定パラメータ
S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)

# PASS判定基準（事前定義）
PASS_SHARPE_OOS_MIN   = 0.878   # REF_N750 + 0.020
PASS_CAGR_OOS_MIN     = 0.2916  # +29.16%
PASS_IS_OOS_GAP_MAX   = 0.0300  # +3.0pp
PASS_MAXDD_MIN        = -0.6445 # > -64.45%
PASS_WORST10Y_MIN     = 0.150   # >= +15.0%

# グリッド定義
GRID = [
    dict(w=1.00, k_lt=0.5, label='REF1 (w=1.00, k=0.5)'),   # pure N750 = 現行ベスト
    dict(w=0.75, k_lt=0.5, label='Combo w=0.75, k=0.5'),
    dict(w=0.50, k_lt=0.5, label='Combo w=0.50, k=0.5'),
    dict(w=0.25, k_lt=0.5, label='Combo w=0.25, k=0.5'),
    dict(w=0.00, k_lt=0.5, label='REF2 (w=0.00, k=0.5)'),   # pure N1500
    dict(w=0.75, k_lt=0.7, label='Combo w=0.75, k=0.7'),
    dict(w=0.50, k_lt=0.7, label='Combo w=0.50, k=0.7'),
    dict(w=0.25, k_lt=0.7, label='Combo w=0.25, k=0.7'),
]


# ---------------------------------------------------------------------------
# ヘルパー: P10_5Y / Worst5Y
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
# 指標計算 (9指標)
# ---------------------------------------------------------------------------

def calc_all_metrics(nav, dates, trades_per_year):
    m = calc_7metrics(nav, dates, trades_per_year=trades_per_year)
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, 10)
    worst10y_star = float(r10.min()) if len(r10) > 0 else np.nan
    p10_5y  = compute_p10_5y(nav.values)
    worst5y = compute_worst5y(nav.values)
    return {
        **m,
        'Worst10Y_star': worst10y_star,
        'P10_5Y':        p10_5y,
        'Worst5Y':       worst5y,
        'IS_OOS_gap':    m['CAGR_IS'] - m['CAGR_OOS'],
        'Trades_yr':     trades_per_year,
    }


# ---------------------------------------------------------------------------
# PASS判定
# ---------------------------------------------------------------------------

def check_pass(r):
    crit_i   = r['Sharpe_OOS']    >= PASS_SHARPE_OOS_MIN
    crit_ii  = r['CAGR_OOS']      >= PASS_CAGR_OOS_MIN
    crit_iii = r['IS_OOS_gap']    <= PASS_IS_OOS_GAP_MAX
    crit_iv  = r['MaxDD_FULL']    > PASS_MAXDD_MIN
    crit_v   = r['Worst10Y_star'] >= PASS_WORST10Y_MIN
    all_pass = crit_i and crit_ii and crit_iii and crit_iv and crit_v
    any_fail = not (crit_i or crit_ii or crit_iii or crit_iv or crit_v)
    verdict = 'PASS' if all_pass else ('FAIL' if any_fail else 'WARN')
    return verdict, dict(i=crit_i, ii=crit_ii, iii=crit_iii, iv=crit_iv, v=crit_v)


# ---------------------------------------------------------------------------
# Markdownレポート生成
# ---------------------------------------------------------------------------

def generate_report(results: list) -> str:
    lines = []
    lines.append('# B11: LT2 Dual-N 加重平均スイープ')
    lines.append('')
    lines.append('**実行日**: 2026-05-23')
    lines.append('**準拠**: EVALUATION_STANDARD §3.12 (v1.1)')
    lines.append('')
    lines.append('## 概要')
    lines.append('')
    lines.append('N750 (Sharpe 0.858, IS-OOS gap +0.18pp) vs N1500 (Sharpe 0.885, IS-OOS gap -0.05pp)')
    lines.append('のトレードオフを線形補間で解消する最良点を探す。')
    lines.append('')
    lines.append('```')
    lines.append('lt_bias_combo = w * bias_N750 + (1-w) * bias_N1500')
    lines.append('lev_mod = apply_lt_mode_b(lev_A, lt_bias_combo, l_min=0.0, l_max=1.0)')
    lines.append('```')
    lines.append('')
    lines.append('**PASS判定基準（事前定義）**:')
    lines.append('- (i) Sharpe_OOS ≥ 0.878 (REF_N750 +0.020)')
    lines.append('- (ii) CAGR_OOS ≥ +29.16%')
    lines.append('- (iii) IS-OOS gap ≤ +3.0pp')
    lines.append('- (iv) MaxDD > -64.45%')
    lines.append('- (v) Worst10Y★ ≥ +15.0%')
    lines.append('')
    lines.append(f'- **IS 期間**: {IS_START} 〜 {IS_END}')
    lines.append(f'- **OOS 期間**: {OOS_START} 〜')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 結果テーブル')
    lines.append('')

    # k_lt=0.5 ブロック
    lines.append('### k_lt = 0.5 ブロック')
    lines.append('')
    lines.append('| w | k_lt | CAGR<br>_OOS | Sharpe | MaxDD | Worst<br>10Y★ | P10▷ | IS-OOS<br>gap | Tr | CI95<br>_lo | WFE |')
    lines.append('|--:|-----:|-------------:|-------:|------:|--------------:|-----:|--------------:|---:|-----------:|----:|')
    for r in results:
        if r['k_lt'] != 0.5:
            continue
        marker = ''
        if r['label'].startswith('REF1'):
            marker = ' ← REF(N750)'
        elif r['label'].startswith('REF2'):
            marker = ' ← REF(N1500)'
        verdict, _ = check_pass(r)
        pass_mark = ' **PASS**' if verdict == 'PASS' else (' WARN' if verdict == 'WARN' else ' FAIL')
        sharpe_mark = ' ★' if r['Sharpe_OOS'] > 0.885 else (' ◎' if r['Sharpe_OOS'] > 0.770 else '')
        lines.append(
            f'| {r["w"]:.2f} | {r["k_lt"]:.1f} '
            f'| {_fp1(r["CAGR_OOS"])} '
            f'| {_ff2(r["Sharpe_OOS"])}{sharpe_mark}{pass_mark} '
            f'| {_fp1(r["MaxDD_FULL"])} '
            f'| {_fp1(r["Worst10Y_star"])} '
            f'| {_fp1(r["P10_5Y"])} '
            f'| {_gap_pp(r["IS_OOS_gap"])} '
            f'| {_tr(r.get("Trades_yr"))} '
            f'| {_wfa(r.get("WFA_CI95_lo"))} '
            f'| {_wfa(r.get("WFA_WFE"))} |{marker}'
        )
    lines.append('')

    # k_lt=0.7 ブロック
    lines.append('### k_lt = 0.7 ブロック')
    lines.append('')
    lines.append('| w | k_lt | CAGR<br>_OOS | Sharpe | MaxDD | Worst<br>10Y★ | P10▷ | IS-OOS<br>gap | Tr | CI95<br>_lo | WFE |')
    lines.append('|--:|-----:|-------------:|-------:|------:|--------------:|-----:|--------------:|---:|-----------:|----:|')
    for r in results:
        if r['k_lt'] != 0.7:
            continue
        verdict, _ = check_pass(r)
        pass_mark = ' **PASS**' if verdict == 'PASS' else (' WARN' if verdict == 'WARN' else ' FAIL')
        sharpe_mark = ' ★' if r['Sharpe_OOS'] > 0.885 else (' ◎' if r['Sharpe_OOS'] > 0.770 else '')
        lines.append(
            f'| {r["w"]:.2f} | {r["k_lt"]:.1f} '
            f'| {_fp1(r["CAGR_OOS"])} '
            f'| {_ff2(r["Sharpe_OOS"])}{sharpe_mark}{pass_mark} '
            f'| {_fp1(r["MaxDD_FULL"])} '
            f'| {_fp1(r["Worst10Y_star"])} '
            f'| {_fp1(r["P10_5Y"])} '
            f'| {_gap_pp(r["IS_OOS_gap"])} '
            f'| {_tr(r.get("Trades_yr"))} '
            f'| {_wfa(r.get("WFA_CI95_lo"))} '
            f'| {_wfa(r.get("WFA_WFE"))} |'
        )
    lines.append('')
    lines.append(MD_WFA_NOTE)
    lines.append('')
    lines.append('---')
    lines.append('')

    # サニティチェック
    ref1 = next((r for r in results if r['label'].startswith('REF1')), None)
    ref2 = next((r for r in results if r['label'].startswith('REF2')), None)
    lines.append('## サニティチェック')
    lines.append('')
    if ref1:
        diff_sharpe = ref1['Sharpe_OOS'] - REF_N750_SHARPE_OOS
        diff_cagr   = (ref1['CAGR_OOS'] - REF_N750_CAGR_OOS) * 100
        ok_sharpe = abs(diff_sharpe) <= 0.020
        ok_cagr   = abs(diff_cagr) <= 0.10
        tag = '✅ OK' if (ok_sharpe and ok_cagr) else '⚠️ WARN'
        lines.append(f'- **REF1** (w=1.00, k=0.5): Sharpe_OOS={ref1["Sharpe_OOS"]:+.3f} '
                     f'(ref +{REF_N750_SHARPE_OOS:.3f}, diff {diff_sharpe:+.3f}) | '
                     f'CAGR_OOS={ref1["CAGR_OOS"]*100:+.2f}% '
                     f'(ref +{REF_N750_CAGR_OOS*100:.2f}%, diff {diff_cagr:+.2f} pp) → {tag}')
    if ref2:
        diff_sharpe2 = ref2['Sharpe_OOS'] - REF_N1500_SHARPE_OOS
        diff_cagr2   = (ref2['CAGR_OOS'] - REF_N1500_CAGR_OOS) * 100
        ok2 = abs(diff_sharpe2) <= 0.020 and abs(diff_cagr2) <= 0.10
        tag2 = '✅ OK' if ok2 else '⚠️ WARN'
        lines.append(f'- **REF2** (w=0.00, k=0.5): Sharpe_OOS={ref2["Sharpe_OOS"]:+.3f} '
                     f'(ref +{REF_N1500_SHARPE_OOS:.3f}, diff {diff_sharpe2:+.3f}) | '
                     f'CAGR_OOS={ref2["CAGR_OOS"]*100:+.2f}% '
                     f'(ref +{REF_N1500_CAGR_OOS*100:.2f}%, diff {diff_cagr2:+.2f} pp) → {tag2}')
    lines.append('')
    lines.append('---')
    lines.append('')

    # 各行の判定サマリー
    lines.append('## PASS判定サマリー')
    lines.append('')
    lines.append('| label | (i) Sharpe≥0.878 | (ii) CAGR≥29.16% | (iii) Gap≤3pp | (iv) DD>-64.45% | (v) W10Y≥15% | 判定 |')
    lines.append('|-------|:----------------:|:----------------:|:-------------:|:---------------:|:------------:|:----:|')
    pass_count = 0
    best_sharpe = -999
    best_row = None
    for r in results:
        verdict, crits = check_pass(r)
        if verdict == 'PASS':
            pass_count += 1
        if r['Sharpe_OOS'] > best_sharpe:
            best_sharpe = r['Sharpe_OOS']
            best_row = r
        ci  = '✅' if crits['i']   else '❌'
        cii = '✅' if crits['ii']  else '❌'
        ciii= '✅' if crits['iii'] else '❌'
        civ = '✅' if crits['iv']  else '❌'
        cv  = '✅' if crits['v']   else '❌'
        lines.append(f'| {r["label"]} | {ci} | {cii} | {ciii} | {civ} | {cv} | **{verdict}** |')
    lines.append('')

    # 総合判定
    if pass_count > 0:
        overall = 'PASS'
        conclusion = f'{pass_count}行がPASS。Dual-N補間は有効。最高Sharpe_OOS = {best_sharpe:.3f} ({best_row["label"]})。'
    else:
        # PASS基準を満たしているか確認（WARNが多数あれば WARN, 全FAILならFAIL）
        warn_count = sum(1 for r in results if check_pass(r)[0] == 'WARN')
        if warn_count > 0:
            overall = 'WARN'
            conclusion = f'PASSなし。{warn_count}行がWARN。Dual-N補間は部分的効果あり。最高Sharpe_OOS = {best_sharpe:.3f} ({best_row["label"]})。'
        else:
            overall = 'FAIL'
            conclusion = f'全行FAIL。Dual-N補間は改善なし。現行ベスト(N750)を維持推奨。'

    lines.append('---')
    lines.append('')
    lines.append('## 結論')
    lines.append('')
    lines.append(f'**総合判定: {overall}**')
    lines.append('')
    lines.append(conclusion)
    lines.append('')
    if best_row:
        lines.append(f'- 最高Sharpe_OOS: **{best_sharpe:.3f}** (w={best_row["w"]:.2f}, k_lt={best_row["k_lt"]:.1f})')
        lines.append(f'- CAGR_OOS: {best_row["CAGR_OOS"]*100:+.2f}%')
        lines.append(f'- IS-OOS gap: {best_row["IS_OOS_gap"]*100:+.2f}pp')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 再現コマンド')
    lines.append('')
    lines.append('```')
    lines.append('python -X utf8 src/b11_lt2_dualn.py')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/b11_lt2_dualn.py`*')
    lines.append('*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md §3.12`*')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('B11: LT2 Dual-N 加重平均スイープ')
    print('実行日: 2026-05-23')
    print('=' * 70)

    # S1: データロード
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} '
          f'({n:,} days, {n_years:.2f} years)')

    # S2: 共有アセット（1回のみ生成）
    print('\nBuilding shared assets...')
    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print('  Shared assets done.')

    # S3: DH Dynシグナル（1回のみ）
    print('Building DH Dyn signal...')
    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    trades_per_year = n_tr / n_years
    print(f'  DH Dyn signal: {n_tr} trades, {trades_per_year:.1f}/yr')

    # S4: S2_VZGated CFDレバ系列（1回のみ）
    print('\nBuilding S2 CFD leverage series...')
    L_s2 = compute_L_s2_vz_gated(
        ret, vz,
        target_vol=S2_FIXED['target_vol'],
        k_vz=S2_FIXED['k_vz'],
        gate_min=S2_FIXED['gate_min'],
        n=S2_FIXED['n'],
        l_min=S2_FIXED['l_min'],
        l_max=S2_FIXED['l_max'],
        step=S2_FIXED['step'],
    )
    print(f'  L_s2 range: [{L_s2.min():.1f}, {L_s2.max():.1f}]  median={L_s2.median():.1f}')

    # S5: LT2シグナル両N（1回のみ）
    print('\nBuilding LT2 signals for N=750 and N=1500...')
    lt_sig_750  = build_lt_signal(close, 'LT2', N=750)
    lt_sig_1500 = build_lt_signal(close, 'LT2', N=1500)
    print(f'  lt_sig_750 range:  [{lt_sig_750.min():+.3f}, {lt_sig_750.max():+.3f}]')
    print(f'  lt_sig_1500 range: [{lt_sig_1500.min():+.3f}, {lt_sig_1500.max():+.3f}]')

    # S6: グリッドスイープ
    results = []
    prev_k_lt = None

    for idx, row_cfg in enumerate(GRID):
        w    = row_cfg['w']
        k_lt = row_cfg['k_lt']
        label = row_cfg['label']

        print(f'\n[{idx+1}/{len(GRID)}] {label}  (w={w:.2f}, k_lt={k_lt:.1f})')

        # k_ltが変わった場合はbias系列を再計算
        if k_lt != prev_k_lt:
            bias_750  = signal_to_bias(lt_sig_750,  k_lt)
            bias_1500 = signal_to_bias(lt_sig_1500, k_lt)
            prev_k_lt = k_lt
            print(f'  Recomputed bias for k_lt={k_lt:.1f}')

        # 加重平均バイアス
        lt_bias_combo = w * bias_750 + (1.0 - w) * bias_1500
        lev_mod = apply_lt_mode_b(lev_A, lt_bias_combo, l_min=0.0, l_max=1.0)

        # NAV構築
        nav = build_nav_strategy(
            close, lev_mod, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
        )

        # 指標計算
        m = calc_all_metrics(nav, dates, trades_per_year)
        m['w']     = w
        m['k_lt']  = k_lt
        m['label'] = label
        m['WFA_CI95_lo'] = np.nan
        m['WFA_WFE']     = np.nan

        results.append(m)

        verdict, crits = check_pass(m)
        print(f'  CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
              f'Sharpe_OOS={m["Sharpe_OOS"]:+.3f}  '
              f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
              f'Worst10Y★={m["Worst10Y_star"]*100:+.2f}%  '
              f'Gap={m["IS_OOS_gap"]*100:+.2f}pp  → {verdict}')

    # S7: サニティチェック
    print()
    print('--- Sanity Check ---')
    ref1 = next((r for r in results if r['label'].startswith('REF1')), None)
    ref2 = next((r for r in results if r['label'].startswith('REF2')), None)
    if ref1:
        diff_sharpe = ref1['Sharpe_OOS'] - REF_N750_SHARPE_OOS
        diff_cagr   = (ref1['CAGR_OOS'] - REF_N750_CAGR_OOS) * 100
        tag = 'OK' if (abs(diff_sharpe) <= 0.020 and abs(diff_cagr) <= 0.10) else 'WARN'
        print(f'[SANITY REF1] Sharpe_OOS={ref1["Sharpe_OOS"]:+.3f} '
              f'(ref +{REF_N750_SHARPE_OOS:.3f}, diff {diff_sharpe:+.3f})  '
              f'CAGR_OOS={ref1["CAGR_OOS"]*100:+.2f}% '
              f'(ref +{REF_N750_CAGR_OOS*100:.2f}%, diff {diff_cagr:+.2f}pp) [{tag}]')
    if ref2:
        diff_sharpe2 = ref2['Sharpe_OOS'] - REF_N1500_SHARPE_OOS
        diff_cagr2   = (ref2['CAGR_OOS'] - REF_N1500_CAGR_OOS) * 100
        tag2 = 'OK' if (abs(diff_sharpe2) <= 0.020 and abs(diff_cagr2) <= 0.10) else 'WARN'
        print(f'[SANITY REF2] Sharpe_OOS={ref2["Sharpe_OOS"]:+.3f} '
              f'(ref +{REF_N1500_SHARPE_OOS:.3f}, diff {diff_sharpe2:+.3f})  '
              f'CAGR_OOS={ref2["CAGR_OOS"]*100:+.2f}% '
              f'(ref +{REF_N1500_CAGR_OOS*100:.2f}%, diff {diff_cagr2:+.2f}pp) [{tag2}]')

    # S8: コンソール結果テーブル
    print()
    print('=' * 120)
    print('B11: LT2 Dual-N Sweep Results')
    print('=' * 120)
    hdr = (f'{"label":<35}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"MaxDD":>9}  {"Worst10Y★":>10}  {"P10_5Y":>8}  {"IS-OOS gap":>11}  {"Trades/yr":>10}')
    print(hdr)
    print('-' * 120)
    for r in results:
        verdict, _ = check_pass(r)
        print(
            f'{r["label"]:<35}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f}'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {r["Worst10Y_star"]*100:>+9.2f}%'
            f'  {r["P10_5Y"]*100:>+7.2f}%'
            f'  {r["IS_OOS_gap"]*100:>+10.2f} pp'
            f'  {r["Trades_yr"]:>9.1f}'
            f'  [{verdict}]'
        )
    print('=' * 120)

    # PASS集計
    pass_count = sum(1 for r in results if check_pass(r)[0] == 'PASS')
    best_row = max(results, key=lambda r: r['Sharpe_OOS'])
    print(f'\nPASS count: {pass_count} / {len(results)}')
    print(f'Best Sharpe_OOS: {best_row["Sharpe_OOS"]:+.3f} ({best_row["label"]})')

    # S9: CSV保存
    df_out = pd.DataFrame([{
        'w':            r['w'],
        'k_lt':         r['k_lt'],
        'label':        r['label'],
        'CAGR_IS':      r['CAGR_IS'],
        'CAGR_OOS':     r['CAGR_OOS'],
        'Sharpe_OOS':   r['Sharpe_OOS'],
        'MaxDD_FULL':   r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'],
        'P10_5Y':       r['P10_5Y'],
        'IS_OOS_gap':   r['IS_OOS_gap'],
        'Trades_yr':    r['Trades_yr'],
    } for r in results])
    csv_path = os.path.join(BASE, 'b11_lt2_dualn_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    # S10: Markdownレポート生成・保存
    md = generate_report(results)
    md_path = os.path.join(BASE, 'B11_LT2_DUALN_2026-05-23.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
