"""
B2: k_lt スイープ — LT2 バイアス強度の最適化
=============================================
目的: k_lt ∈ {0.3, 0.5, 0.7, 1.0} をスイープし、S2+LT2 における
     LT2 バイアス強度 k_lt=0.5 (B1 確定値) が最適かどうかを確認する。

サニティ:
  k_lt=0.5 の CAGR_OOS は B1 結果 +31.16% に ±0.10pp 以内で一致する必要がある。

事前定義判定基準 (PASS):
  (i)  最良 k の Sharpe_OOS  ≥ k=0.5 の 0.858 + 0.020
  (ii) 最良 k の CAGR_OOS    ≥ +29.16% (k=0.5 比 ≥ -2.0pp)
  (iii) 最良 k の IS-OOS gap ≤ +3.0pp
  (iv) 最良 k の MaxDD       > -64.45% (k=0.5 比 ≤ +5.0pp 悪化)

出力:
  - b2_klt_sweep_results.csv
  - B2_KLT_SWEEP_2026-05-21.md
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

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
K_LT_GRID = [0.3, 0.5, 0.7, 1.0]

# サニティ: k=0.5 の CAGR_OOS (B1 確定値)
REF_CAGR_OOS_K05   = 0.3116    # B1 (2) S2+LT2 CAGR_OOS
REF_SHARPE_OOS_K05 = 0.858     # B1 (2) S2+LT2 Sharpe_OOS
REF_MAXDD_K05      = -0.5945   # B1 (2) S2+LT2 MaxDD_FULL

# S2 固定パラメータ
S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)
LT2_N = 750

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

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
    p10_5y  = compute_p10_5y(nav.values)
    worst5y = compute_worst5y(nav.values)
    return {
        **m,
        'Worst10Y_star': worst10y_star,
        'P10_5Y':        p10_5y,
        'Worst5Y':       worst5y,
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


# ---------------------------------------------------------------------------
# Markdown レポート生成
# ---------------------------------------------------------------------------

def generate_report(results: list, sanity_ok: bool, sanity_diff_pp: float,
                    sanity_warn_forced: bool, best_k: float) -> str:
    today = '2026-05-21'
    lines = []
    lines.append('# B2: k_lt スイープ — LT2 バイアス強度の最適化')
    lines.append('')
    lines.append(f'**実行日**: {today}')
    lines.append('**目的**: LT2 バイアス強度 k_lt を {0.3, 0.5(REF), 0.7, 1.0} でスイープし、'
                 'k_lt=0.5 (B1 確定値) が最適かどうかを確認する')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 1. セットアップ')
    lines.append('')
    lines.append('### スイープ設定')
    lines.append('')
    lines.append(f'- **k_lt グリッド**: {K_LT_GRID}')
    lines.append(f'- **LT2 N**: {LT2_N} 日')
    lines.append('- **S2 固定パラメータ**: target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0')
    lines.append('- **CFD スプレッド**: CFD_SPREAD_LOW')
    lines.append(f'- **IS 期間**: {IS_START} 〜 {IS_END}')
    lines.append(f'- **OOS 期間**: {OOS_START} 〜')
    lines.append('')
    lines.append('### lt_bias の式')
    lines.append('')
    lines.append('```')
    lines.append('lt_sig  = compute_lt2(close, N=750)')
    lines.append('lt_bias = signal_to_bias(lt_sig, k_lt)  # = (-k_lt * 0.5 * z).clip(-0.5, +0.5)')
    lines.append('lev_mod = clip(lev_A + lt_bias, 0, 1)')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 2. 結果テーブル')
    lines.append('')
    hdr = ('| k_lt | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD_FULL '
           '| Worst10Y★ | P10_5Y | Worst5Y | IS−OOS gap | 取引/年 |')
    sep = ('|-----:|--------:|---------:|-----------:|-----------:'
           '|----------:|-------:|--------:|-----------:|-------:|')
    lines.append(hdr)
    lines.append(sep)
    for r in results:
        marker = ' ← REF' if abs(r['k_lt'] - 0.5) < 1e-9 else ''
        marker_best = ' ← **BEST**' if abs(r['k_lt'] - best_k) < 1e-9 and abs(r['k_lt'] - 0.5) > 1e-9 else ''
        lines.append(
            f'| {r["k_lt"]:.1f}{marker}{marker_best} '
            f'| {_fp(r["CAGR_IS"])} '
            f'| {_fp(r["CAGR_OOS"])} '
            f'| {_ff(r["Sharpe_OOS"])} '
            f'| {_fp(r["MaxDD_FULL"])} '
            f'| {_fp(r["Worst10Y_star"])} '
            f'| {_fp(r["P10_5Y"])} '
            f'| {_fp(r["Worst5Y"])} '
            f'| {(r["IS_OOS_gap"])*100:+.2f} pp '
            f'| {r["n_trades_yr"]:.1f} |'
        )
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 3. サニティチェック')
    lines.append('')
    r_k05 = next((r for r in results if abs(r['k_lt'] - 0.5) < 1e-9), None)
    if r_k05:
        sanity_tag = '✅ 一致（±0.10 pp 以内）' if sanity_ok else f'⚠️ 乖離 {sanity_diff_pp:+.2f} pp'
        lines.append(f'- k_lt=0.5 CAGR_OOS: **{r_k05["CAGR_OOS"]*100:+.2f}%**')
        lines.append(f'- B1 参照値: **+31.16%**')
        lines.append(f'- 差分: **{sanity_diff_pp:+.2f} pp** → {sanity_tag}')
    if sanity_warn_forced:
        lines.append('- ⚠️ サニティ不一致により総合判定を強制 WARN 降格')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 4. 判定（B2 有効性評価）')
    lines.append('')
    lines.append('### 判定基準（事前定義）')
    lines.append('')
    lines.append('| 基準 | 条件 | 結果 |')
    lines.append('|------|------|------|')

    r_best = next((r for r in results if abs(r['k_lt'] - best_k) < 1e-9), None)
    if r_k05 and r_best:
        sharpe_delta = r_best['Sharpe_OOS'] - REF_SHARPE_OOS_K05
        cagr_oos_best = r_best['CAGR_OOS']
        gap_best = r_best['IS_OOS_gap']
        maxdd_best = r_best['MaxDD_FULL']

        crit1 = sharpe_delta >= 0.020
        crit2 = cagr_oos_best >= 0.2916
        crit3 = gap_best <= 0.030
        crit4 = maxdd_best > -0.6445

        lines.append(f'| (i) 最良 k の Sharpe_OOS ≥ REF+0.020 | '
                     f'Δ = {sharpe_delta:+.3f} (best={r_best["Sharpe_OOS"]:+.3f}, REF=0.858) | '
                     f'{"✅ PASS" if crit1 else "❌ FAIL"} |')
        lines.append(f'| (ii) 最良 k の CAGR_OOS ≥ +29.16% | '
                     f'{cagr_oos_best*100:+.2f}% | '
                     f'{"✅ PASS" if crit2 else "❌ FAIL"} |')
        lines.append(f'| (iii) 最良 k の IS-OOS gap ≤ +3.0pp | '
                     f'{gap_best*100:+.2f} pp | '
                     f'{"✅ PASS" if crit3 else "❌ FAIL"} |')
        lines.append(f'| (iv) 最良 k の MaxDD > -64.45% | '
                     f'{maxdd_best*100:+.2f}% | '
                     f'{"✅ PASS" if crit4 else "❌ FAIL"} |')
        lines.append('')

        if sanity_warn_forced:
            verdict = 'WARN (サニティ不一致)'
        elif crit1 and crit2 and crit3 and crit4:
            verdict = f'PASS — k_lt={best_k:.1f} を新ベストに採用'
        elif crit2 and crit3 and crit4:
            verdict = f'WARN — Sharpe改善不足。k_lt=0.5 を維持'
        else:
            verdict = f'FAIL — k_lt=0.5 を維持'
    else:
        crit1 = crit2 = crit3 = crit4 = False
        verdict = 'WARN'

    lines.append(f'**総合判定: {verdict}**')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 5. 考察')
    lines.append('')
    lines.append('### k_lt と lt_bias の関係')
    lines.append('')
    lines.append('| k_lt | lt_bias 式 | バイアス最大幅 |')
    lines.append('|-----:|-----------|------------|')
    lines.append('| 0.3 | (−0.15×z).clip(−0.5,+0.5) | ±0.15 |')
    lines.append('| 0.5 | (−0.25×z).clip(−0.5,+0.5) | ±0.25 (REF) |')
    lines.append('| 0.7 | (−0.35×z).clip(−0.5,+0.5) | ±0.35 |')
    lines.append('| 1.0 | (−0.50×z).clip(−0.5,+0.5) | ±0.50 |')
    lines.append('')
    lines.append('k_lt が大きいほど LT2 の逆張りバイアスが強まり、クランプ発生頻度が増加する。')
    lines.append('MaxDD・Worst5Y への影響を確認すること。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 6. 結論')
    lines.append('')
    if 'PASS' in verdict:
        lines.append(f'**B2 は k_lt={best_k:.1f} で Sharpe_OOS が B1 (k_lt=0.5) を上回ることを確認。**')
        lines.append(f'k_lt={best_k:.1f} を CURRENT_BEST_STRATEGY.md 更新候補として提案。')
    elif 'WARN' in verdict:
        lines.append(f'**k_lt=0.5 が引き続き最良。** k_lt={best_k:.1f} を試みたが Sharpe 改善が'
                     f'基準 (+0.020) に達しないため B1 ベスト (k_lt=0.5) を維持する。')
    else:
        lines.append(f'**k_lt=0.5 が最良。** 他の k_lt では基準を満たせないため k_lt=0.5 を採用継続。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 7. 再現コマンド')
    lines.append('')
    lines.append('```')
    lines.append('python -X utf8 src/b2_klt_sweep.py')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/b2_klt_sweep.py`*')
    lines.append('*参照: `B1_S2_LT2_2026-05-21.md`, `CURRENT_BEST_STRATEGY.md`*')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('B2: k_lt スイープ — LT2 バイアス強度の最適化')
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

    # S3: DH Dyn シグナル（1回のみ）
    print('Building DH Dyn signal...')
    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_years = n / TRADING_DAYS
    n_trades_yr = n_tr / n_years
    print(f'  DH Dyn signal: {n_tr} trades, {n_trades_yr:.1f}/yr')

    # S4: S2 CFD レバレッジ系列（1回のみ）
    print('Building S2 CFD leverage series...')
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

    # S5: LT2 シグナル（1回のみ — k_lt に独立）
    print(f'Building LT2 signal (N={LT2_N}) ...')
    lt_sig = build_lt_signal(close, 'LT2', LT2_N)
    nonzero_mask = lt_sig != 0.0
    if nonzero_mask.any():
        first_pos = int(nonzero_mask.idxmax())
        first_date = dates.iloc[first_pos]
        print(f'  LT2 first non-zero date: {first_date.date()}  (warmup OK)')

    # S6: k_lt ループ
    results = []
    print('\n--- k_lt sweep ---')
    for k in K_LT_GRID:
        lt_bias = signal_to_bias(lt_sig, k)
        assert len(lev_A) == len(lt_bias), \
            f'Length mismatch: lev_A={len(lev_A)}, lt_bias={len(lt_bias)}'
        lev_mod = apply_lt_mode_b(lev_A, lt_bias, l_min=0.0, l_max=1.0)

        clamp_lo = int((lev_mod == 0.0).sum())
        clamp_hi = int((lev_mod == 1.0).sum())
        print(f'  k_lt={k:.1f}: lt_bias=[{lt_bias.min():+.3f}, {lt_bias.max():+.3f}]  '
              f'clamp_lo={clamp_lo} days  clamp_hi={clamp_hi} days')

        nav = build_nav_strategy(
            close, lev_mod, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, n_trades_yr)
        m['k_lt'] = k
        m['n_trades_yr'] = n_trades_yr
        m['clamp_lo'] = clamp_lo
        m['clamp_hi'] = clamp_hi
        results.append(m)
        print(f'  k_lt={k:.1f}: CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
              f'Sharpe_OOS={m["Sharpe_OOS"]:+.3f}  '
              f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
              f'IS-OOS gap={m["IS_OOS_gap"]*100:+.2f}pp')

    # S7: サニティチェック
    print()
    print('--- Sanity Check ---')
    r_k05 = next(r for r in results if abs(r['k_lt'] - 0.5) < 1e-9)
    sanity_diff_pp = (r_k05['CAGR_OOS'] - REF_CAGR_OOS_K05) * 100
    sanity_ok = abs(sanity_diff_pp) <= 0.10
    sanity_warn_forced = not sanity_ok
    print(f'[SANITY] k_lt=0.5 CAGR_OOS = {r_k05["CAGR_OOS"]*100:+.2f}%  '
          f'(B1 ref +31.16%, diff {sanity_diff_pp:+.2f} pp)')
    if sanity_ok:
        print('  [OK] within 0.10 pp tolerance.')
    else:
        print('  [WARN] diff > 0.10 pp — データパイプライン確認要。総合判定を強制 WARN 降格。')

    # S8: 最良 k の選択（Sharpe_OOS 最大）
    best_r = max(results, key=lambda r: r['Sharpe_OOS'])
    best_k = best_r['k_lt']
    print(f'\nBest k_lt by Sharpe_OOS: {best_k:.1f}  '
          f'(Sharpe_OOS={best_r["Sharpe_OOS"]:+.3f}, CAGR_OOS={best_r["CAGR_OOS"]*100:+.2f}%)')

    # S9: 判定
    sharpe_delta = best_r['Sharpe_OOS'] - REF_SHARPE_OOS_K05
    crit1 = sharpe_delta >= 0.020
    crit2 = best_r['CAGR_OOS'] >= 0.2916
    crit3 = best_r['IS_OOS_gap'] <= 0.030
    crit4 = best_r['MaxDD_FULL'] > -0.6445
    print(f'  Δ Sharpe_OOS vs k=0.5: {sharpe_delta:+.3f}  (crit1 ≥0.020: {"PASS" if crit1 else "FAIL"})')
    print(f'  CAGR_OOS best:          {best_r["CAGR_OOS"]*100:+.2f}%  (crit2 ≥29.16%: {"PASS" if crit2 else "FAIL"})')
    print(f'  IS-OOS gap best:        {best_r["IS_OOS_gap"]*100:+.2f}pp  (crit3 ≤3.0pp: {"PASS" if crit3 else "FAIL"})')
    print(f'  MaxDD best:             {best_r["MaxDD_FULL"]*100:+.2f}%  (crit4 >-64.45%: {"PASS" if crit4 else "FAIL"})')

    # S10: コンソール結果テーブル
    print()
    print('=' * 120)
    print('B2: k_lt Sweep Results')
    print('=' * 120)
    hdr = (f'{"k_lt":>6}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"MaxDD":>9}  {"Worst10Y★":>10}  {"P10_5Y":>8}  {"Worst5Y":>8}  {"IS-OOS":>9}')
    print(hdr)
    print('-' * 120)
    for r in results:
        marker = ' ← BEST' if abs(r['k_lt'] - best_k) < 1e-9 else ''
        ref_tag = ' [REF]' if abs(r['k_lt'] - 0.5) < 1e-9 else ''
        print(
            f'{r["k_lt"]:>6.1f}{ref_tag}{marker}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f}'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {r["Worst10Y_star"]*100:>+9.2f}%'
            f'  {r["P10_5Y"]*100:>+7.2f}%'
            f'  {r["Worst5Y"]*100:>+7.2f}%'
            f'  {r["IS_OOS_gap"]*100:>+8.2f} pp'
        )
    print('=' * 120)

    # S11: CSV 保存
    df_out = pd.DataFrame([{
        'k_lt':          r['k_lt'],
        'CAGR_IS':       r['CAGR_IS'],
        'CAGR_OOS':      r['CAGR_OOS'],
        'Sharpe_OOS':    r['Sharpe_OOS'],
        'MaxDD_FULL':    r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'],
        'P10_5Y':        r['P10_5Y'],
        'Worst5Y':       r['Worst5Y'],
        'IS_OOS_gap':    r['IS_OOS_gap'],
        'n_trades_yr':   r['n_trades_yr'],
        'clamp_lo':      r['clamp_lo'],
        'clamp_hi':      r['clamp_hi'],
    } for r in results])
    csv_path = os.path.join(BASE, 'b2_klt_sweep_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    # S12: Markdown レポート生成・保存
    md = generate_report(results, sanity_ok, sanity_diff_pp, sanity_warn_forced, best_k)
    md_path = os.path.join(BASE, 'B2_KLT_SWEEP_2026-05-21.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
