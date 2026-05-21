"""
B1: S2_VZGated + LT2-N750-k0.5-modeB 組み合わせ検証
====================================================
目的: LT2-modeB（長期逆張りフィルタ）を S2_VZGated の lev シグナルに適用し、
     Sharpe_OOS が改善するかを検証する。

比較戦略:
  (1) S2_VZGated [baseline]        — LT2 なし
  (2) S2_VZGated + LT2-N750-k0.5-modeB — B1 本体
  (3) DH Dyn [A+LT2] TQQQ          — 参照（既存 Strategy 10 の再計算）

LT2 modeB の作用:
  lt_sig  = compute_lt2(close, N=750)         # N-year momentum z-score
  lt_bias = signal_to_bias(lt_sig, k=0.5)    # (-0.25 * z).clip(-0.5, +0.5)
  lev_mod = clip(lev_A + lt_bias, 0, 1)      # lev_A を ±0.25 シフト

出力:
  - コンソール結果テーブル
  - b1_s2_lt2_results.csv   (プロジェクトルート)
  - B1_S2_LT2_2026-05-21.md (プロジェクトルート)
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
    build_nav,
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
# 参照値 (サニティチェック用)
# ---------------------------------------------------------------------------
REF_CAGR_OOS_S2_BASE  = 0.2751   # A6/A1 で確認済み: S2_VZGated (n=20) CAGR_OOS
REF_DHA_LT2_SHARPE    = 0.777    # Phase-1 best LT2-N750-k0.5-modeB (TQQQ)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 固定パラメータ
S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)
LT2_FIXED = dict(signal='LT2', N=750, k_lt=0.5, mode='B')


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
# 指標計算 (7 metrics + Worst10Y★ + P10_5Y + Worst5Y)
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
    }


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
    lines.append('# B1: S2_VZGated + LT2-N750-k0.5-modeB 組み合わせ検証')
    lines.append('')
    lines.append('**実行日**: 2026-05-21')
    lines.append('**目的**: LT2-modeB（長期逆張りフィルタ）を S2_VZGated の lev に適用し、'
                 'Sharpe_OOS が改善するかを検証する')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 1. セットアップ')
    lines.append('')
    lines.append('### 比較戦略')
    lines.append('')
    lines.append('| # | 戦略名 | lev シグナル | CFD レバ | コメント |')
    lines.append('|---|--------|------------|----------|---------|')
    lines.append('| (1) | S2_VZGated [baseline] | `lev_A`（生） | L_s2 動的配列 | サニティ兼ベースライン |')
    lines.append('| (2) | S2_VZGated + LT2-N750-k0.5-modeB | `clip(lev_A + lt_bias, 0, 1)` | L_s2 動的配列（同じ） | **B1 本体** |')
    lines.append('| (3) | DH Dyn [A+LT2] TQQQ | `clip(lev_A + lt_bias, 0, 1)` | TQQQ 固定 3x | 参照（既存 Strategy 10 再計算） |')
    lines.append('')
    lines.append('### LT2 modeB の式')
    lines.append('')
    lines.append('```')
    lines.append('lt_sig  = compute_lt2(close, N=750)')
    lines.append('lt_bias = signal_to_bias(lt_sig, k=0.5)  # = (-0.25 * z).clip(-0.5, +0.5)')
    lines.append('lev_mod = clip(lev_A + lt_bias, 0, 1)    # 実効バイアス ±0.25')
    lines.append('```')
    lines.append('')
    lines.append(f'- **IS 期間**: {IS_START} 〜 {IS_END}')
    lines.append(f'- **OOS 期間**: {OOS_START} 〜')
    lines.append('- **S2 固定パラメータ**: target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0')
    lines.append('- **CFD スプレッド**: CFD_SPREAD_LOW (0.20%/yr) for (1)(2); TQQQ TER for (3)')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 2. 結果テーブル')
    lines.append('')
    hdr = ('| 戦略 | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD_FULL '
           '| Worst10Y★ | P10_5Y | Worst5Y | IS−OOS gap | 取引/年 |')
    sep = ('|------|--------:|---------:|-----------:|-----------:'
           '|----------:|-------:|--------:|-----------:|-------:|')
    lines.append(hdr)
    lines.append(sep)
    for r in results:
        lines.append(
            f'| {r["strategy"]} '
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
    r1 = next((r for r in results if r['strategy'] == '(1) S2_VZGated [baseline]'), None)
    if r1:
        sanity_tag = '✅ 一致（±0.10 pp 以内）' if sanity_ok else f'⚠️ 乖離 {sanity_diff_pp:+.2f} pp'
        lines.append(f'- (1) S2 baseline CAGR_OOS: **{r1["CAGR_OOS"]*100:+.2f}%**')
        lines.append(f'- 参照値（A6/A1 実測）: **+27.51%**')
        lines.append(f'- 差分: **{sanity_diff_pp:+.2f} pp** → {sanity_tag}')
    if sanity_warn_forced:
        lines.append('- ⚠️ サニティ不一致により総合判定を強制 WARN 降格')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 4. 判定（B1 の有効性評価）')
    lines.append('')
    lines.append('### 判定基準（事前定義）')
    lines.append('')
    lines.append('| 基準 | 条件 | 結果 |')
    lines.append('|------|------|------|')

    r2 = next((r for r in results if r['strategy'] == '(2) S2+LT2-N750-k0.5-modeB'), None)
    if r1 and r2:
        sharpe_delta = r2['Sharpe_OOS'] - r1['Sharpe_OOS']
        cagr_delta_pp = (r2['CAGR_OOS'] - r1['CAGR_OOS']) * 100
        gap_delta_pp = (r2['IS_OOS_gap'] - r1['IS_OOS_gap']) * 100

        crit1 = sharpe_delta >= 0.020
        crit2 = cagr_delta_pp >= -2.0
        crit3 = gap_delta_pp <= 2.0

        lines.append(f'| (i) Sharpe_OOS 改善 ≥ +0.020 | Δ = {sharpe_delta:+.3f} | {"✅ PASS" if crit1 else "⚠️ WARN/FAIL"} |')
        lines.append(f'| (ii) CAGR_OOS 低下 ≤ −2.0 pp | Δ = {cagr_delta_pp:+.2f} pp | {"✅ PASS" if crit2 else "⚠️ WARN/FAIL"} |')
        lines.append(f'| (iii) IS−OOS gap 増大 ≤ +2.0 pp | Δ = {gap_delta_pp:+.2f} pp | {"✅ PASS" if crit3 else "⚠️ WARN/FAIL"} |')
        lines.append('')

        all_pass = crit1 and crit2 and crit3
        if sanity_warn_forced:
            verdict = 'WARN'
        elif all_pass:
            verdict = 'PASS'
        elif crit1 or crit2:
            verdict = 'WARN'
        else:
            verdict = 'FAIL'
    else:
        crit1 = crit2 = crit3 = False
        sharpe_delta = cagr_delta_pp = gap_delta_pp = np.nan
        verdict = 'WARN'

    lines.append(f'**総合判定: {verdict}**')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 5. 追加観察')
    lines.append('')

    # LT2 効果の比較: TQQQ vs CFD
    r3 = next((r for r in results if r['strategy'] == '(3) DH Dyn [A+LT2] TQQQ'), None)
    if r1 and r2 and r3:
        sharpe_tqqq_delta = r3['Sharpe_OOS'] - REF_DHA_LT2_SHARPE  # TQQQ re-calc vs Phase-1 ref
        lines.append('### LT2 modeB の効果: TQQQ 固定レバ vs S2_VZGated 動的レバ')
        lines.append('')
        lines.append('| 項目 | DH Dyn [A] → [A+LT2] TQQQ | S2_VZGated → S2+LT2 |')
        lines.append('|------|---------------------------|---------------------|')
        lines.append(f'| Sharpe_OOS 改善 | +0.131（0.646→0.777 Phase-1実績） | {sharpe_delta:+.3f} |')
        lines.append(f'| CAGR_OOS 変化 | −（Phase-1で低下実績） | {cagr_delta_pp:+.2f} pp |')
        lines.append('')
        if abs(sharpe_delta) >= 0.100:
            lines.append('- **解釈**: LT2 は CFD 動的レバとも独立に大きな効果を発揮。'
                         'VZ ゲートは LT2 と異なるシグナル領域をカバーしている。')
        elif sharpe_delta >= 0.020:
            lines.append('- **解釈**: VZ ゲートが LT2 の効果を一部内包しているため改善幅が縮小。'
                         '二重の逆張りフィルタとして機能している。')
        else:
            lines.append('- **解釈**: VZ ゲートと LT2 のシグナル領域が重複しており効果が飽和。'
                         'S2_VZGated 単体運用を推奨。')
        lines.append('')
        lines.append('### 実効レバについての注意')
        lines.append('')
        lines.append('> LT2 が lev_A を下げた局面でも L_cfd（VZ ゲート）が高い場合、'
                     '`wn × lev × L_cfd` の積では実効レバが想定より高くなる可能性がある。'
                     'Worst5Y・MaxDD の変化を必ず確認すること。')

    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 6. 結論')
    lines.append('')
    if r1 and r2:
        if verdict == 'PASS':
            lines.append(f'**B1 は S2_VZGated に対して Sharpe_OOS +{sharpe_delta:.3f} の有意な改善を達成。**'
                         'CURRENT_BEST_STRATEGY.md の更新候補。'
                         '次の検証候補: B2（k_lt ∈ {0.3, 0.7, 1.0} sweep）または OOS 単独検証。')
        elif verdict == 'WARN':
            lines.append(f'LT2 の効果は限定的（Sharpe Δ={sharpe_delta:+.3f}）。'
                         'VZ ゲートが LT2 の一部効果を既に内包している可能性がある。'
                         'B2（k_lt sweep）で最適 k_lt を探索することを推奨。S2_VZGated 単体を現行ベストとして維持。')
        else:
            lines.append(f'LT2 と VZ ゲートのシグナルが衝突し Sharpe が低下（Δ={sharpe_delta:+.3f}）。'
                         'CFD 動的レバ戦略への LT2 組み合わせは推奨しない。'
                         'S2_VZGated 単体運用を継続する。')
    else:
        lines.append('結果が取得できませんでした。ログを確認してください。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 7. 再現コマンド')
    lines.append('')
    lines.append('```')
    lines.append('python -X utf8 src/b1_s2_lt2.py')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/b1_s2_lt2.py`*')
    lines.append('*参照: `A1_NVOL_SWEEP_2026-05-21.md`, `A6_LMAX_SWEEP_2026-05-21.md`, '
                 '`CURRENT_BEST_STRATEGY.md`*')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('B1: S2_VZGated + LT2-N750-k0.5-modeB 組み合わせ検証')
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

    # S4: LT2 シグナル計算
    print(f'\nBuilding LT2 signal (N={LT2_FIXED["N"]}, k={LT2_FIXED["k_lt"]}, mode={LT2_FIXED["mode"]}) ...')
    lt_sig  = build_lt_signal(close, LT2_FIXED['signal'], LT2_FIXED['N'])
    lt_bias = signal_to_bias(lt_sig, LT2_FIXED['k_lt'])
    assert len(lev_A) == len(lt_bias), f'Length mismatch: lev_A={len(lev_A)}, lt_bias={len(lt_bias)}'
    lev_mod = apply_lt_mode_b(lev_A, lt_bias, l_min=0.0, l_max=1.0)  # np.ndarray

    # LT2 ウォームアップ確認 (close は RangeIndex なので dates で日付引き当て)
    nonzero_mask = lt_sig != 0.0
    if nonzero_mask.any():
        first_nonzero_pos = int(nonzero_mask.idxmax())
        first_nonzero_date = dates.iloc[first_nonzero_pos]
        print(f'  LT2 first non-zero date: {first_nonzero_date.date()}  (warmup OK)')

    # lev_mod の変化量ログ
    delta_lev = lev_mod - lev_A
    print(f'  lt_bias range: [{lt_bias.min():+.3f}, {lt_bias.max():+.3f}]')
    print(f'  lev_mod vs lev_A: mean_delta={delta_lev.mean():+.4f}, '
          f'clamp_at_0={int((delta_lev == -lev_A).sum())} days, '
          f'clamp_at_1={int((lev_mod == 1.0).sum())} days')

    # S5: S2_VZGated CFD レバレッジ系列（1回のみ）
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

    # S6: 3 戦略の NAV 構築
    results = []

    # (1) S2_VZGated [baseline]
    print('\n[1/3] Building S2_VZGated [baseline]...')
    nav1 = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m1 = calc_all_metrics(nav1, dates, n_tr / n_years)
    m1['strategy'] = '(1) S2_VZGated [baseline]'
    m1['n_trades_yr'] = n_tr / n_years
    results.append(m1)
    print(f'  CAGR_OOS={m1["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m1["Sharpe_OOS"]:+.3f}')

    # (2) S2_VZGated + LT2-N750-k0.5-modeB
    print('[2/3] Building S2_VZGated + LT2-N750-k0.5-modeB...')
    nav2 = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m2 = calc_all_metrics(nav2, dates, n_tr / n_years)
    m2['strategy'] = '(2) S2+LT2-N750-k0.5-modeB'
    m2['n_trades_yr'] = n_tr / n_years
    results.append(m2)
    print(f'  CAGR_OOS={m2["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m2["Sharpe_OOS"]:+.3f}')

    # (3) DH Dyn [A+LT2] TQQQ (参照: Strategy 10 再計算)
    print('[3/3] Building DH Dyn [A+LT2] TQQQ (reference)...')
    nav3 = build_nav(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr_daily=sofr, apply_tqqq_sofr=True,
    )
    m3 = calc_all_metrics(nav3, dates, n_tr / n_years)
    m3['strategy'] = '(3) DH Dyn [A+LT2] TQQQ'
    m3['n_trades_yr'] = n_tr / n_years
    results.append(m3)
    print(f'  CAGR_OOS={m3["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m3["Sharpe_OOS"]:+.3f}  '
          f'(ref Phase-1: Sharpe_OOS=+{REF_DHA_LT2_SHARPE:.3f})')

    # S7: サニティチェック
    print()
    print('--- Sanity Check ---')
    sanity_diff_pp = (m1['CAGR_OOS'] - REF_CAGR_OOS_S2_BASE) * 100
    sanity_ok = abs(sanity_diff_pp) <= 0.10
    sanity_warn_forced = not sanity_ok
    print(f'[SANITY] (1) S2 baseline CAGR_OOS = {m1["CAGR_OOS"]*100:+.2f}%  '
          f'(ref +27.51%, diff {sanity_diff_pp:+.2f} pp)')
    if sanity_ok:
        print('  [OK] within 0.10 pp tolerance.')
    else:
        print('  [WARN] diff > 0.10 pp — 再現性を確認。総合判定を強制 WARN 降格。')

    # DH Dyn [A+LT2] の Sharpe 参照
    tqqq_lt2_sharpe_diff = m3['Sharpe_OOS'] - REF_DHA_LT2_SHARPE
    print(f'[INFO] (3) DH Dyn [A+LT2] TQQQ Sharpe_OOS = {m3["Sharpe_OOS"]:+.3f}  '
          f'(Phase-1 ref +{REF_DHA_LT2_SHARPE:.3f}, diff {tqqq_lt2_sharpe_diff:+.3f})')

    # S8: コンソール結果テーブル
    print()
    print('=' * 110)
    print('B1: S2_VZGated + LT2-N750-k0.5-modeB Results')
    print('=' * 110)
    hdr = (f'{"Strategy":<40}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"MaxDD":>9}  {"Worst10Y★":>10}  {"P10_5Y":>8}  {"IS-OOS":>9}')
    print(hdr)
    print('-' * 110)
    for r in results:
        print(
            f'{r["strategy"]:<40}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f}'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {r["Worst10Y_star"]*100:>+9.2f}%'
            f'  {r["P10_5Y"]*100:>+7.2f}%'
            f'  {r["IS_OOS_gap"]*100:>+8.2f} pp'
        )
    print('=' * 110)

    # Sharpe 差分サマリー
    sharpe_delta = m2['Sharpe_OOS'] - m1['Sharpe_OOS']
    cagr_delta_pp = (m2['CAGR_OOS'] - m1['CAGR_OOS']) * 100
    print(f'\nΔ Sharpe_OOS (B1 vs baseline): {sharpe_delta:+.3f}')
    print(f'Δ CAGR_OOS   (B1 vs baseline): {cagr_delta_pp:+.2f} pp')

    # S9: CSV 保存
    df_out = pd.DataFrame([{
        'strategy':       r['strategy'],
        'CAGR_IS':        r['CAGR_IS'],
        'CAGR_OOS':       r['CAGR_OOS'],
        'Sharpe_OOS':     r['Sharpe_OOS'],
        'MaxDD_FULL':     r['MaxDD_FULL'],
        'Worst10Y_star':  r['Worst10Y_star'],
        'P10_5Y':         r['P10_5Y'],
        'Worst5Y':        r['Worst5Y'],
        'IS_OOS_gap':     r['IS_OOS_gap'],
        'n_trades_yr':    r['n_trades_yr'],
    } for r in results])
    csv_path = os.path.join(BASE, 'b1_s2_lt2_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    # S10: Markdown レポート生成・保存
    md = generate_report(results, sanity_ok, sanity_diff_pp, sanity_warn_forced)
    md_path = os.path.join(BASE, 'B1_S2_LT2_2026-05-21.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
