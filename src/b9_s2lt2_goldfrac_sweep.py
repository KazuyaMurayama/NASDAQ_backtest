"""
B9: S2+LT2-N750 × Gold_frac / wn_min 2D グリッドスイープ
=========================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-23)

2D 同時スイープ:
  gold_frac ∈ {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
  wn_min    ∈ {0.20, 0.25, 0.30, 0.35}
  合計 28 configs

REF (gold_frac=0.50, wn_min=0.30): CAGR_OOS +31.16%, Sharpe_OOS +0.858

出力:
  - b9_s2lt2_goldfrac_results.csv
  - B9_S2LT2_GOLDFRAC_SWEEP_2026-05-23.md
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
    _fp1, _ff2, _gap_pp, _tr, _wfa,
    MD_WFA_NOTE,
)

# ---------------------------------------------------------------------------
# グリッド定義
# ---------------------------------------------------------------------------
GOLD_FRAC_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]   # 7値
WN_MIN_GRID    = [0.20, 0.25, 0.30, 0.35]                        # 4値
# REF点
REF_GOLD_FRAC = 0.50
REF_WN_MIN    = 0.30

# サニティ参照値 (B1/F1 確定値)
REF_CAGR_OOS   = 0.3116   # +31.16%
REF_SHARPE_OOS = 0.858
REF_MAXDD      = -0.5945

# PASS 判定基準（事前定義）
PASS_SHARPE_DELTA = 0.020   # (i) Sharpe_OOS ≥ REF + 0.020
PASS_CAGR_OOS     = 0.2916  # (ii) CAGR_OOS ≥ +29.16%
PASS_GAP          = 0.030   # (iii) IS-OOS gap ≤ +3.0pp
PASS_MAXDD        = -0.6445 # (iv) MaxDD > -64.45%
PASS_WORST10Y     = 0.150   # (v) Worst10Y★ ≥ +15.0%

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# simulate_rebalance_A_wmin (f1_alloc_sweep.py から移植)
# ---------------------------------------------------------------------------

def simulate_rebalance_A_wmin(raw, vz, threshold=THRESHOLD,
                               wn_min: float = 0.30, wn_max: float = 0.90,
                               gold_frac: float = 0.50):
    """simulate_rebalance_A と同一ロジックだが wn_min/wn_max/gold_frac を引数化。"""
    n = len(raw); raw_v = raw.values; vz_v = vz.values
    lev = np.zeros(n); wn = np.zeros(n); wg = np.zeros(n); wb = np.zeros(n)
    cur_lev = raw_v[0]
    cur_wn = float(np.clip(0.55 + 0.25 * cur_lev - 0.10 * max(vz_v[0], 0), wn_min, wn_max))
    cur_wg = (1 - cur_wn) * gold_frac
    cur_wb = (1 - cur_wn) * (1 - gold_frac)
    lev[0] = cur_lev; wn[0] = cur_wn; wg[0] = cur_wg; wb[0] = cur_wb
    n_trades = 0
    for i in range(1, n):
        t = raw_v[i]
        if (t == 0 and cur_lev > 0) or (cur_lev == 0 and t > 0) or abs(t - cur_lev) > threshold:
            cur_lev = t
            cur_wn = float(np.clip(0.55 + 0.25 * cur_lev - 0.10 * max(vz_v[i], 0), wn_min, wn_max))
            cur_wg = (1 - cur_wn) * gold_frac
            cur_wb = (1 - cur_wn) * (1 - gold_frac)
            n_trades += 1
        lev[i] = cur_lev; wn[i] = cur_wn; wg[i] = cur_wg; wb[i] = cur_wb
    return lev, wn, wg, wb, n_trades


# ---------------------------------------------------------------------------
# 指標計算ヘルパー
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


# ---------------------------------------------------------------------------
# PASS 判定
# ---------------------------------------------------------------------------

def passes_all(r: dict) -> bool:
    return (
        r['Sharpe_OOS'] - REF_SHARPE_OOS >= PASS_SHARPE_DELTA and
        r['CAGR_OOS']      >= PASS_CAGR_OOS and
        r['IS_OOS_gap']    <= PASS_GAP and
        r['MaxDD_FULL']    > PASS_MAXDD and
        r['Worst10Y_star'] >= PASS_WORST10Y
    )


def row_verdict(r: dict, is_ref: bool) -> str:
    if is_ref:
        return '← REF'
    if passes_all(r):
        return 'PASS'
    # 部分判定
    c_sharpe = r['Sharpe_OOS'] - REF_SHARPE_OOS >= PASS_SHARPE_DELTA
    c_cagr   = r['CAGR_OOS'] >= PASS_CAGR_OOS
    c_gap    = r['IS_OOS_gap'] <= PASS_GAP
    c_maxdd  = r['MaxDD_FULL'] > PASS_MAXDD
    c_worst  = r['Worst10Y_star'] >= PASS_WORST10Y
    n_pass = sum([c_sharpe, c_cagr, c_gap, c_maxdd, c_worst])
    if n_pass >= 4:
        return 'WARN'
    return 'FAIL'


# ---------------------------------------------------------------------------
# Markdown レポート生成
# ---------------------------------------------------------------------------

def generate_report(results: list, sanity_ok: bool, sanity_diff_pp: float,
                    best_r: dict, verdict: str) -> str:
    today = '2026-05-23'
    lines = []
    lines.append('# B9: S2+LT2-N750 × Gold_frac / wn_min 2D グリッドスイープ')
    lines.append('')
    lines.append(f'**実行日**: {today}')
    lines.append('**目的**: gold_frac × wn_min の 2D 同時スイープで配分最適化余地を検証する')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 1. セットアップ')
    lines.append('')
    lines.append(f'- **gold_frac グリッド**: {GOLD_FRAC_GRID}  (0.50 = REF)')
    lines.append(f'- **wn_min グリッド**: {WN_MIN_GRID}  (0.30 = REF)')
    lines.append('- **合計 configs**: 28（7 × 4）')
    lines.append(f'- **IS 期間**: {IS_START} 〜 {IS_END}')
    lines.append(f'- **OOS 期間**: {OOS_START} 〜')
    lines.append('- **S2 固定パラメータ**: target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0')
    lines.append('- **LT2 固定パラメータ**: N=750, k_lt=0.5, modeB')
    lines.append('')
    lines.append('### PASS 判定基準（事前定義）')
    lines.append('')
    lines.append('| 基準 | 条件 |')
    lines.append('|------|------|')
    lines.append('| (i)   Sharpe_OOS 改善 | ≥ REF + 0.020 (≥ 0.878) |')
    lines.append('| (ii)  CAGR_OOS        | ≥ +29.16% (REF -2.0pp) |')
    lines.append('| (iii) IS-OOS gap      | ≤ +3.0pp |')
    lines.append('| (iv)  MaxDD           | > -64.45% (REF +5.0pp) |')
    lines.append('| (v)   Worst10Y★       | ≥ +15.0% |')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 2. 結果テーブル (28 configs)')
    lines.append('')

    # カスタムヘッダ (gold_frac + wn_min 2パラメータ)
    hdr = ('| gold_frac | wn_min | CAGR<br>_OOS | Sharpe | MaxDD | Worst<br>10Y★ '
           '| P10▷ | IS-OOS<br>gap | Tr | CI95<br>_lo | WFE | 判定 |')
    sep = ('|----------:|-------:|-------------:|-------:|------:|--------------:'
           '|-----:|--------------:|---:|-----------:|----:|:----:|')
    lines.append(hdr)
    lines.append(sep)

    # REF点のSharpを先取り（マーカ用）
    ref_sharpe = REF_SHARPE_OOS
    best_sharpe = best_r['Sharpe_OOS'] if best_r else -999.0

    for r in results:
        is_ref  = (abs(r['gold_frac'] - REF_GOLD_FRAC) < 1e-9 and
                   abs(r['wn_min']    - REF_WN_MIN   ) < 1e-9)
        is_best = (best_r is not None and
                   abs(r['gold_frac'] - best_r['gold_frac']) < 1e-9 and
                   abs(r['wn_min']    - best_r['wn_min']   ) < 1e-9 and
                   not is_ref)

        vdict = row_verdict(r, is_ref)
        mark = ''
        if r['Sharpe_OOS'] > 0.885:
            mark = ' ★'
        elif r['Sharpe_OOS'] > 0.770:
            mark = ' ◎'

        if is_ref:
            label = f'{r["gold_frac"]:.2f} ← REF'
        elif is_best:
            label = f'{r["gold_frac"]:.2f} ← **BEST**'
        else:
            label = f'{r["gold_frac"]:.2f}'

        lines.append(
            f'| {label} '
            f'| {r["wn_min"]:.2f} '
            f'| {_fp1(r["CAGR_OOS"])} '
            f'| {_ff2(r["Sharpe_OOS"])}{mark} '
            f'| {_fp1(r["MaxDD_FULL"])} '
            f'| {_fp1(r["Worst10Y_star"])} '
            f'| {_fp1(r["P10_5Y"])} '
            f'| {_gap_pp(r["IS_OOS_gap"])} '
            f'| {_tr(r.get("Trades_yr"))} '
            f'| {_wfa(r.get("WFA_CI95_lo"))} '
            f'| {_wfa(r.get("WFA_WFE"))} '
            f'| {vdict} |'
        )

    lines.append('')
    lines.append(MD_WFA_NOTE)
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 3. サニティチェック')
    lines.append('')
    r_ref = next((r for r in results
                  if abs(r['gold_frac'] - REF_GOLD_FRAC) < 1e-9
                  and abs(r['wn_min'] - REF_WN_MIN) < 1e-9), None)
    if r_ref:
        sanity_tag = '一致（±0.10 pp 以内）' if sanity_ok else f'乖離 {sanity_diff_pp:+.2f} pp'
        lines.append(f'- REF (gold_frac=0.50, wn_min=0.30) CAGR_OOS: **{r_ref["CAGR_OOS"]*100:+.2f}%**')
        lines.append(f'- B1/F1 参照値: **+31.16%**')
        lines.append(f'- 差分: **{sanity_diff_pp:+.2f} pp** → {sanity_tag}')
    if not sanity_ok:
        lines.append('- 警告: サニティ不一致により総合判定を強制 WARN 降格')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 4. 判定')
    lines.append('')

    pass_count = sum(1 for r in results
                     if passes_all(r) and not
                     (abs(r['gold_frac'] - REF_GOLD_FRAC) < 1e-9 and
                      abs(r['wn_min'] - REF_WN_MIN) < 1e-9))
    warn_count = sum(1 for r in results
                     if row_verdict(r, False) == 'WARN')

    lines.append(f'- **PASS configs**: {pass_count} / 27 (REF 除く)')
    lines.append(f'- **WARN configs**: {warn_count}')
    lines.append('')

    if best_r is not None:
        lines.append(f'**最高 Sharpe_OOS**: {best_r["Sharpe_OOS"]:+.3f} '
                     f'(gold_frac={best_r["gold_frac"]:.2f}, wn_min={best_r["wn_min"]:.2f})')
        lines.append(f'  - CAGR_OOS:  {best_r["CAGR_OOS"]*100:+.2f}%')
        lines.append(f'  - MaxDD:     {best_r["MaxDD_FULL"]*100:+.2f}%')
        lines.append(f'  - Worst10Y★: {best_r["Worst10Y_star"]*100:+.2f}%')
        lines.append(f'  - IS-OOS gap: {best_r["IS_OOS_gap"]*100:+.2f} pp')
        lines.append('')

    lines.append(f'**総合判定: {verdict}**')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 5. 結論')
    lines.append('')

    if 'PASS' in verdict:
        lines.append(f'**B9 は 2D スイープで Sharpe_OOS を改善する設定を発見。**')
        if best_r:
            lines.append(f'推奨候補: gold_frac={best_r["gold_frac"]:.2f}, wn_min={best_r["wn_min"]:.2f} '
                         f'(Sharpe_OOS={best_r["Sharpe_OOS"]:+.3f})')
        lines.append('次のステップ: CURRENT_BEST_STRATEGY.md 更新候補として G1 WFA 検証を実施すること。')
    elif 'WARN' in verdict:
        lines.append('一部の設定で部分的な改善が見られるが、全基準 PASS には至らない。')
        if best_r:
            lines.append(f'最良設定: gold_frac={best_r["gold_frac"]:.2f}, wn_min={best_r["wn_min"]:.2f} '
                         f'(Sharpe_OOS={best_r["Sharpe_OOS"]:+.3f})')
        lines.append('デフォルト配分 (gold_frac=0.50, wn_min=0.30) の継続を推奨。')
    else:
        lines.append('2D スイープで全基準を満たす設定は存在しない。')
        lines.append('デフォルト配分 (gold_frac=0.50, wn_min=0.30) を現行ベストとして維持する。')

    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 6. 再現コマンド')
    lines.append('')
    lines.append('```')
    lines.append('python -X utf8 src/b9_s2lt2_goldfrac_sweep.py')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/b9_s2lt2_goldfrac_sweep.py`*')
    lines.append('*参照: `B1_S2_LT2_2026-05-21.md`, `F1_ALLOC_SWEEP_2026-05-21.md`, '
                 '`CURRENT_BEST_STRATEGY.md`*')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('B9: S2+LT2-N750 × Gold_frac / wn_min 2D グリッドスイープ')
    print('実行日: 2026-05-23')
    print('=' * 70)
    print(f'グリッド: gold_frac={GOLD_FRAC_GRID}')
    print(f'         wn_min   ={WN_MIN_GRID}')
    print(f'合計 {len(GOLD_FRAC_GRID) * len(WN_MIN_GRID)} configs')

    # S1: データロード
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    print(f'\nData: {dates.iloc[0].date()} to {dates.iloc[-1].date()} '
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
    n_years = n / TRADING_DAYS
    print(f'  DH Dyn signal built.')

    # S4: LT2 シグナル (N=750, k=0.5)
    print('Building LT2 signal (N=750, k=0.5, modeB) ...')
    lt_sig  = build_lt_signal(close, 'LT2', 750)
    lt_bias = signal_to_bias(lt_sig, 0.5)
    print(f'  lt_bias range: [{lt_bias.min():+.3f}, {lt_bias.max():+.3f}]')

    # S5: S2_VZGated CFD レバレッジ系列（1回のみ）
    print('\nBuilding S2 CFD leverage series...')
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)
    print(f'  L_s2 range: [{L_s2.min():.1f}, {L_s2.max():.1f}]  median={L_s2.median():.1f}')

    # S6: 2D スイープ
    print(f'\n=== 2D Sweep: {len(GOLD_FRAC_GRID)} × {len(WN_MIN_GRID)} = '
          f'{len(GOLD_FRAC_GRID) * len(WN_MIN_GRID)} configs ===')
    results = []
    total = len(GOLD_FRAC_GRID) * len(WN_MIN_GRID)
    idx = 0

    for gf in GOLD_FRAC_GRID:
        for wn_min in WN_MIN_GRID:
            idx += 1
            is_ref = (abs(gf - REF_GOLD_FRAC) < 1e-9 and abs(wn_min - REF_WN_MIN) < 1e-9)
            ref_tag = ' [REF]' if is_ref else ''

            # simulate_rebalance_A_wmin で wn/wg/wb を取得
            lev_b, wn_b, wg_b, wb_b, n_tr_b = simulate_rebalance_A_wmin(
                raw_a2, vz, THRESHOLD,
                wn_min=wn_min, wn_max=0.90, gold_frac=gf,
            )
            n_trades_yr = n_tr_b / n_years

            # LT2 modeB を lev_b に適用
            lev_mod = apply_lt_mode_b(lev_b, lt_bias, l_min=0.0, l_max=1.0)

            # NAV 構築
            nav = build_nav_strategy(
                close, lev_mod, wn_b, wg_b, wb_b, dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
            )

            # 指標計算
            m = calc_all_metrics(nav, dates, n_trades_yr)
            m['gold_frac']  = gf
            m['wn_min']     = wn_min
            m['Trades_yr']  = n_trades_yr
            results.append(m)

            print(
                f'  [{idx:>2d}/{total}] gold_frac={gf:.2f} wn_min={wn_min:.2f}{ref_tag}: '
                f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
                f'Sharpe_OOS={m["Sharpe_OOS"]:+.3f}  '
                f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
                f'Worst10Y★={m["Worst10Y_star"]*100:+.2f}%  '
                f'IS-OOS={m["IS_OOS_gap"]*100:+.2f}pp  '
                f'Tr/yr={n_trades_yr:.1f}'
            )

    # S7: サニティチェック
    print()
    print('--- Sanity Check ---')
    r_ref = next((r for r in results
                  if abs(r['gold_frac'] - REF_GOLD_FRAC) < 1e-9
                  and abs(r['wn_min'] - REF_WN_MIN) < 1e-9), None)
    if r_ref:
        sanity_diff_pp = (r_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
        sanity_ok = abs(sanity_diff_pp) <= 0.10
        print(f'[SANITY] REF (0.50, 0.30) CAGR_OOS = {r_ref["CAGR_OOS"]*100:+.2f}%  '
              f'(ref +31.16%, diff {sanity_diff_pp:+.2f} pp)')
        if sanity_ok:
            print('  [OK] within 0.10 pp tolerance.')
        else:
            print('  [WARN] diff > 0.10 pp — 再現性を確認。総合判定を強制 WARN 降格。')
    else:
        sanity_diff_pp = float('nan')
        sanity_ok = False
        print('[WARN] REF点が結果に見つかりませんでした。')

    # S8: 全体サマリー
    print()
    print('--- Summary ---')
    sharpe_vals = [r['Sharpe_OOS'] for r in results]
    cagr_vals   = [r['CAGR_OOS'] * 100 for r in results]
    print(f'Sharpe_OOS range: [{min(sharpe_vals):+.3f}, {max(sharpe_vals):+.3f}]')
    print(f'CAGR_OOS   range: [{min(cagr_vals):+.2f}%, {max(cagr_vals):+.2f}%]')

    # 最高Sharpe
    best_all = max(results, key=lambda r: r['Sharpe_OOS'])
    print(f'\n最高 Sharpe_OOS: {best_all["Sharpe_OOS"]:+.3f} '
          f'(gold_frac={best_all["gold_frac"]:.2f}, wn_min={best_all["wn_min"]:.2f})')
    print(f'  CAGR_OOS:  {best_all["CAGR_OOS"]*100:+.2f}%  '
          f'MaxDD: {best_all["MaxDD_FULL"]*100:+.2f}%  '
          f'Worst10Y★: {best_all["Worst10Y_star"]*100:+.2f}%  '
          f'IS-OOS gap: {best_all["IS_OOS_gap"]*100:+.2f}pp')

    # PASS判定
    pass_list = [r for r in results
                 if passes_all(r) and not
                 (abs(r['gold_frac'] - REF_GOLD_FRAC) < 1e-9 and
                  abs(r['wn_min'] - REF_WN_MIN) < 1e-9)]
    print(f'\nPASS configs (REF除く): {len(pass_list)} / {total - 1}')
    for r in pass_list:
        print(f'  gold_frac={r["gold_frac"]:.2f} wn_min={r["wn_min"]:.2f}: '
              f'Sharpe={r["Sharpe_OOS"]:+.3f}  CAGR={r["CAGR_OOS"]*100:+.2f}%  '
              f'MaxDD={r["MaxDD_FULL"]*100:+.2f}%  Worst10Y★={r["Worst10Y_star"]*100:+.2f}%  '
              f'IS-OOS={r["IS_OOS_gap"]*100:+.2f}pp')

    # 総合判定
    best_passing = max(pass_list, key=lambda r: r['Sharpe_OOS']) if pass_list else None
    if not sanity_ok:
        verdict = 'WARN (サニティ不一致)'
        best_r = best_all
    elif pass_list:
        verdict = 'PASS'
        best_r = best_passing
    else:
        # 部分改善確認
        if (best_all['CAGR_OOS'] >= PASS_CAGR_OOS and
                best_all['MaxDD_FULL'] > PASS_MAXDD and
                best_all['IS_OOS_gap'] <= PASS_GAP):
            verdict = 'WARN'
        else:
            verdict = 'FAIL'
        best_r = best_all
    print(f'\n総合判定: {verdict}')

    # S9: コンソール結果テーブル
    print()
    print('=' * 130)
    print('B9: 全28 configs 結果テーブル')
    print('=' * 130)
    hdr = (f'{"gold_frac":>10}  {"wn_min":>7}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  '
           f'{"Sharpe_OOS":>11}  {"MaxDD":>9}  {"Worst10Y★":>10}  '
           f'{"P10_5Y":>8}  {"IS-OOS":>9}  {"Tr/yr":>6}')
    print(hdr)
    print('-' * 130)
    for r in results:
        tags = ''
        if (abs(r['gold_frac'] - REF_GOLD_FRAC) < 1e-9 and
                abs(r['wn_min'] - REF_WN_MIN) < 1e-9):
            tags += ' [REF]'
        elif passes_all(r):
            tags += ' [PASS]'
        print(
            f'{r["gold_frac"]:>10.2f}'
            f'  {r["wn_min"]:>7.2f}{tags}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f}'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {r["Worst10Y_star"]*100:>+9.2f}%'
            f'  {r["P10_5Y"]*100:>+7.2f}%'
            f'  {r["IS_OOS_gap"]*100:>+8.2f} pp'
            f'  {r["Trades_yr"]:>6.1f}'
        )
    print('=' * 130)

    # S10: CSV 保存
    df_out = pd.DataFrame([{
        'gold_frac':     r['gold_frac'],
        'wn_min':        r['wn_min'],
        'CAGR_IS':       r['CAGR_IS'],
        'CAGR_OOS':      r['CAGR_OOS'],
        'Sharpe_OOS':    r['Sharpe_OOS'],
        'MaxDD_FULL':    r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'],
        'P10_5Y':        r['P10_5Y'],
        'IS_OOS_gap':    r['IS_OOS_gap'],
        'Trades_yr':     r['Trades_yr'],
    } for r in results])
    csv_path = os.path.join(BASE, 'b9_s2lt2_goldfrac_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    # S11: Markdown レポート生成・保存
    md = generate_report(results, sanity_ok, sanity_diff_pp, best_r, verdict)
    md_path = os.path.join(BASE, 'B9_S2LT2_GOLDFRAC_SWEEP_2026-05-23.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
