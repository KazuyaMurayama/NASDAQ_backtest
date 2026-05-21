"""
F1: 資産配分比率スイープ
========================
F1a: Gold/Bond スプリット比率スイープ
  gold_frac ∈ {0.20, 0.30, 0.40, 0.50(REF), 0.60, 0.70, 0.80}
  wg = (1 - wn) * gold_frac
  wb = (1 - wn) * (1 - gold_frac)
  wn は simulate_rebalance_A の出力をそのまま使用（変更なし）

F1b: NASDAQ ウェイト下限スイープ
  wn_min ∈ {0.20, 0.25, 0.30(REF), 0.35, 0.40}
  simulate_rebalance_A_wmin() で clip 下限を変える（内部コピー版）
  gold_frac は F1a 最良値を採用

サニティ:
  F1a gold_frac=0.50 の CAGR_OOS が B1 (+31.16%) に ±0.10pp 以内で一致すること。

事前定義判定基準 (PASS):
  F1a/F1b 共通:
  (i)   最良ケースの Sharpe_OOS ≥ REF + 0.020
  (ii)  最良ケースの CAGR_OOS   ≥ +29.16% (REF -2.0pp)
  (iii) 最良ケースの IS-OOS gap ≤ +3.0pp
  (iv)  最良ケースの MaxDD      > -64.45% (REF +5.0pp)
  (v)   最良ケースの Worst10Y★  ≥ +15.0%

出力:
  - f1_alloc_sweep_results.csv
  - F1_ALLOC_SWEEP_2026-05-21.md
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
GOLD_FRAC_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]   # 0.50 が REF
WN_MIN_GRID    = [0.20, 0.25, 0.30, 0.35, 0.40]                 # 0.30 が REF

# サニティ参照値 (B1 確定値, gold_frac=0.50/wn_min=0.30 のもの)
REF_CAGR_OOS    = 0.3116
REF_SHARPE_OOS  = 0.858
REF_MAXDD       = -0.5945

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# F1 専用ヘルパー: wn_min を引数化した rebalance 関数 (simulate_rebalance_A のコピー)
# ---------------------------------------------------------------------------

def simulate_rebalance_A_wmin(raw, vz, threshold=THRESHOLD,
                               wn_min: float = 0.30, wn_max: float = 0.90,
                               gold_frac: float = 0.50):
    """simulate_rebalance_A と同一ロジックだが wn_min/wn_max を引数化。
    corrected_strategy_backtest.py の既存関数は変更しない。"""
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


def reallocate_gold_bond(wn: np.ndarray, gold_frac: float):
    """wn 固定のまま gold_frac で wg/wb を再配分する。"""
    rest = 1.0 - wn
    return rest * gold_frac, rest * (1.0 - gold_frac)


# ---------------------------------------------------------------------------
# ヘルパー: 指標計算
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
# 判定ロジック
# ---------------------------------------------------------------------------

def passes_all(r: dict, ref_sharpe: float) -> bool:
    """全5基準を満たすかチェック。"""
    return (
        r['Sharpe_OOS'] - ref_sharpe >= 0.020 and
        r['CAGR_OOS']      >= 0.2916 and
        r['IS_OOS_gap']    <= 0.030 and
        r['MaxDD_FULL']    > -0.6445 and
        r['Worst10Y_star'] >= 0.150
    )


def find_best_passing(results: list, ref_sharpe: float, key_col: str) -> tuple:
    """全基準を満たすケースの中で Sharpe_OOS が最大のものを返す。
    なければ None を返す。"""
    passing = [r for r in results if passes_all(r, ref_sharpe)]
    if not passing:
        return None, None
    best = max(passing, key=lambda r: r['Sharpe_OOS'])
    return best, best.get(key_col)


def judge_sweep(results: list, ref_sharpe: float, key_col: str,
                sanity_warn_forced: bool) -> tuple:
    """results リストに対して判定を行い (verdict, best_key, best_r) を返す。
    best_key: gold_frac または wn_min の値。"""
    if sanity_warn_forced:
        best_r = max(results, key=lambda r: r['Sharpe_OOS'])
        return 'WARN (サニティ不一致)', best_r.get(key_col), best_r

    best_r, best_key = find_best_passing(results, ref_sharpe, key_col)
    if best_r is not None:
        return 'PASS', best_key, best_r

    # 全基準 PASS なし → 部分的に改善している最良を確認
    abs_best = max(results, key=lambda r: r['Sharpe_OOS'])
    if (abs_best['CAGR_OOS'] >= 0.2916 and
            abs_best['MaxDD_FULL'] > -0.6445 and
            abs_best['IS_OOS_gap'] <= 0.030):
        return 'WARN', abs_best.get(key_col), abs_best
    return 'FAIL', abs_best.get(key_col), abs_best


# ---------------------------------------------------------------------------
# Markdown レポート生成
# ---------------------------------------------------------------------------

def generate_report(results_a: list, results_b: list,
                    sanity_ok_a: bool, sanity_diff_pp_a: float,
                    sanity_warn_forced: bool,
                    best_gf: float, best_wn_min: float,
                    f1a_verdict: str, f1b_verdict: str) -> str:
    today = '2026-05-21'
    lines = []
    lines.append('# F1: 資産配分比率スイープ')
    lines.append('')
    lines.append(f'**実行日**: {today}')
    lines.append('**目的**: Gold/Bond スプリット (F1a) と NASDAQ ウェイト下限 (F1b) をスイープし、'
                 '資産配分の最適化余地を探る')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 1. セットアップ')
    lines.append('')
    lines.append('### F1a: Gold/Bond スプリット')
    lines.append('')
    lines.append(f'- **gold_frac グリッド**: {GOLD_FRAC_GRID}  (0.50 = REF)')
    lines.append('- `wg = (1 - wn) * gold_frac`  `wb = (1 - wn) * (1 - gold_frac)`')
    lines.append('- wn は simulate_rebalance_A の出力を固定使用（変更なし）')
    lines.append('')
    lines.append('### F1b: NASDAQ ウェイト下限')
    lines.append('')
    lines.append(f'- **wn_min グリッド**: {WN_MIN_GRID}  (0.30 = REF)')
    lines.append('- wn = clip(0.55 + 0.25*lev - 0.10*max(vz,0), **wn_min**, 0.90)')
    lines.append('- F1b では gold_frac = F1a 最良値を採用')
    lines.append('')
    lines.append(f'- **IS 期間**: {IS_START} 〜 {IS_END}')
    lines.append(f'- **OOS 期間**: {OOS_START} 〜')
    lines.append('')
    lines.append('---')
    lines.append('')
    # F1a テーブル
    lines.append('## 2. F1a 結果テーブル')
    lines.append('')
    hdr = ('| gold_frac | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD '
           '| Worst10Y★ | P10_5Y | Worst5Y | IS-OOS gap |')
    sep = ('|---:|--------:|---------:|-----------:|----------:'
           '|----------:|-------:|--------:|-----------:|')
    lines.append(hdr)
    lines.append(sep)
    for r in results_a:
        ref_tag  = ' ← REF'  if abs(r['gold_frac'] - 0.50) < 1e-9 else ''
        best_tag = ' ← **BEST**' if abs(r['gold_frac'] - best_gf) < 1e-9 and abs(r['gold_frac'] - 0.50) > 1e-9 else ''
        lines.append(
            f'| {r["gold_frac"]:.2f}{ref_tag}{best_tag} '
            f'| {_fp(r["CAGR_IS"])} '
            f'| {_fp(r["CAGR_OOS"])} '
            f'| {_ff(r["Sharpe_OOS"])} '
            f'| {_fp(r["MaxDD_FULL"])} '
            f'| {_fp(r["Worst10Y_star"])} '
            f'| {_fp(r["P10_5Y"])} '
            f'| {_fp(r["Worst5Y"])} '
            f'| {r["IS_OOS_gap"]*100:+.2f} pp |'
        )
    lines.append('')
    # F1b テーブル
    lines.append('## 3. F1b 結果テーブル')
    lines.append('')
    lines.append(f'*(F1b では gold_frac = {best_gf:.2f} を使用)*')
    lines.append('')
    hdr = ('| wn_min | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD '
           '| Worst10Y★ | P10_5Y | Worst5Y | IS-OOS gap | 取引/年 |')
    sep = ('|---:|--------:|---------:|-----------:|----------:'
           '|----------:|-------:|--------:|-----------:|-------:|')
    lines.append(hdr)
    lines.append(sep)
    for r in results_b:
        ref_tag  = ' ← REF'  if abs(r['wn_min'] - 0.30) < 1e-9 else ''
        best_tag = ' ← **BEST**' if abs(r['wn_min'] - best_wn_min) < 1e-9 and abs(r['wn_min'] - 0.30) > 1e-9 else ''
        lines.append(
            f'| {r["wn_min"]:.2f}{ref_tag}{best_tag} '
            f'| {_fp(r["CAGR_IS"])} '
            f'| {_fp(r["CAGR_OOS"])} '
            f'| {_ff(r["Sharpe_OOS"])} '
            f'| {_fp(r["MaxDD_FULL"])} '
            f'| {_fp(r["Worst10Y_star"])} '
            f'| {_fp(r["P10_5Y"])} '
            f'| {_fp(r["Worst5Y"])} '
            f'| {r["IS_OOS_gap"]*100:+.2f} pp '
            f'| {r["n_trades_yr"]:.1f} |'
        )
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 4. サニティチェック')
    lines.append('')
    sanity_tag = '✅ 一致（±0.10 pp 以内）' if sanity_ok_a else f'⚠️ 乖離 {sanity_diff_pp_a:+.2f} pp'
    r_ref = next((r for r in results_a if abs(r['gold_frac'] - 0.50) < 1e-9), None)
    if r_ref:
        lines.append(f'- F1a gold_frac=0.50 CAGR_OOS: **{r_ref["CAGR_OOS"]*100:+.2f}%**')
        lines.append(f'- B1 参照値: **+31.16%**')
        lines.append(f'- 差分: **{sanity_diff_pp_a:+.2f} pp** → {sanity_tag}')
    if sanity_warn_forced:
        lines.append('- ⚠️ サニティ不一致により総合判定を強制 WARN 降格')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 5. 判定')
    lines.append('')
    lines.append('### 判定基準（事前定義）')
    lines.append('')
    lines.append('| 基準 | 条件 |')
    lines.append('|------|------|')
    lines.append('| (i)   Sharpe_OOS 改善 | ≥ REF + 0.020 |')
    lines.append('| (ii)  CAGR_OOS        | ≥ +29.16% (REF -2.0pp) |')
    lines.append('| (iii) IS-OOS gap      | ≤ +3.0pp |')
    lines.append('| (iv)  MaxDD           | > -64.45% (REF +5.0pp) |')
    lines.append('| (v)   Worst10Y★       | ≥ +15.0% |')
    lines.append('')
    lines.append(f'**F1a 総合判定: {f1a_verdict}** (最良 gold_frac={best_gf:.2f})')
    lines.append(f'**F1b 総合判定: {f1b_verdict}** (最良 wn_min={best_wn_min:.2f})')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 6. 考察')
    lines.append('')
    lines.append('### Gold vs Bond の長期特性 (1974-2026)')
    lines.append('')
    lines.append('- **Gold 2x**: インフレ保護・実質金利低下局面で有利。ボラティリティが高い。')
    lines.append('- **Bond 3x (TMF)**: デフレ・リスクオフ局面でのキャリーと値上がり。高金利局面では損失。')
    lines.append('- 1974-2026 の超長期では両者のバランスが重要。どちらに偏っても極端な局面で脆弱になる。')
    lines.append('')
    lines.append('### wn_min の効果')
    lines.append('')
    lines.append('- wn_min ↑ → 弱気局面でも NASDAQ 露出を維持 → 横ばい〜下落局面でのドラッグ')
    lines.append('- wn_min ↓ → 弱気局面でより守りに回れる → 取引回数増の可能性')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 7. 結論')
    lines.append('')
    if 'PASS' in f1a_verdict:
        lines.append(f'**F1a: gold_frac={best_gf:.2f} が Sharpe_OOS を改善。CURRENT_BEST 更新候補。**')
    else:
        lines.append(f'**F1a: gold_frac=0.50 (REF) が最良。** 他の比率への変更による有意な改善なし。')
    if 'PASS' in f1b_verdict:
        lines.append(f'**F1b: wn_min={best_wn_min:.2f} が Sharpe_OOS を改善。CURRENT_BEST 更新候補。**')
    elif 'WARN' in f1b_verdict:
        lines.append(f'**F1b: wn_min={best_wn_min:.2f} が部分的に改善。** 採用には追加検証が必要。')
    else:
        lines.append(f'**F1b: wn_min=0.30 (REF) が最良。** 変更による有意な改善なし。')
    if 'PASS' not in f1a_verdict and 'PASS' not in f1b_verdict:
        lines.append('')
        lines.append('**Phase 2+D1+F1 を通じて現行ベスト戦略 S2_VZGated+LT2-N750-k0.5-modeB は変更不要。**')
        lines.append('デフォルト配分 (gold_frac=0.50, wn_min=0.30) が最適であることが確認された。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 8. 再現コマンド')
    lines.append('')
    lines.append('```')
    lines.append('python -X utf8 src/f1_alloc_sweep.py')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/f1_alloc_sweep.py`*')
    lines.append('*参照: `B1_S2_LT2_2026-05-21.md`, `D1_OOS_BOUNDARY_2026-05-21.md`, '
                 '`CURRENT_BEST_STRATEGY.md`*')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('F1: 資産配分比率スイープ')
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

    # S2: 共有資産（1回のみ）
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
    lev_A, wn_A, wg_A_ref, wb_A_ref, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_years = n / TRADING_DAYS
    n_trades_yr_ref = n_tr / n_years
    print(f'  DH Dyn signal: {n_tr} trades, {n_trades_yr_ref:.1f}/yr')

    # S4: S2 CFD レバレッジ系列（1回のみ）
    print('Building S2 CFD leverage series...')
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)
    print(f'  L_s2 range: [{L_s2.min():.1f}, {L_s2.max():.1f}]  median={L_s2.median():.1f}')

    # S5: LT2 シグナル (k_lt=0.5)
    print('Building LT2 signal (N=750, k=0.5) ...')
    lt_sig  = build_lt_signal(close, 'LT2', 750)
    lt_bias = signal_to_bias(lt_sig, 0.5)
    assert len(lev_A) == len(lt_bias)
    # lev_mod は F1a (wn 固定) 用の共通ベース
    lev_mod_ref = apply_lt_mode_b(lev_A, lt_bias, l_min=0.0, l_max=1.0)

    # =========================================================
    # F1a: Gold/Bond スプリットスイープ (wn 固定)
    # =========================================================
    print('\n=== F1a: Gold/Bond Split Sweep ===')
    results_a = []
    for gf in GOLD_FRAC_GRID:
        wg_new, wb_new = reallocate_gold_bond(wn_A, gf)
        nav = build_nav_strategy(
            close, lev_mod_ref, wn_A, wg_new, wb_new, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, n_trades_yr_ref)
        m['gold_frac']   = gf
        m['n_trades_yr'] = n_trades_yr_ref
        results_a.append(m)
        ref_tag = ' [REF]' if abs(gf - 0.50) < 1e-9 else ''
        print(
            f'  gold_frac={gf:.2f}{ref_tag}: CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
            f'Sharpe_OOS={m["Sharpe_OOS"]:+.3f}  MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
            f'Worst10Y★={m["Worst10Y_star"]*100:+.2f}%  IS-OOS={m["IS_OOS_gap"]*100:+.2f}pp'
        )

    # F1a サニティ
    print()
    r_ref_a = next(r for r in results_a if abs(r['gold_frac'] - 0.50) < 1e-9)
    sanity_diff_pp_a = (r_ref_a['CAGR_OOS'] - REF_CAGR_OOS) * 100
    sanity_ok_a = abs(sanity_diff_pp_a) <= 0.10
    sanity_warn_forced = not sanity_ok_a
    print(f'[SANITY F1a] gold_frac=0.50 CAGR_OOS = {r_ref_a["CAGR_OOS"]*100:+.2f}%  '
          f'(B1 ref +31.16%, diff {sanity_diff_pp_a:+.2f} pp)')
    if sanity_ok_a:
        print('  [OK]')
    else:
        print('  [WARN] diff > 0.10 pp — 強制 WARN 降格。')

    # F1a 最良 gold_frac (全基準PASSの中で最大Sharpe)
    f1a_verdict, best_gf, best_r_a = judge_sweep(results_a, REF_SHARPE_OOS, 'gold_frac', sanity_warn_forced)
    if best_r_a is None:
        best_gf = 0.50
        best_r_a = next(r for r in results_a if abs(r['gold_frac'] - 0.50) < 1e-9)
    sharpe_delta_a = best_r_a['Sharpe_OOS'] - REF_SHARPE_OOS
    print(f'\nF1a best gold_frac: {best_gf:.2f}  '
          f'(Sharpe_OOS={best_r_a["Sharpe_OOS"]:+.3f}, Δ={sharpe_delta_a:+.3f} vs REF)')
    print(f'F1a verdict: {f1a_verdict}')

    # =========================================================
    # F1b: NASDAQ ウェイト下限スイープ (gold_frac = best_gf)
    # =========================================================
    print(f'\n=== F1b: NASDAQ wn_min Sweep (gold_frac={best_gf:.2f}) ===')
    results_b = []
    for wn_min in WN_MIN_GRID:
        lev_b, wn_b, wg_b, wb_b, n_tr_b = simulate_rebalance_A_wmin(
            raw_a2, vz, THRESHOLD, wn_min=wn_min, wn_max=0.90, gold_frac=best_gf,
        )
        n_trades_yr_b = n_tr_b / n_years

        # LT2 を lev_b に適用
        lt_bias_b = signal_to_bias(lt_sig, 0.5)
        lev_mod_b = apply_lt_mode_b(lev_b, lt_bias_b, l_min=0.0, l_max=1.0)

        nav = build_nav_strategy(
            close, lev_mod_b, wn_b, wg_b, wb_b, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, n_trades_yr_b)
        m['wn_min']      = wn_min
        m['gold_frac']   = best_gf
        m['n_trades_yr'] = n_trades_yr_b
        results_b.append(m)
        ref_tag = ' [REF]' if abs(wn_min - 0.30) < 1e-9 else ''
        print(
            f'  wn_min={wn_min:.2f}{ref_tag}: CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
            f'Sharpe_OOS={m["Sharpe_OOS"]:+.3f}  MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
            f'Worst10Y★={m["Worst10Y_star"]*100:+.2f}%  '
            f'Trades/yr={n_trades_yr_b:.1f}  IS-OOS={m["IS_OOS_gap"]*100:+.2f}pp'
        )

    # F1b サニティ (wn_min=0.30, gold_frac=best_gf との比較)
    r_ref_b = next((r for r in results_b if abs(r['wn_min'] - 0.30) < 1e-9), None)
    if r_ref_b:
        print(f'\n[SANITY F1b] wn_min=0.30 CAGR_OOS = {r_ref_b["CAGR_OOS"]*100:+.2f}%')

    # F1b 最良 wn_min (全基準PASSの中で最大Sharpe)
    f1b_verdict, best_wn_min, best_r_b = judge_sweep(results_b, REF_SHARPE_OOS, 'wn_min', sanity_warn_forced)
    if best_r_b is None:
        best_wn_min = 0.30
        best_r_b = next(r for r in results_b if abs(r['wn_min'] - 0.30) < 1e-9)
    sharpe_delta_b = best_r_b['Sharpe_OOS'] - REF_SHARPE_OOS
    print(f'F1b best wn_min: {best_wn_min:.2f}  '
          f'(Sharpe_OOS={best_r_b["Sharpe_OOS"]:+.3f}, Δ={sharpe_delta_b:+.3f} vs REF)')
    print(f'F1b verdict: {f1b_verdict}')

    # =========================================================
    # コンソール結果テーブル
    # =========================================================
    print()
    print('=' * 120)
    print('F1a: Gold/Bond Split Results (wn fixed)')
    print('=' * 120)
    hdr = (f'{"gold_frac":>10}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"MaxDD":>9}  {"Worst10Y★":>10}  {"P10_5Y":>8}  {"Worst5Y":>8}  {"IS-OOS":>9}')
    print(hdr)
    print('-' * 120)
    for r in results_a:
        tags = ''
        if abs(r['gold_frac'] - 0.50) < 1e-9:   tags += ' [REF]'
        if abs(r['gold_frac'] - best_gf) < 1e-9 and abs(r['gold_frac'] - 0.50) > 1e-9:
            tags += ' ← BEST'
        print(
            f'{r["gold_frac"]:>10.2f}{tags}'
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

    print()
    print('=' * 120)
    print(f'F1b: wn_min Sweep Results (gold_frac={best_gf:.2f})')
    print('=' * 120)
    hdr = (f'{"wn_min":>8}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"MaxDD":>9}  {"Worst10Y★":>10}  {"P10_5Y":>8}  {"Worst5Y":>8}  '
           f'{"IS-OOS":>9}  {"Trades/yr":>10}')
    print(hdr)
    print('-' * 120)
    for r in results_b:
        tags = ''
        if abs(r['wn_min'] - 0.30) < 1e-9:           tags += ' [REF]'
        if abs(r['wn_min'] - best_wn_min) < 1e-9 and abs(r['wn_min'] - 0.30) > 1e-9:
            tags += ' ← BEST'
        print(
            f'{r["wn_min"]:>8.2f}{tags}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f}'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {r["Worst10Y_star"]*100:>+9.2f}%'
            f'  {r["P10_5Y"]*100:>+7.2f}%'
            f'  {r["Worst5Y"]*100:>+7.2f}%'
            f'  {r["IS_OOS_gap"]*100:>+8.2f} pp'
            f'  {r["n_trades_yr"]:>10.1f}'
        )
    print('=' * 120)

    # S_CSV: CSV 保存
    rows_a = [{
        'sweep': 'F1a',
        'gold_frac': r['gold_frac'],
        'wn_min': 0.30,
        'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
        'Worst5Y': r['Worst5Y'], 'IS_OOS_gap': r['IS_OOS_gap'],
        'n_trades_yr': r['n_trades_yr'],
    } for r in results_a]
    rows_b = [{
        'sweep': 'F1b',
        'gold_frac': r['gold_frac'],
        'wn_min': r['wn_min'],
        'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
        'Worst5Y': r['Worst5Y'], 'IS_OOS_gap': r['IS_OOS_gap'],
        'n_trades_yr': r['n_trades_yr'],
    } for r in results_b]
    df_out = pd.DataFrame(rows_a + rows_b)
    csv_path = os.path.join(BASE, 'f1_alloc_sweep_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    # Markdown レポート
    md = generate_report(results_a, results_b,
                         sanity_ok_a, sanity_diff_pp_a, sanity_warn_forced,
                         best_gf, best_wn_min, f1a_verdict, f1b_verdict)
    md_path = os.path.join(BASE, 'F1_ALLOC_SWEEP_2026-05-21.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
