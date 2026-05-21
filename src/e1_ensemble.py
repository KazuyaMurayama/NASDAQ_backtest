"""
E1: アンサンブル — S2+LT2 (CFD) × DH Dyn [A] TQQQ
======================================================
目的: S2_VZGated+LT2-modeB (k_lt=0.5) と DH Dyn [A] TQQQ (LT2なし) を
     NAV レベルで w:(1-w) ブレンドし、MaxDD 緩和と多様化を図る。

設計:
  - NAV_A: S2+LT2 (CFD, k_lt=0.5) — B1(2) と同一
  - NAV_B: DH Dyn [A] TQQQ (lev_A を使用、LT2 なし)
  - アンサンブル: daily_ret = w*r_A + (1-w)*r_B  → cumprod
  - w スイープ: {0.3, 0.5, 0.7}  (0.5 = REF)

サニティ:
  NAV_A の CAGR_OOS は B1 (k_lt=0.5) の +31.16% に ±0.10pp 以内で一致する必要がある。

事前定義判定基準 (w=0.5 に対して評価, PASS 条件):
  (i)  MaxDD  < -50%       (S2+LT2 単体 -59.45% より ≥ 9.45pp 改善)
  (ii) Sharpe_OOS ≥ 0.808 (S2+LT2 単体 0.858 から ≤ -0.050 低下)
  (iii) CAGR_OOS ≥ +22.0%
  (iv) Worst5Y ≥ -2.83%  (S2+LT2 比 同等以上)

出力:
  - e1_ensemble_results.csv
  - E1_ENSEMBLE_2026-05-21.md
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
# 定数
# ---------------------------------------------------------------------------
W_GRID = [0.3, 0.5, 0.7]    # NAV_A の比重

# サニティ: NAV_A (S2+LT2 k_lt=0.5) の CAGR_OOS (B1 確定値)
REF_CAGR_OOS_A = 0.3116
# NAV_B 参照 (DH Dyn [A] TQQQ 素): 存在確認用のみ (loose check)
REF_CAGR_OOS_B_MIN = 0.08    # 最低ライン (DH Dyn [A] TQQQ は OOS で正収益が期待される)

# S2+LT2 ベースライン (判定用)
REF_SHARPE_A    = 0.858
REF_MAXDD_A     = -0.5945
REF_WORST5Y_A   = -0.0283

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)

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

def generate_report(m_A: dict, m_B: dict, results: list,
                    sanity_ok_A: bool, sanity_diff_pp_A: float,
                    sanity_warn_forced: bool) -> str:
    today = '2026-05-21'
    lines = []
    lines.append('# E1: アンサンブル — S2+LT2 (CFD) × DH Dyn [A] TQQQ')
    lines.append('')
    lines.append(f'**実行日**: {today}')
    lines.append('**目的**: S2_VZGated+LT2-modeB (k_lt=0.5) と DH Dyn [A] TQQQ (LT2なし) を '
                 'w:(1-w) でブレンドし、MaxDD 緩和と多様化を図る')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 1. セットアップ')
    lines.append('')
    lines.append('| コンポーネント | 戦略 | 備考 |')
    lines.append('|---|---|---|')
    lines.append('| NAV_A | S2_VZGated + LT2-N750-k0.5-modeB (CFD) | B1(2) と同一 |')
    lines.append('| NAV_B | DH Dyn [A] TQQQ (LT2 なし) | `lev_A` を直接使用 |')
    lines.append('')
    lines.append('```')
    lines.append('daily_ret = w * r_A + (1 - w) * r_B  # 日次リバランス近似')
    lines.append('nav_ens   = (1 + daily_ret).cumprod()')
    lines.append('```')
    lines.append('')
    lines.append(f'- **w グリッド**: {W_GRID} (NAV_A の比重)')
    lines.append(f'- **IS 期間**: {IS_START} 〜 {IS_END}')
    lines.append(f'- **OOS 期間**: {OOS_START} 〜')
    lines.append('')
    lines.append('### 単体成分の指標')
    lines.append('')
    lines.append('| 戦略 | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD | Worst5Y |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    lines.append(
        f'| NAV_A: S2+LT2 | {_fp(m_A["CAGR_IS"])} | {_fp(m_A["CAGR_OOS"])} '
        f'| {_ff(m_A["Sharpe_OOS"])} | {_fp(m_A["MaxDD_FULL"])} | {_fp(m_A["Worst5Y"])} |'
    )
    lines.append(
        f'| NAV_B: DH Dyn [A] TQQQ | {_fp(m_B["CAGR_IS"])} | {_fp(m_B["CAGR_OOS"])} '
        f'| {_ff(m_B["Sharpe_OOS"])} | {_fp(m_B["MaxDD_FULL"])} | {_fp(m_B["Worst5Y"])} |'
    )
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 2. 結果テーブル')
    lines.append('')
    hdr = ('| w (NAV_A比重) | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD_FULL '
           '| Worst10Y★ | P10_5Y | Worst5Y | IS−OOS gap |')
    sep = ('|---:|--------:|---------:|-----------:|-----------:'
           '|----------:|-------:|--------:|-----------:|')
    lines.append(hdr)
    lines.append(sep)
    for r in results:
        marker = ' ← REF' if abs(r['w'] - 0.5) < 1e-9 else ''
        lines.append(
            f'| {r["w"]:.1f}{marker} '
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
    lines.append('---')
    lines.append('')
    lines.append('## 3. サニティチェック')
    lines.append('')
    sanity_tag = '✅ 一致（±0.10 pp 以内）' if sanity_ok_A else f'⚠️ 乖離 {sanity_diff_pp_A:+.2f} pp'
    lines.append(f'- NAV_A (S2+LT2) CAGR_OOS: **{m_A["CAGR_OOS"]*100:+.2f}%**')
    lines.append(f'- B1 参照値: **+31.16%**')
    lines.append(f'- 差分: **{sanity_diff_pp_A:+.2f} pp** → {sanity_tag}')
    if sanity_warn_forced:
        lines.append('- ⚠️ サニティ不一致により総合判定を強制 WARN 降格')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 4. 判定（w=0.5 基準）')
    lines.append('')
    lines.append('### 判定基準（事前定義）')
    lines.append('')
    lines.append('| 基準 | 条件 | 結果 |')
    lines.append('|------|------|------|')

    r_w05 = next((r for r in results if abs(r['w'] - 0.5) < 1e-9), None)
    if r_w05:
        crit1 = r_w05['MaxDD_FULL'] > -0.50
        crit2 = r_w05['Sharpe_OOS'] >= 0.808
        crit3 = r_w05['CAGR_OOS'] >= 0.220
        crit4 = r_w05['Worst5Y'] >= REF_WORST5Y_A

        lines.append(f'| (i) MaxDD < −50.0% | {r_w05["MaxDD_FULL"]*100:+.2f}% | '
                     f'{"✅ PASS" if crit1 else "❌ FAIL"} |')
        lines.append(f'| (ii) Sharpe_OOS ≥ 0.808 | {r_w05["Sharpe_OOS"]:+.3f} | '
                     f'{"✅ PASS" if crit2 else "❌ FAIL"} |')
        lines.append(f'| (iii) CAGR_OOS ≥ +22.0% | {r_w05["CAGR_OOS"]*100:+.2f}% | '
                     f'{"✅ PASS" if crit3 else "❌ FAIL"} |')
        lines.append(f'| (iv) Worst5Y ≥ {REF_WORST5Y_A*100:+.2f}% (S2+LT2 比同等以上) | '
                     f'{r_w05["Worst5Y"]*100:+.2f}% | '
                     f'{"✅ PASS" if crit4 else "❌ FAIL"} |')
        lines.append('')

        if sanity_warn_forced:
            verdict = 'WARN (サニティ不一致)'
        elif crit1 and crit2 and crit3 and crit4:
            verdict = 'PASS — アンサンブルは MaxDD 改善を達成。CURRENT_BEST 候補として検討'
        elif crit2 and crit3:
            verdict = 'WARN — MaxDD 改善が基準に未達。S2+LT2 単体を維持'
        else:
            verdict = 'FAIL — アンサンブルは Sharpe_OOS / CAGR_OOS を大幅に低下させる。採用しない'
    else:
        crit1 = crit2 = crit3 = crit4 = False
        verdict = 'WARN'

    lines.append(f'**総合判定: {verdict}**')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 5. 考察')
    lines.append('')
    lines.append('### アンサンブル効果の分析')
    lines.append('')
    if r_w05:
        maxdd_imp = (r_w05['MaxDD_FULL'] - m_A['MaxDD_FULL']) * 100
        sharpe_chg = r_w05['Sharpe_OOS'] - m_A['Sharpe_OOS']
        cagr_chg = (r_w05['CAGR_OOS'] - m_A['CAGR_OOS']) * 100
        worst5y_chg = (r_w05['Worst5Y'] - m_A['Worst5Y']) * 100
        lines.append(f'- MaxDD 変化 (w=0.5 vs S2+LT2 単体): **{maxdd_imp:+.2f} pp**')
        lines.append(f'- Sharpe_OOS 変化: **{sharpe_chg:+.3f}**')
        lines.append(f'- CAGR_OOS 変化: **{cagr_chg:+.2f} pp**')
        lines.append(f'- Worst5Y 変化: **{worst5y_chg:+.2f} pp**')
        lines.append('')
        if maxdd_imp > 5.0:
            lines.append('- **解釈**: アンサンブルは MaxDD を有意に改善。CFD 戦略と TQQQ の相関が '
                         '低い局面でリスク低減が効いている。')
        elif maxdd_imp > 0:
            lines.append('- **解釈**: MaxDD は若干改善したが基準 (-50%) に未達。'
                         'CFD と TQQQ は同じ NASDAQ に連動するため相関が高く分散効果が限定的。')
        else:
            lines.append('- **解釈**: MaxDD が改善しなかった。CFD と TQQQ のドローダウン発生時期が '
                         '重複しており分散効果が発揮できていない。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 6. 結論')
    lines.append('')
    if 'PASS' in verdict:
        lines.append('**E1 アンサンブルは MaxDD を基準以下に改善した。**')
        lines.append('B2 (k_lt=0.5 最適確認) と合わせて CURRENT_BEST_STRATEGY.md 更新の候補。')
    elif 'WARN' in verdict and 'サニティ' not in verdict:
        lines.append('**E1 アンサンブルは MaxDD 改善が不十分。** '
                     'CFD と TQQQ は NASDAQ 連動性が高く、ドローダウン期間が重複する。')
        lines.append('S2_VZGated+LT2 単体 (現行ベスト) を維持することを推奨。')
    else:
        lines.append('**E1 アンサンブルは採用しない。** S2_VZGated+LT2 単体を現行ベストとして維持する。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 7. 再現コマンド')
    lines.append('')
    lines.append('```')
    lines.append('python -X utf8 src/e1_ensemble.py')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/e1_ensemble.py`*')
    lines.append('*参照: `B1_S2_LT2_2026-05-21.md`, `B2_KLT_SWEEP_2026-05-21.md`, '
                 '`CURRENT_BEST_STRATEGY.md`*')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('E1: アンサンブル — S2+LT2 (CFD) × DH Dyn [A] TQQQ')
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

    # S5: LT2 シグナル (k_lt=0.5 固定)
    print('Building LT2 signal (N=750, k=0.5) ...')
    lt_sig  = build_lt_signal(close, 'LT2', 750)
    lt_bias = signal_to_bias(lt_sig, 0.5)
    assert len(lev_A) == len(lt_bias)
    lev_mod = apply_lt_mode_b(lev_A, lt_bias, l_min=0.0, l_max=1.0)
    print(f'  lt_bias range: [{lt_bias.min():+.3f}, {lt_bias.max():+.3f}]')

    # S6: NAV_A — S2+LT2 (CFD, k_lt=0.5)
    print('\nBuilding NAV_A: S2+LT2 (CFD, k_lt=0.5) ...')
    nav_A = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_A = calc_all_metrics(nav_A, dates, n_trades_yr)
    m_A['n_trades_yr'] = n_trades_yr
    print(f'  NAV_A CAGR_OOS={m_A["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m_A["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={m_A["MaxDD_FULL"]*100:+.2f}%')

    # S7: NAV_B — DH Dyn [A] TQQQ (lev_A, LT2 なし)
    print('Building NAV_B: DH Dyn [A] TQQQ (no LT2) ...')
    nav_B = build_nav(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr_daily=sofr, apply_tqqq_sofr=True,
    )
    m_B = calc_all_metrics(nav_B, dates, n_trades_yr)
    m_B['n_trades_yr'] = n_trades_yr
    print(f'  NAV_B CAGR_OOS={m_B["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m_B["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={m_B["MaxDD_FULL"]*100:+.2f}%')

    # S8: サニティチェック
    print()
    print('--- Sanity Check ---')
    sanity_diff_pp_A = (m_A['CAGR_OOS'] - REF_CAGR_OOS_A) * 100
    sanity_ok_A = abs(sanity_diff_pp_A) <= 0.10
    sanity_warn_forced = not sanity_ok_A
    print(f'[SANITY] NAV_A CAGR_OOS = {m_A["CAGR_OOS"]*100:+.2f}%  '
          f'(B1 ref +31.16%, diff {sanity_diff_pp_A:+.2f} pp)')
    if sanity_ok_A:
        print('  [OK] within 0.10 pp tolerance.')
    else:
        print('  [WARN] diff > 0.10 pp — 再現性を確認。総合判定を強制 WARN 降格。')

    sanity_ok_B = m_B['CAGR_OOS'] >= REF_CAGR_OOS_B_MIN
    print(f'[INFO] NAV_B CAGR_OOS = {m_B["CAGR_OOS"]*100:+.2f}%  '
          f'(loose check ≥ {REF_CAGR_OOS_B_MIN*100:.0f}%: {"OK" if sanity_ok_B else "WARN"})')

    # S9: w スイープ
    r_A = nav_A.pct_change().fillna(0)
    r_B = nav_B.pct_change().fillna(0)
    results = []
    print('\n--- w sweep ---')
    for w in W_GRID:
        daily_ret = w * r_A + (1.0 - w) * r_B
        nav_ens = (1 + pd.Series(daily_ret.values, index=dates.index)).cumprod()
        m = calc_all_metrics(nav_ens, dates, n_trades_yr)
        m['w'] = w
        m['n_trades_yr'] = n_trades_yr
        results.append(m)
        ref_tag = ' [REF]' if abs(w - 0.5) < 1e-9 else ''
        print(
            f'  w={w:.1f}{ref_tag}: CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
            f'Sharpe_OOS={m["Sharpe_OOS"]:+.3f}  '
            f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
            f'Worst5Y={m["Worst5Y"]*100:+.2f}%  '
            f'IS-OOS gap={m["IS_OOS_gap"]*100:+.2f}pp'
        )

    # S10: コンソール結果テーブル
    print()
    print('=' * 120)
    print('E1: Ensemble Results')
    print('=' * 120)
    hdr = (f'{"w":>5}  {"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"MaxDD":>9}  {"Worst10Y★":>10}  {"P10_5Y":>8}  {"Worst5Y":>8}  {"IS-OOS":>9}')
    print(hdr)
    print('-' * 120)
    for label, m in [('NAV_A', m_A), ('NAV_B', m_B)]:
        print(
            f'{label:>5}'
            f'  {m["CAGR_IS"]*100:>+8.2f}%'
            f'  {m["CAGR_OOS"]*100:>+8.2f}%'
            f'  {m["Sharpe_OOS"]:>+10.3f}'
            f'  {m["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {m["Worst10Y_star"]*100:>+9.2f}%'
            f'  {m["P10_5Y"]*100:>+7.2f}%'
            f'  {m["Worst5Y"]*100:>+7.2f}%'
            f'  {m["IS_OOS_gap"]*100:>+8.2f} pp'
        )
    print('-' * 120)
    for r in results:
        ref_tag = ' [REF]' if abs(r['w'] - 0.5) < 1e-9 else ''
        print(
            f'{r["w"]:>5.1f}{ref_tag}'
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
    rows = []
    for label, m in [('NAV_A_S2+LT2', m_A), ('NAV_B_DHDynTQQQ', m_B)]:
        rows.append({
            'label':         label,
            'w':             np.nan,
            'CAGR_IS':       m['CAGR_IS'],
            'CAGR_OOS':      m['CAGR_OOS'],
            'Sharpe_OOS':    m['Sharpe_OOS'],
            'MaxDD_FULL':    m['MaxDD_FULL'],
            'Worst10Y_star': m['Worst10Y_star'],
            'P10_5Y':        m['P10_5Y'],
            'Worst5Y':       m['Worst5Y'],
            'IS_OOS_gap':    m['IS_OOS_gap'],
        })
    for r in results:
        rows.append({
            'label':         f'Ens_w{r["w"]:.1f}',
            'w':             r['w'],
            'CAGR_IS':       r['CAGR_IS'],
            'CAGR_OOS':      r['CAGR_OOS'],
            'Sharpe_OOS':    r['Sharpe_OOS'],
            'MaxDD_FULL':    r['MaxDD_FULL'],
            'Worst10Y_star': r['Worst10Y_star'],
            'P10_5Y':        r['P10_5Y'],
            'Worst5Y':       r['Worst5Y'],
            'IS_OOS_gap':    r['IS_OOS_gap'],
        })
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(BASE, 'e1_ensemble_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    # S12: Markdown レポート
    md = generate_report(m_A, m_B, results,
                         sanity_ok_A, sanity_diff_pp_A, sanity_warn_forced)
    md_path = os.path.join(BASE, 'E1_ENSEMBLE_2026-05-21.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
