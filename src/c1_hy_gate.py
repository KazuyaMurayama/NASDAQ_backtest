"""
C1: VIX ゲート — 高ストレス時の CFD レバレッジ上限制御
======================================================
目的: VIX が高い局面で CFD レバレッジの上限 (l_max) を制限し、
     Worst5Y CAGR を改善する。

データ:
  - VIX 実値: data/vixcls_daily.csv (1990-01-02〜)
  - VIX プロキシ: 実現ボラ由来 (1974-1990 の穴埋め、calc_vix_proxy 流用)
  - HY spread: data/hy_spread_daily.csv (2023-05-22〜) — 期間が短いため参考値扱い

ゲート方式 (案 a_lmax):
  stress_mask[t] = vix_prev[t] > threshold    # shift(1) でルックアヘッド回避
  L_c1[t] = min(L_s2[t], l_max_eff[t])
  l_max_eff[t] = l_max_stress if stress_mask else l_max_base (= 7.0)

スイープケース:
  threshold ∈ {25, 30}  x  l_max_stress ∈ {3.0, 4.0}  = 4 ケース
  + baseline (ゲートなし)

サニティ:
  Case 0 (baseline) の CAGR_OOS は B1 (k_lt=0.5) の +31.16% に ±0.10pp 以内。

事前定義判定基準 (PASS):
  (i)  Worst5Y > -1.0%    (REF -2.83% より ≥ +1.83pp 改善)
  (ii) MaxDD ≥ -62.0%     (REF -59.45% から ≤ +2.5pp 悪化まで許容)
  (iii) Sharpe_OOS ≥ 0.808 (REF 0.858 から ≤ -0.050 低下)
  (iv) CAGR_OOS ≥ +25.0%  (REF +31.16% から ≤ -6.16pp 低下を許容)

出力:
  - c1_hy_gate_results.csv
  - C1_HY_GATE_2026-05-21.md
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
from test_vix_integration import calc_vix_proxy

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
# スイープケース: (case_id, threshold, l_max_stress)
SWEEP_CASES = [
    ('baseline',    None, None),   # S2+LT2 ゲートなし (サニティ)
    ('vix30_lmax4', 30,   4.0),
    ('vix30_lmax3', 30,   3.0),
    ('vix25_lmax4', 25,   4.0),
    ('vix25_lmax3', 25,   3.0),
]

L_MAX_BASE = 7.0

# サニティ参照値
REF_CAGR_OOS_BASE    = 0.3116   # B1 S2+LT2 k_lt=0.5
REF_SHARPE_OOS_BASE  = 0.858
REF_MAXDD_BASE       = -0.5945
REF_WORST5Y_BASE     = -0.0283

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')


# ---------------------------------------------------------------------------
# VIX 系列の構築
# ---------------------------------------------------------------------------

def build_vix_series(ret: pd.Series, dates: pd.Series) -> pd.Series:
    """VIX 実値 (1990+) と VIX プロキシ (pre-1990) を連結した日次系列を返す。

    Returns: pd.Series, index は dates.index と対応、単位は VIX ポイント (%) 。
    """
    # 1. VIX プロキシ（全期間カバー）
    vix_proxy = calc_vix_proxy(ret)
    vix_proxy.index = dates.index

    # 2. 実 VIX ロード
    vix_path = os.path.join(DATA_DIR, 'vixcls_daily.csv')
    vix_df = pd.read_csv(vix_path, parse_dates=['Date'])
    vix_df = vix_df.rename(columns={'vix': 'VIX'})

    # dates と結合
    dates_df = pd.DataFrame({'Date': dates.values, 'idx': dates.index})
    merged = dates_df.merge(vix_df, on='Date', how='left').set_index('idx')
    real_vix = merged['VIX']
    real_vix.index = dates.index

    # 3. マージ: 実値優先、NaN は proxy で補完
    combined = real_vix.where(real_vix.notna(), other=vix_proxy)

    # 4. 残 NaN (VIX proxy がゼロになる warmup 期間) を proxy で前方補完
    combined = combined.fillna(vix_proxy)

    return combined


def build_stress_mask(vix_series: pd.Series, threshold: float) -> pd.Series:
    """前日の VIX > threshold を stress フラグとする (ルックアヘッド回避)。"""
    return (vix_series.shift(1) > threshold).fillna(False)


def apply_lmax_gate(L_s2: pd.Series, stress_mask: pd.Series,
                    l_max_stress: float, l_max_base: float = L_MAX_BASE) -> pd.Series:
    """stress_mask が True の日は L_s2 を l_max_stress でキャップ。"""
    l_max_eff = np.where(stress_mask.values, l_max_stress, l_max_base)
    return pd.Series(
        np.minimum(L_s2.values, l_max_eff),
        index=L_s2.index,
    )


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
                    sanity_warn_forced: bool, vix_stats: dict) -> str:
    today = '2026-05-21'
    lines = []
    lines.append('# C1: VIX ゲート — 高ストレス時の CFD レバレッジ上限制御')
    lines.append('')
    lines.append(f'**実行日**: {today}')
    lines.append('**目的**: VIX > 閾値 の局面で CFD レバ上限 (l_max) を制限し、Worst5Y CAGR を改善する')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 1. セットアップ')
    lines.append('')
    lines.append('### VIX 系列の構成')
    lines.append('')
    lines.append('| 期間 | データソース |')
    lines.append('|---|---|')
    lines.append(f'| 1974-01-02〜1989-12-31 | VIX プロキシ（実現ボラ由来） |')
    lines.append(f'| 1990-01-02〜{today} | `data/vixcls_daily.csv`（実 VIX） |')
    lines.append(f'| HY spread | `data/hy_spread_daily.csv`（2023-05-22〜、参考値） |')
    lines.append('')
    lines.append('### VIX 統計')
    lines.append('')
    for stat_label, stat_val in vix_stats.items():
        lines.append(f'- {stat_label}: {stat_val}')
    lines.append('')
    lines.append('### ゲートロジック')
    lines.append('')
    lines.append('```')
    lines.append('stress_mask[t] = vix_prev[t] > threshold   # shift(1) でルックアヘッド回避')
    lines.append('l_max_eff[t]   = l_max_stress if stress_mask[t] else 7.0')
    lines.append('L_c1[t]        = min(L_s2[t], l_max_eff[t])')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 2. 結果テーブル')
    lines.append('')
    hdr = ('| ケース | threshold | l_max_stress | gate発動率 | CAGR_IS | CAGR_OOS | Sharpe_OOS '
           '| MaxDD | Worst10Y★ | P10_5Y | Worst5Y | IS−OOS gap |')
    sep = ('|---|---:|---:|---:|--------:|---------:|-----------:|-----------:'
           '|----------:|-------:|--------:|-----------:|')
    lines.append(hdr)
    lines.append(sep)
    for r in results:
        base_tag = ' (REF)' if r['case_id'] == 'baseline' else ''
        thr_str  = f'{r["threshold"]}' if r['threshold'] is not None else '—'
        lmax_str = f'{r["l_max_stress"]}' if r['l_max_stress'] is not None else '—'
        gate_str = f'{r["gate_active_pct"]*100:.1f}%' if r['gate_active_pct'] is not None else '—'
        lines.append(
            f'| {r["case_id"]}{base_tag} '
            f'| {thr_str} '
            f'| {lmax_str} '
            f'| {gate_str} '
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
    r_base = next((r for r in results if r['case_id'] == 'baseline'), None)
    if r_base:
        sanity_tag = '✅ 一致（±0.10 pp 以内）' if sanity_ok else f'⚠️ 乖離 {sanity_diff_pp:+.2f} pp'
        lines.append(f'- baseline CAGR_OOS: **{r_base["CAGR_OOS"]*100:+.2f}%**')
        lines.append(f'- B1 参照値: **+31.16%**')
        lines.append(f'- 差分: **{sanity_diff_pp:+.2f} pp** → {sanity_tag}')
    if sanity_warn_forced:
        lines.append('- ⚠️ サニティ不一致により総合判定を強制 WARN 降格')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 4. 判定（各ケースの PASS/FAIL）')
    lines.append('')
    lines.append('### 判定基準（事前定義）')
    lines.append('')
    lines.append('| 基準 | 条件 |')
    lines.append('|------|------|')
    lines.append('| (i)  Worst5Y  | > −1.0%     （REF −2.83% より ≥ +1.83pp 改善） |')
    lines.append('| (ii) MaxDD    | ≥ −62.0%    （REF −59.45% から ≤ +2.5pp 悪化許容） |')
    lines.append('| (iii) Sharpe_OOS | ≥ 0.808  （REF 0.858 から ≤ −0.050 低下） |')
    lines.append('| (iv) CAGR_OOS | ≥ +25.0%    （REF +31.16% から ≤ −6.16pp 低下許容） |')
    lines.append('')
    lines.append('### ケース別判定')
    lines.append('')
    lines.append('| ケース | (i) Worst5Y | (ii) MaxDD | (iii) Sharpe | (iv) CAGR | 総合 |')
    lines.append('|---|---|---|---|---|---|')

    best_case = None
    best_worst5y = REF_WORST5Y_BASE  # baseline

    for r in results:
        if r['case_id'] == 'baseline':
            continue
        c1 = r['Worst5Y'] > -0.010
        c2 = r['MaxDD_FULL'] >= -0.620
        c3 = r['Sharpe_OOS'] >= 0.808
        c4 = r['CAGR_OOS'] >= 0.250

        if sanity_warn_forced:
            verdict = '⚠️ WARN'
        elif c1 and c2 and c3 and c4:
            verdict = '✅ PASS'
            if r['Worst5Y'] > best_worst5y:
                best_worst5y = r['Worst5Y']
                best_case = r['case_id']
        elif (c2 and c3 and c4):
            verdict = '⚠️ WARN'
        else:
            verdict = '❌ FAIL'

        lines.append(
            f'| {r["case_id"]} '
            f'| {"✅" if c1 else "❌"} {r["Worst5Y"]*100:+.2f}% '
            f'| {"✅" if c2 else "❌"} {r["MaxDD_FULL"]*100:+.2f}% '
            f'| {"✅" if c3 else "❌"} {r["Sharpe_OOS"]:+.3f} '
            f'| {"✅" if c4 else "❌"} {r["CAGR_OOS"]*100:+.2f}% '
            f'| {verdict} |'
        )
    lines.append('')

    if best_case:
        lines.append(f'**最良ケース: `{best_case}` — Worst5Y={best_worst5y*100:+.2f}% で全基準 PASS**')
    else:
        lines.append('**全ケース FAIL または WARN — Worst5Y 目標 (> -1.0%) を満たすケースなし**')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 5. 考察')
    lines.append('')
    lines.append('### VIX ゲートの発動特性')
    lines.append('')
    lines.append('VIX > 30 の局面は主に以下の市場ストレス期間に集中する:')
    lines.append('- 1987年ブラックマンデー (VIX プロキシ)')
    lines.append('- 1998年 LTCM 危機')
    lines.append('- 2001-2002年 ドットコム崩壊')
    lines.append('- 2007-2009年 GFC')
    lines.append('- 2020年 COVID ショック')
    lines.append('- 2022年 NASDAQ 急落期')
    lines.append('')
    lines.append('これらの期間に l_max を制限することでドローダウン・最悪5年 CAGR が改善する一方、')
    lines.append('好調期の CFD レバ機会損失が CAGR_OOS を押し下げる。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 6. 結論')
    lines.append('')
    if best_case:
        lines.append(f'**C1 `{best_case}` は Worst5Y 目標 > -1.0% を達成し全基準 PASS。**')
        lines.append('B2 (k_lt=0.5 確認) と合わせて S2+LT2+C1 の統合評価を実施することを推奨。')
    else:
        lines.append('**C1 の全ケースは Worst5Y の改善目標 (> -1.0%) に届かなかった。**')
        lines.append('ゲート閾値を下げる (例: VIX > 20) か、l_max_stress をさらに絞る '
                     '(例: 2.5) ことで改善できる可能性があるが、CAGR_OOS の低下が懸念される。')
        lines.append('S2_VZGated+LT2 単体 (現行ベスト) を維持する。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 7. 再現コマンド')
    lines.append('')
    lines.append('```')
    lines.append('python -X utf8 src/c1_hy_gate.py')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/c1_hy_gate.py`*')
    lines.append('*参照: `B1_S2_LT2_2026-05-21.md`, `B2_KLT_SWEEP_2026-05-21.md`, '
                 '`E1_ENSEMBLE_2026-05-21.md`, `CURRENT_BEST_STRATEGY.md`*')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('C1: VIX ゲート — 高ストレス時の CFD レバレッジ上限制御')
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

    # S5: LT2 シグナル (k_lt=0.5)
    print('Building LT2 signal (N=750, k=0.5) ...')
    lt_sig  = build_lt_signal(close, 'LT2', 750)
    lt_bias = signal_to_bias(lt_sig, 0.5)
    assert len(lev_A) == len(lt_bias)
    lev_mod = apply_lt_mode_b(lev_A, lt_bias, l_min=0.0, l_max=1.0)

    # S6: VIX 系列の構築
    print('\nBuilding VIX series...')
    vix_series = build_vix_series(ret, dates)
    real_vix_mask = pd.read_csv(
        os.path.join(DATA_DIR, 'vixcls_daily.csv'), parse_dates=['Date']
    ).rename(columns={'vix': 'VIX'})
    n_real = vix_series[dates >= pd.Timestamp('1990-01-02')].notna().sum()
    n_proxy = vix_series[dates < pd.Timestamp('1990-01-02')].count()
    print(f'  VIX real (1990+): {n_real:,} days')
    print(f'  VIX proxy (pre-1990): {n_proxy:,} days')
    print(f'  VIX full period: min={vix_series.min():.1f}  median={vix_series.median():.1f}  '
          f'max={vix_series.max():.1f}')

    vix_stats = {
        'VIX minimum': f'{vix_series.min():.1f}',
        'VIX median': f'{vix_series.median():.1f}',
        'VIX maximum': f'{vix_series.max():.1f}',
        'VIX > 30 日数': f'{(vix_series > 30).sum():,} 日 ({(vix_series > 30).mean()*100:.1f}%)',
        'VIX > 25 日数': f'{(vix_series > 25).sum():,} 日 ({(vix_series > 25).mean()*100:.1f}%)',
    }

    # S7: ケーススイープ
    results = []
    print('\n--- Case sweep ---')
    for case_id, threshold, l_max_stress in SWEEP_CASES:
        print(f'  [{case_id}]')
        if threshold is None:
            # baseline: L_s2 そのまま
            L_used = L_s2.copy()
            gate_active_pct = None
        else:
            stress_mask = build_stress_mask(vix_series, threshold)
            gate_active_pct = float(stress_mask.mean())
            L_used = apply_lmax_gate(L_s2, stress_mask, l_max_stress)
            print(f'    gate_active_pct: {gate_active_pct*100:.1f}%  '
                  f'L_used range: [{L_used.min():.1f}, {L_used.max():.1f}]')

        nav = build_nav_strategy(
            close, lev_mod, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=L_used.values, cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, n_trades_yr)
        m['case_id']          = case_id
        m['threshold']        = threshold
        m['l_max_stress']     = l_max_stress
        m['gate_active_pct']  = gate_active_pct
        m['n_trades_yr']      = n_trades_yr
        results.append(m)
        print(
            f'    CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  Sharpe_OOS={m["Sharpe_OOS"]:+.3f}  '
            f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  Worst5Y={m["Worst5Y"]*100:+.2f}%  '
            f'IS-OOS gap={m["IS_OOS_gap"]*100:+.2f}pp'
        )

    # S8: サニティチェック
    print()
    print('--- Sanity Check ---')
    r_base = next(r for r in results if r['case_id'] == 'baseline')
    sanity_diff_pp = (r_base['CAGR_OOS'] - REF_CAGR_OOS_BASE) * 100
    sanity_ok = abs(sanity_diff_pp) <= 0.10
    sanity_warn_forced = not sanity_ok
    print(f'[SANITY] baseline CAGR_OOS = {r_base["CAGR_OOS"]*100:+.2f}%  '
          f'(B1 ref +31.16%, diff {sanity_diff_pp:+.2f} pp)')
    if sanity_ok:
        print('  [OK] within 0.10 pp tolerance.')
    else:
        print('  [WARN] diff > 0.10 pp — 強制 WARN 降格。')

    # S9: コンソール結果テーブル
    print()
    print('=' * 130)
    print('C1: VIX Gate Results')
    print('=' * 130)
    hdr = (f'{"Case":<20}  {"thr":>4}  {"lmax":>5}  {"gate%":>6}  '
           f'{"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"MaxDD":>9}  {"Worst10Y★":>10}  {"Worst5Y":>8}  {"IS-OOS":>9}')
    print(hdr)
    print('-' * 130)
    for r in results:
        thr_s  = f'{r["threshold"]}' if r['threshold'] is not None else '—'
        lmax_s = f'{r["l_max_stress"]:.1f}' if r['l_max_stress'] is not None else '—'
        gate_s = f'{r["gate_active_pct"]*100:.1f}%' if r['gate_active_pct'] is not None else '—'
        print(
            f'{r["case_id"]:<20}'
            f'  {thr_s:>4}'
            f'  {lmax_s:>5}'
            f'  {gate_s:>6}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f}'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
            f'  {r["Worst10Y_star"]*100:>+9.2f}%'
            f'  {r["Worst5Y"]*100:>+7.2f}%'
            f'  {r["IS_OOS_gap"]*100:>+8.2f} pp'
        )
    print('=' * 130)

    # S10: CSV 保存
    df_out = pd.DataFrame([{
        'case_id':        r['case_id'],
        'threshold':      r['threshold'],
        'l_max_stress':   r['l_max_stress'],
        'gate_active_pct': r['gate_active_pct'],
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
    csv_path = os.path.join(BASE, 'c1_hy_gate_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    # S11: Markdown レポート
    md = generate_report(results, sanity_ok, sanity_diff_pp, sanity_warn_forced, vix_stats)
    md_path = os.path.join(BASE, 'C1_HY_GATE_2026-05-21.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
