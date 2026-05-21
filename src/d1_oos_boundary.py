"""
D1: IS/OOS 境界感度分析
========================
目的: IS/OOS 分割点を {2018, 2019, 2020, 2021-05(REF), 2022} に変えても
     S2_VZGated + LT2-N750-k0.5-modeB の OOS 指標が安定していることを確認する。

設計:
  - NAV は 1 回だけ構築（パラメータ完全固定）
  - 指標計算関数 calc_7metrics_at() で分割点だけを変えて再評価
  - cfd_leverage_backtest.py の定数には一切触らない

サニティ:
  2021-05 分割の CAGR_OOS が B1 (+31.16%) に ±0.10pp 以内で一致すること。

事前定義判定基準 (PASS):
  (i)  全5分割中 CAGR_OOS の最小値 ≥ +18.0%
  (ii) 全5分割中 Sharpe_OOS の最小値 ≥ 0.55
  (iii) 全5分割中 CAGR_OOS の標準偏差 ≤ 8.0pp
  (iv) 全5分割中 |IS-OOS gap| が全ケースで ≤ 7.0pp

出力:
  - d1_oos_boundary_results.csv
  - D1_OOS_BOUNDARY_2026-05-21.md
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
    IS_START, IS_END, OOS_START, FULL_START, FULL_END,
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
# 分割点グリッド: (label, is_end_raw, oos_start_raw)
SPLIT_GRID_RAW = [
    ('2018', '2017-12-29', '2018-01-02'),
    ('2019', '2018-12-31', '2019-01-02'),
    ('2020', '2019-12-31', '2020-01-02'),
    ('2021-05 (REF)', IS_END,  OOS_START),   # 現行分割 (サニティ)
    ('2022', '2021-12-31', '2022-01-03'),
]

# サニティ: 2021-05 分割の CAGR_OOS (B1 確定値)
REF_CAGR_OOS     = 0.3116
REF_SHARPE_OOS   = 0.858

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# ヘルパー: 営業日スナップ
# ---------------------------------------------------------------------------

def snap_backward(dates: pd.Series, ts_str: str) -> pd.Timestamp:
    """ts_str 以前で最後の営業日を返す。"""
    ts = pd.Timestamp(ts_str)
    valid = dates[dates <= ts]
    if len(valid) == 0:
        return dates.iloc[0]
    return pd.Timestamp(valid.iloc[-1])


def snap_forward(dates: pd.Series, ts_str: str) -> pd.Timestamp:
    """ts_str 以降で最初の営業日を返す。"""
    ts = pd.Timestamp(ts_str)
    valid = dates[dates >= ts]
    if len(valid) == 0:
        return dates.iloc[-1]
    return pd.Timestamp(valid.iloc[0])


# ---------------------------------------------------------------------------
# 引数化された指標計算関数
# ---------------------------------------------------------------------------

def calc_7metrics_at(nav: pd.Series, dates: pd.Series, trades_per_year: float,
                     is_start: str, is_end: pd.Timestamp,
                     oos_start: pd.Timestamp,
                     full_start: str = FULL_START,
                     full_end: str   = FULL_END) -> dict:
    """calc_7metrics と同等だが IS/OOS 境界を引数化したバージョン。
    Worst5Y / Worst10Y / WinRate / MaxDD_FULL は FULL ベースで変わらない。
    """
    def _mp(start, end):
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        if mask.sum() < 100:
            return {}
        ns  = nav[mask].copy() / nav[mask].iloc[0]
        r   = ns.pct_change().fillna(0)
        n   = len(ns); yrs = n / TRADING_DAYS
        if yrs <= 0 or ns.iloc[-1] <= 0:
            return {}
        cagr  = float(ns.iloc[-1]) ** (1 / yrs) - 1
        std   = r.std()
        sharpe = (r.mean() * TRADING_DAYS) / (std * np.sqrt(TRADING_DAYS)) if std > 0 else np.nan
        maxdd  = float((ns / ns.cummax() - 1).min())
        n_days = int(mask.sum())
        return {'CAGR': cagr, 'Sharpe': sharpe, 'MaxDD': maxdd, 'n_days': n_days}

    full = _mp(full_start, full_end)
    is_  = _mp(is_start,   str(is_end.date()))
    oos  = _mp(str(oos_start.date()), full_end)

    # FULL ベースの rolling 指標（分割点に依存しない）
    mask_f = (dates >= pd.Timestamp(full_start)) & (dates <= pd.Timestamp(full_end))
    ns_f   = nav[mask_f].copy() / nav[mask_f].iloc[0]

    w5 = w10 = np.nan
    if len(ns_f) >= TRADING_DAYS * 5:
        r5  = (ns_f / ns_f.shift(TRADING_DAYS * 5)) ** (1 / 5) - 1
        w5  = float(r5.min())
    if len(ns_f) >= TRADING_DAYS * 10:
        r10 = (ns_f / ns_f.shift(TRADING_DAYS * 10)) ** (1 / 10) - 1
        w10 = float(r10.min())

    dt_f   = dates[mask_f]
    ndf = pd.DataFrame({'nav': ns_f.values, 'dt': dt_f.values})
    ndf['year'] = pd.to_datetime(ndf['dt']).dt.year
    yn  = ndf.groupby('year')['nav'].last()
    wr  = float((yn.pct_change().dropna() > 0).mean())

    return {
        'CAGR_FULL':    full.get('CAGR',   np.nan),
        'CAGR_IS':      is_.get('CAGR',    np.nan),
        'CAGR_OOS':     oos.get('CAGR',    np.nan),
        'Sharpe_FULL':  full.get('Sharpe', np.nan),
        'Sharpe_IS':    is_.get('Sharpe',  np.nan),
        'Sharpe_OOS':   oos.get('Sharpe',  np.nan),
        'MaxDD_FULL':   full.get('MaxDD',  np.nan),
        'Worst5Y':      w5,
        'Worst10Y':     w10,
        'WinRate':      wr,
        'Trades_yr':    trades_per_year,
        'n_days_IS':    is_.get('n_days',  0),
        'n_days_OOS':   oos.get('n_days',  0),
    }


# ---------------------------------------------------------------------------
# ヘルパー: Worst10Y★ (calendar-year rolling)
# ---------------------------------------------------------------------------

def compute_p10_5y(nav, trading_days=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(trading_days * 5)) ** 0.2 - 1
    return float(r.dropna().quantile(0.10))


def compute_worst5y(nav, trading_days=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(trading_days * 5)) ** 0.2 - 1
    return float(r.dropna().min())


def calc_worst10y_star(nav, dates):
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, 10)
    return float(r10.min()) if len(r10) > 0 else np.nan


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
    today = '2026-05-21'
    lines = []
    lines.append('# D1: IS/OOS 境界感度分析')
    lines.append('')
    lines.append(f'**実行日**: {today}')
    lines.append('**目的**: IS/OOS 分割点を変えても S2_VZGated+LT2-N750-k0.5-modeB の OOS 指標が安定しているかを確認する')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 1. セットアップ')
    lines.append('')
    lines.append('### 分割点グリッド')
    lines.append('')
    lines.append('| 分割ラベル | IS 終端 | OOS 開始 | IS 長 (年) | OOS 長 (年) |')
    lines.append('|---|---|---|---:|---:|')
    for r in results:
        is_yr  = r['n_days_IS']  / TRADING_DAYS
        oos_yr = r['n_days_OOS'] / TRADING_DAYS
        ref_tag = ' ← REF' if 'REF' in r['split_label'] else ''
        lines.append(
            f'| {r["split_label"]}{ref_tag} | {r["is_end_eff"]} | {r["oos_start_eff"]} '
            f'| {is_yr:.1f} | {oos_yr:.1f} |'
        )
    lines.append('')
    lines.append('- パラメータは完全固定: S2_VZGated (tv=0.8, k_vz=0.3, gate_min=0.5, n=20, l_max=7.0) + LT2 (N=750, k_lt=0.5, modeB)')
    lines.append('- NAV は 1 本のみ構築。指標計算時に分割点を変えて再評価。')
    lines.append('- FULL ベース指標 (Worst5Y, Worst10Y★, WinRate, MaxDD_FULL) は全分割で同値。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 2. 結果テーブル')
    lines.append('')
    hdr = ('| 分割 | CAGR_IS | CAGR_OOS | Sharpe_OOS | IS-OOS gap '
           '| MaxDD (FULL) | Worst5Y (FULL) | Worst10Y★ (FULL) |')
    sep = ('|---|--------:|---------:|-----------:|----------:'
           '|-----------:|-------:|--------:|')
    lines.append(hdr)
    lines.append(sep)
    for r in results:
        ref_tag = ' ← REF' if 'REF' in r['split_label'] else ''
        lines.append(
            f'| {r["split_label"]}{ref_tag} '
            f'| {_fp(r["CAGR_IS"])} '
            f'| {_fp(r["CAGR_OOS"])} '
            f'| {_ff(r["Sharpe_OOS"])} '
            f'| {r["IS_OOS_gap"]*100:+.2f} pp '
            f'| {_fp(r["MaxDD_FULL"])} '
            f'| {_fp(r["Worst5Y"])} '
            f'| {_fp(r.get("Worst10Y_star", np.nan))} |'
        )
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 3. サニティチェック')
    lines.append('')
    r_ref = next((r for r in results if 'REF' in r['split_label']), None)
    if r_ref:
        sanity_tag = '✅ 一致（±0.10 pp 以内）' if sanity_ok else f'⚠️ 乖離 {sanity_diff_pp:+.2f} pp'
        lines.append(f'- 2021-05 分割 CAGR_OOS: **{r_ref["CAGR_OOS"]*100:+.2f}%**')
        lines.append(f'- B1 参照値: **+31.16%**')
        lines.append(f'- 差分: **{sanity_diff_pp:+.2f} pp** → {sanity_tag}')
    if sanity_warn_forced:
        lines.append('- ⚠️ サニティ不一致により総合判定を強制 WARN 降格')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 4. 判定')
    lines.append('')
    lines.append('### 判定基準（事前定義）')
    lines.append('')
    lines.append('| 基準 | 条件 |')
    lines.append('|------|------|')
    lines.append('| (i)   CAGR_OOS 最小値 | ≥ +18.0%（全5分割中） |')
    lines.append('| (ii)  Sharpe_OOS 最小値 | ≥ 0.55（全5分割中） |')
    lines.append('| (iii) CAGR_OOS 標準偏差 | ≤ 8.0pp（全5分割中） |')
    lines.append('| (iv)  |IS-OOS gap| 最大値 | ≤ 7.0pp（全5分割中） |')
    lines.append('')

    cagr_oos_vals = [r['CAGR_OOS'] for r in results if not np.isnan(r.get('CAGR_OOS', np.nan))]
    sharpe_oos_vals = [r['Sharpe_OOS'] for r in results if not np.isnan(r.get('Sharpe_OOS', np.nan))]
    gap_abs_vals = [abs(r['IS_OOS_gap']) for r in results if not np.isnan(r.get('IS_OOS_gap', np.nan))]

    crit1 = min(cagr_oos_vals, default=0) >= 0.180
    crit2 = min(sharpe_oos_vals, default=0) >= 0.55
    crit3 = np.std(cagr_oos_vals) <= 0.080 if len(cagr_oos_vals) >= 2 else True
    crit4 = max(gap_abs_vals, default=0) <= 0.070

    lines.append('### 計算値')
    lines.append('')
    lines.append('| 基準 | 計算値 | 結果 |')
    lines.append('|---|---|---|')
    lines.append(f'| (i) CAGR_OOS 最小値 | {min(cagr_oos_vals, default=np.nan)*100:+.2f}% | '
                 f'{"✅ PASS" if crit1 else "❌ FAIL"} |')
    lines.append(f'| (ii) Sharpe_OOS 最小値 | {min(sharpe_oos_vals, default=np.nan):+.3f} | '
                 f'{"✅ PASS" if crit2 else "❌ FAIL"} |')
    lines.append(f'| (iii) CAGR_OOS σ | {np.std(cagr_oos_vals)*100:.2f}pp | '
                 f'{"✅ PASS" if crit3 else "❌ FAIL"} |')
    lines.append(f'| (iv) |IS-OOS gap| 最大値 | {max(gap_abs_vals, default=np.nan)*100:.2f}pp | '
                 f'{"✅ PASS" if crit4 else "❌ FAIL"} |')
    lines.append('')

    if sanity_warn_forced:
        verdict = 'WARN (サニティ不一致)'
    elif crit1 and crit2 and crit3 and crit4:
        verdict = 'PASS — S2+LT2 は分割点に対してロバスト。CURRENT_BEST を維持'
    elif crit1 and crit2:
        verdict = 'WARN — 一部基準（分散 or gap）で軽微な懸念あり。注意して維持'
    else:
        verdict = 'FAIL — 分割点依存性あり。戦略の評価方法を再検討'

    lines.append(f'**総合判定: {verdict}**')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 5. 考察')
    lines.append('')
    lines.append('### 分割点と OOS 期間の特徴')
    lines.append('')
    lines.append('| 分割 | OOS 期間の主要イベント |')
    lines.append('|---|---|')
    lines.append('| 2018〜 | COVID (2020), NASDAQ急落 (2022), 高金利期 |')
    lines.append('| 2019〜 | COVID (2020), NASDAQ急落 (2022) |')
    lines.append('| 2020〜 | COVID 回復期〜, NASDAQ急落 (2022) |')
    lines.append('| 2021-05 (REF) | ポスト緩和期〜, NASDAQ急落 (2022) |')
    lines.append('| 2022〜 | 利上げ局面〜 (最も短い OOS、結果の解釈に注意) |')
    lines.append('')
    lines.append('> ⚠️ **注意**: 2022 分割は OOS が約4年と短く、サンプル数が少ない。')
    lines.append('> この分割の Sharpe_OOS / CAGR_OOS は変動が大きい可能性がある。')
    lines.append('')
    lines.append('### IS-OOS gap の解釈')
    lines.append('')
    lines.append('IS 期間が長くなる (2018 分割) ほど CAGR_IS は '
                 '1974-2018 の超長期成長を含むため高くなりやすく、gap が大きく見える。')
    lines.append('これは過適合の悪化ではなく構造的なバイアス。OOS 指標の絶対値で判断する。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 6. 結論')
    lines.append('')
    if 'PASS' in verdict:
        lines.append('**S2_VZGated + LT2-N750-k0.5-modeB は IS/OOS 分割点に対してロバスト。**')
        lines.append('全分割で Sharpe_OOS ≥ 0.55 かつ CAGR_OOS ≥ +18.0% を維持し、')
        lines.append('CURRENT_BEST_STRATEGY.md の変更は不要。F1 (資産配分スイープ) に進むことを推奨。')
    elif 'WARN' in verdict:
        lines.append('一部基準で軽微な懸念があるが、全分割で正の OOS リターンを維持している。')
        lines.append('現行ベスト戦略の継続を推奨するが、特定の分割点での脆弱性に注意する。')
    else:
        lines.append('**分割点依存性が確認された。** 2021-05 分割が特別に有利な可能性がある。')
        lines.append('より多くの分割点で検証するか、Walk-Forward Analysis への移行を検討すること。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 7. 再現コマンド')
    lines.append('')
    lines.append('```')
    lines.append('python -X utf8 src/d1_oos_boundary.py')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/d1_oos_boundary.py`*')
    lines.append('*参照: `B1_S2_LT2_2026-05-21.md`, `B2_KLT_SWEEP_2026-05-21.md`, '
                 '`CURRENT_BEST_STRATEGY.md`*')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('D1: IS/OOS 境界感度分析')
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
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)
    print(f'  L_s2 range: [{L_s2.min():.1f}, {L_s2.max():.1f}]  median={L_s2.median():.1f}')

    # S5: LT2 シグナル (k_lt=0.5)
    print('Building LT2 signal (N=750, k=0.5) ...')
    lt_sig  = build_lt_signal(close, 'LT2', 750)
    lt_bias = signal_to_bias(lt_sig, 0.5)
    assert len(lev_A) == len(lt_bias)
    lev_mod = apply_lt_mode_b(lev_A, lt_bias, l_min=0.0, l_max=1.0)

    # S6: NAV 構築（1回のみ）
    print('Building S2+LT2 NAV (once) ...')
    nav = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    worst10y_star_full = calc_worst10y_star(nav, dates)
    p10_5y_full   = compute_p10_5y(nav.values)
    worst5y_full  = compute_worst5y(nav.values)
    print(f'  NAV built. FULL worst10y★={worst10y_star_full*100:+.2f}%  worst5y={worst5y_full*100:+.2f}%')

    # S7: 分割点ループ
    print('\n--- Split boundary sweep ---')
    results = []
    for label, is_end_raw, oos_start_raw in SPLIT_GRID_RAW:
        is_end_eff    = snap_backward(dates, is_end_raw)
        oos_start_eff = snap_forward(dates, oos_start_raw)

        m = calc_7metrics_at(
            nav, dates, n_trades_yr,
            is_start=IS_START,
            is_end=is_end_eff,
            oos_start=oos_start_eff,
        )
        m['split_label']   = label
        m['is_end_eff']    = str(is_end_eff.date())
        m['oos_start_eff'] = str(oos_start_eff.date())
        m['IS_OOS_gap']    = m['CAGR_IS'] - m['CAGR_OOS']
        m['Worst10Y_star'] = worst10y_star_full   # FULL ベース (不変)
        m['P10_5Y']        = p10_5y_full          # FULL ベース (不変)
        m['Worst5Y']       = worst5y_full          # FULL ベース (不変)
        results.append(m)

        is_yr  = m['n_days_IS']  / TRADING_DAYS
        oos_yr = m['n_days_OOS'] / TRADING_DAYS
        print(
            f'  [{label:<18}] IS={is_yr:.1f}yr  OOS={oos_yr:.1f}yr  '
            f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
            f'Sharpe_OOS={m["Sharpe_OOS"]:+.3f}  '
            f'IS-OOS gap={m["IS_OOS_gap"]*100:+.2f}pp'
        )

    # S8: サニティチェック
    print()
    print('--- Sanity Check ---')
    r_ref = next(r for r in results if 'REF' in r['split_label'])
    sanity_diff_pp = (r_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    sanity_ok = abs(sanity_diff_pp) <= 0.10
    sanity_warn_forced = not sanity_ok
    print(f'[SANITY] 2021-05 (REF) CAGR_OOS = {r_ref["CAGR_OOS"]*100:+.2f}%  '
          f'(B1 ref +31.16%, diff {sanity_diff_pp:+.2f} pp)')
    if sanity_ok:
        print('  [OK] within 0.10 pp tolerance.')
    else:
        print('  [WARN] diff > 0.10 pp — 強制 WARN 降格。')

    # S9: 判定計算
    cagr_oos_vals   = [r['CAGR_OOS'] for r in results if not np.isnan(r.get('CAGR_OOS', np.nan))]
    sharpe_oos_vals = [r['Sharpe_OOS'] for r in results if not np.isnan(r.get('Sharpe_OOS', np.nan))]
    gap_abs_vals    = [abs(r['IS_OOS_gap']) for r in results if not np.isnan(r.get('IS_OOS_gap', np.nan))]

    crit1 = min(cagr_oos_vals, default=0) >= 0.180
    crit2 = min(sharpe_oos_vals, default=0) >= 0.55
    crit3 = np.std(cagr_oos_vals) <= 0.080 if len(cagr_oos_vals) >= 2 else True
    crit4 = max(gap_abs_vals, default=0) <= 0.070
    print(f'  (i)  CAGR_OOS min:  {min(cagr_oos_vals, default=np.nan)*100:+.2f}%  '
          f'(≥18.0%: {"PASS" if crit1 else "FAIL"})')
    print(f'  (ii) Sharpe_OOS min: {min(sharpe_oos_vals, default=np.nan):+.3f}  '
          f'(≥0.55: {"PASS" if crit2 else "FAIL"})')
    print(f'  (iii) CAGR_OOS σ:   {np.std(cagr_oos_vals)*100:.2f}pp  '
          f'(≤8.0pp: {"PASS" if crit3 else "FAIL"})')
    print(f'  (iv) |gap| max:     {max(gap_abs_vals, default=np.nan)*100:.2f}pp  '
          f'(≤7.0pp: {"PASS" if crit4 else "FAIL"})')

    # S10: コンソール結果テーブル
    print()
    print('=' * 120)
    print('D1: IS/OOS Boundary Sensitivity Results')
    print('=' * 120)
    hdr = (f'{"Split":<22}  {"IS(yr)":>7}  {"OOS(yr)":>7}  '
           f'{"CAGR_IS":>9}  {"CAGR_OOS":>9}  {"Sharpe_OOS":>11}  '
           f'{"IS-OOS gap":>11}  {"MaxDD":>9}')
    print(hdr)
    print('-' * 120)
    for r in results:
        is_yr  = r['n_days_IS']  / TRADING_DAYS
        oos_yr = r['n_days_OOS'] / TRADING_DAYS
        ref_tag = ' [REF]' if 'REF' in r['split_label'] else ''
        print(
            f'{r["split_label"]:<22}{ref_tag}'
            f'  {is_yr:>7.1f}'
            f'  {oos_yr:>7.1f}'
            f'  {r["CAGR_IS"]*100:>+8.2f}%'
            f'  {r["CAGR_OOS"]*100:>+8.2f}%'
            f'  {r["Sharpe_OOS"]:>+10.3f}'
            f'  {r["IS_OOS_gap"]*100:>+10.2f} pp'
            f'  {r["MaxDD_FULL"]*100:>+8.2f}%'
        )
    print('=' * 120)
    print(f'FULL ベース固定値: Worst5Y={worst5y_full*100:+.2f}%  '
          f'Worst10Y★={worst10y_star_full*100:+.2f}%  P10_5Y={p10_5y_full*100:+.2f}%  '
          f'MaxDD={results[0]["MaxDD_FULL"]*100:+.2f}%')

    # S11: CSV 保存
    df_out = pd.DataFrame([{
        'split_label':   r['split_label'],
        'is_end_eff':    r['is_end_eff'],
        'oos_start_eff': r['oos_start_eff'],
        'n_days_IS':     r['n_days_IS'],
        'n_days_OOS':    r['n_days_OOS'],
        'CAGR_IS':       r['CAGR_IS'],
        'CAGR_OOS':      r['CAGR_OOS'],
        'Sharpe_IS':     r['Sharpe_IS'],
        'Sharpe_OOS':    r['Sharpe_OOS'],
        'MaxDD_FULL':    r['MaxDD_FULL'],
        'IS_OOS_gap':    r['IS_OOS_gap'],
        'Worst5Y_FULL':  r['Worst5Y'],
        'Worst10Y_star_FULL': r['Worst10Y_star'],
        'P10_5Y_FULL':   r['P10_5Y'],
        'Trades_yr':     r['Trades_yr'],
    } for r in results])
    csv_path = os.path.join(BASE, 'd1_oos_boundary_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    # S12: Markdown レポート
    md = generate_report(results, sanity_ok, sanity_diff_pp, sanity_warn_forced)
    md_path = os.path.join(BASE, 'D1_OOS_BOUNDARY_2026-05-21.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
