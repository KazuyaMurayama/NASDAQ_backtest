"""
E2: S2+LT2-N1500(70%) + P05_HY×CPI(30%) ハイブリッド検証
========================================================
# Evaluation Standard: v1.1
# Cost Scenario: D
# IS: 1974-01-02〜2021-05-07 / OOS: 2021-05-08〜現在
# Metrics: 標準7 + WFA補助2 (docs/rules/08_evaluation-metrics.md)

目的:
  WFA Opusインサイト（旧評価指標）で提案された
  S2+LT2-N1500(70%) + P05_HY×CPI(30%) 固定ウェイトブレンドを
  EVALUATION_STANDARD v1.1 統一9指標で検証する。

注意:
  P05_HY×CPI は §1.3 参考値戦略（HY/CPI 異コストモデル）。
  ハイブリッドもこの制約を継承し、CAGR_OOS の直接比較は参考値扱い。
  Active 昇格不可（Shortlisted 上限）。

出力:
  - e2_hybrid_70_30_results.csv  (プロジェクトルート)
  - E2_HYBRID_70_30_2026-05-22.md (プロジェクトルート)
"""

import sys
import os
import types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _sweep_format import MD_HEADER_STRAT, fmt_row_strat, MD_WFA_NOTE  # §3.12 標準フォーマッタ

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
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNALS_PATH = os.path.join(BASE, 'data', 'timing_signals_raw.csv')
DATE_STR     = '2026-05-22'

S2_FIXED   = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                  l_min=1.0, l_max=7.0, step=0.5)
N_LT2      = 1500
K_LT2      = 0.5
W_S2LT2    = 0.70
W_P05      = 0.30

# サニティ参照値
REF_S2LT2_CAGR_OOS  = 0.3084   # B6 N=1500
REF_S2LT2_SHARPE    = 0.885
REF_P05_CAGR_OOS    = 0.1565   # STRATEGY_COMPARISON 参照値
REF_P05_SHARPE      = 0.667

IS_B   = 11916
OOS_B  = 1253
FULL_B = 13169

# ---------------------------------------------------------------------------
# P05 ゲートビルダー（p4_overfitting_check.py より）
# ---------------------------------------------------------------------------

def build_hy_gate(hy, z_thresh, slope):
    mu = hy.rolling(252, min_periods=126).mean()
    sd = hy.rolling(252, min_periods=126).std().clip(lower=0.01)
    z  = (hy - mu) / sd
    g  = (1.0 - np.maximum(0.0, z - z_thresh) * slope).clip(0.2, 1.0)
    return g.fillna(1.0)


def build_cpi_gate(cpi_yoy, cpi_accel, cpi_thresh, reduce_factor):
    infl_regime = ((cpi_yoy - cpi_thresh) / 5.0).clip(0.0, 1.0)
    accel_norm  = (cpi_accel / 2.0).clip(0.0, 1.0)
    g = (1.0 - reduce_factor * np.maximum(infl_regime, accel_norm)).clip(
        1.0 - reduce_factor, 1.0
    )
    return g.fillna(1.0)


def apply_gates(wn_A_arr, nas_gate=None, bond_gate=None):
    ones   = np.ones(len(wn_A_arr))
    g_nas  = np.where(np.isnan(nas_gate),  1.0, nas_gate)  if nas_gate  is not None else ones
    g_bond = np.where(np.isnan(bond_gate), 1.0, bond_gate) if bond_gate is not None else ones
    wn   = np.clip(wn_A_arr * g_nas, 0.0, 1.0)
    rest = 1.0 - wn
    wg   = rest * 0.5
    wb   = rest * 0.5 * g_bond
    return wn, wg, wb


# ---------------------------------------------------------------------------
# 指標ヘルパー
# ---------------------------------------------------------------------------

def compute_p10_5y(nav, trading_days=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(trading_days * 5)) ** 0.2 - 1
    return float(r.dropna().quantile(0.10))


def calc_all_metrics(nav, dates, trades_per_year):
    m   = calc_7metrics(nav, dates, trades_per_year=trades_per_year)
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, 10)
    cf  = (1 + m['CAGR_IS']) ** (IS_B / FULL_B) * (1 + m['CAGR_OOS']) ** (OOS_B / FULL_B) - 1
    return {
        **m,
        'CAGR_FULL':     cf,
        'Worst10Y_star': float(r10.min()) if len(r10) > 0 else np.nan,
        'P10_5Y':        compute_p10_5y(nav.values),
        'IS_OOS_gap':    m['CAGR_IS'] - m['CAGR_OOS'],
    }


# ---------------------------------------------------------------------------
# NAVビルダー
# ---------------------------------------------------------------------------

def build_s2lt2_nav(close, lev_A, wn_A, wg_A, wb_A, dates,
                    gold_2x, bond_3x, sofr, L_s2, N, k_lt):
    lt_sig  = build_lt_signal(close, 'LT2', N)
    lt_bias = signal_to_bias(lt_sig, k_lt)
    lev_mod = apply_lt_mode_b(lev_A, lt_bias, l_min=0.0, l_max=1.0)
    nav = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD',
        cfd_leverage=L_s2.values,
        cfd_spread=CFD_SPREAD_LOW,
    )
    return nav


def build_p05_nav(close, lev_A, wn_A, dates,
                  gold_2x, bond_3x, sofr,
                  hy_s, cpi_yoy, cpi_acc):
    g_hy  = build_hy_gate(hy_s, z_thresh=1.0, slope=0.5)
    g_cpi = build_cpi_gate(cpi_yoy, cpi_acc, cpi_thresh=5.0, reduce_factor=0.3)
    g_nas = np.clip(g_hy.values * g_cpi.values, 0.2, 1.0)
    wn, wg, wb = apply_gates(wn_A, nas_gate=g_nas, bond_gate=None)
    nav = build_nav_strategy(
        close, lev_A, wn, wg, wb, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002,
    )
    return nav


def blend_navs(nav_a, nav_b, w_a, w_b):
    a = nav_a / nav_a.iloc[0]
    b = nav_b / nav_b.iloc[0]
    h = w_a * a + w_b * b
    return h


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
# MD レポート生成
# ---------------------------------------------------------------------------

def generate_report(results, sanity_s2_ok, sanity_p05_ok,
                    sanity_s2_diff, sanity_p05_diff):
    r_s2  = next(r for r in results if r['label'] == 'S2+LT2-N1500')
    r_p05 = next(r for r in results if r['label'] == 'P05_HY×CPI')
    r_hyb = next(r for r in results if r['label'] == 'Hybrid_70_30')

    # 判定ロジック
    sharpe_ok  = r_hyb['Sharpe_OOS'] > r_s2['Sharpe_OOS']
    maxdd_ok   = r_hyb['MaxDD_FULL'] > r_s2['MaxDD_FULL']  # 浅い = 大きい(負値)
    gap_ok     = abs(r_hyb['IS_OOS_gap']) <= abs(r_s2['IS_OOS_gap']) + 0.02
    n_improve  = sum([sharpe_ok, maxdd_ok, gap_ok])
    if n_improve == 3:
        verdict = 'Shortlisted上位'
        verdict_note = '3条件すべて改善。§1.3 制約により Active 昇格不可。'
    elif n_improve >= 1:
        verdict = 'Shortlisted'
        verdict_note = f'{n_improve}/3 条件改善。§1.3 制約によりShortlisted止まり。'
    else:
        verdict = 'Rejected'
        verdict_note = '全条件でS2+LT2-N1500を下回る。ハイブリッド効果なし。'

    sanity_s2_tag  = '✅ 一致' if sanity_s2_ok  else f'⚠️ 乖離 {sanity_s2_diff:+.2f} pp'
    sanity_p05_tag = '✅ 一致' if sanity_p05_ok else f'⚠️ 乖離 {sanity_p05_diff:+.2f} pp'

    lines = [
        f'# E2: S2+LT2-N1500(70%) + P05_HY×CPI(30%) ハイブリッド検証',
        '',
        f'作成日: {DATE_STR}',
        'EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**',
        '',
        '> ‡ **§1.3 参考値**: P05_HY×CPI は異コストモデル戦略。',
        '> ハイブリッドはこの制約を継承し、CAGR_OOS の直接比較は参考値扱い。',
        '> Active 昇格不可（Shortlisted 上限）。',
        '',
        '---',
        '',
        '## §1 検証の背景・目的',
        '',
        'セッション内の WFA Opus インサイトレポート（旧5基準）にて、',
        '`S2+LT2(70%) + P05(30%)` のハイブリッドが提案された。',
        '旧指標（Stable_Sharpe/WinRate/WorstK5）は EVALUATION_STANDARD v1.1 で廃止済みのため、',
        '統一9指標フレームワークで改めて検証する。',
        '',
        '**判定基準（事前定義）**:',
        '',
        '| 基準 | 条件 | 参照 |',
        '|------|------|------|',
        f'| (i) Sharpe_OOS 改善 | Hybrid > S2+LT2-N1500 ({r_s2["Sharpe_OOS"]:.3f}) | §3.2 |',
        f'| (ii) MaxDD 改善 | Hybrid > S2+LT2-N1500 ({_fp(r_s2["MaxDD_FULL"])}) | §3.3 |',
        f'| (iii) IS-OOS gap 維持 | |gap_hybrid| ≤ |gap_S2LT2| + 2pp | §3.8 |',
        '',
        '---',
        '',
        '## §2 統合比較表 — 9指標 (EVALUATION_STANDARD v1.1)',
        '',
        '> 単位: CAGR_OOS/MaxDD/Worst10Y★/P10▷ = %, IS-OOS gap = pp, Tr = 回/年',
        '> CAGR_IS / CAGR_FULL は CSV (`e2_hybrid_70_30_results.csv`) に保存。MDは CAGR_OOS のみ（§3.12）。',
        '',
        MD_HEADER_STRAT[0],
        MD_HEADER_STRAT[1],
    ]

    for r in results:
        label = r['label']
        is_ref = label in ('P05_HY×CPI', 'Hybrid_70_30')
        if label == 'Hybrid_70_30':
            label_md = '**Hybrid 70/30 ‡**'
        elif label == 'P05_HY×CPI':
            label_md = 'P05_HY×CPI ‡'
        else:
            label_md = label
        lines.append(fmt_row_strat(
            label_md, r,
            sharpe_ref_mark='‡' if is_ref else None,
            maxdd_ref_mark='‡'  if is_ref else None,
        ))

    lines += ['', MD_WFA_NOTE]

    lines += [
        '',
        '---',
        '',
        '## §3 判定詳細',
        '',
        '| 基準 | S2+LT2-N1500 | Hybrid 70/30 | 差分 | 結果 |',
        '|------|---:|---:|---:|:---:|',
        f'| (i) Sharpe_OOS | {_ff(r_s2["Sharpe_OOS"])} | {_ff(r_hyb["Sharpe_OOS"])} '
        f'| {r_hyb["Sharpe_OOS"]-r_s2["Sharpe_OOS"]:+.3f} | {"✅" if sharpe_ok else "❌"} |',
        f'| (ii) MaxDD | {_fp(r_s2["MaxDD_FULL"])} | {_fp(r_hyb["MaxDD_FULL"])} '
        f'| {(r_hyb["MaxDD_FULL"]-r_s2["MaxDD_FULL"])*100:+.1f}pp | {"✅" if maxdd_ok else "❌"} |',
        f'| (iii) IS-OOS gap | {r_s2["IS_OOS_gap"]*100:+.2f}pp | {r_hyb["IS_OOS_gap"]*100:+.2f}pp '
        f'| {(r_hyb["IS_OOS_gap"]-r_s2["IS_OOS_gap"])*100:+.2f}pp | {"✅" if gap_ok else "❌"} |',
        '',
        f'**総合判定: {verdict}**',
        f'> {verdict_note}',
        '',
        '---',
        '',
        '## §4 サニティチェック',
        '',
        f'- S2+LT2-N1500 CAGR_OOS 参照値: +{REF_S2LT2_CAGR_OOS*100:.2f}% → 実測: {_fp(r_s2["CAGR_OOS"])} ({sanity_s2_tag})',
        f'- P05_HY×CPI CAGR_OOS 参照値: +{REF_P05_CAGR_OOS*100:.2f}% → 実測: {_fp(r_p05["CAGR_OOS"])} ({sanity_p05_tag})',
        '',
        '> P05 参照値は p4_overfitting_check.py と異なるデータロードパス',
        '> (prepare_gold_local vs prepare_gold_data) を使用するため若干の乖離を許容。',
        '',
        '---',
        '',
        '## §5 考察',
        '',
        '### ハイブリッド効果の評価',
        '',
        f'- **CAGR_OOS**: Hybrid {_fp(r_hyb["CAGR_OOS"])} vs S2+LT2 {_fp(r_s2["CAGR_OOS"])} '
        f'(差 {(r_hyb["CAGR_OOS"]-r_s2["CAGR_OOS"])*100:+.2f}pp)',
        f'- **Sharpe_OOS**: {_ff(r_hyb["Sharpe_OOS"])} vs {_ff(r_s2["Sharpe_OOS"])} '
        f'(差 {r_hyb["Sharpe_OOS"]-r_s2["Sharpe_OOS"]:+.3f})',
        f'- **MaxDD**: {_fp(r_hyb["MaxDD_FULL"])} vs {_fp(r_s2["MaxDD_FULL"])} '
        f'(差 {(r_hyb["MaxDD_FULL"]-r_s2["MaxDD_FULL"])*100:+.1f}pp)',
        '',
        '### §1.3 制約の実質的影響',
        '',
        '- P05 は HY スプレッド (FRED BAMLH0A0HYM2) / CPI 前年比ゲートを使用',
        '- 実運用ではこれらシグナルの取得コストが発生',
        '- ハイブリッドは§1.3 参考値扱いのため CURRENT_BEST_STRATEGY.md の',
        '  更新根拠には使用不可。Shortlisted での記録のみ可能。',
        '',
        '---',
        '',
        '## §6 再現コマンド',
        '',
        '```',
        'python -X utf8 src/e2_hybrid_70_30.py',
        '```',
        '',
        '---',
        '',
        f'*生成スクリプト: `src/e2_hybrid_70_30.py`  準拠: `EVALUATION_STANDARD.md v1.1`*',
        f'*参照: `CURRENT_BEST_STRATEGY.md`, `B6_S2_LT2_N_SWEEP_2026-05-22.md`, `STRATEGY_COMPARISON_INTEGRATED_2026-05-22.md`*',
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('E2: S2+LT2-N1500(70%) + P05_HY×CPI(30%) ハイブリッド検証')
    print(f'実行日: {DATE_STR}')
    print('=' * 70)

    assert os.path.exists(SIGNALS_PATH), f"Missing: {SIGNALS_PATH}"

    # ------------------------------------------------------------------
    # S1: データロード
    # ------------------------------------------------------------------
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({n:,} days, {n_years:.2f} yr)')

    # ------------------------------------------------------------------
    # S2: 共有資産
    # ------------------------------------------------------------------
    print('\nBuilding shared assets...')
    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print('  Done.')

    # ------------------------------------------------------------------
    # S3: DH Dyn シグナル
    # ------------------------------------------------------------------
    print('Building DH Dyn signal...')
    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    trades_yr = n_tr / n_years
    print(f'  {n_tr} trades, {trades_yr:.1f}/yr')

    # ------------------------------------------------------------------
    # S4: S2 CFD レバレッジ系列（共有）
    # ------------------------------------------------------------------
    print('Building S2 CFD leverage...')
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)

    # ------------------------------------------------------------------
    # S5: タイミングシグナルロード (P05 用)
    # ------------------------------------------------------------------
    print('Loading timing signals...')
    dates_idx = pd.DatetimeIndex(pd.to_datetime(dates.values))
    sig_raw = pd.read_csv(SIGNALS_PATH, index_col=0, parse_dates=True)
    sig     = sig_raw.reindex(dates_idx)
    hy_s    = pd.Series(sig['hy_spread'].values,          index=close.index)
    cpi_yoy = pd.Series(sig['cpi_yoy'].fillna(0).values,  index=close.index)
    cpi_acc = pd.Series(sig['cpi_accel'].fillna(0).values, index=close.index)
    print('  Done.')

    # ------------------------------------------------------------------
    # S6: NAV 計算
    # ------------------------------------------------------------------
    print(f'\nBuilding S2+LT2-N{N_LT2} NAV...')
    nav_s2lt2 = build_s2lt2_nav(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr, L_s2, N_LT2, K_LT2,
    )
    print(f'  Done. Final NAV = {nav_s2lt2.iloc[-1]:.4f}')

    print('Building P05_HY×CPI NAV...')
    nav_p05 = build_p05_nav(
        close, lev_A, wn_A, dates,
        gold_2x, bond_3x, sofr,
        hy_s, cpi_yoy, cpi_acc,
    )
    print(f'  Done. Final NAV = {nav_p05.iloc[-1]:.4f}')

    print(f'Blending {W_S2LT2:.0%}/{W_P05:.0%}...')
    nav_hybrid = blend_navs(nav_s2lt2, nav_p05, W_S2LT2, W_P05)
    print(f'  Done. Final NAV = {nav_hybrid.iloc[-1]:.4f}')

    # ------------------------------------------------------------------
    # S7: 9指標計算
    # ------------------------------------------------------------------
    print('\nCalculating 9 metrics...')
    m_s2lt2  = calc_all_metrics(nav_s2lt2,  dates, trades_yr)
    m_p05    = calc_all_metrics(nav_p05,    dates, trades_yr)
    m_hybrid = calc_all_metrics(nav_hybrid, dates, trades_yr)

    results = [
        {'label': 'S2+LT2-N1500',  **m_s2lt2},
        {'label': 'P05_HY×CPI',    **m_p05},
        {'label': 'Hybrid_70_30',   **m_hybrid},
    ]

    # ------------------------------------------------------------------
    # S8: サニティチェック
    # ------------------------------------------------------------------
    s2_diff   = (m_s2lt2['CAGR_OOS']  - REF_S2LT2_CAGR_OOS) * 100
    p05_diff  = (m_p05['CAGR_OOS']    - REF_P05_CAGR_OOS)   * 100
    s2_ok     = abs(s2_diff)  <= 0.20
    p05_ok    = abs(p05_diff) <= 1.50  # P05は別ローダーで若干乖離を許容

    print(f'\n[SANITY] S2+LT2-N1500 CAGR_OOS = {m_s2lt2["CAGR_OOS"]*100:+.2f}% '
          f'(ref {REF_S2LT2_CAGR_OOS*100:+.2f}%, diff {s2_diff:+.2f}pp) → {"OK" if s2_ok else "WARN"}')
    print(f'[SANITY] P05 CAGR_OOS          = {m_p05["CAGR_OOS"]*100:+.2f}% '
          f'(ref {REF_P05_CAGR_OOS*100:+.2f}%, diff {p05_diff:+.2f}pp) → {"OK" if p05_ok else "WARN"}')

    # ------------------------------------------------------------------
    # S9: コンソール出力
    # ------------------------------------------------------------------
    print('\n--- 9指標比較 ---')
    print(f'{"戦略":<22} {"CAGR_IS":>8} {"CAGR_OOS":>9} {"CAGR_FULL":>10} '
          f'{"Sharpe":>7} {"MaxDD":>7} {"W10★":>7} {"P10▷":>7} {"gap":>7} {"Tr":>4}')
    for r in results:
        print(f'{r["label"]:<22} '
              f'{r["CAGR_IS"]*100:>+7.2f}% '
              f'{r["CAGR_OOS"]*100:>+8.2f}% '
              f'{r["CAGR_FULL"]*100:>+9.2f}% '
              f'{r["Sharpe_OOS"]:>7.3f} '
              f'{r["MaxDD_FULL"]*100:>+6.1f}% '
              f'{r["Worst10Y_star"]*100:>+6.1f}% '
              f'{r["P10_5Y"]*100:>+6.1f}% '
              f'{r["IS_OOS_gap"]*100:>+6.2f} '
              f'{r["Trades_yr"]:>4.0f}')

    # ------------------------------------------------------------------
    # S10: CSV 出力
    # ------------------------------------------------------------------
    csv_path = os.path.join(BASE, 'e2_hybrid_70_30_results.csv')
    cols = ['label', 'CAGR_IS', 'CAGR_OOS', 'CAGR_FULL', 'Sharpe_OOS',
            'MaxDD_FULL', 'Worst10Y_star', 'P10_5Y', 'IS_OOS_gap', 'Trades_yr']
    pd.DataFrame(results)[cols].to_csv(csv_path, index=False)
    print(f'\nCSV → {csv_path}')

    # ------------------------------------------------------------------
    # S11: MD レポート出力
    # ------------------------------------------------------------------
    md = generate_report(results, s2_ok, p05_ok, s2_diff, p05_diff)
    md_path = os.path.join(BASE, 'E2_HYBRID_70_30_2026-05-22.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'MD  → {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
