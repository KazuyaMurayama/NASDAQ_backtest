"""
H1: コストストレステスト + 過学習感度分析
============================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-26)

目的:
  3戦略 (E4, F10-ε015, vz065+lmax5) を 2スプレッド条件で評価して、
  CFDコストが想定 (LOW=0.20%/yr) から 1.5× (=0.30%/yr) に上振れした場合の
  リターン耐性を比較する。さらに G7 WFA per-window CSV を用いて F10-ε015 の
  「上位N窓を除外した場合」の CI95_lo 感度を分析し、F10 の WFA PASS が
  少数の好調窓に依存していないかを確認する (過学習チェック)。

3戦略:
  (1) E4              — k_lo=0.1, k_hi=0.8, vz_thr=0.7, k_mid=0.5, LT2-N750,
                         L_s2 (l_max=7.0)  ← 現行 Active
  (2) F10-ε015        — E4 base + F8-R5 CALM_BOOST tilt + ε=0.015 deadband
                         L_s2 (l_max=7.0)
  (3) vz065+lmax5     — E4 base, vz_thr=0.65, L_s2 (l_max=5.0)

2スプレッド条件:
  Baseline    : cfd_spread = CFD_SPREAD_LOW = 0.0020
  Stress 1.5x : cfd_spread =                 0.0030

過学習チェック (G7 per-window CSV):
  F10-ε015 の有効窓 (short_flag=False) を CAGR 降順で並べ、上位N=0/1/3/5/10窓
  を除外して CI95_lo / t_p を再計算。CI95_lo > 0 が全 N で維持されれば「強固」と判定。

出力:
  - h1_stress_test_results.csv (6行 = 3戦略 × 2スプレッド × 9指標)
  - H1_STRESS_TEST_2026-05-26.md (レポート)

CURRENT_BEST_STRATEGY.md / tasks.md は更新しない。
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr, build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_a2_signal, simulate_rebalance_A,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy, calc_7metrics,
    CFD_SPREAD_LOW, IS_START, IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import build_lt_signal, apply_lt_mode_b
from _sweep_format import MD_HEADER_STRAT, fmt_row_strat, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
TODAY = '2026-05-26'

# E4 base
K_LO, K_HI, K_MID = 0.1, 0.8, 0.5
VZ_THR_E4         = 0.70
VZ_THR_065        = 0.65
N_LT2             = 750

# F10 ε-deadband (F8-R5 CALM_BOOST)
TILT_R5            = 10.0
VZ_REG             = 0.70
TILT_CAP_CALM      = 0.15
TILT_CAP_BULL_VZ   = 0.10
TILT_CAP_BEAR_VZ   = 0.05
EPS                = 0.015

# S2 leverage 基本パラメータ
S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5,
               n=20, l_min=1.0, step=0.5)

# スプレッドシナリオ
SPREAD_BASE   = CFD_SPREAD_LOW   # 0.0020
SPREAD_STRESS = 0.0030           # 1.5x

# G7 F10 WFA per-window CSV path
G7_PER_WINDOW_CSV = os.path.join(BASE, 'g7_wfa_f10_per_window.csv')

# 参照値（CURRENT_BEST_STRATEGY.md / G7 結果）
E4_REF = dict(
    CAGR_OOS=0.3353, Sharpe_OOS=0.891, MaxDD=-0.6001, Trades_yr=27.1,
    Worst10Y_star=0.1867, IS_OOS_gap=-0.0181,
    WFA_CI95_lo=0.2651, WFA_WFE=1.131,
)
F10_REF = dict(
    CAGR_OOS=0.3684, Sharpe_OOS=0.935, MaxDD=-0.6309, Trades_yr=51.6,
    Worst10Y_star=0.1858, IS_OOS_gap=-0.0431,
    WFA_CI95_lo=0.2791, WFA_WFE=1.208,
)


# ---------------------------------------------------------------------------
# 指標計算ヘルパ
# ---------------------------------------------------------------------------

def compute_p10_5y(nav, td=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    return float(((s / s.shift(td * 5)) ** 0.2 - 1).dropna().quantile(0.10))


def calc_all_metrics(nav, dates, trades_yr):
    m = calc_7metrics(nav, dates, trades_per_year=trades_yr)
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, 10)
    return {**m,
            'Worst10Y_star': float(r10.min()) if len(r10) > 0 else float('nan'),
            'P10_5Y':        compute_p10_5y(nav.values),
            'IS_OOS_gap':    m['CAGR_IS'] - m['CAGR_OOS']}


# ---------------------------------------------------------------------------
# F10 ε-deadband (g7_wfa_f10.py / f10_epsilon_deadband.py と同一)
# ---------------------------------------------------------------------------

def compute_tilt_with_deadband(raw_a2, vz, bull_mask, eps):
    n = len(raw_a2)
    cap_eff = np.where(np.abs(vz) < VZ_REG, TILT_CAP_CALM,
              np.where(vz > VZ_REG, TILT_CAP_BULL_VZ, TILT_CAP_BEAR_VZ))
    tilt_raw = TILT_R5 * (raw_a2 - THRESHOLD) * (1.0 - raw_a2)
    tilt_target = np.minimum(np.maximum(tilt_raw, 0.0), cap_eff)
    tilt_target = np.where(bull_mask, tilt_target, 0.0)

    confirmed = np.zeros(n, dtype=float)
    cur = 0.0
    for i in range(n):
        if i == 0 or abs(tilt_target[i] - cur) >= eps:
            cur = tilt_target[i]
        confirmed[i] = cur
    return confirmed


# ---------------------------------------------------------------------------
# 過学習チェック: 上位N窓除外時の CI95_lo 再計算
# ---------------------------------------------------------------------------

def trimmed_ci95(df, n_trim):
    """CAGR 降順で先頭 n_trim 窓を除外し、CI95_lo / t_p を再計算する。"""
    sorted_df = df.sort_values('CAGR', ascending=False)
    trimmed = sorted_df.iloc[n_trim:]
    cagrs = trimmed['CAGR'].dropna().values
    n = len(cagrs)
    if n < 2:
        return dict(n_trim=n_trim, n=n, mean_CAGR=float('nan'),
                    CI95_lo=float('nan'), t_p=float('nan'))
    mean_c = float(np.mean(cagrs))
    std_c  = float(np.std(cagrs, ddof=1))
    se     = std_c / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci95_lo = mean_c - t_crit * se
    t_stat  = mean_c / se if se > 0 else float('nan')
    t_pval  = float(stats.t.sf(t_stat, df=n - 1)) if not np.isnan(t_stat) else float('nan')
    return dict(n_trim=n_trim, n=n, mean_CAGR=mean_c,
                CI95_lo=ci95_lo, t_p=t_pval,
                trimmed_top_cagrs=sorted_df.iloc[:n_trim][['window_id', 'CAGR']].to_dict(orient='records'))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 72)
    print('H1: コストストレステスト + 過学習感度分析')
    print('=' * 72)
    print(f'戦略: E4, F10-eps015, vz065+lmax5')
    print(f'スプレッド: Baseline={SPREAD_BASE*100:.2f}%/yr, '
          f'Stress 1.5x={SPREAD_STRESS*100:.2f}%/yr')
    print()

    # -----------------------------------------------------------------------
    # 共有資産ロード
    # -----------------------------------------------------------------------
    print('[1/4] Loading shared assets...')
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'  Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} '
          f'({n:,} days, {n_years:.2f} yr)')

    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                          bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_trades_yr_base = n_tr / n_years   # E4 / vz065+lmax5 ベースの Trades/yr

    lt_sig_raw = build_lt_signal(close, 'LT2', N=N_LT2)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    raw_a2_vals = raw_a2.values
    bull_mask   = raw_a2_vals > THRESHOLD

    # -----------------------------------------------------------------------
    # 戦略コンポーネント構築
    # -----------------------------------------------------------------------
    print('[2/4] Building strategy components...')

    # ---- E4 ----
    k_dyn_e4 = np.where(vz_arr > VZ_THR_E4, K_HI,
                np.where(vz_arr < -VZ_THR_E4, K_LO, K_MID))
    lt_bias_e4 = pd.Series(np.clip(-k_dyn_e4 * lt_sig_arr * 0.5, -0.5, 0.5),
                            index=lt_sig_raw.index)
    lev_mod_e4 = apply_lt_mode_b(lev_raw, lt_bias_e4, l_min=0.0, l_max=1.0)
    L_s2_lmax7 = compute_L_s2_vz_gated(ret, vz, l_max=7.0, **S2_BASE)
    print(f'  E4: lev_mod built, L_s2 (l_max=7.0) built')

    # ---- F10-ε015 (E4 base + tilted weights) ----
    tilt_conf = compute_tilt_with_deadband(raw_a2_vals, vz_arr, bull_mask, EPS)
    wn_f10 = wn_A + tilt_conf
    wb_f10 = np.clip(wb_A - tilt_conf, 0.0, wb_A)
    # F10 は E4 と同じ lev_mod (E4 base) と L_s2 (l_max=7.0) を使用
    print(f'  F10-eps015: tilt confirmed (eps={EPS}), wn/wb adjusted')

    # ---- vz065+lmax5 (E4 と同じ tilt なし、vz_thr=0.65, l_max=5.0) ----
    k_dyn_065 = np.where(vz_arr > VZ_THR_065, K_HI,
                 np.where(vz_arr < -VZ_THR_065, K_LO, K_MID))
    lt_bias_065 = pd.Series(np.clip(-k_dyn_065 * lt_sig_arr * 0.5, -0.5, 0.5),
                             index=lt_sig_raw.index)
    lev_mod_065 = apply_lt_mode_b(lev_raw, lt_bias_065, l_min=0.0, l_max=1.0)
    L_s2_lmax5 = compute_L_s2_vz_gated(ret, vz, l_max=5.0, **S2_BASE)
    print(f'  vz065+lmax5: lev_mod_065 built, L_s2 (l_max=5.0) built')

    # -----------------------------------------------------------------------
    # 戦略テンプレ
    # -----------------------------------------------------------------------
    # NAV 構築の引数を 1 dict にまとめ、spread のみ差し替える
    strategies = [
        dict(
            id='E4',
            label='E4 (現行 Active)',
            lev_mod=lev_mod_e4,
            wn=wn_A, wg=wg_A, wb=wb_A,
            cfd_lev=L_s2_lmax7.values,
            trades_yr=n_trades_yr_base,
        ),
        dict(
            id='F10-eps015',
            label='F10 (E4 + F8-R5 tilt + ε=0.015)',
            lev_mod=lev_mod_e4,
            wn=wn_f10, wg=wg_A, wb=wb_f10,
            cfd_lev=L_s2_lmax7.values,
            trades_yr=51.6,   # F10_EPSILON_DEADBAND_2026-05-26.md / G7 値
        ),
        dict(
            id='vz065_lmax5',
            label='vz_thr=0.65 + l_max=5.0',
            lev_mod=lev_mod_065,
            wn=wn_A, wg=wg_A, wb=wb_A,
            cfd_lev=L_s2_lmax5.values,
            trades_yr=n_trades_yr_base,
        ),
    ]

    # -----------------------------------------------------------------------
    # スプレッドループ
    # -----------------------------------------------------------------------
    print('[3/4] Running stress test (3 strategies x 2 spreads)...')
    rows = []
    for spread_label, spread in [('Baseline', SPREAD_BASE),
                                  ('Stress_1.5x', SPREAD_STRESS)]:
        for s in strategies:
            nav = build_nav_strategy(
                close, s['lev_mod'], s['wn'], s['wg'], s['wb'], dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=s['cfd_lev'], cfd_spread=spread,
            )
            m = calc_all_metrics(nav, dates, s['trades_yr'])
            row = dict(
                spread_label=spread_label,
                spread_pct=spread,
                strategy=s['id'],
                strategy_label=s['label'],
                CAGR_IS=m['CAGR_IS'],
                CAGR_OOS=m['CAGR_OOS'],
                Sharpe_OOS=m['Sharpe_OOS'],
                MaxDD_FULL=m['MaxDD_FULL'],
                Worst10Y_star=m['Worst10Y_star'],
                P10_5Y=m['P10_5Y'],
                IS_OOS_gap=m['IS_OOS_gap'],
                Trades_yr=s['trades_yr'],
                WFA_CI95_lo=float('nan'),    # スプレッド変動時 WFA は再計算しない (本ラウンド対象外)
                WFA_WFE=float('nan'),
            )
            rows.append(row)
            print(f'  [{spread_label:<11s}] {s["id"]:<14s}: '
                  f'CAGR_OOS={m["CAGR_OOS"]*100:+6.2f}%  '
                  f'Sharpe={m["Sharpe_OOS"]:+.3f}  '
                  f'MaxDD={m["MaxDD_FULL"]*100:+6.2f}%  '
                  f'W10Y★={m["Worst10Y_star"]*100:+6.2f}%')

    # -----------------------------------------------------------------------
    # CSV 保存
    # -----------------------------------------------------------------------
    csv_path = os.path.join(BASE, 'h1_stress_test_results.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nCSV saved: {csv_path}')

    # -----------------------------------------------------------------------
    # 過学習チェック: G7 F10 per-window CSV の窓トリミング
    # -----------------------------------------------------------------------
    print('\n[4/4] G7 F10 per-window trimming sensitivity...')
    per_df = pd.read_csv(G7_PER_WINDOW_CSV)
    f10_windows = per_df[per_df['strategy'] == 'F10-eps015'].copy()
    f10_windows = f10_windows[~f10_windows['short_flag']].copy()
    n_total = len(f10_windows)
    print(f'  F10-eps015 有効窓: {n_total}')

    trim_results = []
    for ntrim in [0, 1, 3, 5, 10]:
        r = trimmed_ci95(f10_windows, ntrim)
        trim_results.append(r)
        top_str = ''
        if ntrim > 0 and 'trimmed_top_cagrs' in r:
            top_str = ' | trimmed top: ' + ', '.join(
                f'{w["window_id"]}({w["CAGR"]*100:+.1f}%)'
                for w in r['trimmed_top_cagrs']
            )
        print(f'  N={ntrim:>2d}: n={r["n"]:>2d}  '
              f'mean_CAGR={r["mean_CAGR"]*100:+6.2f}%  '
              f'CI95_lo={r["CI95_lo"]*100:+6.2f}%  '
              f't_p={r["t_p"]:.6f}{top_str}')

    # CI95_lo > 0 が全Nで維持されるか判定
    all_positive = all((not np.isnan(r['CI95_lo'])) and r['CI95_lo'] > 0
                       for r in trim_results)
    overfit_verdict = '強固 (全 N で CI95_lo > 0)' if all_positive else '脆弱 (一部 N で CI95_lo ≤ 0)'
    print(f'\n過学習判定: {overfit_verdict}')

    # -----------------------------------------------------------------------
    # MD レポート生成
    # -----------------------------------------------------------------------
    print('\nGenerating MD report...')
    md_path = os.path.join(BASE, f'H1_STRESS_TEST_{TODAY}.md')

    hdr1, hdr2 = MD_HEADER_STRAT

    # ストレステスト 9指標フル表 (6行)
    def make_rows(label_prefix):
        rs = []
        for r in rows:
            label = f'{label_prefix} / {r["strategy"]}'
            d = dict(
                CAGR_OOS=r['CAGR_OOS'],
                Sharpe_OOS=r['Sharpe_OOS'],
                MaxDD_FULL=r['MaxDD_FULL'],
                Worst10Y_star=r['Worst10Y_star'],
                P10_5Y=r['P10_5Y'],
                IS_OOS_gap=r['IS_OOS_gap'],
                Trades_yr=r['Trades_yr'],
                WFA_CI95_lo=r['WFA_CI95_lo'],
                WFA_WFE=r['WFA_WFE'],
            )
            rs.append((label, d))
        return rs

    # 戦略表: スプレッド別にセクション分けして並べる
    base_rows = [r for r in rows if r['spread_label'] == 'Baseline']
    stress_rows = [r for r in rows if r['spread_label'] == 'Stress_1.5x']

    def fmt_full_table(row_list, prefix):
        out = []
        for r in row_list:
            d = dict(
                CAGR_OOS=r['CAGR_OOS'],
                Sharpe_OOS=r['Sharpe_OOS'],
                MaxDD_FULL=r['MaxDD_FULL'],
                Worst10Y_star=r['Worst10Y_star'],
                P10_5Y=r['P10_5Y'],
                IS_OOS_gap=r['IS_OOS_gap'],
                Trades_yr=r['Trades_yr'],
                WFA_CI95_lo=r['WFA_CI95_lo'],
                WFA_WFE=r['WFA_WFE'],
            )
            out.append(fmt_row_strat(f'{prefix} {r["strategy"]}', d))
        return '\n'.join(out)

    table_base   = fmt_full_table(base_rows,   '[B]')
    table_stress = fmt_full_table(stress_rows, '[S]')

    # 差分表 (3戦略 × CAGR/Sharpe/MaxDD の変化量)
    def diff_block():
        lines = [
            '| 戦略 | ΔCAGR_OOS | ΔSharpe_OOS | ΔMaxDD | ΔWorst10Y★ |',
            '|:-----|----------:|------------:|-------:|------------:|',
        ]
        for sid in ['E4', 'F10-eps015', 'vz065_lmax5']:
            b = next(r for r in base_rows   if r['strategy'] == sid)
            s = next(r for r in stress_rows if r['strategy'] == sid)
            d_cagr = (s['CAGR_OOS']   - b['CAGR_OOS'])   * 100
            d_sh   =  s['Sharpe_OOS'] - b['Sharpe_OOS']
            d_dd   = (s['MaxDD_FULL'] - b['MaxDD_FULL']) * 100
            d_w10  = (s['Worst10Y_star'] - b['Worst10Y_star']) * 100
            lines.append(
                f'| {sid:<12s} | {d_cagr:+.2f} pp | {d_sh:+.4f} | '
                f'{d_dd:+.2f} pp | {d_w10:+.2f} pp |'
            )
        return '\n'.join(lines)

    diff_table = diff_block()

    # 過学習感度表
    trim_lines = [
        '| N (除外窓数) | n (有効窓) | mean_CAGR | CI95_lo | t_p | 判定 (CI95_lo > 0) |',
        '|------------:|-----------:|----------:|--------:|----:|:------------------:|',
    ]
    for r in trim_results:
        verdict = 'OK' if (not np.isnan(r['CI95_lo']) and r['CI95_lo'] > 0) else 'NG'
        trim_lines.append(
            f'| {r["n_trim"]:>4d} | {r["n"]:>4d} | '
            f'{r["mean_CAGR"]*100:+6.2f}% | '
            f'{r["CI95_lo"]*100:+6.2f}% | '
            f'{r["t_p"]:.6f} | {verdict} |'
        )
    trim_table = '\n'.join(trim_lines)

    # 除外された上位窓のリスト (N=10 の場合)
    top10_lines = []
    if len(trim_results) > 0 and 'trimmed_top_cagrs' in trim_results[-1]:
        for w in trim_results[-1]['trimmed_top_cagrs']:
            top10_lines.append(f'  - 窓 {int(w["window_id"])}: CAGR={w["CAGR"]*100:+.2f}%')
    top10_block = '\n'.join(top10_lines) if top10_lines else '  - (該当なし)'

    # 総合評価ロジック
    e4_base   = next(r for r in base_rows   if r['strategy'] == 'E4')
    e4_str    = next(r for r in stress_rows if r['strategy'] == 'E4')
    f10_base  = next(r for r in base_rows   if r['strategy'] == 'F10-eps015')
    f10_str   = next(r for r in stress_rows if r['strategy'] == 'F10-eps015')
    vz_base   = next(r for r in base_rows   if r['strategy'] == 'vz065_lmax5')
    vz_str    = next(r for r in stress_rows if r['strategy'] == 'vz065_lmax5')

    d_e4_cagr  = (e4_str['CAGR_OOS']   - e4_base['CAGR_OOS'])   * 100
    d_f10_cagr = (f10_str['CAGR_OOS']  - f10_base['CAGR_OOS'])  * 100
    d_vz_cagr  = (vz_str['CAGR_OOS']   - vz_base['CAGR_OOS'])   * 100
    d_e4_sh    =  e4_str['Sharpe_OOS'] - e4_base['Sharpe_OOS']
    d_f10_sh   =  f10_str['Sharpe_OOS'] - f10_base['Sharpe_OOS']
    d_vz_sh    =  vz_str['Sharpe_OOS'] - vz_base['Sharpe_OOS']

    # F10 が E4 を維持できるか判定
    f10_advantage_base   = (f10_base['CAGR_OOS']  - e4_base['CAGR_OOS'])  * 100
    f10_advantage_stress = (f10_str['CAGR_OOS']   - e4_str['CAGR_OOS'])   * 100
    f10_keeps_lead = f10_advantage_stress > 0
    f10_lead_loss_pp = f10_advantage_base - f10_advantage_stress

    vz_advantage_base   = (vz_base['CAGR_OOS']  - e4_base['CAGR_OOS']) * 100
    vz_advantage_stress = (vz_str['CAGR_OOS']   - e4_str['CAGR_OOS'])  * 100
    vz_keeps_lead = vz_advantage_stress > 0

    # 採用推奨可否 (E4 比較で F10 がストレス下でも Sharpe & CAGR で上回るか)
    f10_keeps_sharpe = f10_str['Sharpe_OOS'] > e4_str['Sharpe_OOS']
    if f10_keeps_sharpe and f10_keeps_lead and all_positive:
        adopt_verdict = (
            f'**F10-ε015 は採用検討可**: Stress 1.5x でも Sharpe ({f10_str["Sharpe_OOS"]:+.3f}) と '
            f'CAGR_OOS ({f10_str["CAGR_OOS"]*100:+.2f}%) が E4 を上回り、過学習感度も全 N で CI95_lo > 0。'
            f'ただし IS-OOS gap (-4.31pp) は E4 (-1.81pp) より広く、将来モニタリング必須。'
            f'CURRENT_BEST_STRATEGY.md の更新は本ラウンドでは行わず、追加検証 (例: WFA G8 で別シード) の後に再評価。'
        )
    elif f10_keeps_sharpe and f10_keeps_lead:
        adopt_verdict = (
            f'**F10-ε015 は条件付き採用候補**: Stress 1.5x でも E4 を上回るが、'
            f'過学習感度で一部 N で CI95_lo ≤ 0 → 上位窓依存リスクあり。E4 維持を推奨。'
        )
    else:
        adopt_verdict = (
            f'**F10-ε015 採用見送り**: Stress 1.5x で E4 比優位が縮小/消失 '
            f'(Sharpe 維持={f10_keeps_sharpe}, CAGR 維持={f10_keeps_lead})。E4 を Active 維持。'
        )

    md = f"""\
# H1: コストストレステスト + 過学習感度分析

作成日: {TODAY}
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D** + スプレッド 1.5× ストレス

## §1 目的

3戦略 (E4, F10-ε015, vz065+lmax5) を以下の 2スプレッド条件で評価し、
CFD スプレッドが想定 (`LOW=0.20%/yr`) から 1.5× (=0.30%/yr) に上振れした場合の
リターン耐性を比較する。さらに G7 WFA per-window CSV を用いて F10-ε015 の
「上位 N 窓を除外した場合」の CI95_lo 感度を分析し、F10 の WFA PASS が
少数の好調窓に依存していないかを確認する (**過学習チェック**)。

### 対象戦略
| ID | 説明 | base 性能 |
|:---|:-----|:----------|
| **E4** | k_lo=0.1, k_hi=0.8, vz_thr=0.7, k_mid=0.5, LT2-N750, L_s2 l_max=7.0 (現行 Active) | CAGR=+{E4_REF['CAGR_OOS']*100:.2f}%, Sharpe=+{E4_REF['Sharpe_OOS']:.3f} |
| **F10-ε015** | E4 base + F8-R5 CALM_BOOST tilt + ε={EPS} deadband (G7 WFA PASS) | CAGR=+{F10_REF['CAGR_OOS']*100:.2f}%, Sharpe=+{F10_REF['Sharpe_OOS']:.3f} |
| **vz065+lmax5** | E4 base, vz_thr=0.65, L_s2 l_max=5.0 (E5b ピーク, WFA 未実施) | CAGR=+33.49%, Sharpe=+0.949 |

### スプレッド条件
| ラベル | cfd_spread | 解釈 |
|:-------|-----------:|:-----|
| Baseline    | {SPREAD_BASE*100:.2f}% /yr | くりっく株365 等の最安クラス想定 (CFD_SPREAD_LOW) |
| Stress 1.5x | {SPREAD_STRESS*100:.2f}% /yr | スプレッドが想定の 1.5× に拡大した状況 |

## §2 ストレステスト結果 (9指標フル表)

### §2.1 Baseline (cfd_spread = {SPREAD_BASE*100:.2f}%/yr)

{hdr1}
{hdr2}
{table_base}

### §2.2 Stress 1.5x (cfd_spread = {SPREAD_STRESS*100:.2f}%/yr)

{hdr1}
{hdr2}
{table_stress}

{MD_WFA_NOTE}

CI95_lo / WFE が `—` なのは、本ラウンドでスプレッド変動時の WFA 再計算は対象外のため。
WFA はスプレッド `Baseline` での値が G3 (E4) / G7 (F10) で確認済み。

## §3 Baseline vs Stress 1.5x 差分

{diff_table}

### §3.1 戦略別 ΔCAGR_OOS / ΔSharpe の解釈

- **E4** (現行 Active): ΔCAGR={d_e4_cagr:+.2f} pp / ΔSharpe={d_e4_sh:+.4f}
  - Trades/yr=27 と少ないため、スプレッド 1.5× の影響は限定的。
- **F10-ε015** (G7 PASS): ΔCAGR={d_f10_cagr:+.2f} pp / ΔSharpe={d_f10_sh:+.4f}
  - Trades/yr=51.6 で E4 比 ~1.9× → コスト感度がやや高い。
- **vz065+lmax5** (WFA 未実施): ΔCAGR={d_vz_cagr:+.2f} pp / ΔSharpe={d_vz_sh:+.4f}
  - Trades/yr=27 (E4 と同) → E4 と同程度のコスト耐性。

### §3.2 E4 比優位の維持

- **F10 vs E4**: Baseline {f10_advantage_base:+.2f} pp → Stress {f10_advantage_stress:+.2f} pp (優位縮小 {f10_lead_loss_pp:+.2f} pp)
  - 優位維持: **{"はい" if f10_keeps_lead else "いいえ"}**
- **vz065+lmax5 vs E4**: Baseline {vz_advantage_base:+.2f} pp → Stress {vz_advantage_stress:+.2f} pp
  - 優位維持: **{"はい" if vz_keeps_lead else "いいえ"}**

## §4 過学習チェック: G7 F10 窓トリミング感度

### §4.1 セットアップ

G7 WFA per-window CSV (`g7_wfa_f10_per_window.csv`) から F10-ε015 の
有効窓 ({n_total} 窓, short_flag=False) を CAGR 降順で並べ、
上位 N=0/1/3/5/10 窓を除外して CI95_lo / t_p を再計算する。

CI95_lo > 0 が全 N で維持されれば「**強固**」と判定 (=PASS が少数好調窓に依存していない)。

### §4.2 結果

{trim_table}

### §4.3 除外された上位 10 窓の内訳

{top10_block}

### §4.4 過学習判定

**判定**: **{overfit_verdict}**

- 全 N で CI95_lo > 0 → F10 の WFA PASS は少数の好調窓に依存しない (頑健)
- 一部 N で CI95_lo ≤ 0 → 上位窓除外で有意性消失 (上位窓依存性あり)

## §5 総合評価

### §5.1 ストレス耐性ランキング (Sharpe_OOS, Stress 1.5x ベース)

| 順位 | 戦略 | Sharpe (Stress) | CAGR_OOS (Stress) | MaxDD (Stress) |
|:----:|:-----|----------------:|------------------:|---------------:|
""".strip() + '\n'

    stress_ranked = sorted(stress_rows, key=lambda r: r['Sharpe_OOS'], reverse=True)
    md_extra_rows = []
    for i, r in enumerate(stress_ranked, 1):
        md_extra_rows.append(
            f'| {i} | {r["strategy"]} | '
            f'{r["Sharpe_OOS"]:+.3f} | '
            f'{r["CAGR_OOS"]*100:+.2f}% | '
            f'{r["MaxDD_FULL"]*100:+.2f}% |'
        )

    md += '\n'.join(md_extra_rows) + '\n\n'

    md += f"""\
### §5.2 採用推奨可否

{adopt_verdict}

### §5.3 注意点

- 本テストは **Scenario D 資産前提**でスプレッドのみ変動させた。SOFR/金利環境の同時変動はカバーしていない。
- **WFA はスプレッド変動下では未再計算**。スプレッド 1.5× での CI95_lo / WFE は別ラウンドで確認推奨。
- F10-ε015 の IS-OOS gap (-4.31pp) は E4 (-1.81pp) より広い。Stress 1.5x でこの gap がさらに拡大しないかを継続モニタリング。

## §6 再現コマンド

```bash
cd "C:\\Users\\user\\Desktop\\投資・不動産\\nasdaq_backtest"
python -X utf8 src/h1_stress_test.py
```

出力:
- `h1_stress_test_results.csv` — 6 行 (3戦略 × 2スプレッド × 9指標)
- `H1_STRESS_TEST_{TODAY}.md` — 本レポート

参照:
- `CURRENT_BEST_STRATEGY.md` — E4 現行 Active
- `G7_WFA_F10_{TODAY}.md` — F10 WFA 結果
- `E5B_FINE_SWEEP_{TODAY}.md` — vz065+lmax5 由来
- `EVALUATION_STANDARD.md` §3.12 — 9指標標準
- `g7_wfa_f10_per_window.csv` — 過学習感度の入力

---

*生成スクリプト: `src/h1_stress_test.py` (本ラウンドで作成)*
"""

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'MD saved: {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
