"""
G10: Walk-Forward Analysis — vz_thr=0.65 × l_max行フロンティア確認
=================================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-27)

目的:
  D5実験でvz_thr=0.65が全l_max値にわたって最高Sharpeを達成した。
  本G10では vz=0.65 行の l_max=[5.0,5.5,6.0,7.0] を WFA で評価し、
  プラトー（l_max=5.0-5.5）が統計的に有効か、最終採用変更根拠とする。

対象戦略 (5本):
  REF-E4:         vz_thr=0.70, l_max=7.0 (G3 サニティ参照, CI95_lo=+26.51%, WFE=+1.131)
  vz065-lmax5:    vz_thr=0.65, l_max=5.0 (G9 PASS, CI95_lo=+24.82%)
  vz065-lmax5p5:  vz_thr=0.65, l_max=5.5 (D5: Sharpe=+0.945)
  vz065-lmax6:    vz_thr=0.65, l_max=6.0 (D5: Sharpe=+0.939)
  vz065-lmax7:    vz_thr=0.65, l_max=7.0 (D5: Sharpe=+0.945, CAGR最大)

窓設計: G3/G4/G5/G7/G8/G9 と完全同一（非重複1年窓、~50窓）

判定基準 (EVALUATION_STANDARD v1.1 §3.9/§3.10):
  α: WFA_CI95_lo > 0 AND t_p < 0.05
  β: WFA_WFE ∈ [0.5, 2.0]
  総合: α+β→PASS / αのみ→WARN / α FAIL→FAIL

出力:
  - g10_wfa_vz065_lmax_row_per_window.csv
  - g10_wfa_vz065_lmax_row_summary.csv
  - G10_WFA_VZ065_LMAX_ROW_2026-05-27.md
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

import numpy as np
import pandas as pd
from scipy import stats

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
    CFD_SPREAD_LOW,
    IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local
from long_cycle_signal import build_lt_signal, apply_lt_mode_b

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
K_LO       = 0.1
K_HI       = 0.8
VZ_THR_E4  = 0.70
VZ_THR_065 = 0.65
K_MID      = 0.5
N_LT2      = 750

S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)
S2_LMAX5   = {**S2_BASE, 'l_max': 5.0}
S2_LMAX5P5 = {**S2_BASE, 'l_max': 5.5}
S2_LMAX6   = {**S2_BASE, 'l_max': 6.0}
S2_LMAX7   = {**S2_BASE, 'l_max': 7.0}

WINDOW_DAYS   = 252
STEP_DAYS     = 252
EVAL_START    = '1977-01-03'
EVAL_END      = '2026-03-26'
IS_END_REF    = IS_END
OOS_START_REF = OOS_START
TODAY         = '2026-05-27'

STRATEGY_IDS = ['REF-E4', 'vz065-lmax5', 'vz065-lmax5p5', 'vz065-lmax6', 'vz065-lmax7']
STRATEGY_LABELS = {
    'REF-E4':        'REF E4 (vz=0.70, lmax=7.0)',
    'vz065-lmax5':   'vz065+lmax5.0 (G9 PASS)',
    'vz065-lmax5p5': 'vz065+lmax5.5',
    'vz065-lmax6':   'vz065+lmax6.0',
    'vz065-lmax7':   'vz065+lmax7.0',
}

# G3 サニティ参照値
REF_G3_CI95_LO = 0.265093
REF_G3_WFE     = 1.130664

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Trades 計数
# ---------------------------------------------------------------------------

def count_trades_in_window(wn_tilted, wb_tilted, lev_arr, s, e):
    n_tr = 0
    for i in range(s + 1, e):
        if (wn_tilted[i] != wn_tilted[i-1] or
            wb_tilted[i] != wb_tilted[i-1] or
            lev_arr[i]   != lev_arr[i-1]):
            n_tr += 1
    return n_tr


# ---------------------------------------------------------------------------
# 共有資産ロード
# ---------------------------------------------------------------------------

def load_shared_assets(data_path: str) -> dict:
    df    = load_data(data_path)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'  Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} '
          f'({n:,} days, {n_years:.2f} yr)')

    sofr = load_sofr(dates)

    gold_1x_local = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x_local, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                          bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print('  Scenario D assets done.')

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    print(f'  DH Dyn A: {n_tr} trades, {n_tr/n_years:.1f}/yr')

    lt_sig_raw = build_lt_signal(close, 'LT2', N=N_LT2)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    print('  LT2 signal done.')

    # E4 レジーム (vz_thr=0.70)
    k_dyn_e4 = np.where(vz_arr >  VZ_THR_E4, K_HI,
                np.where(vz_arr < -VZ_THR_E4, K_LO, K_MID))
    lt_bias_e4 = pd.Series(np.clip(-k_dyn_e4 * lt_sig_arr * 0.5, -0.5, 0.5),
                           index=lt_sig_raw.index)
    lev_mod_e4 = apply_lt_mode_b(lev_raw, lt_bias_e4, l_min=0.0, l_max=1.0)

    # vz065 レジーム (vz_thr=0.65)
    k_dyn_065 = np.where(vz_arr >  VZ_THR_065, K_HI,
                np.where(vz_arr < -VZ_THR_065, K_LO, K_MID))
    lt_bias_065 = pd.Series(np.clip(-k_dyn_065 * lt_sig_arr * 0.5, -0.5, 0.5),
                            index=lt_sig_raw.index)
    lev_mod_065 = apply_lt_mode_b(lev_raw, lt_bias_065, l_min=0.0, l_max=1.0)

    # CFD レバ（l_max 4種）
    L_s2_lmax7   = compute_L_s2_vz_gated(ret, vz, **S2_LMAX7)
    L_s2_lmax5   = compute_L_s2_vz_gated(ret, vz, **S2_LMAX5)
    L_s2_lmax5p5 = compute_L_s2_vz_gated(ret, vz, **S2_LMAX5P5)
    L_s2_lmax6   = compute_L_s2_vz_gated(ret, vz, **S2_LMAX6)
    print('  S2 leverage done (l_max=5.0, 5.5, 6.0, 7.0).')

    return dict(
        close=close, ret=ret, dates=dates, n=n, n_years=n_years, sofr=sofr,
        gold_2x=gold_2x, bond_3x=bond_3x,
        vz=vz, lev_raw=lev_raw, lev_mod_e4=lev_mod_e4, lev_mod_065=lev_mod_065,
        wn_A=wn_A, wg_A=wg_A, wb_A=wb_A,
        L_s2_lmax7=L_s2_lmax7, L_s2_lmax5=L_s2_lmax5,
        L_s2_lmax5p5=L_s2_lmax5p5, L_s2_lmax6=L_s2_lmax6,
    )


# ---------------------------------------------------------------------------
# 戦略 NAV 構築
# ---------------------------------------------------------------------------

def build_all_navs(assets: dict) -> dict:
    a = assets
    close, dates, sofr = a['close'], a['dates'], a['sofr']
    gold_2x, bond_3x   = a['gold_2x'], a['bond_3x']
    lev_raw_arr        = np.asarray(a['lev_raw'])

    out = {}

    def _build(lev_mod, L_s2):
        return build_nav_strategy(
            close, lev_mod, a['wn_A'], a['wg_A'], a['wb_A'], dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD',
            cfd_leverage=L_s2.values,
            cfd_spread=CFD_SPREAD_LOW,
        )

    configs = [
        ('REF-E4',        a['lev_mod_e4'],  a['L_s2_lmax7']),
        ('vz065-lmax5',   a['lev_mod_065'], a['L_s2_lmax5']),
        ('vz065-lmax5p5', a['lev_mod_065'], a['L_s2_lmax5p5']),
        ('vz065-lmax6',   a['lev_mod_065'], a['L_s2_lmax6']),
        ('vz065-lmax7',   a['lev_mod_065'], a['L_s2_lmax7']),
    ]

    for sid, lev_mod, L_s2 in configs:
        nav = _build(lev_mod, L_s2)
        out[sid] = dict(nav=nav, wn=np.asarray(a['wn_A']),
                        wb=np.asarray(a['wb_A']), lev_raw_arr=lev_raw_arr)

    return out


# ---------------------------------------------------------------------------
# Window 生成 (G9 と完全同一)
# ---------------------------------------------------------------------------

def generate_windows(dates, eval_start=EVAL_START, eval_end=EVAL_END,
                     window_days=WINDOW_DAYS, step_days=STEP_DAYS):
    eval_start_ts = pd.Timestamp(eval_start)
    eval_end_ts   = pd.Timestamp(eval_end)
    first_year    = eval_start_ts.year
    last_year     = eval_end_ts.year

    windows = []
    for year in range(first_year, last_year + 1):
        yr_start_ts = max(pd.Timestamp(f'{year}-01-01'), eval_start_ts)
        yr_end_ts   = min(pd.Timestamp(f'{year}-12-31'), eval_end_ts)

        mask_s = dates >= yr_start_ts
        if not mask_s.any():
            break
        s_idx = int(mask_s.values.argmax())

        mask_e = dates <= yr_end_ts
        if not mask_e.any():
            break
        e_idx = int(np.where(mask_e.values)[0][-1])

        if e_idx < s_idx:
            break

        n_days = e_idx - s_idx + 1
        windows.append(dict(
            window_id=year, start_idx=s_idx, end_idx=e_idx,
            start_date=dates.iloc[s_idx], end_date=dates.iloc[e_idx],
            n_days=n_days, short_flag=n_days < int(window_days * 0.8),
        ))

    return windows


# ---------------------------------------------------------------------------
# 窓内指標計算 / 統計集計 / 判定基準 (G9 と完全同一)
# ---------------------------------------------------------------------------

def compute_window_metrics(nav, window, wn=None, wb=None, lev_arr=None,
                           trading_days=TRADING_DAYS):
    s = window['start_idx']
    e = window['end_idx'] + 1
    nav_arr = nav.iloc[s:e].values.astype(float)
    n_days  = len(nav_arr)

    if n_days < 5 or nav_arr[0] <= 0:
        return dict(CAGR=np.nan, Sharpe=np.nan, MaxDD=np.nan,
                    Vol=np.nan, PosDay_pct=np.nan, n_days=n_days, Trades_yr=np.nan)

    daily_ret = np.diff(nav_arr) / nav_arr[:-1]
    years     = n_days / trading_days
    cagr      = (nav_arr[-1] / nav_arr[0]) ** (1.0 / years) - 1
    r_std     = np.std(daily_ret, ddof=1)
    sharpe    = (np.mean(daily_ret) / r_std * np.sqrt(trading_days)
                 if r_std > 1e-10 else np.nan)
    running_max = np.maximum.accumulate(nav_arr)
    max_dd  = float((nav_arr / running_max - 1).min())
    vol     = float(r_std * np.sqrt(trading_days))
    pos_pct = float(np.mean(daily_ret > 0))

    trades_yr = np.nan
    if wn is not None and wb is not None and lev_arr is not None:
        n_tr = count_trades_in_window(wn, wb, lev_arr, s, e)
        trades_yr = n_tr / years if years > 0 else np.nan

    return dict(CAGR=float(cagr), Sharpe=float(sharpe), MaxDD=max_dd,
                Vol=vol, PosDay_pct=pos_pct, n_days=n_days,
                Trades_yr=float(trades_yr) if not np.isnan(trades_yr) else np.nan)


def compute_summary_stats(per_df):
    valid   = per_df[~per_df['short_flag']].copy()
    cagrs   = valid['CAGR'].dropna().values
    sharpes = valid['Sharpe'].dropna().values
    n       = len(cagrs)

    if n == 0:
        return {}

    mean_c = float(np.mean(cagrs))
    med_c  = float(np.median(cagrs))
    std_c  = float(np.std(cagrs, ddof=1)) if n > 1 else np.nan
    se     = std_c / np.sqrt(n) if (not np.isnan(std_c) and n > 1) else np.nan
    t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else np.nan

    ci95_lo = mean_c - t_crit * se if not np.isnan(se) else np.nan
    ci95_hi = mean_c + t_crit * se if not np.isnan(se) else np.nan
    t_stat  = mean_c / se if (not np.isnan(se) and se > 0) else np.nan
    t_pval  = float(stats.t.sf(t_stat, df=n - 1)) if not np.isnan(t_stat) else np.nan

    mean_s = float(np.mean(sharpes)) if len(sharpes) > 0 else np.nan
    std_s  = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else np.nan

    oos_start_ts = pd.Timestamp(OOS_START_REF)
    is_mask   = valid['start_date'] < oos_start_ts
    post_mask = valid['start_date'] >= oos_start_ts
    is_cagrs   = valid.loc[is_mask,   'CAGR'].dropna().values
    post_cagrs = valid.loc[post_mask, 'CAGR'].dropna().values
    mean_is   = float(np.mean(is_cagrs))   if len(is_cagrs)   > 0 else np.nan
    mean_post = float(np.mean(post_cagrs)) if len(post_cagrs) > 0 else np.nan
    wfe_post  = (mean_post / mean_is) if (not np.isnan(mean_is) and mean_is != 0) else np.nan

    tr = valid['Trades_yr'].dropna().values if 'Trades_yr' in valid.columns else []
    mean_tr = float(np.mean(tr)) if len(tr) > 0 else np.nan

    return dict(
        n_windows=n, mean_CAGR=mean_c, median_CAGR=med_c, std_CAGR=std_c,
        min_CAGR=float(np.min(cagrs)), max_CAGR=float(np.max(cagrs)),
        P05_CAGR=float(np.percentile(cagrs, 5)),
        P25_CAGR=float(np.percentile(cagrs, 25)),
        P75_CAGR=float(np.percentile(cagrs, 75)),
        P95_CAGR=float(np.percentile(cagrs, 95)),
        WFA_CI95_lo=ci95_lo, WFA_CI95_hi=ci95_hi,
        t_stat=t_stat, t_pvalue=t_pval,
        mean_Sharpe=mean_s, std_Sharpe=std_s,
        mean_CAGR_IS=mean_is, mean_CAGR_postIS=mean_post, WFA_WFE=wfe_post,
        n_windows_IS=len(is_cagrs), n_windows_postIS=len(post_cagrs),
        mean_Trades_yr=mean_tr,
    )


def evaluate_criteria(summary):
    crit_alpha = (summary.get('WFA_CI95_lo', -1) > 0 and
                  summary.get('t_pvalue', 1.0) < 0.05)
    wfe    = summary.get('WFA_WFE', np.nan)
    n_post = summary.get('n_windows_postIS', 0)
    if n_post < 3:
        crit_beta = True
    else:
        crit_beta = (not np.isnan(wfe)) and (0.5 <= wfe <= 2.0)

    crits = dict(alpha=crit_alpha, beta=crit_beta)
    verdict = 'PASS' if (crit_alpha and crit_beta) else ('WARN' if crit_alpha else 'FAIL')
    return verdict, crits


# ---------------------------------------------------------------------------
# フォーマットヘルパー
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
# Markdown レポート（MD_HEADER_STRAT 形式で5戦略比較）
# ---------------------------------------------------------------------------

def generate_md_report(results, sanity, windows, base_dir):
    from _sweep_format import MD_HEADER_STRAT, fmt_row_strat, MD_METRIC_GLOSSARY

    lines = []
    lines += [
        f'# G10: WFA — vz_thr=0.65 × l_max行フロンティア確認 ({len(windows)}窓)',
        '',
        f'作成日: {TODAY}',
        'EVALUATION_STANDARD: **v1.1**',
        '',
        '## §1 目的・背景',
        '',
        'D5実験（vz_thr × l_max 2D グリッド）で vz_thr=0.65 が全 l_max 値にわたって',
        'Sharpe 最高を達成（E4比 +1.5〜+5.8pp）。本 G10 では vz=0.65 行の',
        'l_max=[5.0, 5.5, 6.0, 7.0] を WFA50窓で評価し、プラトーの統計的有効性を確認する。',
        '',
        '| 戦略 | D5 Sharpe | D5 CAGR | D5 MaxDD | WFA 期待 |',
        '|------|----------:|--------:|--------:|---------|',
        '| REF-E4 (vz=0.70, lmax=7.0) | +0.891 | +33.53% | -60.01% | G3 PASS (サニティ) |',
        '| vz065+lmax5.0 | **+0.949** | +33.49% | -51.82% | G9 PASS (CI95_lo=+24.82%) |',
        '| vz065+lmax5.5 | +0.945 | +34.62% | -53.43% | PASS期待 |',
        '| vz065+lmax6.0 | +0.939 | +35.30% | -54.90% | PASS期待 |',
        '| vz065+lmax7.0 | +0.945 | +36.99% | -58.68% | PASS期待 |',
        '',
        '---', '',
        '## §2 セットアップ', '',
        '| 項目 | 値 |',
        '|------|-----|',
        f'| 評価開始 | {EVAL_START} |',
        f'| 評価終了 | {EVAL_END} |',
        f'| 窓長/ステップ | {WINDOW_DAYS}営業日（非重複1年窓） |',
        f'| 総窓数 | {len(windows)} |',
        f'| IS境界 | {IS_END_REF} |',
        f'| OOS開始 | {OOS_START_REF} |',
        '| k_lo / k_hi / k_mid | 0.1 / 0.8 / 0.5（E4固定）|',
        '| LT2 lookback | 750日 |',
        '',
        '---', '',
    ]

    # §3 WFA サマリテーブル (MD_HEADER_STRAT形式)
    lines += ['## §3 WFA サマリ（5戦略 × 9指標）', '']
    hdr1, hdr2 = MD_HEADER_STRAT
    lines += [hdr1, hdr2]
    for sid in STRATEGY_IDS:
        if sid not in results:
            continue
        s = results[sid]['summary']
        r = {
            'CAGR_OOS': s.get('mean_CAGR', np.nan),
            'Sharpe_OOS': s.get('mean_Sharpe', np.nan),
            'MaxDD_FULL': np.nan,
            'Worst10Y_star': np.nan,
            'P10_5Y': np.nan,
            'IS_OOS_gap': np.nan,
            'Trades_yr': s.get('mean_Trades_yr', np.nan),
            'WFA_CI95_lo': s.get('WFA_CI95_lo', np.nan),
            'WFA_WFE': s.get('WFA_WFE', np.nan),
        }
        verdict = results[sid]['verdict']
        label = f'{STRATEGY_LABELS.get(sid, sid)} → **{verdict}**'
        lines.append(fmt_row_strat(label, r))
    lines += ['', MD_METRIC_GLOSSARY, '', '---', '']

    # §4 判定詳細
    lines += ['## §4 判定詳細（α∩β 2基準）', '']
    lines += [
        '| 戦略 | CI95_lo | t_p | WFE | α | β | 判定 |',
        '|------|--------:|----:|----:|:--:|:--:|:---:|',
    ]
    for sid in STRATEGY_IDS:
        if sid not in results:
            continue
        s  = results[sid]['summary']
        cr = results[sid]['criteria']
        v  = results[sid]['verdict']
        a_mark = 'PASS' if cr.get('alpha') else 'FAIL'
        b_mark = 'PASS' if cr.get('beta')  else 'FAIL'
        lines.append(
            f'| {STRATEGY_LABELS.get(sid, sid)} '
            f'| {_fp(s.get("WFA_CI95_lo"))} '
            f'| {_ff(s.get("t_pvalue"), d=4)} '
            f'| {_ff(s.get("WFA_WFE"))} '
            f'| {a_mark} | {b_mark} | **{v}** |'
        )
    lines += ['', '---', '']

    # §5 サニティ
    ci_diff_pp = sanity.get('ci_diff_pp', np.nan)
    wfe_diff   = sanity.get('wfe_diff', np.nan)
    ci_ok  = (not np.isnan(ci_diff_pp)) and abs(ci_diff_pp) <= 0.1
    wfe_ok = (not np.isnan(wfe_diff))   and abs(wfe_diff)   <= 0.001
    lines += ['## §5 サニティチェック（REF-E4 vs G3）', '']
    lines += [
        '| 指標 | G3 値 | G10 REF-E4 | 差分 | 許容 | 判定 |',
        '|------|------:|-----------:|-----:|-----:|:---:|',
        f'| CI95_lo | +{REF_G3_CI95_LO*100:.4f}% | {_fp(results["REF-E4"]["summary"].get("WFA_CI95_lo"), d=4)} | {ci_diff_pp:+.4f} pp | ±0.1 pp | {"OK" if ci_ok else "WARN"} |',
        f'| WFA_WFE | +{REF_G3_WFE:.6f} | {_ff(results["REF-E4"]["summary"].get("WFA_WFE"), d=6)} | {wfe_diff:+.6f} | ±0.001 | {"OK" if wfe_ok else "WARN"} |',
        '',
    ]
    lines += ['---', '']

    # §6 採用判定
    lines += ['## §6 採用変更判定', '']
    e4_ci = REF_G3_CI95_LO
    pass_list = [sid for sid in STRATEGY_IDS[1:]
                 if results.get(sid, {}).get('verdict') == 'PASS']
    best_ci_sid = max(STRATEGY_IDS[1:],
                      key=lambda s: results.get(s, {}).get('summary', {}).get('WFA_CI95_lo', -99))
    best_ci_val = results.get(best_ci_sid, {}).get('summary', {}).get('WFA_CI95_lo', np.nan)

    if pass_list:
        lines += [
            f'- **PASS configs**: {len(pass_list)}/4 ({", ".join(pass_list)})',
            f'- **Best CI95_lo**: {best_ci_sid} → {_fp(best_ci_val)}',
            f'  - E4 G3 CI95_lo (+{e4_ci*100:.2f}%) 比較: {(best_ci_val - e4_ci)*100:+.2f} pp',
            '',
            '**採用変更推奨条件**:',
            '1. vz065-lmax5 が α+β PASS（G9確認済）',
            '2. 隣接点（lmax=5.5/6.0）が少なくとも1点 α PASS → プラトー確認',
            '3. 上記2条件満たす場合: `CURRENT_BEST_STRATEGY.md` を vz065+lmax5 に更新推奨',
        ]
    else:
        lines += ['- **PASS configs: 0/4** → 採用変更保留。E4 Active 継続。']

    lines += ['', '---', '']
    lines += [
        '## §7 再現コマンド', '',
        '```bash',
        'cd "C:\\Users\\user\\Desktop\\投資・不動産\\nasdaq_backtest"',
        'python -X utf8 src/g10_wfa_vz065_lmax_row.py',
        '```', '',
        '*生成スクリプト: `src/g10_wfa_vz065_lmax_row.py` | 準拠: `EVALUATION_STANDARD.md v1.1`*',
    ]

    md_text = '\n'.join(lines)
    md_path = os.path.join(base_dir, f'G10_WFA_VZ065_LMAX_ROW_{TODAY}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    return md_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 72)
    print('G10: WFA — vz_thr=0.65 × l_max行フロンティア確認')
    print(f'実行日: {TODAY}')
    print('=' * 72)

    print('\n[S1] Loading shared assets...')
    assets = load_shared_assets(DATA_PATH)
    dates  = assets['dates']

    print('\n[S2] Building NAVs for 5 strategies...')
    nav_data = build_all_navs(assets)
    for sid in STRATEGY_IDS:
        d = nav_data[sid]
        print(f'  {sid:<22}: NAV range [{d["nav"].min():.4f}, {d["nav"].max():.4f}]')

    print('\n[S3] Generating windows...')
    windows = generate_windows(dates)
    print(f'  {len(windows)} windows: '
          f'{windows[0]["start_date"].date()} ... {windows[-1]["end_date"].date()}')
    short_wins = sum(w['short_flag'] for w in windows)
    print(f'  Short windows: {short_wins}')

    print('\n[S4] Evaluating all strategies across windows...')
    results     = {}
    all_pw_rows = []
    all_sm_rows = []

    for sid in STRATEGY_IDS:
        nav     = nav_data[sid]['nav']
        wn      = nav_data[sid]['wn']
        wb      = nav_data[sid]['wb']
        lev_arr = nav_data[sid]['lev_raw_arr']

        per_rows = []
        for w in windows:
            m = compute_window_metrics(nav, w, wn=wn, wb=wb, lev_arr=lev_arr)
            m.update(dict(
                strategy=sid, window_id=w['window_id'],
                start_date=w['start_date'], end_date=w['end_date'],
                short_flag=w['short_flag'],
            ))
            per_rows.append(m)
            all_pw_rows.append(m)

        per_df  = pd.DataFrame(per_rows)
        summary = compute_summary_stats(per_df)
        verdict, crits = evaluate_criteria(summary)

        results[sid] = dict(per_window=per_df, summary=summary,
                            verdict=verdict, criteria=crits)

        sm_row = {'strategy': sid, 'verdict': verdict,
                  **summary, **{f'crit_{k}': v for k, v in crits.items()}}
        all_sm_rows.append(sm_row)

        ci_lo  = summary.get('WFA_CI95_lo', np.nan)
        ci_hi  = summary.get('WFA_CI95_hi', np.nan)
        wfe    = summary.get('WFA_WFE', np.nan)
        mean_c = summary.get('mean_CAGR', np.nan)
        tp     = summary.get('t_pvalue', np.nan)
        print(
            f'  {sid:<22}: mean_CAGR={mean_c*100:+.2f}%  '
            f'CI95=[{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]  '
            f't_p={tp:.4f}  WFE={wfe:+.3f}  => {verdict}'
        )

    print('\n[S5] Sanity checks (REF-E4 vs G3)...')
    ref_ci  = results['REF-E4']['summary'].get('WFA_CI95_lo', np.nan)
    ref_wfe = results['REF-E4']['summary'].get('WFA_WFE', np.nan)
    ci_diff_pp = (ref_ci  - REF_G3_CI95_LO) * 100 if not np.isnan(ref_ci)  else np.nan
    wfe_diff   = (ref_wfe - REF_G3_WFE)            if not np.isnan(ref_wfe) else np.nan
    ci_ok  = (not np.isnan(ci_diff_pp)) and abs(ci_diff_pp) <= 0.1
    wfe_ok = (not np.isnan(wfe_diff))   and abs(wfe_diff)   <= 0.001
    print(f'  REF-E4 CI95_lo: {ref_ci*100:+.4f}%  (G3 {REF_G3_CI95_LO*100:+.4f}%, diff {ci_diff_pp:+.4f} pp) => {"OK" if ci_ok else "WARN"}')
    print(f'  REF-E4 WFA_WFE: {ref_wfe:+.6f}  (G3 {REF_G3_WFE:+.6f}, diff {wfe_diff:+.6f}) => {"OK" if wfe_ok else "WARN"}')

    sanity = {'ci_diff_pp': ci_diff_pp, 'wfe_diff': wfe_diff,
              'ci_ok': ci_ok, 'wfe_ok': wfe_ok}

    print('\n[S6] Summary:')
    print('=' * 100)
    print(f'{"Strategy":<22} {"mean_CAGR":>10} {"CI95_lo":>9} {"CI95_hi":>9}'
          f' {"t_p":>9} {"WFE":>7} {"Verdict":>8}')
    print('-' * 100)
    for sid in STRATEGY_IDS:
        s = results[sid]['summary']
        v = results[sid]['verdict']
        print(
            f'{sid:<22}'
            f' {s.get("mean_CAGR", 0)*100:>+9.2f}%'
            f' {s.get("WFA_CI95_lo", np.nan)*100:>+8.2f}%'
            f' {s.get("WFA_CI95_hi", np.nan)*100:>+8.2f}%'
            f' {s.get("t_pvalue", np.nan):>9.6f}'
            f' {s.get("WFA_WFE", np.nan):>7.3f}'
            f' {v:>8}'
        )
    print('=' * 100)

    print('\n[S7] Saving CSVs...')
    pw_path = os.path.join(BASE, 'g10_wfa_vz065_lmax_row_per_window.csv')
    sm_path = os.path.join(BASE, 'g10_wfa_vz065_lmax_row_summary.csv')
    pd.DataFrame(all_pw_rows).to_csv(pw_path, index=False, float_format='%.6f')
    pd.DataFrame(all_sm_rows).to_csv(sm_path, index=False, float_format='%.6f')
    print(f'  Per-window: {pw_path}')
    print(f'  Summary:    {sm_path}')

    print('[S8] Generating Markdown report...')
    md_path = generate_md_report(results, sanity, windows, BASE)
    print(f'  Report: {md_path}')

    print('\nDone.')
    return results


if __name__ == '__main__':
    main()
