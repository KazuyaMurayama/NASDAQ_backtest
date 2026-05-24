"""
G4: Walk-Forward Analysis — F7-v3 定式A (tilt=2.0, cap=0.10)
===============================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

対象戦略 (2本):
  F7v3-A2:        E4 base + 定式A bull-tilt (tilt=2.0, cap=0.10)
  REF-E4-sanity:  G3 と完全同一の E4 戦略 (tilt なし)
                  → CI95_lo=+26.51%, WFE=+1.131 が再現されれば sanity OK

窓設計: G3 と完全同一
  - 評価開始: 1977-01-03 (LT2 warmup)
  - 評価終了: 2026-03-26
  - 窓長: 252営業日 / ステップ 252営業日（非重複、カレンダー年アンカー）

判定基準 (EVALUATION_STANDARD v1.1 §3.9/§3.10):
  α: WFA_CI95_lo > 0 AND t_p < 0.05
  β: WFA_WFE ∈ [0.5, 2.0]
  総合: α+β→PASS / αのみ→WARN / α FAIL→FAIL

出力:
  - g4_wfa_f7v3_per_window.csv
  - g4_wfa_f7v3_summary.csv
  - G4_WFA_F7V3_2026-05-24.md
"""

import sys
import os
import types

# multitasking スタブ (yfinance 依存回避)
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
# E4 base パラメータ (G3 と同じ)
K_LO   = 0.1
K_HI   = 0.8
VZ_THR = 0.7
K_MID  = 0.5
N_LT2  = 750

# F7-v3 定式A パラメータ
TILT_A     = 2.0
TILT_CAP_A = 0.10

WINDOW_DAYS   = 252
STEP_DAYS     = 252
EVAL_START    = '1977-01-03'
EVAL_END      = '2026-03-26'
IS_END_REF    = IS_END       # '2021-05-07'
OOS_START_REF = OOS_START    # '2021-05-08'
TODAY         = '2026-05-24'

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5,
                n=20, l_min=1.0, l_max=7.0, step=0.5)

STRATEGY_IDS = ['F7v3-A2', 'REF-E4-sanity']

STRATEGY_LABELS = {
    'F7v3-A2':       f'F7v3-A2 (E4 + 定式A tilt={TILT_A}, cap={TILT_CAP_A}) ◆',
    'REF-E4-sanity': 'REF-E4-sanity (G3 E4-RegimeKLT と同一) ← sanity ◆',
}

# サニティ参照値 (G3 で確認済み)
REF_CI95_LO = 0.265093   # G3: E4-RegimeKLT WFA_CI95_lo = +26.51%
REF_WFE     = 1.130664   # G3: E4-RegimeKLT WFA_WFE = +1.131

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    # Scenario D 資産
    gold_1x_local = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x_local, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                          bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print('  Scenario D assets done.')

    # DH Dyn A シグナル (raw_a2 / vz / lev / wn / wg / wb)
    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    print(f'  DH Dyn A: {n_tr} trades, {n_tr/n_years:.1f}/yr')

    # LT2 シグナル (N=750)
    lt_sig_raw = build_lt_signal(close, 'LT2', N=N_LT2)
    print('  LT2 signal done.')

    # === E4: Regime-conditional k_lt (vz レジーム依存)  — G3 と同一 ===
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    k_dyn = np.where(vz_arr >  VZ_THR, K_HI,
             np.where(vz_arr < -VZ_THR, K_LO, K_MID))
    lt_bias_e4 = pd.Series(
        np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
        index=lt_sig_raw.index,
    )
    lev_mod_e4 = apply_lt_mode_b(lev_A, lt_bias_e4, l_min=0.0, l_max=1.0)

    # S2 CFD レバレッジ系列
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)
    print('  S2 leverage done.')

    return dict(
        close=close, ret=ret, dates=dates, n=n, n_years=n_years, sofr=sofr,
        gold_2x=gold_2x, bond_3x=bond_3x,
        raw_a2=raw_a2, vz=vz,
        lev_A=lev_A,
        lev_mod_e4=lev_mod_e4,
        wn_A=wn_A, wg_A=wg_A, wb_A=wb_A,
        L_s2=L_s2,
    )


# ---------------------------------------------------------------------------
# 戦略 NAV 構築 (2戦略)
# ---------------------------------------------------------------------------

def build_all_navs(assets: dict) -> dict:
    a = assets
    close, dates, sofr = a['close'], a['dates'], a['sofr']
    gold_2x, bond_3x = a['gold_2x'], a['bond_3x']
    wn_A, wg_A, wb_A = a['wn_A'], a['wg_A'], a['wb_A']
    raw_a2 = a['raw_a2']
    lev_mod_e4 = a['lev_mod_e4']
    L_s2 = a['L_s2']

    navs = {}

    # --- REF-E4-sanity (G3 と完全同一: tilt なし) ---
    navs['REF-E4-sanity'] = build_nav_strategy(
        close, lev_mod_e4, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )

    # --- F7v3-A2: E4 + 定式A bull-tilt (tilt=2.0, cap=0.10) ---
    raw_a2_vals = raw_a2.values if hasattr(raw_a2, 'values') else np.asarray(raw_a2)
    bull_mask   = raw_a2_vals > THRESHOLD

    tilt_amount_raw = TILT_A * (raw_a2_vals - THRESHOLD) * (1.0 - raw_a2_vals)
    tilt_amount     = np.where(bull_mask,
                                np.clip(tilt_amount_raw, 0.0, TILT_CAP_A),
                                0.0)

    wn_tilted = wn_A + tilt_amount
    wb_tilted = np.clip(wb_A - tilt_amount, 0.0, wb_A)
    wg_tilted = wg_A  # gold unchanged

    navs['F7v3-A2'] = build_nav_strategy(
        close, lev_mod_e4, wn_tilted, wg_tilted, wb_tilted, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )

    return navs


# ---------------------------------------------------------------------------
# Window 生成 (カレンダー年アンカー方式 — G1/G2/G3 と同一)
# ---------------------------------------------------------------------------

def generate_windows(dates: pd.Series,
                     eval_start: str = EVAL_START,
                     eval_end: str = EVAL_END,
                     window_days: int = WINDOW_DAYS,
                     step_days: int = STEP_DAYS) -> list:
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
            window_id  = year,
            start_idx  = s_idx,
            end_idx    = e_idx,
            start_date = dates.iloc[s_idx],
            end_date   = dates.iloc[e_idx],
            n_days     = n_days,
            short_flag = n_days < int(window_days * 0.8),
        ))

    return windows


# ---------------------------------------------------------------------------
# 窓内指標計算
# ---------------------------------------------------------------------------

def compute_window_metrics(nav: pd.Series, window: dict,
                            trading_days: int = TRADING_DAYS) -> dict:
    s = window['start_idx']
    e = window['end_idx'] + 1
    nav_arr = nav.iloc[s:e].values.astype(float)
    n_days  = len(nav_arr)

    if n_days < 5 or nav_arr[0] <= 0:
        return dict(CAGR=np.nan, Sharpe=np.nan, MaxDD=np.nan,
                    Vol=np.nan, PosDay_pct=np.nan, n_days=n_days)

    daily_ret = np.diff(nav_arr) / nav_arr[:-1]
    years     = n_days / trading_days
    cagr      = (nav_arr[-1] / nav_arr[0]) ** (1.0 / years) - 1

    r_std  = np.std(daily_ret, ddof=1)
    sharpe = (np.mean(daily_ret) / r_std * np.sqrt(trading_days)
              if r_std > 1e-10 else np.nan)

    running_max = np.maximum.accumulate(nav_arr)
    max_dd  = float((nav_arr / running_max - 1).min())
    vol     = float(r_std * np.sqrt(trading_days))
    pos_pct = float(np.mean(daily_ret > 0))

    return dict(CAGR=float(cagr), Sharpe=float(sharpe), MaxDD=max_dd,
                Vol=vol, PosDay_pct=pos_pct, n_days=n_days)


# ---------------------------------------------------------------------------
# 統計集計
# ---------------------------------------------------------------------------

def compute_summary_stats(per_df: pd.DataFrame) -> dict:
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

    mean_s  = float(np.mean(sharpes)) if len(sharpes) > 0 else np.nan
    std_s   = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else np.nan

    oos_start_ts = pd.Timestamp(OOS_START_REF)
    is_mask   = valid['start_date'] < oos_start_ts
    post_mask = valid['start_date'] >= oos_start_ts
    is_cagrs   = valid.loc[is_mask,   'CAGR'].dropna().values
    post_cagrs = valid.loc[post_mask, 'CAGR'].dropna().values
    mean_is   = float(np.mean(is_cagrs))   if len(is_cagrs)   > 0 else np.nan
    mean_post = float(np.mean(post_cagrs)) if len(post_cagrs) > 0 else np.nan
    wfe_post  = (mean_post / mean_is) if (not np.isnan(mean_is) and mean_is != 0) else np.nan

    return dict(
        n_windows=n,
        mean_CAGR=mean_c, median_CAGR=med_c, std_CAGR=std_c,
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
    )


# ---------------------------------------------------------------------------
# 判定基準
# ---------------------------------------------------------------------------

def evaluate_criteria(summary: dict) -> tuple:
    crit_alpha = (summary.get('WFA_CI95_lo', -1) > 0 and
                  summary.get('t_pvalue', 1.0) < 0.05)
    wfe    = summary.get('WFA_WFE', np.nan)
    n_post = summary.get('n_windows_postIS', 0)
    if n_post < 3:
        crit_beta = True
    else:
        crit_beta = (not np.isnan(wfe)) and (0.5 <= wfe <= 2.0)

    crits = dict(alpha=crit_alpha, beta=crit_beta)
    if crit_alpha and crit_beta:
        verdict = 'PASS'
    elif crit_alpha:
        verdict = 'WARN'
    else:
        verdict = 'FAIL'

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
# Markdown レポート生成
# ---------------------------------------------------------------------------

def generate_md_report(results: dict, sanity: dict, windows: list,
                        base_dir: str) -> str:
    lines = []
    lines += [
        f'# G4: Walk-Forward Analysis — F7-v3 定式A (tilt={TILT_A}) × {len(windows)}窓',
        '',
        f'作成日: {TODAY}',
        f'EVALUATION_STANDARD: v1.1',
        '目的: F7-v3 定式A (tilt=2.0, cap=0.10) の統計的有効性を 50窓 WFA で確認し、'
        'PASS → 正式 Active 昇格の根拠を作る。',
        '',
        '---', '',
        '## 1. セットアップ（G3 と同一窓設計）', '',
        '| 項目 | 値 |',
        '|------|-----|',
        f'| 評価開始 | {EVAL_START} (LT2 warmup完了) |',
        f'| 評価終了 | {EVAL_END} |',
        f'| 窓長 | {WINDOW_DAYS}営業日 (1年) |',
        f'| ステップ | {STEP_DAYS}営業日 (非重複) |',
        f'| 総窓数 | {len(windows)} |',
        f'| IS境界 | {IS_END_REF} |',
        f'| OOS開始 | {OOS_START_REF} |',
        f'| E4 base | k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, k_mid={K_MID}, LT2-N750 |',
        f'| F7-v3 定式A | tilt={TILT_A}, cap={TILT_CAP_A}, THR={THRESHOLD} |',
        '',
        '**判定基準 (EVALUATION_STANDARD v1.1 §3.9/§3.10)**',
        '',
        '| 基準 | 条件 |',
        '|------|------|',
        '| α (統計的有意性) | WFA_CI95_lo > 0 AND t_p < 0.05 |',
        '| β (IS↔OOS安定) | WFA_WFE ∈ [0.5, 2.0] |',
        '| **総合** | α+β PASS → PASS / α のみ → WARN / α FAIL → FAIL |',
        '',
        '**F7-v3 定式A 構築 (E4 base + bull-tilt)**:',
        '```',
        '# E4 base lev_mod (G3 と同一)',
        'k_dyn = np.where(vz > +0.7, 0.8,',
        '         np.where(vz < -0.7, 0.1, 0.5))',
        'lt_bias = clip(-k_dyn * lt_sig * 0.5, -0.5, 0.5)',
        'lev_mod_e4 = clip(lev_A + lt_bias, 0, 1)',
        '',
        '# F7-v3 定式A bull-tilt (tilt=2.0, cap=0.10)',
        'bull_mask = raw_a2 > 0.15',
        'tilt_amount = 2.0 * (raw_a2 - 0.15) * (1.0 - raw_a2)',
        'tilt_amount = where(bull_mask, clip(tilt_amount, 0, 0.10), 0)',
        'wn_tilted = wn_A + tilt_amount',
        'wb_tilted = clip(wb_A - tilt_amount, 0, wb_A)',
        'wg_tilted = wg_A  # unchanged',
        '```',
        '',
        '---', '',
    ]

    # §2 WFA サマリ
    lines += ['## 2. WFA サマリ（2戦略）', '']
    lines += [
        '| 戦略 | n | mean_ret | std_ret | t_stat | t_p | CI95_lo | CI95_hi | WFE | 判定 |',
        '|------|--:|---------:|--------:|-------:|----:|--------:|--------:|----:|:---:|',
    ]
    for sid in STRATEGY_IDS:
        if sid not in results:
            continue
        s = results[sid]['summary']
        v = results[sid]['verdict']
        label = STRATEGY_LABELS.get(sid, sid)
        lines.append(
            f'| {label} '
            f'| {s.get("n_windows", "N/A")} '
            f'| {_fp(s.get("mean_CAGR"))} '
            f'| {_fp(s.get("std_CAGR"))} '
            f'| {_ff(s.get("t_stat"), d=3)} '
            f'| {_ff(s.get("t_pvalue"), d=4)} '
            f'| {_fp(s.get("WFA_CI95_lo"))} '
            f'| {_fp(s.get("WFA_CI95_hi"))} '
            f'| {_ff(s.get("WFA_WFE"))} '
            f'| **{v}** |'
        )
    lines += ['', '---', '']

    # §3 判定詳細
    lines += ['## 3. 判定詳細 (α∩β 2基準)', '']
    lines += [
        '| 戦略 | α: WFA_CI95_lo | α: t_p | β: WFA_WFE | 判定 |',
        '|------|---------------:|-------:|-----------:|:---:|',
    ]
    for sid in STRATEGY_IDS:
        if sid not in results:
            continue
        s  = results[sid]['summary']
        cr = results[sid]['criteria']
        v  = results[sid]['verdict']
        a_mark = 'OK' if cr.get('alpha') else 'NG'
        b_mark = 'OK' if cr.get('beta')  else 'NG'
        lines.append(
            f'| {STRATEGY_LABELS.get(sid, sid)} '
            f'| [{a_mark}] {_fp(s.get("WFA_CI95_lo"))} '
            f'| {_ff(s.get("t_pvalue"), d=4)} '
            f'| [{b_mark}] {_ff(s.get("WFA_WFE"))} '
            f'| **{v}** |'
        )
    lines += ['', '---', '']

    # §4 sanity 確認
    ref_ci  = results.get('REF-E4-sanity', {}).get('summary', {}).get('WFA_CI95_lo', np.nan)
    ref_wfe = results.get('REF-E4-sanity', {}).get('summary', {}).get('WFA_WFE', np.nan)
    ci_diff  = (ref_ci  - REF_CI95_LO) * 100 if not np.isnan(ref_ci)  else np.nan
    wfe_diff = (ref_wfe - REF_WFE)            if not np.isnan(ref_wfe) else np.nan
    ci_ok    = (not np.isnan(ci_diff))  and abs(ci_diff)  <= 1.0   # ±1pp (完全一致期待)
    wfe_ok   = (not np.isnan(wfe_diff)) and abs(wfe_diff) <= 0.050 # ±0.050 (完全一致期待)
    lines += ['## 4. サニティチェック (REF-E4-sanity vs G3 結果)', '']
    lines += [
        '| 項目 | G3 値 | G4 実測 | 差分 | 判定 |',
        '|------|------:|--------:|-----:|:---:|',
        f'| CI95_lo | +{REF_CI95_LO*100:.2f}% | {_fp(ref_ci)} | {ci_diff:+.3f} pp | {"OK" if ci_ok else "WARN"} |',
        f'| WFA_WFE | +{REF_WFE:.3f} | {_ff(ref_wfe)} | {wfe_diff:+.4f} | {"OK" if wfe_ok else "WARN"} |',
        '',
        'REF-E4-sanity は G3 の E4-RegimeKLT と完全同一の戦略設定。'
        '完全一致 (±0pp / ±0.000) が期待される。',
        '',
        '---', '',
    ]

    # §5 正式昇格判定
    lines += ['## 5. F7v3-A2 正式昇格判定', '']
    lines += [
        '| 戦略 | CI95_lo | WFE | t_p | α | β | 総合 | 昇格判断 |',
        '|------|--------:|----:|----:|:--:|:--:|:---:|:--------|',
    ]
    f7v3 = results.get('F7v3-A2', {})
    if f7v3:
        s  = f7v3['summary']
        cr = f7v3['criteria']
        v  = f7v3['verdict']
        a_mark = 'PASS' if cr.get('alpha') else 'FAIL'
        b_mark = 'PASS' if cr.get('beta')  else 'FAIL'
        if v == 'PASS':
            decision = '**正式 Active 昇格** (CURRENT_BEST_STRATEGY.md 更新候補)'
        elif v == 'WARN':
            decision = 'SHORTLISTED 維持 (β基準未達; 要再検討)'
        else:
            decision = '昇格保留 (α基準未達; E4 へ差し戻し)'
        lines.append(
            f'| {STRATEGY_LABELS["F7v3-A2"]} '
            f'| {_fp(s.get("WFA_CI95_lo"))} '
            f'| {_ff(s.get("WFA_WFE"))} '
            f'| {_ff(s.get("t_pvalue"), d=4)} '
            f'| {a_mark} | {b_mark} '
            f'| **{v}** | {decision} |'
        )
    # REF-E4-sanity 行も参考表示
    ref_res = results.get('REF-E4-sanity', {})
    if ref_res:
        s  = ref_res['summary']
        cr = ref_res['criteria']
        v  = ref_res['verdict']
        a_mark = 'PASS' if cr.get('alpha') else 'FAIL'
        b_mark = 'PASS' if cr.get('beta')  else 'FAIL'
        lines.append(
            f'| {STRATEGY_LABELS["REF-E4-sanity"]} '
            f'| {_fp(s.get("WFA_CI95_lo"))} '
            f'| {_ff(s.get("WFA_WFE"))} '
            f'| {_ff(s.get("t_pvalue"), d=4)} '
            f'| {a_mark} | {b_mark} '
            f'| **{v}** | (参考: G3 で同 PASS 既確認) |'
        )
    lines += ['', '---', '']

    # §6 再現コマンド
    lines += [
        '## 6. 再現コマンド', '',
        '```',
        'python -X utf8 src/g4_wfa_f7v3.py',
        '```', '',
        '---', '',
        f'*生成スクリプト: `src/g4_wfa_f7v3.py`  '
        f'準拠: `EVALUATION_STANDARD.md v1.1`*',
        f'*参照: `CURRENT_BEST_STRATEGY.md`, `F7V3_BULL_TILT_2026-05-24.md`, '
        f'`G3_WFA_E4_2026-05-24.md`, `g3_wfa_e4_summary.csv`*',
    ]

    md_text = '\n'.join(lines)
    md_path = os.path.join(base_dir, f'G4_WFA_F7V3_{TODAY}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    return md_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('G4: Walk-Forward Analysis — F7-v3 定式A (tilt=2.0) × 非重複1年窓')
    print(f'実行日: {TODAY}')
    print('=' * 70)

    # S1: 共有資産ロード
    print('\n[S1] Loading shared assets...')
    assets = load_shared_assets(DATA_PATH)
    dates  = assets['dates']

    # S2: 2戦略 NAV 構築
    print('\n[S2] Building NAVs for 2 strategies...')
    navs = build_all_navs(assets)
    for sid in STRATEGY_IDS:
        print(f'  {sid}: NAV range [{navs[sid].min():.4f}, {navs[sid].max():.4f}]')

    # S3: Window 生成
    print('\n[S3] Generating windows...')
    windows = generate_windows(dates, EVAL_START, EVAL_END, WINDOW_DAYS, STEP_DAYS)
    print(f'  {len(windows)} windows: '
          f'{windows[0]["start_date"].date()} ... {windows[-1]["end_date"].date()}')
    short_wins = sum(w['short_flag'] for w in windows)
    print(f'  Short windows (< 80% of target): {short_wins}')

    # 重複なし確認
    for i in range(len(windows) - 1):
        assert windows[i]['end_idx'] < windows[i+1]['start_idx'], \
            f'Overlapping windows at {i} and {i+1}'
    print('  Non-overlap check: OK')

    # S4: 全戦略評価
    print('\n[S4] Evaluating all strategies across windows...')
    results     = {}
    all_pw_rows = []
    all_sm_rows = []

    for sid in STRATEGY_IDS:
        nav = navs[sid]
        per_rows = []
        for w in windows:
            m = compute_window_metrics(nav, w)
            m.update(dict(
                strategy   = sid,
                window_id  = w['window_id'],
                start_date = w['start_date'],
                end_date   = w['end_date'],
                short_flag = w['short_flag'],
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

        mean_c = summary.get('mean_CAGR', np.nan)
        ci_lo  = summary.get('WFA_CI95_lo', np.nan)
        ci_hi  = summary.get('WFA_CI95_hi', np.nan)
        tp     = summary.get('t_pvalue', np.nan)
        wfe    = summary.get('WFA_WFE', np.nan)
        print(
            f'  {sid:<16}: mean_CAGR={mean_c*100:+.2f}%  '
            f'CI95=[{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]  '
            f't_p={tp:.4f}  WFE={wfe:+.3f}  => {verdict}'
        )

    # S5: サニティチェック
    print('\n[S5] Sanity checks (REF-E4-sanity vs G3)...')
    ref_ci  = results['REF-E4-sanity']['summary'].get('WFA_CI95_lo', np.nan)
    ref_wfe = results['REF-E4-sanity']['summary'].get('WFA_WFE', np.nan)
    ci_diff  = (ref_ci  - REF_CI95_LO) * 100 if not np.isnan(ref_ci)  else np.nan
    wfe_diff = (ref_wfe - REF_WFE)            if not np.isnan(ref_wfe) else np.nan
    ci_ok    = (not np.isnan(ci_diff))  and abs(ci_diff)  <= 1.0
    wfe_ok   = (not np.isnan(wfe_diff)) and abs(wfe_diff) <= 0.050
    print(f'  REF CI95_lo: {ref_ci*100:+.2f}%  (G3 {REF_CI95_LO*100:+.2f}%,'
          f' diff {ci_diff:+.3f} pp) => {"OK" if ci_ok else "WARN"}')
    print(f'  REF WFA_WFE: {ref_wfe:+.3f}  (G3 {REF_WFE:+.3f},'
          f' diff {wfe_diff:+.4f}) => {"OK" if wfe_ok else "WARN"}')

    sanity = {'ci_diff_pp': ci_diff, 'wfe_diff': wfe_diff}

    # S6: コンソールサマリ
    print('\n[S6] Summary:')
    print('=' * 100)
    print(f'{"Strategy":<18} {"mean_CAGR":>10} {"CI95_lo":>9} {"CI95_hi":>9}'
          f' {"t_p":>7} {"Sharpe":>7} {"WFE":>7} {"Verdict":>8}')
    print('-' * 100)
    for sid in STRATEGY_IDS:
        s = results[sid]['summary']
        v = results[sid]['verdict']
        print(
            f'{sid:<18}'
            f' {s.get("mean_CAGR", 0)*100:>+9.2f}%'
            f' {s.get("WFA_CI95_lo", np.nan)*100:>+8.2f}%'
            f' {s.get("WFA_CI95_hi", np.nan)*100:>+8.2f}%'
            f' {s.get("t_pvalue", np.nan):>7.4f}'
            f' {s.get("mean_Sharpe", np.nan):>7.3f}'
            f' {s.get("WFA_WFE", np.nan):>7.3f}'
            f' {v:>8}'
        )
    print('=' * 100)

    # S7: CSV 保存
    print('\n[S7] Saving CSVs...')
    pw_path = os.path.join(BASE, 'g4_wfa_f7v3_per_window.csv')
    sm_path = os.path.join(BASE, 'g4_wfa_f7v3_summary.csv')
    pd.DataFrame(all_pw_rows).to_csv(pw_path, index=False, float_format='%.6f')
    pd.DataFrame(all_sm_rows).to_csv(sm_path, index=False, float_format='%.6f')
    print(f'  Per-window: {pw_path}')
    print(f'  Summary:    {sm_path}')

    # S8: MD レポート
    print('[S8] Generating Markdown report...')
    md_path = generate_md_report(results, sanity, windows, BASE)
    print(f'  Report: {md_path}')

    print('\nDone.')

    return results


if __name__ == '__main__':
    main()
