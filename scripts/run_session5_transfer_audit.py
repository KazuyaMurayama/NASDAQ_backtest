"""Session 5 B — Transfer nasdaq_mom63 × M6 defensive overlay to S2 (D5) and E4 (Active).

Session 4 ADOPT'd nasdaq_mom63 × S3 (DH-W1) × M6 defensive.
This script tests whether the same overlay transfers to:
  - S2 (vz=0.65+l_max=5) baseline      — cache: vz065lmax5_nav_cache.pkl
  - E4 (current Active = RegimeKLT)    — cache: e4_nav_cache.pkl

Pipeline per target baseline (matches Session 4 audit grade)
------------------------------------------------------------
1. Build candidate NAV: ret × (lev_mod_native × M6_def(sig_q)) → reuse cached
   wn/wg/wb + L_s2 leg via g14.build_nav_strategy
2. WFA 50 yearly windows → WFE + CI95_lo CAGR
3. Block bootstrap 10,000 (block=60, paired, multi-metric: CAGR/Sharpe/MaxDD)
4. 9+1 metrics (full grade)
5. Per-target Markdown report
6. Combined session5 transfer report
"""
from __future__ import annotations
import os
import sys
import pickle
import types
from pathlib import Path
from datetime import datetime

# multitasking stub (matches integration modules)
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src' / 'integration'))

import numpy as np
import pandas as pd
from scipy import stats

from integration.build_strategy_with_signal import _build_mult_array  # noqa: E402
from g14_wfa_sbi_cfd import generate_windows, compute_window_metrics, build_nav_strategy  # noqa: E402
from g18_daily_trade_cost_wfa import metrics_from_nav, OOS_START_TS, OOS_END_TS  # noqa: E402
from corrected_strategy_backtest import TRADING_DAYS  # noqa: E402
from signals.quantize import quantile_cut  # noqa: E402
from signals.timing import apply_publication_lag  # noqa: E402


CACHE_DIR = REPO / 'audit_results' / '_cache'
OUT_DIR = REPO / 'data' / 'signals' / 'expansion'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MACRO_CSV = REPO / 'data' / 'macro_features.csv'

CFD_SPREAD_LOW = 0.0020

N_BOOTSTRAP = 10000
BLOCK_SIZE = 60
RNG_SEED = 42

# Transfer targets
TARGETS = [
    dict(id='S2_D5',     label='S2 (D5 vz=0.65, lmax=5)',
         cache='vz065lmax5_nav_cache.pkl',
         lev_key='lev_mod_065', leverage_key='L_s2_lmax5',
         wn_key='wn_A', wg_key='wg_A', wb_key='wb_A',
         nav_key='nav_vz065lmax5'),
    dict(id='E4_Active', label='E4 RegimeKLT (current §1 Active)',
         cache='e4_nav_cache.pkl',
         lev_key='lev_mod_e4', leverage_key='L_s2',
         wn_key='wn_A', wg_key='wg_A', wb_key='wb_A',
         nav_key='nav_e4'),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def load_signal(name: str) -> pd.Series:
    df = pd.read_csv(MACRO_CSV, parse_dates=['date'], index_col='date').sort_index()
    return df[name].dropna()


def full_sharpe(nav: pd.Series) -> float:
    r = nav.pct_change().fillna(0).values
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-12:
        return float('nan')
    return float(np.mean(r) / sd * np.sqrt(252))


def ci95_t(arr: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 2:
        return float('nan'), float('nan'), float('nan')
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(n) if n > 1 else float('nan')
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mu - tcrit * se, mu + tcrit * se, mu


def trades_per_yr_from_levraw(lev_arr: np.ndarray, dates: pd.Series,
                              threshold: float = 0.05) -> float:
    lr = np.asarray(lev_arr, dtype=float)
    base = np.where(np.abs(lr[:-1]) > 1e-9, np.abs(lr[:-1]), 1.0)
    rel = np.abs(lr[1:] - lr[:-1]) / base
    flips = int((rel > threshold).sum())
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    return flips / max(years, 1e-9)


# ------------------------------------------------------------------
# Native CFD builder for S2/E4 (both use same g14.build_nav_strategy shape)
# ------------------------------------------------------------------

def build_cfd_native(target: dict, sig_q_lagged: pd.Series,
                     method: str, direction: str) -> tuple[pd.Series, np.ndarray, pd.Series]:
    """Build candidate NAV by injecting (M6 defensive) mult into cached CFD baseline.

    Returns (nav_DatetimeIndex, lev_mod_modulated_array, dates_Series).
    """
    obj = pickle.load(open(CACHE_DIR / target['cache'], 'rb'))
    dates = obj['dates']
    close = obj['close']
    ret = obj['ret']

    mult_arr = _build_mult_array(
        sig_q_lagged, method, direction,
        target_dates=dates,
        ret_values=ret.values if method == 'M4' else None,
    )
    lev_mod_base = np.asarray(obj[target['lev_key']], dtype=float)
    lev_mod_mod = lev_mod_base * mult_arr

    L_s2_obj = obj[target['leverage_key']]
    L_s2_arr = L_s2_obj.values if hasattr(L_s2_obj, 'values') else L_s2_obj

    nav = build_nav_strategy(
        close, lev_mod_mod,
        obj[target['wn_key']], obj[target['wg_key']], obj[target['wb_key']], dates,
        obj['gold_2x'], obj['bond_3x'], obj['sofr'],
        nas_mode='CFD',
        cfd_leverage=L_s2_arr,
        cfd_spread=CFD_SPREAD_LOW,
    )
    # Reindex
    dt_index = pd.DatetimeIndex(pd.to_datetime(dates.values))
    nav_dt = pd.Series(nav.values, index=dt_index).dropna()
    return nav_dt, lev_mod_mod, dates


def load_baseline_nav(target: dict) -> tuple[pd.Series, np.ndarray, pd.Series, pd.Series]:
    """Load baseline NAV + lev_mod + dates + ret from cache."""
    obj = pickle.load(open(CACHE_DIR / target['cache'], 'rb'))
    dates = obj['dates']
    nav_raw = obj[target['nav_key']]
    if hasattr(nav_raw, 'values'):
        nav_vals = nav_raw.values
    else:
        nav_vals = np.asarray(nav_raw)
    dt_index = pd.DatetimeIndex(pd.to_datetime(dates.values))
    nav = pd.Series(nav_vals, index=dt_index).dropna()
    lev_mod_base = np.asarray(obj[target['lev_key']], dtype=float)
    ret = obj['ret']
    return nav, lev_mod_base, dates, ret


# ------------------------------------------------------------------
# WFA 50 windows
# ------------------------------------------------------------------

def run_wfa(nav_b: pd.Series, nav_c: pd.Series, dates: pd.Series) -> tuple[pd.DataFrame, dict]:
    nav_b_pos = pd.Series(nav_b.values, index=range(len(nav_b)))
    nav_c_pos = pd.Series(nav_c.values, index=range(len(nav_c)))

    windows = generate_windows(dates)
    rows = []
    for w in windows:
        m_b = compute_window_metrics(nav_b_pos, w)
        m_c = compute_window_metrics(nav_c_pos, w)
        is_oos = 'OOS' if w['end_date'] >= OOS_START_TS else 'IS'
        rows.append(dict(
            window_id=w['window_id'],
            window_kind=is_oos,
            start_date=str(w['start_date'].date()),
            end_date=str(w['end_date'].date()),
            n_days=w['n_days'],
            short_flag=w['short_flag'],
            baseline_cagr=m_b['CAGR'],
            candidate_cagr=m_c['CAGR'],
            baseline_sharpe=m_b['Sharpe'],
            candidate_sharpe=m_c['Sharpe'],
            baseline_maxdd=m_b['MaxDD'],
            candidate_maxdd=m_c['MaxDD'],
            cand_minus_base_cagr=m_c['CAGR'] - m_b['CAGR'],
            cand_minus_base_sharpe=m_c['Sharpe'] - m_b['Sharpe'],
        ))
    df = pd.DataFrame(rows)

    valid = df[~df['short_flag']].copy()
    oos_mask = valid['window_kind'] == 'OOS'

    full_sh_b = full_sharpe(nav_b)
    full_sh_c = full_sharpe(nav_c)

    mean_sh_b_all = float(valid['baseline_sharpe'].mean())
    mean_sh_c_all = float(valid['candidate_sharpe'].mean())
    wfe_b = mean_sh_b_all / full_sh_b if full_sh_b and not np.isnan(full_sh_b) else float('nan')
    wfe_c = mean_sh_c_all / full_sh_c if full_sh_c and not np.isnan(full_sh_c) else float('nan')

    mean_sh_b_oos = float(valid.loc[oos_mask, 'baseline_sharpe'].mean()) if oos_mask.any() else float('nan')
    mean_sh_c_oos = float(valid.loc[oos_mask, 'candidate_sharpe'].mean()) if oos_mask.any() else float('nan')

    ci_lo_c, ci_hi_c, mu_c = ci95_t(valid['candidate_cagr'].values)
    ci_lo_b, ci_hi_b, mu_b = ci95_t(valid['baseline_cagr'].values)
    ci_lo_sh, ci_hi_sh, mu_sh = ci95_t(valid['candidate_sharpe'].values)

    summary = dict(
        n_windows=int(len(valid)),
        n_oos_windows=int(oos_mask.sum()),
        baseline_full_sharpe=full_sh_b,
        candidate_full_sharpe=full_sh_c,
        baseline_wfe=wfe_b,
        candidate_wfe=wfe_c,
        baseline_mean_oos_sharpe=mean_sh_b_oos,
        candidate_mean_oos_sharpe=mean_sh_c_oos,
        candidate_ci95_lo_cagr=ci_lo_c,
        candidate_ci95_hi_cagr=ci_hi_c,
        baseline_ci95_lo_cagr=ci_lo_b,
        baseline_ci95_hi_cagr=ci_hi_b,
        candidate_ci95_lo_sharpe=ci_lo_sh,
        candidate_ci95_hi_sharpe=ci_hi_sh,
    )
    return df, summary


# ------------------------------------------------------------------
# Multi-metric block bootstrap (matches Session 4)
# ------------------------------------------------------------------

def run_bootstrap_multi(nav_b: pd.Series, nav_c: pd.Series, dates: pd.Series) -> dict:
    oos_mask = (dates >= OOS_START_TS) & (dates <= OOS_END_TS)
    n_oos = int(oos_mask.sum())
    years_oos = n_oos / TRADING_DAYS

    nav_b_pos = pd.Series(nav_b.values, index=range(len(nav_b)))
    nav_c_pos = pd.Series(nav_c.values, index=range(len(nav_c)))
    ret_b = nav_b_pos.pct_change().fillna(0).values[oos_mask.values]
    ret_c = nav_c_pos.pct_change().fillna(0).values[oos_mask.values]

    def _metrics(ret_series: np.ndarray) -> tuple[float, float, float]:
        cum = float(np.prod(1.0 + ret_series))
        cg = cum ** (1.0 / years_oos) - 1.0 if cum > 0 else -1.0
        sd = float(np.std(ret_series, ddof=1))
        sh = float(np.mean(ret_series) / sd * np.sqrt(252)) if sd > 1e-12 else float('nan')
        nav_path = np.cumprod(1.0 + ret_series)
        peak = np.maximum.accumulate(nav_path)
        dd = nav_path / np.where(peak > 0, peak, 1) - 1.0
        md = float(dd.min())
        return cg, sh, md

    cg_b_act, sh_b_act, md_b_act = _metrics(ret_b)
    cg_c_act, sh_c_act, md_c_act = _metrics(ret_c)

    rng = np.random.default_rng(RNG_SEED)
    n_blocks = int(np.ceil(n_oos / BLOCK_SIZE))

    cg_b_arr = np.empty(N_BOOTSTRAP); cg_c_arr = np.empty(N_BOOTSTRAP)
    sh_b_arr = np.empty(N_BOOTSTRAP); sh_c_arr = np.empty(N_BOOTSTRAP)
    md_b_arr = np.empty(N_BOOTSTRAP); md_c_arr = np.empty(N_BOOTSTRAP)

    for i in range(N_BOOTSTRAP):
        starts = rng.integers(0, n_oos - BLOCK_SIZE + 1, size=n_blocks)
        s_b = np.concatenate([ret_b[s:s + BLOCK_SIZE] for s in starts])[:n_oos]
        s_c = np.concatenate([ret_c[s:s + BLOCK_SIZE] for s in starts])[:n_oos]
        cg_b, sh_b, md_b = _metrics(s_b)
        cg_c, sh_c, md_c = _metrics(s_c)
        cg_b_arr[i] = cg_b; sh_b_arr[i] = sh_b; md_b_arr[i] = md_b
        cg_c_arr[i] = cg_c; sh_c_arr[i] = sh_c; md_c_arr[i] = md_c
        if (i + 1) % 2500 == 0:
            log(f'      bootstrap {i+1}/{N_BOOTSTRAP}')

    cg_diff = (cg_c_arr - cg_b_arr) * 100  # pp
    sh_diff = sh_c_arr - sh_b_arr
    md_diff = md_c_arr - md_b_arr

    def _stats(arr: np.ndarray, p_threshold: float = 0.0) -> dict:
        return dict(
            median=float(np.percentile(arr, 50)),
            p2_5=float(np.percentile(arr, 2.5)),
            p97_5=float(np.percentile(arr, 97.5)),
            p_better=float((arr > p_threshold).mean()),
        )

    return dict(
        n_oos_days=n_oos,
        years_oos=years_oos,
        actual_cagr_baseline_pct=cg_b_act * 100,
        actual_cagr_candidate_pct=cg_c_act * 100,
        actual_sharpe_baseline=sh_b_act,
        actual_sharpe_candidate=sh_c_act,
        actual_maxdd_baseline_pct=md_b_act * 100,
        actual_maxdd_candidate_pct=md_c_act * 100,
        cagr_diff=_stats(cg_diff),
        sharpe_diff=_stats(sh_diff),
        maxdd_diff=_stats(md_diff),
    )


# ------------------------------------------------------------------
# 9+1 metrics
# ------------------------------------------------------------------

def compute_9_plus_1(nav_b: pd.Series, nav_c: pd.Series, dates: pd.Series,
                     ret: pd.Series, wfa_sum: dict,
                     lev_b: np.ndarray, lev_c: np.ndarray) -> pd.DataFrame:
    nav_b_idx = pd.Series(nav_b.values, index=range(len(nav_b)))
    nav_c_idx = pd.Series(nav_c.values, index=range(len(nav_c)))
    ret_idx = pd.Series(ret.values, index=range(len(ret)))
    dates_idx = pd.Series(dates.values, index=range(len(dates)))

    m_b = metrics_from_nav(nav_b_idx, dates_idx, ret_idx)
    m_c = metrics_from_nav(nav_c_idx, dates_idx, ret_idx)

    tr_b = trades_per_yr_from_levraw(lev_b, dates) if lev_b is not None else float('nan')
    tr_c = trades_per_yr_from_levraw(lev_c, dates) if lev_c is not None else float('nan')

    wfe_b = wfa_sum['baseline_wfe']
    wfe_c = wfa_sum['candidate_wfe']
    ci_lo_b = wfa_sum['baseline_ci95_lo_cagr']
    ci_lo_c = wfa_sum['candidate_ci95_lo_cagr']

    rows = [
        dict(metric='CAGR_OOS_pct', baseline=m_b['CAGR_OOS'] * 100,
             candidate=m_c['CAGR_OOS'] * 100,
             diff=(m_c['CAGR_OOS'] - m_b['CAGR_OOS']) * 100,
             better_if='positive'),
        dict(metric='IS_OOS_gap_pp', baseline=m_b['IS_OOS_gap'] * 100,
             candidate=m_c['IS_OOS_gap'] * 100,
             diff=(m_c['IS_OOS_gap'] - m_b['IS_OOS_gap']) * 100,
             better_if='negative'),
        dict(metric='Sharpe_OOS', baseline=m_b['Sharpe_OOS'],
             candidate=m_c['Sharpe_OOS'],
             diff=m_c['Sharpe_OOS'] - m_b['Sharpe_OOS'],
             better_if='positive'),
        dict(metric='MaxDD_full_pct', baseline=m_b['MaxDD_FULL'] * 100,
             candidate=m_c['MaxDD_FULL'] * 100,
             diff=(m_c['MaxDD_FULL'] - m_b['MaxDD_FULL']) * 100,
             better_if='positive (less neg)'),
        dict(metric='Worst10Y_pct', baseline=m_b['Worst10Y_star'] * 100,
             candidate=m_c['Worst10Y_star'] * 100,
             diff=(m_c['Worst10Y_star'] - m_b['Worst10Y_star']) * 100,
             better_if='positive'),
        dict(metric='P10_5Y_pct', baseline=m_b['P10_5Y'] * 100,
             candidate=m_c['P10_5Y'] * 100,
             diff=(m_c['P10_5Y'] - m_b['P10_5Y']) * 100,
             better_if='positive'),
        dict(metric='Trades_per_yr', baseline=tr_b, candidate=tr_c,
             diff=tr_c - tr_b, better_if='lower / equal'),
        dict(metric='WFE_full', baseline=wfe_b, candidate=wfe_c,
             diff=wfe_c - wfe_b, better_if='>= 1.0'),
        dict(metric='CI95_lo_window_CAGR_pct',
             baseline=ci_lo_b * 100, candidate=ci_lo_c * 100,
             diff=(ci_lo_c - ci_lo_b) * 100, better_if='> 0'),
    ]

    def _improved(r):
        d = r['diff']
        if 'positive' in r['better_if']:
            return d > 0
        elif 'negative' in r['better_if']:
            return d < 0
        elif 'lower' in r['better_if']:
            return d <= 0
        elif '>=' in r['better_if']:
            return r['candidate'] >= r['baseline']
        elif '>' in r['better_if']:
            return r['candidate'] > r['baseline']
        return False

    n_imp = sum(int(_improved(r)) for r in rows)
    n_deg = sum(int((r['diff'] != 0) and (not _improved(r))) for r in rows)
    rows.append(dict(
        metric='+1 composite (n_imp / n_deg)',
        baseline='-',
        candidate=f'{n_imp} / {n_deg}',
        diff='-',
        better_if='n_deg = 0',
    ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Per-target runner
# ------------------------------------------------------------------

def run_target(target: dict, sig_q_lagged: pd.Series) -> dict:
    tid = target['id']
    log('\n' + '=' * 78)
    log(f'TRANSFER TARGET: {tid} ({target["label"]})')
    log('=' * 78)

    # Step 0: Load baseline + cache info
    log(f'  [0/5] Loading baseline cache: {target["cache"]} ...')
    nav_b, lev_b, dates, ret = load_baseline_nav(target)
    log(f'        baseline NAV: {len(nav_b)} obs, {nav_b.index[0].date()} -> {nav_b.index[-1].date()}')

    # Step 1: Build candidate NAV (M6 defensive)
    log('  [1/5] Building candidate NAV (M6 defensive)...')
    nav_c, lev_c, _ = build_cfd_native(target, sig_q_lagged, 'M6', 'defensive')
    log(f'        candidate NAV: {len(nav_c)} obs')

    # Align by intersection (both should be same calendar)
    nav_c = nav_c.reindex(nav_b.index, method='ffill')

    # Step 2: WFA
    log('  [2/5] WFA 50 windows ...')
    df_wfa, sum_wfa = run_wfa(nav_b, nav_c, dates)
    wfa_csv = OUT_DIR / f'session5_phase_d_wfa_{tid}_20260605.csv'
    df_wfa.to_csv(wfa_csv, index=False, float_format='%.6f')
    log(f'        -> {wfa_csv.name}')
    log(f'        WFE base={sum_wfa["baseline_wfe"]:.3f}  cand={sum_wfa["candidate_wfe"]:.3f}')
    log(f'        CI95_lo cand CAGR = {sum_wfa["candidate_ci95_lo_cagr"]*100:+.2f}%')

    # Step 3: Bootstrap
    log('  [3/5] Block bootstrap 10,000 (multi-metric) ...')
    boot = run_bootstrap_multi(nav_b, nav_c, dates)
    boot_rows = [
        dict(metric='CAGR_pp',
             actual_baseline=boot['actual_cagr_baseline_pct'],
             actual_candidate=boot['actual_cagr_candidate_pct'],
             actual_diff=boot['actual_cagr_candidate_pct'] - boot['actual_cagr_baseline_pct'],
             boot_median_diff=boot['cagr_diff']['median'],
             boot_ci95_lo_diff=boot['cagr_diff']['p2_5'],
             boot_ci95_hi_diff=boot['cagr_diff']['p97_5'],
             P_cand_better=boot['cagr_diff']['p_better']),
        dict(metric='Sharpe',
             actual_baseline=boot['actual_sharpe_baseline'],
             actual_candidate=boot['actual_sharpe_candidate'],
             actual_diff=boot['actual_sharpe_candidate'] - boot['actual_sharpe_baseline'],
             boot_median_diff=boot['sharpe_diff']['median'],
             boot_ci95_lo_diff=boot['sharpe_diff']['p2_5'],
             boot_ci95_hi_diff=boot['sharpe_diff']['p97_5'],
             P_cand_better=boot['sharpe_diff']['p_better']),
        dict(metric='MaxDD_pp(better=less_neg)',
             actual_baseline=boot['actual_maxdd_baseline_pct'],
             actual_candidate=boot['actual_maxdd_candidate_pct'],
             actual_diff=boot['actual_maxdd_candidate_pct'] - boot['actual_maxdd_baseline_pct'],
             boot_median_diff=boot['maxdd_diff']['median'] * 100,
             boot_ci95_lo_diff=boot['maxdd_diff']['p2_5'] * 100,
             boot_ci95_hi_diff=boot['maxdd_diff']['p97_5'] * 100,
             P_cand_better=boot['maxdd_diff']['p_better']),
    ]
    df_boot = pd.DataFrame(boot_rows)
    boot_csv = OUT_DIR / f'session5_phase_d_bootstrap_{tid}_20260605.csv'
    df_boot.to_csv(boot_csv, index=False, float_format='%.6f')
    log(f'        -> {boot_csv.name}')
    log(f'        P(CAGR>base)={boot["cagr_diff"]["p_better"]:.3f}  '
        f'P(Sharpe>base)={boot["sharpe_diff"]["p_better"]:.3f}  '
        f'P(MaxDD better)={boot["maxdd_diff"]["p_better"]:.3f}')

    # Step 4: 9+1
    log('  [4/5] 9+1 metrics ...')
    df_metrics = compute_9_plus_1(nav_b, nav_c, dates, ret, sum_wfa, lev_b, lev_c)

    # Step 5: Verdict
    p_cagr = boot['cagr_diff']['p_better']
    p_sh = boot['sharpe_diff']['p_better']
    p_md = boot['maxdd_diff']['p_better']
    wfe_ok = sum_wfa['candidate_wfe'] >= 1.0
    ci_ok = sum_wfa['candidate_ci95_lo_cagr'] > 0
    any_boot_pass = max(p_cagr, p_sh, p_md) > 0.90

    if wfe_ok and ci_ok and any_boot_pass:
        verdict = 'ADOPT'
    elif (wfe_ok or ci_ok) and any_boot_pass:
        verdict = 'NEEDS_FURTHER_WORK'
    elif max(p_cagr, p_sh, p_md) > 0.80:
        verdict = 'NEEDS_FURTHER_WORK'
    else:
        verdict = 'REJECT'

    log(f'  [5/5] Verdict: {verdict}')

    # Write per-target markdown
    md_path = OUT_DIR / f'session5_phase_d_audit_{tid}_20260605.md'
    write_target_md(md_path, tid, target, sum_wfa, boot, df_metrics, verdict)
    log(f'        -> {md_path.name}')

    return dict(
        id=tid, target=target,
        wfa=sum_wfa, boot=boot, metrics=df_metrics,
        p_cagr=p_cagr, p_sharpe=p_sh, p_maxdd=p_md,
        wfe_c=sum_wfa['candidate_wfe'],
        ci95_lo_c=sum_wfa['candidate_ci95_lo_cagr'],
        verdict=verdict,
    )


def write_target_md(path: Path, tid: str, target: dict, wfa: dict, boot: dict,
                    df_metrics: pd.DataFrame, verdict: str) -> None:
    p_cagr = boot['cagr_diff']['p_better']
    p_sh = boot['sharpe_diff']['p_better']
    p_md = boot['maxdd_diff']['p_better']

    lines = []
    lines.append(f'# Session 5 Transfer Audit — nasdaq_mom63 x M6 defensive on {tid}')
    lines.append('')
    lines.append('作成日: 2026-06-05')
    lines.append('最終更新日: 2026-06-05')
    lines.append('')
    lines.append('## Target Baseline')
    lines.append(f'- ID: `{tid}`')
    lines.append(f'- Label: {target["label"]}')
    lines.append(f'- Cache: `audit_results/_cache/{target["cache"]}`')
    lines.append('')
    lines.append('## Overlay (transferred from Session 4 ADOPT)')
    lines.append('- Signal: `nasdaq_mom63` (NASDAQ 63-day momentum, daily lag)')
    lines.append('- Method: M6 (threshold-proxy continuous tilt)')
    lines.append('- Direction: defensive (high momentum -> reduce leverage)')
    lines.append('- Multiplier: signal_q {0,1,2,3} -> {1.1, 1.0, 0.9, 0.8}')
    lines.append('')
    lines.append('## 9+1 Metrics (Native Audit)')
    lines.append('')
    lines.append('| metric | baseline | candidate | diff | better if |')
    lines.append('|---|---|---|---|---|')
    for _, r in df_metrics.iterrows():
        b = r['baseline']; c = r['candidate']; d = r['diff']
        if isinstance(b, (int, float)) and not pd.isna(b):
            b = f'{b:+.4f}'
        if isinstance(c, (int, float)) and not pd.isna(c):
            c = f'{c:+.4f}'
        if isinstance(d, (int, float)) and not pd.isna(d):
            d = f'{d:+.4f}'
        lines.append(f'| {r["metric"]} | {b} | {c} | {d} | {r["better_if"]} |')
    lines.append('')
    lines.append('## WFA 50 windows')
    lines.append(f'- baseline full Sharpe : {wfa["baseline_full_sharpe"]:+.3f}')
    lines.append(f'- candidate full Sharpe : {wfa["candidate_full_sharpe"]:+.3f}')
    lines.append(f'- baseline WFE          : {wfa["baseline_wfe"]:.3f}')
    lines.append(f'- candidate WFE         : {wfa["candidate_wfe"]:.3f}  (PASS >= 1.0: {"YES" if wfa["candidate_wfe"]>=1.0 else "NO"})')
    lines.append(f'- baseline mean OOS Sharpe : {wfa["baseline_mean_oos_sharpe"]:+.3f}')
    lines.append(f'- candidate mean OOS Sharpe : {wfa["candidate_mean_oos_sharpe"]:+.3f}')
    lines.append(f'- candidate CI95 CAGR   : [{wfa["candidate_ci95_lo_cagr"]*100:+.2f}%, {wfa["candidate_ci95_hi_cagr"]*100:+.2f}%]  (PASS > 0: {"YES" if wfa["candidate_ci95_lo_cagr"]>0 else "NO"})')
    lines.append('')
    lines.append('## Block Bootstrap 10,000 — Multi-Metric')
    lines.append('')
    lines.append('| metric | actual diff | boot median | 95% CI of diff | P(cand better) |')
    lines.append('|---|---|---|---|---|')
    cg_act = boot['actual_cagr_candidate_pct'] - boot['actual_cagr_baseline_pct']
    sh_act = boot['actual_sharpe_candidate'] - boot['actual_sharpe_baseline']
    md_act = boot['actual_maxdd_candidate_pct'] - boot['actual_maxdd_baseline_pct']
    lines.append(f'| CAGR (pp) | {cg_act:+.3f} | {boot["cagr_diff"]["median"]:+.3f} | '
                 f'[{boot["cagr_diff"]["p2_5"]:+.3f}, {boot["cagr_diff"]["p97_5"]:+.3f}] | {p_cagr:.4f} |')
    lines.append(f'| Sharpe | {sh_act:+.4f} | {boot["sharpe_diff"]["median"]:+.4f} | '
                 f'[{boot["sharpe_diff"]["p2_5"]:+.4f}, {boot["sharpe_diff"]["p97_5"]:+.4f}] | {p_sh:.4f} |')
    lines.append(f'| MaxDD (pp, +=better) | {md_act:+.3f} | {boot["maxdd_diff"]["median"]*100:+.3f} | '
                 f'[{boot["maxdd_diff"]["p2_5"]*100:+.3f}, {boot["maxdd_diff"]["p97_5"]*100:+.3f}] | {p_md:.4f} |')
    lines.append('')
    lines.append(f'OOS days = {boot["n_oos_days"]}, years = {boot["years_oos"]:.2f}')
    lines.append('')
    lines.append('## Verdict')
    lines.append('')
    lines.append(f'- WFA WFE >= 1.0      : {"PASS" if wfa["candidate_wfe"]>=1.0 else "FAIL"}  (actual = {wfa["candidate_wfe"]:.3f})')
    lines.append(f'- WFA CI95_lo > 0    : {"PASS" if wfa["candidate_ci95_lo_cagr"]>0 else "FAIL"}  (actual = {wfa["candidate_ci95_lo_cagr"]*100:+.2f}%)')
    lines.append(f'- Bootstrap P > 0.90 : {"PASS" if max(p_cagr,p_sh,p_md)>0.90 else "FAIL"}  (max P = {max(p_cagr,p_sh,p_md):.4f})')
    lines.append('')
    lines.append(f'**Final: {verdict}**')
    lines.append('')

    path.write_text('\n'.join(lines), encoding='utf-8')


# ------------------------------------------------------------------
# Combined session5 report
# ------------------------------------------------------------------

def write_summary_report(results: list[dict], out_path: Path) -> None:
    lines = []
    lines.append('# Session 5 — nasdaq_mom63 x M6 defensive 転用 audit (S2, E4)')
    lines.append('')
    lines.append('作成日: 2026-06-05')
    lines.append('最終更新日: 2026-06-05')
    lines.append('')
    lines.append('## 検証対象')
    lines.append('- S3 (DH-W1, ETF only) -> **既に ADOPT 確定** (Session 4)')
    lines.append('- **S2 (D5 = vz=0.65, lmax=5)** -> 本セッション転用テスト')
    lines.append('- **E4 (現行 §1 Active = S2_VZGated + LT2-N750 + E4 Regime k_lt)** -> 本セッション転用テスト')
    lines.append('')
    lines.append('Overlay: signal = `nasdaq_mom63`, method = M6, direction = defensive, '
                 'mapping = signal_q {0,1,2,3} -> {1.1, 1.0, 0.9, 0.8}')
    lines.append('')
    lines.append('## 転用結果サマリ')
    lines.append('')
    lines.append('| baseline | WFE | CI95_lo CAGR | P_CAGR | P_Sharpe | P_MaxDD better | Verdict |')
    lines.append('|---|---:|---:|---:|---:|---:|---|')
    lines.append('| S3 DH-W1 (Session 4 ADOPT) | 1.005 | +13.00% | 0.295 | **0.930** | **0.988** | **ADOPT** |')
    for r in results:
        wfa = r['wfa']; boot = r['boot']
        lines.append(f'| {r["id"]} | {wfa["candidate_wfe"]:.3f} | '
                     f'{wfa["candidate_ci95_lo_cagr"]*100:+.2f}% | '
                     f'{r["p_cagr"]:.3f} | {r["p_sharpe"]:.3f} | '
                     f'{r["p_maxdd"]:.3f} | **{r["verdict"]}** |')
    lines.append('')

    # 9+1 baseline-vs-candidate detail per target
    lines.append('## 各転用先の 9+1 詳細')
    lines.append('')
    for r in results:
        lines.append(f'### {r["id"]} ({r["target"]["label"]})')
        lines.append('')
        lines.append('| metric | baseline | candidate | diff |')
        lines.append('|---|---:|---:|---:|')
        for _, row in r['metrics'].iterrows():
            b = row['baseline']; c = row['candidate']; d = row['diff']
            if isinstance(b, (int, float)) and not pd.isna(b):
                b = f'{b:+.4f}'
            if isinstance(c, (int, float)) and not pd.isna(c):
                c = f'{c:+.4f}'
            if isinstance(d, (int, float)) and not pd.isna(d):
                d = f'{d:+.4f}'
            lines.append(f'| {row["metric"]} | {b} | {c} | {d} |')
        lines.append('')

    lines.append('## 解釈')
    lines.append('')

    # Auto-interpretation
    s2_res = next((r for r in results if r['id'] == 'S2_D5'), None)
    e4_res = next((r for r in results if r['id'] == 'E4_Active'), None)

    if s2_res:
        lines.append(f'### S2 (D5) 転用結果: **{s2_res["verdict"]}**')
        lines.append(f'- WFE = {s2_res["wfe_c"]:.3f}, CI95_lo = {s2_res["ci95_lo_c"]*100:+.2f}%')
        lines.append(f'- Bootstrap: P(CAGR)={s2_res["p_cagr"]:.3f}, '
                     f'P(Sharpe)={s2_res["p_sharpe"]:.3f}, P(MaxDD better)={s2_res["p_maxdd"]:.3f}')
        lines.append('')
    if e4_res:
        lines.append(f'### E4 (現行 Active) 転用結果: **{e4_res["verdict"]}**')
        lines.append(f'- WFE = {e4_res["wfe_c"]:.3f}, CI95_lo = {e4_res["ci95_lo_c"]*100:+.2f}%')
        lines.append(f'- Bootstrap: P(CAGR)={e4_res["p_cagr"]:.3f}, '
                     f'P(Sharpe)={e4_res["p_sharpe"]:.3f}, P(MaxDD better)={e4_res["p_maxdd"]:.3f}')
        lines.append('')

    lines.append('### 戦略基盤特異性 (Cross-Strategy)')
    n_adopt = sum(1 for r in results if r['verdict'] == 'ADOPT')
    if n_adopt == len(results):
        lines.append('全 baseline で ADOPT -> overlay は **汎用 (strategy-agnostic)** と判定。')
    elif n_adopt == 0:
        lines.append('S2/E4 で REJECT または NEEDS_FURTHER_WORK -> overlay は '
                     '**S3 (DH-W1) 特異** と判定。CFD ベース戦略には転用不可。')
    else:
        lines.append(f'{n_adopt}/{len(results)} で ADOPT -> overlay は **部分的に汎用**。'
                     '採用可能な戦略基盤を限定して記録する。')
    lines.append('')

    lines.append('## 推奨アクション')
    lines.append('')
    if all(r['verdict'] == 'REJECT' for r in results):
        lines.append('- S3 (DH-W1 ETF only) 限定 ADOPT を維持。S2/E4 への適用は推奨しない。')
        lines.append('- CURRENT_BEST_STRATEGY.md の Overlay Candidate セクションに「S3 限定」を明示。')
    elif any(r['verdict'] == 'ADOPT' for r in results):
        adopted_ids = [r['id'] for r in results if r['verdict'] == 'ADOPT']
        lines.append(f'- {", ".join(adopted_ids)} で ADOPT -> CURRENT_BEST_STRATEGY.md に追記対象。')
        lines.append('- 現行 §1 Active との両立 (置換 / 上乗せ) はユーザー判断を要する。')
    else:
        lines.append('- いずれも NEEDS_FURTHER_WORK -> 別 method/direction の組合せ探索 or '
                     'ハードゲート緩和の議論。')
        lines.append('- 暫定的に S3 限定 ADOPT を維持。')
    lines.append('')

    out_path.write_text('\n'.join(lines), encoding='utf-8')


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    log('=' * 80)
    log('Session 5 B — Transfer audit: nasdaq_mom63 x M6 defensive on S2 + E4')
    log('=' * 80)
    log(f'Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # Load signal once
    log('\n[Load signal] nasdaq_mom63 ...')
    sig_raw = load_signal('nasdaq_mom63')
    log(f'  raw obs: {len(sig_raw)}, range {sig_raw.index[0].date()} -> {sig_raw.index[-1].date()}')

    sig_q = quantile_cut(sig_raw, levels=4)
    sig_q = sig_q.dropna().astype('int8')
    sig_lagged = apply_publication_lag(sig_q, 'daily')
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep='last')]
    log(f'  quantized + lagged obs: {len(sig_lagged)}')

    # Run each target
    results = []
    for target in TARGETS:
        try:
            res = run_target(target, sig_lagged)
            results.append(res)
        except Exception as e:
            import traceback
            log(f'\n[FAIL] {target["id"]}: {e}')
            traceback.print_exc()

    # Combined summary CSV
    log('\n' + '=' * 78)
    log('SUMMARY')
    log('=' * 78)

    if not results:
        log('No successful targets — abort.')
        return

    summary_rows = []
    for r in results:
        wfa = r['wfa']; boot = r['boot']
        m = r['metrics']
        # Get key 9+1 diffs
        def _diff_of(metric_name):
            row = m[m['metric'] == metric_name]
            if row.empty:
                return float('nan')
            return float(row.iloc[0]['diff'])

        summary_rows.append(dict(
            baseline=r['id'],
            signal='nasdaq_mom63',
            method='M6',
            direction='defensive',
            cagr_oos_diff_pp=_diff_of('CAGR_OOS_pct'),
            sharpe_diff=_diff_of('Sharpe_OOS'),
            maxdd_diff_pp=_diff_of('MaxDD_full_pct'),
            worst10y_diff_pp=_diff_of('Worst10Y_pct'),
            p10_5y_diff_pp=_diff_of('P10_5Y_pct'),
            trades_yr_diff=_diff_of('Trades_per_yr'),
            wfe_candidate=wfa['candidate_wfe'],
            wfe_baseline=wfa['baseline_wfe'],
            ci95_lo_candidate_cagr_pct=wfa['candidate_ci95_lo_cagr'] * 100,
            ci95_lo_baseline_cagr_pct=wfa['baseline_ci95_lo_cagr'] * 100,
            p_cagr_better=r['p_cagr'],
            p_sharpe_better=r['p_sharpe'],
            p_maxdd_better=r['p_maxdd'],
            verdict=r['verdict'],
        ))

    df_sum = pd.DataFrame(summary_rows)
    sum_csv = OUT_DIR / 'session5_transfer_audit_20260605.csv'
    df_sum.to_csv(sum_csv, index=False, float_format='%.6f')
    log(f'-> {sum_csv}')

    # Combined report
    report_md = OUT_DIR / 'session5_transfer_report_20260605.md'
    write_summary_report(results, report_md)
    log(f'-> {report_md}')

    # Final stdout summary
    log('\n=== Final ===')
    for r in results:
        log(f'  {r["id"]:<12s}  WFE={r["wfe_c"]:.3f}  CI95_lo={r["ci95_lo_c"]*100:+.2f}%  '
            f'P(CAGR)={r["p_cagr"]:.3f}  P(Sharpe)={r["p_sharpe"]:.3f}  '
            f'P(MaxDD)={r["p_maxdd"]:.3f}  -> {r["verdict"]}')
    log(f'\nDone: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
