"""
g13_realistic_cost_full_comparison.py
======================================
21戦略を SBI CFD 実績スプレッド（3.0%/yr）で全再シミュレーション

目的:
  現行 STRATEGY_PERFORMANCE_COMPARISON は cfd_spread=0.20%/yr（くりっく株365想定）で
  計算済み。NASDAQ 対応 CFD は SBI CFD NQ100 のみで実際スプレッドは 3.0%/yr。
  全 21 戦略の現実的なパフォーマンスを定量化して意思決定の根拠とする。

WFA 欄（CI95_lo / Overfit(WFE)）は既存 WFA CSV から読み込み（旧スプレッドでの近似値）。

出力:
  - g13_sbi_cfd_full_results.csv
  - STRATEGY_PERFORMANCE_COMPARISON_SBI_CFD_2026-05-28.md
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr, build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_a2_signal, simulate_rebalance_A,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import build_nav_strategy, calc_7metrics
from compute_cfd_worst10y import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import build_lt_signal, apply_lt_mode_b
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from a2_dyn_lmax import (
    compute_L_s2_dyn_lmax, signal_to_bias_dynamic,
    compute_p10_5y, calc_all_metrics,
    S2_BASE, K_LO, K_HI, K_MID,
)
from _sweep_format import MD_HEADER_STRAT, fmt_row_strat, MD_METRIC_GLOSSARY

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# コスト定数
# ---------------------------------------------------------------------------
SBI_CFD_SPREAD = 0.0300   # 3.0%/yr（SBI CFD NQ100 業者マージン実績）

# ---------------------------------------------------------------------------
# 共通定数
# ---------------------------------------------------------------------------
LT_N = 750
S2_FIXED_7 = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)
S2_FIXED_5 = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=5.0, step=0.5)
K_MID_E4 = K_MID  # 0.50

# A2/A2B 動的 l_max パラメータ
VOL_REF_A2 = 0.20
L_FLOOR_A2 = 4.5
L_CEIL_A2  = 6.5

# F10 / C2 tilt パラメータ
TILT_F   = 10.0
VZ_REG_F = 0.7
CAP_CALM = 0.15
CAP_BULL = 0.10
CAP_BEAR = 0.05


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def _k_lt_regime(vz_arr, k_lo, k_hi, vz_thr):
    return np.where(vz_arr > +vz_thr, k_hi,
           np.where(vz_arr < -vz_thr, k_lo, K_MID_E4))


def _k_lt_sigmoid(vz_arr, alpha, k_lo=K_LO, k_hi=K_HI):
    x = np.clip(alpha * vz_arr, -500.0, 500.0)
    return k_lo + (k_hi - k_lo) / (1.0 + np.exp(-x))


def _build_lev_mod(lev_raw, lt_sig_raw, vz, k_lo, k_hi, vz_thr):
    k_dyn = _k_lt_regime(np.asarray(vz), k_lo, k_hi, vz_thr)
    lt_bias = pd.Series(signal_to_bias_dynamic(lt_sig_raw.values, k_dyn),
                        index=lt_sig_raw.index)
    return apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)


def _build_lev_mod_sigmoid(lev_raw, lt_sig_raw, vz, alpha):
    k_dyn = _k_lt_sigmoid(np.asarray(vz), alpha)
    lt_bias = pd.Series(signal_to_bias_dynamic(lt_sig_raw.values, k_dyn),
                        index=lt_sig_raw.index)
    return apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)


def _tilt_deadband(raw_a2_vals, vz_vals, bull_mask, eps_or_arr):
    n = len(raw_a2_vals)
    cap_eff = np.where(np.abs(vz_vals) < VZ_REG_F, CAP_CALM,
              np.where(vz_vals > VZ_REG_F, CAP_BULL, CAP_BEAR))
    tilt_raw    = TILT_F * (raw_a2_vals - THRESHOLD) * (1.0 - raw_a2_vals)
    tilt_target = np.clip(tilt_raw, 0.0, cap_eff)
    tilt_target = np.where(bull_mask, tilt_target, 0.0)
    eps_arr = (np.full(n, float(eps_or_arr)) if np.isscalar(eps_or_arr)
               else np.asarray(eps_or_arr, float))
    confirmed = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if i == 0 or abs(tilt_target[i] - cur) >= eps_arr[i]:
            cur = tilt_target[i]
        confirmed[i] = cur
    return confirmed


def _compute_vov_zscore(ret, win_short=20, win_long=60, win_norm=252):
    vol_short = ret.rolling(win_short, min_periods=win_short // 2).std() * np.sqrt(TRADING_DAYS)
    vov       = vol_short.rolling(win_long, min_periods=win_long // 2).std()
    vov_mean  = vov.expanding(min_periods=win_norm).mean()
    vov_std   = vov.expanding(min_periods=win_norm).std().replace(0, np.nan)
    return ((vov - vov_mean) / vov_std).fillna(0.0)


def _yang_zhang_vol(ohlc_df, n=20):
    O  = np.log(ohlc_df['Open']  / ohlc_df['Close'].shift(1))
    lh = np.log(ohlc_df['High']  / ohlc_df['Open'])
    ll = np.log(ohlc_df['Low']   / ohlc_df['Open'])
    lc = np.log(ohlc_df['Close'] / ohlc_df['Open'])
    sigma2_o  = (O**2).rolling(n, min_periods=max(5, n // 2)).mean()
    sigma2_c  = (lc**2).rolling(n, min_periods=max(5, n // 2)).mean()
    rs        = lh * (lh - lc) + ll * (ll - lc)
    sigma2_rs = rs.rolling(n, min_periods=max(5, n // 2)).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt((sigma2_o + k * sigma2_c + (1 - k) * sigma2_rs).clip(lower=1e-8) * TRADING_DAYS)


def _nav(close, lev_mod, wn, wg, wb, dates, gold_2x, bond_3x, sofr, L_s2_vals):
    return build_nav_strategy(
        close, lev_mod, wn, wg, wb, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2_vals, cfd_spread=SBI_CFD_SPREAD,
    )


def _metrics(nav, dates, trades_yr, ci95=None, wfe=None):
    m = calc_all_metrics(nav, dates, trades_yr)
    m['WFA_CI95_lo'] = ci95
    m['WFA_WFE']     = wfe
    m['Trades_yr']   = trades_yr
    return m


def _load_wfa(csv_name, strat_col='strategy'):
    path = os.path.join(BASE, csv_name)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        out[str(row[strat_col])] = (float(row['WFA_CI95_lo']), float(row['WFA_WFE']))
    return out


def _csv_val(csv_name, col, **filters):
    path = os.path.join(BASE, csv_name)
    if not os.path.exists(path):
        return float('nan')
    df = pd.read_csv(path)
    mask = pd.Series([True] * len(df), index=df.index)
    for c, v in filters.items():
        if isinstance(v, float):
            mask &= (df[c] - v).abs() < 1e-5
        else:
            mask &= df[c].astype(str) == str(v)
    rows = df[mask]
    return float(rows.iloc[0][col]) if not rows.empty else float('nan')


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 75)
    print('G13: 21戦略 全 SBI CFD 実績コスト（3.0%/yr）再シミュレーション')
    print('=' * 75)

    # ── 共通データロード ──
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({n:,}日 / {n_years:.1f}年)')

    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_trades_yr = n_tr / n_years
    print(f'Trades/yr (rebalance): {n_trades_yr:.1f}')

    lt_sig_raw  = build_lt_signal(close, 'LT2', LT_N)
    vz_arr      = vz.values
    raw_a2_arr  = raw_a2.values

    # 共通 L_s2
    L_s2_7 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED_7)   # l_max=7.0
    L_s2_5 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED_5)   # l_max=5.0

    # vol252（A2/A2B 用）
    vol252 = ret.rolling(252, min_periods=126).std() * np.sqrt(TRADING_DAYS)
    vol252 = vol252.fillna(ret.expanding(min_periods=30).std() * np.sqrt(TRADING_DAYS))

    # WFA 読み込み
    wfa_g3  = _load_wfa('g3_wfa_e4_summary.csv')
    wfa_g7  = _load_wfa('g7_wfa_f10_summary.csv')
    wfa_g8  = _load_wfa('g8_wfa_lmax5_summary.csv')
    wfa_g10 = _load_wfa('g10_wfa_vz065_lmax_row_summary.csv')
    wfa_g11 = _load_wfa('g11_wfa_all_remaining_summary.csv')

    results = []

    def _add(label, nav, wfa_key, wfa_dict, trades_yr=None):
        tyr = trades_yr if trades_yr is not None else n_trades_yr
        ci95, wfe = wfa_dict.get(wfa_key, (None, None))
        m = _metrics(nav, dates, tyr, ci95=ci95, wfe=wfe)
        m['strategy'] = label
        results.append(m)
        print(f'  {label[:60]:60s} OOS={m["CAGR_OOS"]*100:+.1f}%  '
              f'Sh={m["Sharpe_OOS"]:+.3f}  MaxDD={m["MaxDD_FULL"]*100:+.1f}%')

    # ── F10 / C2 Trades/yr（既存CSVから）──
    f10_trades = _csv_val('f10_epsilon_deadband_results.csv', 'Trades_yr', eps=0.015)
    if np.isnan(f10_trades):
        f10_trades = 52.0
    c2_trades = _csv_val('c2_adaptive_deadband_results.csv', 'Trades_yr',
                         eps_0=0.020, mode='adaptive')
    if np.isnan(c2_trades):
        c2_trades = 51.0
    f10l5_trades = _csv_val('g8_wfa_lmax5_summary.csv', 'mean_Trades_yr',
                            strategy='F10-eps015-lmax5')
    if np.isnan(f10l5_trades):
        f10l5_trades = 52.0

    # ====================================================================
    # Block 1: E4 + Active 候補（F10系 / D5 vz065 lmax5）
    # ====================================================================
    print('\n■ Block 1: E4 / F10 系 / D5-Active')

    # 01 E4
    lev_e4 = _build_lev_mod(lev_raw, lt_sig_raw, vz, k_lo=K_LO, k_hi=K_HI, vz_thr=0.7)
    nav_e4 = _nav(close, lev_e4, wn_A, wg_A, wb_A, dates, gold_2x, bond_3x, sofr, L_s2_7.values)
    _add('E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7) ◆', nav_e4, 'E4-RegimeKLT', wfa_g3)

    # 02 F10 ε=0.015
    bull_mask = raw_a2_arr > THRESHOLD
    tilt_f10  = _tilt_deadband(raw_a2_arr, vz_arr, bull_mask, 0.015)
    wn_f10 = wn_A + tilt_f10
    wb_f10 = np.clip(wb_A - tilt_f10, 0.0, wb_A)
    nav_f10 = _nav(close, lev_e4, wn_f10, wg_A, wb_f10, dates, gold_2x, bond_3x, sofr, L_s2_7.values)
    _add('[Active候補] F10 ε=0.015 ✅⚠', nav_f10, 'F10-eps015', wfa_g7, trades_yr=f10_trades)

    # 03 F10+lmax5
    nav_f10l5 = _nav(close, lev_e4, wn_f10, wg_A, wb_f10, dates, gold_2x, bond_3x, sofr, L_s2_5.values)
    _add('[Active候補] F10+lmax5 ✅⚠', nav_f10l5, 'F10-eps015-lmax5', wfa_g8, trades_yr=f10l5_trades)

    # 04 D5 vz=0.65/lmax=5.0
    lev_d5_65 = _build_lev_mod(lev_raw, lt_sig_raw, vz, k_lo=K_LO, k_hi=K_HI, vz_thr=0.65)
    L_d5_65_5 = compute_L_s2_dyn_lmax(ret, vz, pd.Series(5.0, index=ret.index), **S2_BASE)
    nav_d5_65_5 = _nav(close, lev_d5_65, wn_A, wg_A, wb_A, dates, gold_2x, bond_3x, sofr, L_d5_65_5.values)
    _add('[Active候補] D5 vz=0.65/lmax=5.0 ✅⚠', nav_d5_65_5, 'vz065-lmax5', wfa_g10)

    # ====================================================================
    # Block 2: B4
    # ====================================================================
    print('\n■ Block 2: B4')
    lev_b4 = _build_lev_mod(lev_raw, lt_sig_raw, vz, k_lo=0.0, k_hi=0.7, vz_thr=0.7)
    nav_b4 = _nav(close, lev_b4, wn_A, wg_A, wb_A, dates, gold_2x, bond_3x, sofr, L_s2_7.values)
    _add('[B] B4 k_lo=0/k_hi=0.7/vz=0.7', nav_b4, 'B4-klo0', wfa_g11)

    # ====================================================================
    # Block 3: A1 (sigmoid k_lt)
    # ====================================================================
    print('\n■ Block 3: A1 (sigmoid k_lt)')
    for alpha in [2.0, 3.0, 5.0, 8.0]:
        lev_a1 = _build_lev_mod_sigmoid(lev_raw, lt_sig_raw, vz, alpha)
        nav_a1 = _nav(close, lev_a1, wn_A, wg_A, wb_A, dates, gold_2x, bond_3x, sofr, L_s2_7.values)
        _add(f'[A] A1 α={int(alpha)} (soft regime)', nav_a1, f'A1-alpha{int(alpha)}', wfa_g11)

    # ====================================================================
    # Block 4: A2 (dynamic l_max, vol-linked)
    # ====================================================================
    print('\n■ Block 4: A2 (動的 l_max)')
    lmax_base_a2 = 6.0; vol_sens_a2 = 2.0
    l_max_a2 = (lmax_base_a2 - vol_sens_a2 * (vol252 / VOL_REF_A2 - 1)).clip(L_FLOOR_A2, L_CEIL_A2)
    l_max_a2 = l_max_a2.reindex(ret.index).fillna(lmax_base_a2)
    L_s2_a2 = compute_L_s2_dyn_lmax(ret, vz, l_max_a2, **S2_BASE)
    lev_a2  = _build_lev_mod(lev_raw, lt_sig_raw, vz, k_lo=K_LO, k_hi=K_HI, vz_thr=0.7)
    nav_a2  = _nav(close, lev_a2, wn_A, wg_A, wb_A, dates, gold_2x, bond_3x, sofr, L_s2_a2.values)
    _add('[A] A2 lmax_base=6/vol_sens=2', nav_a2, 'A2-lmax6vs2', wfa_g11)

    # ====================================================================
    # Block 5: A2B (rolling VOL_REF)
    # ====================================================================
    print('\n■ Block 5: A2B (rolling VOL_REF)')
    vol_ref_t = vol252.rolling(2520, min_periods=504).median()
    vol_ref_t = vol_ref_t.fillna(vol252.expanding(min_periods=5).median()).fillna(vol252).clip(lower=0.05)
    l_max_a2b = (lmax_base_a2 - vol_sens_a2 * (vol252 / vol_ref_t - 1)).clip(L_FLOOR_A2, L_CEIL_A2)
    L_s2_a2b  = compute_L_s2_dyn_lmax(ret, vz, l_max_a2b, **S2_BASE)
    nav_a2b   = _nav(close, lev_a2, wn_A, wg_A, wb_A, dates, gold_2x, bond_3x, sofr, L_s2_a2b.values)
    _add('[A] A2B rolling VOL_REF', nav_a2b, 'A2B-rolling', wfa_g11)

    # ====================================================================
    # Block 6: A3 (VoV dual gate → weight tilt)
    # ====================================================================
    print('\n■ Block 6: A3 (VoV dual gate)')
    vov_z     = _compute_vov_zscore(ret)
    vov_thr_a3 = 1.3; alpha_max_a3 = 0.2
    stress    = (vov_z.values > vov_thr_a3) & (vz_arr > VZ_REG_F)
    alpha_a3  = np.where(stress, alpha_max_a3, 0.0)
    wn_a3     = wn_A * (1.0 - alpha_a3)
    wg_a3     = wg_A + wn_A * alpha_a3 * 0.5
    wb_a3     = wb_A + wn_A * alpha_a3 * 0.5
    L_s2_a3   = compute_L_s2_dyn_lmax(ret, vz, pd.Series(7.0, index=ret.index), **S2_BASE)
    nav_a3    = _nav(close, lev_e4, wn_a3, wg_a3, wb_a3, dates, gold_2x, bond_3x, sofr, L_s2_a3.values)
    _add('[A] A3 VoV dual gate (vov=1.3/α=0.2)', nav_a3, 'A3-vov', wfa_g11)

    # ====================================================================
    # Block 7: C2 (adaptive ε deadband tilt)
    # ====================================================================
    print('\n■ Block 7: C2 (adaptive deadband)')
    vol_20d   = ret.rolling(20).std() * np.sqrt(252)
    vol_mean  = vol_20d.rolling(250).mean()
    vol_ratio = (vol_20d / vol_mean).fillna(1.0).clip(0.3, 3.0)
    eps_t_c2  = (0.020 * vol_ratio).values
    tilt_c2   = _tilt_deadband(raw_a2_arr, vz_arr, bull_mask, eps_t_c2)
    wn_c2     = wn_A + tilt_c2
    wb_c2     = np.clip(wb_A - tilt_c2, 0.0, wb_A)
    nav_c2    = _nav(close, lev_e4, wn_c2, wg_A, wb_c2, dates, gold_2x, bond_3x, sofr, L_s2_7.values)
    _add('[C] C2 adaptive deadband (ε₀=0.020)', nav_c2, 'C2-adaptive', wfa_g11, trades_yr=c2_trades)

    # ====================================================================
    # Block 8: C3 (Yang-Zhang vol replacement)
    # ====================================================================
    print('\n■ Block 8: C3 (Yang-Zhang vol)')
    ohlc_path = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')
    if os.path.exists(ohlc_path):
        try:
            ohlc_df  = pd.read_csv(ohlc_path, parse_dates=['Date'])
            ohlc_df  = ohlc_df.set_index('Date').reindex(close.index).ffill()
            yz_vol   = _yang_zhang_vol(ohlc_df, n=10)
            yz_mean  = yz_vol.rolling(252, min_periods=20).mean()
            yz_std   = yz_vol.rolling(252, min_periods=20).std().replace(0, 0.001)
            vz_yz    = ((yz_vol - yz_mean) / yz_std).fillna(0.0)
            lev_raw_c3, wn_c3, wg_c3, wb_c3, n_tr_c3 = simulate_rebalance_A(raw_a2, vz_yz, THRESHOLD)
            L_s2_c3  = compute_L_s2_vz_gated(ret, vz_yz, **S2_FIXED_7)
            lev_c3   = _build_lev_mod(lev_raw_c3, lt_sig_raw, vz_yz, k_lo=K_LO, k_hi=K_HI, vz_thr=0.7)
            nav_c3   = _nav(close, lev_c3, wn_c3, wg_c3, wb_c3, dates, gold_2x, bond_3x, sofr, L_s2_c3.values)
            _add('[C] C3 Yang-Zhang (yz_n=10/vz=0.7)', nav_c3, 'C3-yz', wfa_g11,
                 trades_yr=n_tr_c3 / n_years)
        except Exception as e:
            print(f'  [ERROR] C3: {e}')
            results.append({'strategy': '[C] C3 Yang-Zhang (yz_n=10/vz=0.7)',
                            'CAGR_OOS': float('nan'), 'Sharpe_OOS': float('nan'),
                            'MaxDD_FULL': float('nan'), 'Worst10Y_star': float('nan'),
                            'P10_5Y': float('nan'), 'IS_OOS_gap': float('nan'),
                            'Trades_yr': float('nan'), 'WFA_CI95_lo': None, 'WFA_WFE': None})
    else:
        print(f'  [SKIP] C3: OHLC ファイルなし ({ohlc_path})')
        # C3 旧値を旧スプレッドCSVから読み込んで代替
        try:
            c3_raw = pd.read_csv(os.path.join(BASE, 'c3_yang_zhang_results.csv'))
            row = c3_raw[(abs(c3_raw['yz_n'] - 10) < 1e-5) & (abs(c3_raw['vz_thr'] - 0.7) < 1e-5)].iloc[0]
            ci95_c3, wfe_c3 = wfa_g11.get('C3-yz', (None, None))
            r_c3 = {k: float(row[k]) if k in row else float('nan')
                    for k in ['CAGR_OOS', 'Sharpe_OOS', 'MaxDD_FULL', 'Worst10Y_star',
                               'P10_5Y', 'IS_OOS_gap', 'Trades_yr']}
            r_c3['WFA_CI95_lo'] = ci95_c3
            r_c3['WFA_WFE']     = wfe_c3
            r_c3['strategy']    = '[C] C3 Yang-Zhang (yz_n=10/vz=0.7) ※旧コスト'
            results.append(r_c3)
            print(f'  C3: OHLC なし → 旧コストCSV値を掲載（参考値）')
        except Exception:
            results.append({'strategy': '[C] C3 Yang-Zhang (yz_n=10/vz=0.7) [SKIP]',
                            'CAGR_OOS': float('nan'), 'Sharpe_OOS': float('nan'),
                            'MaxDD_FULL': float('nan'), 'Worst10Y_star': float('nan'),
                            'P10_5Y': float('nan'), 'IS_OOS_gap': float('nan'),
                            'Trades_yr': float('nan'), 'WFA_CI95_lo': None, 'WFA_WFE': None})

    # ====================================================================
    # Block 9: D5 グリッド（7 configs）
    # ====================================================================
    print('\n■ Block 9: D5 (vz × lmax グリッド)')
    d5_configs = [
        ('[D5] vz=0.60/lmax=4.5',      0.60, 4.5, 'D5-vz060-lmax45',  wfa_g11),
        ('[D5] vz=0.60/lmax=5.0',      0.60, 5.0, 'D5-vz060-lmax50',  wfa_g11),
        ('[D5] vz=0.65/lmax=4.5',      0.65, 4.5, 'D5-vz065-lmax45',  wfa_g11),
        ('[D5] vz=0.65/lmax=5.5 ✅',   0.65, 5.5, 'vz065-lmax5p5',    wfa_g10),
        ('[D5] vz=0.65/lmax=6.0 ✅',   0.65, 6.0, 'vz065-lmax6',      wfa_g10),
        ('[D5] vz=0.65/lmax=7.0 ✅⚠',  0.65, 7.0, 'vz065-lmax7',      wfa_g10),
        ('[D5] vz=0.70/lmax=5.0',      0.70, 5.0, 'D5-vz070-lmax50',  wfa_g11),
    ]
    for cfg_name, vz_thr, l_max, wfa_key, wfa_dict in d5_configs:
        l_max_s = pd.Series(l_max, index=ret.index)
        L_d5    = compute_L_s2_dyn_lmax(ret, vz, l_max_s, **S2_BASE)
        lev_d5  = _build_lev_mod(lev_raw, lt_sig_raw, vz, k_lo=K_LO, k_hi=K_HI, vz_thr=vz_thr)
        nav_d5  = _nav(close, lev_d5, wn_A, wg_A, wb_A, dates, gold_2x, bond_3x, sofr, L_d5.values)
        _add(cfg_name, nav_d5, wfa_key, wfa_dict)

    # ====================================================================
    # CSV 出力
    # ====================================================================
    df_out = pd.DataFrame(results)
    csv_path = os.path.join(BASE, 'g13_sbi_cfd_full_results.csv')
    df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\nCSV: {csv_path}')

    _write_md(results)
    print('Done.')


def _write_md(results):
    H1, H2 = MD_HEADER_STRAT
    lines = []
    W = lines.append

    W('# 戦略パフォーマンス比較表（SBI CFD実績コスト版）— 2026-05-28 21戦略')
    W('')
    W('作成日: 2026-05-28')
    W('最終更新日: 2026-05-28')
    W('EVALUATION_STANDARD: **v1.3** | コスト: **SBI CFD NQ100 実績（3.0%/yr above SOFR）**')
    W('生成スクリプト: `src/g13_realistic_cost_full_comparison.py`')
    W('')
    W('> ### ⚠ コスト前提の変更')
    W('> 旧比較表（v1.9）は CFD スプレッド **0.20%/yr**（くりっく株365想定）で計算。')
    W('> 本表は **SBI CFD NQ100 実績スプレッド 3.0%/yr** で全 21 戦略を再計算した現実的な値。')
    W('> **WFA 欄（CI95_lo / Overfit(WFE)）は旧スプレッド計算の近似値**（参考掲載）。')
    W('')
    W('---')
    W('')
    W('## 📋 §1 比較前提')
    W('')
    W('| 項目 | 定義 |')
    W('|------|------|')
    W('| **IS** | 1974-01-02 〜 2021-05-07（47.3年） |')
    W('| **OOS** | 2021-05-08 〜 2026-03-26（4.9年） |')
    W('| **CFD スプレッド** | **3.0%/yr**（SBI CFD NQ100 業者マージン実績） |')
    W('| **旧前提との差** | +2.8%/yr → 平均 eff_L≈2〜3 倍 × 2.8% ≈ **−5〜8pp CAGR_OOS** |')
    W('| **WFA** | 旧スプレッドでの計算値を参照（新スプレッドでの再計算は Phase 2） |')
    W('')
    W('---')
    W('')
    W('## 📊 §2 全戦略 統合比較表（21戦略 × 9指標 / 10列）')
    W('')
    W('> **単位**: CAGR_OOS / Worst10Y★ / P10▷ / MaxDD = %、IS-OOS gap = pp、Tr = 回/年')
    W('> ★ = Sharpe_OOS > +0.885 / ◎ = > +0.770（S2ベースライン）')
    W('> **Overfit(WFE)**: WFA は旧スプレッド計算の近似値')
    W('')
    W(H1)
    W(H2)

    for m in results:
        label = str(m.get('strategy', ''))
        r = {}
        for k in ['CAGR_OOS', 'Sharpe_OOS', 'MaxDD_FULL', 'Worst10Y_star',
                  'P10_5Y', 'IS_OOS_gap', 'Trades_yr', 'WFA_CI95_lo', 'WFA_WFE']:
            v = m.get(k)
            r[k] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else float('nan')
        W(fmt_row_strat(label, r))

    W('')
    W(MD_METRIC_GLOSSARY)
    W('')
    W('> WFA CI95_lo / Overfit(WFE): 旧 CFD スプレッド（0.20%/yr）計算値（近似）。')
    W('')
    W('---')
    W('')

    # §3 サマリー
    W('## 🏆 §3 SBI CFD 実績コスト下: 採用判断ガイド')
    W('')

    df = pd.DataFrame(results)
    valid = df.dropna(subset=['CAGR_OOS', 'Sharpe_OOS', 'MaxDD_FULL'])
    if not valid.empty:
        bi_cagr   = valid.loc[valid['CAGR_OOS'].idxmax()]
        bi_sharpe = valid.loc[valid['Sharpe_OOS'].idxmax()]
        bi_maxdd  = valid.loc[valid['MaxDD_FULL'].idxmax()]   # 最小負値 = 最良

        W('| 指標 | 最良戦略（SBI CFD 3.0%） | 値 |')
        W('|------|---------|---|')
        W(f'| CAGR_OOS 最高 | {bi_cagr["strategy"][:55]} | {bi_cagr["CAGR_OOS"]*100:+.1f}% |')
        W(f'| Sharpe_OOS 最高 | {bi_sharpe["strategy"][:55]} | {bi_sharpe["Sharpe_OOS"]:+.3f} |')
        W(f'| MaxDD 最良（浅い） | {bi_maxdd["strategy"][:55]} | {bi_maxdd["MaxDD_FULL"]*100:+.1f}% |')

    W('')
    W('### コスト影響サマリー（vs バックテスト前提 0.20%/yr）')
    W('')
    W('| 戦略系統 | 平均eff_L | 追加コスト/yr | CAGR_OOS 変化 |')
    W('|---------|----------|-------------|--------------|')
    W('| E4 / lmax=7.0  | ≈2.9x | (2.9-1)×2.8%≈**5.3%** | −5〜6pp |')
    W('| D5 / lmax=5.0  | ≈2.4x | (2.4-1)×2.8%≈**3.9%** | −4〜5pp |')
    W('| D5 / lmax=4.5  | ≈2.2x | (2.2-1)×2.8%≈**3.4%** | −4pp |')
    W('')
    W('---')
    W('')
    W('## 📝 §4 改訂履歴')
    W('')
    W('| 版 | 日付 | 変更内容 |')
    W('|----|------|---------|')
    W('| **G13** | 2026-05-28 | SBI CFD 実績スプレッド（3.0%/yr）で全 21 戦略を再計算。CAGR/Sharpe/MaxDD 等を更新。WFA 欄は近似値。 |')
    W('')
    W('---')
    W('')
    W('*生成: `src/g13_realistic_cost_full_comparison.py` | SBI CFD NQ100 スプレッド=3.0%/yr | IS=1974-2021 / OOS=2021-2026*')

    md_path = os.path.join(BASE, 'STRATEGY_PERFORMANCE_COMPARISON_SBI_CFD_2026-05-28.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'MD:  {md_path}')


if __name__ == '__main__':
    main()
