import sys, os, types, pickle

# multitasking stub (must come before sys.path manipulation)
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

# Path resolution: this file is at src/audit/, so BASE is 3 levels up
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, 'src'))

AUDIT_DIR = os.path.join(BASE, 'audit_results')
CACHE_DIR = os.path.join(BASE, 'audit_results', '_cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'best_nav_cache.pkl')

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr, build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_a2_signal, simulate_rebalance_A,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy, calc_7metrics, CFD_SPREAD_LOW,
    IS_START, IS_END, OOS_START, DELAY,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import build_lt_signal, signal_to_bias, apply_lt_mode_b

# Fixed parameters (same values as b1_s2_lt2.py)
S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)
LT2_FIXED = dict(signal='LT2', N=750, k_lt=0.5, mode='B')
NAV_FLOOR = -0.999

# E4 Regime k_lt 確定パラメータ (CURRENT_BEST_STRATEGY.md 2026-05-24 確定)
E4_PARAMS = dict(k_lo=0.1, k_hi=0.8, k_mid=0.5, vz_thr=0.7)
E4_CACHE_FILE = os.path.join(CACHE_DIR, 'e4_nav_cache.pkl')
E4_SHARPE_OOS_EXPECTED = 0.891   # サニティ確認用


def _signal_to_bias_dynamic(lt_sig_arr: np.ndarray, k_arr: np.ndarray) -> np.ndarray:
    """E4用: element-wise lt_bias = (-k * sig * 0.5).clip(-0.5, 0.5)"""
    return np.clip(-k_arr * lt_sig_arr * 0.5, -0.5, 0.5)


def build_best_strategy_assets(force_rebuild: bool = False) -> dict:
    """ベスト戦略のNAV+共有資産をキャッシュ込みで返す。

    Returns dict keys:
      dates, close, ret, sofr, gold_2x, bond_3x,
      lev_A, wn_A, wg_A, wb_A, lev_mod, L_s2 (pd.Series), n_tr, n_years,
      nav_baseline (pd.Series, CFD_SPREAD_LOW=0.20%)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_FILE) and not force_rebuild:
        print(f'[CACHE] Loading from {CACHE_FILE}')
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print('[BUILD] Building best strategy assets (this may take 2-5 min)...')
    df = load_data(DATA_PATH)
    close = df['Close']
    ret = close.pct_change().fillna(0)
    dates = df['Date']
    n = len(df)
    n_years = n / TRADING_DAYS

    sofr = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)

    lt_sig = build_lt_signal(close, LT2_FIXED['signal'], LT2_FIXED['N'])
    lt_bias = signal_to_bias(lt_sig, LT2_FIXED['k_lt'])
    lev_mod = apply_lt_mode_b(lev_A, lt_bias, l_min=0.0, l_max=1.0)

    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)

    nav_baseline = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )

    cache = dict(
        dates=dates, close=close, ret=ret, sofr=sofr,
        gold_2x=gold_2x, bond_3x=bond_3x,
        lev_A=lev_A, wn_A=wn_A, wg_A=wg_A, wb_A=wb_A,
        lev_mod=lev_mod, L_s2=L_s2,
        n_tr=int(n_tr), n_years=float(n_years),
        nav_baseline=nav_baseline,
    )
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f'[CACHE] Saved to {CACHE_FILE}')
    return cache


def _build_nav_full_funding(assets: dict, L_arr, full_funding_ann: float) -> pd.Series:
    """L全体にfull_funding_annをかけるモデルでNAVを構築（くりっく365/ロール方式）"""
    close = assets['close']
    dates = assets['dates']
    lev_mod = assets['lev_mod']
    wn = assets['wn_A']
    wg = assets['wg_A']
    wb = assets['wb_A']
    gold_2x = assets['gold_2x']
    bond_3x = assets['bond_3x']

    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3 = pd.Series(bond_3x).pct_change().fillna(0).values

    idx = dates.index
    L = np.asarray(L_arr, dtype=float)
    if L.ndim == 0:
        L = np.full(len(r_nas), float(L_arr))

    # DELAY シフト（build_nav_strategy と同じ規約: DELAY=2）
    L_shifted = pd.Series(L, index=idx).shift(DELAY).fillna(1.0).values
    lev_s = pd.Series(lev_mod, index=idx).shift(DELAY).fillna(0).values
    wn_s  = pd.Series(wn, index=idx).shift(DELAY).fillna(0).values
    wg_s  = pd.Series(wg, index=idx).shift(DELAY).fillna(0).values
    wb_s  = pd.Series(wb, index=idx).shift(DELAY).fillna(0).values

    # くりっく365/ロール: L全体にfundingをかける（借入分だけでなくL倍分全体）
    nas_ret = L_shifted * r_nas - L_shifted * (full_funding_ann / TRADING_DAYS)
    daily = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    blowup_days = int((daily < NAV_FLOOR).sum())
    daily_clipped = np.maximum(daily, NAV_FLOOR)
    nav = (1 + pd.Series(daily_clipped, index=idx)).cumprod()
    nav.attrs['blowup_days'] = blowup_days
    return nav


def build_best_strategy_nav_for_scenario(
    assets: dict,
    cfd_spread: float = 0.0,
    extra_funding_ann: float = 0.0,
    full_funding_ann=None,
    cfd_leverage_override=None,
) -> pd.Series:
    L_use = assets['L_s2'].values if cfd_leverage_override is None else cfd_leverage_override

    if full_funding_ann is None:
        effective_spread = float(cfd_spread) + float(extra_funding_ann)
        return build_nav_strategy(
            assets['close'], assets['lev_mod'],
            assets['wn_A'], assets['wg_A'], assets['wb_A'], assets['dates'],
            assets['gold_2x'], assets['bond_3x'], assets['sofr'],
            nas_mode='CFD', cfd_leverage=L_use, cfd_spread=effective_spread,
        )
    else:
        return _build_nav_full_funding(assets, L_use, full_funding_ann)


def build_e4_strategy_assets(force_rebuild: bool = False) -> dict:
    """E4 Regime k_lt (k_lo=0.1, k_hi=0.8, k_mid=0.5, vz_thr=0.7) のNAV+共有資産をキャッシュ込みで返す。

    Returns dict keys (旧キャッシュに加えて以下が追加):
      vz, lt_sig, k_dyn, lt_bias_e4, lev_mod_e4, nav_e4, e4_params
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(E4_CACHE_FILE) and not force_rebuild:
        print(f'[CACHE] Loading E4 from {E4_CACHE_FILE}')
        with open(E4_CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print('[BUILD] Building E4 strategy assets (this may take 2-5 min)...')
    df = load_data(DATA_PATH)
    close = df['Close']
    ret = close.pct_change().fillna(0)
    dates = df['Date']
    n = len(df)
    n_years = n / TRADING_DAYS

    sofr = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)

    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)

    # LT2 シグナル (N=750)
    lt_sig = build_lt_signal(close, 'LT2', 750)

    # E4 レジーム条件付き k_dyn
    vz_arr = vz.values
    lt_sig_arr = lt_sig.values
    k_lo, k_hi, k_mid, vz_thr = (
        E4_PARAMS['k_lo'], E4_PARAMS['k_hi'],
        E4_PARAMS['k_mid'], E4_PARAMS['vz_thr']
    )
    k_dyn = np.where(vz_arr > vz_thr, k_hi,
                     np.where(vz_arr < -vz_thr, k_lo, k_mid))

    # E4 動的バイアス
    lt_bias_arr = _signal_to_bias_dynamic(lt_sig_arr, k_dyn)
    lt_bias_e4 = pd.Series(lt_bias_arr, index=close.index)
    lev_mod_e4 = np.asarray(apply_lt_mode_b(lev_A, lt_bias_e4, l_min=0.0, l_max=1.0), dtype=float)

    # E4 NAV
    nav_e4 = build_nav_strategy(
        close, lev_mod_e4, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )

    # サニティ: Sharpe_OOS確認
    oos_mask = dates >= OOS_START
    r_oos = nav_e4[oos_mask].pct_change().dropna()
    sr_oos = float(r_oos.mean() / r_oos.std(ddof=1) * np.sqrt(TRADING_DAYS))
    diff = abs(sr_oos - E4_SHARPE_OOS_EXPECTED)
    status = 'OK' if diff < 0.005 else 'WARN'
    print(f'[SANITY] Sharpe_OOS={sr_oos:.4f} (expected {E4_SHARPE_OOS_EXPECTED}) → {status}')

    cache = dict(
        dates=dates, close=close, ret=ret, sofr=sofr,
        gold_2x=gold_2x, bond_3x=bond_3x,
        lev_A=lev_A, wn_A=wn_A, wg_A=wg_A, wb_A=wb_A,
        L_s2=L_s2,
        n_tr=int(n_tr), n_years=float(n_years),
        vz=vz, lt_sig=lt_sig,
        k_dyn=k_dyn, lt_bias_e4=lt_bias_e4,
        lev_mod_e4=lev_mod_e4,
        nav_e4=nav_e4,
        e4_params=dict(E4_PARAMS),
    )
    with open(E4_CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f'[CACHE] E4 saved to {E4_CACHE_FILE}')
    return cache


def build_e4_strategy_nav_for_scenario(
    assets: dict,
    cfd_spread: float = 0.0,
    extra_funding_ann: float = 0.0,
    full_funding_ann=None,
    cfd_leverage_override=None,
) -> pd.Series:
    """E4戦略の lev_mod_e4 を使ったシナリオ別 NAV 構築。
    build_best_strategy_nav_for_scenario の E4 版（lev_mod → lev_mod_e4）。
    """
    L_use = assets['L_s2'].values if cfd_leverage_override is None else cfd_leverage_override
    if full_funding_ann is None:
        effective_spread = float(cfd_spread) + float(extra_funding_ann)
        return build_nav_strategy(
            assets['close'], assets['lev_mod_e4'],
            assets['wn_A'], assets['wg_A'], assets['wb_A'], assets['dates'],
            assets['gold_2x'], assets['bond_3x'], assets['sofr'],
            nas_mode='CFD', cfd_leverage=L_use, cfd_spread=effective_spread,
        )
    else:
        return _build_nav_full_funding_e4(assets, L_use, full_funding_ann)


def _build_nav_full_funding_e4(assets: dict, L_arr, full_funding_ann: float) -> pd.Series:
    """E4版: L全体にfull_funding_annをかけるモデルでNAVを構築（くりっく365/ロール方式）"""
    close = assets['close']
    dates = assets['dates']
    lev_mod = assets['lev_mod_e4']  # ← E4版の差分
    wn = assets['wn_A']
    wg = assets['wg_A']
    wb = assets['wb_A']
    gold_2x = assets['gold_2x']
    bond_3x = assets['bond_3x']

    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3 = pd.Series(bond_3x).pct_change().fillna(0).values

    idx = dates.index
    L = np.asarray(L_arr, dtype=float)
    if L.ndim == 0:
        L = np.full(len(r_nas), float(L_arr))

    L_shifted = pd.Series(L, index=idx).shift(DELAY).fillna(1.0).values
    lev_s = pd.Series(lev_mod, index=idx).shift(DELAY).fillna(0).values
    wn_s  = pd.Series(wn, index=idx).shift(DELAY).fillna(0).values
    wg_s  = pd.Series(wg, index=idx).shift(DELAY).fillna(0).values
    wb_s  = pd.Series(wb, index=idx).shift(DELAY).fillna(0).values

    nas_ret = L_shifted * r_nas - L_shifted * (full_funding_ann / TRADING_DAYS)
    daily = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    blowup_days = int((daily < NAV_FLOOR).sum())
    daily_clipped = np.maximum(daily, NAV_FLOOR)
    nav = (1 + pd.Series(daily_clipped, index=idx)).cumprod()
    nav.attrs['blowup_days'] = blowup_days
    return nav


def build_e4_assets_with_override(
    param_name: str,
    value,
    base_assets: dict,
) -> dict:
    """E4戦略を指定パラメータ1つだけオーバーライドして再構築（感度分析用）。
    キャッシュ不使用・毎回NAV再計算。
    base_assets は build_e4_strategy_assets() の戻り値（共有資産を使い回す）。
    """
    from dynamic_leverage_strategies import compute_L_s2_vz_gated
    from long_cycle_signal import build_lt_signal, apply_lt_mode_b

    # 中心値
    k_lo   = float(E4_PARAMS['k_lo'])
    k_hi   = float(E4_PARAMS['k_hi'])
    k_mid  = float(E4_PARAMS['k_mid'])
    vz_thr = float(E4_PARAMS['vz_thr'])
    N_lt2  = int(LT2_FIXED['N'])
    s2_params = dict(S2_FIXED)

    # オーバーライド
    if param_name == 'k_lo':
        k_lo = float(value)
    elif param_name == 'k_hi':
        k_hi = float(value)
    elif param_name == 'k_mid':
        k_mid = float(value)
    elif param_name == 'vz_thr':
        vz_thr = float(value)
    elif param_name == 'N_lt2':
        N_lt2 = int(value)
    elif param_name in s2_params:
        s2_params[param_name] = float(value)
    else:
        raise ValueError(f'Unknown param_name: {param_name}')

    close  = base_assets['close']
    ret    = base_assets['ret']
    dates  = base_assets['dates']
    vz     = base_assets['vz']
    lev_A  = base_assets['lev_A']
    wn_A   = base_assets['wn_A']
    wg_A   = base_assets['wg_A']
    wb_A   = base_assets['wb_A']
    sofr   = base_assets['sofr']
    gold_2x = base_assets['gold_2x']
    bond_3x = base_assets['bond_3x']

    # L_s2 再計算（s2_params が変わった場合のみ）
    if param_name in S2_FIXED:
        L_s2 = compute_L_s2_vz_gated(ret, vz, **s2_params)
    else:
        L_s2 = base_assets['L_s2']

    # lt_sig 再計算（N_lt2 が変わった場合のみ）
    if param_name == 'N_lt2':
        lt_sig = build_lt_signal(close, 'LT2', N_lt2)
    else:
        lt_sig = base_assets['lt_sig']

    # k_dyn 再計算
    vz_arr     = vz.values
    lt_sig_arr = lt_sig.values
    k_dyn = np.where(vz_arr > vz_thr, k_hi,
                     np.where(vz_arr < -vz_thr, k_lo, k_mid))

    lt_bias_arr = _signal_to_bias_dynamic(lt_sig_arr, k_dyn)
    lt_bias_e4  = pd.Series(lt_bias_arr, index=close.index)
    lev_mod_e4  = np.asarray(apply_lt_mode_b(lev_A, lt_bias_e4, l_min=0.0, l_max=1.0), dtype=float)

    nav_e4 = build_nav_strategy(
        close, lev_mod_e4, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )

    result = dict(base_assets)
    result.update(dict(
        k_dyn=k_dyn, lt_sig=lt_sig, lt_bias_e4=lt_bias_e4,
        lev_mod_e4=lev_mod_e4, nav_e4=nav_e4, L_s2=L_s2,
        override_param=param_name, override_value=value,
    ))
    return result


# ===========================================================================
# F10: F8-R5 (tilt=10.0, CALM_BOOST) + ε=0.015 Deadband
# Architecture:
#   - Same E4 lev_mod_e4 (vz_thr=0.70, k_lo=0.1, k_hi=0.8, k_mid=0.5)
#   - Same S2 L_s2 (l_max=7.0)
#   - ε-deadband tilt applied to wn_A/wb_A (F8-R5 CALM_BOOST logic)
#   - NAV = build_nav_strategy(close, lev_mod_e4, wn_tilted, wg_A, wb_tilted, ...)
# ===========================================================================

F10_PARAMS = dict(
    eps=0.015, tilt=10.0, vz_reg=0.7,
    cap_calm=0.15, cap_bull=0.10, cap_bear=0.05,
)
F10_CACHE_FILE = os.path.join(CACHE_DIR, 'f10_nav_cache.pkl')
F10_SHARPE_OOS_EXPECTED = 0.935


def _compute_tilt_with_deadband(
    raw_a2_vals: np.ndarray,
    vz_vals: np.ndarray,
    threshold: float,
    eps: float = 0.015,
    tilt: float = 10.0,
    vz_reg: float = 0.7,
    cap_calm: float = 0.15,
    cap_bull: float = 0.10,
    cap_bear: float = 0.05,
) -> np.ndarray:
    """F10 ε-deadband tilt (g7_wfa_f10.py compute_tilt_with_deadband と同一ロジック)。"""
    cap_eff = np.where(np.abs(vz_vals) < vz_reg, cap_calm,
              np.where(vz_vals > vz_reg, cap_bull, cap_bear))
    tilt_raw    = tilt * (raw_a2_vals - threshold) * (1.0 - raw_a2_vals)
    tilt_target = np.minimum(np.maximum(tilt_raw, 0.0), cap_eff)
    bull_mask   = raw_a2_vals > threshold
    tilt_target = np.where(bull_mask, tilt_target, 0.0)

    confirmed = np.zeros(len(raw_a2_vals), dtype=float)
    cur = 0.0
    for i in range(len(raw_a2_vals)):
        if i == 0 or abs(tilt_target[i] - cur) >= eps:
            cur = tilt_target[i]
        confirmed[i] = cur
    return confirmed


def build_f10_strategy_assets(force_rebuild: bool = False) -> dict:
    """F10 (ε=0.015 deadband) strategy assets。

    Returns dict keys (E4と共通):
      dates, close, ret, sofr, gold_2x, bond_3x,
      lev_A, wn_A, wg_A, wb_A, L_s2, vz, lt_sig, k_dyn, lt_bias_e4, lev_mod_e4,
      n_tr, n_years,
      F10固有:
      raw_a2, tilt_confirmed, wn_tilted, wb_tilted, nav_f10, f10_params
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(F10_CACHE_FILE) and not force_rebuild:
        print(f'[CACHE] Loading F10 from {F10_CACHE_FILE}')
        with open(F10_CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print('[BUILD] Building F10 strategy assets (this may take 2-5 min)...')
    df = load_data(DATA_PATH)
    close = df['Close']
    ret = close.pct_change().fillna(0)
    dates = df['Date']
    n_years = len(df) / TRADING_DAYS

    sofr = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)

    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)  # l_max=7.0

    lt_sig = build_lt_signal(close, 'LT2', 750)
    vz_arr     = vz.values
    lt_sig_arr = lt_sig.values
    k_lo, k_hi, k_mid, vz_thr = (
        E4_PARAMS['k_lo'], E4_PARAMS['k_hi'], E4_PARAMS['k_mid'], E4_PARAMS['vz_thr']
    )
    k_dyn = np.where(vz_arr > vz_thr, k_hi,
                     np.where(vz_arr < -vz_thr, k_lo, k_mid))
    lt_bias_arr = _signal_to_bias_dynamic(lt_sig_arr, k_dyn)
    lt_bias_e4  = pd.Series(lt_bias_arr, index=close.index)
    lev_mod_e4  = np.asarray(apply_lt_mode_b(lev_A, lt_bias_e4, l_min=0.0, l_max=1.0), dtype=float)

    # ε-deadband tilt
    raw_a2_vals = raw_a2.values if hasattr(raw_a2, 'values') else np.asarray(raw_a2)
    tilt_confirmed = _compute_tilt_with_deadband(
        raw_a2_vals, vz_arr, THRESHOLD,
        eps=F10_PARAMS['eps'], tilt=F10_PARAMS['tilt'], vz_reg=F10_PARAMS['vz_reg'],
        cap_calm=F10_PARAMS['cap_calm'], cap_bull=F10_PARAMS['cap_bull'],
        cap_bear=F10_PARAMS['cap_bear'],
    )
    wn_tilted = np.asarray(wn_A) + tilt_confirmed
    wb_tilted = np.clip(np.asarray(wb_A) - tilt_confirmed, 0.0, np.asarray(wb_A))

    nav_f10 = build_nav_strategy(
        close, lev_mod_e4, wn_tilted, wg_A, wb_tilted, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )

    oos_mask = dates >= OOS_START
    r_oos = nav_f10[oos_mask].pct_change().dropna()
    sr_oos = float(r_oos.mean() / r_oos.std(ddof=1) * np.sqrt(TRADING_DAYS))
    diff = abs(sr_oos - F10_SHARPE_OOS_EXPECTED)
    status = 'OK' if diff < 0.01 else 'WARN'
    print(f'[SANITY] F10 Sharpe_OOS={sr_oos:.4f} (expected {F10_SHARPE_OOS_EXPECTED}) → {status}')

    cache = dict(
        dates=dates, close=close, ret=ret, sofr=sofr,
        gold_2x=gold_2x, bond_3x=bond_3x,
        lev_A=lev_A, wn_A=wn_A, wg_A=wg_A, wb_A=wb_A,
        L_s2=L_s2, n_tr=int(n_tr), n_years=float(n_years),
        vz=vz, lt_sig=lt_sig, raw_a2=raw_a2,
        k_dyn=k_dyn, lt_bias_e4=lt_bias_e4, lev_mod_e4=lev_mod_e4,
        tilt_confirmed=tilt_confirmed,
        wn_tilted=wn_tilted, wb_tilted=wb_tilted,
        nav_f10=nav_f10, f10_params=dict(F10_PARAMS),
    )
    with open(F10_CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f'[CACHE] F10 saved to {F10_CACHE_FILE}')
    return cache


def build_f10_strategy_nav_for_scenario(
    assets: dict,
    cfd_spread: float = 0.0,
    extra_funding_ann: float = 0.0,
    full_funding_ann=None,
    cfd_leverage_override=None,
) -> pd.Series:
    """F10戦略のシナリオ別 NAV 構築 (wn_tilted/wb_tilted 使用)。"""
    L_use = assets['L_s2'].values if cfd_leverage_override is None else cfd_leverage_override
    if full_funding_ann is None:
        effective_spread = float(cfd_spread) + float(extra_funding_ann)
        return build_nav_strategy(
            assets['close'], assets['lev_mod_e4'],
            assets['wn_tilted'], assets['wg_A'], assets['wb_tilted'], assets['dates'],
            assets['gold_2x'], assets['bond_3x'], assets['sofr'],
            nas_mode='CFD', cfd_leverage=L_use, cfd_spread=effective_spread,
        )
    else:
        return _build_nav_full_funding_f10(assets, L_use, full_funding_ann)


def _build_nav_full_funding_f10(assets: dict, L_arr, full_funding_ann: float) -> pd.Series:
    close = assets['close']
    dates = assets['dates']
    lev_mod = assets['lev_mod_e4']
    wn  = assets['wn_tilted']
    wg  = assets['wg_A']
    wb  = assets['wb_tilted']
    gold_2x = assets['gold_2x']
    bond_3x = assets['bond_3x']

    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3  = pd.Series(bond_3x).pct_change().fillna(0).values

    idx = dates.index
    L   = np.asarray(L_arr, dtype=float)
    if L.ndim == 0:
        L = np.full(len(r_nas), float(L_arr))

    L_shifted  = pd.Series(L, index=idx).shift(DELAY).fillna(1.0).values
    lev_s      = pd.Series(lev_mod, index=idx).shift(DELAY).fillna(0).values
    wn_s       = pd.Series(wn, index=idx).shift(DELAY).fillna(0).values
    wg_s       = pd.Series(wg, index=idx).shift(DELAY).fillna(0).values
    wb_s       = pd.Series(wb, index=idx).shift(DELAY).fillna(0).values

    nas_ret = L_shifted * r_nas - L_shifted * (full_funding_ann / TRADING_DAYS)
    daily   = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    blowup_days   = int((daily < NAV_FLOOR).sum())
    daily_clipped = np.maximum(daily, NAV_FLOOR)
    nav = (1 + pd.Series(daily_clipped, index=idx)).cumprod()
    nav.attrs['blowup_days'] = blowup_days
    return nav


def build_f10_assets_with_override(param_name: str, value, base_assets: dict) -> dict:
    """F10戦略を指定パラメータ1つオーバーライドして再構築（感度分析用）。"""
    k_lo   = float(E4_PARAMS['k_lo'])
    k_hi   = float(E4_PARAMS['k_hi'])
    k_mid  = float(E4_PARAMS['k_mid'])
    vz_thr = float(E4_PARAMS['vz_thr'])
    N_lt2  = 750
    eps    = float(F10_PARAMS['eps'])
    s2_params = dict(S2_FIXED)

    if param_name == 'k_lo':       k_lo   = float(value)
    elif param_name == 'k_hi':     k_hi   = float(value)
    elif param_name == 'k_mid':    k_mid  = float(value)
    elif param_name == 'vz_thr':   vz_thr = float(value)
    elif param_name == 'N_lt2':    N_lt2  = int(value)
    elif param_name == 'eps':      eps    = float(value)
    elif param_name in s2_params:  s2_params[param_name] = float(value)
    else:
        raise ValueError(f'Unknown param_name: {param_name}')

    close   = base_assets['close']
    ret     = base_assets['ret']
    dates   = base_assets['dates']
    vz      = base_assets['vz']
    raw_a2  = base_assets['raw_a2']
    lev_A   = base_assets['lev_A']
    wn_A    = base_assets['wn_A']
    wg_A    = base_assets['wg_A']
    wb_A    = base_assets['wb_A']
    sofr    = base_assets['sofr']
    gold_2x = base_assets['gold_2x']
    bond_3x = base_assets['bond_3x']

    L_s2 = compute_L_s2_vz_gated(ret, vz, **s2_params) if param_name in S2_FIXED else base_assets['L_s2']

    lt_sig = build_lt_signal(close, 'LT2', N_lt2) if param_name == 'N_lt2' else base_assets['lt_sig']

    vz_arr     = vz.values
    lt_sig_arr = lt_sig.values
    k_dyn = np.where(vz_arr > vz_thr, k_hi, np.where(vz_arr < -vz_thr, k_lo, k_mid))
    lt_bias_arr = _signal_to_bias_dynamic(lt_sig_arr, k_dyn)
    lt_bias_e4  = pd.Series(lt_bias_arr, index=close.index)
    lev_mod_e4  = np.asarray(apply_lt_mode_b(lev_A, lt_bias_e4, l_min=0.0, l_max=1.0), dtype=float)

    raw_a2_vals = raw_a2.values if hasattr(raw_a2, 'values') else np.asarray(raw_a2)
    tilt_confirmed = _compute_tilt_with_deadband(
        raw_a2_vals, vz_arr, THRESHOLD, eps=eps,
        tilt=F10_PARAMS['tilt'], vz_reg=F10_PARAMS['vz_reg'],
        cap_calm=F10_PARAMS['cap_calm'], cap_bull=F10_PARAMS['cap_bull'],
        cap_bear=F10_PARAMS['cap_bear'],
    )
    wn_tilted = np.asarray(wn_A) + tilt_confirmed
    wb_tilted = np.clip(np.asarray(wb_A) - tilt_confirmed, 0.0, np.asarray(wb_A))

    nav_f10 = build_nav_strategy(
        close, lev_mod_e4, wn_tilted, wg_A, wb_tilted, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )

    result = dict(base_assets)
    result.update(dict(
        k_dyn=k_dyn, lt_sig=lt_sig, lt_bias_e4=lt_bias_e4,
        lev_mod_e4=lev_mod_e4, tilt_confirmed=tilt_confirmed,
        wn_tilted=wn_tilted, wb_tilted=wb_tilted,
        nav_f10=nav_f10, L_s2=L_s2,
        override_param=param_name, override_value=value,
    ))
    return result


# ===========================================================================
# vz065+lmax5: E4 regime k_lt with vz_thr=0.65 + l_max=5.0
# Architecture:
#   - vz_thr=0.65 for k_dyn (k_lo=0.1, k_hi=0.8, k_mid=0.5) → lev_mod_065
#   - l_max=5.0 → L_s2_lmax5
#   - Same wn_A, wg_A, wb_A (no tilt)
#   - NAV = build_nav_strategy(close, lev_mod_065, wn_A, wg_A, wb_A, ..., L_s2_lmax5)
# ===========================================================================

S2_LMAX5_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=5.0, step=0.5)
VZ065LMAX5_PARAMS = dict(vz_thr=0.65, l_max=5.0, k_lo=0.1, k_hi=0.8, k_mid=0.5)
VZ065LMAX5_CACHE_FILE = os.path.join(CACHE_DIR, 'vz065lmax5_nav_cache.pkl')
VZ065LMAX5_SHARPE_OOS_EXPECTED = 0.949


def build_vz065lmax5_strategy_assets(force_rebuild: bool = False) -> dict:
    """vz065+lmax5 strategy assets (vz_thr=0.65, l_max=5.0)。

    Returns dict keys:
      dates, close, ret, sofr, gold_2x, bond_3x,
      lev_A, wn_A, wg_A, wb_A, L_s2_lmax5, L_s2 (l_max=7 for reference),
      vz, lt_sig, k_dyn_065, lt_bias_065, lev_mod_065,
      n_tr, n_years, nav_vz065lmax5, vz065lmax5_params
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(VZ065LMAX5_CACHE_FILE) and not force_rebuild:
        print(f'[CACHE] Loading vz065lmax5 from {VZ065LMAX5_CACHE_FILE}')
        with open(VZ065LMAX5_CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print('[BUILD] Building vz065+lmax5 strategy assets (this may take 2-5 min)...')
    df = load_data(DATA_PATH)
    close = df['Close']
    ret = close.pct_change().fillna(0)
    dates = df['Date']
    n_years = len(df) / TRADING_DAYS

    sofr = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)

    L_s2       = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)       # l_max=7.0 (参照用)
    L_s2_lmax5 = compute_L_s2_vz_gated(ret, vz, **S2_LMAX5_FIXED)  # l_max=5.0

    lt_sig = build_lt_signal(close, 'LT2', 750)
    vz_arr     = vz.values
    lt_sig_arr = lt_sig.values

    vz_thr = VZ065LMAX5_PARAMS['vz_thr']  # 0.65
    k_lo   = VZ065LMAX5_PARAMS['k_lo']
    k_hi   = VZ065LMAX5_PARAMS['k_hi']
    k_mid  = VZ065LMAX5_PARAMS['k_mid']

    k_dyn_065 = np.where(vz_arr > vz_thr, k_hi,
                         np.where(vz_arr < -vz_thr, k_lo, k_mid))
    lt_bias_arr = _signal_to_bias_dynamic(lt_sig_arr, k_dyn_065)
    lt_bias_065 = pd.Series(lt_bias_arr, index=close.index)
    lev_mod_065 = np.asarray(apply_lt_mode_b(lev_A, lt_bias_065, l_min=0.0, l_max=1.0), dtype=float)

    nav_vz065lmax5 = build_nav_strategy(
        close, lev_mod_065, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2_lmax5.values, cfd_spread=CFD_SPREAD_LOW,
    )

    oos_mask = dates >= OOS_START
    r_oos = nav_vz065lmax5[oos_mask].pct_change().dropna()
    sr_oos = float(r_oos.mean() / r_oos.std(ddof=1) * np.sqrt(TRADING_DAYS))
    diff = abs(sr_oos - VZ065LMAX5_SHARPE_OOS_EXPECTED)
    status = 'OK' if diff < 0.01 else 'WARN'
    print(f'[SANITY] vz065lmax5 Sharpe_OOS={sr_oos:.4f} (expected {VZ065LMAX5_SHARPE_OOS_EXPECTED}) → {status}')

    cache = dict(
        dates=dates, close=close, ret=ret, sofr=sofr,
        gold_2x=gold_2x, bond_3x=bond_3x,
        lev_A=lev_A, wn_A=wn_A, wg_A=wg_A, wb_A=wb_A,
        L_s2=L_s2, L_s2_lmax5=L_s2_lmax5,
        n_tr=int(n_tr), n_years=float(n_years),
        vz=vz, lt_sig=lt_sig, raw_a2=raw_a2,
        k_dyn_065=k_dyn_065, lt_bias_065=lt_bias_065, lev_mod_065=lev_mod_065,
        nav_vz065lmax5=nav_vz065lmax5, vz065lmax5_params=dict(VZ065LMAX5_PARAMS),
    )
    with open(VZ065LMAX5_CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f'[CACHE] vz065lmax5 saved to {VZ065LMAX5_CACHE_FILE}')
    return cache


def build_vz065lmax5_strategy_nav_for_scenario(
    assets: dict,
    cfd_spread: float = 0.0,
    extra_funding_ann: float = 0.0,
    full_funding_ann=None,
    cfd_leverage_override=None,
) -> pd.Series:
    """vz065+lmax5戦略のシナリオ別 NAV 構築 (lev_mod_065 + L_s2_lmax5 使用)。"""
    L_use = assets['L_s2_lmax5'].values if cfd_leverage_override is None else cfd_leverage_override
    if full_funding_ann is None:
        effective_spread = float(cfd_spread) + float(extra_funding_ann)
        return build_nav_strategy(
            assets['close'], assets['lev_mod_065'],
            assets['wn_A'], assets['wg_A'], assets['wb_A'], assets['dates'],
            assets['gold_2x'], assets['bond_3x'], assets['sofr'],
            nas_mode='CFD', cfd_leverage=L_use, cfd_spread=effective_spread,
        )
    else:
        return _build_nav_full_funding_vz065lmax5(assets, L_use, full_funding_ann)


def _build_nav_full_funding_vz065lmax5(assets: dict, L_arr, full_funding_ann: float) -> pd.Series:
    close   = assets['close']
    dates   = assets['dates']
    lev_mod = assets['lev_mod_065']
    wn = assets['wn_A']; wg = assets['wg_A']; wb = assets['wb_A']
    gold_2x = assets['gold_2x']; bond_3x = assets['bond_3x']

    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3  = pd.Series(bond_3x).pct_change().fillna(0).values
    idx   = dates.index
    L     = np.asarray(L_arr, dtype=float)
    if L.ndim == 0:
        L = np.full(len(r_nas), float(L_arr))

    L_shifted = pd.Series(L, index=idx).shift(DELAY).fillna(1.0).values
    lev_s = pd.Series(lev_mod, index=idx).shift(DELAY).fillna(0).values
    wn_s  = pd.Series(wn, index=idx).shift(DELAY).fillna(0).values
    wg_s  = pd.Series(wg, index=idx).shift(DELAY).fillna(0).values
    wb_s  = pd.Series(wb, index=idx).shift(DELAY).fillna(0).values

    nas_ret = L_shifted * r_nas - L_shifted * (full_funding_ann / TRADING_DAYS)
    daily   = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    blowup_days   = int((daily < NAV_FLOOR).sum())
    daily_clipped = np.maximum(daily, NAV_FLOOR)
    nav = (1 + pd.Series(daily_clipped, index=idx)).cumprod()
    nav.attrs['blowup_days'] = blowup_days
    return nav


def build_vz065lmax5_assets_with_override(param_name: str, value, base_assets: dict) -> dict:
    """vz065+lmax5戦略を指定パラメータ1つオーバーライドして再構築（感度分析用）。"""
    vz_thr = VZ065LMAX5_PARAMS['vz_thr']
    k_lo   = VZ065LMAX5_PARAMS['k_lo']
    k_hi   = VZ065LMAX5_PARAMS['k_hi']
    k_mid  = VZ065LMAX5_PARAMS['k_mid']
    N_lt2  = 750
    s2_params = dict(S2_LMAX5_FIXED)

    if param_name == 'vz_thr':     vz_thr = float(value)
    elif param_name == 'k_lo':     k_lo   = float(value)
    elif param_name == 'k_hi':     k_hi   = float(value)
    elif param_name == 'k_mid':    k_mid  = float(value)
    elif param_name == 'N_lt2':    N_lt2  = int(value)
    elif param_name in s2_params:  s2_params[param_name] = float(value)
    else:
        raise ValueError(f'Unknown param_name: {param_name}')

    close = base_assets['close']; ret = base_assets['ret']
    dates = base_assets['dates']; vz  = base_assets['vz']
    lev_A = base_assets['lev_A']; wn_A = base_assets['wn_A']
    wg_A  = base_assets['wg_A']; wb_A  = base_assets['wb_A']
    sofr  = base_assets['sofr']; gold_2x = base_assets['gold_2x']
    bond_3x = base_assets['bond_3x']

    L_s2_lmax5 = compute_L_s2_vz_gated(ret, vz, **s2_params) if param_name in S2_LMAX5_FIXED else base_assets['L_s2_lmax5']

    lt_sig = build_lt_signal(close, 'LT2', N_lt2) if param_name == 'N_lt2' else base_assets['lt_sig']

    vz_arr     = vz.values
    lt_sig_arr = lt_sig.values
    k_dyn_065 = np.where(vz_arr > vz_thr, k_hi, np.where(vz_arr < -vz_thr, k_lo, k_mid))
    lt_bias_arr = _signal_to_bias_dynamic(lt_sig_arr, k_dyn_065)
    lt_bias_065 = pd.Series(lt_bias_arr, index=close.index)
    lev_mod_065 = np.asarray(apply_lt_mode_b(lev_A, lt_bias_065, l_min=0.0, l_max=1.0), dtype=float)

    nav_vz065lmax5 = build_nav_strategy(
        close, lev_mod_065, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2_lmax5.values, cfd_spread=CFD_SPREAD_LOW,
    )

    result = dict(base_assets)
    result.update(dict(
        k_dyn_065=k_dyn_065, lt_sig=lt_sig, lt_bias_065=lt_bias_065,
        lev_mod_065=lev_mod_065, nav_vz065lmax5=nav_vz065lmax5, L_s2_lmax5=L_s2_lmax5,
        override_param=param_name, override_value=value,
    ))
    return result


# ===========================================================================
# F10+lmax5: F10 (ε=0.015 deadband) + l_max=5.0
# Architecture:
#   - Same E4 lev_mod_e4 (vz_thr=0.70)
#   - ε-deadband tilt on wn/wb (same as F10)
#   - l_max=5.0 → L_s2_lmax5
# ===========================================================================

F10LMAX5_PARAMS = dict(
    eps=0.015, tilt=10.0, vz_reg=0.7,
    cap_calm=0.15, cap_bull=0.10, cap_bear=0.05,
    l_max=5.0,
)
F10LMAX5_CACHE_FILE = os.path.join(CACHE_DIR, 'f10lmax5_nav_cache.pkl')
F10LMAX5_SHARPE_OOS_EXPECTED = 0.937


def build_f10lmax5_strategy_assets(force_rebuild: bool = False) -> dict:
    """F10+lmax5 strategy assets (ε=0.015 deadband + l_max=5.0)。

    Returns dict keys: same as F10 but L_s2_lmax5 instead of L_s2, nav_f10lmax5
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(F10LMAX5_CACHE_FILE) and not force_rebuild:
        print(f'[CACHE] Loading F10+lmax5 from {F10LMAX5_CACHE_FILE}')
        with open(F10LMAX5_CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print('[BUILD] Building F10+lmax5 strategy assets (this may take 2-5 min)...')
    df = load_data(DATA_PATH)
    close = df['Close']
    ret = close.pct_change().fillna(0)
    dates = df['Date']
    n_years = len(df) / TRADING_DAYS

    sofr = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)

    L_s2       = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)       # l_max=7.0 (参照用)
    L_s2_lmax5 = compute_L_s2_vz_gated(ret, vz, **S2_LMAX5_FIXED)  # l_max=5.0

    lt_sig = build_lt_signal(close, 'LT2', 750)
    vz_arr     = vz.values
    lt_sig_arr = lt_sig.values

    k_lo, k_hi, k_mid, vz_thr = (
        E4_PARAMS['k_lo'], E4_PARAMS['k_hi'], E4_PARAMS['k_mid'], E4_PARAMS['vz_thr']
    )
    k_dyn = np.where(vz_arr > vz_thr, k_hi, np.where(vz_arr < -vz_thr, k_lo, k_mid))
    lt_bias_arr = _signal_to_bias_dynamic(lt_sig_arr, k_dyn)
    lt_bias_e4  = pd.Series(lt_bias_arr, index=close.index)
    lev_mod_e4  = np.asarray(apply_lt_mode_b(lev_A, lt_bias_e4, l_min=0.0, l_max=1.0), dtype=float)

    raw_a2_vals = raw_a2.values if hasattr(raw_a2, 'values') else np.asarray(raw_a2)
    tilt_confirmed = _compute_tilt_with_deadband(
        raw_a2_vals, vz_arr, THRESHOLD,
        eps=F10LMAX5_PARAMS['eps'], tilt=F10LMAX5_PARAMS['tilt'],
        vz_reg=F10LMAX5_PARAMS['vz_reg'],
        cap_calm=F10LMAX5_PARAMS['cap_calm'], cap_bull=F10LMAX5_PARAMS['cap_bull'],
        cap_bear=F10LMAX5_PARAMS['cap_bear'],
    )
    wn_tilted = np.asarray(wn_A) + tilt_confirmed
    wb_tilted = np.clip(np.asarray(wb_A) - tilt_confirmed, 0.0, np.asarray(wb_A))

    nav_f10lmax5 = build_nav_strategy(
        close, lev_mod_e4, wn_tilted, wg_A, wb_tilted, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2_lmax5.values, cfd_spread=CFD_SPREAD_LOW,
    )

    oos_mask = dates >= OOS_START
    r_oos = nav_f10lmax5[oos_mask].pct_change().dropna()
    sr_oos = float(r_oos.mean() / r_oos.std(ddof=1) * np.sqrt(TRADING_DAYS))
    diff = abs(sr_oos - F10LMAX5_SHARPE_OOS_EXPECTED)
    status = 'OK' if diff < 0.01 else 'WARN'
    print(f'[SANITY] F10+lmax5 Sharpe_OOS={sr_oos:.4f} (expected {F10LMAX5_SHARPE_OOS_EXPECTED}) → {status}')

    cache = dict(
        dates=dates, close=close, ret=ret, sofr=sofr,
        gold_2x=gold_2x, bond_3x=bond_3x,
        lev_A=lev_A, wn_A=wn_A, wg_A=wg_A, wb_A=wb_A,
        L_s2=L_s2, L_s2_lmax5=L_s2_lmax5,
        n_tr=int(n_tr), n_years=float(n_years),
        vz=vz, lt_sig=lt_sig, raw_a2=raw_a2,
        k_dyn=k_dyn, lt_bias_e4=lt_bias_e4, lev_mod_e4=lev_mod_e4,
        tilt_confirmed=tilt_confirmed,
        wn_tilted=wn_tilted, wb_tilted=wb_tilted,
        nav_f10lmax5=nav_f10lmax5, f10lmax5_params=dict(F10LMAX5_PARAMS),
    )
    with open(F10LMAX5_CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f'[CACHE] F10+lmax5 saved to {F10LMAX5_CACHE_FILE}')
    return cache


def build_f10lmax5_strategy_nav_for_scenario(
    assets: dict,
    cfd_spread: float = 0.0,
    extra_funding_ann: float = 0.0,
    full_funding_ann=None,
    cfd_leverage_override=None,
) -> pd.Series:
    """F10+lmax5戦略のシナリオ別 NAV 構築 (lev_mod_e4 + wn_tilted + L_s2_lmax5 使用)。"""
    L_use = assets['L_s2_lmax5'].values if cfd_leverage_override is None else cfd_leverage_override
    if full_funding_ann is None:
        effective_spread = float(cfd_spread) + float(extra_funding_ann)
        return build_nav_strategy(
            assets['close'], assets['lev_mod_e4'],
            assets['wn_tilted'], assets['wg_A'], assets['wb_tilted'], assets['dates'],
            assets['gold_2x'], assets['bond_3x'], assets['sofr'],
            nas_mode='CFD', cfd_leverage=L_use, cfd_spread=effective_spread,
        )
    else:
        return _build_nav_full_funding_f10lmax5(assets, L_use, full_funding_ann)


def _build_nav_full_funding_f10lmax5(assets: dict, L_arr, full_funding_ann: float) -> pd.Series:
    close   = assets['close']
    dates   = assets['dates']
    lev_mod = assets['lev_mod_e4']
    wn = assets['wn_tilted']; wg = assets['wg_A']; wb = assets['wb_tilted']
    gold_2x = assets['gold_2x']; bond_3x = assets['bond_3x']

    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3  = pd.Series(bond_3x).pct_change().fillna(0).values
    idx   = dates.index
    L     = np.asarray(L_arr, dtype=float)
    if L.ndim == 0:
        L = np.full(len(r_nas), float(L_arr))

    L_shifted = pd.Series(L, index=idx).shift(DELAY).fillna(1.0).values
    lev_s = pd.Series(lev_mod, index=idx).shift(DELAY).fillna(0).values
    wn_s  = pd.Series(wn, index=idx).shift(DELAY).fillna(0).values
    wg_s  = pd.Series(wg, index=idx).shift(DELAY).fillna(0).values
    wb_s  = pd.Series(wb, index=idx).shift(DELAY).fillna(0).values

    nas_ret = L_shifted * r_nas - L_shifted * (full_funding_ann / TRADING_DAYS)
    daily   = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    blowup_days   = int((daily < NAV_FLOOR).sum())
    daily_clipped = np.maximum(daily, NAV_FLOOR)
    nav = (1 + pd.Series(daily_clipped, index=idx)).cumprod()
    nav.attrs['blowup_days'] = blowup_days
    return nav


def build_f10lmax5_assets_with_override(param_name: str, value, base_assets: dict) -> dict:
    """F10+lmax5戦略を指定パラメータ1つオーバーライドして再構築（感度分析用）。"""
    k_lo   = float(E4_PARAMS['k_lo'])
    k_hi   = float(E4_PARAMS['k_hi'])
    k_mid  = float(E4_PARAMS['k_mid'])
    vz_thr = float(E4_PARAMS['vz_thr'])
    N_lt2  = 750
    eps    = float(F10LMAX5_PARAMS['eps'])
    s2_params = dict(S2_LMAX5_FIXED)

    if param_name == 'k_lo':       k_lo   = float(value)
    elif param_name == 'k_hi':     k_hi   = float(value)
    elif param_name == 'k_mid':    k_mid  = float(value)
    elif param_name == 'vz_thr':   vz_thr = float(value)
    elif param_name == 'N_lt2':    N_lt2  = int(value)
    elif param_name == 'eps':      eps    = float(value)
    elif param_name in s2_params:  s2_params[param_name] = float(value)
    else:
        raise ValueError(f'Unknown param_name: {param_name}')

    close = base_assets['close']; ret = base_assets['ret']
    dates = base_assets['dates']; vz  = base_assets['vz']
    raw_a2 = base_assets['raw_a2']
    lev_A = base_assets['lev_A']; wn_A = base_assets['wn_A']
    wg_A  = base_assets['wg_A']; wb_A  = base_assets['wb_A']
    sofr  = base_assets['sofr']; gold_2x = base_assets['gold_2x']
    bond_3x = base_assets['bond_3x']

    L_s2_lmax5 = compute_L_s2_vz_gated(ret, vz, **s2_params) if param_name in S2_LMAX5_FIXED else base_assets['L_s2_lmax5']

    lt_sig = build_lt_signal(close, 'LT2', N_lt2) if param_name == 'N_lt2' else base_assets['lt_sig']

    vz_arr     = vz.values
    lt_sig_arr = lt_sig.values
    k_dyn = np.where(vz_arr > vz_thr, k_hi, np.where(vz_arr < -vz_thr, k_lo, k_mid))
    lt_bias_arr = _signal_to_bias_dynamic(lt_sig_arr, k_dyn)
    lt_bias_e4  = pd.Series(lt_bias_arr, index=close.index)
    lev_mod_e4  = np.asarray(apply_lt_mode_b(lev_A, lt_bias_e4, l_min=0.0, l_max=1.0), dtype=float)

    raw_a2_vals = raw_a2.values if hasattr(raw_a2, 'values') else np.asarray(raw_a2)
    tilt_confirmed = _compute_tilt_with_deadband(
        raw_a2_vals, vz_arr, THRESHOLD, eps=eps,
        tilt=F10LMAX5_PARAMS['tilt'], vz_reg=F10LMAX5_PARAMS['vz_reg'],
        cap_calm=F10LMAX5_PARAMS['cap_calm'], cap_bull=F10LMAX5_PARAMS['cap_bull'],
        cap_bear=F10LMAX5_PARAMS['cap_bear'],
    )
    wn_tilted = np.asarray(wn_A) + tilt_confirmed
    wb_tilted = np.clip(np.asarray(wb_A) - tilt_confirmed, 0.0, np.asarray(wb_A))

    nav_f10lmax5 = build_nav_strategy(
        close, lev_mod_e4, wn_tilted, wg_A, wb_tilted, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2_lmax5.values, cfd_spread=CFD_SPREAD_LOW,
    )

    result = dict(base_assets)
    result.update(dict(
        k_dyn=k_dyn, lt_sig=lt_sig, lt_bias_e4=lt_bias_e4,
        lev_mod_e4=lev_mod_e4, tilt_confirmed=tilt_confirmed,
        wn_tilted=wn_tilted, wb_tilted=wb_tilted,
        nav_f10lmax5=nav_f10lmax5, L_s2_lmax5=L_s2_lmax5,
        override_param=param_name, override_value=value,
    ))
    return result


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print('=== Building / loading best strategy assets (旧戦略) ===')
    a = build_best_strategy_assets(force_rebuild=False)
    print(f'  dates : {a["dates"].iloc[0].date()} -> {a["dates"].iloc[-1].date()}')
    print(f'  n_tr  : {a["n_tr"]} ({a["n_tr"]/a["n_years"]:.1f}/yr)')
    print(f'  nav_baseline: {float(a["nav_baseline"].iloc[-1]):.4f} (final NAV)')

    print()
    print('=== Building / loading E4 strategy assets ===')
    e4 = build_e4_strategy_assets(force_rebuild=False)
    print(f'  lev_mod_e4 range: [{e4["lev_mod_e4"].min():.3f}, {e4["lev_mod_e4"].max():.3f}]')
    print(f'  k_dyn distribution: lo={float((e4["k_dyn"] == E4_PARAMS["k_lo"]).mean()*100):.1f}%  '
          f'mid={float((e4["k_dyn"] == E4_PARAMS["k_mid"]).mean()*100):.1f}%  '
          f'hi={float((e4["k_dyn"] == E4_PARAMS["k_hi"]).mean()*100):.1f}%')
    print(f'  nav_e4 final: {float(e4["nav_e4"].iloc[-1]):.4f}')
    print(f'  E4 Cache: {E4_CACHE_FILE}')

    print()
    print('=== Building / loading F10 strategy assets ===')
    f10 = build_f10_strategy_assets(force_rebuild=False)
    print(f'  tilt_confirmed range: [{f10["tilt_confirmed"].min():.4f}, {f10["tilt_confirmed"].max():.4f}]')
    print(f'  nav_f10 final: {float(f10["nav_f10"].iloc[-1]):.4f}')

    print()
    print('=== Building / loading vz065+lmax5 strategy assets ===')
    vz65 = build_vz065lmax5_strategy_assets(force_rebuild=False)
    print(f'  lev_mod_065 range: [{vz65["lev_mod_065"].min():.3f}, {vz65["lev_mod_065"].max():.3f}]')
    print(f'  nav_vz065lmax5 final: {float(vz65["nav_vz065lmax5"].iloc[-1]):.4f}')

    print()
    print('=== Building / loading F10+lmax5 strategy assets ===')
    f10l5 = build_f10lmax5_strategy_assets(force_rebuild=False)
    print(f'  nav_f10lmax5 final: {float(f10l5["nav_f10lmax5"].iloc[-1]):.4f}')
