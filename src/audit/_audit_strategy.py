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


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print('Building / loading best strategy assets ...')
    a = build_best_strategy_assets(force_rebuild=False)
    print(f'  dates : {a["dates"].iloc[0].date()} -> {a["dates"].iloc[-1].date()}')
    print(f'  n_tr  : {a["n_tr"]} ({a["n_tr"]/a["n_years"]:.1f}/yr)')
    print(f'  L_s2  : [{a["L_s2"].min():.1f}, {a["L_s2"].max():.1f}] median={a["L_s2"].median():.1f}')
    print(f'  nav_baseline: {float(a["nav_baseline"].iloc[-1]):.4f} (final NAV)')
    print(f'  Cache saved: {CACHE_FILE}')
