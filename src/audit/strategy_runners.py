"""
src/audit/strategy_runners.py
==============================
E4 戦略 & vz065 戦略 (l5/l7) の NAV を生データから再生成し、コスト基準を切替可能にする薄いラッパ。

公開 API:
    run_e4(basis: str) -> dict
        basis ∈ {'scenarioD', 'realistic'}
        返り値: {
            'nav':            pd.Series,   # 日次 NAV (DatetimeIndex)
            'trades_per_year': float,
            **compute_10metrics(nav, trades_per_year),  # 10 指標
        }

    run_vz065(lmax: float, basis: str) -> dict
        lmax ∈ {5.0, 7.0} (他の値も受理可)
        basis ∈ {'scenarioD', 'realistic'}
        返り値: run_e4 と同一スキーマ。
        scenarioD: g27 と同一コスト前提
            (overnight = (L-1)×(sofr+SBI_CFD_SPREAD/252), 売買コスト SPREAD_RT=0.05%)
        realistic: E4 realistic と同一手法
            (overnight = (sofr+CFD_OVERNIGHT_SPREAD/252)×L_t (フルNotional),
             売買コスト = CFD_SPREAD_ONE_WAY × pos_change)

設計方針:
  - 正典ファイル (src/*.py) は一切改変しない。
  - 既存の E4 組み立てロジック（e4_regime_klt.py）を最大限再利用する。
  - コスト層だけ差し替える: 'realistic' (Option A) では歴史変動SOFR + 3.0%/yr の
    フルNotional課金を日次適用し、売買スプレッドも CFD_SPREAD_ONE_WAY で計上。
  - 'scenarioD': 既存の build_nav_strategy(cfd_spread=CFD_SPREAD_LOW) そのまま。

vz065 コスト設計:
  scenarioD: g27 と同一 (overnight = (L-1)×(sofr+SBI_CFD_SPREAD/252),
             SPREAD_RT=0.00050 moderate 片道コスト)
  realistic: E4 と同一手法 (overnight = cfd_overnight_daily(sofr_t, L_t),
             売買コスト = CFD_SPREAD_ONE_WAY × pos_change)

CFD財務コスト比較:
  scenarioD: (L-1) × (sofr_daily + CFD_SPREAD_LOW/252)
             = (L-1) × (sofr_daily + 0.0020/252)
  realistic (Option A): cfd_overnight_daily(sofr_daily_t, L_t)
             = (sofr_daily_t + CFD_OVERNIGHT_SPREAD/252) × L_t
             = (時変SOFR/252 + 3.0%/252) × L_t  (フルNotional、L-1倍ではない)
             + CFD_SPREAD_ONE_WAY × pos_change (ポジション変化日に往復スプレッド控除)

  ※ realistic (Option A) は歴史的時変SOFR（DTB3系列）を使用するため、
    1974〜2021 IS 期間の高金利期（1980年代 SOFR≈15%）のコストが正確に反映される。
    旧実装の固定SOFR（SOFR_2026=3.63%）より IS 期間コストが大幅に増加する。

注: 本モジュールは実行時に src/ をパスに追加する（unified_metrics.py と同様）。
"""

from __future__ import annotations

import os
import sys
import types

# ---------------------------------------------------------------------------
# Patch: multitasking stub + sys.path
# ---------------------------------------------------------------------------
if "multitasking" not in sys.modules:
    _m = types.ModuleType("multitasking")
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules["multitasking"] = _m

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical imports (read-only)
# ---------------------------------------------------------------------------
from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr,
    build_bond_1x_nav_corrected,
    build_gold_2x,
    build_bond_3x,
    build_a2_signal,
    simulate_rebalance_A,
    DATA_PATH,
    TRADING_DAYS,
    THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy,
    CFD_SPREAD_LOW,
    NAV_FLOOR,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local
from long_cycle_signal import build_lt_signal, apply_lt_mode_b
from e4_regime_klt import signal_to_bias_dynamic, K_MID, S2_FIXED
from g14_wfa_sbi_cfd import (
    SBI_CFD_SPREAD,
    K_LO as VZ_K_LO,
    K_HI as VZ_K_HI,
    K_MID as VZ_K_MID,
    VZ_THR_065,
    S2_BASE as VZ_S2_BASE,
    N_LT2,
    compute_tilt_with_deadband,
)
from g19a_f10_eps_extended import build_f10_wn_for_eps
from g18_daily_trade_cost_wfa import build_cfd_nav_with_cost

# Audit helpers
from src.audit.unified_metrics import compute_10metrics
from src.audit.product_costs_realistic_20260610 import (
    cfd_overnight_annual,
    cfd_overnight_daily,
    CFD_SPREAD_ONE_WAY,
)

# E4 採用 config
E4_CONFIG = dict(k_lo=0.1, k_hi=0.8, vz_thr=0.7)


# ---------------------------------------------------------------------------
# 内部ヘルパー: 共有アセット & シグナル (キャッシュ)
# ---------------------------------------------------------------------------
_SHARED: dict | None = None


def _load_shared() -> dict:
    """共有アセット・シグナルをロードしてキャッシュ (初回のみ計算)。"""
    global _SHARED
    if _SHARED is not None:
        return _SHARED

    df = load_data(DATA_PATH)
    close = df["Close"]
    ret = close.pct_change().fillna(0)
    dates = df["Date"]
    n = len(df)
    n_years = n / TRADING_DAYS

    sofr = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(
        dates, use_time_varying_duration=True, bond_maturity=22.0
    )
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_trades_yr = n_tr / n_years

    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)

    lt_sig_raw = build_lt_signal(close, "LT2", 750)
    lt_sig_arr = lt_sig_raw.values
    vz_arr = vz.values
    lev_arr = lev_raw

    # E4 採用 config: k_lo=0.1, k_hi=0.8, vz_thr=0.7
    k_lo = E4_CONFIG["k_lo"]
    k_hi = E4_CONFIG["k_hi"]
    vz_thr = E4_CONFIG["vz_thr"]

    regime_hi = vz_arr > +vz_thr
    regime_lo = vz_arr < -vz_thr
    k_dyn = np.where(regime_hi, k_hi, np.where(regime_lo, k_lo, K_MID))
    lt_bias_dyn = pd.Series(
        signal_to_bias_dynamic(lt_sig_arr, k_dyn),
        index=lt_sig_raw.index,
    )
    lev_mod = apply_lt_mode_b(lev_arr, lt_bias_dyn, l_min=0.0, l_max=1.0)

    _SHARED = dict(
        close=close,
        dates=dates,
        sofr=sofr,
        gold_2x=gold_2x,
        bond_3x=bond_3x,
        lev_mod=lev_mod,
        wn_A=wn_A,
        wg_A=wg_A,
        wb_A=wb_A,
        L_s2=L_s2,
        n_trades_yr=n_trades_yr,
        n=n,
        n_years=n_years,
    )
    return _SHARED


# ---------------------------------------------------------------------------
# NAV 構築: realistic コスト置換
# ---------------------------------------------------------------------------
_DELAY = 1  # corrected_strategy_backtest.DELAY と同値


def _build_nav_realistic(
    close: pd.Series,
    lev_mod: np.ndarray,
    wn_A: np.ndarray,
    wg_A: np.ndarray,
    wb_A: np.ndarray,
    dates: pd.Series,
    gold_2x_nav: np.ndarray,
    bond_3x_nav: np.ndarray,
    sofr_daily: np.ndarray,
    L_s2: pd.Series,
) -> pd.Series:
    """
    Realistic コスト (Option A): 歴史変動SOFR + CFDブローカースプレッド3.0%/yr + フルNotional課金。

    CFD 日次オーバーナイト = cfd_overnight_daily(sofr_daily_t, L_t)
        = (sofr_daily_t + CFD_OVERNIGHT_SPREAD/252) * L_t
    すなわち (時変SOFR年率/252 + 3.0%/252) × L_t (フルNotional、L-1倍ではない)

    売買コスト: ポジション変化 × 2 × CFD_SPREAD_ONE_WAY (往復スプレッド)

    NAV 崩壊フロア (-99.9%) 込み。
    """
    from cfd_leverage_backtest import NAV_FLOOR

    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(gold_2x_nav).pct_change().fillna(0).values
    r_b3 = pd.Series(bond_3x_nav).pct_change().fillna(0).values

    idx = dates.index
    lev_s = pd.Series(lev_mod, index=idx).shift(_DELAY).fillna(0).values
    wn_s = pd.Series(wn_A, index=idx).shift(_DELAY).fillna(0).values
    wg_s = pd.Series(wg_A, index=idx).shift(_DELAY).fillna(0).values
    wb_s = pd.Series(wb_A, index=idx).shift(_DELAY).fillna(0).values

    L_arr = np.asarray(L_s2.values, dtype=float)
    L_shifted = pd.Series(L_arr, index=idx).shift(_DELAY).fillna(1.0).values

    # 時変 SOFR: sofr_daily は load_sofr() が返す日次系列 (DTB3/252相当)
    sofr_arr = np.asarray(sofr_daily, dtype=float)

    # 日次 CFD オーバーナイトコスト (Option A: 歴史変動SOFR × フルNotional)
    # = (sofr_daily_t + CFD_OVERNIGHT_SPREAD/252) * L_t
    overnight_daily_arr = np.vectorize(cfd_overnight_daily)(sofr_arr, L_shifted)

    # ポジション変化検出 (DELAY シフト後の L_shifted の差分)
    L_prev = np.concatenate([[L_shifted[0]], L_shifted[:-1]])
    pos_change = np.abs(L_shifted - L_prev)
    # 往復スプレッド: 変化量 × 2 × CFD_SPREAD_ONE_WAY (片道の2倍)
    spread_cost = pos_change * 2.0 * CFD_SPREAD_ONE_WAY

    nas_ret = L_shifted * r_nas - overnight_daily_arr - spread_cost

    daily = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    blowup_days = int((daily < NAV_FLOOR).sum())
    daily_clipped = np.maximum(daily, NAV_FLOOR)

    nav = (1 + pd.Series(daily_clipped, index=idx)).cumprod()
    nav.attrs["blowup_days"] = blowup_days
    return nav


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def run_e4(basis: str) -> dict:
    """E4 (S2_VZGated + LT2-N750 + E4 Regime k_lt, k_lo=0.1, k_hi=0.8, vz_thr=0.7) の
    NAV を生データから再生成し、10 指標 + NAV を返す。

    Parameters
    ----------
    basis : str
        'scenarioD' — 既存コードそのまま (CFD_SPREAD_LOW=0.20%/yr, SOFR 実績値)
        'realistic' — Option A: CFD オーバーナイト = (時変SOFR_daily + 3.0%/252) × L_t
                       (歴史変動SOFR使用, フルNotional課金)
                       + 売買スプレッド = pos_change × 2 × CFD_SPREAD_ONE_WAY

    Returns
    -------
    dict with keys:
        nav              : pd.Series
        trades_per_year  : float
        CAGR_IS, CAGR_OOS, CAGR_FULL, IS_OOS_gap_pp,
        Sharpe_OOS, MaxDD_FULL, Worst10Y_star, P10_5Y, Worst5Y, Trades_yr
    """
    if basis not in ("scenarioD", "realistic"):
        raise ValueError(f"basis must be 'scenarioD' or 'realistic', got {basis!r}")

    shared = _load_shared()
    close = shared["close"]
    dates = shared["dates"]
    sofr = shared["sofr"]
    gold_2x = shared["gold_2x"]
    bond_3x = shared["bond_3x"]
    lev_mod = shared["lev_mod"]
    wn_A = shared["wn_A"]
    wg_A = shared["wg_A"]
    wb_A = shared["wb_A"]
    L_s2 = shared["L_s2"]
    n_trades_yr = shared["n_trades_yr"]

    if basis == "scenarioD":
        nav = build_nav_strategy(
            close,
            lev_mod,
            wn_A,
            wg_A,
            wb_A,
            dates,
            gold_2x,
            bond_3x,
            sofr,
            nas_mode="CFD",
            cfd_leverage=L_s2.values,
            cfd_spread=CFD_SPREAD_LOW,
        )
    else:  # realistic
        nav = _build_nav_realistic(
            close,
            lev_mod,
            wn_A,
            wg_A,
            wb_A,
            dates,
            gold_2x,
            bond_3x,
            sofr,
            L_s2,
        )

    # calc_7metrics (inside compute_10metrics) needs dates comparable to pd.Timestamp.
    # build_nav_strategy returns a Series with integer index matching dates.index.
    # Reassign a DatetimeIndex so that unified_metrics.py can build a valid dates Series.
    dates_dt = pd.to_datetime(dates.values)
    nav_dt = pd.Series(nav.values, index=pd.DatetimeIndex(dates_dt))
    nav_dt.attrs.update(nav.attrs)

    metrics = compute_10metrics(nav_dt, n_trades_yr)
    return {"nav": nav_dt, "trades_per_year": n_trades_yr, **metrics}


# ---------------------------------------------------------------------------
# vz065 戦略共通定数
# ---------------------------------------------------------------------------
# g27 SPREAD_RT=0.00050 (moderate, round-trip) → 片道 0.00025
VZ065_SPREAD_RT = 0.00050
VZ065_EPS_F10 = 0.015
VZ065_S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)

# 共有アセットキャッシュ (vz065 用; E4 _SHARED とは別に保持)
_VZ065_SHARED: dict | None = None


def _load_vz065_shared() -> dict:
    """vz065 戦略に必要な共有アセット・シグナルをキャッシュ。
    E4 と共通のデータをベースに、vz_thr=0.65 の lev_mod と
    F10 ε=0.015 の wn/wb を構築する。
    """
    global _VZ065_SHARED
    if _VZ065_SHARED is not None:
        return _VZ065_SHARED

    df = load_data(DATA_PATH)
    close = df["Close"]
    ret = close.pct_change().fillna(0)
    dates = df["Date"]
    n = len(df)
    n_years = n / TRADING_DAYS

    sofr = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(
        dates, use_time_varying_duration=True, bond_maturity=22.0
    )
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_trades_yr_base = n_tr / n_years

    # vz_thr=0.65 の k_dyn → lt_bias_065 → lev_mod_065
    lt_sig_raw = build_lt_signal(close, "LT2", N=N_LT2)
    lt_sig_arr = lt_sig_raw.values
    vz_arr = vz.values

    k_dyn_065 = np.where(vz_arr > VZ_THR_065, VZ_K_HI,
                np.where(vz_arr < -VZ_THR_065, VZ_K_LO, VZ_K_MID))
    lt_bias_065 = pd.Series(
        np.clip(-k_dyn_065 * lt_sig_arr * 0.5, -0.5, 0.5),
        index=lt_sig_raw.index,
    )
    lev_mod_065 = apply_lt_mode_b(lev_raw, lt_bias_065, l_min=0.0, l_max=1.0)

    # F10 ε=0.015 tilt (g27 と同一)
    raw_a2_vals = raw_a2.values
    bull_mask = raw_a2_vals > THRESHOLD
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2_vals, vz_arr, wn_A, wb_A, np.asarray(lev_raw), bull_mask, VZ065_EPS_F10
    )
    wg_f10 = np.asarray(wg_A)

    _VZ065_SHARED = dict(
        close=close,
        dates=dates,
        sofr=sofr,
        gold_2x=gold_2x,
        bond_3x=bond_3x,
        ret=ret,
        vz=vz,
        vz_arr=vz_arr,
        lev_raw=lev_raw,
        lev_mod_065=lev_mod_065,
        wn_A=wn_A,
        wg_A=wg_A,
        wb_A=wb_A,
        wn_f10=wn_f10,
        wg_f10=wg_f10,
        wb_f10=wb_f10,
        n=n,
        n_years=n_years,
        n_trades_yr_base=n_trades_yr_base,
    )
    return _VZ065_SHARED


def _build_nav_vz065_realistic(
    close: pd.Series,
    lev_mod_065: np.ndarray,
    wn_f10: np.ndarray,
    wg_f10: np.ndarray,
    wb_f10: np.ndarray,
    dates: pd.Series,
    gold_2x_nav: np.ndarray,
    bond_3x_nav: np.ndarray,
    sofr_daily: np.ndarray,
    L_s2: pd.Series,
) -> pd.Series:
    """vz065 用 realistic NAV (E4 realistic と完全同一手法)。

    overnight = cfd_overnight_daily(sofr_daily_t, L_t) = (sofr_t/252 + 3%/252) × L_t
    売買コスト = |Δ(L_shifted)| × 2 × CFD_SPREAD_ONE_WAY
    """
    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(gold_2x_nav).pct_change().fillna(0).values
    r_b3 = pd.Series(bond_3x_nav).pct_change().fillna(0).values

    idx = dates.index
    lev_s = pd.Series(np.asarray(lev_mod_065, dtype=float), index=idx).shift(_DELAY).fillna(0).values
    wn_s = pd.Series(np.asarray(wn_f10, dtype=float), index=idx).shift(_DELAY).fillna(0).values
    wg_s = pd.Series(np.asarray(wg_f10, dtype=float), index=idx).shift(_DELAY).fillna(0).values
    wb_s = pd.Series(np.asarray(wb_f10, dtype=float), index=idx).shift(_DELAY).fillna(0).values

    L_arr = np.asarray(L_s2.values, dtype=float)
    L_shifted = pd.Series(L_arr, index=idx).shift(_DELAY).fillna(1.0).values

    sofr_arr = np.asarray(sofr_daily, dtype=float)
    overnight_daily_arr = np.vectorize(cfd_overnight_daily)(sofr_arr, L_shifted)

    L_prev = np.concatenate([[L_shifted[0]], L_shifted[:-1]])
    pos_change = np.abs(L_shifted - L_prev)
    spread_cost = pos_change * 2.0 * CFD_SPREAD_ONE_WAY

    nas_ret = L_shifted * r_nas - overnight_daily_arr - spread_cost

    daily = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    blowup_days = int((daily < NAV_FLOOR).sum())
    daily_clipped = np.maximum(daily, NAV_FLOOR)

    nav = (1 + pd.Series(daily_clipped, index=idx)).cumprod()
    nav.attrs["blowup_days"] = blowup_days
    return nav


# ---------------------------------------------------------------------------
# DH-W1 ETF コスト定数
# ---------------------------------------------------------------------------
# Realistic TER (ETF actual vs sim proxy)
_TER_UGL_REAL   = 0.0095   # UGL (Gold 2x): actual 0.95% vs sim 0.49%
_TER_TMF_REAL   = 0.0106   # TMF: actual 1.06% vs sim 0.91%
_TER_TQQQ_REAL  = 0.0086   # TQQQ: actual 0.86% (unchanged)
_TER_UGL_SIM    = 0.0049   # sim proxy (already embedded in build_dh_nav_with_cost via gold_2x)
_TER_TMF_SIM    = 0.0091   # sim proxy (already embedded via bond_3x)
_TER_TQQQ_SIM   = 0.0086   # sim proxy (already embedded)
# Incremental TER drag to add on top of sim (daily)
_TER_GOLD2X_EXTRA_DAILY = (_TER_UGL_REAL  - _TER_UGL_SIM)  / 252.0   # +0.46%/yr daily
_TER_TMF_EXTRA_DAILY    = (_TER_TMF_REAL  - _TER_TMF_SIM)  / 252.0   # +0.15%/yr daily
_TER_TQQQ_EXTRA_DAILY   = (_TER_TQQQ_REAL - _TER_TQQQ_SIM) / 252.0   # 0 (unchanged)

_DH_PER_UNIT = 0.0010  # scenarioD: moderate per-unit turnover cost (same as g23a)

# DH-W1 shared cache (computed once)
_DHW1_SHARED: dict | None = None


def _load_dhw1_shared() -> None:
    """DH-W1 用共有アセットを一度だけロードしてキャッシュ。"""
    global _DHW1_SHARED
    if _DHW1_SHARED is not None:
        return

    from g14_wfa_sbi_cfd import load_shared_assets
    from g23a_dh_refinement_variants import hold_mask_W1, DH_PER_UNIT
    from g18_daily_trade_cost_wfa import build_dh_nav_with_cost

    a = load_shared_assets()
    mask = hold_mask_W1(a)
    wn = np.asarray(a["wn_A"]) * mask
    wg = np.asarray(a["wg_A"]) * mask
    wb = np.asarray(a["wb_A"]) * mask
    lev_raw_masked = np.asarray(a["lev_raw"]) * mask

    # scenarioD NAV (build once, reuse)
    nav_base, _ = build_dh_nav_with_cost(
        a["close"], lev_raw_masked, wn, wg, wb,
        a["dates"], a["gold_2x"], a["bond_3x"], a["sofr"], DH_PER_UNIT,
    )

    _DHW1_SHARED = dict(
        assets=a,
        mask=mask,
        wn=wn,
        wg=wg,
        wb=wb,
        lev_raw_masked=lev_raw_masked,
        nav_base=nav_base,
    )


def _compute_dhw1_trades_per_year(lev_raw_masked: np.ndarray, dates: pd.Series) -> float:
    """lev_raw_masked の変化イベント数から Trades/yr を算出。

    定義: リバランスイベント数 = |lev[i] != lev[i-1]| のカウント (wn/wb 変化含む)。
    """
    n = len(lev_raw_masked)
    n_years = n / 252.0
    change_flag = np.zeros(n, dtype=bool)
    change_flag[1:] = lev_raw_masked[1:] != lev_raw_masked[:-1]
    n_trades = int(change_flag.sum())
    return n_trades / n_years if n_years > 0 else float("nan")


def _build_dhw1_nav_realistic(
    a: dict,
    wn: np.ndarray,
    wg: np.ndarray,
    wb: np.ndarray,
    lev_raw_masked: np.ndarray,
    dates: pd.Series,
) -> pd.Series:
    """DH-W1 realistic NAV: sim NAV に実TERとETF売買コストを追加控除する。

    アプローチ:
      1. scenarioD NAV (sim TER 込み) を build_dh_nav_with_cost で構築。
      2. 実TERとsim TERの差分 (incrementalTER) を日次 × 各アセット配分で追加控除。
      3. US ETF 売買コスト ($22上限モデル, us_etf_trade_cost_annual) を年率→日次で追加控除。
      4. FX ヘッジなし (米ETF未ヘッジ)。
      5. CFD オーバーナイトは適用しない (ETF 群)。

    SOFR financing: sim と同様に時変SOFR×2 を維持 (sim内で bond_3x/gold_2xに埋め込み済み)。
    """
    from g18_daily_trade_cost_wfa import build_dh_nav_with_cost
    from src.audit.product_costs_realistic_20260610 import us_etf_trade_cost_annual

    # Step 1: sim NAV with scenarioD per_unit turnover
    nav_sim, _ = build_dh_nav_with_cost(
        a["close"], lev_raw_masked, wn, wg, wb,
        dates, a["gold_2x"], a["bond_3x"], a["sofr"], _DH_PER_UNIT,
    )

    # Step 2: incremental TER drag (daily rate)
    # wg = Gold2x weight * mask, wb = Bond3x(TMF) weight * mask, wn = TQQQ weight * mask
    wn_arr = np.asarray(wn, dtype=float)
    wg_arr = np.asarray(wg, dtype=float)
    wb_arr = np.asarray(wb, dtype=float)

    # Daily incremental TER drag: weight × extra_TER_daily
    # (proportional to portfolio allocation, already accounts for exposure)
    ter_drag_daily = (
        wn_arr * _TER_TQQQ_EXTRA_DAILY
        + wg_arr * _TER_GOLD2X_EXTRA_DAILY
        + wb_arr * _TER_TMF_EXTRA_DAILY
    )

    # Step 3: US ETF trade cost (annual, $22 cap model)
    trades_per_year = _compute_dhw1_trades_per_year(lev_raw_masked, dates)
    etf_trade_cost_annual = us_etf_trade_cost_annual(trades_per_year)
    etf_trade_cost_daily = etf_trade_cost_annual / 252.0

    # Step 4: build adjusted NAV
    r_sim = np.nan_to_num(nav_sim.pct_change().fillna(0).values, nan=0.0)
    r_adj = r_sim - ter_drag_daily - etf_trade_cost_daily
    nav_adj = pd.Series(np.cumprod(1.0 + r_adj), index=nav_sim.index)
    return nav_adj


def run_dhw1(basis: str) -> dict:
    """DH-W1 (Asymm+Hysteresis, ENTER0.7/EXIT0.3) の NAV を再生成し、10 指標 + NAV を返す。

    ETF 群コスト体系:
      scenarioD : build_dh_nav_with_cost(per_unit_cost=DH_PER_UNIT=0.0010)
                  — SOFR financing は _make_nav 内の build_nav_strategy (SBI_CFD_SPREAD) 経由
      realistic  : TER を実製品値に差替 (UGL 0.95%, TMF 1.06%, TQQQ 0.86%) + 米ETF売買コスト
                  + 2×SOFR financing (時変) 維持。CFDオーバーナイトは適用しない。

    Parameters
    ----------
    basis : str
        'scenarioD' or 'realistic'

    Returns
    -------
    dict with keys: nav, trades_per_year, CAGR_IS, CAGR_OOS, CAGR_FULL,
        IS_OOS_gap_pp, Sharpe_OOS, MaxDD_FULL, Worst10Y_star, P10_5Y, Worst5Y, Trades_yr
    """
    if basis not in ("scenarioD", "realistic"):
        raise ValueError(f"basis must be 'scenarioD' or 'realistic', got {basis!r}")

    _load_dhw1_shared()
    shared = _DHW1_SHARED
    a = shared["assets"]
    nav_base = shared["nav_base"]
    wn = shared["wn"]
    wg = shared["wg"]
    wb = shared["wb"]
    lev_raw_masked = shared["lev_raw_masked"]
    mask = shared["mask"]
    dates = a["dates"]
    dates_dt = pd.to_datetime(dates.values)

    if basis == "scenarioD":
        nav = nav_base
    else:  # realistic
        nav = _build_dhw1_nav_realistic(a, wn, wg, wb, lev_raw_masked, dates)

    nav_dt = pd.Series(nav.values, index=pd.DatetimeIndex(dates_dt))

    trades_per_year = _compute_dhw1_trades_per_year(lev_raw_masked, dates)

    metrics = compute_10metrics(nav_dt, trades_per_year)
    return {"nav": nav_dt, "trades_per_year": trades_per_year, **metrics}


def run_overlay(variant: str, basis: str) -> dict:
    """DH-W1 + nasdaq_mom63 M6 overlay (V0 defensive / V7 boost) を再生成し、10 指標を返す。

    V0 mapping: {0: 1.10, 1: 1.00, 2: 0.90, 3: 0.80} (defensive)
    V7 mapping: {0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00} (boost)

    Pipeline: quantile_cut(levels=4) → publication_lag(+1BD) → lev×mask_W1×mult

    Parameters
    ----------
    variant : str
        'V0' or 'V7'
    basis : str
        'scenarioD' or 'realistic'

    Returns
    -------
    dict with keys: nav, trades_per_year, CAGR_IS, CAGR_OOS, ... (same as run_dhw1)
    """
    if variant not in ("V0", "V7"):
        raise ValueError(f"variant must be 'V0' or 'V7', got {variant!r}")
    if basis not in ("scenarioD", "realistic"):
        raise ValueError(f"basis must be 'scenarioD' or 'realistic', got {basis!r}")

    _load_dhw1_shared()
    shared = _DHW1_SHARED
    a = shared["assets"]
    mask = shared["mask"]
    dates = a["dates"]
    dates_dt = pd.to_datetime(dates.values)
    date_index = pd.DatetimeIndex(dates_dt)

    # V0/V7 mapping catalogue (from tune_s3_overlay_20260607.py)
    _OVERLAY_MAPPINGS = {
        "V0": {0: 1.10, 1: 1.00, 2: 0.90, 3: 0.80},  # defensive
        "V7": {0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00},  # pure boost
    }
    mapping = _OVERLAY_MAPPINGS[variant]

    # Load macro signal
    _macro_path = os.path.join(
        _SRC_DIR, "..", "data", "macro_features.csv"
    )
    macro = pd.read_csv(_macro_path, parse_dates=["date"], index_col="date").sort_index()
    signal_raw = macro["nasdaq_mom63"].dropna()

    # quantile_cut (full sample) then publication lag +1BD
    from signals.quantize import quantile_cut as _quantile_cut
    from signals.timing import apply_publication_lag as _apply_lag

    sig_q = _quantile_cut(signal_raw, levels=4)
    sig_q = sig_q.dropna().astype("int8")
    sig_lagged = _apply_lag(sig_q, "daily")
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]

    # Align to strategy dates
    sig_aligned = sig_lagged.reindex(date_index).ffill()
    mult_arr = sig_aligned.map(
        lambda s: mapping.get(int(s), 1.0) if pd.notna(s) else 1.0
    ).fillna(1.0).values
    mult_arr = np.clip(np.asarray(mult_arr, dtype=float), 0.0, 3.0)

    # Apply multiplier to lev_raw_masked (W1 mask already applied)
    lev_raw_masked_base = shared["lev_raw_masked"]
    lev_raw_mod = lev_raw_masked_base * mult_arr

    wn = shared["wn"]
    wg = shared["wg"]
    wb = shared["wb"]

    from g18_daily_trade_cost_wfa import build_dh_nav_with_cost

    if basis == "scenarioD":
        nav, _ = build_dh_nav_with_cost(
            a["close"], lev_raw_mod, wn, wg, wb,
            dates, a["gold_2x"], a["bond_3x"], a["sofr"], _DH_PER_UNIT,
        )
    else:  # realistic
        nav = _build_dhw1_nav_realistic(a, wn, wg, wb, lev_raw_mod, dates)

    nav_dt = pd.Series(nav.values, index=date_index)

    trades_per_year = _compute_dhw1_trades_per_year(lev_raw_mod, dates)

    metrics = compute_10metrics(nav_dt, trades_per_year)
    return {"nav": nav_dt, "trades_per_year": trades_per_year, **metrics}


def run_vz065(lmax: float, basis: str) -> dict:
    """vz=0.65 + lmax=X + F10ε=0.015 の NAV を再生成し、10 指標 + NAV を返す。

    Parameters
    ----------
    lmax : float
        CFD レバレッジ上限。通常 5.0 または 7.0。
    basis : str
        'scenarioD' — g27 と同一コスト前提
            overnight = (L-1)×(sofr+SBI_CFD_SPREAD/252),
            売買コスト = SPREAD_RT/2 = 0.00025 (moderate)
        'realistic' — E4 realistic と同一手法
            overnight = cfd_overnight_daily(sofr_t, L_t) (フルNotional, 時変SOFR),
            売買コスト = CFD_SPREAD_ONE_WAY × pos_change

    Returns
    -------
    dict with keys:
        nav              : pd.Series (DatetimeIndex)
        trades_per_year  : float
        CAGR_IS, CAGR_OOS, CAGR_FULL, IS_OOS_gap_pp,
        Sharpe_OOS, MaxDD_FULL, Worst10Y_star, P10_5Y, Worst5Y, Trades_yr
    """
    if basis not in ("scenarioD", "realistic"):
        raise ValueError(f"basis must be 'scenarioD' or 'realistic', got {basis!r}")

    shared = _load_vz065_shared()
    close = shared["close"]
    dates = shared["dates"]
    sofr = shared["sofr"]
    gold_2x = shared["gold_2x"]
    bond_3x = shared["bond_3x"]
    ret = shared["ret"]
    vz = shared["vz"]
    lev_mod_065 = shared["lev_mod_065"]
    wn_f10 = shared["wn_f10"]
    wg_f10 = shared["wg_f10"]
    wb_f10 = shared["wb_f10"]

    # lmax に対応する L_s2 を毎回生成 (キャッシュ対象外; lmax は可変引数)
    L_s2 = compute_L_s2_vz_gated(ret, vz, **{**VZ065_S2_BASE, "l_max": lmax})

    if basis == "scenarioD":
        # g27 build_variant と同一: build_cfd_nav_with_cost(spread_one_way=SPREAD_RT/2)
        nav, _ = build_cfd_nav_with_cost(
            close,
            lev_mod_065,
            wn_f10,
            wg_f10,
            wb_f10,
            dates,
            gold_2x,
            bond_3x,
            sofr,
            L_s2.values,
            VZ065_SPREAD_RT / 2.0,
        )
    else:  # realistic
        nav = _build_nav_vz065_realistic(
            close,
            np.asarray(lev_mod_065.values if hasattr(lev_mod_065, "values") else lev_mod_065),
            np.asarray(wn_f10),
            np.asarray(wg_f10),
            np.asarray(wb_f10),
            dates,
            gold_2x,
            bond_3x,
            sofr,
            L_s2,
        )

    # Trades per year: g27 と同様に wn_f10 or wb_f10 or L_s2 が変化した日をカウント
    # (g14.count_trades_in_window と同一ロジックを全期間に適用)
    wn_arr = np.asarray(wn_f10, dtype=float)
    wb_arr = np.asarray(wb_f10, dtype=float)
    L_arr = np.asarray(L_s2.values, dtype=float)
    n = len(L_arr)
    n_years = n / TRADING_DAYS
    # count_trades_in_window: wn[i] != wn[i-1] OR wb[i] != wb[i-1] OR lev[i] != lev[i-1]
    change_flag = np.zeros(n, dtype=bool)
    change_flag[1:] = (
        (wn_arr[1:] != wn_arr[:-1])
        | (wb_arr[1:] != wb_arr[:-1])
        | (L_arr[1:] != L_arr[:-1])
    )
    n_trades = int(change_flag.sum())
    trades_per_year = n_trades / n_years if n_years > 0 else float("nan")

    # DatetimeIndex に変換
    dates_dt = pd.to_datetime(dates.values)
    nav_dt = pd.Series(nav.values, index=pd.DatetimeIndex(dates_dt))
    if hasattr(nav, "attrs"):
        nav_dt.attrs.update(nav.attrs)

    metrics = compute_10metrics(nav_dt, trades_per_year)
    return {"nav": nav_dt, "trades_per_year": trades_per_year, **metrics}
