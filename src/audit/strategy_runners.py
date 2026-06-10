"""
src/audit/strategy_runners.py
==============================
E4 戦略の NAV を生データから再生成し、コスト基準を切替可能にする薄いラッパ。

公開 API:
    run_e4(basis: str) -> dict
        basis ∈ {'scenarioD', 'realistic'}
        返り値: {
            'nav':            pd.Series,   # 日次 NAV (DatetimeIndex)
            'trades_per_year': float,
            **compute_10metrics(nav, trades_per_year),  # 10 指標
        }

設計方針:
  - 正典ファイル (src/*.py) は一切改変しない。
  - 既存の E4 組み立てロジック（e4_regime_klt.py）を最大限再利用する。
  - コスト層だけ差し替える: 'realistic' では cfd_overnight_annual(L_t)/252 を
    1日ごとのオーバーナイトコストとして差し引き、売買スプレッドも CFD_SPREAD_ONE_WAY で計上。
  - 'scenarioD': 既存の build_nav_strategy(cfd_spread=CFD_SPREAD_LOW) そのまま。

CFD財務コスト比較:
  scenarioD: (L-1) × (sofr_daily + CFD_SPREAD_LOW/252)
             = (L-1) × (sofr_daily + 0.0020/252)
  realistic:  (SOFR_2026 + CFD_OVERNIGHT_SPREAD) × L / 252
             = (0.0363 + 0.030) × L / 252  (定数 SOFR 2026 年)
             + CFD_SPREAD_ONE_WAY × pos_change (ポジション変化日に往復スプレッド控除)

  ※ realistic は動的レバ L_t に対して日々のオーバーナイトコストを乗算するため、
    高レバ日の財務コストを正確に反映する。SOFR を定数 (SOFR_2026=3.63%) で固定している
    点に注意: 歴史的 SOFR（1974〜2021 IS 期間）の平均は現在より大きく異なる可能性があるが、
    「2026 年現在の実コスト感応度」を確認する目的では妥当。
    歴史的 SOFR 変動を含む完全 realistic は scenarioD コードを複製して sofr_daily ×
    (1+spread_pct) に差し替えることで実現可能（将来拡張）。

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
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local
from long_cycle_signal import build_lt_signal, apply_lt_mode_b
from e4_regime_klt import signal_to_bias_dynamic, K_MID, S2_FIXED

# Audit helpers
from src.audit.unified_metrics import compute_10metrics
from src.audit.product_costs_realistic_20260610 import (
    cfd_overnight_annual,
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
    Realistic コスト: CFD オーバーナイト = (SOFR_2026 + 3.0%) × L_t / 252 を
    日次ごとに適用し、ポジション変化日に往復スプレッド CFD_SPREAD_ONE_WAY × 2 を追加控除。

    NAV 崩壊フロア (-99.9%) 込み。

    ※ SOFR は定数 SOFR_2026=3.63% を使用。歴史的 SOFR 変動は含まない。
    """
    from cfd_leverage_backtest import NAV_FLOOR, TRADING_DAYS as TD

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

    # 日次 CFD 収益 (realistic)
    # r_cfd = L * r_nas - cfd_overnight_annual(L) / 252
    # cfd_overnight_annual(L) = (SOFR_2026 + CFD_OVERNIGHT_SPREAD) * L
    # 注: scenarioD の borrow = (L-1)×(sofr_daily + cfd_spread/252) と異なり
    #     realistic は L 全体に乗算（IB 系 CFD の典型的なモデル）。
    from src.audit.product_costs_realistic_20260610 import (
        CFD_OVERNIGHT_SPREAD,
        SOFR_2026,
    )

    overnight_annual_rate = SOFR_2026 + CFD_OVERNIGHT_SPREAD  # 0.0363 + 0.030 = 0.0663
    overnight_daily = overnight_annual_rate * L_shifted / TD   # 日次オーバーナイトコスト

    # ポジション変化検出 (DELAY シフト後の L_shifted の差分)
    L_prev = np.concatenate([[L_shifted[0]], L_shifted[:-1]])
    pos_change = np.abs(L_shifted - L_prev)
    # 往復スプレッド: 変化量 × 2 × CFD_SPREAD_ONE_WAY (片道の2倍)
    spread_cost = pos_change * 2.0 * CFD_SPREAD_ONE_WAY

    nas_ret = L_shifted * r_nas - overnight_daily - spread_cost

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
        'realistic' — CFD オーバーナイト = (SOFR_2026=3.63% + 3.0%) × L_t / 252、
                       売買スプレッド = pos_change × 2 × CFD_SPREAD_ONE_WAY

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
