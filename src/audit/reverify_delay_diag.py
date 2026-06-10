"""
src/audit/reverify_delay_diag.py
================================
敵対的再検証: realistic 経路の DELAY 修正(2回目=DELAY=2) と初版(1回目=DELAY=1)、
どちらが scenarioD の時間規約と一致するかを数値で決着させる診断スクリプト。

正典(src/*.py)・audit コア(strategy_runners.py の関数本体)は一切改変しない。
本スクリプトは:
  - 既存 _load_shared() / _load_vz065_shared() を呼んでアセット・シグナルを取得
  - scenarioD と「コスト完全同一・DELAY可変」の realistic-構造 NAV を別途構築して diff
  - 実際の realistic ビルダーを DELAY=2/1 で実走して指標差を出す (module._DELAY を一時切替→復元)

実行:
    python -m src.audit.reverify_delay_diag
"""
from __future__ import annotations

import os
import sys
import types

if "multitasking" not in sys.modules:
    _m = types.ModuleType("multitasking")
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules["multitasking"] = _m

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SRC_DIR)
for p in (_SRC_DIR, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

import src.audit.strategy_runners as SR
from cfd_leverage_backtest import (
    build_nav_strategy,
    build_cfd_nas_sleeve,
    CFD_SPREAD_LOW,
    NAV_FLOOR,
)
from g18_daily_trade_cost_wfa import build_cfd_nav_with_cost
from src.audit.unified_metrics import compute_10metrics


# ---------------------------------------------------------------------------
# scenarioD-equivalent NAV built with the realistic builder's *structure*
# but scenarioD's *costs*, parameterised by DELAY.
# This isolates DELAY as the only free variable for the round-trip test.
# ---------------------------------------------------------------------------
def build_scenD_struct(close, lev_mod, wn_A, wg_A, wb_A, dates,
                       gold_2x, bond_3x, sofr, L_s2, delay):
    """Replicate _build_nav_realistic's algebra exactly, but with scenarioD cost:
       nas_ret = build_cfd_nas_sleeve(r_nas, L_shifted, sofr, CFD_SPREAD_LOW)
                 = L*r_nas - (L-1)*(sofr + 0.0020/252)         (no separate spread cost)
       This is *identical* to build_nav_strategy(nas_mode='CFD', cfd_spread=CFD_SPREAD_LOW)
       when delay == canonical DELAY.
    """
    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(np.asarray(gold_2x)).pct_change().fillna(0).values
    r_b3 = pd.Series(np.asarray(bond_3x)).pct_change().fillna(0).values

    idx = dates.index
    lev_s = pd.Series(np.asarray(lev_mod, float), index=idx).shift(delay).fillna(0).values
    wn_s = pd.Series(np.asarray(wn_A, float), index=idx).shift(delay).fillna(0).values
    wg_s = pd.Series(np.asarray(wg_A, float), index=idx).shift(delay).fillna(0).values
    wb_s = pd.Series(np.asarray(wb_A, float), index=idx).shift(delay).fillna(0).values

    L_arr = np.asarray(L_s2.values, float)
    L_shifted = pd.Series(L_arr, index=idx).shift(delay).fillna(1.0).values

    nas_ret = build_cfd_nas_sleeve(r_nas, L_shifted, np.asarray(sofr, float), CFD_SPREAD_LOW)
    daily = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3
    daily_clipped = np.maximum(daily, NAV_FLOOR)
    return (1 + pd.Series(daily_clipped, index=idx)).cumprod()


def max_abs_diff(a: pd.Series, b: pd.Series) -> float:
    av = np.asarray(a.values, float)
    bv = np.asarray(b.values, float)
    n = min(len(av), len(bv))
    return float(np.max(np.abs(av[:n] - bv[:n])))


def _to_dt(nav, dates):
    dates_dt = pd.to_datetime(dates.values)
    return pd.Series(np.asarray(nav.values), index=pd.DatetimeIndex(dates_dt))


def metrics_for_nav(nav, dates, trades):
    return compute_10metrics(_to_dt(nav, dates), trades)


# ---------------------------------------------------------------------------
def run_e4_diag():
    print("=" * 78)
    print("V2-E4: round-trip (scenarioD struct, scenarioD cost, DELAY swept)")
    print("=" * 78)
    s = SR._load_shared()
    close, dates, sofr = s["close"], s["dates"], s["sofr"]
    gold_2x, bond_3x = s["gold_2x"], s["bond_3x"]
    lev_mod, wn_A, wg_A, wb_A = s["lev_mod"], s["wn_A"], s["wg_A"], s["wb_A"]
    L_s2, ntr = s["L_s2"], s["n_trades_yr"]

    # Actual scenarioD NAV (canonical build_nav_strategy, internal DELAY=2)
    nav_scenD = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates, gold_2x, bond_3x, sofr,
        nas_mode="CFD", cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )

    results = {}
    for delay in (2, 1):
        nav_struct = build_scenD_struct(
            close, lev_mod, wn_A, wg_A, wb_A, dates, gold_2x, bond_3x, sofr, L_s2, delay)
        d = max_abs_diff(nav_scenD, nav_struct)
        print(f"  DELAY={delay}: max|NAV_scenD - NAV_struct| = {d:.3e}")
        results[delay] = d
    print()
    return results


def run_vz_diag():
    print("=" * 78)
    print("V2-vz065(l5): round-trip vs g18 build_cfd_nav_with_cost (scenarioD), DELAY swept")
    print("=" * 78)
    s = SR._load_vz065_shared()
    close, dates, sofr = s["close"], s["dates"], s["sofr"]
    gold_2x, bond_3x, ret, vz = s["gold_2x"], s["bond_3x"], s["ret"], s["vz"]
    lev_mod_065 = s["lev_mod_065"]
    wn_f10, wg_f10, wb_f10 = s["wn_f10"], s["wg_f10"], s["wb_f10"]

    from dynamic_leverage_strategies import compute_L_s2_vz_gated
    L_s2 = compute_L_s2_vz_gated(ret, vz, **{**SR.VZ065_S2_BASE, "l_max": 5.0})

    # scenarioD NAV for vz065 = g18 build_cfd_nav_with_cost (this is what run_vz065 scenarioD uses)
    nav_scenD, _ = build_cfd_nav_with_cost(
        close, lev_mod_065, wn_f10, wg_f10, wb_f10, dates,
        gold_2x, bond_3x, sofr, L_s2.values, SR.VZ065_SPREAD_RT / 2.0)

    # Replicate g18 algebra with scenarioD cost (overnight (L-1)x, spread one_way) at swept DELAY.
    # build_cfd_nav_with_cost internally = build_nav_strategy(CFD, SBI_CFD_SPREAD) - dpos*spread_ow.
    # We compare against build_scenD_struct using SBI spread + dpos cost to keep it apples-to-apples.
    from g14_wfa_sbi_cfd import SBI_CFD_SPREAD

    def build_g18_struct(delay):
        r_nas = close.pct_change().fillna(0).values
        r_g2 = pd.Series(np.asarray(gold_2x)).pct_change().fillna(0).values
        r_b3 = pd.Series(np.asarray(bond_3x)).pct_change().fillna(0).values
        idx = dates.index
        lev_s = pd.Series(np.asarray(lev_mod_065, float), index=idx).shift(delay).fillna(0).values
        wn_s = pd.Series(np.asarray(wn_f10, float), index=idx).shift(delay).fillna(0).values
        wg_s = pd.Series(np.asarray(wg_f10, float), index=idx).shift(delay).fillna(0).values
        wb_s = pd.Series(np.asarray(wb_f10, float), index=idx).shift(delay).fillna(0).values
        L_arr = np.asarray(L_s2.values, float)
        L_shifted = pd.Series(L_arr, index=idx).shift(delay).fillna(1.0).values
        nas_ret = build_cfd_nas_sleeve(r_nas, L_shifted, np.asarray(sofr, float), SBI_CFD_SPREAD)
        # g18 spread cost: pos = wn*lm*L_s2 (UNSHIFTED in g18), dpos*spread_ow
        pos = np.nan_to_num(np.asarray(wn_f10, float)) * \
              np.nan_to_num(np.asarray(lev_mod_065, float)) * \
              np.nan_to_num(L_arr)
        dpos = np.zeros_like(pos); dpos[1:] = np.abs(np.diff(pos))
        trade_cost = dpos * (SR.VZ065_SPREAD_RT / 2.0)
        daily = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3 - trade_cost
        daily_clipped = np.maximum(daily, NAV_FLOOR)
        return (1 + pd.Series(daily_clipped, index=idx)).cumprod()

    results = {}
    for delay in (2, 1):
        nav_struct = build_g18_struct(delay)
        d = max_abs_diff(nav_scenD, nav_struct)
        print(f"  DELAY={delay}: max|NAV_scenD(g18) - NAV_struct| = {d:.3e}")
        results[delay] = d
    print()
    return results


def run_realistic_delay_compare():
    """V4-②: actual realistic builder at DELAY=2 vs DELAY=1 → reproduce +21.83 / +15.35."""
    print("=" * 78)
    print("V4: actual realistic builder, DELAY=2 vs DELAY=1 (full notional)")
    print("=" * 78)
    rows = []
    orig = SR._DELAY
    try:
        for delay in (2, 1):
            SR._DELAY = delay
            e4 = SR.run_e4("realistic", cfd_notional="full")
            rows.append(("E4 realistic full", delay, e4["CAGR_OOS"], e4["CAGR_IS"],
                         e4["Sharpe_OOS"], e4["MaxDD_FULL"]))
            vz = SR.run_vz065(5.0, "realistic", cfd_notional="full")
            rows.append(("vz065_l5 realistic full", delay, vz["CAGR_OOS"], vz["CAGR_IS"],
                         vz["Sharpe_OOS"], vz["MaxDD_FULL"]))
    finally:
        SR._DELAY = orig
    print(f"  {'strategy':28s} {'DELAY':>5} {'CAGR_OOS':>9} {'CAGR_IS':>9} {'Sharpe':>7} {'MaxDD':>8}")
    for name, dl, oos, is_, sh, dd in rows:
        print(f"  {name:28s} {dl:>5} {oos*100:>8.2f}% {is_*100:>8.2f}% {sh:>7.3f} {dd*100:>7.2f}%")
    print()
    return rows


def run_ranking():
    """V4-③④: full ranking at DELAY=2 (canonical), realistic basis, full notional + borrowed."""
    print("=" * 78)
    print("V4-③④: 7-strategy ranking, realistic DELAY=2")
    print("=" * 78)
    assert SR._DELAY == 2, "module _DELAY must be restored to 2"
    table = []

    # CFD group (full notional)
    e4 = SR.run_e4("realistic", cfd_notional="full")
    table.append(("E4 (CFD)", e4))
    e4b = SR.run_e4("realistic", cfd_notional="borrowed")
    vz7 = SR.run_vz065(7.0, "realistic", cfd_notional="full")
    table.append(("vz065_l7 (CFD)", vz7))
    vz5 = SR.run_vz065(5.0, "realistic", cfd_notional="full")
    table.append(("vz065_l5 (CFD)", vz5))

    # ETF / fund group
    table.append(("P7 (fund)", SR.run_p7("realistic")))
    table.append(("V7 (ETF overlay)", SR.run_overlay("V7", "realistic")))
    table.append(("DH-W1 (ETF)", SR.run_dhw1("realistic")))
    table.append(("V0 (ETF overlay)", SR.run_overlay("V0", "realistic")))

    # also scenarioD for E4 (③ borrowed/full reference)
    e4_scenD = SR.run_e4("scenarioD")

    print(f"  {'strategy':18s} {'CAGR_OOS':>9} {'min(IS,OOS)':>11} {'aftertax_OOS':>12} {'MaxDD':>8} {'Sharpe':>7}")
    rows = []
    for name, m in table:
        oos = m["CAGR_OOS"]
        mn = min(m["CAGR_IS"], m["CAGR_OOS"])
        is_cfd = "(CFD)" in name
        # after-tax CAGR_OOS: CFD JP gains taxed 0.20315 on positive; ETF NISA non-tax assumed.
        if is_cfd:
            at = (1.0 + oos) ** 1.0  # placeholder, compute via after-tax factor below
            at = oos * (1.0 - 0.20315) if oos > 0 else oos
        else:
            at = oos  # NISA non-taxable
        rows.append((name, oos, mn, at, m["MaxDD_FULL"], m["Sharpe_OOS"]))
    # sort by CAGR_OOS desc
    for name, oos, mn, at, dd, sh in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:18s} {oos*100:>8.2f}% {mn*100:>10.2f}% {at*100:>11.2f}% {dd*100:>7.2f}% {sh:>7.3f}")
    print()
    print(f"  [ref] E4 scenarioD CAGR_OOS = {e4_scenD['CAGR_OOS']*100:.2f}%")
    print(f"  [ref] E4 realistic borrowed CAGR_OOS = {e4b['CAGR_OOS']*100:.2f}%")
    print(f"  [ref] E4 realistic full     CAGR_OOS = {e4['CAGR_OOS']*100:.2f}%")
    print()
    # also print min(IS,OOS) ranking and aftertax ranking
    print("  -- ranking by min(IS,OOS) --")
    for name, oos, mn, at, dd, sh in sorted(rows, key=lambda r: -r[2]):
        print(f"    {name:18s} min={mn*100:>7.2f}%")
    print("  -- ranking by after-tax CAGR_OOS --")
    for name, oos, mn, at, dd, sh in sorted(rows, key=lambda r: -r[3]):
        print(f"    {name:18s} aftertax={at*100:>7.2f}%")
    return rows


if __name__ == "__main__":
    r_e4 = run_e4_diag()
    r_vz = run_vz_diag()
    r_cmp = run_realistic_delay_compare()
    r_rank = run_ranking()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"V2-E4    DELAY=2 maxdiff={r_e4[2]:.3e}  DELAY=1 maxdiff={r_e4[1]:.3e}")
    print(f"V2-vz065 DELAY=2 maxdiff={r_vz[2]:.3e}  DELAY=1 maxdiff={r_vz[1]:.3e}")
    verdict2 = "DELAY=2 CORRECT (review right)" if (r_e4[2] < 1e-6 and r_e4[1] > 1e-6) else "INDETERMINATE"
    print(f"VERDICT V1/V2: {verdict2}")
