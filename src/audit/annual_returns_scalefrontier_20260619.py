"""
src/audit/annual_returns_scalefrontier_20260619.py
===================================================
5系列の暦年（暦年）リターン算出スクリプト。

対象5系列:
  1. scale1.35強map (Bext_str_sc1.35)
  2. scale1.40強map (Bext_str_sc1.40)
  3. scale1.50強map (Bext_str_sc1.50)
  4. scale1.60強map (Bext_str_sc1.60)
  5. NASDAQ-100 1倍 Buy&Hold（ベンチマーク）

算出方法:
  - 1〜4の戦略NAVは _build_full_c1 (k365_recost_20260612) で構築。
    v7_map=STRONG_MAP={0:1.60,1:1.50,2:1.10,3:1.00}、各lev_scale。
    全コスト（TQQQ borrow・k365 EXCESS_EXTRA=0.0025・TER・売買）込み。
  - 暦年リターン = _calendar_year_returns (run_p01_backtest_20260611) 使用。
  - 戦略NAVはコスト後・譲渡益税前（AFTER_TAX=0.8273 適用なし）。
    全系列を同一基準（コスト後・税前）で揃える。
  - NASDAQ B&H: NASDAQ_extended_to_2026.csv の Close から。
  - 期間: 全系列で揃う最初の暦年から 2025 まで（2026年は未完のため除外）。

出力:
  audit_results/annual_returns_scalefrontier_20260619.csv
  Markdown表は stdout に出力。

ASCII-only prints (Windows cp932). No git. No temp files.
"""

from __future__ import annotations

import os
import sys
import types

# ---- multitasking stub ----
if "multitasking" not in sys.modules:
    _m = types.ModuleType("multitasking")
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules["multitasking"] = _m

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

import src.audit.strategy_runners as sr
from src.audit.k365_recost_20260612 import (
    _build_full_c1,
    EXCESS_EXTRA_K365_CENTRE,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
    FEE_GOLD, FEE_BOND,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
B3A_MAP_STRONG = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025

SWEEP_SCALES = [1.35, 1.40, 1.50, 1.60]

# Sanity expected values (from leverext_high_20260618.py / unified_metrics)
# Note: +26.77% is CAGR_IS after-tax (IS period = before 2021-05-07)
# Full-period pretax CAGR will be ~+32% (52 yrs); min9 aftertax = +23.83%
SANITY_EXPECT = {
    1.35: {
        "min9_aftertax": 0.2383,   # min(IS,OOS) after-tax CAGR = +23.83%
        "maxdd": -0.4504,           # MaxDD = -45.04%
        "tol_min9": 0.0010,         # +/-0.10pp
        "tol_maxdd": 0.0010,        # +/-0.10pp
    },
}

NASDAQ_CSV_PATH = os.path.join(_REPO_DIR, "NASDAQ_extended_to_2026.csv")


def _nav_to_calendar_year_returns(nav_dt: pd.Series) -> pd.Series:
    """暦年リターン: 各年の年初→年末NAV変化率。
    _calendar_year_returns と同じロジック。
    """
    return _calendar_year_returns(nav_dt)


def _cagr_from_nav(nav_dt: pd.Series) -> float:
    """全期間CAGR (税前・複利)"""
    start = nav_dt.iloc[0]
    end = nav_dt.iloc[-1]
    n_years = (nav_dt.index[-1] - nav_dt.index[0]).days / 365.25
    return (end / start) ** (1.0 / n_years) - 1.0


def _load_nasdaq_bh() -> pd.Series:
    """NASDAQ-100指数の1倍B&Hの暦年リターン。
    NASDAQ_extended_to_2026.csvのClose列から算出。
    """
    df = pd.read_csv(NASDAQ_CSV_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    close = df["Close"].dropna()
    # 暦年last価格で年次リターン算出
    yearly = close.resample("YE").last()
    ann = yearly.pct_change().dropna()
    # 2026年は未完のため除外
    ann = ann[ann.index.year <= 2025]
    return ann


def main():
    print("=" * 100)
    print("ANNUAL RETURNS: Scale Frontier (Strong Map)  2026-06-19")
    print("v7_map STRONG: {0:1.60, 1:1.50, 2:1.10, 3:1.00}")
    print("lev_scale in {1.35, 1.40, 1.50, 1.60}")
    print("Cost: k365 EXCESS_EXTRA=0.0025  All costs included  Pre-tax NAV")
    print("Benchmark: NASDAQ-100 1x Buy&Hold (NASDAQ_extended_to_2026.csv)")
    print("=" * 100)

    # ---- Load shared data ----
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)
    print("Data range: %s to %s  (%.1f years)" % (
        dates_dt[0].strftime("%Y-%m-%d"),
        dates_dt[-1].strftime("%Y-%m-%d"),
        n_years))

    # ---- Gold/Bond auxiliary series ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask_arr = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    wg_iv, wb_iv = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    # =========================================================================
    # Build NAV for each scale (Strong map)
    # =========================================================================
    print("\nBuilding NAVs for %d scales ..." % len(SWEEP_SCALES))
    nav_by_scale = {}
    for sc in SWEEP_SCALES:
        lbl = "Bext_str_sc%.2f" % sc
        print("  [%s] scale=%.2f ..." % (lbl, sc))
        nav_dt, r, tpy, exc = _build_full_c1(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
            v7_map=B3A_MAP_STRONG, lev_scale=sc, excess_extra=EXCESS_EXTRA)
        nav_by_scale[sc] = {"nav": nav_dt, "r": r, "tpy": tpy, "exc": exc}
        cagr = _cagr_from_nav(nav_dt)
        print("    Full CAGR (pretax): %+.2f%%  tpy=%.1f  exc_days=%d" % (
            cagr * 100, tpy, exc))

    # =========================================================================
    # Sanity check: scale=1.35 min9 aftertax = +23.83%, MaxDD = -45.04%
    # (from leverext_high_20260618.py / leverext_scale_20260616.py sanity gates)
    # =========================================================================
    print("\n--- SANITY CHECK (scale=1.35 strong map) ---")
    sc135 = nav_by_scale[1.35]
    # Import compute_10metrics and _apply_aftertax to get min9
    from src.audit.unified_metrics import compute_10metrics
    sc135_pre = compute_10metrics(sc135["nav"], sc135["tpy"])
    sc135_aft = _apply_aftertax(sc135_pre)
    min9_got   = min(sc135_aft["CAGR_IS"], sc135_aft["CAGR_OOS"])
    maxdd_got  = sc135_pre["MaxDD_FULL"]
    expect_min9  = SANITY_EXPECT[1.35]["min9_aftertax"]
    expect_maxdd = SANITY_EXPECT[1.35]["maxdd"]
    tol_min9   = SANITY_EXPECT[1.35]["tol_min9"]
    tol_maxdd  = SANITY_EXPECT[1.35]["tol_maxdd"]
    ok_min9  = abs(min9_got - expect_min9)  <= tol_min9
    ok_maxdd = abs(maxdd_got - expect_maxdd) <= tol_maxdd
    sanity_ok = ok_min9 and ok_maxdd
    print("  min9 aftertax: got %+.4f%%  expect %+.4f%%+/-%.2fpp -> %s" % (
        min9_got * 100, expect_min9 * 100, tol_min9 * 100, "OK" if ok_min9 else "FAIL"))
    print("  MaxDD:         got %+.4f%%  expect %+.4f%%+/-%.2fpp -> %s" % (
        maxdd_got * 100, expect_maxdd * 100, tol_maxdd * 100, "OK" if ok_maxdd else "FAIL"))
    print("  SANITY: %s" % ("PASS" if sanity_ok else "FAIL -- check NAV construction"))
    # Also report full-period pretax CAGR for reference
    cagr_135 = _cagr_from_nav(sc135["nav"])

    # =========================================================================
    # Calendar-year returns for 4 strategy series
    # =========================================================================
    print("\nComputing calendar-year returns ...")
    cy_by_scale = {}
    for sc in SWEEP_SCALES:
        nav_dt = nav_by_scale[sc]["nav"]
        cy = _nav_to_calendar_year_returns(nav_dt)
        # 2026年は未完のため除外
        cy = cy[cy.index <= 2025]
        cy_by_scale[sc] = cy

    # =========================================================================
    # NASDAQ 1x B&H calendar-year returns
    # =========================================================================
    print("Loading NASDAQ 1x B&H annual returns ...")
    cy_ndx = _load_nasdaq_bh()
    print("  NASDAQ B&H range: %d-%d  (%d years)" % (
        int(cy_ndx.index[0].year), int(cy_ndx.index[-1].year), len(cy_ndx)))

    # Re-index cy_ndx by integer year for easier lookup
    cy_ndx_by_year = pd.Series(cy_ndx.values, index=cy_ndx.index.year)

    # Sanity: check representative years for NASDAQ B&H
    print("\n--- SANITY CHECK: NASDAQ 1x B&H representative years ---")
    check_years = {2000: (-0.44, -0.30), 2008: (-0.47, -0.32),
                   2020: (0.40, 0.55), 2022: (-0.40, -0.25)}
    for yr, (lo, hi) in check_years.items():
        if yr in cy_ndx_by_year.index:
            val = cy_ndx_by_year.loc[yr]
            ok = lo <= val <= hi
            print("  %d: got %+.1f%%  expected [%+.1f%%, %+.1f%%] -> %s" % (
                yr, val * 100, lo * 100, hi * 100, "OK" if ok else "WARN"))
        else:
            print("  %d: not in data" % yr)

    # =========================================================================
    # Align all series to common year range
    # =========================================================================
    # Re-index cy_ndx_by_year was already built above
    # Find the start year: max of all series' first year (integer years)
    all_first = [int(cy_by_scale[sc].index[0]) for sc in SWEEP_SCALES]
    all_first.append(int(cy_ndx_by_year.index[0]))
    start_year = max(all_first)

    # End year: min of last year across all series
    all_last = [int(cy_by_scale[sc].index[-1]) for sc in SWEEP_SCALES]
    all_last.append(int(cy_ndx_by_year.index[-1]))
    end_year = min(all_last)

    print("\nCommon year range: %d-%d" % (start_year, end_year))
    years = list(range(start_year, end_year + 1))

    # Align strategy series (integer index)
    for sc in SWEEP_SCALES:
        cy_by_scale[sc] = cy_by_scale[sc][
            (cy_by_scale[sc].index >= start_year) &
            (cy_by_scale[sc].index <= end_year)]
    # Align NASDAQ series (integer year index)
    cy_ndx_aligned = cy_ndx_by_year[
        (cy_ndx_by_year.index >= start_year) &
        (cy_ndx_by_year.index <= end_year)]

    # =========================================================================
    # Aggregate statistics
    # =========================================================================
    def _cagr_from_annual(cy: pd.Series) -> float:
        return (np.prod(1 + cy.values) ** (1.0 / len(cy))) - 1.0

    def _max_year(cy: pd.Series):
        idx = cy.idxmax()
        return idx, cy.loc[idx]

    def _min_year(cy: pd.Series):
        idx = cy.idxmin()
        return idx, cy.loc[idx]

    print("\n=== AGGREGATE STATISTICS (pre-tax, cost-included) ===")
    print("%-22s | %8s | %12s | %12s" % ("series", "CAGR%", "MaxYear(Ret%)", "MinYear(Ret%)"))
    print("-" * 65)

    agg = {}
    for sc in SWEEP_SCALES:
        cy = cy_by_scale[sc]
        cagr = _cagr_from_annual(cy)
        mx_yr, mx_val = _max_year(cy)
        mn_yr, mn_val = _min_year(cy)
        agg[sc] = {"cagr": cagr, "max_yr": mx_yr, "max_val": mx_val,
                   "min_yr": mn_yr, "min_val": mn_val, "cy": cy}
        lbl = "sc%.2f_strong" % sc
        print("%-22s | %+7.2f%% | %d(%+.1f%%) | %d(%+.1f%%)" % (
            lbl, cagr * 100, mx_yr, mx_val * 100, mn_yr, mn_val * 100))

    # NASDAQ B&H
    cagr_ndx = _cagr_from_annual(cy_ndx_aligned)
    mx_yr_ndx, mx_val_ndx = _max_year(cy_ndx_aligned)
    mn_yr_ndx, mn_val_ndx = _min_year(cy_ndx_aligned)
    agg["ndx"] = {"cagr": cagr_ndx, "max_yr": mx_yr_ndx, "max_val": mx_val_ndx,
                  "min_yr": mn_yr_ndx, "min_val": mn_val_ndx, "cy": cy_ndx_aligned}
    print("%-22s | %+7.2f%% | %d(%+.1f%%) | %d(%+.1f%%)" % (
        "NASDAQ_1x_BH", cagr_ndx * 100, mx_yr_ndx, mx_val_ndx * 100,
        mn_yr_ndx, mn_val_ndx * 100))

    # =========================================================================
    # Build combined DataFrame
    # =========================================================================
    df_out = pd.DataFrame(index=years)
    df_out.index.name = "year"

    for sc in SWEEP_SCALES:
        col = "sc%.2f_strong" % sc
        cy = cy_by_scale[sc]
        for yr in years:
            if yr in cy.index:
                df_out.loc[yr, col] = cy.loc[yr]
            else:
                df_out.loc[yr, col] = float("nan")

    # NASDAQ B&H
    for yr in years:
        if yr in cy_ndx_aligned.index:
            df_out.loc[yr, "NASDAQ_1x_BH"] = cy_ndx_aligned.loc[yr]
        else:
            df_out.loc[yr, "NASDAQ_1x_BH"] = float("nan")

    # =========================================================================
    # Print Markdown table
    # =========================================================================
    cols = ["sc1.35_strong", "sc1.40_strong", "sc1.50_strong", "sc1.60_strong", "NASDAQ_1x_BH"]
    hdr_disp = ["scale1.35", "scale1.40", "scale1.50", "scale1.60", "NASDAQ 1x B&H"]

    print("\n")
    print("=" * 110)
    print("ANNUAL RETURNS TABLE  (pre-tax, cost-included; %+.1f format)")
    print("=" * 110)

    # Header
    hdr = "| Year |"
    for h in hdr_disp:
        hdr += " %-12s |" % h
    print(hdr)
    sep = "|------|"
    for _ in hdr_disp:
        sep += "--------------|"
    print(sep)

    for yr in years:
        row = "| %d |" % yr
        for col in cols:
            val = df_out.loc[yr, col] if col in df_out.columns else float("nan")
            if pd.isna(val):
                row += " %12s |" % "N/A"
            else:
                row += " %+11.1f%% |" % (val * 100)
        print(row)

    # =========================================================================
    # CSV output
    # =========================================================================
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "annual_returns_scalefrontier_20260619.csv")

    # Format as percentage strings for readability (also keep raw float)
    df_csv = df_out.copy() * 100  # convert to %
    df_csv.columns = [c + "_pct" for c in df_csv.columns]
    df_csv.to_csv(csv_path, float_format="%.2f", encoding="utf-8-sig")
    print("\nSaved CSV: %s  (%d rows)" % (csv_path, len(df_csv)))

    # =========================================================================
    # Final sanity summary
    # =========================================================================
    print("\n=== SANITY SUMMARY ===")
    print("1. scale=1.35 strong map:")
    print("   min9 aftertax: %+.2f%%  (expect %+.2f%%) -> %s" % (
        min9_got * 100, expect_min9 * 100, "OK" if ok_min9 else "FAIL"))
    print("   MaxDD:         %+.2f%%  (expect %+.2f%%) -> %s" % (
        maxdd_got * 100, expect_maxdd * 100, "OK" if ok_maxdd else "FAIL"))
    print("   Full-period CAGR pretax (1975-2025): %+.2f%% [reference only]" % (
        cagr_135 * 100))

    print("2. NASDAQ 1x B&H representative year checks:")
    for yr, (lo, hi) in check_years.items():
        if yr in cy_ndx_aligned.index:
            val = cy_ndx_aligned.loc[yr]
            ok = lo <= val <= hi
            print("   %d: %+.1f%%  -> %s" % (yr, val * 100, "OK" if ok else "WARN"))
        else:
            print("   %d: not in aligned range" % yr)

    # =========================================================================
    # Footnote
    # =========================================================================
    print("\n--- FOOTNOTE (コピー用) ---")
    print("- 基準: 戦略NAVはコスト後・譲渡益税前（AFTER_TAX=0.8273の年次スケーリング未適用）。")
    print("  全系列を同一基準（コスト後・税前）で算出。")
    print("- 開始年: %d（strategy_runners mom63/DH-W1ウォームアップ後の最初の完全暦年）。" % start_year)
    print("- 終了年: %d（2026年は未完のため除外）。" % end_year)
    print("- コスト前提: TQQQ borrow = SOFR + SWAP_SPREAD(50bps); TER_TQQQ; ")
    print("  >3x超過分 k365コスト EXCESS_EXTRA=0.25%%/yr (K365_SPREAD=75bps - TQQQ swap 50bps)。")
    print("  売買コスト: us_etf_trade_cost_annual(Trades/yr)/252 適用済み。")
    print("  C1 OUTフィル: inverse-vol 63日加重 Gold/Bond ブレンド（TER込み）。")
    print("- NASDAQ 1x B&H: NASDAQ_extended_to_2026.csv Close, レバ無・コスト無・タイミング無。")
    print("- 注意: scale=1.60はStage-0でMaxDD=-51.95%%により硬否決（参考値として掲載）。")
    print("- 戦略NAV 実績値。推定・外挿なし。")

    print("\nDone.")
    return {
        "csv_path": csv_path,
        "start_year": start_year,
        "end_year": end_year,
        "sanity_cagr_135_pct": round(cagr_135 * 100, 2),
        "sanity_ok": sanity_ok,
        "agg": {
            k: {
                "cagr_pct": round(v["cagr"] * 100, 2),
                "max_year": int(v["max_yr"]),
                "max_pct": round(v["max_val"] * 100, 1),
                "min_year": int(v["min_yr"]),
                "min_pct": round(v["min_val"] * 100, 1),
            }
            for k, v in agg.items()
        },
    }


if __name__ == "__main__":
    main()
