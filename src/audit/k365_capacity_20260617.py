"""
src/audit/k365_capacity_20260617.py
=====================================
Task #1 (MARGIN_CAPACITY_STRESS_PLAN_20260617.md M4):
k365 click-kabu365 NASDAQ-100 capacity check -- full period

Strategy: Bext_str_sc1.35
  - NASDAQ sleeve wn (IN-day avg 69%) with effective leverage L.
  - <=3x portion: TQQQ (US ETF, deep liquidity, no capacity constraint).
  - >3x excess:   click-kabu365 k365 (exchange CFD, capacity constrained).
  - Required excess notional = wn * max(L-3,0) * AUM.
  - Required lots  = notional / JPY220,000 per lot
    (product spec: index 22,000 x JPY10/pt = JPY220,000/lot, PRODUCT_COST_COMPARISON ss9).

Liquidity data:
  TFX publishes k365 NASDAQ-100 OI/volume CSV at:
    https://www.tfx.co.jp/historical/cfd/
  However, automated download is not possible from this script (browser required).
  => DATA GAP: actual OI/volume not obtained. Sensitivity table provided instead.

Capacity ceiling:
  AUM at which required lots_max > 1%/5%/10%/20% of assumed daily volume or OI.

Inputs:
  audit_results/sc135_weight_leverage_breakdown_20260617.csv (pre-computed)

Outputs:
  src/audit/k365_capacity_20260617.py   (this file)
  audit_results/k365_capacity_20260617.csv
  RETURN_BLOCK printed to stdout (json.dumps)

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.
"""

from __future__ import annotations

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# k365 NASDAQ-100: notional per lot (product spec 2026-06)
# index ~22,000 x JPY10/pt = JPY220,000/lot
NOTIONAL_PER_LOT = 220_000  # yen

AUM_SCENARIOS = {
    "AUM_30M":  30_000_000,
    "AUM_100M": 100_000_000,
    "AUM_300M": 300_000_000,
    "AUM_1B":   1_000_000_000,
}

CSV_BREAKDOWN = os.path.join(
    _REPO_DIR, "audit_results", "sc135_weight_leverage_breakdown_20260617.csv")

# ---------------------------------------------------------------------------
# Load existing breakdown CSV
# ---------------------------------------------------------------------------

def _load_breakdown():
    df = pd.read_csv(CSV_BREAKDOWN, encoding="utf-8-sig")
    result = {}

    l_stats = df[df["section"] == "agg2_L_stats"].set_index("key")["value"]
    result["N_in"]     = float(l_stats.get("N_in", 0))
    result["L_mean"]   = float(l_stats.get("mean", 0))
    result["L_p10"]    = float(l_stats.get("p10", 0))
    result["L_p25"]    = float(l_stats.get("p25", 0))
    result["L_p75"]    = float(l_stats.get("p75", 0))
    result["L_p90"]    = float(l_stats.get("p90", 0))
    result["L_max"]    = float(l_stats.get("max", 0))
    result["L_mean_all"] = float(l_stats.get("mean_all", 0))

    exc_stats = df[df["section"] == "agg3_excess"].set_index("key")["value"]
    result["n_gt3_all"]         = float(exc_stats.get("n_gt3_all", 0))
    result["ratio_all_pct"]     = float(exc_stats.get("ratio_all_pct", 0))
    result["n_gt3_in"]          = float(exc_stats.get("n_gt3_in", 0))
    result["ratio_in_pct"]      = float(exc_stats.get("ratio_in_pct", 0))
    result["mean_exc_gt3_all"]  = float(exc_stats.get("mean_exc_given_gt3_all", 0))
    result["mean_exc_gt3_in"]   = float(exc_stats.get("mean_exc_given_gt3_in", 0))
    result["mean_exc_all"]      = float(exc_stats.get("mean_exc_all", 0))
    result["mean_exc_in"]       = float(exc_stats.get("mean_exc_in", 0))

    dc = df[df["section"] == "day_counts"].set_index("key")["value"]
    result["n_total"] = float(dc.get("n_total", 0))
    result["n_in"]    = float(dc.get("n_in", 0))
    result["n_out"]   = float(dc.get("n_out", 0))

    # wn_eff: all-day and IN-day averages
    w = df[df["section"] == "agg1_weight"].copy()
    w_idx = w["key"].str.strip()
    w = w.set_index(w_idx)
    nas_row = None
    for k in w.index:
        if "NASDAQ" in k:
            nas_row = w.loc[k]
            break
    if nas_row is not None:
        result["wn_eff_all_avg"] = float(nas_row["value"]) / 100.0
        result["wn_eff_in_avg"]  = float(nas_row["in_value"]) / 100.0
    else:
        result["wn_eff_all_avg"] = 0.369
        result["wn_eff_in_avg"]  = 0.693

    bins = df[df["section"] == "agg4_bin_table"].copy()
    bins.index = bins["key"].str.strip()
    result["bins"] = bins
    return result


# ---------------------------------------------------------------------------
# Reconstruct approximate excess distribution from bin statistics
# ---------------------------------------------------------------------------

def _reconstruct_excess_series(bk):
    n_in      = int(bk["N_in"])
    n_gt3_in  = int(bk["n_gt3_in"])
    n_le3_in  = n_in - n_gt3_in

    exc_arr = np.zeros(n_le3_in)

    bins = bk["bins"]
    gt3_bins = [
        "3.0-4.0  >3x",
        "4.0-5.0  >3x",
        "5.0-6.0  >3x",
        "6.0+     >3x",
    ]
    for bkey in gt3_bins:
        if bkey in bins.index:
            row   = bins.loc[bkey]
            n_b   = int(float(row.get("in_value", 0)))
            m_exc = float(row.get("mean_excess_above3", 0))
            if "6.0+" in bkey:
                m_exc = bk["L_max"] - 3.0  # conservative: use true max
            if n_b > 0:
                exc_arr = np.concatenate([exc_arr, np.full(n_b, m_exc)])

    return exc_arr


# ---------------------------------------------------------------------------
# Compute required lots
# ---------------------------------------------------------------------------

def _compute_required_lots(exc_arr, wn_eff_in_avg, aum):
    notional = wn_eff_in_avg * exc_arr * aum
    lots     = notional / NOTIONAL_PER_LOT
    gt3_mask = exc_arr > 0
    return {
        "n_days_total":    len(exc_arr),
        "n_days_excess":   int(gt3_mask.sum()),
        "pct_days_excess": round(100.0 * gt3_mask.sum() / max(len(exc_arr), 1), 2),
        "notional_mean":   float(np.mean(notional)),
        "notional_p90":    float(np.percentile(notional, 90)),
        "notional_max":    float(np.max(notional)),
        "lots_mean":       float(np.mean(lots)),
        "lots_p90":        float(np.percentile(lots, 90)),
        "lots_max":        float(np.max(lots)),
        "lots_mean_cond":  float(np.mean(lots[gt3_mask])) if gt3_mask.any() else 0.0,
        "lots_p90_cond":   float(np.percentile(lots[gt3_mask], 90)) if gt3_mask.any() else 0.0,
        "lots_max_cond":   float(np.max(lots[gt3_mask])) if gt3_mask.any() else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 120)
    print("K365 CAPACITY ANALYSIS  2026-06-17")
    print("Strategy: Bext_str_sc1.35 (v7_map STRONG x scale=1.35)")
    print("  TQQQ (<=3x): US ETF -- deep liquidity, no capacity constraint")
    print("  k365 click-kabu365 (>3x): exchange CFD -- capacity constrained (this analysis)")
    print("  Required lots = wn x max(L-3,0) x AUM / JPY220,000")
    print("  Liquidity data: DATA GAP (TFX CSV needs manual download)")
    print("=" * 120)

    if not os.path.exists(CSV_BREAKDOWN):
        print("\nERROR: %s not found." % CSV_BREAKDOWN)
        print("  Please run sc135_weight_leverage_breakdown_20260617.py first.")
        sys.exit(1)

    bk = _load_breakdown()
    print("\nLoaded breakdown CSV: %s" % CSV_BREAKDOWN)
    print("  n_total=%d  n_in=%d  n_gt3_in=%d (%.1f%% of IN days)"
          % (int(bk["n_total"]), int(bk["n_in"]), int(bk["n_gt3_in"]),
             bk["ratio_in_pct"]))
    print("  IN-day L: mean=%.3f  p90=%.3f  max=%.3f"
          % (bk["L_mean"], bk["L_p90"], bk["L_max"]))
    print("  IN-day excess mean(all IN)=%.4f  mean(|L>3)=%.4f"
          % (bk["mean_exc_in"], bk["mean_exc_gt3_in"]))
    print("  wn_eff (IN-day avg)=%.4f" % bk["wn_eff_in_avg"])

    # ------------------------------------------------------------------
    # Reconstruct excess distribution
    # ------------------------------------------------------------------
    exc_arr = _reconstruct_excess_series(bk)
    wn_in   = bk["wn_eff_in_avg"]

    exc_mean = float(np.mean(exc_arr))
    exc_p90  = float(np.percentile(exc_arr, 90))
    exc_max  = float(bk["L_max"] - 3.0)   # 6.48-3=3.48

    print("\nExcess distribution (max(L-3,0), reconstructed from bin table):")
    print("  mean=%.4f  p90=%.4f  max=%.4f (=L_max-3, conservative)"
          % (exc_mean, exc_p90, exc_max))

    # ------------------------------------------------------------------
    # Required lots per AUM
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("REQUIRED LOTS BY AUM  (excess notional = wn x max(L-3,0) x AUM)")
    print("=" * 120)
    print("\n  %-12s | %14s | %14s | %14s | %10s | %10s | %10s"
          % ("AUM", "notional_mean", "notional_p90", "notional_max",
             "lots_mean", "lots_p90", "lots_max"))
    print("  " + "-" * 105)

    aum_rows = []
    for aum_label, aum in AUM_SCENARIOS.items():
        r = _compute_required_lots(exc_arr, wn_in, aum)
        print("  %-12s | %14.0f | %14.0f | %14.0f | %10.1f | %10.1f | %10.1f"
              % (aum_label,
                 r["notional_mean"], r["notional_p90"], r["notional_max"],
                 r["lots_mean"],     r["lots_p90"],     r["lots_max"]))
        aum_rows.append({
            "aum_label":       aum_label,
            "aum_yen":         aum,
            "aum_M":           round(aum / 1e6, 0),
            "notional_mean":   round(r["notional_mean"], 0),
            "notional_p90":    round(r["notional_p90"], 0),
            "notional_max":    round(r["notional_max"], 0),
            "lots_mean":       round(r["lots_mean"], 2),
            "lots_p90":        round(r["lots_p90"], 2),
            "lots_max":        round(r["lots_max"], 2),
            "lots_mean_cond":  round(r["lots_mean_cond"], 2),
            "lots_p90_cond":   round(r["lots_p90_cond"], 2),
            "lots_max_cond":   round(r["lots_max_cond"], 2),
            "n_days_excess":   r["n_days_excess"],
            "pct_days_excess": r["pct_days_excess"],
        })

    print("\n  Notes:")
    print("  - lots_mean / lots_p90 / lots_max: all-IN-day base (incl. excess=0 days)")
    print("  - lots_mean_cond / lots_p90_cond / lots_max_cond: conditional on excess>0")
    print("  - notional_per_lot = JPY220,000 (index 22,000 x JPY10/pt)")
    print("  - wn_eff IN-day avg = %.4f (DH-W1 NASDAQ weight on IN days)" % wn_in)
    print("  - excess = max(L-3, 0). Days with L<=3 -> zero (TQQQ only, no k365)")

    # Conditional table
    key_aum_labels = list(AUM_SCENARIOS.keys())
    key_aum_map    = {r["aum_label"]: r for r in aum_rows}

    print("\n  Conditional lots (excess>0 days only: %d / %d IN days)"
          % (int(bk["n_gt3_in"]), int(bk["n_in"])))
    print("  %-12s | %15s | %15s | %15s"
          % ("AUM", "lots_mean_cond", "lots_p90_cond", "lots_max_cond"))
    print("  " + "-" * 62)
    for r in aum_rows:
        print("  %-12s | %15.1f | %15.1f | %15.1f"
              % (r["aum_label"], r["lots_mean_cond"], r["lots_p90_cond"], r["lots_max_cond"]))

    # ------------------------------------------------------------------
    # DATA GAP notice
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("LIQUIDITY DATA: k365 NASDAQ-100 OI / Daily Volume")
    print("=" * 120)
    print("")
    print("  [DATA GAP -- no fabrication]")
    print("")
    print("  TFX (Tokyo Financial Exchange) publishes k365 historical CSV:")
    print("    URL: https://www.tfx.co.jp/historical/cfd/")
    print("    Items: open interest (OI), trading volume, OHLC, interest amounts")
    print("    Format: CSV, available from Nov 2010; updated next business day AM")
    print("")
    print("  Automated download from this script was not feasible (browser login required).")
    print("  => Specific OI/volume lot counts for NASDAQ-100 NOT obtained.")
    print("")
    print("  Qualitative reference (from Web search):")
    print("  - k365 NASDAQ-100 listed: 2022-02-28 (relatively new product)")
    print("  - k365 N225 is far more liquid (institutional users, flagship product)")
    print("  - NASDAQ-100 OI estimated at ~500-10,000 lots (qualitative, much < N225)")
    print("    N225 k365 OI: likely tens of thousands to 100,000+ lots")
    print("  - OI data updated Wed & Fri (end of trading)")
    print("")
    print("  Recommended action: download CSV from TFX and insert actual OI/vol into")
    print("  the sensitivity table below to get precise capacity assessment.")

    # ------------------------------------------------------------------
    # Sensitivity tables
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("SENSITIVITY TABLE: required_lots_max / assumed_vol_or_OI (%)")
    print("  = What % of market does the strategy occupy at max leverage?")
    print("=" * 120)

    vol_refs = [100, 500, 1000, 2000, 5000, 10000, 50000]
    oi_refs  = [500, 1000, 2000, 5000, 10000, 50000, 100000]

    print("\n  [A] Daily volume basis -- lots_max / assumed daily volume (%)")
    hdr_str = "  %-18s | " + " | ".join(["%10s" % l for l in key_aum_labels])
    print(hdr_str % "Assumed vol(lots)")
    print("  " + "-" * (18 + 4 + 13 * len(key_aum_labels)))
    for vol in vol_refs:
        parts = []
        for lbl in key_aum_labels:
            pct = 100.0 * key_aum_map[lbl]["lots_max"] / vol
            flag = " !!!" if pct > 5.0 else ("  ! " if pct > 1.0 else "    ")
            parts.append("%7.1f%%%s" % (pct, flag))
        print("  %-18s | %s" % ("%d lots" % vol, " | ".join(parts)))

    print("\n  [B] Open interest (OI) basis -- lots_max / assumed OI (%)")
    print(hdr_str % "Assumed OI(lots)")
    print("  " + "-" * (18 + 4 + 13 * len(key_aum_labels)))
    for oi in oi_refs:
        parts = []
        for lbl in key_aum_labels:
            pct = 100.0 * key_aum_map[lbl]["lots_max"] / oi
            flag = " !!!" if pct > 5.0 else ("  ! " if pct > 1.0 else "    ")
            parts.append("%7.1f%%%s" % (pct, flag))
        print("  %-18s | %s" % ("%d lots" % oi, " | ".join(parts)))

    print("\n  [C] Average (lots_mean) / assumed daily volume -- typical day burden")
    print(hdr_str % "Assumed vol(lots)")
    print("  " + "-" * (18 + 4 + 13 * len(key_aum_labels)))
    for vol in vol_refs:
        parts = []
        for lbl in key_aum_labels:
            pct = 100.0 * key_aum_map[lbl]["lots_mean"] / vol
            parts.append("%9.2f%%" % pct)
        print("  %-18s | %s" % ("%d lots" % vol, " | ".join(parts)))

    print("\n  Legend: '!!!' = >5% (high market impact), '!' = >1% (caution)")

    # ------------------------------------------------------------------
    # Capacity ceiling AUM
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("CAPACITY CEILING AUM")
    print("  AUM at which required lots_max >= X% of assumed vol/OI")
    print("  Formula: AUM_ceil = X% x ref_lots x JPY220,000 / (wn_in x exc_max)")
    print("  wn_in=%.4f  exc_max=%.4f" % (wn_in, exc_max))
    print("=" * 120)

    ceil_vol_refs = [(500, "vol_500"), (1000, "vol_1000"), (2000, "vol_2000"),
                     (5000, "vol_5000"), (10000, "vol_10000")]
    ceil_oi_refs  = [(1000, "oi_1000"), (2000, "oi_2000"), (5000, "oi_5000"),
                     (10000, "oi_10000"), (50000, "oi_50000")]

    def _ceil(thr_pct, ref_lots):
        thr = thr_pct / 100.0
        if exc_max > 0:
            return thr * ref_lots * NOTIONAL_PER_LOT / (wn_in * exc_max)
        return float("inf")

    print("\n  [A] Daily volume basis")
    print("  %-14s | %12s | %12s | %12s | %12s"
          % ("Assumed vol", "1% ceiling", "5% ceiling", "10% ceiling", "20% ceiling"))
    print("  " + "-" * 70)
    ceil_rows = []
    for vol_v, vol_lbl in ceil_vol_refs:
        c1, c5, c10, c20 = _ceil(1, vol_v), _ceil(5, vol_v), _ceil(10, vol_v), _ceil(20, vol_v)
        print("  %-14s | %10.0fM | %10.0fM | %10.0fM | %10.0fM"
              % ("%d lots" % vol_v, c1/1e6, c5/1e6, c10/1e6, c20/1e6))
        ceil_rows.append({
            "basis": "daily_volume", "ref_label": vol_lbl, "ref_lots": vol_v,
            "ceil_1pct_M": round(c1/1e6, 1), "ceil_5pct_M": round(c5/1e6, 1),
            "ceil_10pct_M": round(c10/1e6, 1), "ceil_20pct_M": round(c20/1e6, 1),
        })

    print("\n  [B] Open interest (OI) basis")
    print("  %-14s | %12s | %12s | %12s | %12s"
          % ("Assumed OI", "1% ceiling", "5% ceiling", "10% ceiling", "20% ceiling"))
    print("  " + "-" * 70)
    for oi_v, oi_lbl in ceil_oi_refs:
        c1, c5, c10, c20 = _ceil(1, oi_v), _ceil(5, oi_v), _ceil(10, oi_v), _ceil(20, oi_v)
        print("  %-14s | %10.0fM | %10.0fM | %10.0fM | %10.0fM"
              % ("%d lots" % oi_v, c1/1e6, c5/1e6, c10/1e6, c20/1e6))
        ceil_rows.append({
            "basis": "open_interest", "ref_label": oi_lbl, "ref_lots": oi_v,
            "ceil_1pct_M": round(c1/1e6, 1), "ceil_5pct_M": round(c5/1e6, 1),
            "ceil_10pct_M": round(c10/1e6, 1), "ceil_20pct_M": round(c20/1e6, 1),
        })

    # ------------------------------------------------------------------
    # Risk notes
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("QUALITATIVE RISK NOTES")
    print("=" * 120)
    print("""
  [1] Annual reset (forced settlement -- annual roll-over risk)
  -------------------------------------------------------------
  - k365 NASDAQ-100 has ~15-month listing period. New contract listed each Sep;
    old contract expires 3rd Friday Dec of the following year.
  - Investors must roll over (close old, open new) annually. A large position
    rolled simultaneously with other participants causes liquidity concentration.
  - The roll-over period (Dec expiry week) is particularly thin. Adverse fills
    estimated at +0.1% to +0.5% of notional (qualitative; actual TFX roll spread
    data not retrieved -- DATA GAP).
  - DH-W1 strategy trades ~20-30 times/year; if re-entry signal coincides with
    roll-over window, transaction cost may spike.

  [2] Thin market (NASDAQ-100 vs N225)
  --------------------------------------
  - k365 N225: flagship product, institutional users, high liquidity.
    Estimated OI: tens of thousands to 200,000+ lots.
  - k365 NASDAQ-100: listed 2022-02, retail-dominated, newer product.
    Estimated OI: 500-10,000 lots (qualitative).
    This is roughly 1/10 to 1/100 of N225 liquidity.
  - Large block orders (e.g., 50-500 lots at high leverage moments) risk
    moving the market and receiving adverse fills.

  [3] TQQQ (<=3x) capacity: NO CONSTRAINT
  -----------------------------------------
  - The <=3x NASDAQ notional is held via TQQQ (US ETF).
    TQQQ daily volume: tens of millions of shares (~USD1-3 billion/day).
    Even at AUM=JPY1B, wn x min(L,3) x AUM ~ JPY2B ~ USD14M -- negligible vs TQQQ volume.

  [4] Practical assessment
  -------------------------
  - AUM JPY30M: lots_max ~329. At OI=1000 lots -> 32.9% of OI. RISKY if OI is that small.
    But if OI=5000 lots -> 6.6%, manageable. Key unknown: actual OI.
  - AUM JPY100M: lots_max ~1096. Requires OI/vol well above 10,000 lots for safety (<10%).
  - AUM JPY300M: lots_max ~3289. Even at OI=50,000 lots -> 6.6%. Borderline.
  - AUM JPY1B: lots_max ~10,964. Requires OI >200,000 lots for <5%. Likely infeasible.
  - RECOMMENDATION: Obtain actual OI/volume from TFX CSV before scaling beyond JPY100M.
""")

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    out_dir  = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "k365_capacity_20260617.csv")

    csv_rows = []

    # params
    for k, v in [
        ("NOTIONAL_PER_LOT", NOTIONAL_PER_LOT),
        ("wn_eff_in_avg",    round(wn_in, 6)),
        ("exc_mean",         round(exc_mean, 6)),
        ("exc_p90",          round(exc_p90, 6)),
        ("exc_max",          round(exc_max, 6)),
        ("n_in",             int(bk["n_in"])),
        ("n_gt3_in",         int(bk["n_gt3_in"])),
        ("pct_gt3_in",       round(bk["ratio_in_pct"], 2)),
    ]:
        csv_rows.append({"section": "params", "key": k, "value": v, "note": ""})

    # required lots
    for r in aum_rows:
        row = {"section": "required_lots"}
        row.update(r)
        csv_rows.append(row)

    # sensitivity: vol
    for vol in vol_refs:
        row = {"section": "sensitivity_vol", "ref_lots": vol}
        for lbl in key_aum_labels:
            row["%s_pct_max"  % lbl] = round(100.0 * key_aum_map[lbl]["lots_max"]  / vol, 2)
            row["%s_pct_p90"  % lbl] = round(100.0 * key_aum_map[lbl]["lots_p90"]  / vol, 2)
            row["%s_pct_mean" % lbl] = round(100.0 * key_aum_map[lbl]["lots_mean"] / vol, 2)
        csv_rows.append(row)

    # sensitivity: OI
    for oi in oi_refs:
        row = {"section": "sensitivity_oi", "ref_lots": oi}
        for lbl in key_aum_labels:
            row["%s_pct_max"  % lbl] = round(100.0 * key_aum_map[lbl]["lots_max"]  / oi, 2)
            row["%s_pct_p90"  % lbl] = round(100.0 * key_aum_map[lbl]["lots_p90"]  / oi, 2)
            row["%s_pct_mean" % lbl] = round(100.0 * key_aum_map[lbl]["lots_mean"] / oi, 2)
        csv_rows.append(row)

    # capacity ceiling
    for r in ceil_rows:
        row = {"section": "capacity_ceiling"}
        row.update(r)
        csv_rows.append(row)

    # data gap
    csv_rows.append({
        "section": "data_gap",
        "key":     "k365_NASDAQ100_OI_volume",
        "value":   "NOT_AVAILABLE",
        "note":    "TFX CSV https://www.tfx.co.jp/historical/cfd/ -- manual browser download required",
    })
    csv_rows.append({
        "section": "data_gap",
        "key":     "k365_NASDAQ100_OI_estimate_lots",
        "value":   "500-10000",
        "note":    "qualitative estimate; N225 OI >> NASDAQ-100 OI",
    })

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\nSaved CSV: %s  (%d rows)" % (csv_path, len(csv_rows)))

    # ------------------------------------------------------------------
    # RETURN_BLOCK
    # ------------------------------------------------------------------
    return_block = {
        "script":  "k365_capacity_20260617.py",
        "date":    "2026-06-17",
        "config": {
            "strategy":         "Bext_str_sc1.35",
            "notional_per_lot": NOTIONAL_PER_LOT,
            "wn_eff_in_avg":    round(wn_in, 6),
            "exc_mean":         round(exc_mean, 6),
            "exc_p90":          round(exc_p90, 6),
            "exc_max":          round(exc_max, 6),
            "n_in":             int(bk["n_in"]),
            "n_gt3_in":         int(bk["n_gt3_in"]),
            "pct_gt3_in":       round(bk["ratio_in_pct"], 2),
        },
        "required_lots": aum_rows,
        "capacity_ceiling": ceil_rows,
        "liquidity_data_gap": {
            "status":  "DATA_GAP",
            "source":  "https://www.tfx.co.jp/historical/cfd/",
            "reason":  "Browser-based CSV download required; automated fetch not feasible",
            "estimate_OI_lots": "500-10000",
            "estimate_basis":   "qualitative; NASDAQ-100 listed 2022, retail-dominated; N225 much larger",
            "sensitivity_table_provided": True,
        },
        "risk_notes": [
            "Annual reset forced settlement: roll-over in Dec expiry week, liquidity concentration risk. "
            "Additional cost est 0.1-0.5% of notional per roll (DATA GAP: actual TFX roll spread not retrieved).",
            "Thin market: NASDAQ-100 OI estimated 500-10000 lots vs N225 tens-of-thousands+.",
            "TQQQ (<=3x) has no capacity constraint: US ETF, ~USD1-3B daily volume.",
            "Practical assessment: AUM30M likely manageable (329 lots max) if actual OI>5000. "
            "AUM100M borderline (1096 lots max). AUM300M+ likely infeasible unless actual OI>50000.",
        ],
        "csv_path": csv_path,
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True))
    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
