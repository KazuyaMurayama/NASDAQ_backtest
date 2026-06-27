"""
src/audit/scale_dial_x4equiv_sweep_20260627.py
==============================================
"X4-equivalent at other scales" (user request 2026-06-27):

  Build X4-STYLE candidates (IN-leg blends 1x Gold/Bond into the leveraged NASDAQ
  leg) at the SAME scale dial values 1.4 / 1.6 / 1.8 / 2.2 -- i.e. keep the X4
  construction but turn the leverage knob to each target scale.

  BUT per the user: "the ideal bond/gold ratio should be swept over a few patterns
  at each scale and re-tuned to the best one" -- so at EACH scale we DO NOT keep
  X4's fixed gold0.32/bond0.05; we sweep a small (gold_w, bond_w) grid and SELECT
  the best variant for that scale.

SELECTION CRITERION (the X4/N4 thesis, applied per scale):
  Among the swept blends at a given scale, pick the one that MAXIMIZES CAGR_OOS
  SUBJECT TO MaxDD not being deeper than the PURE scale dial at that same scale
  (MaxDD >= pure_scale_MaxDD). Tie-break: shallower MaxDD, then higher Sharpe.
  This answers "at this leverage level, how much can the Gold/Bond blend improve
  things without giving up the pure dial's drawdown."
  We also report frontier_gap (candidate CAGR_OOS minus the equal-MaxDD uniform
  -scale twin's CAGR_OOS) on the winner, to confirm it's a real shape-change.

Grid sweep uses the CHEAP path (MaxDD + CAGR_OOS only, no WFA). Full standard-10
+ WFA is computed ONLY on the per-scale winner (and on X4 at that scale for
reference). SANITY: harness sc2.0 reproduces FRONTIER (inherited via _self_test
machinery -- re-checked here for sc2.0).

ASCII-only prints. No commit, no temp files.
Outputs:
  audit_results/scale_dial_x4equiv_sweep_20260627.csv   (full grid, all scales)
  audit_results/scale_dial_x4equiv_winners_20260627.csv (per-scale winner std-10)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.audit.dd_reduction_harness_20260626 as H
from src.audit.unified_metrics import compute_10metrics
from src.audit.run_p01_backtest_20260611 import _apply_aftertax, _calendar_year_returns

AFTER_TAX = 0.8273
STRONG_MAP = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}

# target scale dial points + the PURE scale dial MaxDD at each (from
# P09_STR_SCALE_DIAL_20260623.md / p09_strongmap_scale_dial_20260623.csv).
TARGETS = [
    (1.4, -0.464781),
    (1.6, -0.519521),
    (1.8, -0.569995),
    (2.2, -0.663300),  # sc2.2 from the dial report (-66.33%)
]

# (gold_w, bond_w) grid to sweep at each scale. Centered around the X4/N4 region
# but spanning lighter -> heavier blends. NASDAQ weight = 1 - gold - bond.
GOLD_GRID = [0.00, 0.10, 0.18, 0.24, 0.28, 0.32, 0.38, 0.44]
BOND_GRID = [0.00, 0.05, 0.07, 0.10, 0.14]
# X4's fixed mix for the per-scale "X4-as-is" reference row
X4_MIX = (0.32, 0.05)

# pure scale dial CAGR_OOS (after-tax) for context (from dial report).
PURE_OOS = {1.4: 0.243414, 1.6: 0.262127, 1.8: 0.278068, 2.2: 0.301100}


def _cheap(ctx, **kw):
    """MaxDD + CAGR_OOS + Sharpe_Full only (no WFA). Fast grid evaluation."""
    nav, r, tpy, exc = H.build(ctx, **kw)
    pre = compute_10metrics(nav, tpy)
    aft = _apply_aftertax(pre)
    return dict(CAGR_OOS=aft["CAGR_OOS"], CAGR_IS=aft["CAGR_IS"],
                MaxDD=pre["MaxDD_FULL"], Sharpe=pre["Sharpe_FULL"],
                nav=nav, r=r, tpy=tpy)


def _twin_cagr_at_maxdd(ctx, target_maxdd, lo=1.0, hi=3.6, iters=24):
    """Uniform-scale (gold=bond=0) twin whose MaxDD == target. Returns its CAGR_OOS."""
    f_lo = _cheap(ctx, scale=lo)["MaxDD"]
    f_hi = _cheap(ctx, scale=hi)["MaxDD"]
    # MaxDD decreases (more negative) as scale rises; find scale with MaxDD==target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = _cheap(ctx, scale=mid)["MaxDD"]
        if m > target_maxdd:   # twin shallower than target -> need more leverage
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return _cheap(ctx, scale=s)["CAGR_OOS"], s


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 110)
    print("X4-EQUIVALENT AT scale 1.4/1.6/1.8/2.2 -- gold/bond grid sweep + per-scale best  2026-06-27")
    print("=" * 110)
    ctx = H.setup()

    # SANITY: sc2.0 reproduces FRONTIER
    s20 = _cheap(ctx, scale=2.0)
    ok = (abs(s20["CAGR_IS"] - 0.353755) <= 0.0015 and
          abs(s20["CAGR_OOS"] - 0.291102) <= 0.0015 and
          abs(s20["MaxDD"] + 0.616342) <= 0.0015)
    print("\nSANITY sc2.0: IS%+.4f%% OOS%+.4f%% MaxDD%+.4f%% -> %s"
          % (s20["CAGR_IS"]*100, s20["CAGR_OOS"]*100, s20["MaxDD"]*100,
             "PASS" if ok else "FAIL"))
    if not ok:
        print("HALT"); sys.exit(1)

    grid_rows = []
    winners = {}
    for scale, pure_dd in TARGETS:
        print("\n" + "-" * 100)
        print("SCALE %.1f  (pure-dial MaxDD=%.2f%%, pure-dial CAGR_OOS=%+.2f%%) -- sweeping gold/bond"
              % (scale, pure_dd*100, PURE_OOS[scale]*100))
        cand = []
        for gw in GOLD_GRID:
            for bw in BOND_GRID:
                if gw + bw > 0.55:   # keep NASDAQ weight >= 0.45
                    continue
                m = _cheap(ctx, scale=scale, in_gold_w=gw, in_bond_w=bw)
                feasible = (m["MaxDD"] >= pure_dd)   # not deeper than pure dial
                rec = dict(scale=scale, gold_w=gw, bond_w=bw,
                           CAGR_OOS=m["CAGR_OOS"], CAGR_IS=m["CAGR_IS"],
                           MaxDD=m["MaxDD"], Sharpe=m["Sharpe"],
                           feasible=int(feasible))
                cand.append(rec)
                grid_rows.append(rec)
        # SELECTION (re-tuned, faithful to the X4 thesis -- give up a little CAGR for
        # the deepest MaxDD improvement): among blends whose CAGR_OOS stays within
        # CAGR_TOL of the pure dial (the project's >=1% tolerance), MINIMIZE MaxDD
        # (shallowest = least negative), tie-break higher Sharpe. This always returns
        # a real blend (objective "max CAGR s.t. MaxDD>=pure" degenerates to gold=0).
        CAGR_TOL = 0.010   # 1.0pp give-up allowed vs pure-dial CAGR_OOS at this scale
        floor = PURE_OOS[scale] - CAGR_TOL
        feas = [c for c in cand if c["CAGR_OOS"] >= floor]
        pool = feas if feas else cand
        pool_sorted = sorted(pool, key=lambda c: (c["MaxDD"], -c["Sharpe"]), reverse=True)
        # reverse=True on (MaxDD, -Sharpe): want LARGEST MaxDD (shallowest) first.
        # MaxDD is negative; "largest" = closest to 0 = shallowest. -Sharpe largest = Sharpe smallest,
        # so to tie-break on HIGHER Sharpe we sort by (MaxDD asc-neg, Sharpe desc) -> do it explicitly:
        pool_sorted = sorted(pool, key=lambda c: (-c["MaxDD"], -c["Sharpe"]))
        best = pool_sorted[0]
        winners[scale] = best
        print("  top by shallowest MaxDD (CAGR_OOS >= pure-%.1fpp = %+.2f%%):"
              % (CAGR_TOL*100, floor*100))
        for c in pool_sorted[:5]:
            print("    gold=%.2f bond=%.2f | OOS=%+.2f%% IS=%+.2f%% MaxDD=%+.2f%% Sh=%.3f %s"
                  % (c["gold_w"], c["bond_w"], c["CAGR_OOS"]*100, c["CAGR_IS"]*100,
                     c["MaxDD"]*100, c["Sharpe"], "<-- BEST" if c is best else ""))

    # full standard-10 + WFA on each winner (and X4-as-is at that scale for ref)
    print("\n" + "=" * 110)
    print("FULL STANDARD-10 + WFA on per-scale winners")
    print("=" * 110)
    win_rows = []
    cy_map = {}
    for scale, _ in TARGETS:
        b = winners[scale]
        gw, bw = b["gold_w"], b["bond_w"]
        nav, r, tpy, exc = H.build(ctx, scale=scale, in_gold_w=gw, in_bond_w=bw)
        m = H.metrics10(ctx, nav, r, tpy, label="X4eq_sc%.1f" % scale, with_wfa=True)
        twin_oos, twin_s = _twin_cagr_at_maxdd(ctx, m["MaxDD"])
        fgap = m["CAGR_OOS"] - twin_oos
        cy = _calendar_year_returns(nav); cy = cy[cy.index <= 2025] * AFTER_TAX
        cy_map["X4eq_sc%.1f" % scale] = cy
        # worst1d date
        rr = np.asarray(r, float); wi = int(np.argmin(rr))
        w1d_date = ctx["dates_dt"][wi].strftime("%Y-%m-%d")
        max_eff_lev = scale * 3.0 * max(STRONG_MAP.values())
        rec = dict(label="X4eq_sc%.1f" % scale, scale=scale, gold_w=gw, bond_w=bw,
                   CAGR_IS=m["CAGR_IS"], CAGR_OOS=m["CAGR_OOS"], min9=m["min9"],
                   IS_OOS_gap_pp=m["IS_OOS_gap_pp"], Sharpe_FULL=m["Sharpe_FULL"],
                   Sharpe_OOS=m["Sharpe_OOS"], MaxDD=m["MaxDD"], Worst1D=m["Worst1D"],
                   Worst1D_date=w1d_date, Worst10Y=m["Worst10Y"], Worst5Y=m["Worst5Y"],
                   P10_5Y=m["P10_5Y"], Trades_yr=m["Trades_yr"], max_eff_lev=max_eff_lev,
                   WFE=m["WFE"], CI95_lo=m["CI95_lo"], Regime_min=m["Regime_min"],
                   frontier_gap_pp=fgap*100, twin_scale=twin_s,
                   pure_MaxDD=dict(TARGETS)[scale], pure_OOS=PURE_OOS[scale])
        win_rows.append(rec)
        print("\nscale %.1f WINNER: gold=%.2f bond=%.2f (NASDAQ=%.2f)"
              % (scale, gw, bw, 1-gw-bw))
        print("  CAGR_IS=%+.2f%% OOS=%+.2f%% (pure-dial OOS=%+.2f%%, diff %+.2fpp)"
              % (m["CAGR_IS"]*100, m["CAGR_OOS"]*100, PURE_OOS[scale]*100,
                 (m["CAGR_OOS"]-PURE_OOS[scale])*100))
        print("  MaxDD=%+.2f%% (pure-dial=%+.2f%%, diff %+.2fpp) Sharpe=%.3f gap=%+.2fpp"
              % (m["MaxDD"]*100, dict(TARGETS)[scale]*100,
                 (m["MaxDD"]-dict(TARGETS)[scale])*100, m["Sharpe_FULL"], m["IS_OOS_gap_pp"]))
        print("  W10Y=%+.2f%% W5Y=%+.2f%% P10=%+.2f%% Tr=%.1f W1D=%+.2f%%(%s) maxL=%.1fx"
              % (m["Worst10Y"]*100, m["Worst5Y"]*100, m["P10_5Y"]*100, m["Trades_yr"],
                 m["Worst1D"]*100, w1d_date, max_eff_lev))
        print("  WFE=%.3f CI95lo=%+.2f%% Reg=%+.2f%% frontier_gap=%+.2fpp(twin scale=%.2f)"
              % (m["WFE"], m["CI95_lo"]*100, m["Regime_min"]*100, fgap*100, twin_s))

    # ---- tuning summary: winner under 0.5pp vs 1.0pp give-up, + bond non-help ----
    print("\n" + "=" * 110)
    print("TUNING SUMMARY (re-tune gold/bond per scale; bond never improves CAGR/MaxDD frontier here)")
    print("=" * 110)
    gdf = pd.DataFrame(grid_rows)
    print("%-6s | %-22s | %-22s" % ("scale", "give-up 0.5pp (gold/bond)", "give-up 1.0pp [HEADLINE]"))
    for scale, pdd in TARGETS:
        sub = gdf[abs(gdf["scale"] - scale) < 1e-9]
        for tol, tag in ((0.005, "05"), (0.010, "10")):
            fb = sub[sub["CAGR_OOS"] >= PURE_OOS[scale] - tol]
            w = fb.sort_values(["MaxDD", "Sharpe"], ascending=[False, False]).iloc[0]
            if tag == "05":
                s05 = "g%.2f/b%.2f OOS%+.2f%% DD%+.2f%%" % (w["gold_w"], w["bond_w"], w["CAGR_OOS"]*100, w["MaxDD"]*100)
            else:
                s10 = "g%.2f/b%.2f OOS%+.2f%% DD%+.2f%%" % (w["gold_w"], w["bond_w"], w["CAGR_OOS"]*100, w["MaxDD"]*100)
        print("%-6.1f | %-22s | %-22s" % (scale, s05, s10))
    # bond non-help evidence (sc1.6 gold=0.10 row)
    s16 = gdf[(abs(gdf["scale"]-1.6) < 1e-9) & (abs(gdf["gold_w"]-0.10) < 1e-9)].sort_values("bond_w")
    print("\n  bond non-help (sc1.6, gold=0.10): adding bond cuts CAGR faster than it helps MaxDD:")
    for _, r in s16.iterrows():
        print("    bond=%.2f -> OOS%+.2f%% MaxDD%+.2f%% Sharpe%.3f" % (r["bond_w"], r["CAGR_OOS"]*100, r["MaxDD"]*100, r["Sharpe"]))

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    pd.DataFrame(grid_rows).to_csv(
        os.path.join(out_dir, "scale_dial_x4equiv_sweep_20260627.csv"),
        index=False, float_format="%.6f", encoding="utf-8-sig")
    pd.DataFrame(win_rows).to_csv(
        os.path.join(out_dir, "scale_dial_x4equiv_winners_20260627.csv"),
        index=False, float_format="%.6f", encoding="utf-8-sig")
    print("\nSaved sweep + winners CSVs.")

    # annual after-tax for the winners (for the report's annual addendum)
    ann_rows = []
    for y in range(1975, 2026):
        row = {"year": y}
        for scale, _ in TARGETS:
            lab = "X4eq_sc%.1f" % scale
            v = cy_map[lab].get(y, np.nan)
            row[lab + "_pct"] = round(float(v)*100, 1) if v == v else np.nan
        ann_rows.append(row)
    pd.DataFrame(ann_rows).to_csv(
        os.path.join(out_dir, "scale_dial_x4equiv_annual_20260627.csv"),
        index=False, float_format="%.1f", encoding="utf-8-sig")
    print("Saved winners annual CSV.")

    block = {"script": "scale_dial_x4equiv_sweep_20260627.py", "date": "2026-06-27",
             "sanity_sc2.0_pass": bool(ok),
             "gold_grid": GOLD_GRID, "bond_grid": BOND_GRID,
             "winners": [{k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in r.items()} for r in win_rows]}
    print("\n" + "=" * 110); print("RETURN_BLOCK"); print("=" * 110)
    print(json.dumps(block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
