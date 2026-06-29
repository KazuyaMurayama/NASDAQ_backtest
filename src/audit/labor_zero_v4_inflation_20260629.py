"""
src/audit/labor_zero_v4_inflation_20260629.py
=============================================
v4 -- INFLATION-AWARE labor-zero. Builds on the v3 Round-G winner (sc2.6,
run-fraction lever, hold-if-crash refill, guaranteed floor + opportunistic top)
and asks: under a REALISTIC Japan inflation assumption, how high a GUARANTEED
REAL floor can we sustain across all 46 starts (1975-2020), still labor=0/ruin=0?

WHY this is the right question
------------------------------
The v2/v3 numbers are NOMINAL. A real retiree's 7.2M must buy the same basket in
year 20 as year 1. The existing robustness check (labor_zero_round2_robustness)
showed that indexing a FIXED full spend at +2%/yr breaks nominal-fixed labor-zero.
BUT the v3 winner does NOT spend a fixed full amount -- it spends a guaranteed
floor and only tops to full when the withdrawal rate is comfortable. So the right
model is: inflate BOTH the floor and the full-spend target each year, and find the
real floor that still survives. That is the honest inflation-aware design.

Inflation model (deterministic, constant g) -- Japan-realistic scenarios:
  g = 0.000  deflation/zero era (1990s-2010s Japan, ~0%/yr)            [floor of realism]
  g = 0.010  mild                                                       [conservative]
  g = 0.020  BoJ target / central planning assumption                   [CENTER]
  g = 0.030  recent actual (2022-2024 Japan CPI ~2.5-3.3%)              [stress]
  g = 0.035  persistent-elevated                                        [severe]
Spend in year k (0-based) is multiplied by (1+g)**k for BOTH floor and full target.
"floor_real" is the year-1 yen; its purchasing power is held constant by indexation.

Levers carried from v3 Round G:
  strat=sc2.6, run-fraction (run0/40M), reserve=1x bond, refill all-in at run<thr
  but ONLY in a year whose run-return >= 0 (hold-if-crash), guaranteed floor every
  year, top to full (also indexed) when full/total <= top_wr.

NEW v4 levers to fight inflation (rounds below):
  H1  COLA-on-floor only (top target also indexed) -- baseline inflation port of G.
  H2  reserve as bond/gold blend (gold has historically tracked inflation better).
  H3  run-fraction re-optimised UNDER inflation (higher growth may be needed).
  H4  glide the top_wr (be greedier late when the sleeve has compounded).

Outputs (audit_results/):
  labor_zero_v4_inflation_grid_20260629.csv     (full sweep)
  labor_zero_v4_inflation_frontier_20260629.csv (max real floor per g)
SELF-TEST: at g=0 the v4 sim must reproduce the v3 Round-G nominal numbers
(guaranteed floor 6.0M at sc2.6/run0.50/top0.20). If not, HALT.
ASCII-only. No commit, no temp files.
"""
from __future__ import annotations

import importlib.util
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

_spec = importlib.util.spec_from_file_location(
    "v3", os.path.join(_THIS_DIR, "labor_zero_v3_harness_20260628.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)

M = v3.M
AR_DIR = os.path.join(_REPO_DIR, "audit_results")
FULL_REAL = 7.2 * M           # year-1 full-spend target (real)
DATA_END = 2025
ALL_STARTS = list(range(1975, 2021))   # 46 starts

INFL = [0.0, 0.010, 0.020, 0.030, 0.035]


def sim_v4(rets, reserve_series, start, horizon, *, strat, run0, reserve0, thr,
           floor_real, top_wr, g, hold_if_crash=True, data_end=DATA_END):
    """Inflation-aware Round-G. floor and full target both grow at (1+g)**k.
    Returns labor/shortfall/worst REAL spend (deflated back to yr1)/totals."""
    run, res = float(run0), float(reserve0)
    s = rets[strat]
    n = min(horizon, data_end - start + 1)
    fired = False
    labor = short = 0
    real_spends = []        # spend deflated to year-1 purchasing power
    for k in range(n):
        yr = start + k
        infl = (1.0 + g) ** k
        r_this = float(s.loc[yr])
        # refill (hold-if-crash): thr is nominal-on-balance; balance is nominal too
        if (not fired) and run < thr and res > 1e-6 and ((not hold_if_crash) or r_this >= 0):
            run += res
            res = 0.0
            fired = True
        total = run + res
        floor_nom = floor_real * infl
        full_nom = FULL_REAL * infl
        spend = floor_nom
        if total > 1e-9 and full_nom / total <= top_wr:
            spend = full_nom
        if total + 1e-6 < spend:
            # can't even fund the (indexed) floor -> labor year; take all
            labor += 1
            spend = total
        if spend < full_nom - 1e-6:
            short += 1
        # deflate realised spend back to yr1 to measure purchasing power
        real_spends.append(spend / infl)
        if run >= spend:
            run -= spend
        else:
            res -= (spend - run)
            run = 0.0
        run *= (1.0 + r_this)
        res *= (1.0 + float(reserve_series.loc[yr]))
    real_spends = np.array(real_spends)
    return dict(labor_years=labor, shortfall_years=short,
                worst_real_spend_M=round(float(real_spends.min()) / M, 2),
                med_real_spend_M=round(float(np.median(real_spends)) / M, 2),
                total_real_spend_M=round(float(real_spends.sum()) / M, 1),
                terminal_nom_M=round((run + res) / M, 1),
                ruin=int((run + res) <= 1e-6), n_years=n)


def _agg(rets, reserve, *, strat, runfrac, floor_real, top_wr, g):
    run0 = runfrac * 40 * M
    res0 = 40 * M - run0
    thr = min(14 * M, run0)
    labor = short = ruin = 0
    wmin = 1e18
    tots = []
    terms = []
    for sy in ALL_STARTS:
        r = sim_v4(rets, reserve, sy, 20, strat=strat, run0=run0, reserve0=res0,
                   thr=thr, floor_real=floor_real, top_wr=top_wr, g=g)
        labor += r["labor_years"]
        short += r["shortfall_years"]
        wmin = min(wmin, r["worst_real_spend_M"])
        ruin += r["ruin"]
        tots.append(r["total_real_spend_M"])
        terms.append(r["terminal_nom_M"])
    return dict(strat=strat, runfrac=runfrac, floor_real_M=round(floor_real / M, 2),
                top_wr=top_wr, g=g, labor_total=labor, shortfall_years_total=short,
                worst_real_spend_M=round(wmin, 2),
                median_total_real_M=round(float(np.median(tots)), 1),
                median_terminal_nom_M=round(float(np.median(terms)), 1), ruin=ruin)


def self_test(rets, bond):
    print("=" * 95)
    print("v4 SELF-TEST: at g=0 reproduce v3 Round-G nominal (sc2.6/run0.50/floor6.0/top0.20)")
    print("=" * 95)
    r = _agg(rets, bond, strat="sc2.6", runfrac=0.50, floor_real=6.0 * M, top_wr=0.20, g=0.0)
    print("  g=0 sc2.6 run0.50 floor6.0 top0.20 -> labor=%d ruin=%d worstReal=%.2fM medTotal=%.1f medTermNom=%.0f"
          % (r["labor_total"], r["ruin"], r["worst_real_spend_M"],
             r["median_total_real_M"], r["median_terminal_nom_M"]))
    ok = (r["labor_total"] == 0 and r["ruin"] == 0 and abs(r["worst_real_spend_M"] - 6.0) < 0.01)
    print("  -> SELF-TEST %s" % ("PASS (Round-G reproduced at g=0)" if ok
                                 else "FAIL -- v4 base diverges from v3; HALT"))
    return ok


def round_H1(rets, bond):
    """COLA on floor+target, v2/v3 geometry. Find max real floor per g (bond reserve)."""
    print("\n" + "=" * 95)
    print("ROUND H1: inflation-indexed floor (bond reserve, run-frac + hold-if-crash from G)")
    print("=" * 95)
    rows = []
    for g in INFL:
        for runfrac in [0.45, 0.50, 0.60]:
            for floor_M in [4.0, 4.5, 5.0, 5.5, 6.0, 6.37]:
                for top_wr in [0.16, 0.18, 0.20]:
                    rows.append(_agg(rets, bond, strat="sc2.6", runfrac=runfrac,
                                     floor_real=floor_M * M, top_wr=top_wr, g=g))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_v4_roundH1_bond_20260629.csv"),
              index=False, encoding="utf-8-sig")
    return df


def _agg_mix(rets, reserve, *, strat, runfrac, floor_real, top_wr, g,
             top_wr_late=None, late_k=10):
    """Like _agg but the reserve series is passed in (allows gold blend) and
    top_wr can GLIDE: years >= late_k use top_wr_late (H4). Implemented by a thin
    wrapper that mutates the top each year via a closure-free reimplementation."""
    run0 = runfrac * 40 * M
    res0 = 40 * M - run0
    thr = min(14 * M, run0)
    labor = short = ruin = 0
    wmin = 1e18
    tots = []
    terms = []
    for sy in ALL_STARTS:
        run, res = float(run0), float(res0)
        s = rets[strat]
        n = min(20, DATA_END - sy + 1)
        fired = False
        real_spends = []
        lab = 0
        for k in range(n):
            yr = sy + k
            infl = (1.0 + g) ** k
            r_this = float(s.loc[yr])
            tw = top_wr if (top_wr_late is None or k < late_k) else top_wr_late
            if (not fired) and run < thr and res > 1e-6 and r_this >= 0:
                run += res
                res = 0.0
                fired = True
            total = run + res
            floor_nom = floor_real * infl
            full_nom = FULL_REAL * infl
            spend = floor_nom
            if total > 1e-9 and full_nom / total <= tw:
                spend = full_nom
            if total + 1e-6 < spend:
                lab += 1
                spend = total
            if spend < full_nom - 1e-6:
                short += 1
            real_spends.append(spend / infl)
            if run >= spend:
                run -= spend
            else:
                res -= (spend - run)
                run = 0.0
            run *= (1.0 + r_this)
            res *= (1.0 + float(reserve[yr]))
        labor += lab
        if lab > 0:
            pass
        rs = np.array(real_spends)
        wmin = min(wmin, float(rs.min()) / M)
        tots.append(float(rs.sum()) / M)
        terms.append((run + res) / M)
        ruin += int((run + res) <= 1e-6)
    return dict(strat=strat, runfrac=runfrac, floor_real_M=round(floor_real / M, 2),
                top_wr=top_wr, top_wr_late=top_wr_late, g=g, labor_total=labor,
                shortfall_years_total=short, worst_real_spend_M=round(wmin, 2),
                median_total_real_M=round(float(np.median(tots)), 1),
                median_terminal_nom_M=round(float(np.median(terms)), 1), ruin=ruin)


def round_H234(rets):
    """H2 (gold/bond blend reserve) x H3 (run-fraction) x H4 (top_wr glide).
    Goal: lift the GUARANTEED REAL floor under realistic inflation above H1's bond-only."""
    print("\n" + "=" * 95)
    print("ROUND H2/H3/H4: gold-blend reserve + run-fraction + top_wr glide (anti-inflation levers)")
    print("=" * 95)
    # reserve mixes: bond-only (H1 ref), +gold sleeves (gold tracks inflation better)
    mixes = {
        "bond100": {"bond": 1.0},
        "b80g20": {"bond": 0.8, "gold": 0.2},
        "b60g40": {"bond": 0.6, "gold": 0.4},
        "b50g50": {"bond": 0.5, "gold": 0.5},
    }
    series = {k: v3.load_mixed_reserve(w) for k, w in mixes.items()}
    rows = []
    for g in INFL:
        for mixname, rs in series.items():
            for runfrac in [0.45, 0.50, 0.60]:
                for floor_M in [5.0, 5.5, 6.0, 6.37, 6.6]:
                    for top_wr in [0.16, 0.18, 0.20]:
                        # H4: also try a later-greedier glide (top_wr_late higher)
                        for tw_late in [None, 0.24, 0.28]:
                            r = _agg_mix(rets, rs, strat="sc2.6", runfrac=runfrac,
                                         floor_real=floor_M * M, top_wr=top_wr, g=g,
                                         top_wr_late=tw_late)
                            r["reserve_mix"] = mixname
                            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_v4_roundH234_20260629.csv"),
              index=False, encoding="utf-8-sig")
    print("\nFRONTIER per g (best GUARANTEED REAL floor, labor0/ruin0), H2/3/4 vs H1 bond-only:")
    front = []
    for g in INFL:
        ok = df[(df.g == g) & (df.labor_total == 0) & (df.ruin == 0)].sort_values(
            ["worst_real_spend_M", "median_total_real_M"], ascending=[False, False])
        bond_ok = ok[ok.reserve_mix == "bond100"]
        bref = bond_ok.iloc[0]["worst_real_spend_M"] if len(bond_ok) else 0.0
        if len(ok):
            b = ok.iloc[0]
            front.append(b.to_dict())
            twl = "none" if b["top_wr_late"] is None else ("%.0f%%" % (b["top_wr_late"] * 100))
            print("  g=%.1f%%: BEST realFloor %.2fM [%s run%.2f top%.0f%%->late%s] vs bond-only %.2fM  (+%.2fM)  cut=%d term=%.0f"
                  % (g * 100, b["worst_real_spend_M"], b["reserve_mix"], b["runfrac"],
                     b["top_wr"] * 100, twl, bref, b["worst_real_spend_M"] - bref,
                     b["shortfall_years_total"], b["median_terminal_nom_M"]))
        else:
            print("  g=%.1f%%: no labor0/ruin0" % (g * 100))
    pd.DataFrame(front).to_csv(
        os.path.join(AR_DIR, "labor_zero_v4_inflation_frontier_H234_20260629.csv"),
        index=False, encoding="utf-8-sig")
    return df, front


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rets = v3.load_rets()
    bond = v3.load_mixed_reserve({"bond": 1.0})
    if not self_test(rets, bond):
        return
    dfH1 = round_H1(rets, bond)
    dfH234, frontH234 = round_H234(rets)
    # frontier: max real floor with labor=0/ruin=0 per g
    print("\n" + "=" * 95)
    print("FRONTIER (bond reserve): max GUARANTEED REAL floor with labor=0/ruin=0, per inflation g")
    print("=" * 95)
    front = []
    for g in INFL:
        ok = dfH1[(dfH1.g == g) & (dfH1.labor_total == 0) & (dfH1.ruin == 0)]
        if len(ok):
            best = ok.sort_values(["worst_real_spend_M", "median_total_real_M"],
                                  ascending=[False, False]).iloc[0]
            front.append(best.to_dict())
            print("  g=%.1f%%: real floor %.2fM  (run%.2f top%.0f%%)  cutYrs=%d/920  medTotalReal=%.0f  medTermNom=%.0f"
                  % (g * 100, best["worst_real_spend_M"], best["runfrac"],
                     best["top_wr"] * 100, best["shortfall_years_total"],
                     best["median_total_real_M"], best["median_terminal_nom_M"]))
        else:
            front.append(dict(g=g, worst_real_spend_M=0.0, note="no labor0/ruin0 config"))
            print("  g=%.1f%%: NO labor0/ruin0 config in grid" % (g * 100))
    pd.DataFrame(front).to_csv(
        os.path.join(AR_DIR, "labor_zero_v4_inflation_frontier_20260629.csv"),
        index=False, encoding="utf-8-sig")
    print("\nRETURN_BLOCK")
    print(json.dumps({"frontier": front}, indent=2, ensure_ascii=True, default=str))
    print("\nDone (Round H1 + frontier).")


if __name__ == "__main__":
    main()
