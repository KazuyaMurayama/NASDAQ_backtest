"""
src/audit/labor_zero_round2_configs_20260627.py
===============================================
ROUND 2 of labor-zero (assets 40M, strict spend 7.2M/yr, 31 starts 1975-2005, 20y).
Round 1 proved NO single lever reaches labor-0 and NONE even saves 1988.

DIAGNOSIS (verified live, see labor_zero_round2_diag): the 1988 start dies not from
the -24% year-1 but from a 6-7 YEAR GRIND 1988-1994 whose cumulative strat growth is
only ~2.1x for ALL leverage levels (2.05x at sc1.6 .. 2.16x at sc2.6) while 7.2M/yr
(18%) bleeds the run sleeve. The prior top-up dumps the ENTIRE 20M reserve in 1989,
leaving zero buffer; the run grinds to 0.73M by 1995, just before the +149/+41/+52/
+112/+204% boom it can no longer ride. Higher leverage (F) deepens the single-year
holes (1990 -29.9% -> -34.7%) without raising the grind-window cumulative -> useless.

ROUND-2 THESIS -> three families that attack the GRIND, not the year-1 dip:
  CONFIG 1  "reverse glide + staged top-up + bond-tilt sleeve"
            high leverage EARLY (sc2.4) to outrun withdrawals in good starts, then
            DE-lever to sc1.6 after K years; reserve invested in BOND and fed in
            STAGED chunks (not all-at-once) so a buffer survives the grind.
  CONFIG 2  "in-sleeve bond tilt (85/15) + bond reserve + staged top-up"
            blend 15% bond INSIDE the run sleeve -> same 1988-94 cumulative (2.18x)
            but worst year -33%->-28.5%; reserve bond, staged feed.
  CONFIG 3  "conventional low-early glide + big bond bucket bridge"
            sc1.6 first 2y then sc2.2, PLUS a 5y cash/bond bridge bucket drawn first
            to span the grind, reserve in bond.

This harness adds the MISSING lever from round 1: STAGED top-up (feed reserve in
fixed annual chunks with a reserve floor) and an in-sleeve asset blend. It reuses
h2's loaders + return convention exactly. Self-checks vs h2 baseline first.

ASCII-only. No commit, no temp files.
Outputs: audit_results/labor_zero_round2_configs_20260627.csv
"""
from __future__ import annotations
import os, sys, json, importlib.util
import numpy as np, pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_THIS); _REPO = os.path.dirname(_SRC)
spec = importlib.util.spec_from_file_location(
    "h2", os.path.join(_THIS, "labor_zero_harness_v2_20260627.py"))
h2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h2)
M = h2.M; SPEND = h2.SPEND; HORIZON = h2.HORIZON; START_YEARS = h2.START_YEARS
AR = os.path.join(_REPO, "audit_results")


def strat_ret(rets, sleeves, cfg, yr, k):
    """Return-sleeve growth for calendar yr / elapsed k under cfg.
    Supports: reverse/forward glide (list of (k_threshold, key)),
    static single, and an in-sleeve blend {key:weight} possibly with a sleeve asset."""
    # 1. pick base strategy key by glide schedule (largest threshold <= k)
    if cfg.get("glide"):
        key = cfg["glide"][0][1]
        for thr_k, sk in cfg["glide"]:
            if k >= thr_k:
                key = sk
    else:
        key = cfg["single"]
    r = float(rets[key].loc[yr])
    # 2. optional in-sleeve blend with a sleeve asset (e.g. 15% bond inside the run)
    blend = cfg.get("insleeve_blend")  # (asset_name, weight_on_asset)
    if blend:
        asset, w = blend
        r = (1.0 - w) * r + w * float(sleeves[asset].loc[yr])
    return r


def simulate(rets, sleeves, start, cfg):
    """One start, strict spend. cfg keys:
       single / glide / insleeve_blend  -> run sleeve return
       run0, reserve0, reserve_mode
       topup_thr, topup_chunk (None=ALL), reserve_floor (keep >= this in reserve when feeding)
       init_bucket_years, bucket_mode (cash|bond) -> bridge bucket drawn first (and grows if bond)
       draw_order
    """
    run = float(cfg["run0"]); reserve = float(cfg["reserve0"])
    bucket = float(cfg.get("init_bucket_years", 0)) * SPEND
    bucket_mode = cfg.get("bucket_mode", "cash")
    sret = sleeves[cfg.get("reserve_mode", "cash")]
    bret = sleeves[bucket_mode] if bucket_mode != "cash" else None
    thr = cfg.get("topup_thr", 20 * M)
    chunk = cfg.get("topup_chunk", None)           # None = ALL
    rfloor = cfg.get("reserve_floor", 0.0)         # keep this much in reserve while feeding
    draw_order = cfg.get("draw_order", "run_first")
    labor = 0; topups = 0; min_total = run + reserve + bucket
    for k in range(HORIZON):
        yr = start + k
        # 1. STAGED top-up: feed reserve -> run, but never below reserve_floor,
        #    and at most `chunk` per year (None => all available above floor).
        if run < thr and reserve > rfloor + 1e-6:
            avail = reserve - rfloor
            move = avail if chunk is None else min(chunk, avail)
            run += move; reserve -= move; topups += 1
        # 2. strict spend.  bucket_when='first' (bridge spans the START, drawn before
        #    selling the run) or 'last' (END bridge: run/reserve first, bucket only as
        #    last resort so it survives to bridge the 2000-02 trough).
        bucket_when = cfg.get("bucket_when", "first")
        total = run + reserve + bucket
        if total + 1e-6 < SPEND:
            labor += 1; run = reserve = bucket = 0.0
        else:
            need = SPEND
            if bucket_when == "first":
                take = min(bucket, need); bucket -= take; need -= take
            if need > 1e-9:
                r_this = strat_ret(rets, sleeves, cfg, yr, k)
                down = r_this < 0
                if draw_order == "reserve_first_on_down" and down:
                    take = min(reserve, need); reserve -= take; need -= take
                    if need > 1e-9:
                        take = min(run, need); run -= take; need -= take
                else:
                    take = min(run, need); run -= take; need -= take
                    if need > 1e-9:
                        take = min(reserve, need); reserve -= take; need -= take
            if bucket_when == "last" and need > 1e-9:
                take = min(bucket, need); bucket -= take; need -= take
        # 3. growth
        r_this = strat_ret(rets, sleeves, cfg, yr, k)
        run *= (1.0 + r_this)
        reserve *= (1.0 + float(sret.loc[yr]))
        if bret is not None:
            bucket *= (1.0 + float(bret.loc[yr]))
        total = run + reserve + bucket
        if total < min_total:
            min_total = total
    ruin = (run + reserve + bucket) <= 1e-6
    return dict(labor_years=labor, topups=topups, ruin=int(ruin),
                terminal=run + reserve + bucket, min_total=min_total)


def run_all(rets, sleeves, cfg):
    labor = topup = ruin = 0; terms = []; floors = []; fails = []; labmap = {}
    for sy in START_YEARS:
        res = simulate(rets, sleeves, sy, cfg)
        labor += res["labor_years"]; topup += res["topups"]; ruin += res["ruin"]
        if res["labor_years"] > 0:
            fails.append(sy); labmap[sy] = res["labor_years"]
        if res["ruin"] == 0:
            terms.append(res["terminal"])
        floors.append(res["min_total"])
    return dict(labor_years_total=labor, starts_with_labor=len(fails), fails=fails,
                labmap=labmap, saved_1988=(1988 not in fails),
                topup_events_total=topup, ruin_total=ruin,
                terminal_median_M=(float(np.median(terms)) / M if terms else 0.0),
                terminal_min_M=(float(np.min(terms)) / M if terms else 0.0),
                worst_floor_M=float(np.min(floors)) / M)


def selftest(rets, sleeves):
    # reproduce h2 baseline through this engine: sc2.2 run20/res20 cash thr20 ALL
    r = run_all(rets, sleeves, dict(single="sc2.2", run0=20 * M, reserve0=20 * M,
                                    reserve_mode="cash", topup_thr=20 * M, topup_chunk=None))
    ok = (r["labor_years_total"] == 12 and r["fails"] == [1988])
    print("SELF-TEST (engine vs h2 baseline): labor=%d fails=%s -> %s"
          % (r["labor_years_total"], r["fails"], "MATCH" if ok else "MISMATCH!"))
    return ok


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rets = h2.load_returns(); sleeves = h2.load_sleeve_returns(); rets.update(h2.load_extended())
    print("=" * 100)
    print("LABOR-ZERO ROUND 2 -- attack the 1988-94 GRIND (staged top-up, bond tilt, reverse glide)")
    print("=" * 100)
    if not selftest(rets, sleeves):
        print("ABORT: engine mismatch"); return

    rows = []

    def rec(family, label, cfg):
        r = run_all(rets, sleeves, cfg)
        rows.append(dict(family=family, label=label,
                         labor=r["labor_years_total"], starts_fail=r["starts_with_labor"],
                         saved_1988=int(r["saved_1988"]),
                         fails=";".join(str(x) for x in r["fails"]),
                         termMed_M=round(r["terminal_median_M"], 1),
                         termMin_M=round(r["terminal_min_M"], 1),
                         floor_M=round(r["worst_floor_M"], 2),
                         topups=r["topup_events_total"], ruin=r["ruin_total"]))
        return r

    # ============ FAMILY 0: isolate the STAGED TOP-UP lever alone ============
    # (round 1 only ever did ALL or fixed 10M from thr; here: chunk + reserve_floor,
    #  bond reserve so the held-back buffer also grows.)
    print("\n[0] staged top-up isolation (bond reserve, chunk + floor) ...")
    for chunk in (5 * M, 7.2 * M, 10 * M):
        for floor in (0, 5 * M, 8 * M):
            for thr in (15 * M, 20 * M):
                rec("0_staged", "sc2.2_chunk%.0f_floor%.0f_thr%.0f" % (chunk / M, floor / M, thr / M),
                    dict(single="sc2.2", run0=20 * M, reserve0=20 * M, reserve_mode="bond",
                         topup_thr=thr, topup_chunk=chunk, reserve_floor=floor))

    # ============ CONFIG 1: REVERSE glide + staged bond top-up ============
    # high lev early (outrun withdrawals), de-lever later; bond reserve fed in chunks.
    print("[1] reverse glide (hi early -> lo late) + staged bond top-up ...")
    for hi in ("sc2.2", "sc2.4", "sc2.6"):
        for lo in ("sc1.6", "sc2.0"):
            for ksw in (3, 5, 7):
                for chunk in (7.2 * M, 10 * M):
                    for floor in (5 * M, 8 * M):
                        rec("1_revglide",
                            "rev_%s_to_%s_k%d_chunk%.0f_floor%.0f" % (hi, lo, ksw, chunk / M, floor / M),
                            dict(glide=[(0, hi), (ksw, lo)], single=hi,
                                 run0=20 * M, reserve0=20 * M, reserve_mode="bond",
                                 topup_thr=20 * M, topup_chunk=chunk, reserve_floor=floor))

    # ============ CONFIG 2: in-sleeve bond tilt + bond reserve + staged top-up ============
    print("[2] in-sleeve bond tilt + bond reserve + staged top-up ...")
    for base in ("sc2.2", "sc2.4", "sc2.6"):
        for w in (0.10, 0.15, 0.20, 0.25):
            for chunk in (7.2 * M, 10 * M, None):
                for floor in (0, 5 * M, 8 * M):
                    ctag = "ALL" if chunk is None else "%.0f" % (chunk / M)
                    rec("2_insleeve",
                        "%s_bond%.0f_chunk%s_floor%.0f" % (base, w * 100, ctag, floor / M),
                        dict(single=base, insleeve_blend=("bond", w),
                             run0=20 * M, reserve0=20 * M, reserve_mode="bond",
                             topup_thr=20 * M, topup_chunk=chunk, reserve_floor=floor))

    # ============ CONFIG 3: 50:50 split, BOND reserve, top-up THRESHOLD sweep ============
    # The diagnostic showed the threshold/timing of the single 1989 reserve injection
    # is the live lever (thr<=20M => inject right after the -24% dip => labor 7-9;
    # thr>=22M => inject too early, -24% hits the bigger base => labor 14).
    print("[3] threshold-timed single injection (run20/res20 bond, leverage x thr) ...")
    for lev in ("sc2.2", "sc2.4", "sc2.6"):
        for w in (0.0, 0.12, 0.18):
            for thr in (10 * M, 14 * M, 18 * M, 20 * M):
                rec("3_thrtime",
                    "%s_bond%.0f_thr%.0f" % (lev, w * 100, thr / M),
                    dict(single=lev, insleeve_blend=(("bond", w) if w > 0 else None),
                         run0=20 * M, reserve0=20 * M, reserve_mode="bond",
                         topup_thr=thr, topup_chunk=None))

    # ============ CONFIG 4: forward low-early glide + SMALL bond bridge (END), 50:50ish
    # bucket must come out of TOTAL 40M: run + reserve + bucket = 40M. (round-1 bug:
    # bucket+run overflowed 40M and every config was skipped.)
    print("[4] forward low-early glide + small END bond bridge (alloc-fixed) ...")
    for nb in (1, 2, 3):
        bucket = nb * SPEND
        for run in (18 * M, 20 * M):
            res = 40 * M - run - bucket
            if res < 1 * M:
                continue
            for hi in ("sc2.4", "sc2.6"):
                for w in (0.0, 0.12):
                    rec("4_fglide_bridge",
                        "glide_sc1.6_to_%s_bond%.0f_bridge%dy_run%.0f" % (hi, w * 100, nb, run / M),
                        dict(glide=[(0, "sc1.6"), (2, hi)], single=hi,
                             insleeve_blend=(("bond", w) if w > 0 else None),
                             run0=run, reserve0=res, reserve_mode="bond",
                             init_bucket_years=nb, bucket_mode="bond", bucket_when="last",
                             topup_thr=20 * M, topup_chunk=None))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR, "labor_zero_round2_configs_20260627.csv"),
              index=False, encoding="utf-8-sig")
    print("\nSaved round2 grid: %d configs" % len(df))

    # ---- per-family best ----
    print("\n--- per-family min labor (prior best=12; lower=better) ---")
    for fam in sorted(df["family"].unique()):
        sub = df[df["family"] == fam].sort_values(["labor", "termMed_M"], ascending=[True, False])
        b = sub.iloc[0]
        print("  %-12s minLabor=%2d  saved1988=%d  fails=[%s]  termMed=%.0fM  | %s"
              % (fam, int(b["labor"]), int(b["saved_1988"]), b["fails"], b["termMed_M"], b["label"]))

    saved = df[df["saved_1988"] == 1].sort_values(["labor", "termMed_M"], ascending=[True, False])
    print("\nCONFIGS THAT SAVE 1988: %d" % len(saved))
    for _, r in saved.head(20).iterrows():
        print("  [%s] %-48s labor=%2d fails=[%s] termMed=%.0fM floor=%.1fM"
              % (r["family"], r["label"], int(r["labor"]), r["fails"], r["termMed_M"], r["floor_M"]))

    zero = df[(df["labor"] == 0) & (df["ruin"] == 0)].sort_values("termMed_M", ascending=False)
    print("\n*** LABOR-ZERO (and no-ruin) CONFIGS: %d ***" % len(zero))
    for _, r in zero.iterrows():
        print("  [%s] %-50s termMed=%.0fM termMin=%.0fM floor=%.1fM"
              % (r["family"], r["label"], r["termMed_M"], r["termMin_M"], r["floor_M"]))

    block = {"script": "labor_zero_round2_configs_20260627.py", "date": "2026-06-27",
             "n_configs": int(len(df)),
             "per_family_min_labor": {fam: int(df[df["family"] == fam]["labor"].min())
                                      for fam in sorted(df["family"].unique())},
             "n_saved_1988": int(len(saved)),
             "n_labor_zero": int(len(zero)),
             "best_overall": (df.sort_values(["labor", "termMed_M"], ascending=[True, False])
                              .iloc[0][["family", "label", "labor", "fails", "termMed_M"]].to_dict())}
    print("\n" + "=" * 100); print("RETURN_BLOCK"); print(json.dumps(block, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
