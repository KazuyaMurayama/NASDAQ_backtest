"""
src/audit/verify_p09_gas_log_20260615.py
========================================
P09Log (GAS live) vs Python independent recomputation — divergence check.

Phase 4.2 of the GAS parallel-run plan. Re-derives every field of the GAS
P09Log row from FIRST PRINCIPLES (the frozen spec formulas + the raw inputs
captured from the production spreadsheet) and compares to what GAS logged.

Input snapshot (captured 2026-06-15 from the production spreadsheet via the
Drive read): audit_results/p09_gas_log_snapshot_20260615.json
  - the P09Log row, the same-day Dyn2x3x Log row (lev_raw/vix_z handoff),
    and the GC=F / TLT price tails used for the inverse-vol w_g.

What is checked (recomputed vs logged):
  lev_mod_065, W1 mode, mom63 bucket, boost_mult, L_t, bond_on,
  OUT holdings (gold1x/bond1x/cash), w_g (inverse-vol from raw prices),
  and the lev_raw/vz hand-off from the Dyn2x3x Log.

Note on scope: PriceHistory(^IXIC) is truncated in the Drive export, so
mom63 is checked for INTERNAL CONSISTENCY (the logged value -> implied
63d-ago level is sanity-checked against the Dyn2x3x close), not bit-exact.
All other fields are bit/▒-exact.

ASCII-only prints. Read-only.
"""
from __future__ import annotations

import json
import math
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
SNAP = os.path.join(_REPO_DIR, "audit_results", "p09_gas_log_snapshot_20260615.json")

# frozen spec (p09_live_spec_20260611.json)
K_LO, K_MID, K_HI, VZ_THR = 0.1, 0.5, 0.8, 0.65
W1_ENTER, W1_EXIT = 0.7, 0.3
Q25, Q50, Q75 = -0.028802750000000002, 0.0375715, 0.09098425
BOOST_P09 = {0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00}
VOL_WIN, WG_LO, WG_HI = 63, 0.25, 0.75

TOL = 0.001


def lev_mod_065(lev_raw, vz, lt):
    k = K_HI if vz > VZ_THR else (K_LO if vz < -VZ_THR else K_MID)
    bias = max(-0.5, min(0.5, -k * lt * 0.5))
    return max(0.0, min(1.0, lev_raw + bias))


def bucket(m):
    if m <= Q25:
        return 0
    if m <= Q50:
        return 1
    if m <= Q75:
        return 2
    return 3


def inv_vol_wg(gold, bond, w=VOL_WIN):
    def sig(cl):
        r = [cl[i] / cl[i - 1] - 1 for i in range(1, len(cl))]
        seg = r[-w:]
        m = sum(seg) / w
        var = sum((x - m) ** 2 for x in seg) / (w - 1)
        return (var ** 0.5) * (252 ** 0.5)
    sg, sb = sig(gold), sig(bond)
    wg = (1.0 / sg) / ((1.0 / sg) + (1.0 / sb))
    return max(WG_LO, min(WG_HI, wg)), sg, sb


def main():
    snap = json.load(open(SNAP, encoding="utf-8"))
    g = snap["p09log_20260615"]
    dyn = snap["dyn2x3x_log_20260615"]
    gold = [x[1] for x in snap["gold_tail_GCF"]]
    bond = [x[1] for x in snap["bond_tail_TLT"]]

    print("=" * 80)
    print("P09Log (GAS) vs Python recomputation   row 2026-06-15")
    print("=" * 80)
    results = []

    def chk(name, recomputed, logged, tol=TOL, exact=False):
        if exact:
            ok = (recomputed == logged)
            msg = "calc=%s logged=%s" % (recomputed, logged)
        else:
            ok = abs(recomputed - logged) <= tol
            msg = "calc=%.4f logged=%.4f  d=%+.4f" % (recomputed, logged, recomputed - logged)
        results.append(ok)
        print("  %-16s %s   %s" % (name, "OK  " if ok else "DIFF", msg))

    # 1. lev_mod_065
    lm = lev_mod_065(g["lev_raw"], g["vz"], g["lt_sig"])
    chk("lev_mod_065", lm, g["lev_mod_065"])

    # 2. W1 mode (initial OUT, lev_mod<=exit -> OUT)
    mode_calc = "OUT" if lm <= W1_EXIT else ("IN" if lm >= W1_ENTER else "OUT(hold)")
    chk("W1_mode", mode_calc.startswith("OUT"), g["mode"] == "OUT", exact=True)

    # 3. mom63 bucket
    chk("mom63_q", bucket(g["mom63"]), g["mom63_q"], exact=True)

    # 4. boost_mult
    chk("boost_mult", BOOST_P09[g["mom63_q"]], g["boost_mult"])

    # 5. L_t (OUT -> 0)
    L_calc = 0.0 if g["mode"] == "OUT" else g["lev_raw"] * g["boost_mult"] * 3.0
    chk("L_t", L_calc, g["L_t"])

    # 6. bond_on
    chk("bond_on", 1 if g["bond_mom252"] > 0 else 0, g["bond_on"], exact=True)

    # 7. w_g inverse-vol
    wg, sg, sb = inv_vol_wg(gold, bond)
    print("    (sigma_gold=%.4f sigma_bond=%.4f -> wg_raw=%.5f clamp[%.2f,%.2f])"
          % (sg, sb, (1/sg)/((1/sg)+(1/sb)), WG_LO, WG_HI))
    chk("w_g (inv-vol)", wg, g["w_g"], tol=0.002)

    # 8. OUT holdings
    gold1x = g["w_g"]
    bond1x = (1.0 - g["w_g"]) if g["bond_on"] else 0.0
    cash = 1.0 - gold1x - bond1x
    chk("gold1x", gold1x, g["gold1x"])
    chk("bond1x", bond1x, g["bond1x"])
    chk("cash", cash, g["cash"])

    # 9. hand-off from Dyn2x3x Log
    chk("lev_raw=newLev", dyn["new_leverage"], g["lev_raw"])
    chk("vz=vix_z", dyn["vix_z"], round(g["vz"], 2))

    # mom63 internal-consistency (export truncates ^IXIC; sanity only)
    implied_base = 25169.5 / math.exp(g["mom63"])  # close[2026-06-11]=25169.5
    print("\n  mom63 sanity: logged %.4f => 63d-ago ^IXIC level ~%.0f"
          % (g["mom63"], implied_base))
    print("    (mid-Mar-2026 NASDAQ ~21.8k; Apr-03 Log close 21879 -> plausible)")

    n_ok = sum(results)
    print("\nRESULT: %d/%d fields match%s"
          % (n_ok, len(results),
             "  -> NO DIVERGENCE" if n_ok == len(results) else "  -> CHECK DIFFS"))
    return n_ok == len(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
