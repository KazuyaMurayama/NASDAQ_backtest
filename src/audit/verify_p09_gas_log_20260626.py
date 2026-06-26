"""
src/audit/verify_p09_gas_log_20260626.py
========================================
P09Log (GAS live) vs Python independent recomputation -- 5-day divergence
check (2026-06-15 .. 2026-06-19, the first parallel-run week).

Re-derives every field of each P09Log row from FIRST PRINCIPLES (frozen-
spec formulas + raw spreadsheet inputs) and compares to GAS.

Input: audit_results/p09_gas_log_snapshot_20260626.json captured from the
production spreadsheet (5 P09Log rows + GC=F/TLT tails + same-day Dyn2x3x
Log new_leverage/vix_z for the hand-off).

Per-day checks (recomputed vs logged):
  lev_mod_065, W1 hysteresis transition (replayed across days),
  mom63 bucket, boost_mult, L_t, bond_on, OUT holdings, rebalance flag,
  and the lev_raw/vz hand-off.
w_g (inverse-vol) is recomputed from the GC=F/TLT tails for the days the
export still covers (through 2026-06-18); later days fall back to
internal consistency (w_g held / clamp).

ASCII-only. Read-only.
"""
from __future__ import annotations

import json
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
SNAP = os.path.join(_REPO_DIR, "audit_results", "p09_gas_log_snapshot_20260626.json")

K_LO, K_MID, K_HI, VZ_THR = 0.1, 0.5, 0.8, 0.65
W1_ENTER, W1_EXIT = 0.7, 0.3
Q25, Q50, Q75 = -0.028802750000000002, 0.0375715, 0.09098425
BOOST_P09 = {0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00}
VOL_WIN, WG_LO, WG_HI = 63, 0.25, 0.75
L_DELTA, W_DELTA = 0.25, 0.15
TOL = 0.0011


def f(x):
    return float(x)


def lev_mod_065(lev_raw, vz, lt):
    k = K_HI if vz > VZ_THR else (K_LO if vz < -VZ_THR else K_MID)
    bias = max(-0.5, min(0.5, -k * lt * 0.5))
    return max(0.0, min(1.0, lev_raw + bias))


def bucket(m):
    return 0 if m <= Q25 else 1 if m <= Q50 else 2 if m <= Q75 else 3


def inv_vol_wg(gold, bond):
    def sig(cl):
        r = [cl[i] / cl[i - 1] - 1 for i in range(1, len(cl))]
        seg = r[-VOL_WIN:]
        m = sum(seg) / VOL_WIN
        var = sum((x - m) ** 2 for x in seg) / (VOL_WIN - 1)
        return (var ** 0.5) * (252 ** 0.5)
    sg, sb = sig(gold), sig(bond)
    wg = (1.0 / sg) / ((1.0 / sg) + (1.0 / sb))
    return max(WG_LO, min(WG_HI, wg))


def _contiguous_tail(series):
    """Drop everything before the last >30-day calendar gap (removes the
    2016-2017 ^IXIC values that leaked into the GC=F magnitude filter)."""
    import datetime
    ds = [(datetime.date.fromisoformat(d), d, v) for d, v in series]
    cut = 0
    for i in range(1, len(ds)):
        if (ds[i][0] - ds[i - 1][0]).days > 30:
            cut = i
    return [(d, v) for _, d, v in ds[cut:]]


def main():
    snap = json.load(open(SNAP, encoding="utf-8"))
    rows = snap["rows"]
    gold_list = _contiguous_tail(snap["gold"])
    bond_list = _contiguous_tail(snap["bond"])
    gold = {d: v for d, v in gold_list}
    bond = {d: v for d, v in bond_list}
    gold_dates = [d for d, _ in gold_list]
    bond_dates = [d for d, _ in bond_list]
    dyn = snap["dyn"]
    # GAS updates w_g only on run_count==1 (first run) and every 5th run.
    # Here run_count = day index+1 over this window -> updates on 1st and 5th day.
    # GAS updates w_g on run_count==1 then every 5th (run 5,10,15,...).
    # Over this contiguous window run_count = day-index+1.
    update_days = set()
    for k, rr in enumerate(rows):
        rc = k + 1
        if rc == 1 or rc % 5 == 0:
            update_days.add(rr["date"])

    print("=" * 92)
    print("P09Log (GAS) vs Python recomputation  --  %d days  %s .. %s"
          % (len(rows), rows[0]["date"], rows[-1]["date"]))
    print("=" * 92)

    # W1 hysteresis replay (initial state OUT per frozen spec)
    state_in = False
    total = 0
    fails = 0
    prev = None  # previous day's (mode, L, gold1x, bond1x) for rebalance replay

    for r in rows:
        dt = r["date"]
        lev_raw, vz, lt = f(r["lev_raw"]), f(r["vz"]), f(r["lt_sig"])
        diffs = []

        # lev_mod_065
        lm = lev_mod_065(lev_raw, vz, lt)
        diffs.append(("lev_mod_065", abs(lm - f(r["lev_mod_065"])) <= TOL,
                      "%.4f|%.4f" % (lm, f(r["lev_mod_065"]))))

        # W1 transition
        if not state_in and lm >= W1_ENTER:
            state_in = True
        elif state_in and lm <= W1_EXIT:
            state_in = False
        mode_calc = "IN" if state_in else "OUT"
        diffs.append(("W1_mode", mode_calc == r["mode"], "%s|%s" % (mode_calc, r["mode"])))

        # mom63 bucket / boost
        q = bucket(f(r["mom63"]))
        diffs.append(("mom63_q", q == int(r["mom63_q"]), "%d|%s" % (q, r["mom63_q"])))
        bm = BOOST_P09[q]
        diffs.append(("boost_mult", abs(bm - f(r["boost_mult"])) <= TOL, "%.2f|%s" % (bm, r["boost_mult"])))

        # L_t
        L = lev_raw * bm * 3.0 if mode_calc == "IN" else 0.0
        diffs.append(("L_t", abs(L - f(r["L_t"])) <= TOL, "%.3f|%s" % (L, r["L_t"])))

        # bond_on
        bon = 1 if f(r["bond_mom252"]) > 0 else 0
        diffs.append(("bond_on", bon == int(r["bond_on"]), "%d|%s" % (bon, r["bond_on"])))

        # w_g: GAS updates only on update_days; otherwise it HOLDS the prior value.
        wg_logged = f(r["w_g"])
        if dt in update_days:
            # recompute inverse-vol from prices AVAILABLE at run time: the GAS
            # job runs 07:00 JST on `dt`, before that day's US close, so the
            # latest GC=F/TLT close it has is the prior trading day (< dt).
            avail = [d for d in gold_dates if d < dt and d in bond]
            if len(avail) > VOL_WIN:
                last = avail[-1]
                gser = [gold[d] for d in gold_dates if d <= last]
                bser = [bond[d] for d in bond_dates if d <= last]
                wg = inv_vol_wg(gser, bser)
                diffs.append(("w_g UPDATE", abs(wg - wg_logged) <= 0.003,
                              "calc %.4f|logged %.4f (asof %s)" % (wg, wg_logged, last)))
            else:
                diffs.append(("w_g UPDATE(clamp)", WG_LO <= wg_logged <= WG_HI,
                              "%.4f price-window short" % wg_logged))
        else:
            held_ok = abs(wg_logged - f(prev["wg"])) <= TOL if prev else True
            diffs.append(("w_g HELD", held_ok,
                          "%.4f == prev %.4f" % (wg_logged, f(prev["wg"]) if prev else wg_logged)))

        # OUT holdings
        if mode_calc == "OUT":
            g1 = wg_logged
            b1 = (1.0 - wg_logged) if bon else 0.0
            cash = 1.0 - g1 - b1
            diffs.append(("gold1x", abs(g1 - f(r["gold1x"])) <= TOL, "%.4f|%s" % (g1, r["gold1x"])))
            diffs.append(("bond1x", abs(b1 - f(r["bond1x"])) <= TOL, "%.4f|%s" % (b1, r["bond1x"])))
            diffs.append(("cash", abs(cash - f(r["cash"])) <= TOL, "%.4f|%s" % (cash, r["cash"])))

        # rebalance replay (mode flip / |dL|>=0.25 / OUT-weight >0.15)
        if prev is not None:
            mode_ch = prev["mode"] != mode_calc
            l_ch = abs(L - prev["L"]) >= L_DELTA
            w_ch = False
            if mode_calc == "OUT" and prev["mode"] == "OUT":
                w_ch = (abs(f(r["gold1x"]) - prev["g1"]) > W_DELTA or
                        abs(f(r["bond1x"]) - prev["b1"]) > W_DELTA)
            reb = "YES" if (mode_ch or l_ch or w_ch) else "NO"
            diffs.append(("rebalance", reb == r["rebalance"],
                          "%s|%s (mode%d L%d w%d)" % (reb, r["rebalance"], mode_ch, l_ch, w_ch)))
        else:
            diffs.append(("rebalance(1st)", r["rebalance"] == "NO", "first run->NO|%s" % r["rebalance"]))
        prev = {"mode": mode_calc, "L": L, "g1": f(r["gold1x"]),
                "b1": f(r["bond1x"]), "wg": r["w_g"]}

        # hand-off
        dd = dyn.get(dt, {})
        if dd:
            diffs.append(("lev_raw=newLev", abs(f(dd["new_lev"]) - lev_raw) <= TOL,
                          "%s|%.4f" % (dd["new_lev"], lev_raw)))
            diffs.append(("vz=vix_z", abs(f(dd["vix_z"]) - round(vz, 2)) <= TOL,
                          "%s|%.2f" % (dd["vix_z"], round(vz, 2))))

        day_fail = sum(1 for _, ok, _ in diffs if not ok)
        total += len(diffs); fails += day_fail
        status = "OK" if day_fail == 0 else "**%d DIFF**" % day_fail
        print("\n%s  %s  reb=%s  %s" % (dt, r["mode"], r["rebalance"], status))
        for name, ok, msg in diffs:
            print("    %-16s %s  %s" % (name, "ok " if ok else "DIF", msg))

    print("\n" + "=" * 92)
    print("TOTAL: %d/%d field-checks match across %d days  ->  %s"
          % (total - fails, total, len(rows),
             "NO DIVERGENCE" if fails == 0 else "%d DIFFERENCES" % fails))
    return fails == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
