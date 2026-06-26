"""
src/audit/dd_export_series_20260626.py
======================================
Export the daily NAV/return series for X4 and N4 (and sc2.0 reference) so an
INDEPENDENT agent can recompute the standard-10 metrics from scratch without
touching my metrics10() code. This script outputs ONLY raw series (date, nav,
daily return) -- NO metric computation -- so the metric logic can be cross-checked
by a clean-room reimplementation.

Candidates (from SC2_DD_REDUCTION_RESULTS_20260626.md):
  sc2.0 : scale=2.0
  N4    : scale=2.85, in_gold_w=0.28, in_bond_w=0.07
  X4    : scale=3.0,  in_gold_w=0.32, in_bond_w=0.05

Output: audit_results/dd_series_export_20260626.csv with columns
  date, ret_sc20, nav_sc20, ret_N4, nav_N4, ret_X4, nav_X4
plus a small meta file dd_series_meta_20260626.txt documenting IS/OOS split.
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

import src.audit.dd_reduction_harness_20260626 as H
from src.audit.unified_metrics import IS_END, OOS_START

CANDS = {
    "sc20": dict(scale=2.0),
    "N4":   dict(scale=2.85, in_gold_w=0.28, in_bond_w=0.07),
    "X4":   dict(scale=3.0, in_gold_w=0.32, in_bond_w=0.05),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ctx = H.setup()
    dates_dt = ctx["dates_dt"]
    out = {"date": [d.strftime("%Y-%m-%d") for d in dates_dt]}
    for name, kw in CANDS.items():
        nav_dt, r, tpy, exc = H.build(ctx, **kw)
        out["ret_%s" % name] = np.asarray(r, float)
        out["nav_%s" % name] = np.asarray(nav_dt.values, float)
        out["tpy_%s" % name] = [round(float(tpy), 4)] * len(dates_dt)
    df = pd.DataFrame(out)
    p = os.path.join(_REPO_DIR, "audit_results", "dd_series_export_20260626.csv")
    df.to_csv(p, index=False, float_format="%.10f", encoding="utf-8-sig")

    meta = os.path.join(_REPO_DIR, "audit_results", "dd_series_meta_20260626.txt")
    with open(meta, "w", encoding="utf-8") as f:
        f.write("DD candidate series export 2026-06-26\n")
        f.write("rows=%d  first=%s  last=%s\n" % (len(df), out["date"][0], out["date"][-1]))
        f.write("IS_END=%s  OOS_START=%s\n" % (pd.Timestamp(IS_END).strftime("%Y-%m-%d"),
                                               pd.Timestamp(OOS_START).strftime("%Y-%m-%d")))
        f.write("TRADING_DAYS=252  AFTER_TAX=0.8273\n")
        f.write("Trades/yr (tpy) per candidate: " +
                ", ".join("%s=%.2f" % (k, df["tpy_%s" % k].iloc[0]) for k in CANDS) + "\n")
    print("Saved series: %s (rows=%d)" % (p, len(df)))
    print("Saved meta:   %s" % meta)
    # quick sanity (not the verification): print endpoints
    for name in CANDS:
        nav = df["nav_%s" % name].values
        print("  %-5s nav_first=%.6f nav_last=%.4f" % (name, nav[0], nav[-1]))


if __name__ == "__main__":
    main()
