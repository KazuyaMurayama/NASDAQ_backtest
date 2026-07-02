"""
src/audit/click365_feasibility_20260702.py
==========================================
Task 2 of the P09C1/scale critical-verification campaign: cost-model
reconciliation vs click365 (Kuritsuku-kabu 365 NASDAQ-100 reset-tsuki) actual
economics, and the margin / forced-liquidation path.

PRODUCT FACTS (verified 2026-07-02, TFX/broker pages):
  - unit: NASDAQ-100 x 10 JPY per contract (~293,210 JPY notional at 29,321)
  - margin standard ~8,450 JPY/contract  -> margin ratio m ~ 2.88 percent of
    notional (product max leverage ~35x; revised weekly, vol-linked)
  - interest amount (buyer PAYS): (settle x 10) x rate x days/365 on the FULL
    notional; rate ~4.38-4.5 percent when USD short rate ~4.3 (spread ~0-0.3pp)
  - dividend amount (buyer RECEIVES) on full notional; index is the PRICE index
  - daily mark-to-market; margin deficit must be topped up next day; brokers
    force-liquidate near maintenance 100 percent (intraday)

REPO MODEL (k365_recost _build_nav_v7_tqqq_param):
  financing = max(L-1,0) x (SOFR + 0.50pc) + TER 0.86pc + max(L-3,0) x 0.25pc
  i.e. a TQQQ-swap structure where the 1x equity leg implicitly earns SOFR.

CLICK365 ECONOMICS (per unit equity, leverage L):
  L x (r_price + div) - L x (SOFR + delta) - commissions/spread on turnover,
  margin cash earns ~0.

This harness outputs:
  [1] annual pp reconciliation: repo-model charge vs click365-style charge on
      the same realized L path (delta in {0, 0.3pc} x div in {0, 0.8pc}),
      i.e. is the repo cost model optimistic or conservative, and by how much.
  [2] margin path: daily maintenance headroom under daily rebalancing
      (ratio = 1/(L x m)); the intraday index drop that would trigger
      forced liquidation at each day's L; min headroom day; comparison with
      the worst realized daily index moves (and a 1.5x intraday factor).
  [3] contract-count / capacity sanity for the labor-zero run sleeve (20M JPY).

ASCII-only prints. CSV -> audit_results/click365_feasibility_20260702.csv.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (_THIS, _REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MARGIN_RATIO = 0.0288          # 8,450 / 293,210 (2026-05 example)
UNIT_JPY = 293_210.0           # notional per contract at NDX 29,321
SLEEVE = 20_000_000.0          # labor-zero run sleeve
DIV_Y = 0.008                  # NDX dividend yield approx (price index basis)
DELTAS = (0.000, 0.003)        # k365 rate spread over SOFR scenarios
TRADING_DAYS = 252


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import src.audit.dd_reduction_harness_20260626 as H
    import src.audit.k365_recost_20260612 as K
    ctx = H.setup()
    shared = ctx["shared"]
    a = shared["assets"]
    dates = pd.DatetimeIndex(ctx["dates_dt"])
    sofr = np.asarray(a["sofr"], float)                    # daily rate
    close = a["close"]
    r_nas = pd.Series(close).pct_change().fillna(0).values

    print("=" * 100)
    print("CLICK365 FEASIBILITY (Task 2): cost reconciliation + margin path, sc2.6 / sc1.6")
    print("  product: unit=NDXx10JPY (~%.0f JPY), margin ratio m=%.2f%%, MTM daily"
          % (UNIT_JPY, MARGIN_RATIO * 100))
    print("=" * 100)

    rows = []
    for scale in (1.6, 2.6):
        # effective leverage path L (same construction as the NAV builder)
        lev_raw = np.asarray(shared["lev_raw_masked"], float)
        mult_v7 = K._build_v7_mult_custom(ctx["dates_dt"], H.STRONG_MAP)
        mult_v7 = np.asarray(mult_v7, float) * scale
        L = pd.Series(lev_raw * mult_v7 * 3.0, index=dates).shift(K.V7_DELAY).fillna(1.0).values
        wn = pd.Series(np.asarray(shared["wn"], float), index=dates).shift(K.V7_DELAY).fillna(0).values
        L_eff = np.where(wn > 0, L, 0.0)                    # NDX-leg leverage only
        in_days = L_eff > 0

        print("\n[scale %.1f] L stats (IN days, n=%d): mean=%.2f p95=%.2f max=%.2f"
              % (scale, in_days.sum(), L_eff[in_days].mean(),
                 np.percentile(L_eff[in_days], 95), L_eff[in_days].max()))

        # ---- [1] cost reconciliation (annualized pp per unit equity) ----
        # repo model charge on NDX leg:
        repo_chg = (np.maximum(L_eff - 1.0, 0.0) * (sofr + K.SWAP_SPREAD / TRADING_DAYS)
                    + np.where(in_days, K.TER_TQQQ / TRADING_DAYS, 0.0)
                    + np.maximum(L_eff - K.LEV_CAP, 0.0) * 0.0025 / TRADING_DAYS)
        for delta in DELTAS:
            for div in (0.0, DIV_Y):
                k365_chg = L_eff * (sofr + delta / TRADING_DAYS) - L_eff * div / TRADING_DAYS
                gap_annual = (np.sum(k365_chg - repo_chg) / len(sofr)) * TRADING_DAYS
                rows.append(dict(scale=scale, section="cost", delta=delta, div=div,
                                 gap_pp_per_yr=gap_annual * 100))
                print("  cost gap (k365-actual minus repo-model): delta=%.1f%% div=%.1f%%"
                      " -> %+.2f pp/yr (positive = repo model OPTIMISTIC)"
                      % (delta * 100, div * 100, gap_annual * 100))

        # ---- [2] margin path under daily rebalancing ----
        # maintenance ratio each day (post-rebalance) = equity/(m * L * equity) = 1/(m L)
        with np.errstate(divide="ignore"):
            maint = np.where(L_eff > 0, 1.0 / (MARGIN_RATIO * np.maximum(L_eff, 1e-9)), np.inf)
        # intraday index drop that would push equity below required margin intraday:
        # equity(1 + L r) < m L equity (1+r)  ~=  r* = (m L - 1) / (L (1 - m))
        with np.errstate(divide="ignore"):
            r_star = np.where(L_eff > 0, (MARGIN_RATIO * L_eff - 1.0)
                              / (L_eff * (1.0 - MARGIN_RATIO)), -np.inf)
        i_min = int(np.argmin(np.where(in_days, maint, np.inf)))
        print("  maintenance ratio (post-rebalance): min=%.0f%% on %s (L=%.2f)"
              % (maint[i_min] * 100, dates[i_min].date(), L_eff[i_min]))
        print("  forced-liquidation intraday index move at max L: %.1f%%"
              % (r_star[i_min] * 100))
        # worst realized daily index moves vs threshold (with 1.5x intraday factor)
        worst_idx = np.argsort(r_nas)[:5]
        breach_close = int(np.sum(r_nas < r_star))
        breach_intra = int(np.sum(1.5 * r_nas < r_star))
        print("  worst NDX daily closes: %s"
              % ", ".join("%s %.1f%%" % (dates[i].date(), r_nas[i] * 100) for i in worst_idx))
        print("  days breaching forced-liq at that day's L: close-basis=%d, 1.5x-intraday=%d"
              % (breach_close, breach_intra))
        # margin-call (not forced-liq) frequency: daily loss > (1 - m L ... ) simpler:
        # deficit occurs when equity after MTM < required margin for the SAME position:
        eq_after = 1.0 + L_eff * r_nas
        req_same = MARGIN_RATIO * L_eff * (1.0 + r_nas)
        margin_call_days = int(np.sum((eq_after < req_same) & in_days))
        rows.append(dict(scale=scale, section="margin", delta=np.nan, div=np.nan,
                         min_maint_pct=maint[i_min] * 100,
                         r_star_pct=r_star[i_min] * 100,
                         breach_close=breach_close, breach_intra=breach_intra,
                         margin_call_days=margin_call_days))
        print("  margin-deficit (top-up next day) days: %d" % margin_call_days)

        # ---- [3] capacity ----
        contracts = SLEEVE * L_eff[in_days].max() / UNIT_JPY
        print("  contracts at max L for %.0fM sleeve: %.0f (TFX per-account caps are O(10^4) -> OK)"
              % (SLEEVE / 1e6, contracts))
        rows.append(dict(scale=scale, section="capacity", delta=np.nan, div=np.nan,
                         contracts_max=contracts))

    out = os.path.join(_REPO, "audit_results", "click365_feasibility_20260702.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s" % out)
    print("Done (click365 feasibility).")


if __name__ == "__main__":
    main()
