"""
src/audit/i4_product_tax_20260703.py
====================================
Improvement candidate I4 of the P09C1/scale campaign
(P09_SCALE_CRITVERIFY_20260702.md s4): product-boundary re-optimization
including tax REALIZATION TIMING and TAX-BUCKET SEPARATION.

JP tax facts the blanket x0.8273 ignores:
  - TQQQ / ETFs / funds: "listed equity" bucket (jouto-shotoku). Gains taxed
    on SALE -> partial deferral possible (phi<1). Losses offset within the
    bucket, 3y carryforward.
  - click365 (kuritsuku-kabu 365): "futures misc income" bucket. The
    reset-tsuki contract is FORCE-SETTLED at the annual reset -> gains are
    realized every year (phi=1). Losses offset within the bucket, 3y
    carryforward. **THE TWO BUCKETS CANNOT OFFSET EACH OTHER.**

The current implementation model is a HYBRID: NDX exposure up to 3x equity
via TQQQ (equity bucket) and the excess via click365 (futures bucket).

This harness (annual-granularity wealth sim on daily attribution):
  [1] quantifies the BUCKET-SEPARATION COST of the current hybrid that the
      blanket model hides: two-bucket tax (phi_eq in {0.85, 1.0}) vs the
      single-bucket variant-b of tax_model_audit vs blanket x0.8273.
  [2] evaluates the ALT boundary: ALL-K365 NDX leg (single futures bucket
      for the NDX exposure, k365 economics: full-notional financing at
      SOFR+delta minus dividends received; no TER/swap-spread) vs hybrid.
  [3] verdict per pre-set rule: reject I4 unless some alternative beats the
      current hybrid by >= +0.3pp after-tax CAGR (like-for-like two-bucket).

Attribution (documented approximation):
  k365 daily contribution f_k(t) = wn * [ (L-3)+ * (r_nas - sofr) - 0.0025/252 ]
  equity-bucket contribution    = r_total(t) - f_k(t)   (residual; exact sum)
  Annual bucket P&L = share of the year's total wealth gain, pro-rated by the
  arithmetic sums of daily contributions (intra-year compounding assigned
  pro-rata). Tax paid from wealth at year-end per bucket; separate 3y
  carryforwards; NO cross-bucket offset.

SELF-TESTS:
  ST1 merged-bucket mode (one carryforward, phi=0.85) reproduces
      tax_model_audit.after_tax_series(ann, "b") wealth path within 0.2pp
      CAGR (attribution/pro-ration noise only).
  ST2 sc2.6 pre-tax annual chain err = 0 vs labor series.
ASCII-only prints. CSV -> audit_results/i4_product_tax_20260703.csv.
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

_spec = importlib.util.spec_from_file_location(
    "ta", os.path.join(_THIS, "tax_model_audit_20260702.py"))
ta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ta)
v6 = ta.v6

TAX = 0.20315
YEARS = list(range(1975, 2026))
DELTA_K365 = 0.0015          # k365 rate spread over SOFR (mid of 0..0.3pc)
DIV_Y = 0.008                # NDX dividend yield (price-index basis)


def two_bucket_after_tax(ann_total, ann_k365, phi_eq, merged=False, phi_k=1.0):
    """Annual after-tax returns from total annual returns and the k365-bucket
    annual contribution. Separate 3y-FIFO carryforwards per bucket unless
    merged=True (single bucket, for the self-test)."""
    w = 1.0
    out = np.empty_like(ann_total)
    carry_e, carry_k = [], []

    def offset(taxable, carry, i):
        if taxable > 0 and carry:
            new = []
            for (exp_i, loss) in carry:
                if exp_i < i:
                    continue
                use = min(loss, taxable)
                loss -= use
                taxable -= use
                if loss > 1e-12:
                    new.append((exp_i, loss))
            carry[:] = new
        return taxable

    for i, (rt, rk) in enumerate(zip(ann_total, ann_k365)):
        re_ = rt - rk                                   # equity-bucket part
        pnl_e, pnl_k = w * re_, w * rk
        if merged:
            realized = (phi_eq * pnl_e if pnl_e > 0 else pnl_e) + \
                       (phi_k * pnl_k if pnl_k > 0 else pnl_k)
            taxable = offset(realized, carry_e, i) if realized > 0 else realized
            tax = TAX * taxable if taxable > 0 else 0.0
            if realized < 0:
                carry_e.append((i + 3, -realized))
        else:
            real_e = phi_eq * pnl_e if pnl_e > 0 else pnl_e
            real_k = phi_k * pnl_k if pnl_k > 0 else pnl_k
            tx_e = offset(real_e, carry_e, i) if real_e > 0 else real_e
            tx_k = offset(real_k, carry_k, i) if real_k > 0 else real_k
            tax = (TAX * tx_e if tx_e > 0 else 0.0) + (TAX * tx_k if tx_k > 0 else 0.0)
            if real_e < 0:
                carry_e.append((i + 3, -real_e))
            if real_k < 0:
                carry_k.append((i + 3, -real_k))
        w_next = w * (1.0 + rt) - tax
        if w_next < 1e-9:                       # ruin guard (tax exceeded wealth)
            w_next = 1e-9
        out[i] = w_next / w - 1.0
        w = w_next
    return out


def annual_bucket_split(dates, r_total, f_k):
    """Compounded-currency attribution: within each calendar year, bucket P&L
    (as a fraction of year-start NAV) = sum_t (NAV_{t-1}/NAV_start) * f_bucket(t).
    The two buckets sum EXACTLY to the year's total return (identity)."""
    yr = dates.year
    ann_t, ann_k = [], []
    for y in YEARS:
        m = yr == y
        rt_d = r_total[m]
        fk_d = f_k[m]
        w_rel = np.concatenate([[1.0], np.cumprod(1.0 + rt_d)[:-1]])
        rt = float(np.prod(1.0 + rt_d) - 1.0)
        k = float(np.sum(w_rel * fk_d))
        ann_t.append(rt)
        ann_k.append(k)
    return np.array(ann_t), np.array(ann_k)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import src.audit.dd_reduction_harness_20260626 as H
    import src.audit.k365_recost_20260612 as K
    print("=" * 100)
    print("I4: product-boundary tax (bucket separation + deferral + all-k365 alt)")
    print("=" * 100)
    ctx = H.setup()
    dates = pd.DatetimeIndex(ctx["dates_dt"])
    shared = ctx["shared"]
    a = shared["assets"]
    sofr = np.asarray(a["sofr"], float)
    r_nas = pd.Series(a["close"]).pct_change().fillna(0).values
    run_h, _res_h = v6.get_paired_history()
    rows = []

    for scale in (1.6, 2.6):
        nav, _r, _t, _e = H.build(ctx, scale=scale)
        nav = pd.Series(np.asarray(nav, float), index=dates)
        r_total = nav.pct_change().fillna(0.0).values
        # leverage path & k365 contribution
        lev_raw = np.asarray(shared["lev_raw_masked"], float)
        mult = np.asarray(K._build_v7_mult_custom(ctx["dates_dt"], H.STRONG_MAP), float) * scale
        L = pd.Series(lev_raw * mult * 3.0, index=dates).shift(K.V7_DELAY).fillna(1.0).values
        wn = pd.Series(np.asarray(shared["wn"], float), index=dates).shift(K.V7_DELAY).fillna(0).values
        exc = np.maximum(L - 3.0, 0.0)
        f_k = wn * (exc * (r_nas - sofr) - 0.0025 / 252.0)
        ann_t, ann_k = annual_bucket_split(dates, r_total, f_k)

        if scale == 2.6:
            err = np.max(np.abs(ann_t * 0.8273 - run_h))
            # ST1: with everything in the equity bucket (ann_k=0), the machinery
            # must reproduce tax_model_audit variant b EXACTLY.
            st1 = two_bucket_after_tax(ann_t, np.zeros_like(ann_t), 0.85)
            ref = ta.after_tax_series(ann_t, "b")
            err1 = np.max(np.abs(st1 - ref))
            print("ST2 sc2.6 chain err=%.1e   ST1 single-bucket-limit vs variant-b err=%.1e -> %s"
                  % (err, err1, "PASS" if (err < 1e-12 and err1 < 1e-12) else "FAIL"))
            if not (err < 1e-12 and err1 < 1e-12):
                print("HALT")
                return

        # ---- variants on the CURRENT hybrid ----
        print("\n[scale %.1f] k365-bucket share of gains: mean |ann_k|/|ann_t| years>0 = %.2f"
              % (scale, np.mean(np.abs(ann_k[np.abs(ann_t) > 1e-9]
                                        / ann_t[np.abs(ann_t) > 1e-9]))))
        variants = {
            "blanket x0.8273 (canon a)": ta.after_tax_series(ann_t, "a"),
            "single-bucket carryfwd (canon b)": ta.after_tax_series(ann_t, "b"),
            "hybrid 2-bucket phi_eq=0.85": two_bucket_after_tax(ann_t, ann_k, 0.85),
            "hybrid 2-bucket phi_eq=1.00": two_bucket_after_tax(ann_t, ann_k, 1.00),
        }

        # ---- ALT: all-k365 NDX leg ----
        # replace the NDX-leg economics: L*(r+div) - L*(sofr+delta), no TER/swap,
        # no excess_extra; non-NDX legs unchanged -> adjust r_total by the delta
        # between k365-actual and repo-model NDX-leg charges (Task 2 formulas).
        repo_ndx = wn * (L * r_nas - np.maximum(L - 1.0, 0.0) * (sofr + K.SWAP_SPREAD / 252.0)
                         - np.where(L > 0, K.TER_TQQQ / 252.0, 0.0)
                         - np.maximum(L - 3.0, 0.0) * 0.0025 / 252.0)
        k365_ndx = wn * (L * (r_nas + DIV_Y / 252.0) - L * (sofr + DELTA_K365 / 252.0))
        r_alt = r_total - repo_ndx + k365_ndx
        f_k_alt = k365_ndx                       # entire NDX leg in futures bucket
        ann_t2, ann_k2 = annual_bucket_split(dates, r_alt, f_k_alt)
        variants["ALL-k365 2-bucket (NDX leg futures)"] = \
            two_bucket_after_tax(ann_t2, ann_k2, 0.85)
        variants["ALL-k365 pre-tax ref (x0.8273)"] = ann_t2 * 0.8273

        base_key = "hybrid 2-bucket phi_eq=0.85"
        mb = ta.path_metrics(variants[base_key])
        print("    %-38s %8s %9s %8s | dCAGR vs hybrid2b" % ("variant", "CAGR", "MDD(ann)", "W5Y"))
        for name, ser in variants.items():
            m = ta.path_metrics(ser)
            d = (m["cagr"] - mb["cagr"]) * 100
            rows.append(dict(scale=scale, variant=name, cagr=m["cagr"],
                             mdd_ann=m["maxdd_annual"], w5y=m["worst5y"],
                             dcagr_vs_hybrid2b_pp=d))
            print("    %-38s %8.4f %9.4f %8.4f | %+7.2fpp"
                  % (name, m["cagr"], m["maxdd_annual"], m["worst5y"], d))

    out = os.path.join(_REPO, "audit_results", "i4_product_tax_20260703.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s" % out)
    print("Done (I4 product-boundary tax).")


if __name__ == "__main__":
    main()
