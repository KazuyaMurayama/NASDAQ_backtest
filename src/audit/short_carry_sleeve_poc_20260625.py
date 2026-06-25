"""
src/audit/short_carry_sleeve_poc_20260625.py
============================================
Stage A PoC for the OUT-period SHORT carry sleeve (くりっく株365).

Question (sign only): on DH-W1 OUT days, does a SHORT position's expected
daily return on equity -- price P&L (-L*ret_ndx) PLUS net carry
(+(sofr+spread-div)*L/252) -- beat sitting in CASH (0%)?

Measures, on OUT days only, for L in {1.0, 2.0, 3.0}:
  - mean OUT-day NDX price return (annualized)  [the tailwind for a short]
  - annualized gross short price return  = -L * that
  - annualized net carry on equity       = +(mean_sofr + SPREAD - DIV)*L
  - annualized short total (pre-tax, pre-roll)
  - after-tax short total (x0.8273) minus roll cost
Compares to CASH=0%. ASCII-only prints. No temp files, no commit here.
Writes audit_results/short_carry_sleeve_poc_20260625.csv.
"""
from __future__ import annotations
import os, sys, types

# cash_sleeve_sim stubs 'multitasking'; replicate so yfinance-style imports don't pull threads.
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_THIS)
_REPO = os.path.dirname(_SRC)
os.chdir(_REPO)
sys.path.insert(0, _SRC)

from g14_wfa_sbi_cfd import load_shared_assets
from g23a_dh_refinement_variants import build_W1
from product_costs import (
    K365_FINANCING_SPREAD, K365_MARGIN_ROLL_COST, TQQQ,
)

TRADING_DAYS = 252
AFTER_TAX = 0.8273
DIV_YIELD = TQQQ.dividend_yield        # NDX dividend yield proxy = 0.003
SPREAD = K365_FINANCING_SPREAD         # 0.0075
ROLL = K365_MARGIN_ROLL_COST           # 0.0020 /yr on notional
LEVERAGES = [1.0, 2.0, 3.0]


def ann_from_daily_mean(daily_mean):
    """Annualize a daily simple-return mean by compounding 252 of them."""
    return (1.0 + daily_mean) ** TRADING_DAYS - 1.0


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 90)
    print('SHORT CARRY SLEEVE PoC (OUT days) -- expected-value SIGN check  2026-06-25')
    print('=' * 90)

    a = load_shared_assets()
    nav_w1, cost_w1, mask, wn_w1, lev_w1 = build_W1(a)
    mask = np.asarray(mask, dtype=float)
    out = (mask < 0.5)

    close = np.asarray(a['close'], dtype=float)
    ret_ndx = np.concatenate([[0.0], np.diff(close) / close[:-1]])
    ret_ndx = np.nan_to_num(ret_ndx, nan=0.0)
    # NOTE: load_sofr() returns a PER-DAY rate (it divides annual DTB3 by 252
    # internally). So a['sofr'] is daily; SPREAD/DIV_YIELD are annual -> /252.
    sofr_daily = np.asarray(a['sofr'], dtype=float)  # PER-DAY rate (decimal)

    n_out = int(out.sum())
    out_ndx = ret_ndx[out]
    out_sofr_daily = sofr_daily[out]
    mean_ndx_daily = float(np.mean(out_ndx))
    ann_ndx = ann_from_daily_mean(mean_ndx_daily)    # OUT-day NDX annualized (expect ~ -7.6%)
    mean_sofr_ann = float(np.mean(out_sofr_daily)) * TRADING_DAYS   # annualized

    print('OUT days = %d  (%.1f%% of %d)' % (n_out, 100*n_out/len(out), len(out)))
    print('OUT-day NDX annualized price return = %+.2f%%  (short tailwind if negative)'
          % (ann_ndx * 100))
    print('OUT-day mean SOFR (annual) = %.2f%%   spread=%.2f%%  div=%.2f%%'
          % (mean_sofr_ann*100, SPREAD*100, DIV_YIELD*100))

    rows = []
    cash_ann = 0.0
    for L in LEVERAGES:
        # Per-day short return on equity, OUT days only, then annualize by compounding.
        # sofr is already per-day; spread & div are annual -> /252.
        carry_daily = (out_sofr_daily + (SPREAD - DIV_YIELD) / TRADING_DAYS) * L
        price_daily = -L * out_ndx
        short_daily = price_daily + carry_daily
        cum = float(np.prod(1.0 + short_daily))
        yrs = len(short_daily) / TRADING_DAYS
        ann_short_pre = cum ** (1.0/yrs) - 1.0 if cum > 0 else -1.0
        # carry-only and price-only annualized for attribution
        ann_carry = (mean_sofr_ann + SPREAD - DIV_YIELD) * L      # ~ linear, report as such
        ann_price = -L * ann_ndx
        # after-tax + roll (roll on notional = L * ROLL)
        ann_short_aftertax = ann_short_pre * AFTER_TAX - L * ROLL
        rows.append(dict(
            leverage=L,
            ann_price_pct=round(ann_price*100, 2),
            ann_carry_pct=round(ann_carry*100, 2),
            ann_short_pretax_pct=round(ann_short_pre*100, 2),
            ann_short_aftertax_pct=round(ann_short_aftertax*100, 2),
            beats_cash=bool(ann_short_aftertax > cash_ann),
        ))
        print('  L=%.1f | price=%+6.2f%% carry=%+6.2f%% | short_pretax=%+6.2f%% '
              'short_aftertax(-roll)=%+6.2f%%  beats_cash=%s'
              % (L, ann_price*100, ann_carry*100, ann_short_pre*100,
                 ann_short_aftertax*100, ann_short_aftertax > cash_ann))

    pd.DataFrame(rows).to_csv(
        os.path.join(_REPO, 'audit_results', 'short_carry_sleeve_poc_20260625.csv'),
        index=False, encoding='utf-8-sig')
    print('\nSaved PoC CSV. CASH baseline on OUT days = +0.00%/yr by definition.')
    print('DECISION SIGN: short beats cash at L where beats_cash=True.')


if __name__ == '__main__':
    main()
