# OUT-Period Short Carry Sleeve (くりっく株365) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backtest whether, during DH-W1 OUT periods (regime gate = cash, L=0), taking a SHORT くりっく株365 position to harvest net carry (金利相当額 received − 配当相当額 paid) plus the price move improves the strategy versus sitting in cash or in a 1x Gold/Bond sleeve.

**Architecture:** Mirror the existing cash-sleeve simulator (`analysis_cash_sleeve/cash_sleeve_sim.py`). It already splices alternative-asset returns into OUT days via `r_enh = np.where(fund_active, r_sleeve, r_w1)`. We add a SHORT sleeve whose OUT-day daily return is `r_short = -L*ret_ndx + (sofr_daily + spread - div_yield)*L/252 - costs`, evaluate it on the same standard-metrics + WFA harness with the same T+? lag and after-tax factor, and compare head-to-head against BASELINE cash and the adopted Gold/Bond sleeves. A two-stage approach: **Stage A = a cheap PoC** that measures the expected-value SIGN before any full build; **Stage B = the full sleeve + metrics + WFA** only if Stage A is favorable.

**Tech Stack:** Python 3, numpy, pandas. Reuses `src/g14_wfa_sbi_cfd.py` (`load_shared_assets`, `generate_windows`), `src/g23a_dh_refinement_variants.py` (`build_W1`), `src/g18_daily_trade_cost_wfa.py` (`metrics_from_nav`, `apply_tax_etf_decimal`, `wfa_metrics`), `src/product_costs.py` (k365 constants), `src/compute_cfd_worst10y.py` (`prepare_gold_local`), `src/corrected_strategy_backtest.py` (`build_bond_1x_nav_corrected`). No new external data.

---

## Domain Background (read before starting)

**The thesis.** During OUT days the DH-W1 regime gate sits in cash (L=0). The cash-sleeve study (`analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md`, Part 1) measured that **NASDAQ 1x during OUT days returned ≈ −7.63%/yr at Vol 26.3%** — i.e. the gate tends to be OUT when NASDAQ is falling. A SHORT during those days earns the *opposite* price move (≈ +7.6%/yr gross, before carry) PLUS net carry. That single number is why this is worth testing — and also why the PoC must re-measure it precisely from current code, not assume it.

**The carry sign (verified from primary sources + `src/product_costs.py`).** くりっく株365 (Tokyo Financial Exchange CFD):
- **Long** pays 金利相当額, receives 配当相当額.
- **Short** RECEIVES 金利相当額, PAYS 配当相当額.
- 金利相当額 (gross) ≈ SOFR + 0.75pp (`K365_FINANCING_SPREAD = 0.0075`). 配当相当額 ≈ NDX dividend yield ≈ 0.3%/yr (`TQQQ.dividend_yield = 0.003`).
- Net carry for a short, per unit notional ≈ `(sofr + 0.0075 − 0.003)` per year. At L× notional, the carry on equity ≈ that × L. This carry is **proportional to notional (= equity × effective leverage L), NOT to the financing RATE scaling with L**. The rate stays one rate; leverage multiplies the notional.

**Sign conventions (lock these in — getting a sign wrong silently inverts the result):**
- `ret_ndx[t]` = NDX daily price return.
- Short price P&L on equity for the day = `−L * ret_ndx[t]` (NDX up ⇒ short loses).
- Carry on equity per day = `+(sofr[t] + K365_FINANCING_SPREAD − DIV_YIELD) * L / 252` (positive ⇒ income).
- Net OUT-day short return on equity = price P&L + carry − per-day costs.

**Why a PoC first.** The carry is a near-certain positive (~+9–11%/yr on equity at L=3). The price leg is a coin-flip-ish ± that leverage AMPLIFIES. The whole question is whether OUT-day price moves are negative *enough, often enough* that `(short price P&L + carry)` beats cash. Stage A answers the sign cheaply before we invest in the full WFA build.

**Existing splice template (the pattern to copy).** In `cash_sleeve_sim.py`:
```python
fund_active = np.zeros(n, dtype=bool)
fund_active[LAG_DAYS:] = out[:-LAG_DAYS]          # forward-shift the OUT mask by the execution lag
r_enh = np.where(fund_active, r_blend - fee_daily, r_w1)   # OUT days -> sleeve return; else baseline
```
The short sleeve replaces `r_blend - fee_daily` with the short-carry return defined above. Everything else (metrics, tax, WFA) is identical to the cash-sleeve flow.

**Lag.** Cash sleeve used `LAG_DAYS = 5` for 1x mutual funds (T+5). くりっく株365 is an exchange CFD with immediate execution — use **`LAG_DAYS = 1`** (next-day fill, conservative vs T+0; matches the leveraged-leg `DELAY=2`/CFD convention being faster than funds). Expose it as a constant so a sensitivity run is one edit.

**Cost & margin premises (canonical — do NOT re-litigate; `src/product_costs.py` L152-172, `MARGIN_CAPACITY_STRESS_RESULTS_20260617.md`).** Margin is collateral, not idle cash; it creates NO continuous CAGR drag. The only recurring cost is financing (already in the carry term as the spread) plus a small roll bid-ask (`K365_MARGIN_ROLL_COST = 0.0020`/yr on notional) and per-trade cost on the position change. Do NOT subtract "margin × SOFR" or "idle-cash foregone return" — those premises are REJECTED in the canon.

**Evaluation reporting constraint (PERMANENT, from project memory).** Strategy reports must NOT include VETO / hard-veto / pass-fail labels. Report raw measured values and factual descriptions only (MaxDD, Worst10Y, Regime stats are measurements, not verdicts).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/audit/short_carry_sleeve_poc_20260625.py` (Create) | **Stage A PoC.** Measures OUT-day NDX return, builds the short-carry OUT-day return for L∈{1,2,3}, reports expected-value sign vs cash. ~120 lines. Throwaway-grade but committed (it is the evidence for the go/no-go). |
| `src/audit/short_carry_sleeve_sim_20260625.py` (Create, Stage B) | **Stage B full sim.** Splices the short sleeve into OUT days, runs standard metrics + WFA + after-tax, writes grid/yearly CSVs. Mirrors `cash_sleeve_sim.py` structure. ~200 lines. |
| `audit_results/short_carry_sleeve_poc_20260625.csv` (Generated) | PoC per-leverage expected-value table. |
| `audit_results/short_carry_sleeve_metrics_20260625.csv` (Generated) | Stage B standard-metrics table (short variants + baselines). |
| `audit_results/short_carry_sleeve_yearly_20260625.csv` (Generated) | Stage B after-tax yearly returns for the focus strategies. |
| `SHORT_CARRY_SLEEVE_RESULTS_20260625.md` (Create, Stage B) | Findings report. Decision: adopt / reject / needs-work. Pushed to GitHub, URL-verified. |

**Decision gate between stages:** After Stage A, STOP and report the sign to the user. Only proceed to Stage B if the PoC shows the short sleeve's OUT-day expected value (price + carry, after-tax) beats cash at some leverage. This is an explicit human checkpoint.

---

## Stage A — Proof of Concept (expected-value sign)

### Task 1: PoC script — measure OUT-day NDX return and short-carry expected value

**Files:**
- Create: `src/audit/short_carry_sleeve_poc_20260625.py`
- Generated: `audit_results/short_carry_sleeve_poc_20260625.csv`

- [ ] **Step 1: Write the PoC script**

```python
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
    sofr = np.asarray(a['sofr'], dtype=float)        # daily annual SOFR (decimal)

    n_out = int(out.sum())
    out_ndx = ret_ndx[out]
    out_sofr = sofr[out]
    mean_ndx_daily = float(np.mean(out_ndx))
    ann_ndx = ann_from_daily_mean(mean_ndx_daily)    # OUT-day NDX annualized (expect ~ -7.6%)
    mean_sofr = float(np.mean(out_sofr))

    print('OUT days = %d  (%.1f%% of %d)' % (n_out, 100*n_out/len(out), len(out)))
    print('OUT-day NDX annualized price return = %+.2f%%  (short tailwind if negative)'
          % (ann_ndx * 100))
    print('OUT-day mean SOFR (annual) = %.2f%%   spread=%.2f%%  div=%.2f%%'
          % (mean_sofr*100, SPREAD*100, DIV_YIELD*100))

    rows = []
    cash_ann = 0.0
    for L in LEVERAGES:
        # Build per-day short return on equity, OUT days only, then annualize by compounding.
        carry_daily = (out_sofr + SPREAD - DIV_YIELD) * L / TRADING_DAYS
        price_daily = -L * out_ndx
        short_daily = price_daily + carry_daily
        cum = float(np.prod(1.0 + short_daily))
        yrs = len(short_daily) / TRADING_DAYS
        ann_short_pre = cum ** (1.0/yrs) - 1.0 if cum > 0 else -1.0
        # carry-only and price-only annualized for attribution
        ann_carry = (mean_sofr + SPREAD - DIV_YIELD) * L      # ~ linear, report as such
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
```

- [ ] **Step 2: Run the PoC and capture output**

Run:
```bash
cd "/c/Users/user/Desktop/投資・不動産/nasdaq_backtest"; python src/audit/short_carry_sleeve_poc_20260625.py
```
Expected: prints OUT-day count and `OUT-day NDX annualized price return` ≈ −5% to −9% (the cash-sleeve study found ≈ −7.63%; allow drift since data/engine may have moved). For each L, prints price/carry/short totals. Likely outcome: carry is solidly positive (~+4% per unit L), price leg positive (short profits when NDX falls). `beats_cash=True` expected at least at L=1; higher L amplifies both legs.

- [ ] **Step 3: Sanity-check the sign against the cash-sleeve study**

Run (one-off verification, no file kept):
```bash
cd "/c/Users/user/Desktop/投資・不動産/nasdaq_backtest"; python analysis_cash_sleeve/cash_sleeve_sim.py 2>&1 | grep -i NASDAQ_1x
```
Expected: a line reporting NASDAQ_1x OUT-day annualized return near −7.6%. Confirm your PoC's `OUT-day NDX annualized price return` is within ~2pp of it (small drift from annualization method is fine; a sign flip or a >5pp gap means a bug — STOP and diagnose).

- [ ] **Step 4: Commit the PoC + CSV**

```bash
cd "/c/Users/user/Desktop/投資・不動産/nasdaq_backtest"; git add src/audit/short_carry_sleeve_poc_20260625.py audit_results/short_carry_sleeve_poc_20260625.csv; git commit -m "feat(audit): OUT-period short carry sleeve PoC (expected-value sign check)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: HUMAN DECISION GATE**

Report to the user: OUT-day NDX return, per-leverage short after-tax expected value, and whether it beats cash. Ask whether to proceed to Stage B (full sleeve + WFA) or stop. **Do NOT start Stage B without this confirmation** — if the sign is negative or marginal, the user may prefer to stop here.

---

## Stage B — Full Sleeve Simulation (only if Stage A is favorable)

### Task 2: Full short-sleeve simulator — splice short into OUT days, standard metrics + WFA

**Files:**
- Create: `src/audit/short_carry_sleeve_sim_20260625.py`
- Generated: `audit_results/short_carry_sleeve_metrics_20260625.csv`, `audit_results/short_carry_sleeve_yearly_20260625.csv`

- [ ] **Step 1: Write the full simulator (mirrors cash_sleeve_sim.py)**

```python
"""
src/audit/short_carry_sleeve_sim_20260625.py
============================================
Stage B: full OUT-period SHORT carry sleeve on DH-W1, evaluated on the standard
metrics + WFA harness (same flow as analysis_cash_sleeve/cash_sleeve_sim.py).

OUT-day short daily return on equity (L = short leverage):
    r_short[t] = -L*ret_ndx[t] + (sofr[t] + SPREAD - DIV)*L/252 - L*ROLL/252
HOLD days: baseline DH-W1 return (unchanged). Splice with a LAG_DAYS forward shift.

Variants: SHORT_L1.0, SHORT_L1.5, SHORT_L2.0 plus BASELINE_DH-W1 (cash) for ref.
After-tax x0.8273 on yearly. ASCII prints. No temp files.
Writes audit_results/short_carry_sleeve_{metrics,yearly}_20260625.csv.
"""
from __future__ import annotations
import os, sys, types
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

from g14_wfa_sbi_cfd import load_shared_assets, generate_windows
from g23a_dh_refinement_variants import build_W1
from g18_daily_trade_cost_wfa import metrics_from_nav, apply_tax_etf_decimal, wfa_metrics
from product_costs import K365_FINANCING_SPREAD, K365_MARGIN_ROLL_COST, TQQQ

TRADING_DAYS = 252
LAG_DAYS = 1                 # exchange CFD next-day fill (vs T+5 for funds)
DIV_YIELD = TQQQ.dividend_yield   # 0.003
SPREAD = K365_FINANCING_SPREAD    # 0.0075
ROLL = K365_MARGIN_ROLL_COST      # 0.0020 /yr on notional
OUT_DIR = os.path.join(_REPO, 'audit_results')
SHORT_LEVS = [1.0, 1.5, 2.0]


def _geo(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan
    c = float(np.prod(1.0 + x))
    return c ** (1.0 / len(x)) - 1.0 if c > 0 else -1.0


def worst10y_p10_from_yearly(yr_values):
    arr = np.asarray(yr_values, dtype=float)
    w10 = []
    for i in range(len(arr) - 10 + 1):
        c = float(np.prod(1.0 + arr[i:i+10]))
        w10.append(c ** (1.0/10) - 1.0 if c > 0 else -1.0)
    worst10y = float(np.min(w10)) if w10 else np.nan
    r5 = []
    for i in range(len(arr) - 5 + 1):
        c = float(np.prod(1.0 + arr[i:i+5]))
        r5.append(c ** (1.0/5) - 1.0 if c > 0 else -1.0)
    p10 = float(np.percentile(r5, 10)) if r5 else np.nan
    return worst10y, p10


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 90)
    print('SHORT CARRY SLEEVE (OUT days) full sim -- standard metrics + WFA  2026-06-25')
    print('=' * 90)

    a = load_shared_assets()
    dates = a['dates'].reset_index(drop=True)
    n = len(dates)
    nav_w1, cost_w1, mask, wn_w1, lev_w1 = build_W1(a)
    mask = np.asarray(mask, dtype=float)
    r_w1 = pd.Series(np.asarray(nav_w1, dtype=float)).pct_change().fillna(0).values
    out = (mask < 0.5)

    close = np.asarray(a['close'], dtype=float)
    ret_ndx = np.nan_to_num(np.concatenate([[0.0], np.diff(close)/close[:-1]]), nan=0.0)
    sofr = np.asarray(a['sofr'], dtype=float)

    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out[:-LAG_DAYS]
    print('  OUT(cash)=%.1f%%  fund_active(lagged)=%.1f%%  n=%d'
          % (100*out.mean(), 100*fund_active.mean(), n))

    windows = generate_windows(dates)

    # Effective weights for WFA trade counting: HOLD=DH dynamic, OUT-short=|L| in NDX leg.
    wn_base = np.asarray(a['wn_A'], dtype=float) * mask
    wb_base = np.asarray(a['wb_A'], dtype=float) * mask
    lev_base = np.asarray(a['lev_raw'], dtype=float) * mask * 3.0

    def evaluate(label, r_daily, wn_eff, wb_eff, lev_eff):
        nav_pre = pd.Series(np.cumprod(1.0 + r_daily), index=dates.index)
        m = metrics_from_nav(nav_pre, dates, a['ret'])
        yr_aft = m['yearly'].apply(apply_tax_etf_decimal)
        is_sub  = yr_aft.loc[[y for y in yr_aft.index if 1974 <= y <= 2020]]
        oos_sub = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
        cagr_is, cagr_oos = _geo(is_sub.values), _geo(oos_sub.values)
        oos_mult = float(np.prod(1.0 + oos_sub.values))
        worst10y, p10 = worst10y_p10_from_yearly(yr_aft.values)
        try:
            wfa = wfa_metrics(nav_pre, dates, windows, lev_arr=lev_eff, wn_arr=wn_eff, wb_arr=wb_eff)
            ci95, wfe, trades = (wfa.get('WFA_CI95_lo', np.nan),
                                 wfa.get('WFA_WFE', np.nan),
                                 wfa.get('mean_Trades_yr', np.nan))
        except Exception as e:
            print('    WFA failed %s: %s' % (label, e)); ci95 = wfe = trades = np.nan
        row = dict(strategy=label,
                   CAGR_OOS_pct=round(cagr_oos*100, 2), CAGR_IS_pct=round(cagr_is*100, 2),
                   OOS_mult=round(oos_mult, 3), IS_OOS_gap_pp=round((cagr_is-cagr_oos)*100, 2),
                   Sharpe_OOS=round(m['Sharpe_OOS'], 3), MaxDD_pct=round(m['MaxDD_FULL']*100, 2),
                   Worst10Y_pct=round(worst10y*100, 2), P10_5Y_pct=round(p10*100, 2),
                   Trades_yr=round(trades, 1) if not np.isnan(trades) else np.nan,
                   WFE=round(wfe, 3) if not np.isnan(wfe) else np.nan,
                   CI95_lo_pct=round(ci95*100, 2) if not np.isnan(ci95) else np.nan)
        print('  %-16s OOS=%+6.2f%% IS=%+6.2f%% gap=%+5.2f Sh=%+.2f DD=%+6.1f '
              'W10=%+5.2f P10=%+5.2f Tr=%s WFE=%s CI=%s'
              % (label, row['CAGR_OOS_pct'], row['CAGR_IS_pct'], row['IS_OOS_gap_pp'],
                 row['Sharpe_OOS'], row['MaxDD_pct'], row['Worst10Y_pct'], row['P10_5Y_pct'],
                 row['Trades_yr'], row['WFE'], row['CI95_lo_pct']))
        return row, yr_aft

    rows = []
    yearly_focus = {}
    base_row, base_yr = evaluate('BASELINE_DH-W1', r_w1, wn_base, wb_base, lev_base)
    rows.append(base_row); yearly_focus['BASELINE_DH-W1'] = base_yr

    for L in SHORT_LEVS:
        carry = (sofr + SPREAD - DIV_YIELD) * L / TRADING_DAYS
        roll = (L * ROLL) / TRADING_DAYS
        r_short = -L * ret_ndx + carry - roll
        r_enh = np.where(fund_active, r_short, r_w1)
        wn_eff = np.where(fund_active, L, wn_base)        # short notional in NDX leg
        wb_eff = np.where(fund_active, 0.0, wb_base)
        lev_eff = np.where(fund_active, L, lev_base)
        label = 'SHORT_L%.1f' % L
        row, yr_aft = evaluate(label, r_enh, wn_eff, wb_eff, lev_eff)
        rows.append(row); yearly_focus[label] = yr_aft

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'short_carry_sleeve_metrics_20260625.csv'),
              index=False, float_format='%.4f')

    years = sorted(set(y for s in yearly_focus.values() for y in s.index))
    ycols = {'year': years}
    for label, ser in yearly_focus.items():
        ycols[label] = [round(float(ser.get(y, np.nan))*100, 2) if y in ser.index else np.nan
                        for y in years]
    pd.DataFrame(ycols).to_csv(
        os.path.join(OUT_DIR, 'short_carry_sleeve_yearly_20260625.csv'),
        index=False, float_format='%.2f')
    print('\n[DONE] CSVs -> audit_results/ (metrics, yearly)')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the full sim**

Run:
```bash
cd "/c/Users/user/Desktop/投資・不動産/nasdaq_backtest"; python src/audit/short_carry_sleeve_sim_20260625.py
```
Expected: prints a metrics line per strategy (BASELINE_DH-W1, SHORT_L1.0, SHORT_L1.5, SHORT_L2.0). Each must have a finite CAGR_OOS, Sharpe, MaxDD, and a WFA CI95_lo (not NaN). If WFA prints "failed", read the exception — most likely a shape mismatch in `lev_eff/wn_eff` length vs `dates`; they must all be length `n`.

- [ ] **Step 3: Cross-check vs the PoC**

Run (verify the full-sim short variants moved OOS in the direction the PoC predicted):
```bash
cd "/c/Users/user/Desktop/投資・不動産/nasdaq_backtest"; python -c "import pandas as pd; m=pd.read_csv('audit_results/short_carry_sleeve_metrics_20260625.csv'); p=pd.read_csv('audit_results/short_carry_sleeve_poc_20260625.csv'); print(m[['strategy','CAGR_OOS_pct','MaxDD_pct','Sharpe_OOS','CI95_lo_pct']].to_string(index=False)); print('PoC beats_cash:', dict(zip(p.leverage, p.beats_cash)))"
```
Expected: if PoC said `beats_cash=True` at L≥1, the full sim's SHORT_L* CAGR_OOS should be ≥ BASELINE's (the OUT-day improvement compounds through). A SHORT variant with CAGR_OOS *below* baseline despite PoC saying beats_cash means the price-leg variance hurt risk-adjusted compounding — note it, don't "fix" it (it's a real finding).

- [ ] **Step 4: Commit the full sim + CSVs**

```bash
cd "/c/Users/user/Desktop/投資・不動産/nasdaq_backtest"; git add src/audit/short_carry_sleeve_sim_20260625.py audit_results/short_carry_sleeve_metrics_20260625.csv audit_results/short_carry_sleeve_yearly_20260625.csv; git commit -m "feat(audit): full OUT-period short carry sleeve sim with standard metrics + WFA

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 3: Independent QC — attack premises, not just arithmetic

**Files:**
- No new code (verification task). Uses a throwaway hand-trace deleted after use.

- [ ] **Step 1: Re-derive OUT-day short return for ONE specific year by hand**

Pick the worst OUT-heavy year (e.g. 2008 or 2022). In a throwaway script (delete after — do NOT leave it in the repo), independently recompute that year's `SHORT_L1.0` return from `ret_ndx`, `sofr`, mask, and the carry formula, WITHOUT importing the sim module. Compare to `short_carry_sleeve_yearly_20260625.csv`. They must match within rounding.

```bash
cd "/c/Users/user/Desktop/投資・不動産/nasdaq_backtest"; python /tmp/_qc_short_handtrace.py   # script you author in a system temp dir, then delete
```
Expected: hand-traced year return == CSV value (±0.1pp). A mismatch means a sign or lag bug.

- [ ] **Step 2: Attack the three premises that could invert the result**

Check each explicitly and write one sentence each into the report:
1. **Carry sign**: confirm short RECEIVES financing (positive carry). Re-read `cash_sleeve_sim.py` long-financing sign and confirm we flipped it. (If carry came out negative for a short, the whole study is inverted.)
2. **OUT≠down guarantee**: state plainly that OUT-day NDX return is an *average* of ≈−7.6%; individual OUT spells include up-moves that a leveraged short amplifies into losses. Report the WORST single OUT-spell short P&L (max consecutive-OUT cumulative `-L*ret_ndx`), not just the average.
3. **No double-counting**: confirm the carry term is on equity (×L/252) and the price term is on equity (−L×ret), so leverage is applied once to each, not squared.

- [ ] **Step 3: Compare on equal footing against the adopted Gold/Bond sleeve**

The decision is not "short vs cash" but "short vs the BEST already-adopted OUT filler". Read `analysis_cash_sleeve/cash_sleeve_4strategies_metrics.csv` (P2_GOLD100 / P7_GOLD75_BOND25 numbers) and place the short variants' CAGR_OOS / MaxDD / Sharpe / CI95_lo next to them in the report. The short sleeve must beat Gold/Bond on a risk-adjusted basis to be worth the tail risk, not merely beat cash.

- [ ] **Step 4: Delete the throwaway hand-trace script**

```bash
rm -f /tmp/_qc_short_handtrace.py
```
Confirm `git status` shows no stray files under the repo.

### Task 4: Findings report + deliverable

**Files:**
- Create: `SHORT_CARRY_SLEEVE_RESULTS_20260625.md`

- [ ] **Step 1: Write the report**

Structure (NO veto/pass-fail labels — measurements and facts only, per project memory):
- H1 + `作成日: 2026-06-25` / `最終更新日: 2026-06-25`.
- §1 Question & mechanism (long pays / short receives 金利相当額; net carry on equity ≈ (SOFR+0.75pp−0.3%)×L; price leg −L×ret on OUT days).
- §2 Stage A PoC table (per-L price / carry / after-tax short total vs cash 0%).
- §3 Stage B standard-metrics table (BASELINE + SHORT_L1.0/1.5/2.0): CAGR_OOS, IS, gap, Sharpe, MaxDD, Worst10Y, P10_5Y, Trades_yr, WFE, CI95_lo.
- §4 Head-to-head vs adopted Gold/Bond cash sleeve (P2/P7) — the real benchmark.
- §5 Tail-risk measurement: worst consecutive-OUT short P&L (the amplified-up-move scenario), stated as a raw measured value.
- §6 QC: premise checks (carry sign, OUT≠down, no double-count), hand-trace match.
- §7 Reading: under what condition (if any) the short sleeve is preferable; explicitly note this is analysis, not a live-trading recommendation; active strategy unchanged.
- §8 Footnotes: cost premises (margin = collateral, no CAGR drag), data range, after-tax basis.

- [ ] **Step 2: Branch check, then commit on main**

```bash
cd "/c/Users/user/Desktop/投資・不動産/nasdaq_backtest"; git branch --show-current
```
Expected: `main` (if not, `git checkout main` first per CLAUDE.md §4). Then:
```bash
cd "/c/Users/user/Desktop/投資・不動産/nasdaq_backtest"; git add SHORT_CARRY_SLEEVE_RESULTS_20260625.md; git commit -m "docs: OUT-period short carry sleeve findings report

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Push and verify the URL (deliverables policy)**

```bash
cd "/c/Users/user/Desktop/投資・不動産/nasdaq_backtest"; git pull --rebase origin main; git push origin main; git rev-parse --abbrev-ref HEAD
```
Then verify the raw URL returns 200 before reporting:
```bash
curl -s -o /dev/null -w "%{http_code}" https://raw.githubusercontent.com/KazuyaMurayama/NASDAQ_backtest/main/SHORT_CARRY_SLEEVE_RESULTS_20260625.md
```
Expected: `200`. Report the deliverable in the CLAUDE.md §9 three-column table with the `/blob/main/` link.

---

## Self-Review

**1. Spec coverage:**
- "売りで金利を受け取る" carry mechanism → Task 1 carry term, Task 3 Step 2 premise #1. ✓
- "金利 × レバ倍" (carry scales with notional = equity×L) → Task 1 `ann_carry = (...)*L`, documented in Background. ✓
- "フォワードリターン低下時=OUT時にショート" → splice on `out` mask (Task 1/2). ✓
- "戦略改善につながるか" → Stage B metrics + head-to-head vs Gold/Bond (Task 2, Task 3 Step 3). ✓
- After-tax / WFA / standard-metrics parity with existing studies → Task 2 reuses `apply_tax_etf_decimal`, `metrics_from_nav`, `wfa_metrics`. ✓
- No-veto reporting rule → Task 4 §-structure note. ✓
- Deliverables push + URL verify → Task 4 Step 3. ✓
- Cheap-PoC-first (user's stated preference for the 2-step) → Stage A gate before Stage B. ✓

**2. Placeholder scan:** No TBD/TODO; every code step has full code; commands have expected output. The only intentional "author then delete" is the Task 3 throwaway hand-trace, placed in a system temp dir (not the repo), per the no-local-temp rule. ✓

**3. Type consistency:** `build_W1(a)` unpacks to `(nav_w1, cost_w1, mask, wn_w1, lev_w1)` consistently in PoC and full sim (matches `cash_sleeve_sim.py:99`). `a['close']`, `a['sofr']`, `a['ret']`, `a['dates']`, `a['wn_A']`, `a['wb_A']`, `a['lev_raw']` all match keys used in `cash_sleeve_sim.py`. `metrics_from_nav(nav, dates, ret)` returns a dict with `'yearly'`, `'Sharpe_OOS'`, `'MaxDD_FULL'` — used identically to the template. Constants `K365_FINANCING_SPREAD`, `K365_MARGIN_ROLL_COST`, `TQQQ.dividend_yield` confirmed present in `src/product_costs.py`. ✓

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-06-25-out-period-short-carry-sleeve.md`.
