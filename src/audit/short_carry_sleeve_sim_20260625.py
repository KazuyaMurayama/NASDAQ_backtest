"""
src/audit/short_carry_sleeve_sim_20260625.py
============================================
Stage B: full OUT-period SHORT carry sleeve on DH-W1, evaluated on the standard
metrics + WFA harness (same flow as analysis_cash_sleeve/cash_sleeve_sim.py).

OUT-day short daily return on equity (L = short leverage):
    r_short[t] = -L*ret_ndx[t] + (sofr_daily[t] + (SPREAD-DIV)/252)*L - L*ROLL/252
  (sofr_daily from load_sofr() is ALREADY per-day; SPREAD/DIV/ROLL are annual -> /252.)
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


def worst_out_spell_short(out_mask, ret_ndx, sofr_daily, L):
    """Worst (most negative) cumulative SHORT P&L over any single consecutive
    OUT spell, on equity. This is the amplified-up-move tail: NDX rallies while
    we are short. Returns (worst_spell_return_pct, spell_len_days)."""
    worst = 0.0
    worst_len = 0
    i = 0
    n = len(out_mask)
    while i < n:
        if not out_mask[i]:
            i += 1
            continue
        j = i
        cum = 1.0
        while j < n and out_mask[j]:
            carry = sofr_daily[j] + (SPREAD - DIV_YIELD) / TRADING_DAYS - ROLL / TRADING_DAYS
            r = -L * ret_ndx[j] + L * carry
            cum *= (1.0 + r)
            j += 1
        spell_ret = cum - 1.0
        if spell_ret < worst:
            worst = spell_ret
            worst_len = j - i
        i = j
    return worst, worst_len


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
    sofr_daily = np.asarray(a['sofr'], dtype=float)   # PER-DAY rate

    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out[:-LAG_DAYS]
    print('  OUT(cash)=%.1f%%  fund_active(lagged)=%.1f%%  n=%d'
          % (100*out.mean(), 100*fund_active.mean(), n))

    windows = generate_windows(dates)

    # Effective weights for WFA trade counting: HOLD=DH dynamic, OUT-short=L in NDX leg.
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

    print('\n  -- worst single consecutive-OUT short spell (amplified up-move tail) --')
    for L in SHORT_LEVS:
        carry = sofr_daily + (SPREAD - DIV_YIELD) / TRADING_DAYS
        roll = (L * ROLL) / TRADING_DAYS
        r_short = -L * ret_ndx + L * carry - roll
        r_enh = np.where(fund_active, r_short, r_w1)
        wn_eff = np.where(fund_active, L, wn_base)        # short notional in NDX leg
        wb_eff = np.where(fund_active, 0.0, wb_base)
        lev_eff = np.where(fund_active, L, lev_base)
        label = 'SHORT_L%.1f' % L
        row, yr_aft = evaluate(label, r_enh, wn_eff, wb_eff, lev_eff)
        ws, wl = worst_out_spell_short(out, ret_ndx, sofr_daily, L)
        row['worst_OUT_spell_pct'] = round(ws*100, 2)
        row['worst_OUT_spell_days'] = int(wl)
        print('    %-12s worst consecutive-OUT short spell = %+.2f%% over %d days'
              % (label, ws*100, wl))
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
