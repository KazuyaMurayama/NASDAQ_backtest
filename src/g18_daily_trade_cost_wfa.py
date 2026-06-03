"""
G18: 日次レベル 取引コスト反映 WFA + 9指標 再計算
=================================================================
4戦略 × 4 取引コストケース で、日次レベルで取引コストを NAV に反映し、
IS/OOS の CAGR・Sharpe・MaxDD・Worst10Y・P10_5Y・IS-OOS gap・Trades_yr・
Overfit(WFE)・CI95_lo を全て再計算する。

CFD 3戦略 (E4 / F10 / D5 vz=0.65/lmax=5.5):
  daily_position(t) = wn(t) × lev_mod(t) × L_s2(t)
  daily_cost(t) = |Δposition(t)| × spread_one_way
  spread_one_way = spread_round_trip / 2

DH Dyn 2x3x [A] (TQQQ + TMF + 2036):
  daily_turnover(t) = |Δw_TQQQ| × 3 + |Δw_TMF| × 3 + |Δw_2036| × 2
  daily_cost(t) = daily_turnover(t) × per_unit_cost_one_way

Spread cases (CFD round-trip):
  0.020% — GMO 米国NQ100ミニ 2026/4 実測
  0.030% — 楽観
  0.050% — 中庸
  0.100% — 保守 (baseline)

DH cost cases (per-unit turnover one-way):
  0.30% — large-NAV (cap effective)
  0.50% — moderate
  1.00% — small-NAV (no cap)

IS:  1977-01-03 〜 2021-05-07
OOS: 2021-05-08 〜 2026-03-26

出力:
  - g18_daily_cost_metrics.csv (4戦略 × 4ケース × 9指標)
  - g18_daily_nav_with_cost.csv (戦略 × ケース × 日次NAV、参照用)
  - g18_yearly_with_daily_cost.csv (1974-2026 × 4戦略 × 4ケース)
"""
import sys
import os
import types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

from g14_wfa_sbi_cfd import (
    load_shared_assets, _make_nav, SBI_CFD_SPREAD, BASE, TODAY,
    generate_windows, compute_window_metrics, compute_summary_stats,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IS_END_TS    = pd.Timestamp('2021-05-07')
OOS_START_TS = pd.Timestamp('2021-05-08')
OOS_END_TS   = pd.Timestamp('2026-03-26')

# §3-A tax model
UNCAPTURED_DRAG_CFD = 0.0066
TAX_FACTOR = 0.8273

# CFD spread cases (round-trip decimal)
SPREAD_CASES_CFD = {
    'measured (GMO 2026/4)': 0.00020,
    'optimistic':            0.00030,
    'moderate':              0.00050,
    'conservative (base)':   0.00100,
}

# DH cost cases (per-unit turnover one-way, ETF actual cost — no leverage weighting)
# Includes: SBI commission (cap-effective) + bid-ask + minor FX
COST_CASES_DH = {
    'large-NAV (cap eff.)':  0.0005,   # cap $22 dominates, 0.05% effective
    'moderate (base)':        0.0010,   # retail $100k, 0.10%
    'small-NAV (no cap)':     0.0030,   # retail $50k, no cap effect 0.30%
}


def apply_tax_cfd_decimal(pre):
    """逐年 §3-A tax: r_aft = (r_pre - 0.66%) × 0.8273"""
    return (pre - UNCAPTURED_DRAG_CFD) * TAX_FACTOR


def apply_tax_etf_decimal(pre):
    """ETF: r_aft = r_pre × 0.8273"""
    return pre * TAX_FACTOR


def build_cfd_nav_with_cost(close, lev_mod, wn, wg, wb, dates,
                            gold_2x, bond_3x, sofr, L_s2_values,
                            spread_one_way):
    """g14 _make_nav と同じ NAV を構築するが、日次 |Δposition| × spread_one_way を差し引く。

    position(t) = wn(t) × lev_mod(t) × L_s2(t)
    """
    # まず元のNAV (取引コスト未反映、SBIスプレッド 3%/yr 込み)
    nav_base = _make_nav(close, lev_mod, wn, wg, wb, dates,
                          gold_2x, bond_3x, sofr, L_s2_values)

    # daily position series — fill NaN (LT2 warmup) with 0
    L_s2 = np.nan_to_num(np.asarray(L_s2_values, dtype=float), nan=0.0)
    lm   = np.nan_to_num(np.asarray(lev_mod,     dtype=float), nan=0.0)
    wn_a = np.nan_to_num(np.asarray(wn,          dtype=float), nan=0.0)
    pos = wn_a * lm * L_s2
    dpos = np.zeros_like(pos)
    dpos[1:] = np.abs(np.diff(pos))
    daily_trade_cost = dpos * spread_one_way  # decimal per day

    # base daily returns
    r_base = np.nan_to_num(nav_base.pct_change().fillna(0).values, nan=0.0)

    # adjusted daily returns
    r_adj = r_base - daily_trade_cost
    nav_adj = pd.Series(np.cumprod(1.0 + r_adj), index=nav_base.index)
    return nav_adj, daily_trade_cost.sum() / (len(dates)/252.0)  # yr cost approx


def build_dh_nav_with_cost(close, lev_raw, wn_A, wg_A, wb_A, dates,
                           gold_2x, bond_3x, sofr, per_unit_cost):
    """DH Dyn [A] NAV を取引コスト未反映で構築し、
    日次 turnover × per_unit_cost を差し引く。

    daily_turnover(t) = |Δw_TQQQ| × 3 + |Δw_TMF| × 3 + |Δw_Gold| × 2
    """
    # 既存の DH NAV (g14 と同じ build_nav_strategy, nas_mode='CFD' で CFD wrap)
    # ただし DH は元来 TQQQ + TMF + Gold ETF 想定。
    # ここではシミュレーション基盤として L_s2=lev_raw を CFD レバとして渡す。
    # g14 と同様 _make_nav で構築。
    nav_base = _make_nav(close, np.ones(len(close)), wn_A, wg_A, wb_A,
                          dates, gold_2x, bond_3x, sofr,
                          np.asarray(lev_raw) * 3.0)

    # turnover series (アセット配分の変化)
    wn = np.asarray(wn_A); wg = np.asarray(wg_A); wb = np.asarray(wb_A)
    dwn = np.zeros_like(wn); dwn[1:] = np.abs(np.diff(wn))
    dwg = np.zeros_like(wg); dwg[1:] = np.abs(np.diff(wg))
    dwb = np.zeros_like(wb); dwb[1:] = np.abs(np.diff(wb))
    # ETF actual trade volume (cost is on dollar traded, NOT on internal leverage)
    turnover = dwn + dwb + dwg
    turnover = np.nan_to_num(turnover, nan=0.0)
    daily_trade_cost = turnover * per_unit_cost

    r_base = np.nan_to_num(nav_base.pct_change().fillna(0).values, nan=0.0)
    r_adj = r_base - daily_trade_cost
    nav_adj = pd.Series(np.cumprod(1.0 + r_adj), index=nav_base.index)
    return nav_adj, daily_trade_cost.sum() / (len(dates)/252.0)


def build_dh_nav_with_timing_cost(close, lev_raw, wn_A, wg_A, wb_A, dates,
                                  gold_2x, bond_3x, sofr, per_unit_cost,
                                  lev_mod=None, L_s2_values=None):
    """DH NAV + optional CFD-style timing.

    lev_mod=None and L_s2_values=None → build_dh_nav_with_cost と完全一致。
    lev_mod  : (n,) np.ndarray or None — vz-gated NASDAQ exposure factor (0〜1)
    L_s2_values : (n,) np.ndarray or None — vz-gated leverage cap (lev_raw×3 にキャップ)

    daily_turnover(t) = |Δw_TQQQ| + |Δw_TMF| + |Δw_Gold| (modified weights を使用)
    """
    n = len(close)
    if lev_mod is None:
        lev_mod_arr = np.ones(n)
    else:
        lev_mod_arr = np.nan_to_num(np.asarray(lev_mod, dtype=float), nan=1.0)

    lev_eff = np.asarray(lev_raw, dtype=float) * 3.0
    if L_s2_values is not None:
        L_s2 = np.nan_to_num(np.asarray(L_s2_values, dtype=float), nan=lev_eff.max())
        lev_eff = np.minimum(lev_eff, L_s2)

    nav_base = _make_nav(close, lev_mod_arr, wn_A, wg_A, wb_A,
                          dates, gold_2x, bond_3x, sofr, lev_eff)

    wn = np.asarray(wn_A); wg = np.asarray(wg_A); wb = np.asarray(wb_A)
    dwn = np.zeros_like(wn); dwn[1:] = np.abs(np.diff(wn))
    dwg = np.zeros_like(wg); dwg[1:] = np.abs(np.diff(wg))
    dwb = np.zeros_like(wb); dwb[1:] = np.abs(np.diff(wb))
    turnover = dwn + dwb + dwg
    turnover = np.nan_to_num(turnover, nan=0.0)
    daily_trade_cost = turnover * per_unit_cost

    r_base = np.nan_to_num(nav_base.pct_change().fillna(0).values, nan=0.0)
    r_adj = r_base - daily_trade_cost
    nav_adj = pd.Series(np.cumprod(1.0 + r_adj), index=nav_base.index)
    return nav_adj, daily_trade_cost.sum() / (len(dates)/252.0)


def metrics_from_nav(nav, dates, ret_nas, is_end=IS_END_TS, oos_start=OOS_START_TS):
    """NAV から IS/OOS の CAGR, Sharpe, MaxDD, IS-OOS gap, Worst10Y, P10_5Y を算出。"""
    nav = nav.astype('float64')
    r = nav.pct_change().fillna(0)
    # IS period
    is_mask  = dates <= is_end
    oos_mask = dates >= oos_start
    full_mask = pd.Series(True, index=dates.index)

    def _cagr_sharpe(mask):
        n_days = int(mask.sum())
        if n_days < 50:
            return np.nan, np.nan
        rr = r[mask.values]
        cum = float(nav[mask.values].iloc[-1] / nav[mask.values].iloc[0])
        years = n_days / 252.0
        cagr = cum**(1.0/years) - 1.0 if cum > 0 else -1.0
        mu = float(rr.mean()) * 252
        sd = float(rr.std(ddof=1)) * np.sqrt(252)
        sharpe = mu / sd if sd > 0 else np.nan
        return cagr, sharpe

    cagr_is,   sharpe_is   = _cagr_sharpe(is_mask)
    cagr_oos,  sharpe_oos  = _cagr_sharpe(oos_mask)
    cagr_full, sharpe_full = _cagr_sharpe(full_mask)
    is_oos_gap = (cagr_is - cagr_oos)

    # MaxDD on FULL NAV
    nav_arr = nav.values
    peak = np.maximum.accumulate(nav_arr)
    dd = nav_arr / np.where(peak > 0, peak, 1) - 1.0
    maxdd_full = float(dd.min())

    # Worst10Y / P10_5Y from yearly returns (calendar year compounded)
    yearly = nav.groupby(dates.dt.year).apply(
        lambda s: float(s.iloc[-1]/s.iloc[0]) - 1.0 if len(s) > 1 else 0.0
    )
    yearly = yearly.dropna()
    # rolling 10Y CAGR
    w10 = []
    arr = yearly.values
    for i in range(len(arr) - 10 + 1):
        cum_w = float(np.prod(1.0 + arr[i:i+10]))
        w10.append(cum_w**(1.0/10) - 1.0 if cum_w > 0 else -1.0)
    worst10y = float(np.min(w10)) if w10 else np.nan
    # rolling 5Y CAGR
    r5 = []
    for i in range(len(arr) - 5 + 1):
        cum_w = float(np.prod(1.0 + arr[i:i+5]))
        r5.append(cum_w**(1.0/5) - 1.0 if cum_w > 0 else -1.0)
    p10_5y = float(np.percentile(r5, 10)) if r5 else np.nan

    return dict(
        CAGR_IS=cagr_is, CAGR_OOS=cagr_oos, CAGR_FULL=cagr_full,
        Sharpe_IS=sharpe_is, Sharpe_OOS=sharpe_oos,
        MaxDD_FULL=maxdd_full,
        IS_OOS_gap=is_oos_gap,
        Worst10Y_star=worst10y, P10_5Y=p10_5y,
        yearly=yearly,
    )


def yearly_after_tax_from_nav(nav, dates, is_cfd, daily_trade_cost_arr=None):
    """暦年リターン (after-tax) を NAV から算出。"""
    # yearly pre-tax returns
    nav_year_end = nav.groupby(dates.dt.year).last()
    nav_year_first = nav.groupby(dates.dt.year).first()
    yr_pre = (nav_year_end / nav_year_first - 1.0)
    # apply §3-A tax
    if is_cfd:
        yr_aft = yr_pre.apply(apply_tax_cfd_decimal)
    else:
        yr_aft = yr_pre.apply(apply_tax_etf_decimal)
    return yr_aft


def wfa_metrics(nav, dates, windows, lev_arr, wn_arr, wb_arr):
    """g14 と同等の WFA を nav に対して実施し summary を返す。"""
    rows = []
    for w in windows:
        m = compute_window_metrics(nav, w, wn=wn_arr, wb=wb_arr, lev_arr=lev_arr)
        m.update(window_id=w['window_id'], short_flag=w.get('short_flag', False),
                 start_date=w['start_date'], end_date=w['end_date'])
        rows.append(m)
    per = pd.DataFrame(rows)
    summary = compute_summary_stats(per)
    return summary


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print(f'G18: 日次取引コスト反映 WFA + 9指標 再計算')
    print(f'実行日: {TODAY}')
    print('=' * 80)

    print('\n[S1] Loading shared assets...')
    a = load_shared_assets()
    dates = a['dates']
    close = a['close']
    ret_nas = a['ret']
    windows = generate_windows(dates)
    print(f'  Windows: {len(windows)}')

    rows = []
    yearly_records = {}
    nav_records = {}

    print('\n[S2] Building NAVs with daily trade cost (CFD 3 strats × 4 spreads)...')
    cfd_strats = [
        ('E4 Regime k_lt', 'lev_mod_e4', a['wn_A'], a['wg_A'], a['wb_A'], a['L_s2_lmax7']),
        ('F10 eps015',     'lev_mod_e4', a['wn_f10'], a['wg_f10'], a['wb_f10'], a['L_s2_lmax7']),
        ('D5 v065 l5p5',   'lev_mod_065', a['wn_A'], a['wg_A'], a['wb_A'], a['L_s2_lmax5p5']),
    ]
    for label, lev_mod_key, wn, wg, wb, L_s2 in cfd_strats:
        lev_mod = a[lev_mod_key]
        for case, spread_rt in SPREAD_CASES_CFD.items():
            spread_ow = spread_rt / 2.0
            nav_adj, yr_cost_approx = build_cfd_nav_with_cost(
                close, lev_mod, wn, wg, wb, dates,
                a['gold_2x'], a['bond_3x'], a['sofr'], L_s2.values,
                spread_ow,
            )
            m = metrics_from_nav(nav_adj, dates, ret_nas)
            # apply tax to yearly returns
            yr_pre = m['yearly']
            yr_aft = yr_pre.apply(apply_tax_cfd_decimal)
            yearly_records[(label, case)] = yr_aft

            # WFA on raw nav_adj (no tax)
            try:
                wfa = wfa_metrics(nav_adj, dates, windows, lev_arr=np.asarray(L_s2.values),
                                   wn_arr=wn, wb_arr=wb)
                wfa_ci95 = wfa.get('WFA_CI95_lo', np.nan)
                wfa_wfe  = wfa.get('WFA_WFE', np.nan)
                trades_yr = wfa.get('mean_Trades_yr', np.nan)
            except Exception as e:
                print(f'  WFA failed for {label}/{case}: {e}')
                wfa_ci95 = np.nan; wfa_wfe = np.nan; trades_yr = np.nan

            # tax-adjusted overall CAGR_IS / CAGR_OOS for table display
            # Use yearly compounded after-tax
            is_yrs = [y for y in yr_aft.index if y <= 2021]
            oos_yrs = [y for y in yr_aft.index if y >= 2021]  # 2021 partial (May-Dec)
            # Use 1977-2020 for IS, 2021-2026 for OOS for consistency
            is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1977 <= y <= 2020]]
            oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
            def _geo(x):
                if len(x) == 0: return np.nan
                c = float(np.prod(1.0 + x.values))
                return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
            cagr_is_aft  = _geo(is_subset)
            cagr_oos_aft = _geo(oos_subset)

            rows.append(dict(
                strategy=label, case=case, is_cfd=True,
                spread_rt=spread_rt,
                CAGR_IS=cagr_is_aft, CAGR_OOS=cagr_oos_aft,
                IS_OOS_gap=cagr_is_aft - cagr_oos_aft,
                Sharpe_OOS=m['Sharpe_OOS'],
                MaxDD_FULL=m['MaxDD_FULL'],
                Worst10Y_star=m['Worst10Y_star'],
                P10_5Y=m['P10_5Y'],
                Trades_yr=trades_yr,
                WFA_WFE=wfa_wfe,
                WFA_CI95_lo=wfa_ci95,
                yr_cost_approx=yr_cost_approx,
            ))
            nav_records[(label, case)] = nav_adj
            print(f'  [{label}|{case}] CAGR_IS={cagr_is_aft*100:+.2f}%, '
                  f'CAGR_OOS={cagr_oos_aft*100:+.2f}%, '
                  f'gap={(cagr_is_aft-cagr_oos_aft)*100:+.2f}pp')

    print('\n[S3] Building DH Dyn [A] NAVs (3 cost cases)...')
    for case, per_unit_cost in COST_CASES_DH.items():
        nav_adj, yr_cost_approx = build_dh_nav_with_cost(
            close, a['lev_raw'], a['wn_A'], a['wg_A'], a['wb_A'], dates,
            a['gold_2x'], a['bond_3x'], a['sofr'], per_unit_cost,
        )
        m = metrics_from_nav(nav_adj, dates, ret_nas)
        yr_pre = m['yearly']
        yr_aft = yr_pre.apply(apply_tax_etf_decimal)
        yearly_records[('DH Dyn 2x3x [A]', case)] = yr_aft

        # WFA — use lev_raw*3 as effective leverage
        L_eff = np.asarray(a['lev_raw']) * 3.0
        try:
            wfa = wfa_metrics(nav_adj, dates, windows, lev_arr=L_eff,
                               wn_arr=a['wn_A'], wb_arr=a['wb_A'])
            wfa_ci95 = wfa.get('WFA_CI95_lo', np.nan)
            wfa_wfe  = wfa.get('WFA_WFE', np.nan)
            trades_yr = wfa.get('mean_Trades_yr', np.nan)
        except Exception as e:
            print(f'  WFA failed for DH/{case}: {e}')
            wfa_ci95 = np.nan; wfa_wfe = np.nan; trades_yr = np.nan

        is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1974 <= y <= 2020]]
        oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
        def _geo(x):
            if len(x) == 0: return np.nan
            c = float(np.prod(1.0 + x.values))
            return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
        cagr_is_aft  = _geo(is_subset)
        cagr_oos_aft = _geo(oos_subset)
        rows.append(dict(
            strategy='DH Dyn 2x3x [A]', case=case, is_cfd=False,
            spread_rt=None,
            CAGR_IS=cagr_is_aft, CAGR_OOS=cagr_oos_aft,
            IS_OOS_gap=cagr_is_aft - cagr_oos_aft,
            Sharpe_OOS=m['Sharpe_OOS'],
            MaxDD_FULL=m['MaxDD_FULL'],
            Worst10Y_star=m['Worst10Y_star'],
            P10_5Y=m['P10_5Y'],
            Trades_yr=trades_yr,
            WFA_WFE=wfa_wfe,
            WFA_CI95_lo=wfa_ci95,
            yr_cost_approx=yr_cost_approx,
        ))
        nav_records[('DH Dyn 2x3x [A]', case)] = nav_adj
        print(f'  [DH|{case}] CAGR_IS={cagr_is_aft*100:+.2f}%, '
              f'CAGR_OOS={cagr_oos_aft*100:+.2f}%, '
              f'gap={(cagr_is_aft-cagr_oos_aft)*100:+.2f}pp')

    # Save summary
    summary_df = pd.DataFrame(rows)
    sp = os.path.join(BASE, 'g18_daily_cost_metrics.csv')
    summary_df.to_csv(sp, index=False, float_format='%.6f')
    print(f'\n[S4] Summary saved: {sp}')

    # Save yearly table
    years = sorted(set(y for s in yearly_records.values() for y in s.index))
    yearly_df = pd.DataFrame({'year': years})
    for (label, case), ser in yearly_records.items():
        col = f'{label} [{case}]'
        yearly_df[col] = yearly_df['year'].map(ser)
    yp = os.path.join(BASE, 'g18_yearly_with_daily_cost.csv')
    yearly_df.to_csv(yp, index=False, float_format='%.6f')
    print(f'[S5] Yearly saved: {yp}')

    print('\n[DONE]')


if __name__ == '__main__':
    main()
