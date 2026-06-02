"""
G19E: 3 新戦略 (F8 R5_CALM_BOOST, F7v3+E4 A:tilt=2.0, NDX 1x B&H) を
       日次取引コストで評価し v6.2 統合レポートに追加する
=================================================================
v6.1 (g18) は 4 戦略のみ。本 script は 3 追加戦略を g18 同等の方法で評価:
  - F8 R5_CALM_BOOST : F10 の ε=0 baseline (CALM_BOOST cap, deadband なし)
  - F7v3+E4 A:tilt=2.0 : tilt=2.0/cap=0.10 (regime 無し), E4 base
  - NDX 1x B&H        : ベンチマーク (CFD/レバ不要)

実装方針:
  CFD 2戦略 (F8 R5, F7v3): g18 build_cfd_nav_with_cost を再利用
  B&H 1x: 純粋 NDX Close リターン (コスト=0, 税のみ ×0.8273)

出力:
  - g19e_3strategies_daily_cost.csv
  - g19e_3strategies_yearly.csv
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import (
    load_shared_assets, SBI_CFD_SPREAD, BASE, TODAY,
    compute_tilt_with_deadband,
    K_LO, K_HI, VZ_THR_E4, K_MID, N_LT2,
    TILT_R5, EPS_REF,
    THRESHOLD,
    generate_windows, compute_window_metrics, compute_summary_stats,
)
from g18_daily_trade_cost_wfa import (
    build_cfd_nav_with_cost, metrics_from_nav,
    SPREAD_CASES_CFD, apply_tax_cfd_decimal, apply_tax_etf_decimal,
    UNCAPTURED_DRAG_CFD, TAX_FACTOR,
    IS_END_TS, OOS_START_TS, OOS_END_TS,
    wfa_metrics,
)
from corrected_strategy_backtest import (
    simulate_rebalance_A, TRADING_DAYS,
)

# F7v3 定式 A: tilt=2.0, cap=0.10 (uniform, 非 regime)
F7V3_TILT = 2.0
F7V3_CAP  = 0.10


def build_f7v3_wn(raw_a2_vals, wn_A, wb_A):
    """F7v3 A:tilt=2.0 の wn/wb を構築。
    tilt_amount = clip(2.0 × (raw_a2 - 0.15) × (1 - raw_a2), 0, 0.10)
    bull mask (raw_a2 > THRESHOLD) のみ適用、ε=0 (毎日更新)
    """
    bull_mask = raw_a2_vals > THRESHOLD
    tilt_raw = F7V3_TILT * (raw_a2_vals - THRESHOLD) * (1.0 - raw_a2_vals)
    tilt = np.clip(tilt_raw, 0.0, F7V3_CAP)
    tilt = np.where(bull_mask, tilt, 0.0)
    wn_arr = np.asarray(wn_A, dtype=float)
    wb_arr = np.asarray(wb_A, dtype=float)
    wn_f7v3 = np.clip(wn_arr + tilt, 0.0, 1.0)
    delta = wn_f7v3 - wn_arr
    wb_f7v3 = np.maximum(wb_arr - delta, 0.0)
    return wn_f7v3, wb_f7v3


def compute_bnh_metrics_after_tax(close, dates):
    """NDX 1x B&H NAV を作り、§3-A 税 (×0.8273) を年次に適用。
    CFD コスト=0 (純粋指数保有)。
    """
    nav = close / close.iloc[0]
    # metrics
    r = nav.pct_change().fillna(0)
    is_mask  = dates <= IS_END_TS
    oos_mask = dates >= OOS_START_TS

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

    cagr_is, sharpe_is = _cagr_sharpe(is_mask)
    cagr_oos, sharpe_oos = _cagr_sharpe(oos_mask)
    nav_arr = nav.values
    peak = np.maximum.accumulate(nav_arr)
    dd = nav_arr / np.where(peak > 0, peak, 1) - 1.0
    maxdd_full = float(dd.min())

    # yearly compounding
    yearly = nav.groupby(dates.dt.year).apply(
        lambda s: float(s.iloc[-1]/s.iloc[0]) - 1.0 if len(s) > 1 else 0.0
    )
    yearly = yearly.dropna()
    # B&H 税モデル: §3-A 比例税 ×0.8273 (NISA非適用前提)
    yr_aft = yearly.apply(apply_tax_etf_decimal)

    # rolling 10Y on yearly compounded
    arr = yearly.values  # pre-tax for Worst10Y reference
    w10_pre = []
    for i in range(len(arr) - 10 + 1):
        cum_w = float(np.prod(1.0 + arr[i:i+10]))
        w10_pre.append(cum_w**(1.0/10) - 1.0 if cum_w > 0 else -1.0)
    worst10y_pre = float(np.min(w10_pre)) if w10_pre else np.nan
    # rolling 5Y
    r5 = []
    for i in range(len(arr) - 5 + 1):
        cum_w = float(np.prod(1.0 + arr[i:i+5]))
        r5.append(cum_w**(1.0/5) - 1.0 if cum_w > 0 else -1.0)
    p10_5y_pre = float(np.percentile(r5, 10)) if r5 else np.nan

    # apply tax to worst10y and p10_5y
    worst10y_aft = worst10y_pre * TAX_FACTOR
    p10_5y_aft = p10_5y_pre * TAX_FACTOR

    # tax-adjusted IS/OOS CAGR using yearly compounding
    is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1977 <= y <= 2020]]
    oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
    def _geo(x):
        if len(x) == 0: return np.nan
        c = float(np.prod(1.0 + x.values))
        return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
    cagr_is_aft  = _geo(is_subset)
    cagr_oos_aft = _geo(oos_subset)

    return {
        'CAGR_IS': cagr_is_aft,
        'CAGR_OOS': cagr_oos_aft,
        'IS_OOS_gap': cagr_is_aft - cagr_oos_aft,
        'Sharpe_OOS': sharpe_oos,
        'MaxDD_FULL': maxdd_full,
        'Worst10Y_star': worst10y_aft,
        'P10_5Y': p10_5y_aft,
        'yearly_pre': yearly,
        'yearly_aft': yr_aft,
    }


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print(f'G19E: 3 新戦略 daily cost 評価 (F8 R5, F7v3+E4, NDX 1x B&H)')
    print(f'実行日: 2026-06-02')
    print('=' * 80)

    print('\n[S1] Loading shared assets...')
    a = load_shared_assets()
    dates = a['dates']
    close = a['close']
    ret_nas = a['ret']
    windows = generate_windows(dates)
    print(f'  Windows: {len(windows)}')

    raw_a2_vals = a['raw_a2'].values
    vz_vals = a['vz'].values
    bull_mask = raw_a2_vals > THRESHOLD

    # ---- F8 R5_CALM_BOOST (ε=0 baseline of F10) ----
    # 既に load_shared_assets で wn_ref_f8 として準備済み
    wn_f8 = a['wn_ref_f8']
    wg_f8 = a['wg_ref_f8']
    wb_f8 = a['wb_ref_f8']

    # ---- F7v3+E4 A:tilt=2.0 ----
    wn_f7v3, wb_f7v3 = build_f7v3_wn(raw_a2_vals, a['wn_A'], a['wb_A'])
    wg_f7v3 = a['wg_A']

    rows = []
    yearly_records = {}

    print('\n[S2] CFD 2戦略 × 4 spread = 8 configs')
    cfd_strats = [
        ('F8 R5_CALM_BOOST', 'lev_mod_e4', wn_f8, wg_f8, wb_f8, a['L_s2_lmax7']),
        ('F7v3+E4 A:tilt=2.0', 'lev_mod_e4', wn_f7v3, wg_f7v3, wb_f7v3, a['L_s2_lmax7']),
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
            yr_pre = m['yearly']
            yr_aft = yr_pre.apply(apply_tax_cfd_decimal)
            yearly_records[(label, case)] = yr_aft

            # WFA
            try:
                wfa = wfa_metrics(nav_adj, dates, windows,
                                   lev_arr=np.asarray(L_s2.values),
                                   wn_arr=wn, wb_arr=wb)
                wfa_ci95 = wfa.get('WFA_CI95_lo', np.nan)
                wfa_wfe  = wfa.get('WFA_WFE', np.nan)
                trades_yr = wfa.get('mean_Trades_yr', np.nan)
            except Exception as e:
                print(f'  WFA failed: {e}')
                wfa_ci95 = np.nan; wfa_wfe = np.nan; trades_yr = np.nan

            # tax-adjusted IS/OOS via yearly compounded
            is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1977 <= y <= 2020]]
            oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
            def _geo(x):
                if len(x) == 0: return np.nan
                c = float(np.prod(1.0 + x.values))
                return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
            cagr_is_aft = _geo(is_subset)
            cagr_oos_aft = _geo(oos_subset)

            rows.append(dict(
                strategy=label, case=case, is_cfd=True, spread_rt=spread_rt,
                CAGR_IS=cagr_is_aft, CAGR_OOS=cagr_oos_aft,
                IS_OOS_gap=cagr_is_aft - cagr_oos_aft,
                Sharpe_OOS=m['Sharpe_OOS'], MaxDD_FULL=m['MaxDD_FULL'],
                Worst10Y_star=m['Worst10Y_star'], P10_5Y=m['P10_5Y'],
                Trades_yr=trades_yr,
                WFA_WFE=wfa_wfe, WFA_CI95_lo=wfa_ci95,
                yr_cost_approx=yr_cost_approx,
            ))
            print(f'  [{label}|{case}] CAGR_IS={cagr_is_aft*100:+.2f}%, '
                  f'CAGR_OOS={cagr_oos_aft*100:+.2f}%, gap={(cagr_is_aft-cagr_oos_aft)*100:+.2f}pp, '
                  f'Sharpe={m["Sharpe_OOS"]:+.3f}')

    # ---- NDX 1x B&H ----
    print('\n[S3] NDX 1x B&H (cost=0, tax only)')
    bnh = compute_bnh_metrics_after_tax(close, dates)
    yearly_records[('NDX 1x B&H', 'benchmark')] = bnh['yearly_aft']
    rows.append(dict(
        strategy='NDX 1x B&H', case='benchmark', is_cfd=False, spread_rt=None,
        CAGR_IS=bnh['CAGR_IS'], CAGR_OOS=bnh['CAGR_OOS'],
        IS_OOS_gap=bnh['IS_OOS_gap'],
        Sharpe_OOS=bnh['Sharpe_OOS'], MaxDD_FULL=bnh['MaxDD_FULL'],
        Worst10Y_star=bnh['Worst10Y_star'], P10_5Y=bnh['P10_5Y'],
        Trades_yr=0.0, WFA_WFE=None, WFA_CI95_lo=None, yr_cost_approx=0.0,
    ))
    print(f'  [NDX 1x B&H] CAGR_IS={bnh["CAGR_IS"]*100:+.2f}%, '
          f'CAGR_OOS={bnh["CAGR_OOS"]*100:+.2f}%, gap={bnh["IS_OOS_gap"]*100:+.2f}pp, '
          f'Sharpe={bnh["Sharpe_OOS"]:+.3f}, MaxDD={bnh["MaxDD_FULL"]*100:+.2f}%')

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(BASE, 'g19e_3strategies_daily_cost.csv')
    df_out.to_csv(out_csv, index=False)
    print(f'\n→ CSV saved: {out_csv}')

    # Yearly CSV
    yr_rows = []
    for (label, case), yr in yearly_records.items():
        for year, v in yr.items():
            yr_rows.append(dict(strategy=label, case=case, year=year, return_pct=v*100))
    yr_df = pd.DataFrame(yr_rows)
    yr_csv = os.path.join(BASE, 'g19e_3strategies_yearly.csv')
    yr_df.to_csv(yr_csv, index=False)
    print(f'→ Yearly CSV: {yr_csv}')

    # Summary (moderate ケース)
    print('\n[S4] Summary (moderate case for CFD, benchmark for B&H):')
    sub = df_out[df_out['case'].isin(['moderate', 'benchmark'])]
    print(f'{"Strategy":>22s} {"CAGR_OOS":>10s} {"IS-OOS_gap":>12s} {"Sharpe_OOS":>11s} {"MaxDD":>9s} {"Worst10Y":>10s}')
    print('-' * 80)
    for _, r in sub.iterrows():
        print(f'{r["strategy"]:>22s} {r["CAGR_OOS"]*100:>+8.2f}% {r["IS_OOS_gap"]*100:>+10.2f}pp '
              f'{r["Sharpe_OOS"]:>+11.3f} {r["MaxDD_FULL"]*100:>+7.2f}% {r["Worst10Y_star"]*100:>+8.2f}%')


if __name__ == '__main__':
    main()
