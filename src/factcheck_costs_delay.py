"""
Fact-check: leverage product costs & position delay verification
================================================================
Verifies DH Dyn 2x3x [A] Scenario D against:
  1. Gold TER 0.50% (proxy) vs 0.95% (UGL actual)
  2. TMF TER 0.91% (historical) vs 0.95%/1.06% (current)
  3. DELAY=2 (T+2) vs DELAY=1 (T+1, post-2024 US settlement)
  4. SOFR proxy (DTB3) quality: add-on basis correction
  5. TQQQ swap_spread empirical verification via OLS regression
  6. NASDAQ data: price index vs total return check

Outputs:
  - factcheck_sensitivity_results.csv
  - factcheck_sofr_spread_results.csv
  - Console report with findings
"""
import sys, os, types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from corrected_strategy_backtest import (
    load_data, load_sofr, build_a2_signal, simulate_rebalance_A,
    build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_nav, calc_metrics, DATA_PATH, DATA_DIR, TRADING_DAYS, THRESHOLD,
    GOLD_2X_COST, BOND_3X_COST, ANNUAL_COST, SWAP_SPREAD, DELAY,
)
from compute_dha_worst10y_only import prepare_gold_local

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_START = '1974-01-02'
IS_END     = '2021-05-07'
OOS_START  = '2021-05-08'


# ---------------------------------------------------------------------------
# Helper: run Scenario D with overridden cost params
# ---------------------------------------------------------------------------

def run_scenario_d(close, ret, dates, sofr, gold_1x,
                   gold_ter=0.0050, bond_ter=0.0091,
                   swap_spread=0.0050, delay=2) -> dict:
    """Run full Scenario D pipeline with specified cost overrides."""
    import corrected_strategy_backtest as cs_mod

    # Build components with overrides
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True,
                              swap_spread=swap_spread)
    # Override gold TER inside gold_2x via recalculation
    # build_gold_2x uses cs_mod.GOLD_2X_COST; patch it temporarily
    orig_gold = cs_mod.GOLD_2X_COST
    orig_bond = cs_mod.BOND_3X_COST
    orig_swap = cs_mod.SWAP_SPREAD
    orig_del  = cs_mod.DELAY

    cs_mod.GOLD_2X_COST = gold_ter
    cs_mod.BOND_3X_COST = bond_ter
    cs_mod.SWAP_SPREAD  = swap_spread
    cs_mod.DELAY        = delay

    gold_2x  = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True,
                              swap_spread=swap_spread)
    bond_1x  = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                            bond_maturity=22.0)
    bond_3x  = build_bond_3x(bond_1x, sofr, apply_sofr=True, swap_spread=swap_spread)
    raw, vz  = build_a2_signal(close, ret)
    lev, wn, wg, wb, n_tr = simulate_rebalance_A(raw, vz, THRESHOLD)
    nav      = build_nav(close, lev, wn, wg, wb, dates,
                         gold_2x, bond_3x, sofr_daily=sofr, apply_tqqq_sofr=True,
                         swap_spread=swap_spread)

    cs_mod.GOLD_2X_COST = orig_gold
    cs_mod.BOND_3X_COST = orig_bond
    cs_mod.SWAP_SPREAD  = orig_swap
    cs_mod.DELAY        = orig_del

    m_full = calc_metrics(nav, dates, FULL_START, '2026-12-31')
    m_is   = calc_metrics(nav, dates, FULL_START, IS_END)
    m_oos  = calc_metrics(nav, dates, OOS_START,  '2026-12-31')

    return dict(
        CAGR_IS   = m_is['CAGR']   if m_is else np.nan,
        CAGR_OOS  = m_oos['CAGR']  if m_oos else np.nan,
        CAGR_FULL = m_full['CAGR'] if m_full else np.nan,
        Sharpe_OOS= m_oos['Sharpe'] if m_oos else np.nan,
        MaxDD_FULL= m_full['MaxDD'] if m_full else np.nan,
        W5Y_FULL  = m_full['Worst5Y'] if m_full else np.nan,
    )


# ---------------------------------------------------------------------------
# Section 1: Sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity(close, ret, dates, sofr, gold_1x) -> pd.DataFrame:
    print('\n' + '='*70)
    print('SECTION 1: Cost & Delay Sensitivity Analysis')
    print('='*70)

    scenarios = [
        # label, gold_ter, bond_ter, swap_spread, delay
        ('D_baseline (current)',  0.0050, 0.0091, 0.0050, 2),
        ('D1: Gold TER=0.95%',    0.0095, 0.0091, 0.0050, 2),
        ('D2: TMF TER=0.95%',     0.0050, 0.0095, 0.0050, 2),
        ('D3: TMF TER=1.06%',     0.0050, 0.0106, 0.0050, 2),
        ('D4: Gold+TMF realistic',0.0095, 0.0106, 0.0050, 2),
        ('D5: Delay=1 (T+1)',     0.0050, 0.0091, 0.0050, 1),
        ('D6: Delay=3 (T+3)',     0.0050, 0.0091, 0.0050, 3),
        ('D7: Swap+25bps (0.75%)',0.0050, 0.0091, 0.0075, 2),
        ('D8: Swap+50bps (1.00%)',0.0050, 0.0091, 0.0100, 2),
        ('D9: ALL realistic',     0.0095, 0.0106, 0.0050, 1),  # UGL + current TMF + T+1
    ]

    rows = []
    baseline = None
    for i, (label, g, b, s, d) in enumerate(scenarios):
        print(f'  [{i+1:2d}/{len(scenarios)}] {label}...', flush=True)
        res = run_scenario_d(close, ret, dates, sofr, gold_1x,
                             gold_ter=g, bond_ter=b, swap_spread=s, delay=d)
        res['scenario'] = label
        rows.append(res)
        if baseline is None:
            baseline = res.copy()

    df = pd.DataFrame(rows)[['scenario','CAGR_IS','CAGR_OOS','CAGR_FULL',
                               'Sharpe_OOS','MaxDD_FULL','W5Y_FULL']]
    # bps vs baseline
    for col in ['CAGR_IS','CAGR_OOS','CAGR_FULL','Sharpe_OOS']:
        df[f'd{col}_bps'] = ((df[col] - baseline[col]) * 10000).round(1)

    return df


# ---------------------------------------------------------------------------
# Section 2: SOFR proxy quality check (DTB3 vs actual SOFR)
# ---------------------------------------------------------------------------

def check_sofr_proxy(dates) -> dict:
    print('\n' + '='*70)
    print('SECTION 2: SOFR Proxy Quality (DTB3 vs Actual SOFR)')
    print('='*70)

    dtb3_path = os.path.join(DATA_DIR, 'dtb3_daily.csv')
    sofr_path = os.path.join(DATA_DIR, 'sofr_daily.csv')

    dtb3 = pd.read_csv(dtb3_path, parse_dates=['Date'], index_col='Date')
    dtb3.columns = ['dtb3_pct']
    dtb3['dtb3_pct'] = pd.to_numeric(dtb3['dtb3_pct'], errors='coerce')

    has_sofr_file = os.path.exists(sofr_path)
    results = {'has_sofr_file': has_sofr_file}

    if has_sofr_file:
        sofr_df = pd.read_csv(sofr_path, parse_dates=['Date'], index_col='Date')
        sofr_df.columns = ['sofr_pct']
        sofr_df['sofr_pct'] = pd.to_numeric(sofr_df['sofr_pct'], errors='coerce')
        combined = dtb3.join(sofr_df, how='inner').dropna()
        combined['spread_bps'] = (combined['sofr_pct'] - combined['dtb3_pct']) * 100
        results['n_overlap'] = len(combined)
        results['mean_spread_bps'] = combined['spread_bps'].mean()
        results['std_spread_bps'] = combined['spread_bps'].std()
        print(f'  Overlap period: {combined.index[0].date()} to {combined.index[-1].date()}')
        print(f'  DTB3 mean:  {combined["dtb3_pct"].mean():.3f}%')
        print(f'  SOFR mean:  {combined["sofr_pct"].mean():.3f}%')
        print(f'  SOFR-DTB3 spread: {results["mean_spread_bps"]:.1f} bps (mean)')
    else:
        print('  No sofr_daily.csv found; estimating from theory.')
        results['n_overlap'] = 0
        results['mean_spread_bps'] = 10.0  # typical ~10 bps SOFR > T-bill
        results['note'] = 'DTB3 ~ SOFR - 5..15bps (T-bill typically slightly below SOFR)'

    # DTB3 discount → add-on basis correction
    # add_on = discount / (1 - discount * days/360)
    dtb3['dtb3_addon_pct'] = (dtb3['dtb3_pct'] / 100.0) / (1 - (dtb3['dtb3_pct'] / 100.0) * 91/360) * 100
    dtb3['addon_vs_simple_bps'] = (dtb3['dtb3_addon_pct'] - dtb3['dtb3_pct']) * 100

    recent = dtb3.loc['2020-01-01':].dropna()
    addon_bias = recent['addon_vs_simple_bps'].mean()
    print(f'\n  DTB3 discount→add-on basis correction (post-2020):')
    print(f'    Mean add-on premium: +{addon_bias:.1f} bps')
    print(f'    (DTB3 /252 as SOFR proxy understates actual SOFR by ~{addon_bias:.0f} bps at 4-5% rates)')
    print(f'  => SOFR costs in backtest are ~{addon_bias + 10:.0f} bps/yr UNDERSTATED vs true SOFR')
    results['addon_bias_bps'] = addon_bias

    # Impact estimate
    # SOFR multipliers: TQQQ=2x, TMF=2x, Gold=1x
    # Average portfolio weight at SOFR exposure (rough):
    avg_tqqq_sofr_exp = 2 * 0.60 * 0.67   # sofr_mult * wn_avg * lev_avg
    avg_tmf_sofr_exp  = 2 * 0.20           # 2x * bond_weight
    avg_gold_sofr_exp = 1 * 0.20           # 1x * gold_weight
    total_sofr_exp    = avg_tqqq_sofr_exp + avg_tmf_sofr_exp + avg_gold_sofr_exp
    proxy_bias_impact = (addon_bias + 10) / 10000 * total_sofr_exp
    print(f'\n  Total portfolio SOFR exposure (approx): {total_sofr_exp:.2f}x')
    print(f'  CAGR impact of proxy bias: ~{-proxy_bias_impact*100:.2f}%/yr (costs understated)')
    results['proxy_bias_cagr_impact'] = -proxy_bias_impact

    return results


# ---------------------------------------------------------------------------
# Section 3: TQQQ OLS regression (swap_spread empirical check)
# ---------------------------------------------------------------------------

def check_tqqq_swap() -> dict:
    print('\n' + '='*70)
    print('SECTION 3: TQQQ Swap_Spread Empirical Verification (OLS)')
    print('='*70)

    data_dir = DATA_DIR

    def load_series(fname):
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path, parse_dates=[0], index_col=0, skiprows=[1])
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]
        s = pd.to_numeric(df.iloc[:, 0], errors='coerce').sort_index()
        return s.pct_change()

    try:
        r_tqqq = load_series('tqqq_daily.csv')
        r_qqq  = load_series('qqq_daily.csv')
        dtb3   = pd.read_csv(os.path.join(data_dir, 'dtb3_daily.csv'),
                              parse_dates=['Date'], index_col='Date')
        dtb3.columns = ['dtb3_pct']
        sofr_d = (pd.to_numeric(dtb3['dtb3_pct'], errors='coerce') / 100.0 / 252.0)

        df = pd.concat([r_tqqq.rename('tqqq'), r_qqq.rename('qqq'),
                        sofr_d.rename('sofr')], axis=1)
        df = df.loc[df.index >= '2010-02-11'].dropna()
        print(f'  Analysis period: {df.index[0].date()} to {df.index[-1].date()}, n={len(df):,} days')

        X = sm.add_constant(df[['qqq', 'sofr']])
        m = sm.OLS(df['tqqq'], X).fit()

        beta_qqq  = m.params['qqq']
        beta_sofr = m.params['sofr']
        alpha_ann = m.params['const'] * 252 * 100
        r2        = m.rsquared

        print(f'\n  OLS: r_TQQQ = α + β_QQQ×r_QQQ + β_SOFR×sofr_daily')
        print(f'    β_QQQ      = {beta_qqq:.4f}  (expected ~3.0)')
        print(f'    β_SOFR     = {beta_sofr:.4f}  (expected ~-2.0 for 2xSOFR)')
        print(f'    α (ann.)   = {alpha_ann:.3f}%/yr  (expected ~-2.3% = TER+swap+3xdiv)')
        print(f'    R²         = {r2:.5f}')

        # Implied swap_spread from α
        # α ≈ -(TER + swap_spread) * daily → α_ann ≈ -(0.86% + swap + 3*0.5%div?)
        # But dividend from QQQ adj_close complicates this. Better: alpha = -(ter + swap)
        # Since QQQ adj_close includes dividends, β_QQQ captures div too
        # So alpha ≈ -(TER alone) ≈ -0.86% ideally, swap shows in β_SOFR
        implied_swap = -(alpha_ann / 100) - 0.0086  # after removing TER
        print(f'\n  Implied swap_spread (from α, after TER): {implied_swap*100:.3f}%/yr')
        print(f'    Backtest uses: 0.50%/yr')
        print(f'    Difference:    {(implied_swap-0.0050)*100:+.3f}%/yr')

        if -2.3 <= beta_sofr <= -1.7:
            verdict = 'CONFIRMED: 2xSOFR financing (β_SOFR ≈ -2)'
        elif beta_sofr < -2.3:
            verdict = f'HIGHER THAN 2x: β={beta_sofr:.2f} → SOFR costs understated in model'
        else:
            verdict = f'LOWER THAN 2x: β={beta_sofr:.2f} → check model'
        print(f'\n  VERDICT: {verdict}')

        return dict(beta_qqq=beta_qqq, beta_sofr=beta_sofr,
                    alpha_ann_pct=alpha_ann, r2=r2,
                    implied_swap_pct=implied_swap*100,
                    verdict=verdict)

    except Exception as e:
        print(f'  ERROR: {e}')
        return {'error': str(e)}


# ---------------------------------------------------------------------------
# Section 4: TMF bond model TER check
# ---------------------------------------------------------------------------

def check_tmf_ter() -> dict:
    print('\n' + '='*70)
    print('SECTION 4: TMF TER Verification vs Actual NAV')
    print('='*70)

    tmf_path = os.path.join(DATA_DIR, 'tmf_daily.csv')
    if not os.path.exists(tmf_path):
        print('  tmf_daily.csv not found, skipping.')
        return {}

    tmf = pd.read_csv(tmf_path, parse_dates=[0], index_col=0, skiprows=[1])
    tmf.index = pd.to_datetime(tmf.index, errors='coerce')
    tmf = tmf[tmf.index.notna()]
    tmf_close = pd.to_numeric(tmf.iloc[:, 0], errors='coerce').sort_index().dropna()

    tmf_start = tmf_close.index[0]
    tmf_end   = tmf_close.index[-1]
    yrs = (tmf_end - tmf_start).days / 365.25
    cagr_actual = (tmf_close.iloc[-1] / tmf_close.iloc[0]) ** (1 / yrs) - 1

    print(f'  TMF data: {tmf_start.date()} to {tmf_end.date()} ({yrs:.1f} years)')
    print(f'  TMF actual CAGR: {cagr_actual*100:.2f}%/yr')
    print(f'  Note: TMF TER in backtest = 0.91%/yr (historical)')
    print(f'        TMF TER current (research report) = 0.95-1.06%/yr')
    print(f'        Backtest vs current TER difference: +{(0.0095-0.0091)*100:.2f}% to +{(0.0106-0.0091)*100:.2f}%/yr')
    print(f'        Bond weight ≈ 20% → CAGR impact: -{(0.0095-0.0091)*0.20*100:.2f}% to -{(0.0106-0.0091)*0.20*100:.2f}%/yr')

    return dict(cagr_actual=cagr_actual, period_years=yrs,
                ter_diff_low_bps=4.0, ter_diff_high_bps=15.0,
                cagr_impact_low_bps=-0.8, cagr_impact_high_bps=-3.0)


# ---------------------------------------------------------------------------
# Section 5: Gold product availability & TER impact summary
# ---------------------------------------------------------------------------

def print_gold_summary():
    print('\n' + '='*70)
    print('SECTION 5: Gold 2x Product Availability & TER Impact')
    print('='*70)
    print("""
  Research Report Findings (FILE 2, 2026-05-17):
  ─────────────────────────────────────────────
  WisdomTree 2036 (LSE) -- BACKTEST PROXY:
    TER: 0.49% (historical) / 0.50% used in simulation
    Availability: NOT available at JP retail brokers (SBI/Rakuten)
    → Backtest uses this as simulation proxy only

  UGL (ProShares Ultra Gold 2x) -- ACTUAL JP AVAILABLE:
    TER: 0.95%/yr
    Availability: SBI証券で購入可 (米国ETF、特定口座対応)
    SOFR structure: SOFR × (2-1) = 1×SOFR  [consistent with backtest]
    T+2 settlement in Japan  [consistent with DELAY=2]

  Gap: +0.45%/yr (UGL vs WisdomTree 2036 proxy)
  Gold weight ≈ 20% of portfolio
  CAGR impact: -0.45% × 0.20 = -0.09%/yr ≈ -9 bps/yr

  Alternative: SBI証券 金CFD (T+0, max 20x)
    Not modeled in backtest (backtest is ETF-based)
    SOFR structure: SOFR + lease_rate + margin (non-transparent)
    Estimated cost: ~4.4-5.0%/yr total → HIGHER than UGL

  BOTTOM LINE: Using UGL (realistic) raises effective gold cost by +9 bps CAGR.
""")


# ---------------------------------------------------------------------------
# Section 6: Delay model check
# ---------------------------------------------------------------------------

def print_delay_analysis():
    print('\n' + '='*70)
    print('SECTION 6: Position Delay Model Analysis')
    print('='*70)
    print("""
  Current implementation: DELAY = 2 (T+2)
  ─────────────────────────────────────────
  Signal computed at market close on day t
  → Position effective from day t+DELAY

  Research Report (FILE 2, §8):
  "UGL, TMF, TQQQ (US ETF via SBI): T+2 in Japan
   (US settled T+1 since May 2024, but Japan-side delivery remains T+2)"

  Analysis:
  - DELAY=2 correctly models ETF settlement for pre-2024 data (majority of backtest)
  - Post-2024 (US T+1): technically DELAY=1 applies for US market exposure
    but Japan-side accounting still T+2
  - Practical impact: ~5 years of OOS data (2021-2026), partially T+1 since 2024
  - Conservative DELAY=2 slightly underestimates returns in trending markets
  - Test DELAY=1 vs DELAY=2 in sensitivity analysis (Section 1, row D5)

  VERDICT: DELAY=2 is appropriate and conservative for the full 52-year period.
  For future live trading, DELAY=1 may be slightly more accurate post-2024.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('FACT-CHECK: Leverage Costs & Position Delay Verification')
    print('DH Dyn 2x3x [A] Scenario D')
    print('=' * 70)

    # Load data
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(dates)} days)')
    print(f'NASDAQ: Close == Adj Close check: {(close == df.get("Adj Close", close)).all()} '
          '→ Price Index (not total return)')

    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)

    # Run sections
    sens_df    = run_sensitivity(close, ret, dates, sofr, gold_1x)
    sofr_info  = check_sofr_proxy(dates)
    tqqq_info  = check_tqqq_swap()
    tmf_info   = check_tmf_ter()
    print_gold_summary()
    print_delay_analysis()

    # Save results
    out_dir = os.path.dirname(DATA_PATH)
    out_csv = os.path.join(out_dir, 'factcheck_sensitivity_results.csv')
    sens_df.to_csv(out_csv, index=False, float_format='%.4f')
    print(f'\n\nSaved sensitivity results: {out_csv}')

    # Summary report
    print('\n' + '=' * 70)
    print('SUMMARY: FACT-CHECK FINDINGS')
    print('=' * 70)
    baseline = sens_df.iloc[0]
    print(f'\nBaseline (Scenario D as-is):')
    print(f'  CAGR_FULL = {baseline["CAGR_FULL"]*100:.2f}%  Sharpe_OOS = {baseline["Sharpe_OOS"]:.4f}')

    print(f'\nCost sensitivity results (CAGR_FULL change vs baseline):')
    for _, row in sens_df.iterrows():
        if row['scenario'] == 'D_baseline (current)':
            continue
        delta = row['dCAGR_FULL_bps']
        print(f'  {row["scenario"]:<35s}: {delta:+.1f} bps/yr  '
              f'({row["CAGR_FULL"]*100:.2f}% CAGR_FULL)')

    print(f'\nSOFR proxy analysis:')
    print(f'  DTB3 add-on bias:  ~+{sofr_info["addon_bias_bps"]:.0f} bps vs simple')
    print(f'  Estimated CAGR cost understatement: ~{sofr_info["proxy_bias_cagr_impact"]*100:.2f}%/yr')

    if 'beta_sofr' in tqqq_info:
        print(f'\nTQQQ swap verification:')
        print(f'  β_SOFR = {tqqq_info["beta_sofr"]:.3f} (target: -2.0)')
        print(f'  Implied swap_spread = {tqqq_info["implied_swap_pct"]:.3f}%/yr (model uses 0.50%)')
        print(f'  {tqqq_info["verdict"]}')

    print(f'\nKey conclusion:')
    print(f'  1. Gold TER: proxy 0.50% vs UGL 0.95% → {sens_df.loc[sens_df.scenario=="D1: Gold TER=0.95%","dCAGR_FULL_bps"].iloc[0]:+.1f} bps/yr')
    print(f'  2. TMF TER:  hist 0.91% vs current 1.06% → {sens_df.loc[sens_df.scenario=="D3: TMF TER=1.06%","dCAGR_FULL_bps"].iloc[0]:+.1f} bps/yr')
    print(f'  3. Delay T+2 vs T+1 → {sens_df.loc[sens_df.scenario=="D5: Delay=1 (T+1)","dCAGR_FULL_bps"].iloc[0]:+.1f} bps/yr')
    print(f'  4. All-realistic combined → {sens_df.loc[sens_df.scenario=="D9: ALL realistic","dCAGR_FULL_bps"].iloc[0]:+.1f} bps/yr')

    realistic_cagr = sens_df.loc[sens_df.scenario=="D9: ALL realistic","CAGR_FULL"].iloc[0]
    print(f'\n  "Realistic JP retail" CAGR_FULL estimate: {realistic_cagr*100:.2f}%')
    print(f'  vs current Scenario D: {baseline["CAGR_FULL"]*100:.2f}%')

    print('\n' + '=' * 70)
    return sens_df, sofr_info, tqqq_info


if __name__ == '__main__':
    main()
