"""
Step 2: Bond Timing Signal Integration into Dyn 2x3x
Tests whether adding Bond-specific signals improves portfolio performance.
Run: python research_step2_bond_timing.py [partA|partB|partC|partD|partE|partF|all]
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'NASDAQ_extended_to_2026.csv')
DELAY = 2; ANNUAL_COST = 0.0086; BASE_LEV = 3.0; THRESHOLD = 0.20
GOLD_2X_COST = 0.005; BOND_3X_COST = 0.0091
OOS_DATE = '2021-05-07'

# =============================================================================
# Shared: Data & Signal Building
# =============================================================================
def load_and_build():
    """Load data, build A2 signals, leveraged NAVs."""
    from backtest_engine import load_data, calc_dd_signal
    from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
    from test_vix_integration import calc_vix_proxy
    from test_portfolio_diversification import prepare_gold_data, prepare_bond_data

    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    n = len(df); years = n / 252

    # A2 optimized signals
    dd = calc_dd_signal(close, 0.82, 0.92)
    var = pd.Series(index=returns.index, dtype=float)
    var.iloc[0] = returns.iloc[:20].var() if len(returns) > 20 else 0.0001
    for i in range(1, len(returns)):
        r = returns.iloc[i]; a = 2 / (11 if r < 0 else 31)
        var.iloc[i] = (1 - a) * var.iloc[i - 1] + a * (r ** 2)
    av = np.sqrt(var * 252)
    ma150 = close.rolling(150).mean(); ratio = close / ma150
    ttv = (0.10 + 0.20 * (ratio - 0.85) / 0.30).clip(0.10, 0.30).fillna(0.20)
    vt = (ttv / av).clip(0, 1.0)
    ma200 = close.rolling(200).mean(); sl = ma200.pct_change()
    sm = sl.rolling(60).mean(); ss = sl.rolling(60).std().replace(0, 0.0001)
    z = (sl - sm) / ss; slope = (0.9 + 0.35 * z).clip(0.3, 1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp = calc_vix_proxy(returns)
    vma = vp.rolling(252).mean(); vs = vp.rolling(252).std().replace(0, 0.001)
    vz = (vp - vma) / vs; vm = (1.0 - 0.25 * vz).clip(0.5, 1.15)
    raw = dd * vt * slope * mom * vm; raw = raw.clip(0, 1.0).fillna(0)
    lev = rebalance_threshold(raw, THRESHOLD)

    # A2 NAV
    lr = returns * BASE_LEV; dc = ANNUAL_COST / 252
    dl = lev.shift(DELAY); sr = dl * (lr - dc); sr = sr.fillna(0)
    nav_a2 = (1 + sr).cumprod()

    # Gold/Bond
    gold_1x = prepare_gold_data(dates)
    bond_1x = prepare_bond_data(dates)

    # Bond yield (for correlation signal)
    import yfinance as yf
    tnx = yf.Ticker('^TNX')
    tnx_data = tnx.history(start='1974-01-01', end='2026-04-01')
    tnx_data = tnx_data.reset_index()
    tnx_data['Date'] = pd.to_datetime(tnx_data['Date']).dt.tz_localize(None)
    tnx_daily = tnx_data[['Date', 'Close']].rename(columns={'Close': 'Yield_pct'})
    ndf = pd.DataFrame({'Date': dates})
    merged = ndf.merge(tnx_daily, on='Date', how='left')
    merged['Yield_pct'] = merged['Yield_pct'].ffill().bfill()

    # Leveraged NAVs
    g2 = np.ones(n); b3 = np.ones(n)
    for i in range(1, n):
        gr = gold_1x[i] / gold_1x[i - 1] - 1 if gold_1x[i - 1] > 0 else 0
        g2[i] = g2[i - 1] * (1 + gr * 2 - GOLD_2X_COST / 252)
        br = bond_1x[i] / bond_1x[i - 1] - 1 if bond_1x[i - 1] > 0 else 0
        b3[i] = b3[i - 1] * (1 + br * 3 - BOND_3X_COST / 252)

    # Bond signals for timing
    nasdaq_ret = returns.values
    bond_ret = np.diff(bond_1x, prepend=bond_1x[0]) / np.maximum(bond_1x, 1e-10)
    bond_ret[0] = 0
    bond_nav_s = pd.Series(bond_1x)

    # MA5: NQ-Bond rolling correlation (252-day)
    nq_r = pd.Series(nasdaq_ret); bn_r = pd.Series(bond_ret)
    nq_bond_corr = nq_r.rolling(252).corr(bn_r).fillna(0).values

    # TF5: Bond MACD (26/52/18)
    ema_fast = bond_nav_s.ewm(span=26).mean()
    ema_slow = bond_nav_s.ewm(span=52).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=18).mean()
    bond_macd = ((macd_line - signal_line) / bond_nav_s).fillna(0).values

    # VL2: Bond VolZ (42/126)
    bond_vol = bn_r.rolling(42).std() * np.sqrt(252)
    bv_mean = bond_vol.rolling(126).mean()
    bv_std = bond_vol.rolling(126).std().replace(0, 0.001)
    bond_volz = ((bond_vol - bv_mean) / bv_std).fillna(0).values

    return {
        'dates': dates, 'close': close, 'n': n, 'years': years,
        'nav_a2': nav_a2.values, 'raw_lev': raw.values, 'vix_z': vz.fillna(0).values,
        'gold_2x': g2, 'bond_3x': b3,
        'nq_bond_corr': nq_bond_corr, 'bond_macd': bond_macd, 'bond_volz': bond_volz,
    }


def build_portfolio(nav_nq, nav_gold, nav_bond, wn, wg, wb):
    """Build 3-asset portfolio with given weight arrays."""
    n = len(nav_nq)
    port = np.ones(n)
    cur_wn, cur_wg, cur_wb = wn[0], wg[0], wb[0]

    for i in range(1, n):
        rn = nav_nq[i] / nav_nq[i - 1] - 1 if nav_nq[i - 1] > 0 else 0
        rg = nav_gold[i] / nav_gold[i - 1] - 1 if nav_gold[i - 1] > 0 else 0
        rb = nav_bond[i] / nav_bond[i - 1] - 1 if nav_bond[i - 1] > 0 else 0
        port_ret = cur_wn * rn + cur_wg * rg + cur_wb * rb
        port[i] = port[i - 1] * (1 + port_ret)

        # Update current weights (drift)
        cur_wn *= (1 + rn); cur_wg *= (1 + rg); cur_wb *= (1 + rb)
        total = cur_wn + cur_wg + cur_wb
        if total > 0:
            cur_wn /= total; cur_wg /= total; cur_wb /= total

        # Rebalance if drift exceeds threshold or target changed significantly
        drift = abs(cur_wn - wn[i]) + abs(cur_wg - wg[i]) + abs(cur_wb - wb[i])
        if drift > 0.10 or (i % 63 == 0):  # 10% drift or quarterly
            cur_wn, cur_wg, cur_wb = wn[i], wg[i], wb[i]

    return port


def calc_metrics(nav, dates, name):
    """Calculate standard metrics."""
    n = len(nav); yrs = n / 252
    ret = np.diff(nav, prepend=nav[0]) / np.maximum(nav, 1e-10)
    ret[0] = 0
    cagr = nav[-1] ** (1 / yrs) - 1
    dd = (nav / np.maximum.accumulate(nav) - 1).min()
    r_s = pd.Series(ret)
    sh = (r_s.mean() * 252) / (r_s.std() * np.sqrt(252)) if r_s.std() > 0 else 0

    # Worst 5Y, 10Y
    nav_s = pd.Series(nav)
    w5 = ((nav_s / nav_s.shift(252 * 5)) ** (1 / 5) - 1).min() if n > 252 * 5 else np.nan
    w10 = ((nav_s / nav_s.shift(252 * 10)) ** (1 / 10) - 1).min() if n > 252 * 10 else np.nan

    # OOS metrics
    d = pd.to_datetime(dates)
    oos_idx = d[d >= OOS_DATE].index
    if len(oos_idx) > 0:
        si = oos_idx[0]
        oos_nav = nav[si:] / nav[si]
        oos_yrs = len(oos_nav) / 252
        oos_cagr = oos_nav[-1] ** (1 / oos_yrs) - 1 if oos_yrs > 0 else 0
        oos_ret = np.diff(oos_nav, prepend=oos_nav[0]) / np.maximum(oos_nav, 1e-10)
        oos_ret[0] = 0
        oos_r = pd.Series(oos_ret)
        oos_sh = (oos_r.mean() * 252) / (oos_r.std() * np.sqrt(252)) if oos_r.std() > 0 else 0
        oos_dd = (oos_nav / np.maximum.accumulate(oos_nav) - 1).min()
    else:
        oos_cagr = oos_sh = oos_dd = np.nan

    # Yearly win rate
    ndf = pd.DataFrame({'nav': nav, 'date': dates.values})
    ndf['year'] = pd.to_datetime(ndf['date']).dt.year
    yn = ndf.groupby('year')['nav'].last(); ar = yn.pct_change().dropna()
    wr = (ar > 0).mean() if len(ar) > 0 else 0

    return {
        'Strategy': name, 'CAGR': cagr, 'Sharpe': sh, 'MaxDD': dd,
        'Worst5Y': w5, 'Worst10Y': w10, 'WinRate': wr,
        'OOS_CAGR': oos_cagr, 'OOS_Sharpe': oos_sh, 'OOS_MaxDD': oos_dd,
    }


def sharpe_period(nav, dates, start, end):
    """Sharpe for a specific date range."""
    mask = (dates >= start) & (dates < end)
    if mask.sum() < 100:
        return np.nan
    idx = dates[mask].index
    n = pd.Series(nav).iloc[idx[0]:idx[-1] + 1]
    n = n / n.iloc[0]
    r = n.pct_change().fillna(0)
    if r.std() == 0:
        return 0
    return (r.mean() * 252) / (r.std() * np.sqrt(252))

WF_WINDOWS = [
    ('2010-01-01', '2015-12-31'),
    ('2015-01-01', '2020-12-31'),
    ('2020-01-01', '2026-12-31'),
]


# =============================================================================
# Baseline: Current Dyn 2x3x (no bond timing)
# =============================================================================
def build_baseline(ctx):
    """Build current Dyn 2x3x (B0.55/L0.25/V0.1/G0.5)."""
    n = ctx['n']
    wn, wg, wb = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        lv = ctx['raw_lev'][i]; vzv = max(ctx['vix_z'][i], 0)
        w = np.clip(0.55 + 0.25 * lv - 0.10 * vzv, 0.30, 0.90)
        wn[i] = w; wg[i] = (1 - w) * 0.50; wb[i] = (1 - w) * 0.50
    nav = build_portfolio(ctx['nav_a2'], ctx['gold_2x'], ctx['bond_3x'], wn, wg, wb)
    return nav


# =============================================================================
# Enhanced: With Bond timing signals
# =============================================================================
def build_enhanced(ctx, use_corr=True, use_macd=True, use_volz=True,
                   corr_coeff=0.10, macd_coeff=5.0, volz_coeff=0.05,
                   max_tilt=0.15):
    """Build enhanced Dyn 2x3x with Bond timing signals.

    bond_tilt adjusts the Gold/Bond split within the non-NASDAQ portion.
    Positive tilt = more Bond, less Gold.
    """
    n = ctx['n']
    wn, wg, wb = np.zeros(n), np.zeros(n), np.zeros(n)

    for i in range(n):
        lv = ctx['raw_lev'][i]; vzv = max(ctx['vix_z'][i], 0)
        w = np.clip(0.55 + 0.25 * lv - 0.10 * vzv, 0.30, 0.90)
        wn[i] = w

        # Bond tilt (positive = favor bond over gold)
        tilt = 0.0
        if use_corr:
            # When NQ-Bond corr is high (positive), bonds are NOT good hedge → reduce
            # Delayed by DELAY days
            ci = max(0, i - DELAY)
            tilt -= corr_coeff * ctx['nq_bond_corr'][ci]
        if use_macd:
            ci = max(0, i - DELAY)
            tilt += macd_coeff * ctx['bond_macd'][ci]
        if use_volz:
            ci = max(0, i - DELAY)
            tilt -= volz_coeff * ctx['bond_volz'][ci]  # low vol = positive tilt

        tilt = np.clip(tilt, -max_tilt, max_tilt)

        # Apply tilt: base is 0.50/0.50, tilt shifts toward bond
        bond_share = np.clip(0.50 + tilt, 0.20, 0.80)
        gold_share = 1.0 - bond_share
        wb[i] = (1 - w) * bond_share
        wg[i] = (1 - w) * gold_share

    nav = build_portfolio(ctx['nav_a2'], ctx['gold_2x'], ctx['bond_3x'], wn, wg, wb)
    return nav, wn, wg, wb


# =============================================================================
# Part A: Baseline metrics
# =============================================================================
def run_partA(ctx):
    print("=" * 80)
    print("Part A: Baseline Dyn 2x3x (B0.55/L0.25/V0.1/G0.5)")
    print("=" * 80)
    nav = build_baseline(ctx)
    m = calc_metrics(nav, ctx['dates'], 'Baseline Dyn 2x3x')
    print_metrics(m)

    # WF Sharpe
    wfs = [sharpe_period(nav, ctx['dates'], s, e) for s, e in WF_WINDOWS]
    m['WF_avg'] = np.nanmean(wfs)
    m['WF1'] = wfs[0]; m['WF2'] = wfs[1]; m['WF3'] = wfs[2]
    print(f"  WF: {wfs[0]:.3f} / {wfs[1]:.3f} / {wfs[2]:.3f} → Avg {m['WF_avg']:.4f}")

    pd.DataFrame([m]).to_csv(os.path.join(BASE_DIR, 'step2_baseline.csv'), index=False)
    print("Saved: step2_baseline.csv")
    return m, nav


# =============================================================================
# Part B: MA5 (NQ-Bond Correlation) only
# =============================================================================
def run_partB(ctx):
    print("\n" + "=" * 80)
    print("Part B: MA5 (NQ-Bond Correlation) only — sensitivity analysis")
    print("=" * 80)

    results = []
    for cc in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        nav, wn, wg, wb = build_enhanced(ctx, use_corr=True, use_macd=False,
                                          use_volz=False, corr_coeff=cc)
        m = calc_metrics(nav, ctx['dates'], f'MA5_cc{cc}')
        wfs = [sharpe_period(nav, ctx['dates'], s, e) for s, e in WF_WINDOWS]
        m['WF_avg'] = np.nanmean(wfs)
        m['WF1'] = wfs[0]; m['WF2'] = wfs[1]; m['WF3'] = wfs[2]
        m['corr_coeff'] = cc
        results.append(m)
        print(f"  cc={cc:.2f}: CAGR {m['CAGR']*100:.2f}% Sharpe {m['Sharpe']:.4f} "
              f"MaxDD {m['MaxDD']*100:.1f}% WF_avg {m['WF_avg']:.4f}")

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(BASE_DIR, 'step2_partB_ma5.csv'), index=False)
    print("Saved: step2_partB_ma5.csv")
    return rdf


# =============================================================================
# Part C: MA5 + MACD
# =============================================================================
def run_partC(ctx):
    print("\n" + "=" * 80)
    print("Part C: MA5 + MACD — sensitivity analysis")
    print("=" * 80)

    results = []
    for cc in [0.08, 0.10, 0.12]:
        for mc in [3.0, 5.0, 7.0]:
            nav, wn, wg, wb = build_enhanced(ctx, use_corr=True, use_macd=True,
                                              use_volz=False, corr_coeff=cc, macd_coeff=mc)
            m = calc_metrics(nav, ctx['dates'], f'MA5+MACD_cc{cc}_mc{mc}')
            wfs = [sharpe_period(nav, ctx['dates'], s, e) for s, e in WF_WINDOWS]
            m['WF_avg'] = np.nanmean(wfs)
            m['WF1'] = wfs[0]; m['WF2'] = wfs[1]; m['WF3'] = wfs[2]
            m['corr_coeff'] = cc; m['macd_coeff'] = mc
            results.append(m)
            print(f"  cc={cc} mc={mc}: CAGR {m['CAGR']*100:.2f}% Sharpe {m['Sharpe']:.4f} "
                  f"MaxDD {m['MaxDD']*100:.1f}% WF_avg {m['WF_avg']:.4f}")

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(BASE_DIR, 'step2_partC_ma5_macd.csv'), index=False)
    print("Saved: step2_partC_ma5_macd.csv")
    return rdf


# =============================================================================
# Part D: MA5 + MACD + VolZ
# =============================================================================
def run_partD(ctx):
    print("\n" + "=" * 80)
    print("Part D: MA5 + MACD + VolZ — sensitivity analysis")
    print("=" * 80)

    results = []
    for cc in [0.08, 0.10, 0.12]:
        for mc in [3.0, 5.0, 7.0]:
            for vc in [0.03, 0.05, 0.07]:
                nav, wn, wg, wb = build_enhanced(ctx, use_corr=True, use_macd=True,
                                                  use_volz=True, corr_coeff=cc,
                                                  macd_coeff=mc, volz_coeff=vc)
                m = calc_metrics(nav, ctx['dates'], f'Full_cc{cc}_mc{mc}_vc{vc}')
                wfs = [sharpe_period(nav, ctx['dates'], s, e) for s, e in WF_WINDOWS]
                m['WF_avg'] = np.nanmean(wfs)
                m['WF1'] = wfs[0]; m['WF2'] = wfs[1]; m['WF3'] = wfs[2]
                m['corr_coeff'] = cc; m['macd_coeff'] = mc; m['volz_coeff'] = vc
                results.append(m)

    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values('WF_avg', ascending=False)
    rdf.to_csv(os.path.join(BASE_DIR, 'step2_partD_full.csv'), index=False)
    print(f"Saved: step2_partD_full.csv ({len(rdf)} patterns)")
    print("\nTop 10 by WF_avg:")
    for _, r in rdf.head(10).iterrows():
        print(f"  {r['Strategy']}: CAGR {r['CAGR']*100:.2f}% Sharpe {r['Sharpe']:.4f} "
              f"MaxDD {r['MaxDD']*100:.1f}% WF {r['WF_avg']:.4f} OOS_Sh {r['OOS_Sharpe']:.4f}")
    return rdf


# =============================================================================
# Part E: Walk-Forward validation of best candidates
# =============================================================================
def run_partE(ctx):
    print("\n" + "=" * 80)
    print("Part E: Walk-Forward validation — best vs baseline")
    print("=" * 80)

    # Load best from each part
    candidates = []

    # Baseline
    nav_base = build_baseline(ctx)
    candidates.append(('Baseline (G0.5 fixed)', nav_base, {}))

    # Best from Part B
    try:
        b = pd.read_csv(os.path.join(BASE_DIR, 'step2_partB_ma5.csv'))
        best_b = b.sort_values('WF_avg', ascending=False).iloc[0]
        nav_b, _, _, _ = build_enhanced(ctx, use_corr=True, use_macd=False,
                                         use_volz=False, corr_coeff=best_b['corr_coeff'])
        candidates.append((f"MA5 only (cc={best_b['corr_coeff']})", nav_b,
                           {'corr_coeff': best_b['corr_coeff']}))
    except:
        pass

    # Best from Part C
    try:
        c = pd.read_csv(os.path.join(BASE_DIR, 'step2_partC_ma5_macd.csv'))
        best_c = c.sort_values('WF_avg', ascending=False).iloc[0]
        nav_c, _, _, _ = build_enhanced(ctx, use_corr=True, use_macd=True,
                                         use_volz=False, corr_coeff=best_c['corr_coeff'],
                                         macd_coeff=best_c['macd_coeff'])
        candidates.append((f"MA5+MACD (cc={best_c['corr_coeff']},mc={best_c['macd_coeff']})",
                           nav_c, {'corr_coeff': best_c['corr_coeff'],
                                    'macd_coeff': best_c['macd_coeff']}))
    except:
        pass

    # Best from Part D
    try:
        d = pd.read_csv(os.path.join(BASE_DIR, 'step2_partD_full.csv'))
        best_d = d.sort_values('WF_avg', ascending=False).iloc[0]
        nav_d, _, _, _ = build_enhanced(ctx, use_corr=True, use_macd=True,
                                         use_volz=True, corr_coeff=best_d['corr_coeff'],
                                         macd_coeff=best_d['macd_coeff'],
                                         volz_coeff=best_d['volz_coeff'])
        candidates.append((f"Full (cc={best_d['corr_coeff']},mc={best_d['macd_coeff']},vc={best_d['volz_coeff']})",
                           nav_d, {'corr_coeff': best_d['corr_coeff'],
                                    'macd_coeff': best_d['macd_coeff'],
                                    'volz_coeff': best_d['volz_coeff']}))
    except:
        pass

    # Extended WF (7 windows)
    wf7 = [
        ('1990-01-01', '1995-12-31'),
        ('1995-01-01', '2000-12-31'),
        ('2000-01-01', '2005-12-31'),
        ('2005-01-01', '2010-12-31'),
        ('2010-01-01', '2015-12-31'),
        ('2015-01-01', '2020-12-31'),
        ('2020-01-01', '2026-12-31'),
    ]

    results = []
    print(f"\n{'Strategy':<50} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} " +
          " ".join(f"WF{i+1:d}" for i in range(7)) + f" {'WF_avg':>7}")
    print("-" * 120)

    for name, nav, params in candidates:
        m = calc_metrics(nav, ctx['dates'], name)
        wfs = [sharpe_period(nav, ctx['dates'], s, e) for s, e in wf7]
        m['WF7_avg'] = np.nanmean(wfs)
        for i, wf in enumerate(wfs):
            m[f'WF7_{i+1}'] = wf
        m.update(params)
        results.append(m)

        wf_str = " ".join(f"{w:>5.3f}" for w in wfs)
        print(f"  {name:<48} {m['Sharpe']:>7.4f} {m['CAGR']*100:>6.2f}% "
              f"{m['MaxDD']*100:>7.1f}% {wf_str} {m['WF7_avg']:>7.4f}")

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(BASE_DIR, 'step2_partE_wf.csv'), index=False)
    print(f"\nSaved: step2_partE_wf.csv")
    return rdf


# =============================================================================
# Part F: Summary comparison
# =============================================================================
def run_partF(ctx):
    print("\n" + "=" * 80)
    print("Part F: Final Summary — Baseline vs Best Enhanced")
    print("=" * 80)

    try:
        wf = pd.read_csv(os.path.join(BASE_DIR, 'step2_partE_wf.csv'))
    except:
        print("ERROR: Run Part E first")
        return

    print(f"\n{'Strategy':<50} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'W5Y':>7} "
          f"{'W10Y':>7} {'OOS_CAGR':>8} {'OOS_Sh':>7} {'WF7':>7}")
    print("-" * 120)

    for _, r in wf.iterrows():
        w5 = f"{r['Worst5Y']*100:+.1f}%" if not pd.isna(r.get('Worst5Y')) else 'N/A'
        w10 = f"{r['Worst10Y']*100:+.1f}%" if not pd.isna(r.get('Worst10Y')) else 'N/A'
        print(f"  {r['Strategy']:<48} {r['CAGR']*100:>6.2f}% {r['Sharpe']:>7.4f} "
              f"{r['MaxDD']*100:>7.1f}% {w5:>7} {w10:>7} "
              f"{r['OOS_CAGR']*100:>7.2f}% {r['OOS_Sharpe']:>7.4f} {r['WF7_avg']:>7.4f}")

    # Improvement assessment
    base = wf.iloc[0]  # baseline
    best = wf.sort_values('WF7_avg', ascending=False).iloc[0]

    print(f"\n--- Assessment ---")
    print(f"Baseline WF7_avg: {base['WF7_avg']:.4f}")
    print(f"Best     WF7_avg: {best['WF7_avg']:.4f} ({best['Strategy']})")
    delta_wf = best['WF7_avg'] - base['WF7_avg']
    delta_sh = best['Sharpe'] - base['Sharpe']
    delta_dd = best['MaxDD'] - base['MaxDD']
    print(f"WF7 improvement:  {delta_wf:+.4f}")
    print(f"Sharpe improvement: {delta_sh:+.4f}")
    print(f"MaxDD change:     {delta_dd*100:+.1f}%")

    if delta_wf > 0.01 and delta_dd >= -0.03:
        print(f"\n✓ RECOMMENDATION: Bond timing improves WF by {delta_wf:.4f} "
              f"without significant MaxDD degradation. Worth adopting.")
    elif delta_wf > 0:
        print(f"\n△ MARGINAL: Small WF improvement ({delta_wf:.4f}). "
              f"May not justify added complexity.")
    else:
        print(f"\n✗ NO IMPROVEMENT: Bond timing does not improve Walk-Forward performance. "
              f"Keep current strategy.")

    out = os.path.join(BASE_DIR, 'step2_summary.csv')
    wf.to_csv(out, index=False)
    print(f"\nSaved: {out}")


def print_metrics(m):
    w5 = f"{m['Worst5Y']*100:+.1f}%" if not pd.isna(m.get('Worst5Y')) else 'N/A'
    w10 = f"{m['Worst10Y']*100:+.1f}%" if not pd.isna(m.get('Worst10Y')) else 'N/A'
    print(f"  CAGR: {m['CAGR']*100:.2f}%, Sharpe: {m['Sharpe']:.4f}, "
          f"MaxDD: {m['MaxDD']*100:.1f}%, W5Y: {w5}, W10Y: {w10}")
    print(f"  OOS CAGR: {m['OOS_CAGR']*100:.2f}%, OOS Sharpe: {m['OOS_Sharpe']:.4f}, "
          f"OOS MaxDD: {m['OOS_MaxDD']*100:.1f}%")


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'

    print("Loading data and building signals...")
    ctx = load_and_build()
    print(f"Data ready: {ctx['n']} rows\n")

    if part in ['A', 'partA', 'all']:
        run_partA(ctx)
    if part in ['B', 'partB', 'all']:
        run_partB(ctx)
    if part in ['C', 'partC', 'all']:
        run_partC(ctx)
    if part in ['D', 'partD', 'all']:
        run_partD(ctx)
    if part in ['E', 'partE', 'all']:
        run_partE(ctx)
    if part in ['F', 'partF', 'all']:
        run_partF(ctx)

    print("\n=== DONE ===")
