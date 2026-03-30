"""
Yearly Returns for 7 strategies (1974-2026)
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np
from backtest_engine import load_data, calc_dd_signal
from test_ens2_strategies import strategy_ens2_asym_slope, calc_asym_ewma_vol as aev_orig
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data
from test_dynamic_portfolio import build_dynamic_portfolio, build_static_portfolio

ANNUAL_COST = 0.0086; DELAY = 2; BASE_LEV = 3.0; THRESHOLD = 0.20
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'NASDAQ_extended_to_2026.csv')

def run_bt(close, leverage, delay=DELAY, base_lev=BASE_LEV, cost=ANNUAL_COST):
    returns = close.pct_change()
    lr = returns * base_lev; dc = cost / 252
    dl = leverage.shift(delay)
    sr = dl * (lr - dc); sr = sr.fillna(0)
    return (1 + sr).cumprod(), sr

def calc_asym_ewma(returns, su=30, sd=10):
    var = pd.Series(index=returns.index, dtype=float)
    var.iloc[0] = returns.iloc[:20].var() if len(returns)>20 else 0.0001
    for i in range(1, len(returns)):
        r = returns.iloc[i]; a = 2/(sd+1) if r<0 else 2/(su+1)
        var.iloc[i] = (1-a)*var.iloc[i-1]+a*(r**2)
    return np.sqrt(var * 252)

def yearly_returns(nav, dates):
    """Calculate yearly returns from NAV series."""
    df = pd.DataFrame({'nav': nav.values if hasattr(nav, 'values') else nav, 'date': dates.values})
    df['year'] = pd.to_datetime(df['date']).dt.year
    yearly_nav = df.groupby('year')['nav'].agg(['first', 'last'])
    yearly_nav['return'] = (yearly_nav['last'] / yearly_nav['first'] - 1) * 100
    return yearly_nav['return']

def main():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()}\n")

    # Build all NAVs
    navs = {}

    # 1. BH 1x
    print("[1/7] BH 1x")
    navs['BH 1x'] = close / close.iloc[0]

    # 2. BH 3x
    print("[2/7] BH 3x")
    lev = pd.Series(1.0, index=close.index)
    navs['BH 3x'], _ = run_bt(close, lev, delay=0)

    # 3. DD Only
    print("[3/7] DD Only")
    dd3 = calc_dd_signal(close, 0.82, 0.92)
    lev3 = rebalance_threshold(dd3, THRESHOLD)
    navs['DD Only'], _ = run_bt(close, lev3)

    # 4. Ens2(Asym+Slope)
    print("[4/7] Ens2(Asym+Slope)")
    lev4, dd4 = strategy_ens2_asym_slope(close, returns, 0.82, 0.92, 0.25, 20, 5, 1.0)
    lev4 = rebalance_threshold(lev4, THRESHOLD)
    navs['Ens2(Asym+Slope)'], _ = run_bt(close, lev4)

    # 5. A2 Optimized
    print("[5/7] A2 Optimized")
    dd = calc_dd_signal(close, 0.82, 0.92)
    av = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean(); ratio = close/ma150
    ttv = (0.10+(0.30-0.10)*(ratio-0.85)/(1.15-0.85)).clip(0.10,0.30).fillna(0.20)
    vt = (ttv/av).clip(0,1.0)
    ma200 = close.rolling(200).mean(); sl = ma200.pct_change()
    sm = sl.rolling(60).mean(); ss = sl.rolling(60).std().replace(0,0.0001)
    z = (sl-sm)/ss; slope = (0.9+0.35*z).clip(0.3,1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp = calc_vix_proxy(returns)
    vma = vp.rolling(252).mean(); vs = vp.rolling(252).std().replace(0,0.001)
    vz = (vp-vma)/vs; vm = (1.0-0.25*vz).clip(0.5,1.15)
    raw = dd*vt*slope*mom*vm; raw = raw.clip(0,1.0).fillna(0)
    lev5 = rebalance_threshold(raw, THRESHOLD)
    nav5, ret5 = run_bt(close, lev5)
    navs['A2 Optimized'] = nav5

    # 6+7. Dyn-Hybrid (need Gold/Bond)
    print("[6-7/7] Fetching Gold/Bond for Dyn-Hybrid...")
    gold = prepare_gold_data(dates); bond = prepare_bond_data(dates)
    signals = {'nav': nav5.values, 'raw_leverage': raw.values,
               'dd_signal': dd.values, 'vix_z': vz.fillna(0).values}

    # 6. DH Static (35/30/35)
    snav = build_static_portfolio(signals['nav'], gold, bond, 0.35, 0.30, 0.35)
    navs['DH Static (35/30/35)'] = pd.Series(snav, index=dates.index)

    # 7. DH Dynamic CAGR25+ (B0.5/L0.25/V0.1)
    n = len(signals['raw_leverage'])
    wn, wg, wb = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        lv = signals['raw_leverage'][i]; vzv = max(signals['vix_z'][i], 0)
        w = np.clip(0.50 + 0.25*lv - 0.10*vzv, 0.30, 0.90)
        wn[i] = w; wg[i] = (1-w)*0.55; wb[i] = (1-w)*0.45
    dnav = build_dynamic_portfolio(signals['nav'], gold, bond, wn, wg, wb)
    navs['DH Dynamic CAGR25+'] = pd.Series(dnav, index=dates.index)

    print("\nCalculating yearly returns...")

    # Calculate yearly returns for all
    order = ['DH Static (35/30/35)', 'DH Dynamic CAGR25+', 'A2 Optimized',
             'Ens2(Asym+Slope)', 'DD Only', 'BH 3x', 'BH 1x']

    yr_df = pd.DataFrame()
    for name in order:
        yr = yearly_returns(navs[name], dates)
        yr_df[name] = yr

    # Save CSV
    out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'yearly_returns_7strategies.csv')
    yr_df.to_csv(out_csv)
    print(f"Saved CSV: {out_csv}")

    # Print table
    print(f"\n{'Year':<6}", end="")
    for name in order:
        short = name[:12]
        print(f"{short:>14}", end="")
    print()
    print("-" * (6 + 14 * len(order)))

    for year in yr_df.index:
        print(f"{year:<6}", end="")
        for name in order:
            v = yr_df.loc[year, name]
            print(f"{v:>+13.1f}%", end="")
        print()

    # Summary stats
    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}")
    print(f"{'Stat':<12}", end="")
    for name in order:
        short = name[:12]
        print(f"{short:>14}", end="")
    print()
    print("-" * (12 + 14 * len(order)))

    for stat_name, func in [('Mean', lambda s: s.mean()),
                             ('Median', lambda s: s.median()),
                             ('Max', lambda s: s.max()),
                             ('Min', lambda s: s.min()),
                             ('StdDev', lambda s: s.std()),
                             ('+Years', lambda s: (s > 0).sum()),
                             ('-Years', lambda s: (s <= 0).sum())]:
        print(f"{stat_name:<12}", end="")
        for name in order:
            v = func(yr_df[name])
            if stat_name in ['+Years', '-Years']:
                print(f"{int(v):>14}", end="")
            else:
                print(f"{v:>+13.1f}%", end="")
        print()

    # Generate Markdown
    generate_md(yr_df, order, dates)

def generate_md(yr_df, order, dates):
    short_names = {
        'DH Static (35/30/35)': 'DH Static',
        'DH Dynamic CAGR25+': 'DH Dyn CAGR25+',
        'A2 Optimized': 'A2 Opt',
        'Ens2(Asym+Slope)': 'Ens2',
        'DD Only': 'DD Only',
        'BH 3x': 'BH 3x',
        'BH 1x': 'BH 1x',
    }

    md = f"""# 7戦略 年次リターン（1974-2026）

## 検証条件

| 項目 | 値 |
|------|-----|
| データ期間 | {dates.iloc[0].date()} 〜 {dates.iloc[-1].date()} |
| 実行遅延 | 2営業日 |
| 経費率 | 年0.86%（TQQQ準拠） |
| リバランス閾値 | 20% |

> \\* DH Static / DH Dyn CAGR25+ は3資産ポートフォリオ（NASDAQ 3x + Gold + Bond）

---

## 年次リターン表（%）

| Year | DH Static | DH Dyn CAGR25+ | A2 Opt | Ens2 | DD Only | BH 3x | BH 1x |
|------|-----------|----------------|--------|------|---------|-------|-------|
"""

    for year in yr_df.index:
        md += f"| {year} |"
        for name in order:
            v = yr_df.loc[year, name]
            md += f" {v:+.1f}% |"
        md += "\n"

    # Summary
    md += """
---

## 統計サマリー

| 統計量 | DH Static | DH Dyn CAGR25+ | A2 Opt | Ens2 | DD Only | BH 3x | BH 1x |
|--------|-----------|----------------|--------|------|---------|-------|-------|
"""

    for stat_name, func in [('平均', lambda s: s.mean()),
                             ('中央値', lambda s: s.median()),
                             ('最大', lambda s: s.max()),
                             ('最小', lambda s: s.min()),
                             ('標準偏差', lambda s: s.std()),
                             ('プラス年数', lambda s: (s > 0).sum()),
                             ('マイナス年数', lambda s: (s <= 0).sum())]:
        md += f"| {stat_name} |"
        for name in order:
            v = func(yr_df[name])
            if stat_name in ['プラス年数', 'マイナス年数']:
                md += f" {int(v)} |"
            else:
                md += f" {v:+.1f}% |"
        md += "\n"

    md += """
---

*Generated: 2026-03-30*
"""

    md_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'YEARLY_RETURNS_REPORT.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Saved MD: {md_path}")

if __name__ == '__main__':
    main()
