"""
方向性A 検証: リバランス頻度最適化（ボラポンプ効果）
=====================================================
現状: drift > 0.10 or i % 63 == 0  (quarterly forced)
比較: monthly(21), weekly(5), drift-only, hybrid(monthly + tighter drift)

ボラポンプ仮説: TQQQ×TMF の逆相関（-0.3〜-0.5）下でリバランス頻度が上がると
Shannon's Demon効果でリターンが向上する可能性を検証する。
"""
import sys, os, types

# multitasking stub
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest_engine import load_data, calc_dd_signal
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'NASDAQ_extended_to_2026.csv')


# ─── Core backtest (same as step_update_dyn2x3x.py) ─────────────────────────

def run_bt(close, leverage):
    returns = close.pct_change()
    lr = returns * 3.0
    dc = 0.0086 / 252
    dl = leverage.shift(2)
    sr = dl * (lr - dc)
    sr = sr.fillna(0)
    return (1 + sr).cumprod(), sr


def calc_asym_ewma(returns, su=30, sd=10):
    var = pd.Series(index=returns.index, dtype=float)
    var.iloc[0] = returns.iloc[:20].var() if len(returns) > 20 else 0.0001
    for i in range(1, len(returns)):
        r = returns.iloc[i]
        a = 2 / (sd + 1) if r < 0 else 2 / (su + 1)
        var.iloc[i] = (1 - a) * var.iloc[i - 1] + a * (r ** 2)
    return np.sqrt(var * 252)


# ─── Portfolio builder with configurable rebalancing ─────────────────────────

def build_portfolio_with_freq(nasdaq_nav, gold_prices, bond_nav,
                               w_nasdaq_daily, w_gold_daily, w_bond_daily,
                               drift_threshold=0.10, calendar_freq=63):
    """
    portfolio builder parameterized by:
      drift_threshold: trigger rebalance when sum |actual - target| exceeds this
      calendar_freq  : force rebalance every N days (0 = disabled)
    """
    n = len(nasdaq_nav)
    nasdaq_ret = np.zeros(n)
    gold_ret = np.zeros(n)
    bond_ret = np.zeros(n)

    for i in range(1, n):
        nasdaq_ret[i] = nasdaq_nav[i] / nasdaq_nav[i-1] - 1 if nasdaq_nav[i-1] > 0 else 0
        gold_ret[i]   = gold_prices[i] / gold_prices[i-1] - 1 if gold_prices[i-1] > 0 else 0
        bond_ret[i]   = bond_nav[i] / bond_nav[i-1] - 1 if bond_nav[i-1] > 0 else 0

    portfolio_nav = np.ones(n)
    cw_n = w_nasdaq_daily[0]
    cw_g = w_gold_daily[0]
    cw_b = w_bond_daily[0]

    rebalance_count = 0

    for i in range(1, n):
        port_ret = cw_n * nasdaq_ret[i] + cw_g * gold_ret[i] + cw_b * bond_ret[i]
        portfolio_nav[i] = portfolio_nav[i-1] * (1 + port_ret)

        total = (cw_n * (1 + nasdaq_ret[i]) +
                 cw_g * (1 + gold_ret[i]) +
                 cw_b * (1 + bond_ret[i]))
        if total > 0:
            cw_n = cw_n * (1 + nasdaq_ret[i]) / total
            cw_g = cw_g * (1 + gold_ret[i]) / total
            cw_b = cw_b * (1 + bond_ret[i]) / total

        tw_n = w_nasdaq_daily[i]
        tw_g = w_gold_daily[i]
        tw_b = w_bond_daily[i]
        drift = abs(cw_n - tw_n) + abs(cw_g - tw_g) + abs(cw_b - tw_b)

        calendar_trigger = (calendar_freq > 0 and i % calendar_freq == 0)
        if drift > drift_threshold or calendar_trigger:
            cw_n, cw_g, cw_b = tw_n, tw_g, tw_b
            rebalance_count += 1

    return portfolio_nav, rebalance_count


# ─── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(nav, dates):
    """Calculate CAGR, MaxDD, Sharpe, Worst5Y from NAV array."""
    nav_s = pd.Series(nav, index=pd.to_datetime(dates.values))
    ret = nav_s.pct_change().dropna()

    years = len(nav_s) / 252
    cagr = (nav_s.iloc[-1] ** (1 / years)) - 1

    roll_max = nav_s.cummax()
    dd = (nav_s - roll_max) / roll_max
    max_dd = dd.min()

    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0

    # Worst 5Y rolling CAGR (252*5 window)
    worst5y = float('inf')
    w = 252 * 5
    for j in range(w, len(nav)):
        r = (nav[j] / nav[j - w]) ** (1 / 5) - 1
        if r < worst5y:
            worst5y = r
    if worst5y == float('inf'):
        worst5y = float('nan')

    # OOS Sharpe (2021+)
    oos_ret = ret[ret.index >= '2021-01-01']
    oos_sharpe = oos_ret.mean() / oos_ret.std() * np.sqrt(252) if len(oos_ret) > 10 and oos_ret.std() > 0 else float('nan')

    # Yearly returns
    nav_df = pd.DataFrame({'nav': nav, 'date': dates.values})
    nav_df['year'] = pd.to_datetime(nav_df['date']).dt.year
    year_end = nav_df.groupby('year')['nav'].last()
    yearly = year_end.pct_change()
    yearly.iloc[0] = (year_end.iloc[0] / nav[0]) - 1

    return {
        'cagr': cagr,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'worst5y': worst5y,
        'oos_sharpe': oos_sharpe,
        'yearly': yearly,
    }


def main():
    print("=" * 80)
    print("方向性A 検証: リバランス頻度最適化（ボラポンプ効果）")
    print("=" * 80)

    # ── Load data & build signals (identical to step_update_dyn2x3x.py) ──────
    df = load_data(DATA_PATH)
    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} → {dates.iloc[-1].date()} ({len(df)} rows)")

    # A2 optimized signals
    dd = calc_dd_signal(close, 0.82, 0.92)
    av = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean()
    ratio = close / ma150
    ttv = (0.10 + (0.20) * (ratio - 0.85) / 0.30).clip(0.10, 0.30).fillna(0.20)
    vt = (ttv / av).clip(0, 1.0)
    ma200 = close.rolling(200).mean()
    sl = ma200.pct_change()
    sm = sl.rolling(60).mean()
    ss = sl.rolling(60).std().replace(0, 0.0001)
    z = (sl - sm) / ss
    slope = (0.9 + 0.35 * z).clip(0.3, 1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp = calc_vix_proxy(returns)
    vma = vp.rolling(252).mean()
    vs = vp.rolling(252).std().replace(0, 0.001)
    vz = (vp - vma) / vs
    vm = (1.0 - 0.25 * vz).clip(0.5, 1.15)
    raw = dd * vt * slope * mom * vm
    raw = raw.clip(0, 1.0).fillna(0)
    lev = rebalance_threshold(raw, 0.20)
    nav_a2, _ = run_bt(close, lev)

    # Gold/Bond leveraged
    gold_1x = prepare_gold_data(dates)
    bond_1x = prepare_bond_data(dates)
    n = len(dates)
    g2 = np.ones(n)
    b3 = np.ones(n)
    for i in range(1, n):
        gr = gold_1x[i] / gold_1x[i-1] - 1 if gold_1x[i-1] > 0 else 0
        g2[i] = g2[i-1] * (1 + gr * 2 - 0.005 / 252)
        br = bond_1x[i] / bond_1x[i-1] - 1 if bond_1x[i-1] > 0 else 0
        b3[i] = b3[i-1] * (1 + br * 3 - 0.0091 / 252)

    # Dyn 2x3x weights
    signals_raw = raw.values
    signals_vz = vz.fillna(0).values
    wn = np.zeros(n)
    wg = np.zeros(n)
    wb = np.zeros(n)
    for i in range(n):
        lv = signals_raw[i]
        vzv = max(signals_vz[i], 0)
        w = np.clip(0.55 + 0.25 * lv - 0.10 * vzv, 0.30, 0.90)
        wn[i] = w
        wg[i] = (1 - w) * 0.50
        wb[i] = (1 - w) * 0.50

    nasdaq_nav = nav_a2.values

    # ── Test configurations ───────────────────────────────────────────────────
    configs = [
        # (label, drift_threshold, calendar_freq)
        ("現状 Quarterly (drift>10%|63d)",  0.10, 63),   # baseline
        ("Monthly     (drift>10%|21d)",     0.10, 21),
        ("Bi-monthly  (drift>10%|42d)",     0.10, 42),
        ("Weekly      (drift>10%| 5d)",     0.10,  5),
        ("Daily       (drift>10%| 1d)",     0.10,  1),   # = always rebalance
        ("Threshold-only (drift>10%)",      0.10,  0),   # no calendar
        ("Tight+Monthly (drift>5% |21d)",   0.05, 21),
        ("Tight+Quarterly(drift>5% |63d)",  0.05, 63),
    ]

    print(f"\n{'─'*90}")
    print(f"{'設定':<35} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>8} {'Worst5Y':>9} {'OOS_Sh':>8} {'Rebals':>7}")
    print(f"{'─'*90}")

    results = []
    for label, drift_thr, cal_freq in configs:
        nav_arr, rebal_cnt = build_portfolio_with_freq(
            nasdaq_nav, g2, b3, wn, wg, wb,
            drift_threshold=drift_thr,
            calendar_freq=cal_freq
        )
        m = calc_metrics(nav_arr, dates)
        results.append({
            'label': label,
            'drift': drift_thr,
            'freq': cal_freq,
            **m,
            'rebalances': rebal_cnt,
        })

        flag = ""
        # flag improvements vs baseline (index 0)
        if results and len(results) > 1:
            base = results[0]
            if m['cagr'] > base['cagr'] and m['sharpe'] > base['sharpe']:
                flag = " ◀ BETTER"

        print(f"{label:<35} {m['cagr']*100:>6.2f}% {m['max_dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.4f} {m['worst5y']*100:>8.2f}% "
              f"{m['oos_sharpe']:>8.4f} {rebal_cnt:>7d}{flag}")

    baseline = results[0]

    # ── Crisis year analysis ──────────────────────────────────────────────────
    crisis_years = [2000, 2001, 2002, 2008, 2009, 2020, 2022, 2023, 2024, 2025]
    print(f"\n{'─'*90}")
    print("危機年・主要年 パフォーマンス (%)")
    print(f"{'設定':<35}", end="")
    for y in crisis_years:
        print(f" {y:>6}", end="")
    print()
    print(f"{'─'*90}")

    for cfg, r in zip(configs, results):
        label = cfg[0]
        drift_thr = cfg[1]
        cal_freq = cfg[2]
        nav_arr, _ = build_portfolio_with_freq(
            nasdaq_nav, g2, b3, wn, wg, wb,
            drift_threshold=drift_thr, calendar_freq=cal_freq
        )
        nav_df = pd.DataFrame({'nav': nav_arr, 'date': dates.values})
        nav_df['year'] = pd.to_datetime(nav_df['date']).dt.year
        year_end = nav_df.groupby('year')['nav'].last()
        yearly_pct = year_end.pct_change() * 100

        print(f"{label:<35}", end="")
        for y in crisis_years:
            if y in yearly_pct.index and not pd.isna(yearly_pct[y]):
                print(f" {yearly_pct[y]:>6.1f}", end="")
            else:
                print(f" {'N/A':>6}", end="")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("分析サマリー")
    print(f"{'='*80}")

    sorted_by_sharpe = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    best = sorted_by_sharpe[0]
    print(f"\n[Best by Sharpe]  {best['label']}")
    print(f"  CAGR: {best['cagr']*100:.2f}%  Sharpe: {best['sharpe']:.4f}  MaxDD: {best['max_dd']*100:.2f}%  Worst5Y: {best['worst5y']*100:.2f}%")

    sorted_by_cagr = sorted(results, key=lambda x: x['cagr'], reverse=True)
    best_cagr = sorted_by_cagr[0]
    print(f"\n[Best by CAGR]    {best_cagr['label']}")
    print(f"  CAGR: {best_cagr['cagr']*100:.2f}%  Sharpe: {best_cagr['sharpe']:.4f}  MaxDD: {best_cagr['max_dd']*100:.2f}%  Worst5Y: {best_cagr['worst5y']*100:.2f}%")

    print(f"\n[Baseline]  {baseline['label']}")
    print(f"  CAGR: {baseline['cagr']*100:.2f}%  Sharpe: {baseline['sharpe']:.4f}  MaxDD: {baseline['max_dd']*100:.2f}%  Worst5Y: {baseline['worst5y']*100:.2f}%")

    print(f"\n[CAGR差分 vs 現状 Quarterly]")
    for r in results:
        delta_cagr = (r['cagr'] - baseline['cagr']) * 100
        delta_sharpe = r['sharpe'] - baseline['sharpe']
        sign = "▲" if delta_cagr > 0 else "▼"
        print(f"  {r['label']:<35} CAGR {sign}{abs(delta_cagr):.2f}%  Sharpe {delta_sharpe:+.4f}")

    # ── Volatility pumping analysis ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print("ボラポンプ効果 分析")
    print(f"{'='*80}")
    print("\nNote: 3資産の日次リターン相関（TQQQ×TMF が逆相関のため、")
    print("      リバランス頻度増加 → Shannon's Demon効果で超過リターン期待）")

    # Compute TQQQ vs TMF correlation as proxy
    nasdaq_ret = pd.Series(nasdaq_nav).pct_change().dropna()
    bond_ret = pd.Series(b3).pct_change().dropna()
    gold_ret = pd.Series(g2).pct_change().dropna()

    min_len = min(len(nasdaq_ret), len(bond_ret), len(gold_ret))
    corr_nb = np.corrcoef(nasdaq_ret.values[-min_len:], bond_ret.values[-min_len:])[0, 1]
    corr_ng = np.corrcoef(nasdaq_ret.values[-min_len:], gold_ret.values[-min_len:])[0, 1]
    corr_gb = np.corrcoef(gold_ret.values[-min_len:], bond_ret.values[-min_len:])[0, 1]

    print(f"\n  相関係数 (全期間):")
    print(f"    TQQQ × TMF  (bond):  {corr_nb:.3f}")
    print(f"    TQQQ × Gold (gold):  {corr_ng:.3f}")
    print(f"    Gold × TMF  (bond):  {corr_gb:.3f}")

    # Vol
    print(f"\n  年率ボラティリティ (全期間):")
    print(f"    TQQQ: {nasdaq_ret.std()*np.sqrt(252)*100:.1f}%")
    print(f"    TMF:  {bond_ret.std()*np.sqrt(252)*100:.1f}%")
    print(f"    Gold: {gold_ret.std()*np.sqrt(252)*100:.1f}%")

    # Save CSV
    out_path = os.path.join(BASE_DIR, 'rebalance_frequency_results.csv')
    rows = []
    for r in results:
        rows.append({
            'label': r['label'],
            'drift_threshold': r['drift'],
            'calendar_freq_days': r['freq'],
            'CAGR_pct': round(r['cagr'] * 100, 3),
            'MaxDD_pct': round(r['max_dd'] * 100, 3),
            'Sharpe': round(r['sharpe'], 4),
            'Worst5Y_pct': round(r['worst5y'] * 100, 3),
            'OOS_Sharpe': round(r['oos_sharpe'], 4),
            'Rebalances': r['rebalances'],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nCSV保存: {out_path}")


if __name__ == '__main__':
    main()
