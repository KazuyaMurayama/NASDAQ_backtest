"""
追加検証: 方向性D + F 複合効果
==========================================
方向性D（DD期T-bill）と方向性F（delay=1）を同時適用した場合の効果を測定。

単純加算予測: +0.72% + 1.40% = +2.12%
実測値との差: 相互作用の有無を確認

ベースライン: Dyn 2x3x G0.5 (delay=2, 0% cash)
  CAGR=31.40% / Sharpe=1.297 / MaxDD=-33.4%
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from backtest_engine import load_data, calc_dd_signal
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'NASDAQ_extended_to_2026.csv')


def calc_asym_ewma(returns, su=30, sd=10):
    var = pd.Series(index=returns.index, dtype=float)
    var.iloc[0] = returns.iloc[:20].var() if len(returns) > 20 else 0.0001
    for i in range(1, len(returns)):
        r = returns.iloc[i]
        a = 2 / (sd + 1) if r < 0 else 2 / (su + 1)
        var.iloc[i] = (1 - a) * var.iloc[i - 1] + a * (r ** 2)
    return np.sqrt(var * 252)


def build_a2_signals(close, returns):
    dd = calc_dd_signal(close, 0.82, 0.92)
    av = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean()
    ratio = close / ma150
    ttv = (0.10 + 0.20 * (ratio - 0.85) / 0.30).clip(0.10, 0.30).fillna(0.20)
    vt = (ttv / av).clip(0, 1.0)
    ma200 = close.rolling(200).mean()
    sl = ma200.pct_change()
    sm = sl.rolling(60).mean(); ss = sl.rolling(60).std().replace(0, 0.0001)
    z = (sl - sm) / ss
    slope = (0.9 + 0.35 * z).clip(0.3, 1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp = calc_vix_proxy(returns)
    vma = vp.rolling(252).mean(); vs = vp.rolling(252).std().replace(0, 0.001)
    vz = (vp - vma) / vs
    vm = (1.0 - 0.25 * vz).clip(0.5, 1.15)
    raw = (dd * vt * slope * mom * vm).clip(0, 1.0).fillna(0)
    return raw, vz


def make_tbill_daily_rate(dates_dt):
    years = np.array([d.year for d in dates_dt])
    annual = np.where(years < 1982, 0.08,
              np.where(years < 2002, 0.05,
                np.where(years < 2010, 0.03,
                  np.where(years < 2022, 0.007, 0.05))))
    return annual / 252


def build_a2_nav(close, returns, raw, delay, tbill_daily=None):
    """
    A2 NASDAQ NAV with delay and optional T-bill during cash periods.
    tbill_daily: None → 0% during cash; array → T-bill rate during cash
    """
    lev = rebalance_threshold(raw, 0.20)
    lr = returns * 3.0
    dc = 0.0086 / 252
    dl = lev.shift(delay)
    dl_arr = dl.fillna(0).values

    sr_base = (dl * (lr - dc)).fillna(0).values

    if tbill_daily is None:
        return (1 + pd.Series(sr_base, index=close.index)).cumprod(), dl_arr
    else:
        cash_mask = (dl_arr == 0)
        sr = np.where(cash_mask, tbill_daily, sr_base)
        nav = np.cumprod(1 + sr)
        return pd.Series(nav, index=close.index), dl_arr


def build_portfolio_3asset(nasdaq_nav, gold_nav, bond_nav, wn, wg, wb):
    n = len(nasdaq_nav)
    nt = np.zeros(n); ng = np.zeros(n); nb = np.zeros(n)
    for i in range(1, n):
        nt[i] = nasdaq_nav[i] / nasdaq_nav[i-1] - 1 if nasdaq_nav[i-1] > 0 else 0
        ng[i] = gold_nav[i]  / gold_nav[i-1]  - 1 if gold_nav[i-1]  > 0 else 0
        nb[i] = bond_nav[i]  / bond_nav[i-1]  - 1 if bond_nav[i-1]  > 0 else 0
    pnav = np.ones(n)
    ct, cg, cb = wn[0], wg[0], wb[0]
    for i in range(1, n):
        port_ret = ct*nt[i] + cg*ng[i] + cb*nb[i]
        pnav[i] = pnav[i-1] * (1 + port_ret)
        total = ct*(1+nt[i]) + cg*(1+ng[i]) + cb*(1+nb[i])
        if total > 0:
            ct = ct*(1+nt[i])/total; cg = cg*(1+ng[i])/total; cb = cb*(1+nb[i])/total
        if abs(ct-wn[i]) + abs(cg-wg[i]) + abs(cb-wb[i]) > 0.10 or i % 63 == 0:
            ct, cg, cb = wn[i], wg[i], wb[i]
    return pnav


def calc_metrics(nav, dates):
    nav_s = pd.Series(nav, index=pd.to_datetime(dates.values))
    ret = nav_s.pct_change().dropna()
    years = len(nav_s) / 252
    cagr = float(nav_s.iloc[-1] ** (1 / years) - 1)
    maxdd = float((nav_s / nav_s.cummax() - 1).min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    worst5y = float(((nav_s / nav_s.shift(252*5)) ** (1/5) - 1).min())
    oos = ret[ret.index >= '2021-01-01']
    oos_sh = float(oos.mean()/oos.std()*np.sqrt(252)) if len(oos) > 10 and oos.std() > 0 else float('nan')
    return {'cagr': cagr, 'maxdd': maxdd, 'sharpe': sharpe, 'worst5y': worst5y, 'oos_sharpe': oos_sh}


def main():
    print("=" * 80)
    print("追加検証: 方向性D + F 複合効果")
    print("=" * 80)

    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    n = len(dates)
    dates_dt = pd.to_datetime(dates.values)
    print(f"データ期間: {dates.iloc[0].date()} → {dates.iloc[-1].date()} ({n} rows)")

    raw, vz = build_a2_signals(close, returns)
    raw_v = raw.values; vz_v = vz.fillna(0).values

    gold_1x = prepare_gold_data(dates)
    bond_1x = prepare_bond_data(dates)
    g2 = np.ones(n); b3 = np.ones(n)
    for i in range(1, n):
        gr = gold_1x[i]/gold_1x[i-1]-1 if gold_1x[i-1] > 0 else 0
        g2[i] = g2[i-1] * (1 + gr*2 - 0.005/252)
        br = bond_1x[i]/bond_1x[i-1]-1 if bond_1x[i-1] > 0 else 0
        b3[i] = b3[i-1] * (1 + br*3 - 0.0091/252)

    wn_arr = np.zeros(n); wg_arr = np.zeros(n); wb_arr = np.zeros(n)
    for i in range(n):
        w = np.clip(0.55 + 0.25*raw_v[i] - 0.10*max(vz_v[i], 0), 0.30, 0.90)
        wn_arr[i] = w; wg_arr[i] = (1-w)*0.50; wb_arr[i] = (1-w)*0.50

    tbill = make_tbill_daily_rate(dates_dt)

    # ── 4シナリオ比較
    scenarios = [
        (2, None,   'ベースライン       (d=2, 0%cash)'),
        (1, None,   '方向性F のみ      (d=1, 0%cash)'),
        (2, tbill,  '方向性D のみ      (d=2, T-bill)'),
        (1, tbill,  '方向性D+F 複合    (d=1, T-bill)'),
    ]

    results = []
    base_ref = None
    for delay, tb, label in scenarios:
        nav_a2, _ = build_a2_nav(close, returns, raw, delay, tb)
        pnav = build_portfolio_3asset(nav_a2.values, g2, b3, wn_arr, wg_arr, wb_arr)
        m = calc_metrics(pnav, dates)
        m['label'] = label
        results.append(m)
        if delay == 2 and tb is None:
            base_ref = m

    # ── 全期間結果
    print(f"\n{'─'*80}")
    print(f"全期間 (1974-2026)")
    print(f"{'─'*80}")
    print(f"  {'シナリオ':<32} {'CAGR':>7} {'ΔCAGR':>7} {'Sharpe':>7} {'ΔSharpe':>8} {'MaxDD':>7} {'W5Y':>6} {'OOSSh':>7}")
    print(f"  {'-'*78}")
    for m in results:
        dc = (m['cagr'] - base_ref['cagr']) * 100
        ds = m['sharpe'] - base_ref['sharpe']
        oos_str = f"{m['oos_sharpe']:.3f}" if not (m['oos_sharpe'] != m['oos_sharpe']) else " N/A"
        dc_s = f"{'+' if dc >= 0 else ''}{dc:.2f}%"
        ds_s = f"{'+' if ds >= 0 else ''}{ds:.4f}"
        print(f"  {m['label']:<32} {m['cagr']*100:>6.2f}% {dc_s:>7} {m['sharpe']:>7.4f} {ds_s:>8} "
              f"{m['maxdd']*100:>6.1f}% {m['worst5y']*100:>5.1f}% {oos_str:>7}")

    # ── 相互作用の分析
    f_only = results[1]['cagr'] - base_ref['cagr']
    d_only = results[2]['cagr'] - base_ref['cagr']
    df_comb = results[3]['cagr'] - base_ref['cagr']
    interact = df_comb - (f_only + d_only)

    print(f"\n{'─'*80}")
    print(f"相互作用の分析")
    print(f"{'─'*80}")
    print(f"  方向性F 単独効果:            {f_only*100:+.2f}%")
    print(f"  方向性D 単独効果:            {d_only*100:+.2f}%")
    print(f"  単純加算予測:                {(f_only+d_only)*100:+.2f}%")
    print(f"  D+F 複合実測:                {df_comb*100:+.2f}%")
    print(f"  相互作用項 (実測-予測):      {interact*100:+.2f}%")
    if abs(interact) < 0.001:
        print(f"  → ほぼ独立（相互作用なし）")
    elif interact > 0:
        print(f"  → 正の相乗効果あり")
    else:
        print(f"  → 負の相互作用（干渉あり）")

    # ── OOS（2021-2026）での複合効果
    print(f"\n{'─'*80}")
    print(f"OOS (2021-2026) 比較")
    print(f"{'─'*80}")
    for m in results:
        oos_str = f"{m['oos_sharpe']:.4f}" if not (m['oos_sharpe'] != m['oos_sharpe']) else "N/A"
        print(f"  {m['label']:<32}: OOS Sharpe = {oos_str}")

    # ── 結論
    comb = results[3]
    print(f"\n{'='*80}")
    print(f"複合検証 最終結論")
    print(f"{'='*80}")
    print(f"  ベースライン: CAGR={base_ref['cagr']*100:.2f}%, Sharpe={base_ref['sharpe']:.4f}")
    print(f"  D+F 複合:     CAGR={comb['cagr']*100:.2f}% ({df_comb*100:+.2f}%), "
          f"Sharpe={comb['sharpe']:.4f} ({comb['sharpe']-base_ref['sharpe']:+.4f})")
    print(f"  MaxDD: {comb['maxdd']*100:.1f}% (ベース: {base_ref['maxdd']*100:.1f}%)")
    print(f"  Worst5Y: {comb['worst5y']*100:.1f}% (ベース: {base_ref['worst5y']*100:.1f}%)")


if __name__ == '__main__':
    main()
