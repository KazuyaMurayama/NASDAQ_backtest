"""
方向性F 検証: 注文執行タイミングの最適化（遅延感度分析）
===========================================================
現行: DELAY=2（シグナル当日夜→翌日発注→T+2受渡）
検証: DELAY=0,1,2,3,5 での Dyn 2x3x G0.5 パフォーマンス比較

製品マッピング:
  delay=0 : CFD（即時約定）
  delay=1 : TQQQ 成行→翌日反映（最良ケース）
  delay=2 : TQQQ 通常ケース（現行前提）★ baseline
  delay=3 : TQQQ 最悪ケース（祝日等）
  delay=5 : NASDAQ100 3倍ブル（投資信託）

ベースライン (delay=2):
  CAGR=31.40% / Sharpe=1.297 / MaxDD=-33.4%（test_soxl_addition.py と同実装）
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


# ─── A2 シグナル（test_soxl_addition.py と同実装）────────────────────────────

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


def build_a2_nav_with_delay(close, returns, raw, delay):
    """A2 NASDAQ NAV with specified delay."""
    lev = rebalance_threshold(raw, 0.20)
    lr = returns * 3.0
    dc = 0.0086 / 252
    dl = lev.shift(delay)
    sr = dl * (lr - dc)
    return (1 + sr.fillna(0)).cumprod()


# ─── ポートフォリオ構築（test_soxl_addition.py と同実装）────────────────────

def build_portfolio_3asset(nasdaq_nav, gold_nav, bond_nav, wn, wg, wb):
    """
    3資産ポートフォリオ。drift > 10% または 63日毎にリバランス。
    """
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
            ct = ct*(1+nt[i])/total
            cg = cg*(1+ng[i])/total
            cb = cb*(1+nb[i])/total
        drift = abs(ct-wn[i]) + abs(cg-wg[i]) + abs(cb-wb[i])
        if drift > 0.10 or i % 63 == 0:
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
    print("方向性F 検証: 注文執行タイミング（遅延感度）Dyn 2x3x G0.5")
    print("=" * 80)

    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    n = len(dates)
    print(f"データ期間: {dates.iloc[0].date()} → {dates.iloc[-1].date()} ({n} rows)")

    # ── A2 シグナル（遅延非依存）
    raw, vz = build_a2_signals(close, returns)
    raw_v = raw.values; vz_v = vz.fillna(0).values

    # ── Gold 2x / TMF 3x NAV（遅延非依存）
    gold_1x = prepare_gold_data(dates)
    bond_1x = prepare_bond_data(dates)
    g2 = np.ones(n); b3 = np.ones(n)
    for i in range(1, n):
        gr = gold_1x[i]/gold_1x[i-1]-1 if gold_1x[i-1] > 0 else 0
        g2[i] = g2[i-1] * (1 + gr*2 - 0.005/252)
        br = bond_1x[i]/bond_1x[i-1]-1 if bond_1x[i-1] > 0 else 0
        b3[i] = b3[i-1] * (1 + br*3 - 0.0091/252)

    # ── Dyn 2x3x G0.5 ウェイト（遅延非依存）
    wn_arr = np.zeros(n); wg_arr = np.zeros(n); wb_arr = np.zeros(n)
    for i in range(n):
        w = np.clip(0.55 + 0.25*raw_v[i] - 0.10*max(vz_v[i], 0), 0.30, 0.90)
        wn_arr[i] = w; wg_arr[i] = (1-w)*0.50; wb_arr[i] = (1-w)*0.50

    delays = [0, 1, 2, 3, 5]
    product_map = {0: 'CFD', 1: 'TQQQ最良', 2: 'TQQQ現行(基準)', 3: 'TQQQ最悪', 5: '投信'}

    results = []
    baseline_ref = None

    print(f"\n{'遅延':>4} {'製品':>16} {'CAGR':>8} {'Δ vs d=2':>10} {'Sharpe':>8} {'ΔSharpe':>9} {'MaxDD':>8} {'OOS Sh':>8}")
    print("-" * 80)

    for delay in delays:
        nav_a2 = build_a2_nav_with_delay(close, returns, raw, delay)
        pnav = build_portfolio_3asset(nav_a2.values, g2, b3, wn_arr, wg_arr, wb_arr)
        m = calc_metrics(pnav, dates)
        m['delay'] = delay
        results.append(m)
        if delay == 2:
            baseline_ref = m

    for r in results:
        d_val = r['delay']
        dcagr = (r['cagr'] - baseline_ref['cagr']) * 100
        dsh = r['sharpe'] - baseline_ref['sharpe']
        oos_str = f"{r['oos_sharpe']:.4f}" if not (r['oos_sharpe'] != r['oos_sharpe']) else "N/A"
        dcagr_str = f"{'+' if dcagr >= 0 else ''}{dcagr:.2f}%"
        dsh_str = f"{'+' if dsh >= 0 else ''}{dsh:.4f}"
        marker = " ← 基準" if d_val == 2 else ""
        print(f"  {d_val:>2} {product_map[d_val]:>16} {r['cagr']*100:>7.2f}% {dcagr_str:>10} "
              f"{r['sharpe']:>8.4f} {dsh_str:>9} {r['maxdd']*100:>7.1f}% {oos_str:>8}{marker}")

    # ── OOS（2021-2026）
    print(f"\n{'─'*80}")
    print("OOS検証 (2021-2026)")
    for r in results:
        oos_str = f"{r['oos_sharpe']:.4f}" if not (r['oos_sharpe'] != r['oos_sharpe']) else "N/A"
        print(f"  delay={r['delay']} ({product_map[r['delay']]:>10}): OOS Sharpe={oos_str}")

    # ── 結論
    base = baseline_ref
    print(f"\n{'='*80}")
    print("方向性F 検証結果サマリー")
    print(f"{'='*80}")
    print(f"  ベースライン (d=2): CAGR={base['cagr']*100:.2f}%, Sharpe={base['sharpe']:.4f}")

    d1_r = next(r for r in results if r['delay'] == 1)
    d0_r = results[0]; d5_r = results[-1]
    print(f"  TQQQ最良 (d=1): CAGR={d1_r['cagr']*100:.2f}% "
          f"({'+' if (d1_r['cagr']-base['cagr'])*100>=0 else ''}{(d1_r['cagr']-base['cagr'])*100:.2f}%), "
          f"Sharpe={d1_r['sharpe']:.4f}")
    print(f"  CFD (d=0) vs 現行: CAGR {'+' if (d0_r['cagr']-base['cagr'])*100>=0 else ''}{(d0_r['cagr']-base['cagr'])*100:.2f}%, "
          f"Sharpe {'+' if d0_r['sharpe']-base['sharpe']>=0 else ''}{d0_r['sharpe']-base['sharpe']:.4f}")
    print(f"  投信 (d=5) vs 現行: CAGR {'+' if (d5_r['cagr']-base['cagr'])*100>=0 else ''}{(d5_r['cagr']-base['cagr'])*100:.2f}%, "
          f"Sharpe {'+' if d5_r['sharpe']-base['sharpe']>=0 else ''}{d5_r['sharpe']-base['sharpe']:.4f}")

    max_r = max(results[1:], key=lambda r: r['cagr'])  # exclude d=0 (unrealistic)
    print(f"\n  実現可能な最大改善 (d=1): CAGR {'+' if (max_r['cagr']-base['cagr'])*100>=0 else ''}{(max_r['cagr']-base['cagr'])*100:.2f}%")
    print(f"  計画での期待改善: +0.5〜1.5%")

    # ── delay=0 の注意
    print(f"\n  ⚠️  delay=0 (CFD) の +{(d0_r['cagr']-base['cagr'])*100:.1f}% は先読みバイアスを含む可能性")
    print(f"     シグナル当日の終値で同日中に約定させるのはバックテスト上の理想値。")
    print(f"     実際には delay=1（TQQQ翌日寄付き）が最短の現実的遅延。")


if __name__ == '__main__':
    main()
