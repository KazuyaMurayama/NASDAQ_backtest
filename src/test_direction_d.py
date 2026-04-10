"""
方向性D 検証: DD回避期の短期国債投資
=========================================
現行: DD=0（CASH状態）のNASDAQスロット → 収益 0%
改善案: 短期国債（SHY/BIL相当）で運用 → 年率〜4%

実装:
  lev=0（DD回避中）の日に、NASDAQスロットへの配分部分が
  T-bill 相当の利息を得るよう A2 NAV に加算。

T-bill 近似金利（歴史的 FF 金利 参考）:
  1974-1981 : 8%（スタグフレーション期）
  1982-2001 : 5%（高金利→低下）
  2002-2009 : 3%
  2010-2021 : 0.7%（量的緩和、ゼロ金利）
  2022-     : 5%（急速利上げ）

ベースライン: Dyn 2x3x G0.5
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


def make_tbill_daily_rate(dates_dt):
    """歴史的 T-bill 金利の近似（日次）"""
    years = np.array([d.year for d in dates_dt])
    annual = np.where(years < 1982, 0.08,
              np.where(years < 2002, 0.05,
                np.where(years < 2010, 0.03,
                  np.where(years < 2022, 0.007, 0.05))))
    return annual / 252


def build_a2_nav_tbill(close, returns, raw, tbill_daily, tbill_mode):
    """
    A2 NASDAQ NAV。tbill_mode='zero' は通常ベース。
    tbill_mode='tbill' の場合、lev=0 の日に T-bill 利息を加算。
    """
    lev = rebalance_threshold(raw, 0.20)
    lr = returns * 3.0
    dc = 0.0086 / 252
    dl = lev.shift(2)
    dl_arr = dl.fillna(0).values

    if tbill_mode == 'zero':
        sr = dl * (lr - dc)
        nav = (1 + sr.fillna(0)).cumprod()
        return nav, dl_arr
    else:
        sr_base = (dl * (lr - dc)).fillna(0).values
        # lev=0（キャッシュ）の日にT-bill加算
        cash_mask = (dl_arr == 0)
        sr_tbill = np.where(cash_mask, tbill_daily, sr_base)
        nav = np.cumprod(1 + sr_tbill)
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
    print("方向性D 検証: DD回避期（lev=0）の短期国債投資")
    print("=" * 80)

    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    n = len(dates)
    dates_dt = pd.to_datetime(dates.values)
    print(f"データ期間: {dates.iloc[0].date()} → {dates.iloc[-1].date()} ({n} rows)")

    # ── A2 シグナル
    raw, vz = build_a2_signals(close, returns)
    raw_v = raw.values; vz_v = vz.fillna(0).values

    # ── Gold / Bond NAV
    gold_1x = prepare_gold_data(dates)
    bond_1x = prepare_bond_data(dates)
    g2 = np.ones(n); b3 = np.ones(n)
    for i in range(1, n):
        gr = gold_1x[i]/gold_1x[i-1]-1 if gold_1x[i-1] > 0 else 0
        g2[i] = g2[i-1] * (1 + gr*2 - 0.005/252)
        br = bond_1x[i]/bond_1x[i-1]-1 if bond_1x[i-1] > 0 else 0
        b3[i] = b3[i-1] * (1 + br*3 - 0.0091/252)

    # ── ポートフォリオ重み（ベース）
    wn_arr = np.zeros(n); wg_arr = np.zeros(n); wb_arr = np.zeros(n)
    for i in range(n):
        w = np.clip(0.55 + 0.25*raw_v[i] - 0.10*max(vz_v[i], 0), 0.30, 0.90)
        wn_arr[i] = w; wg_arr[i] = (1-w)*0.50; wb_arr[i] = (1-w)*0.50

    # ── T-bill 日次金利（歴史的近似）
    tbill_hist = make_tbill_daily_rate(dates_dt)

    # ── キャッシュ期間の分析
    nav_a2_base, dl_arr = build_a2_nav_tbill(close, returns, raw, tbill_hist, 'zero')
    cash_mask = (dl_arr == 0)
    cash_frac = cash_mask.mean()
    print(f"\nDD=0（キャッシュ）期間: {cash_frac*100:.1f}% (全期間中)")

    # 期間別キャッシュ率
    print("\n期間別キャッシュ率:")
    for label, start, end in [
        ('1974-1982 (スタグフレーション)', 1974, 1982),
        ('1982-2000 (長期強気相場)',       1982, 2000),
        ('2000-2009 (2回の暴落)',          2000, 2009),
        ('2009-2022 (量的緩和・低金利)',   2009, 2022),
        ('2022-2026 (金利急上昇)',         2022, 2026),
    ]:
        mask = np.array([start <= d.year < end for d in dates_dt])
        if mask.any():
            cf = cash_mask[mask].mean()
            print(f"  {label}: {cf*100:.1f}%")

    # ── シナリオ別検証
    scenarios = [
        ('zero',    '現行: 0%利息（ベース）', None),
        ('tbill',   '歴史的 FF 金利近似',     tbill_hist),
        ('tbill',   '一律 3%',                np.full(n, 0.03/252)),
        ('tbill',   '一律 4%',                np.full(n, 0.04/252)),
        ('tbill',   '一律 5%',                np.full(n, 0.05/252)),
    ]

    print(f"\n{'シナリオ':>22} {'CAGR':>8} {'Δ CAGR':>8} {'Sharpe':>8} {'ΔSharpe':>8} {'MaxDD':>8} {'OOS Sh':>8}")
    print("-" * 80)

    baseline_ref = None
    results = []
    for mode, label, tbill in scenarios:
        if mode == 'zero':
            nav_a2 = nav_a2_base
        else:
            nav_a2, _ = build_a2_nav_tbill(close, returns, raw, tbill, mode)
        pnav = build_portfolio_3asset(
            nav_a2.values if hasattr(nav_a2, 'values') else nav_a2,
            g2, b3, wn_arr, wg_arr, wb_arr
        )
        m = calc_metrics(pnav, dates)
        m['label'] = label
        results.append(m)
        if mode == 'zero':
            baseline_ref = m

    for m in results:
        dcagr = (m['cagr'] - baseline_ref['cagr']) * 100
        dsh = m['sharpe'] - baseline_ref['sharpe']
        oos_str = f"{m['oos_sharpe']:.4f}" if not (m['oos_sharpe'] != m['oos_sharpe']) else "N/A"
        dcagr_str = f"{'+' if dcagr >= 0 else ''}{dcagr:.2f}%"
        dsh_str = f"{'+' if dsh >= 0 else ''}{dsh:.4f}"
        print(f"  {m['label']:>22} {m['cagr']*100:>7.2f}% {dcagr_str:>8} "
              f"{m['sharpe']:>8.4f} {dsh_str:>8} {m['maxdd']*100:>7.1f}% {oos_str:>8}")

    # ── 理論的推定との比較
    base = baseline_ref
    hist_r = results[1]
    print(f"\n{'='*80}")
    print("方向性D 検証結果サマリー")
    print(f"{'='*80}")
    print(f"  ベースライン: CAGR={base['cagr']*100:.2f}%, Sharpe={base['sharpe']:.4f}")
    print(f"\n  NASDAQスロット重み(平均): ~{wn_arr.mean()*100:.0f}%")
    print(f"  キャッシュ期間: {cash_frac*100:.1f}%")
    print(f"  T-bill 4%の場合の理論改善: {wn_arr.mean() * 0.04 * cash_frac * 100:.2f}%/yr")
    hist_delta = (hist_r['cagr'] - base['cagr']) * 100
    print(f"  歴史的金利で実測した改善:  {'+' if hist_delta >= 0 else ''}{hist_delta:.2f}%/yr")
    print(f"  計画での期待改善: +0.3〜1.2%")

    # ── 注意事項
    print(f"\n  注意事項:")
    print(f"  - 低金利期（2010-2021）のキャッシュ期間ではT-bill利回りがほぼ0%")
    print(f"  - 歴史的金利モデルはその期間の実効改善を反映")
    print(f"  - OOS（2021-2026）では高金利環境のため改善効果が大きくなる可能性")
    oos_str = f"{hist_r['oos_sharpe']:.4f}" if not (hist_r['oos_sharpe'] != hist_r['oos_sharpe']) else "N/A"
    print(f"  - 歴史的金利モデルの OOS Sharpe: {oos_str} (ベース: {base['oos_sharpe']:.4f})")


if __name__ == '__main__':
    main()
