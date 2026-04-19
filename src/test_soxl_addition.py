"""
方向性B 検証: 高rawLev局面でのSOXL少量追加
============================================
計画の設計:
  w_soxl_frac = max(0, (rawLev - 0.8) × scale)   # NASDAQスロット内のSOXL比率
  w_tqqq = w_nasdaq × (1 - w_soxl_frac)
  w_soxl = w_nasdaq × w_soxl_frac

SOXデータ: 1994-05-04〜（Yahoo Finance ^SOX）
SOXL simulation: SOX日次リターン × 3 − 経費率(0.76%/yr)
※ 1994年以前: SOXL成分ゼロ（データなし）

複数スケール検証:
  scale=0.10 → 最大 rawLev=1.0 時に NASDAQスロットの2%をSOXL
  scale=0.50 → 最大 10%
  scale=1.00 → 最大 20%
  さらに「絶対置換」パターン: rawLev>0.8 時にNASDAQスロットの固定X%をSOXL
"""
import sys, os, types

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
SOX_PATH  = os.path.join(BASE_DIR, 'data', 'sox_daily.csv')

SOXL_COST = 0.0076  # 経費率 0.76%/yr


# ─── SOXL simulation from SOX index ─────────────────────────────────────────

def prepare_soxl_data(dates):
    """
    SOX 1x → SOXL (3x) シミュレーション。
    データなし期間（〜1994-05-03）は前日比ゼロ（NAV継続）として扱う。
    戻り値: np.ndarray (NAV, shape=(len(dates),))
    """
    sox = pd.read_csv(SOX_PATH, parse_dates=['Date'])
    sox = sox.set_index('Date')['Close']

    dates_dt = pd.to_datetime(dates.values)
    soxl_nav = np.ones(len(dates))

    for i in range(1, len(dates)):
        d_prev = dates_dt[i - 1]
        d_curr = dates_dt[i]

        # SOXデータが両日存在する場合のみ計算
        if d_curr in sox.index and d_prev in sox.index:
            sox_ret = sox[d_curr] / sox[d_prev] - 1
            soxl_nav[i] = soxl_nav[i - 1] * (1 + sox_ret * 3 - SOXL_COST / 252)
        else:
            soxl_nav[i] = soxl_nav[i - 1]   # データなし期間は変動なし

    return soxl_nav


# ─── A2 signal computation ────────────────────────────────────────────────────

def calc_asym_ewma(returns, su=30, sd=10):
    var = pd.Series(index=returns.index, dtype=float)
    var.iloc[0] = returns.iloc[:20].var() if len(returns) > 20 else 0.0001
    for i in range(1, len(returns)):
        r = returns.iloc[i]
        a = 2 / (sd + 1) if r < 0 else 2 / (su + 1)
        var.iloc[i] = (1 - a) * var.iloc[i - 1] + a * (r ** 2)
    return np.sqrt(var * 252)


def build_a2_signals(close, returns):
    """Returns (raw, vz) Series."""
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


def build_a2_nav(close, returns, raw):
    """Build A2-optimized NASDAQ NAV."""
    lev = rebalance_threshold(raw, 0.20)
    lr = returns * 3.0; dc = 0.0086 / 252
    dl = lev.shift(2); sr = dl * (lr - dc)
    return (1 + sr.fillna(0)).cumprod()


# ─── Portfolio builder: 4-asset (TQQQ + SOXL + Gold2x + TMF) ────────────────

def build_4asset_portfolio(nasdaq_nav, soxl_nav, gold_nav, bond_nav,
                            wt, ws, wg, wb):
    """
    wt, ws, wg, wb: np arrays of target weights (sum ≈ 1 each day).
    Rebalance rule: drift > 10% OR every 63 days.
    Returns portfolio NAV array.
    """
    n = len(nasdaq_nav)
    assert len(soxl_nav) == n

    nt = np.zeros(n); ns = np.zeros(n); ng = np.zeros(n); nb_ = np.zeros(n)
    for i in range(1, n):
        nt[i] = nasdaq_nav[i] / nasdaq_nav[i-1] - 1 if nasdaq_nav[i-1] > 0 else 0
        ns[i] = soxl_nav[i]  / soxl_nav[i-1]  - 1 if soxl_nav[i-1]  > 0 else 0
        ng[i] = gold_nav[i]  / gold_nav[i-1]  - 1 if gold_nav[i-1]  > 0 else 0
        nb_[i]= bond_nav[i]  / bond_nav[i-1]  - 1 if bond_nav[i-1]  > 0 else 0

    pnav = np.ones(n)
    ct, cs, cg, cb = wt[0], ws[0], wg[0], wb[0]
    rebal_count = 0

    for i in range(1, n):
        port_ret = ct*nt[i] + cs*ns[i] + cg*ng[i] + cb*nb_[i]
        pnav[i] = pnav[i-1] * (1 + port_ret)

        total = (ct*(1+nt[i]) + cs*(1+ns[i]) + cg*(1+ng[i]) + cb*(1+nb_[i]))
        if total > 0:
            ct = ct*(1+nt[i])/total
            cs = cs*(1+ns[i])/total
            cg = cg*(1+ng[i])/total
            cb = cb*(1+nb_[i])/total

        drift = abs(ct-wt[i]) + abs(cs-ws[i]) + abs(cg-wg[i]) + abs(cb-wb[i])
        if drift > 0.10 or i % 63 == 0:
            ct, cs, cg, cb = wt[i], ws[i], wg[i], wb[i]
            rebal_count += 1

    return pnav, rebal_count


# ─── Metrics ─────────────────────────────────────────────────────────────────

def calc_metrics(nav, dates):
    nav_s = pd.Series(nav, index=pd.to_datetime(dates.values))
    ret = nav_s.pct_change().dropna()
    years = len(nav_s) / 252
    cagr = float(nav_s.iloc[-1] ** (1 / years) - 1)
    maxdd = float((nav_s / nav_s.cummax() - 1).min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    worst5y = float(((nav_s / nav_s.shift(252*5)) ** (1/5) - 1).min())
    return {'cagr': cagr, 'maxdd': maxdd, 'sharpe': sharpe, 'worst5y': worst5y}


def print_row(label, m, baseline_cagr=None):
    delta = f"{'▲' if m['cagr']>=baseline_cagr else '▼'}{abs(m['cagr']-baseline_cagr)*100:.2f}%" if baseline_cagr is not None else ""
    print(f"  {label:<42} CAGR={m['cagr']*100:6.2f}% {delta:<10} Sharpe={m['sharpe']:.4f}  MaxDD={m['maxdd']*100:6.1f}%  W5Y={m['worst5y']*100:5.1f}%")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("方向性B 検証: 高rawLev局面でのSOXL少量追加（Dyn 2x3x G0.5ベース）")
    print("=" * 90)

    # ── Load NASDAQ data
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    n = len(dates)
    print(f"NASDAQ: {dates.iloc[0].date()} → {dates.iloc[-1].date()} ({n} rows)")

    # ── Build A2 signals
    raw, vz = build_a2_signals(close, returns)
    nav_a2 = build_a2_nav(close, returns, raw)

    # ── Leveraged asset NAVs
    gold_1x = prepare_gold_data(dates)
    bond_1x = prepare_bond_data(dates)
    g2 = np.ones(n); b3 = np.ones(n)
    for i in range(1, n):
        gr = gold_1x[i]/gold_1x[i-1]-1 if gold_1x[i-1]>0 else 0
        g2[i] = g2[i-1] * (1 + gr*2 - 0.005/252)
        br = bond_1x[i]/bond_1x[i-1]-1 if bond_1x[i-1]>0 else 0
        b3[i] = b3[i-1] * (1 + br*3 - 0.0091/252)

    # ── SOXL NAV（SOX 3x simulation）
    print("SOXLデータ準備中...")
    soxl = prepare_soxl_data(dates)
    sox_start = pd.to_datetime('1994-05-04')
    sox_start_idx = next((i for i, d in enumerate(pd.to_datetime(dates.values)) if d >= sox_start), n)
    print(f"  SOX data from idx={sox_start_idx} ({dates.iloc[sox_start_idx].date()}), "
          f"coverage = {(n-sox_start_idx)/n*100:.1f}% of total period")

    # ── Baseline: Dyn 2x3x G0.5 (no SOXL)
    raw_v = raw.values; vz_v = vz.fillna(0).values
    wn_base = np.zeros(n); wg_base = np.zeros(n); wb_base = np.zeros(n)
    for i in range(n):
        w = np.clip(0.55 + 0.25*raw_v[i] - 0.10*max(vz_v[i],0), 0.30, 0.90)
        wn_base[i] = w
        wg_base[i] = (1-w)*0.50
        wb_base[i] = (1-w)*0.50

    # Baseline: use the 3-asset builder (TQQQ, Gold2x, TMF)
    ws_zero = np.zeros(n)
    pnav_base, _ = build_4asset_portfolio(nav_a2.values, soxl, g2, b3,
                                           wn_base, ws_zero, wg_base, wb_base)
    m_base = calc_metrics(pnav_base, dates)
    print(f"\n{'─'*90}")
    print_row("ベースライン（SOXL なし）", m_base)
    print(f"{'─'*90}\n")

    results = [{'label': 'ベースライン（SOXL なし）', **m_base}]

    # ── SOXL addition tests
    print("SOXL追加テスト（NASDAQスロット内比率 = max(0, (raw-0.8) × scale)）")
    print(f"  {'設定':<42} CAGR          Sharpe   MaxDD     Worst5Y")
    print(f"  {'-'*88}")

    for scale in [0.10, 0.25, 0.50, 1.00, 2.00]:
        # w_soxl_frac = max(0, (raw - 0.8) × scale)  →  NASDAQ スロット内比率
        wt = np.zeros(n); ws = np.zeros(n); wg = np.zeros(n); wb = np.zeros(n)
        for i in range(n):
            w_nasdaq = np.clip(0.55 + 0.25*raw_v[i] - 0.10*max(vz_v[i],0), 0.30, 0.90)
            # SOXデータなし期間はSOXL=0
            if i < sox_start_idx:
                soxl_frac = 0.0
            else:
                soxl_frac = min(max(0.0, (raw_v[i] - 0.8) * scale), 0.50)  # cap 50%
            w_soxl = w_nasdaq * soxl_frac
            wt[i] = w_nasdaq - w_soxl
            ws[i] = w_soxl
            wg[i] = (1 - w_nasdaq) * 0.50
            wb[i] = (1 - w_nasdaq) * 0.50

        max_soxl_w = float(ws.max())
        mean_soxl_w = float(ws[ws > 0].mean()) if (ws > 0).any() else 0.0
        pnav, _ = build_4asset_portfolio(nav_a2.values, soxl, g2, b3, wt, ws, wg, wb)
        m = calc_metrics(pnav, dates)

        label = f"scale={scale:.2f} (max_frac={(min((1.0-0.8)*scale,0.50))*100:.0f}%→max_w={max_soxl_w*100:.1f}%,avg={mean_soxl_w*100:.1f}%)"
        print_row(label, m, m_base['cagr'])
        results.append({'label': label, 'scale': scale,
                         'max_soxl_w': max_soxl_w, 'mean_soxl_w': mean_soxl_w, **m})

    print(f"\nSOXL追加テスト（固定置換: rawLev>0.8時にNASDAQスロットのX%をSOXL）")
    print(f"  {'設定':<42} CAGR          Sharpe   MaxDD     Worst5Y")
    print(f"  {'-'*88}")

    for fixed_frac in [0.05, 0.10, 0.15, 0.20]:
        wt = np.zeros(n); ws = np.zeros(n); wg = np.zeros(n); wb = np.zeros(n)
        for i in range(n):
            w_nasdaq = np.clip(0.55 + 0.25*raw_v[i] - 0.10*max(vz_v[i],0), 0.30, 0.90)
            if i < sox_start_idx or raw_v[i] <= 0.8:
                soxl_frac = 0.0
            else:
                soxl_frac = fixed_frac
            w_soxl = w_nasdaq * soxl_frac
            wt[i] = w_nasdaq - w_soxl
            ws[i] = w_soxl
            wg[i] = (1 - w_nasdaq) * 0.50
            wb[i] = (1 - w_nasdaq) * 0.50

        max_soxl_w = float(ws.max())
        mean_soxl_w = float(ws[ws > 0].mean()) if (ws > 0).any() else 0.0
        pnav, _ = build_4asset_portfolio(nav_a2.values, soxl, g2, b3, wt, ws, wg, wb)
        m = calc_metrics(pnav, dates)

        label = f"固定frac={fixed_frac*100:.0f}% (max_w={max_soxl_w*100:.1f}%,avg={mean_soxl_w*100:.1f}%)"
        print_row(label, m, m_base['cagr'])
        results.append({'label': label, 'fixed_frac': fixed_frac,
                         'max_soxl_w': max_soxl_w, 'mean_soxl_w': mean_soxl_w, **m})

    # ── SOXL特性の確認（期待リターン比較）
    print("\n── SOXL vs TQQQ 期間別パフォーマンス（SOXデータ期間: 1994-）──")
    sox_dates_mask = pd.to_datetime(dates.values) >= sox_start
    if sox_dates_mask.any():
        soxl_s = pd.Series(soxl[sox_dates_mask], index=pd.to_datetime(dates.values)[sox_dates_mask])
        nasdaq_nav_s = pd.Series(nav_a2.values[sox_dates_mask], index=soxl_s.index)
        soxl_years = len(soxl_s) / 252
        soxl_cagr = float(soxl_s.iloc[-1] ** (1/soxl_years) - 1)
        nasdaq_cagr = float(nasdaq_nav_s.iloc[-1] ** (1/soxl_years) - 1)
        soxl_maxdd = float((soxl_s / soxl_s.cummax() - 1).min())
        soxl_vol = float(soxl_s.pct_change().std() * np.sqrt(252))
        nasdaq_vol = float(nasdaq_nav_s.pct_change().std() * np.sqrt(252))
        print(f"  SOXL sim (1994-2026): CAGR={soxl_cagr*100:.1f}%, MaxDD={soxl_maxdd*100:.1f}%, Vol={soxl_vol*100:.1f}%/yr")
        print(f"  TQQQ sim (1994-2026): CAGR={nasdaq_cagr*100:.1f}%, Vol={nasdaq_vol*100:.1f}%/yr")
        corr = pd.Series(soxl).pct_change().corr(pd.Series(nav_a2.values).pct_change())
        print(f"  SOXL vs TQQQ 相関: {corr:.3f}")

    # ── Best config summary
    best = max(results, key=lambda x: x['sharpe'])
    best_cagr = max(results, key=lambda x: x['cagr'])
    print(f"\n{'='*90}")
    print("方向性B 検証結果サマリー")
    print(f"{'='*90}")
    print(f"  ベースライン: CAGR={m_base['cagr']*100:.2f}%, Sharpe={m_base['sharpe']:.4f}")
    print(f"  Best by Sharpe: {best['label']}")
    print(f"    → CAGR={best['cagr']*100:.2f}%, Sharpe={best['sharpe']:.4f}, MaxDD={best['maxdd']*100:.1f}%")
    print(f"  Best by CAGR:   {best_cagr['label']}")
    print(f"    → CAGR={best_cagr['cagr']*100:.2f}%, Sharpe={best_cagr['sharpe']:.4f}, MaxDD={best_cagr['maxdd']*100:.1f}%")
    delta_cagr = best_cagr['cagr'] - m_base['cagr']
    print(f"\n  最大CAGR改善: {'+' if delta_cagr >= 0 else ''}{delta_cagr*100:.2f}%")
    print(f"  計画での期待改善: +1〜4%")

    # ── Save results
    out = [{k: v for k, v in r.items() if k != 'yearly'} for r in results]
    pd.DataFrame(out).to_csv(os.path.join(BASE_DIR, 'soxl_addition_results.csv'), index=False)
    print(f"\n  結果保存: soxl_addition_results.csv")


if __name__ == '__main__':
    main()
