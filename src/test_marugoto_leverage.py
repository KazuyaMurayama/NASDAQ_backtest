"""
方向性C 検証: まるごとレバレッジをGoldスロットの一部に活用
==========================================================
計画の設計:
  旧: w_gold = (1 - w_nasdaq) × 0.50
  新: w_gold_pure  = (1 - w_nasdaq) × 0.25   # Gold 2x ETN: 半減
      w_marugoto   = (1 - w_nasdaq) × 0.25   # まるごとレバレッジ: 追加
      w_bond(TMF)  = (1 - w_nasdaq) × 0.50   # 変更なし

まるごとレバレッジ合成モデル:
  - 米国3倍4資産リスク分散ファンド（実際の運用方針: リスクパリティ）
  - 構成: Bond(30yr) / Gold / S&P500 / REIT を逆ボラ加重 → 各3倍レバレッジ
  - 推定ウェイト（逆ボラ加重、正規化後）:
      Bond 45.0% / Gold 21.1% / S&P500 17.8% / REIT 16.1%
  - 実効エクスポージャー: Bond 135% / Gold 63% / S&P500 53% / REIT 48%
  - 経費率: 0.4675%/yr（実際の信託報酬）
  - REITデータ: VNQ(2004〜), 2004年以前はS&P500で代替

複数パターン検証:
  - Goldスロットのまるごとレバレッジ置換比率を 0%〜100% で変化
  - 計画案（25%）を含む
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
DATA_PATH  = os.path.join(BASE_DIR, 'NASDAQ_extended_to_2026.csv')
SP500_PATH = os.path.join(BASE_DIR, 'data', 'sp500_daily.csv')
VNQ_PATH   = os.path.join(BASE_DIR, 'data', 'vnq_daily.csv')

# まるごとレバレッジ 経費率
MARUGOTO_COST = 0.004675  # 0.4675%/yr

# リスクパリティ（逆ボラ加重、正規化）
_vols = [0.075, 0.16, 0.19, 0.21]
_inv  = [1/v for v in _vols]
_sum  = sum(_inv)
RP_WEIGHTS = [v / _sum for v in _inv]  # bond, gold, sp500, reit
# = [0.450, 0.211, 0.178, 0.161]


# ─── まるごとレバレッジ NAV 構築 ─────────────────────────────────────────────

def prepare_sp500(dates):
    sp = pd.read_csv(SP500_PATH, parse_dates=['Date']).set_index('Date')['Close']
    result = np.ones(len(dates))
    dates_dt = pd.to_datetime(dates.values)
    for i in range(1, len(dates)):
        d = dates_dt[i]; dp = dates_dt[i - 1]
        if d in sp.index and dp in sp.index:
            result[i] = sp[d] / sp[dp]
        else:
            result[i] = 1.0  # データなし → 変動ゼロ
    return result  # 日次比率配列（1 + return）


def prepare_reit(dates):
    """VNQ(2004〜) + S&P500代替（〜2004）の日次比率配列を返す。"""
    vnq = pd.read_csv(VNQ_PATH, parse_dates=['Date']).set_index('Date')['Close']
    sp  = pd.read_csv(SP500_PATH, parse_dates=['Date']).set_index('Date')['Close']
    vnq_start = vnq.index.min()
    result = np.ones(len(dates))
    dates_dt = pd.to_datetime(dates.values)
    for i in range(1, len(dates)):
        d = dates_dt[i]; dp = dates_dt[i - 1]
        if d >= vnq_start and d in vnq.index and dp in vnq.index:
            result[i] = vnq[d] / vnq[dp]
        elif d in sp.index and dp in sp.index:
            result[i] = sp[d] / sp[dp]
        else:
            result[i] = 1.0
    return result


def build_marugoto_nav(dates, bond_1x, gold_1x):
    """
    まるごとレバレッジ合成NAV。
    各コンポーネントは1x価格から日次リターンを取り → ×3 → リスクパリティ合算。
    経費率 0.4675%/yr を差し引き。
    """
    n = len(dates)
    sp500_ratio = prepare_sp500(dates)   # 1+return arrays
    reit_ratio  = prepare_reit(dates)

    w_b, w_g, w_s, w_r = RP_WEIGHTS

    nav = np.ones(n)
    for i in range(1, n):
        r_bond = bond_1x[i] / bond_1x[i-1] - 1 if bond_1x[i-1] > 0 else 0
        r_gold = gold_1x[i] / gold_1x[i-1] - 1 if gold_1x[i-1] > 0 else 0
        r_sp   = sp500_ratio[i] - 1
        r_reit = reit_ratio[i] - 1

        r_fund = 3.0 * (w_b*r_bond + w_g*r_gold + w_s*r_sp + w_r*r_reit)
        nav[i] = nav[i-1] * (1 + r_fund - MARUGOTO_COST / 252)

    return nav


# ─── A2 signals ──────────────────────────────────────────────────────────────

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


# ─── Portfolio builder: 4-asset (TQQQ + Gold2x + Marugoto + TMF) ────────────

def build_4asset_portfolio(nasdaq_nav, gold2x_nav, marugoto_nav, bond3x_nav,
                            wn, wg, wm, wb):
    """
    4資産ポートフォリオ。
    リバランス: drift > 10% OR 63日に1回（現行と同じ設定）。
    """
    n = len(nasdaq_nav)
    rn = np.zeros(n); rg = np.zeros(n); rm = np.zeros(n); rb = np.zeros(n)
    for i in range(1, n):
        rn[i] = nasdaq_nav[i] / nasdaq_nav[i-1] - 1 if nasdaq_nav[i-1] > 0 else 0
        rg[i] = gold2x_nav[i] / gold2x_nav[i-1] - 1 if gold2x_nav[i-1] > 0 else 0
        rm[i] = marugoto_nav[i] / marugoto_nav[i-1] - 1 if marugoto_nav[i-1] > 0 else 0
        rb[i] = bond3x_nav[i] / bond3x_nav[i-1] - 1 if bond3x_nav[i-1] > 0 else 0

    pnav = np.ones(n)
    cn, cg, cm, cb = wn[0], wg[0], wm[0], wb[0]
    rebal_count = 0

    for i in range(1, n):
        ret = cn*rn[i] + cg*rg[i] + cm*rm[i] + cb*rb[i]
        pnav[i] = pnav[i-1] * (1 + ret)

        total = (cn*(1+rn[i]) + cg*(1+rg[i]) + cm*(1+rm[i]) + cb*(1+rb[i]))
        if total > 0:
            cn = cn*(1+rn[i])/total
            cg = cg*(1+rg[i])/total
            cm = cm*(1+rm[i])/total
            cb = cb*(1+rb[i])/total

        drift = abs(cn-wn[i]) + abs(cg-wg[i]) + abs(cm-wm[i]) + abs(cb-wb[i])
        if drift > 0.10 or i % 63 == 0:
            cn, cg, cm, cb = wn[i], wg[i], wm[i], wb[i]
            rebal_count += 1

    return pnav, rebal_count


# ─── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(nav, dates):
    nav_s = pd.Series(nav, index=pd.to_datetime(dates.values))
    ret = nav_s.pct_change().dropna()
    years = len(nav_s) / 252
    cagr = float(nav_s.iloc[-1] ** (1 / years) - 1)
    maxdd = float((nav_s / nav_s.cummax() - 1).min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    worst5y = float(((nav_s / nav_s.shift(252*5)) ** (1/5) - 1).min())
    return {'cagr': cagr, 'maxdd': maxdd, 'sharpe': sharpe, 'worst5y': worst5y}


def print_row(label, m, base_cagr=None):
    if base_cagr is not None:
        delta = m['cagr'] - base_cagr
        d_str = f"{'▲' if delta>=0 else '▼'}{abs(delta)*100:.2f}%"
    else:
        d_str = ""
    print(f"  {label:<46} CAGR={m['cagr']*100:6.2f}% {d_str:<10} "
          f"Sharpe={m['sharpe']:.4f}  MaxDD={m['maxdd']*100:6.1f}%  W5Y={m['worst5y']*100:5.1f}%")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 95)
    print("方向性C 検証: まるごとレバレッジをGoldスロットの一部に活用")
    print("=" * 95)

    # ── Load NASDAQ
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    n = len(dates)
    print(f"NASDAQ: {dates.iloc[0].date()} → {dates.iloc[-1].date()} ({n} rows)")

    # ── A2 signals
    raw, vz = build_a2_signals(close, returns)

    # ── A2 NASDAQ NAV
    lev_a2 = rebalance_threshold(raw, 0.20)
    lr = returns * 3.0; dc = 0.0086 / 252
    dl = lev_a2.shift(2); sr = dl * (lr - dc)
    nav_a2 = (1 + sr.fillna(0)).cumprod()

    # ── Leveraged asset NAVs
    gold_1x  = prepare_gold_data(dates)
    bond_1x  = prepare_bond_data(dates)

    # Gold 2x
    g2 = np.ones(n)
    for i in range(1, n):
        gr = gold_1x[i]/gold_1x[i-1]-1 if gold_1x[i-1]>0 else 0
        g2[i] = g2[i-1] * (1 + gr*2 - 0.005/252)

    # TMF (bond 3x)
    b3 = np.ones(n)
    for i in range(1, n):
        br = bond_1x[i]/bond_1x[i-1]-1 if bond_1x[i-1]>0 else 0
        b3[i] = b3[i-1] * (1 + br*3 - 0.0091/252)

    # まるごとレバレッジ 合成NAV
    print("\nまるごとレバレッジ合成NAV構築中...")
    print(f"  リスクパリティウェイト: Bond={RP_WEIGHTS[0]*100:.1f}%, Gold={RP_WEIGHTS[1]*100:.1f}%, "
          f"SP500={RP_WEIGHTS[2]*100:.1f}%, REIT={RP_WEIGHTS[3]*100:.1f}%")
    print(f"  REIT: VNQ(2004〜) + S&P500代替(〜2004)")
    marugoto = build_marugoto_nav(dates, bond_1x, gold_1x)

    # まるごとレバレッジの特性確認
    mar_s = pd.Series(marugoto, index=pd.to_datetime(dates.values))
    years_total = n / 252
    mar_cagr = float(mar_s.iloc[-1] ** (1/years_total) - 1)
    mar_maxdd = float((mar_s / mar_s.cummax() - 1).min())
    mar_vol = float(mar_s.pct_change().std() * np.sqrt(252))
    print(f"\n  まるごとレバレッジ合成 (1974-2026): "
          f"CAGR={mar_cagr*100:.1f}%, MaxDD={mar_maxdd*100:.1f}%, Vol={mar_vol*100:.1f}%/yr")

    # Crisis year check
    mar_df = pd.DataFrame({'nav': marugoto, 'date': dates.values})
    mar_df['year'] = pd.to_datetime(mar_df['date']).dt.year
    yearly = mar_df.groupby('year')['nav'].agg(['first','last'])
    yearly_ret = (yearly['last']/yearly['first'] - 1) * 100
    for yr in [2000, 2001, 2002, 2008, 2020, 2022]:
        if yr in yearly_ret.index:
            print(f"    {yr}: {yearly_ret[yr]:+.1f}%")

    raw_v = raw.values; vz_v = vz.fillna(0).values

    # ── Baseline: Dyn 2x3x G0.5 (Gold 2x 50%, TMF 50%)
    wn0 = np.zeros(n); wg0 = np.zeros(n); wm0 = np.zeros(n); wb0 = np.zeros(n)
    for i in range(n):
        w = np.clip(0.55 + 0.25*raw_v[i] - 0.10*max(vz_v[i],0), 0.30, 0.90)
        wn0[i] = w
        wg0[i] = (1 - w) * 0.50   # Gold 2x
        wm0[i] = 0.0               # まるごとレバレッジ なし
        wb0[i] = (1 - w) * 0.50   # TMF

    pnav_base, _ = build_4asset_portfolio(nav_a2.values, g2, marugoto, b3, wn0, wg0, wm0, wb0)
    m_base = calc_metrics(pnav_base, dates)

    print(f"\n{'─'*95}")
    print_row("ベースライン G0.5（Gold100% / Maru0% / TMF100%）", m_base)
    print(f"{'─'*95}\n")

    results = [{'label': 'ベースライン（Goldスロット=Gold2x 100%）',
                'maru_ratio': 0.0, **m_base}]

    # ── まるごとレバレッジ置換テスト
    print("Gold スロット内のまるごとレバレッジ置換比率テスト")
    print(f"  ※ Gold スロット = (1-w_nasdaq) × 0.50 の部分")
    print(f"  {'設定':<46} CAGR          Sharpe   MaxDD     Worst5Y")
    print(f"  {'-'*91}")

    for maru_ratio in [0.10, 0.25, 0.50, 0.75, 1.00]:
        wn = np.zeros(n); wg = np.zeros(n); wm = np.zeros(n); wb = np.zeros(n)
        for i in range(n):
            w = np.clip(0.55 + 0.25*raw_v[i] - 0.10*max(vz_v[i],0), 0.30, 0.90)
            gold_slot = (1 - w) * 0.50  # Gold スロット合計
            wn[i] = w
            wg[i] = gold_slot * (1 - maru_ratio)   # Gold 2x の残り
            wm[i] = gold_slot * maru_ratio           # まるごとレバレッジ
            wb[i] = (1 - w) * 0.50                  # TMF 変更なし

        pnav, _ = build_4asset_portfolio(nav_a2.values, g2, marugoto, b3, wn, wg, wm, wb)
        m = calc_metrics(pnav, dates)

        label = (f"Gold→Maru {maru_ratio*100:.0f}% "
                 f"(Gold={wg.mean()*100:.1f}%, Maru={wm.mean()*100:.1f}%, TMF={wb.mean()*100:.1f}%avg)")
        print_row(label, m, m_base['cagr'])
        results.append({'label': label, 'maru_ratio': maru_ratio, **m})

    # ── TMFの一部をまるごとレバレッジで代替するテスト（方向性Eの参照）
    print(f"\nTMF スロット内のまるごとレバレッジ置換テスト（方向性E 参考）")
    print(f"  {'設定':<46} CAGR          Sharpe   MaxDD     Worst5Y")
    print(f"  {'-'*91}")

    for tmf_maru_ratio in [0.25, 0.50]:
        wn = np.zeros(n); wg = np.zeros(n); wm = np.zeros(n); wb = np.zeros(n)
        for i in range(n):
            w = np.clip(0.55 + 0.25*raw_v[i] - 0.10*max(vz_v[i],0), 0.30, 0.90)
            bond_slot = (1 - w) * 0.50
            gold_slot = (1 - w) * 0.50
            wn[i] = w
            wg[i] = gold_slot          # Gold 2x 変更なし
            wm[i] = bond_slot * tmf_maru_ratio   # TMFの一部をまるごとへ
            wb[i] = bond_slot * (1 - tmf_maru_ratio)

        pnav, _ = build_4asset_portfolio(nav_a2.values, g2, marugoto, b3, wn, wg, wm, wb)
        m = calc_metrics(pnav, dates)

        label = (f"TMF→Maru {tmf_maru_ratio*100:.0f}% "
                 f"(Gold={wg.mean()*100:.1f}%, Maru={wm.mean()*100:.1f}%, TMF={wb.mean()*100:.1f}%avg)")
        print_row(label, m, m_base['cagr'])
        results.append({'label': label, 'tmf_maru_ratio': tmf_maru_ratio, **m})

    # ── 計画案そのまま: Gold 25%→Gold2x, 25%→Maru, TMF 50%
    print(f"\n計画書の設計案そのまま（Gold2x: 25% / まるごとMaru: 25% / TMF: 50%）")
    # これは maru_ratio=0.50 に相当（Goldスロット50%をまるごとに）→ 上で計算済み
    plan_result = [r for r in results if 'maru_ratio' in r and abs(r.get('maru_ratio',0)-0.50)<0.01]
    if plan_result:
        pr = plan_result[0]
        print(f"  CAGR={pr['cagr']*100:.2f}%, Sharpe={pr['sharpe']:.4f}, "
              f"MaxDD={pr['maxdd']*100:.1f}%, W5Y={pr['worst5y']*100:.1f}%")
        print(f"  vs ベースライン: CAGR {'+' if pr['cagr']>=m_base['cagr'] else ''}"
              f"{(pr['cagr']-m_base['cagr'])*100:.2f}%")

    # ── Summary
    best_sh = max(results, key=lambda x: x['sharpe'])
    best_cagr = max(results, key=lambda x: x['cagr'])

    print(f"\n{'='*95}")
    print("方向性C 検証結果サマリー")
    print(f"{'='*95}")
    print(f"  ベースライン: CAGR={m_base['cagr']*100:.2f}%, Sharpe={m_base['sharpe']:.4f}, "
          f"MaxDD={m_base['maxdd']*100:.1f}%, Worst5Y={m_base['worst5y']*100:.1f}%")
    print(f"  Best by Sharpe : {best_sh['label']}")
    print(f"    → CAGR={best_sh['cagr']*100:.2f}%, Sharpe={best_sh['sharpe']:.4f}, "
          f"MaxDD={best_sh['maxdd']*100:.1f}%, Worst5Y={best_sh['worst5y']*100:.1f}%")
    print(f"  Best by CAGR   : {best_cagr['label']}")
    print(f"    → CAGR={best_cagr['cagr']*100:.2f}%, Sharpe={best_cagr['sharpe']:.4f}, "
          f"MaxDD={best_cagr['maxdd']*100:.1f}%, Worst5Y={best_cagr['worst5y']*100:.1f}%")

    delta_best = best_cagr['cagr'] - m_base['cagr']
    print(f"\n  最大CAGR改善: {'+' if delta_best>=0 else ''}{delta_best*100:.2f}%")
    print(f"  計画での期待改善: +0.5〜1.5%")

    # Save
    pd.DataFrame(results).to_csv(
        os.path.join(BASE_DIR, 'marugoto_leverage_results.csv'), index=False)
    print(f"\n  結果保存: marugoto_leverage_results.csv")


if __name__ == '__main__':
    main()
