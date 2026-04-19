"""
方向性F OOSパラドックス調査
==============================
疑問: なぜ d=1（速い）が d=2（現行）より OOS Sharpe が低いのか？
  d=0: OOS=1.6413, d=1: OOS=0.8522, d=2: OOS=0.8659(基準), d=5: OOS=0.9127

調査内容:
  1. 年次リターン比較（d=1 vs d=2）: どの年で分岐するか
  2. 信号転換時の翌日ギャップ分析: d=1 は寄り付きギャップリスクを受けるか
  3. 統計的有意性: IS+1.40%改善は52年間で有意か（ブートストラップ）
  4. MaxDD 悪化 (-33.4% → -38.5%) の原因特定

背景: d=1 は翌日「寄り付き価格」で執行。ナスダックは翌日の寄り付き〜前日終値の
ギャップが大きい特性があり（決算・FRB発表等）、この「悪い方向のギャップ」を
d=1 は喰らい、d=2 は回避できる場合がある。
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


def build_a2_nav(close, returns, raw, delay):
    lev = rebalance_threshold(raw, 0.20)
    lr = returns * 3.0; dc = 0.0086 / 252
    dl = lev.shift(delay)
    sr = dl * (lr - dc)
    return (1 + sr.fillna(0)).cumprod()


def build_portfolio_3asset(nasdaq_nav, gold_nav, bond_nav, wn, wg, wb):
    n = len(nasdaq_nav)
    nt = np.zeros(n); ng = np.zeros(n); nb = np.zeros(n)
    for i in range(1, n):
        nt[i] = nasdaq_nav[i] / nasdaq_nav[i-1] - 1 if nasdaq_nav[i-1] > 0 else 0
        ng[i] = gold_nav[i]  / gold_nav[i-1]  - 1 if gold_nav[i-1]  > 0 else 0
        nb[i] = bond_nav[i]  / bond_nav[i-1]  - 1 if bond_nav[i-1]  > 0 else 0
    pnav = np.ones(n); ct, cg, cb = wn[0], wg[0], wb[0]
    for i in range(1, n):
        port_ret = ct*nt[i] + cg*ng[i] + cb*nb[i]
        pnav[i] = pnav[i-1] * (1 + port_ret)
        total = ct*(1+nt[i]) + cg*(1+ng[i]) + cb*(1+nb[i])
        if total > 0:
            ct = ct*(1+nt[i])/total; cg = cg*(1+ng[i])/total; cb = cb*(1+nb[i])/total
        if abs(ct-wn[i])+abs(cg-wg[i])+abs(cb-wb[i]) > 0.10 or i % 63 == 0:
            ct, cg, cb = wn[i], wg[i], wb[i]
    return pnav


def yearly_return(nav_s):
    df = pd.DataFrame({'nav': nav_s.values, 'year': nav_s.index.year})
    yr = df.groupby('year')['nav'].agg(['first', 'last'])
    return (yr['last'] / yr['first'] - 1) * 100


def main():
    print("=" * 80)
    print("方向性F OOSパラドックス調査: なぜ d=1 は d=2 より OOS Sharpe が低いか")
    print("=" * 80)

    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    n = len(dates)
    dates_idx = pd.to_datetime(dates.values)

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

    # ── 各遅延でポートフォリオ構築
    navs = {}
    for d in [1, 2, 5]:
        nav_a2 = build_a2_nav(close, returns, raw, d)
        pnav = build_portfolio_3asset(nav_a2.values, g2, b3, wn_arr, wg_arr, wb_arr)
        navs[d] = pd.Series(pnav, index=dates_idx)

    # ─── 1. 年次リターン比較 ─────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("1. 年次リターン比較（d=1 vs d=2 の差が大きい年を特定）")
    print("─"*80)
    yr1 = yearly_return(navs[1])
    yr2 = yearly_return(navs[2])
    yr5 = yearly_return(navs[5])
    diff = yr1 - yr2

    print(f"  {'年':>5} {'d=1':>7} {'d=2':>7} {'d=5':>7} {'d1-d2':>7} {'注記':>20}")
    print(f"  {'-'*55}")
    for year in sorted(diff.index):
        if year not in yr2.index or year not in yr1.index:
            continue
        d_val = diff.loc[year] if year in diff.index else float('nan')
        note = ""
        if abs(d_val) >= 2.0:
            note = "← 大差 (d1 優位)" if d_val > 0 else "← 大差 (d2 優位)"
        y5 = yr5.loc[year] if year in yr5.index else float('nan')
        print(f"  {year:>5} {yr1.loc[year]:>6.1f}% {yr2.loc[year]:>6.1f}% {y5:>6.1f}% {d_val:>+6.1f}% {note}")

    # ─── 2. OOS期間（2021-2026）の詳細 ───────────────────────────────────────
    print("\n" + "─"*80)
    print("2. OOS期間（2021-2026）の月次リターン差（d=1 - d=2）")
    print("─"*80)
    oos_mask = navs[1].index >= '2021-01-01'
    for d_val in [1, 2, 5]:
        nav_oos = navs[d_val][oos_mask]
        ret_oos = nav_oos.pct_change().dropna()
        years_oos = len(nav_oos) / 252
        # OOS期間内の相対リターンで CAGR 計算（絶対NAV値ではなく）
        cagr_oos = float((nav_oos.iloc[-1] / nav_oos.iloc[0]) ** (1/years_oos) - 1)
        sh_oos = float(ret_oos.mean() / ret_oos.std() * np.sqrt(252))
        mdd_oos = float((nav_oos / nav_oos.cummax() - 1).min())
        print(f"  d={d_val}: CAGR={cagr_oos*100:.2f}%, Sharpe={sh_oos:.4f}, MaxDD={mdd_oos*100:.1f}%")

    # ─── 3. シグナル転換時のギャップ分析 ────────────────────────────────────
    print("\n" + "─"*80)
    print("3. シグナル転換日の翌日リターン（ギャップリスク）")
    print("─"*80)
    lev = rebalance_threshold(raw, 0.20)
    transitions = lev.diff().abs() > 0.01
    transition_days = transitions[transitions].index

    # 転換翌日（d=1で影響を受ける日）のリターン
    all_returns = returns.fillna(0)
    post_transition_rets = []
    for td in transition_days:
        # 転換日の翌日が d=1 の執行日
        idx = all_returns.index.get_loc(td)
        if idx + 1 < len(all_returns):
            next_ret = all_returns.iloc[idx + 1]
            direction = lev.loc[td]  # 転換後の方向（1=買い, 0=売り）
            post_transition_rets.append({'ret': next_ret, 'dir': direction, 'date': td})

    if post_transition_rets:
        ptr = pd.DataFrame(post_transition_rets)
        buy_rets = ptr[ptr['dir'] > 0]['ret']    # 買い転換翌日
        sell_rets = ptr[ptr['dir'] == 0]['ret']  # 売り転換翌日

        print(f"  総シグナル転換数: {len(ptr)}")
        print(f"  買い転換 ({len(buy_rets)}回) 翌日NASDAQ平均リターン: {buy_rets.mean()*100:+.2f}%")
        print(f"  売り転換 ({len(sell_rets)}回) 翌日NASDAQ平均リターン: {sell_rets.mean()*100:+.2f}%")
        print(f"  買い転換翌日 vs 通常日: {buy_rets.mean()*100:+.2f}% vs {all_returns.mean()*100:+.3f}%")
        print(f"  → d=1はこの翌日リターンをd=2より1日早く受ける")

        # 特に OOS 期間の転換
        ptr_oos = ptr[pd.to_datetime(ptr['date']) >= '2021-01-01']
        if len(ptr_oos) > 0:
            print(f"\n  OOS期間（2021-2026）の転換: {len(ptr_oos)}回")
            buy_oos = ptr_oos[ptr_oos['dir'] > 0]['ret']
            sell_oos = ptr_oos[ptr_oos['dir'] == 0]['ret']
            print(f"  OOS 買い転換翌日平均: {buy_oos.mean()*100:+.2f}%" if len(buy_oos)>0 else "  OOS 買い転換: なし")
            print(f"  OOS 売り転換翌日平均: {sell_oos.mean()*100:+.2f}%" if len(sell_oos)>0 else "  OOS 売り転換: なし")

    # ─── 4. MaxDD 悪化の原因特定 ────────────────────────────────────────────
    print("\n" + "─"*80)
    print("4. MaxDD悪化の原因（d=1: -38.5% vs d=2: -33.4%）")
    print("─"*80)
    for d_val in [1, 2]:
        nav_s = navs[d_val]
        dd_s = (nav_s / nav_s.cummax() - 1) * 100
        worst_date = dd_s.idxmin()
        worst_dd = dd_s.min()
        # DD開始を特定
        peak_before = nav_s[:worst_date].idxmax()
        print(f"  d={d_val}: MaxDD={worst_dd:.1f}% on {worst_date.date()} "
              f"(peak: {peak_before.date()})")
        # その期間の年次リターン
        year_wdd = worst_date.year
        if year_wdd in yr1.index:
            print(f"    該当年 ({year_wdd}) リターン: d=1={yr1.loc[year_wdd]:.1f}%, d=2={yr2.loc[year_wdd]:.1f}%")

    # ─── 5. ブートストラップ有意性検定（d=1 vs d=2）───────────────────────
    print("\n" + "─"*80)
    print("5. ブートストラップ有意性検定（d=1 CAGR > d=2 CAGR ?）")
    print("─"*80)
    np.random.seed(42)
    ret1 = navs[1].pct_change().dropna().values
    ret2 = navs[2].pct_change().dropna().values
    BLOCK = 63  # 四半期ブロック
    N_BOOT = 500

    obs_diff = navs[1].iloc[-1] ** (1/(n/252)) - navs[2].iloc[-1] ** (1/(n/252))
    boot_diffs = []
    num_blocks = len(ret1) // BLOCK
    for _ in range(N_BOOT):
        idx = np.random.choice(num_blocks - 1, size=num_blocks, replace=True)
        blocks1 = np.concatenate([ret1[i*BLOCK:(i+1)*BLOCK] for i in idx])[:len(ret1)]
        blocks2 = np.concatenate([ret2[i*BLOCK:(i+1)*BLOCK] for i in idx])[:len(ret2)]
        yrs = len(blocks1) / 252
        cagr1 = (np.prod(1 + blocks1)) ** (1/yrs) - 1
        cagr2 = (np.prod(1 + blocks2)) ** (1/yrs) - 1
        boot_diffs.append(cagr1 - cagr2)

    boot_diffs = np.array(boot_diffs)
    p_val = (boot_diffs <= 0).mean()  # p(d1 <= d2)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    print(f"  観測差 (d=1 CAGR - d=2 CAGR): {obs_diff*100:+.2f}%")
    print(f"  95%信頼区間: [{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]")
    print(f"  p値 (差 ≤ 0 の確率): {p_val:.3f}")
    print(f"  統計的有意性 (p<0.05): {'✅ 有意' if p_val < 0.05 else '❌ 非有意'}")

    # ─── 結論 ───────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("OOSパラドックス 総合結論")
    print("="*80)
    print("""
  【IS改善 +1.40% は統計的に有意か？】
  → ブートストラップ結果を参照

  【なぜ OOS で d=1 < d=2 か？】
  → 年次リターン比較より: OOS特定年の逆転が原因
  → ギャップリスク: d=1は転換翌日のギャップを受ける（d=2は1日分バッファあり）
  → 2021-2026はボラが高く、翌日の大幅ギャップ（FRB発表等）が多かった可能性

  【実務的な含意】
  → d=1採用は IS では +1.40% の改善
  → OOS Sharpe の差は小さい（0.8522 vs 0.8659 = -0.014差）
  → 52年 IS での統計的有意性が確認できれば採用価値あり
  → MaxDD の -5pp 悪化（-33.4% → -38.5%）は注意が必要
""")


if __name__ == '__main__':
    main()
