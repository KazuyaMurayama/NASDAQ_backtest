"""
Dyn 2x3x Regime Analysis: Forward return statistics by allocation bin.
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np
from backtest_engine import load_data, calc_dd_signal
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'NASDAQ_extended_to_2026.csv')
COST = 0.0086; DELAY = 2; THRESHOLD = 0.20
GOLD_2X_COST = 0.005; BOND_3X_COST = 0.0091
MIN_PCT = 0.01  # 1% minimum for bin inclusion

# ── Part 1: Build signals and portfolio NAV ──
def build_signals_and_nav():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    n = len(df)

    # A2 signals
    dd = calc_dd_signal(close, 0.82, 0.92)
    var = pd.Series(index=returns.index, dtype=float)
    var.iloc[0] = returns.iloc[:20].var()
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
    vp = calc_vix_proxy(returns); vma = vp.rolling(252).mean()
    vs = vp.rolling(252).std().replace(0, 0.001)
    vz = (vp - vma) / vs; vm = (1.0 - 0.25 * vz).clip(0.5, 1.15)
    raw = (dd * vt * slope * mom * vm).clip(0, 1.0).fillna(0)
    lev = rebalance_threshold(raw, THRESHOLD)

    # A2 NAV
    lr = returns * 3.0; dc = COST / 252
    dl = lev.shift(DELAY); sr = dl * (lr - dc); sr = sr.fillna(0)
    nav_a2 = (1 + sr).cumprod().values

    # Gold/Bond
    gold_1x = prepare_gold_data(dates)
    bond_1x = prepare_bond_data(dates)
    g2 = np.ones(n); b3 = np.ones(n)
    for i in range(1, n):
        gr = gold_1x[i] / gold_1x[i - 1] - 1 if gold_1x[i - 1] > 0 else 0
        g2[i] = g2[i - 1] * (1 + gr * 2 - GOLD_2X_COST / 252)
        br = bond_1x[i] / bond_1x[i - 1] - 1 if bond_1x[i - 1] > 0 else 0
        b3[i] = b3[i - 1] * (1 + br * 3 - BOND_3X_COST / 252)

    # w_nasdaq, rawLev
    raw_lev = raw.values
    vz_vals = vz.fillna(0).values
    w_nasdaq = np.clip(0.55 + 0.25 * raw_lev - 0.10 * np.maximum(vz_vals, 0), 0.30, 0.90)

    # Portfolio NAV (daily rebuild)
    wn = w_nasdaq; wg = (1 - wn) * 0.50; wb = (1 - wn) * 0.50
    port_nav = np.ones(n)
    cur_wn, cur_wg, cur_wb = wn[0], wg[0], wb[0]
    for i in range(1, n):
        rn = nav_a2[i] / nav_a2[i - 1] - 1 if nav_a2[i - 1] > 0 else 0
        rg = g2[i] / g2[i - 1] - 1 if g2[i - 1] > 0 else 0
        rb = b3[i] / b3[i - 1] - 1 if b3[i - 1] > 0 else 0
        port_ret = cur_wn * rn + cur_wg * rg + cur_wb * rb
        port_nav[i] = port_nav[i - 1] * (1 + port_ret)
        cur_wn *= (1 + rn); cur_wg *= (1 + rg); cur_wb *= (1 + rb)
        total = cur_wn + cur_wg + cur_wb
        if total > 0:
            cur_wn /= total; cur_wg /= total; cur_wb /= total
        drift = abs(cur_wn - wn[i]) + abs(cur_wg - wg[i]) + abs(cur_wb - wb[i])
        if drift > 0.10 or (i % 63 == 0):
            cur_wn, cur_wg, cur_wb = wn[i], wg[i], wb[i]

    return {
        'n': n, 'dates': dates, 'w_nasdaq': w_nasdaq, 'raw_lev': raw_lev,
        'port_nav': port_nav,
    }


# ── Part 2: Compute forward returns and bin statistics ──
def compute_stats(ctx):
    n = ctx['n']; nav = ctx['port_nav']
    w_nasdaq = ctx['w_nasdaq']; raw_lev = ctx['raw_lev']

    w_bins = [(0.30, 0.50, '30-50%'), (0.50, 0.70, '50-70%'), (0.70, 0.901, '70-90%')]
    r_bins = [(0.00, 0.005, '0%(CASH)'), (0.005, 0.30, '1-30%'),
              (0.30, 0.60, '30-60%'), (0.60, 0.85, '60-85%'), (0.85, 1.01, '85-100%')]
    horizons = [5, 10]

    # Forward returns
    fwd = {}
    for h in horizons:
        fr = np.full(n, np.nan)
        for i in range(n - h):
            fr[i] = nav[i + h] / nav[i] - 1
        fwd[h] = fr

    results = []
    for wlo, whi, wlabel in w_bins:
        for rlo, rhi, rlabel in r_bins:
            mask = (w_nasdaq >= wlo) & (w_nasdaq < whi) & (raw_lev >= rlo) & (raw_lev < rhi)
            cnt = mask.sum()
            pct = cnt / n
            if pct < MIN_PCT:
                continue

            # Effective allocation (midpoint)
            w_mid = (wlo + min(whi, 0.90)) / 2
            r_mid = (rlo + min(rhi, 1.0)) / 2
            nq_eff = w_mid * r_mid * 100
            g_eff = (1 - w_mid) * 50
            b_eff = (1 - w_mid) * 50
            c_eff = w_mid * (1 - r_mid) * 100

            for h in horizons:
                rets = fwd[h][mask]
                rets = rets[~np.isnan(rets)]
                if len(rets) < 20:
                    continue

                # CAGR annualized (geometric mean)
                geo_mean = np.exp(np.mean(np.log(1 + rets))) - 1
                cagr_ann = (1 + geo_mean) ** (252 / h) - 1

                results.append({
                    'w_bin': wlabel, 'r_bin': rlabel,
                    'horizon': h, 'samples': len(rets),
                    'pct_of_total': pct * 100,
                    'NQ_eff': nq_eff, 'Gold_eff': g_eff,
                    'Bond_eff': b_eff, 'Cash_eff': c_eff,
                    'CAGR_ann': cagr_ann * 100,
                    'median': np.median(rets) * 100,
                    'p10': np.percentile(rets, 10) * 100,
                    'p25': np.percentile(rets, 25) * 100,
                    'p75': np.percentile(rets, 75) * 100,
                    'pos_pct': (rets > 0).mean() * 100,
                })

    rdf = pd.DataFrame(results)
    out = os.path.join(BASE_DIR, 'regime_analysis_stats.csv')
    rdf.to_csv(out, index=False)
    print(f"Saved: {out} ({len(rdf)} rows)")
    return rdf


# ── Part 3: Sanity checks ──
def run_checks(rdf, ctx):
    print("\n=== Sanity Checks ===")
    n = ctx['n']

    # Check 1: Total samples across bins (for each horizon)
    for h in [5, 10]:
        hdf = rdf[rdf['horizon'] == h]
        total_samples = hdf['samples'].sum()
        total_pct = hdf['pct_of_total'].sum()
        excluded_pct = 100 - total_pct
        print(f"  Horizon {h}d: {len(hdf)} bins, {total_samples} samples "
              f"({total_pct:.1f}% covered, {excluded_pct:.1f}% excluded)")

    # Check 2: Overall CAGR consistency
    nav = ctx['port_nav']
    yrs = n / 252
    actual_cagr = (nav[-1] ** (1 / yrs) - 1) * 100
    print(f"  Actual portfolio CAGR: {actual_cagr:.2f}%")

    # Check 3: Monotonicity — higher rawLev bins should have higher median returns
    for h in [5, 10]:
        hdf = rdf[rdf['horizon'] == h]
        for wlabel in ['30-50%', '50-70%', '70-90%']:
            sub = hdf[hdf['w_bin'] == wlabel].sort_values('r_bin')
            if len(sub) > 1:
                meds = sub['median'].values
                monotone = all(meds[i] <= meds[i + 1] for i in range(len(meds) - 1))
                print(f"  {h}d {wlabel}: median monotone={monotone} "
                      f"({', '.join(f'{m:.2f}' for m in meds)})")

    print("=== Checks Done ===\n")


# ── Part 4: Generate MD report ──
def generate_md(rdf):
    md = """# Dyn 2x3x 配分レジーム分析

## 戦略の仕組み

Dyn 2x3x は以下の **2つの調整軸** でポートフォリオを動的に管理します。

### 調整軸

| 軸 | 内容 | 範囲 | 決定要因 |
|----|------|------|----------|
| **① NASDAQ枠 (w\\_nasdaq)** | NASDAQ 3x vs Gold2x+Bond3x の配分 | 30%〜90% | rawLeverage + vix\\_z |
| **② rawLeverage** | NASDAQ枠内の投資 vs CASH 比率 | 0%〜100% | A2シグナル（DD×VT×Slope×Mom×VIX）|

### 構造の特徴

- **Gold 2x / Bond 3x にはOFFスイッチがない**（常にフル投資）
- CASH が発生するのは **NASDAQ枠の未投資分のみ**
- 暴落時は NASDAQ → 0% になり、Gold+Bond の比率が **自動的に増加**

```
ポートフォリオ = w_nasdaq × [rawLev × NASDAQ3x + (1-rawLev) × CASH]
               + (1 - w_nasdaq) × [50% Gold2x + 50% Bond3x]
```

---

## 各レジームの実効配分

| ① NASDAQ枠 | ② rawLev | NQ 3x | Gold 2x | Bond 3x | CASH | 局面 |
|-------------|----------|-------|---------|---------|------|------|
| 30-50% | 0% (CASH) | 0% | 30% | 30% | 40% | 暴落退避 |
| 50-70% | 0% (CASH) | 0% | 20% | 20% | 60% | 下落警戒 |
| 50-70% | 1-30% | 9% | 20% | 20% | 51% | 回復初期 |
| 50-70% | 30-60% | 27% | 20% | 20% | 33% | 回復途上 |
| 70-90% | 60-85% | 58% | 10% | 10% | 22% | 上昇基調 |
| 70-90% | 85-100% | 74% | 10% | 10% | 6% | 強気フル投資 |

---

## サンプル数分布（1974-2026, 全13,169日）

"""

    # Sample distribution table
    h5 = rdf[rdf['horizon'] == 5]
    md += "| ① NASDAQ枠 | ② rawLev | 日数 | 全体比率 |\n"
    md += "|-------------|----------|-----:|--------:|\n"
    for _, r in h5.iterrows():
        md += f"| {r['w_bin']} | {r['r_bin']} | {int(r['samples']):,} | {r['pct_of_total']:.1f}% |\n"
    md += f"\n> 全体の1%未満（≒132日未満）のビンは除外\n"

    # Forward return tables
    for h in [5, 10]:
        hdf = rdf[rdf['horizon'] == h]
        md += f"""
---

## フォワードリターン分析（{h}営業日後）

| ① NASDAQ枠 | ② rawLev | CAGR年率 | 中央値 | 下位10% | 下位25% | 上位25% | プラス確率 |
|-------------|----------|--------:|------:|-------:|-------:|-------:|---------:|\n"""

        for _, r in hdf.iterrows():
            md += (f"| {r['w_bin']} | {r['r_bin']} "
                   f"| {r['CAGR_ann']:+.1f}% "
                   f"| {r['median']:+.2f}% "
                   f"| {r['p10']:+.2f}% "
                   f"| {r['p25']:+.2f}% "
                   f"| {r['p75']:+.2f}% "
                   f"| {r['pos_pct']:.0f}% |\n")

    # Findings
    md += """
---

## 主要な発見

### 1. 強気フル投資（70-90% × 85-100%）が最も高頻度・高リターン
"""
    h5_bull = h5[(h5['w_bin'] == '70-90%') & (h5['r_bin'] == '85-100%')]
    if len(h5_bull) > 0:
        b = h5_bull.iloc[0]
        md += f"- 全日数の **{b['pct_of_total']:.1f}%** を占める最頻レジーム\n"
        md += f"- 5日後CAGR年率 **{b['CAGR_ann']:+.1f}%**、プラス確率 **{b['pos_pct']:.0f}%**\n"

    md += "\n### 2. 暴落退避時のGold+Bondの防御効果\n"
    h5_cash = h5[h5['r_bin'] == '0%(CASH)']
    for _, r in h5_cash.iterrows():
        md += f"- {r['w_bin']}枠: 5日後中央値 {r['median']:+.2f}%, 下位10% {r['p10']:+.2f}%\n"

    md += "\n### 3. リスク・リターンのトレードオフ\n"
    md += "- rawLeverageが高いほど上位25%のリターンが大きいが、下位10%のリスクも拡大\n"
    md += "- Gold+Bond比率が高いレジームは下位10%が緩和され、下落防御に寄与\n"

    md += """
---

## 計算前提

- ポートフォリオ全体の日次NAVからフォワードリターンを算出
- CAGR年率 = 各ビンのリターンの幾何平均を252日に年率換算
- 経費率: TQQQ 0.86%, Gold ETN 0.5%, TMF 0.91%（日割り控除済み）
- 実行遅延: 2営業日

---

*Generated: 2026-04-04*
"""

    out = os.path.join(BASE_DIR, 'REGIME_ANALYSIS_REPORT.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Saved: {out}")
    return out


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if part in ['1', 'signals', 'all']:
        print("Part 1: Building signals and NAV...")
        ctx = build_signals_and_nav()
        # Save context for reuse
        np.savez(os.path.join(BASE_DIR, 'regime_ctx.npz'),
                 w_nasdaq=ctx['w_nasdaq'], raw_lev=ctx['raw_lev'],
                 port_nav=ctx['port_nav'], n=np.array([ctx['n']]))
        print(f"  Saved context ({ctx['n']} days)")

    if part in ['2', 'stats', 'all']:
        if 'ctx' not in dir() or ctx is None:
            data = np.load(os.path.join(BASE_DIR, 'regime_ctx.npz'))
            ctx = {'w_nasdaq': data['w_nasdaq'], 'raw_lev': data['raw_lev'],
                   'port_nav': data['port_nav'], 'n': int(data['n'][0])}
        print("Part 2: Computing stats...")
        rdf = compute_stats(ctx)

    if part in ['3', 'check', 'all']:
        if 'ctx' not in dir():
            data = np.load(os.path.join(BASE_DIR, 'regime_ctx.npz'))
            ctx = {'w_nasdaq': data['w_nasdaq'], 'raw_lev': data['raw_lev'],
                   'port_nav': data['port_nav'], 'n': int(data['n'][0])}
        if 'rdf' not in dir():
            rdf = pd.read_csv(os.path.join(BASE_DIR, 'regime_analysis_stats.csv'))
        print("Part 3: Sanity checks...")
        run_checks(rdf, ctx)

    if part in ['4', 'report', 'all']:
        if 'rdf' not in dir():
            rdf = pd.read_csv(os.path.join(BASE_DIR, 'regime_analysis_stats.csv'))
        print("Part 4: Generating MD report...")
        generate_md(rdf)

    print("\n=== DONE ===")
