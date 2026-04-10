"""
7 Strategy Comparison — 代表的な7戦略の比較
============================================
条件: delay=2営業日, cost=0.86%/年 (TQQQ), NASDAQ_extended_to_2026.csv
"""
import sys
import os
import types

# Fake multitasking module (workaround for missing C extension)
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None
m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f)
m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest_engine import (
    load_data, calc_dd_signal, calc_metrics, calc_ewma_vol,
    strategy_baseline_dd_only
)
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol,
    strategy_ens2_asym_slope
)
from test_delay_robust import (
    rebalance_threshold, run_backtest_realistic, count_rebalances,
    calc_momentum_decel_mult
)
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import (
    prepare_gold_data, prepare_bond_data
)
from test_dynamic_portfolio import (
    alloc_dyn_hybrid, build_dynamic_portfolio, build_static_portfolio
)

# =============================================================================
# Parameters (user-specified)
# =============================================================================
ANNUAL_COST = 0.0086      # TQQQ expense ratio
DEFAULT_DELAY = 2          # 2 business day execution delay
BASE_LEVERAGE = 3.0
REBALANCE_THRESHOLD = 0.20
OOS_SPLIT_DATE = '2021-05-07'

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'NASDAQ_extended_to_2026.csv')

# =============================================================================
# Helpers
# =============================================================================
def run_backtest_custom(close, leverage, delay=DEFAULT_DELAY,
                        annual_cost=ANNUAL_COST, base_leverage=BASE_LEVERAGE):
    """Run backtest with custom parameters."""
    returns = close.pct_change()
    leveraged_returns = returns * base_leverage
    daily_cost = annual_cost / 252
    delayed_leverage = leverage.shift(delay)
    strategy_returns = delayed_leverage * (leveraged_returns - daily_cost)
    strategy_returns = strategy_returns.fillna(0)
    nav = (1 + strategy_returns).cumprod()
    return nav, strategy_returns


def calc_full_and_oos_metrics(nav, strat_ret, dd_signal, dates, name):
    """Calculate full-period and OOS metrics."""
    # Full-period metrics
    m = calc_metrics(nav, strat_ret, dd_signal, dates)

    # OOS metrics
    split_idx = dates[dates >= OOS_SPLIT_DATE].index[0]
    nav_oos = nav.iloc[split_idx:] / nav.iloc[split_idx]
    ret_oos = strat_ret.iloc[split_idx:]
    oos_days = len(nav_oos)
    oos_years = oos_days / 252
    oos_cagr = (nav_oos.iloc[-1] ** (1 / oos_years)) - 1 if oos_years > 0 else 0
    oos_sharpe = (ret_oos.mean() * 252) / (ret_oos.std() * np.sqrt(252)) if ret_oos.std() > 0 else 0

    return {
        'Strategy': name,
        'CAGR': m['CAGR'],
        'Sharpe': m['Sharpe'],
        'MaxDD': m['MaxDD'],
        'Worst5Y': m['Worst5Y'],
        'WinRate': m['WinRate'],
        'Trades': m['Trades'],
        'OOS_CAGR': oos_cagr,
        'OOS_Sharpe': oos_sharpe,
    }


def calc_portfolio_full_and_oos(nav_array, dates, name):
    """Calculate metrics for a portfolio NAV array."""
    nav = pd.Series(nav_array, index=dates.index)
    returns = nav.pct_change().fillna(0)
    total_years = len(nav) / 252

    cagr = (nav.iloc[-1] ** (1 / total_years)) - 1
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_dd = drawdown.min()
    annual_ret = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

    # Worst 5Y
    if len(nav) >= 252 * 5:
        nav_5y = nav.shift(252 * 5)
        rolling_5y = (nav / nav_5y) ** (1/5) - 1
        worst_5y = rolling_5y.min()
    else:
        worst_5y = np.nan

    # Win rate (annual)
    nav_df = pd.DataFrame({'nav': nav.values, 'date': dates.values})
    nav_df['year'] = pd.to_datetime(nav_df['date']).dt.year
    yearly_nav = nav_df.groupby('year')['nav'].last()
    annual_returns = yearly_nav.pct_change().dropna()
    win_rate = (annual_returns > 0).mean() if len(annual_returns) > 0 else 0

    # OOS
    split_idx = dates[dates >= OOS_SPLIT_DATE].index[0]
    nav_oos = nav.iloc[split_idx:] / nav.iloc[split_idx]
    ret_oos = returns.iloc[split_idx:]
    oos_years = len(nav_oos) / 252
    oos_cagr = (nav_oos.iloc[-1] ** (1 / oos_years)) - 1 if oos_years > 0 else 0
    oos_sharpe = (ret_oos.mean() * 252) / (ret_oos.std() * np.sqrt(252)) if ret_oos.std() > 0 else 0

    return {
        'Strategy': name,
        'CAGR': cagr,
        'Sharpe': sharpe,
        'MaxDD': max_dd,
        'Worst5Y': worst_5y,
        'WinRate': win_rate,
        'Trades': 'N/A',
        'OOS_CAGR': oos_cagr,
        'OOS_Sharpe': oos_sharpe,
    }


# =============================================================================
# Strategy Implementations
# =============================================================================
def strategy_bh_1x(close):
    """Buy & Hold 1x — no leverage, no cost."""
    leverage = pd.Series(1.0, index=close.index)
    nav = close / close.iloc[0]
    strat_ret = close.pct_change().fillna(0)
    dd_signal = pd.Series(1.0, index=close.index)
    return nav, strat_ret, dd_signal, 'Buy & Hold 1x'


def strategy_bh_3x(close):
    """Buy & Hold 3x — 3x leverage with TQQQ cost."""
    leverage = pd.Series(1.0, index=close.index)
    nav, strat_ret = run_backtest_custom(close, leverage, delay=0,
                                          annual_cost=ANNUAL_COST, base_leverage=3.0)
    dd_signal = pd.Series(1.0, index=close.index)
    return nav, strat_ret, dd_signal, 'Buy & Hold 3x'


def strategy_dd_only(close):
    """DD(-18/92) Only — drawdown control, no VT."""
    returns = close.pct_change()
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    lev = rebalance_threshold(dd_signal, REBALANCE_THRESHOLD)
    nav, strat_ret = run_backtest_custom(close, lev)
    return nav, strat_ret, dd_signal, 'DD(-18/92) Only'


def strategy_ens2_asym_slope_custom(close):
    """Ens2(Asym+Slope) — AsymEWMA + SlopeMult."""
    returns = close.pct_change()
    lev, dd_signal = strategy_ens2_asym_slope(close, returns, 0.82, 0.92,
                                               0.25, 20, 5, 1.0)
    lev = rebalance_threshold(lev, REBALANCE_THRESHOLD)
    nav, strat_ret = run_backtest_custom(close, lev)
    return nav, strat_ret, dd_signal, 'Ens2(Asym+Slope)'


def strategy_a2_custom(close):
    """A2 (VIX+MD60) — VIX Mean Reversion + MomDecel(60/180)."""
    returns = close.pct_change()
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    lev = rebalance_threshold(raw_leverage, REBALANCE_THRESHOLD)
    nav, strat_ret = run_backtest_custom(close, lev)
    return nav, strat_ret, dd_signal, 'A2 (VIX+MD60)'


def get_a2_signals_custom(close, dates):
    """Get A2 raw leverage and signals for Dyn-Hybrid."""
    returns = close.pct_change()
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    lev = rebalance_threshold(raw_leverage, REBALANCE_THRESHOLD)
    nav, strat_ret = run_backtest_custom(close, lev)

    return {
        'nav': nav.values,
        'ret': strat_ret.values,
        'raw_leverage': raw_leverage.values,
        'dd_signal': dd_signal.values,
        'vix_z': vix_z.fillna(0).values,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 100)
    print("7 STRATEGY COMPARISON")
    print(f"Conditions: delay={DEFAULT_DELAY}d, cost={ANNUAL_COST*100:.2f}%, "
          f"base_lev={BASE_LEVERAGE}x, threshold={REBALANCE_THRESHOLD*100:.0f}%")
    print("=" * 100)

    df = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} rows)")
    print(f"OOS split: {OOS_SPLIT_DATE}\n")

    results = []

    # ---- Strategy 1: Buy & Hold 1x ----
    print("[1/7] Buy & Hold 1x...")
    nav, ret, dd, name = strategy_bh_1x(close)
    r = calc_full_and_oos_metrics(nav, ret, dd, dates, name)
    results.append(r)
    print(f"  Sharpe={r['Sharpe']:.4f}, CAGR={r['CAGR']*100:.2f}%")

    # ---- Strategy 2: Buy & Hold 3x ----
    print("[2/7] Buy & Hold 3x...")
    nav, ret, dd, name = strategy_bh_3x(close)
    r = calc_full_and_oos_metrics(nav, ret, dd, dates, name)
    results.append(r)
    print(f"  Sharpe={r['Sharpe']:.4f}, CAGR={r['CAGR']*100:.2f}%")

    # ---- Strategy 3: DD(-18/92) Only ----
    print("[3/7] DD(-18/92) Only...")
    nav, ret, dd, name = strategy_dd_only(close)
    r = calc_full_and_oos_metrics(nav, ret, dd, dates, name)
    results.append(r)
    print(f"  Sharpe={r['Sharpe']:.4f}, CAGR={r['CAGR']*100:.2f}%")

    # ---- Strategy 4: Ens2(Asym+Slope) ----
    print("[4/7] Ens2(Asym+Slope)...")
    nav, ret, dd, name = strategy_ens2_asym_slope_custom(close)
    r = calc_full_and_oos_metrics(nav, ret, dd, dates, name)
    results.append(r)
    print(f"  Sharpe={r['Sharpe']:.4f}, CAGR={r['CAGR']*100:.2f}%")

    # ---- Strategy 5: A2 (VIX+MD60) ----
    print("[5/7] A2 (VIX+MD60)...")
    nav, ret, dd, name = strategy_a2_custom(close)
    r = calc_full_and_oos_metrics(nav, ret, dd, dates, name)
    results.append(r)
    print(f"  Sharpe={r['Sharpe']:.4f}, CAGR={r['CAGR']*100:.2f}%")

    # ---- Strategies 6 & 7: Dyn-Hybrid (need Gold/Bond data) ----
    print("\n[6-7/7] Dyn-Hybrid strategies (fetching Gold/Bond data)...")
    try:
        signals = get_a2_signals_custom(close, dates)
        gold_prices = prepare_gold_data(dates)
        bond_nav = prepare_bond_data(dates)
        print(f"  Gold: first={gold_prices[0]:.2f}, last={gold_prices[-1]:.2f}")
        print(f"  Bond NAV: first={bond_nav[0]:.2f}, last={bond_nav[-1]:.2f}")

        # Strategy 6: Dyn-Hybrid Static Best (50/25/25)
        print("  [6/7] Dyn-Hybrid Static (50/25/25)...")
        static_nav = build_static_portfolio(signals['nav'], gold_prices, bond_nav,
                                             0.50, 0.25, 0.25)
        r = calc_portfolio_full_and_oos(static_nav, dates, 'Dyn-Hybrid Static (50/25/25) *')
        results.append(r)
        print(f"    Sharpe={r['Sharpe']:.4f}, CAGR={r['CAGR']*100:.2f}%")

        # Strategy 7: Dyn-Hybrid (lev+vix)
        print("  [7/7] Dyn-Hybrid (lev+vix)...")
        wn, wg, wb = alloc_dyn_hybrid(signals)
        dyn_nav = build_dynamic_portfolio(signals['nav'], gold_prices, bond_nav,
                                           wn, wg, wb)
        r = calc_portfolio_full_and_oos(dyn_nav, dates, 'Dyn-Hybrid (lev+vix) *')
        results.append(r)
        print(f"    Sharpe={r['Sharpe']:.4f}, CAGR={r['CAGR']*100:.2f}%")

    except Exception as e:
        print(f"  ERROR: Could not compute Dyn-Hybrid: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Print Results Table
    # =========================================================================
    print(f"\n{'=' * 130}")
    print("RESULTS TABLE (sorted by Sharpe)")
    print("=" * 130)

    results_sorted = sorted(results, key=lambda x: -x['Sharpe'])

    print(f"\n{'#':<3} {'Strategy':<32} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} "
          f"{'W5Y':>7} {'WinRate':>8} {'Trades':>7} {'OOS CAGR':>9} {'OOS Sh':>7}")
    print("-" * 110)

    for i, r in enumerate(results_sorted, 1):
        trades_str = str(int(r['Trades'])) if isinstance(r['Trades'], (int, float)) and r['Trades'] != 'N/A' else str(r['Trades'])
        w5y_str = f"{r['Worst5Y']*100:+.2f}%" if not pd.isna(r.get('Worst5Y', np.nan)) else "N/A"
        print(f"{i:<3} {r['Strategy']:<32} {r['CAGR']*100:>6.2f}% {r['Sharpe']:>7.4f} "
              f"{r['MaxDD']*100:>7.2f}% {w5y_str:>7} {r['WinRate']*100:>7.1f}% "
              f"{trades_str:>7} {r['OOS_CAGR']*100:>8.2f}% {r['OOS_Sharpe']:>7.4f}")

    # Save raw results
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'strategy_comparison_results.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # =========================================================================
    # Generate Markdown
    # =========================================================================
    generate_markdown(results_sorted, dates)

    return results


def generate_markdown(results, dates):
    """Generate STRATEGY_COMPARISON.md"""
    md = f"""# 3x NASDAQ投資戦略 7戦略比較

## 検証概要

| 項目 | 値 |
|------|-----|
| **データ期間** | {dates.iloc[0].date()} 〜 {dates.iloc[-1].date()}（{len(dates)/252:.0f}年間） |
| **対象指数** | NASDAQ Composite Index |
| **ベースレバレッジ** | 3倍（日次リバランス） |
| **経費率** | 年0.86%（TQQQ準拠） |
| **実行遅延** | 2営業日 |
| **リバランス閾値** | 20%（レバレッジ変動が20%超で実行） |
| **OOS期間** | 2021年5月7日以降 |

---

## 比較表

| # | 戦略 | CAGR | Sharpe | MaxDD | Worst5Y | Win Rate | Trades | OOS CAGR | OOS Sharpe |
|---|------|------|--------|-------|---------|----------|--------|----------|------------|
"""

    for i, r in enumerate(results, 1):
        trades_str = str(int(r['Trades'])) if isinstance(r['Trades'], (int, float)) and r['Trades'] != 'N/A' else str(r['Trades'])
        w5y_str = f"{r['Worst5Y']*100:+.2f}%" if not pd.isna(r.get('Worst5Y', np.nan)) else "N/A"
        md += (f"| {i} | {r['Strategy']} | {r['CAGR']*100:.2f}% | {r['Sharpe']:.3f} | "
               f"{r['MaxDD']*100:.2f}% | {w5y_str} | {r['WinRate']*100:.1f}% | "
               f"{trades_str} | {r['OOS_CAGR']*100:.2f}% | {r['OOS_Sharpe']:.3f} |\n")

    md += """
> **\\*** 3資産ポートフォリオ（NASDAQ 3x + Gold 447A + Bond 2621）。他の戦略はNASDAQ単一資産。

---

## 評価指標の定義

| 指標 | 定義 |
|------|------|
| **CAGR** | 年率複利成長率。全期間の年率リターン |
| **Sharpe** | (年率リターン) / (年率ボラティリティ)。リスク調整後リターン |
| **MaxDD** | ピークからの最大下落率。ドローダウンの深さ |
| **Worst5Y** | ローリング5年CAGRの最小値。最悪の5年間のパフォーマンス |
| **Win Rate** | 年次リターンがプラスの割合 |
| **Trades** | HOLD↔CASH状態遷移の総回数 |
| **OOS CAGR** | Out-of-Sample期間（2021/5以降）のCAGR |
| **OOS Sharpe** | Out-of-Sample期間のSharpe Ratio |

---

## 各戦略の概要

### 1. Buy & Hold 1x（ベンチマーク）
レバレッジなしのNASDAQ Composite指数をそのまま保有。全戦略の比較基準。

### 2. Buy & Hold 3x（ベンチマーク）
NASDAQ 3倍レバレッジ（TQQQ相当）を無管理で保有。レバレッジの恩恵とリスクを示す。

### 3. DD(-18/92) Only
ドローダウン制御のみ。200日高値から-18%下落でCASH退避、92%回復でHOLD復帰。VTなし。

### 4. Ens2(Asym+Slope)
DD制御 + 非対称EWMAボラティリティターゲティング（下落時Span=5/上昇時Span=20） + MA200傾き乗数。旧FINAL推奨戦略。

### 5. A2 (VIX+MD60)
Ens2ベースに VIX Proxy Mean Reversion（恐怖指数Z-scoreによるレバレッジ調整）+ MomDecel(60/180)（モメンタム減速検出）を追加。最新の単一資産最良戦略。

### 6. Dyn-Hybrid Static (50/25/25) *
A2戦略のNAVに Gold(447A) と Bond(2621) を加えた3資産ポートフォリオ。静的配分（NASDAQ 50% / Gold 25% / Bond 25%）、四半期リバランス。

### 7. Dyn-Hybrid (lev+vix) *
A2の raw_leverage と vix_z を使って3資産の配分比率を動的に調整。リスクオン時はNASDAQ比率を上げ、リスクオフ時はGold/Bondに退避。

---

## 計算前提

- **レバレッジ**: 日次3倍リバランス（TQQQ相当）
- **経費率**: 年0.86%（投資時のみ日割り控除）
- **現金保有時**: リターン0%
- **ポジション反映遅延**: 2営業日（TQQQの約定→受渡）
- **取引カウント**: バイナリ状態遷移のみ（HOLD↔CASH）
- **Dyn-Hybrid**: Gold=GC=F + 月次補間、Bond=^TNX合成トータルリターン

---

*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*
"""

    md_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'STRATEGY_COMPARISON.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Markdown saved to {md_path}")


if __name__ == '__main__':
    results = main()
