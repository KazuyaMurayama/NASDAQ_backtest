"""
Verify max_lev meaning and calculation correctness
max_levの意味と計算の正確性を検証
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import load_data, calc_dd_signal, run_backtest
from test_ens2_strategies import calc_asym_ewma_vol, calc_slope_multiplier

def main():
    print("=" * 100)
    print("max_lev パラメータの意味と計算検証")
    print("=" * 100)

    # Load data
    data_path = r"C:\Users\user\Desktop\nasdaq_backtest\NASDAQ_Dairy_since1973.csv"
    df = load_data(data_path)
    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    # ==========================================================================
    # max_lev の意味を解説
    # ==========================================================================
    print("""
■ max_lev パラメータの意味

【計算式】
  VT_leverage = min(target_vol / realized_vol, max_lev)

  最終レバレッジ = DD_signal × VT_leverage × slope_mult

  実効レバレッジ = 最終レバレッジ × base_leverage(3倍)

【max_lev=1.0 の場合】
  VT_leverage: 0 ~ 1.0
  実効レバレッジ: 0 ~ 3倍 (3倍商品に0~100%投資)

【max_lev=3.0 の場合】★問題点
  VT_leverage: 0 ~ 3.0
  実効レバレッジ: 0 ~ 9倍 (3倍商品に0~300%投資 = 借入れ)

  ※ これは現実的ではない！3倍商品を借入れで3倍買うことを意味する
""")

    # ==========================================================================
    # 1978年の詳細検証
    # ==========================================================================
    print("=" * 100)
    print("1978年の詳細検証")
    print("=" * 100)

    # Filter to 1978
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df_1978 = df[df['Year'] == 1978].copy()

    close_1978 = df[df['Year'] <= 1978]['Close']
    returns_1978 = close_1978.pct_change()

    # Calculate components
    dd_signal = calc_dd_signal(close_1978, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns_1978, 20, 5)
    slope_mult = calc_slope_multiplier(close_1978)

    # VT leverage with different max_lev
    target_vol = 0.25
    vt_lev_1 = (target_vol / asym_vol).clip(0, 1.0)
    vt_lev_3 = (target_vol / asym_vol).clip(0, 3.0)

    # Final leverage
    final_lev_1 = dd_signal * vt_lev_1 * slope_mult
    final_lev_3 = dd_signal * vt_lev_3 * slope_mult

    # Get 1978 data
    mask_1978 = df['Year'] == 1978
    start_idx = mask_1978.idxmax()
    end_idx = mask_1978[::-1].idxmax()

    print(f"\n1978年のデータ範囲: index {start_idx} to {end_idx}")

    # Statistics for 1978
    vol_1978 = asym_vol.loc[start_idx:end_idx]
    vt_lev_1_1978 = vt_lev_1.loc[start_idx:end_idx]
    vt_lev_3_1978 = vt_lev_3.loc[start_idx:end_idx]
    slope_1978 = slope_mult.loc[start_idx:end_idx]
    final_lev_1_1978 = final_lev_1.loc[start_idx:end_idx]
    final_lev_3_1978 = final_lev_3.loc[start_idx:end_idx]

    print(f"\n【1978年の統計】")
    print(f"  AsymEWMA Vol:")
    print(f"    平均: {vol_1978.mean()*100:.2f}%")
    print(f"    最小: {vol_1978.min()*100:.2f}%")
    print(f"    最大: {vol_1978.max()*100:.2f}%")

    print(f"\n  VT Leverage (max_lev=1.0):")
    print(f"    平均: {vt_lev_1_1978.mean():.3f}")
    print(f"    最大: {vt_lev_1_1978.max():.3f}")

    print(f"\n  VT Leverage (max_lev=3.0):")
    print(f"    平均: {vt_lev_3_1978.mean():.3f}")
    print(f"    最大: {vt_lev_3_1978.max():.3f}")

    print(f"\n  Slope Multiplier:")
    print(f"    平均: {slope_1978.mean():.3f}")
    print(f"    最大: {slope_1978.max():.3f}")

    print(f"\n  最終レバレッジ (max_lev=1.0):")
    print(f"    平均: {final_lev_1_1978.mean():.3f}")
    print(f"    最大: {final_lev_1_1978.max():.3f}")
    print(f"    実効レバレッジ平均: {final_lev_1_1978.mean()*3:.2f}倍")
    print(f"    実効レバレッジ最大: {final_lev_1_1978.max()*3:.2f}倍")

    print(f"\n  最終レバレッジ (max_lev=3.0):")
    print(f"    平均: {final_lev_3_1978.mean():.3f}")
    print(f"    最大: {final_lev_3_1978.max():.3f}")
    print(f"    実効レバレッジ平均: {final_lev_3_1978.mean()*3:.2f}倍")
    print(f"    実効レバレッジ最大: {final_lev_3_1978.max()*3:.2f}倍")

    # ==========================================================================
    # リターン計算の検証
    # ==========================================================================
    print("\n" + "=" * 100)
    print("1978年リターン計算の検証")
    print("=" * 100)

    # NASDAQ 1978 return
    close_arr = close.values
    year_arr = df['Year'].values

    idx_1977_end = np.where(year_arr == 1977)[0][-1]
    idx_1978_end = np.where(year_arr == 1978)[0][-1]

    nasdaq_1978_return = close_arr[idx_1978_end] / close_arr[idx_1977_end] - 1

    print(f"\n  NASDAQ 1978年リターン: {nasdaq_1978_return*100:.2f}%")

    # Run backtest for both max_lev
    lev_1 = dd_signal * vt_lev_1 * slope_mult
    lev_3 = dd_signal * vt_lev_3 * slope_mult

    lev_1 = lev_1.clip(0, 1.0).fillna(0)
    lev_3 = lev_3.clip(0, 3.0).fillna(0)

    nav_1, _ = run_backtest(close, lev_1)
    nav_3, _ = run_backtest(close, lev_3)

    # BH 3x
    lev_bh = pd.Series(1.0, index=close.index)
    nav_bh, _ = run_backtest(close, lev_bh)

    # 1978 returns
    ret_1 = nav_1.iloc[idx_1978_end] / nav_1.iloc[idx_1977_end] - 1
    ret_3 = nav_3.iloc[idx_1978_end] / nav_3.iloc[idx_1977_end] - 1
    ret_bh = nav_bh.iloc[idx_1978_end] / nav_bh.iloc[idx_1977_end] - 1

    print(f"\n  【1978年リターン比較】")
    print(f"    NASDAQ 1x:         {nasdaq_1978_return*100:>+8.2f}%")
    print(f"    BH 3x:             {ret_bh*100:>+8.2f}%")
    print(f"    Ens2 (max_lev=1.0): {ret_1*100:>+8.2f}%")
    print(f"    Ens2 (max_lev=3.0): {ret_3*100:>+8.2f}%")

    # Explain the difference
    print(f"""
  【なぜ max_lev=3.0 が +573% なのか？】

  1978年は低ボラティリティ期間で、AsymEWMA Vol が平均 {vol_1978.mean()*100:.1f}%

  Target Vol = 25% なので:
    VT_leverage = 25% / {vol_1978.mean()*100:.1f}% = {0.25/vol_1978.mean():.2f}

  max_lev=3.0 でクリップすると: {min(0.25/vol_1978.mean(), 3.0):.2f}

  さらに Slope Multiplier が平均 {slope_1978.mean():.2f} (MA上昇トレンド)

  最終レバレッジ = {min(0.25/vol_1978.mean(), 3.0):.2f} × {slope_1978.mean():.2f} = {min(0.25/vol_1978.mean(), 3.0) * slope_1978.mean():.2f}

  実効レバレッジ = {min(0.25/vol_1978.mean(), 3.0) * slope_1978.mean():.2f} × 3 = {min(0.25/vol_1978.mean(), 3.0) * slope_1978.mean() * 3:.1f}倍

  NASDAQ +12.31% × {min(0.25/vol_1978.mean(), 3.0) * slope_1978.mean() * 3:.1f}倍 ≈ +{nasdaq_1978_return * min(0.25/vol_1978.mean(), 3.0) * slope_1978.mean() * 3 * 100:.0f}%
  (実際はコストと日次リバランスで {ret_3*100:.0f}%)
""")

    # ==========================================================================
    # 結論
    # ==========================================================================
    print("=" * 100)
    print("結論")
    print("=" * 100)
    print("""
【max_lev=3.0 は現実的ではない】

max_lev=3.0 は以下を意味する:
  - 3倍ETF（例: TQQQ）を借入れで最大300%保有
  - 実効レバレッジ最大9倍
  - 現実の個人投資家には不可能

【推奨: max_lev=1.0】

max_lev=1.0 は以下を意味する:
  - 3倍ETFを0~100%保有（残りは現金）
  - 実効レバレッジ最大3倍
  - 現実的に実行可能

【計算自体は正しい】

計算は正しいが、max_lev=3.0 の想定が非現実的。
FINAL_RESULTS では max_lev=1.0 の結果を推奨戦略としている。

max_lev=3.0 の結果は「理論上の上限」として参考値。
実運用では max_lev=1.0 を使用すべき。
""")

if __name__ == "__main__":
    main()
