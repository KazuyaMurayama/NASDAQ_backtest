"""
P0 Critical Verification Script — Opus指摘の要検証項目
=======================================================
1. SOFR日次値の単位確認 (DTB3が日次小数か年率%かの検証)
2. vt_mult クリッピング率 (P2/S2でtarget_vol/sigma>=1の割合)
3. CFD 5x固定 vs P2 vs S2 のWorst5Y直接比較
4. Worst5Y定義の確認 (252×5 or 260×5, CAGR計算式)
"""

import sys, os, types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import load_sofr, build_a2_signal, TRADING_DAYS
from cfd_leverage_backtest import (
    FULL_START, FULL_END, IS_START, IS_END, OOS_START,
    CFD_SPREAD_LOW, CFD_TER,
    build_cfd_nas_sleeve,
)
from dynamic_leverage_strategies import compute_L_vol_target, compute_L_s2_vz_gated

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'NASDAQ_extended_to_2026.csv')
DELAY = 2

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=== データ読み込み ===")
df = load_data(DATA_PATH)
df.columns = [c.lower() for c in df.columns]
if not isinstance(df.index, pd.DatetimeIndex):
    df = df.set_index('date')

close = df['close']
returns = close.pct_change().fillna(0)
sofr_daily = pd.Series(load_sofr(close.index.to_series()), index=close.index)

print(f"データ期間: {close.index[0].date()} 〜 {close.index[-1].date()}")
print(f"総営業日数: {len(close)}")

# ---------------------------------------------------------------------------
# 1. SOFR単位検証
# ---------------------------------------------------------------------------
print("\n=== [1] SOFR日次値の単位検証 ===")
sofr_notna = sofr_daily[sofr_daily > 0].dropna()
print(f"非ゼロ日数: {len(sofr_notna)}")
print(f"日次値の統計:")
print(f"  Min:    {sofr_notna.min():.8f}")
print(f"  Median: {sofr_notna.median():.8f}")
print(f"  Mean:   {sofr_notna.mean():.8f}")
print(f"  Max:    {sofr_notna.max():.8f}")

sofr_ann_implied = sofr_notna.median() * TRADING_DAYS
print(f"\n中央値を年率換算 (×{TRADING_DAYS}): {sofr_ann_implied:.4f} ({sofr_ann_implied*100:.2f}%/年)")

# 高金利期のサンプル (2023)
mask_2023 = (sofr_daily.index >= '2023-01-01') & (sofr_daily.index <= '2024-01-01')
sofr_2023 = sofr_daily[mask_2023]
sofr_2023_ann = sofr_2023.mean() * TRADING_DAYS
print(f"\n2023年の日次値(平均): {sofr_2023.mean():.8f}")
print(f"2023年の年率換算: {sofr_2023_ann:.4f} ({sofr_2023_ann*100:.2f}%/年)")
print(f"→ 期待値: 2023年FFレート≈5.25% 前後が妥当なら単位は正しい")

# 1980年代のピーク確認
mask_1981 = (sofr_daily.index >= '1981-01-01') & (sofr_daily.index <= '1982-01-01')
sofr_1981 = sofr_daily[mask_1981]
sofr_1981_ann = sofr_1981.mean() * TRADING_DAYS
print(f"\n1981年の年率換算: {sofr_1981_ann:.4f} ({sofr_1981_ann*100:.2f}%/年)")
print(f"→ 期待値: 1981年FFレート≈15-18% 前後が妥当")

# ---------------------------------------------------------------------------
# 2. vt_mult クリッピング率 (P2とS2)
# ---------------------------------------------------------------------------
print("\n=== [2] vt_mult クリッピング率検証 ===")

n = 20
sigma_rolling = returns.rolling(n, min_periods=5).std() * np.sqrt(TRADING_DAYS)
sigma_valid = sigma_rolling[sigma_rolling > 0.01].dropna()

print(f"\n  NASDAQ実現ボラ統計 (20日ローリング、年率):")
print(f"    Min:    {sigma_valid.min()*100:.1f}%")
print(f"    Median: {sigma_valid.median()*100:.1f}%")
print(f"    Mean:   {sigma_valid.mean()*100:.1f}%")
print(f"    Max:    {sigma_valid.max()*100:.1f}%")

for target_vol in [0.60, 0.70, 0.80]:
    ratio = target_vol / sigma_valid
    clip_rate = (ratio >= 1.0).mean()
    print(f"\n  target_vol={target_vol:.2f}:")
    print(f"    ratio Min={ratio.min():.3f}  Median={ratio.median():.3f}  Mean={ratio.mean():.3f}  Max={ratio.max():.3f}")
    print(f"    ratio>=1.0 (レバ上限クリップ日): {clip_rate:.1%}")
    print(f"    → vol-targetingが実効的に機能している割合: {1-clip_rate:.1%}")

# OOS期間
oos_mask_s = sigma_rolling.index >= '2021-05-08'
sigma_oos = sigma_rolling[oos_mask_s & (sigma_rolling > 0.01)].dropna()
print(f"\n  OOS期間 (2021-05-08〜) 実現ボラ統計:")
print(f"    Median: {sigma_oos.median()*100:.1f}%  Mean: {sigma_oos.mean()*100:.1f}%")
for tv in [0.60, 0.80]:
    clip_oos = (tv / sigma_oos >= 1.0).mean()
    print(f"    target_vol={tv}: clip率={clip_oos:.1%}")

# ---------------------------------------------------------------------------
# 3. NAV構築 (CFD sleeve のみ、DH portfolioなし)
# ---------------------------------------------------------------------------
print("\n=== [3] CFD NASDAQスリーブ単体のWorst5Y比較 ===")

r_arr = returns.values
sofr_arr = sofr_daily.values

def build_cfd_only_nav(returns_s, sofr_s, leverage_s):
    """CFD NAVを構築 (NASDAQスリーブ単体)"""
    r = returns_s.values
    sf = sofr_s.values
    if isinstance(leverage_s, (int, float)):
        L = np.full(len(r), float(leverage_s))
        L_shifted = L  # 固定レバは shift不要
    else:
        L_arr = np.asarray(leverage_s, dtype=float)
        L_shifted = pd.Series(L_arr, index=returns_s.index).shift(DELAY).fillna(1.0).values

    r_cfd = build_cfd_nas_sleeve(r, L_shifted, sf, CFD_SPREAD_LOW, CFD_TER)
    nav = pd.Series((1 + r_cfd).cumprod(), index=returns_s.index)
    nav = nav.clip(lower=1e-6)
    return nav

# CFD 5x固定
nav_cfd5 = build_cfd_only_nav(returns, sofr_daily, 5.0)

# P2 (target_vol=0.8)
L_p2 = compute_L_vol_target(returns, target_vol=0.80, n=20, l_min=1.0, l_max=7.0)
nav_p2 = build_cfd_only_nav(returns, sofr_daily, L_p2)

# S2 推奨パラメータ (k_vz=0.3, gate_min=0.2, tv=0.8)
raw_a2, vz, _ = build_a2_signal(close, returns, return_components=True)
L_s2 = compute_L_s2_vz_gated(returns, vz, target_vol=0.80, k_vz=0.30, gate_min=0.20, n=20, l_min=1.0, l_max=7.0)
nav_s2 = build_cfd_only_nav(returns, sofr_daily, L_s2)

def calc_worst5y(nav: pd.Series, window: int = 252*5) -> float:
    log_nav = np.log(nav.clip(lower=1e-6))
    log_r5y = log_nav - log_nav.shift(window)
    cagr_5y = np.exp(log_r5y / 5) - 1
    return cagr_5y.dropna().min()

def find_worst5y_period(nav: pd.Series, window: int = 252*5):
    log_nav = np.log(nav.clip(lower=1e-6))
    log_r5y = log_nav - log_nav.shift(window)
    cagr_5y = np.exp(log_r5y / 5) - 1
    idx = cagr_5y.dropna().idxmin()
    loc = nav.index.get_loc(idx)
    start = nav.index[max(loc - window, 0)]
    return start, idx, cagr_5y.min()

for window_label, window in [('252×5', 252*5), ('260×5', 260*5)]:
    print(f"\n  window={window_label}:")
    for name, nav in [('CFD 5x固定', nav_cfd5), ('P2 tv=0.8', nav_p2), ('S2 k=0.3,g=0.2,tv=0.8', nav_s2)]:
        w5 = calc_worst5y(nav, window)
        start, end, _ = find_worst5y_period(nav, window)
        print(f"    {name}: Worst5Y={w5*100:+.2f}%  期間={start.date()}〜{end.date()}")

# ---------------------------------------------------------------------------
# 4. OOS CAGR/Sharpe の直接計算
# ---------------------------------------------------------------------------
print("\n=== [4] OOS指標の直接計算 (CFDスリーブ単体) ===")

def calc_metrics(nav: pd.Series, start: str, end: str = None):
    mask = nav.index >= start
    if end:
        mask &= nav.index <= end
    nav_sub = nav[mask]
    if len(nav_sub) < 100:
        return {}
    r = nav_sub.pct_change().dropna()
    n_years = len(nav_sub) / TRADING_DAYS
    cagr = nav_sub.iloc[-1] ** (1/n_years) - 1
    sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0
    dd = (nav_sub / nav_sub.cummax() - 1).min()
    w5 = calc_worst5y(nav_sub)
    return {'CAGR': cagr, 'Sharpe': sharpe, 'MaxDD': dd, 'Worst5Y': w5}

for label, nav in [('CFD 5x固定', nav_cfd5), ('P2 tv=0.8', nav_p2), ('S2 k=0.3,g=0.2,tv=0.8', nav_s2)]:
    m_full = calc_metrics(nav, FULL_START)
    m_oos  = calc_metrics(nav, OOS_START)
    print(f"\n  {label}:")
    print(f"    FULL: CAGR={m_full.get('CAGR',0)*100:+.2f}%  Sharpe={m_full.get('Sharpe',0):.3f}  MaxDD={m_full.get('MaxDD',0)*100:.1f}%  Worst5Y={m_full.get('Worst5Y',0)*100:+.2f}%")
    print(f"    OOS:  CAGR={m_oos.get('CAGR',0)*100:+.2f}%  Sharpe={m_oos.get('Sharpe',0):.3f}  MaxDD={m_oos.get('MaxDD',0)*100:.1f}%")

# ---------------------------------------------------------------------------
# 5. レバレッジ分布の確認
# ---------------------------------------------------------------------------
print("\n=== [5] P2/S2 レバレッジ分布 ===")
oos_mask = returns.index >= '2021-05-08'

for name, L in [('P2 tv=0.8', L_p2), ('S2 k=0.3,g=0.2,tv=0.8', L_s2)]:
    print(f"\n  {name}:")
    print(f"    FULL: Mean={L.mean():.3f}x  Std={L.std():.3f}x  Min={L.min():.1f}x  Max={L.max():.1f}x")
    print(f"          =7x日: {(L==7.0).mean():.1%}  =1x日: {(L==1.0).mean():.1%}")
    L_oos = L[oos_mask]
    print(f"    OOS:  Mean={L_oos.mean():.3f}x  Std={L_oos.std():.3f}x")

print("\n=== 検証完了 ===")
