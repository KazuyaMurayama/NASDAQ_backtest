"""
Gold/Bond スリーブ拡張ビルダー (2026-05-17)
==============================================
H1〜H5仮説検証用の代替Gold/Bond NAV ビルダー群。

全関数の出力: 既存 build_nav_strategy が pct_change() を取れる NAV np.ndarray。

参照: GOLD_BOND_STRATEGY_PLAN_2026-05-17.md
"""

import numpy as np
import pandas as pd

from corrected_strategy_backtest import (
    TRADING_DAYS, SWAP_SPREAD, GOLD_2X_COST, BOND_3X_COST,
)


def build_gold_cfd(gold_1x_prices, L_g, sofr_daily, spread_annual=0.012):
    """SBI金CFD型 (vol drag なし、線形コスト)

    r_t = L_g * r_gold - (L_g-1) * (sofr_d + spread/252)

    Args:
        gold_1x_prices: 金現物価格 (USD/oz相当)
        L_g: レバレッジ倍率 (3.0 / 5.0 など、固定)
        sofr_daily: 日次SOFRレート (decimal)
        spread_annual: 年率スプレッド (default 1.2%/yr)
    """
    n = len(gold_1x_prices)
    spread_d = spread_annual / TRADING_DAYS
    nav = np.ones(n)
    for i in range(1, n):
        r_g = (gold_1x_prices[i] / gold_1x_prices[i-1] - 1
               if gold_1x_prices[i-1] > 0 else 0.0)
        r_t = L_g * r_g - (L_g - 1.0) * (sofr_daily[i] + spread_d)
        nav[i] = nav[i-1] * (1 + r_t)
    return nav


def build_gold_tocom(gold_1x_prices, L_g, sofr_daily, roll_cost_annual=0.02):
    """TOCOM金先物型 (vol drag なし、SOFR借入なし＝先物の建値で吸収、ロールコスト)

    r_t = L_g * r_gold - (L_g-1) * sofr_d - roll_cost/252

    注: 厳密には先物は借入コストを含まないが、限日取引はSOFR連動とみなして近似。
    保守側でSOFR借入を残し、ロールコストを別途加算。
    """
    n = len(gold_1x_prices)
    roll_d = roll_cost_annual / TRADING_DAYS
    nav = np.ones(n)
    for i in range(1, n):
        r_g = (gold_1x_prices[i] / gold_1x_prices[i-1] - 1
               if gold_1x_prices[i-1] > 0 else 0.0)
        r_t = L_g * r_g - (L_g - 1.0) * sofr_daily[i] - roll_d
        nav[i] = nav[i-1] * (1 + r_t)
    return nav


def build_gold_s2_dynamic(gold_1x_prices, sofr_daily,
                            target_vol=0.30, k_vz=0.20,
                            gate_min=1.0, gate_max=5.0,
                            vz_hl=20, spread_annual=0.012):
    """Gold動的レバレッジ (S2_Gold型、CFD型でvol dragなし)

    sigma_gold = 20日EWMA std × √252
    L_raw = target_vol / sigma_gold
    vz_gold = (sigma_gold - mean) / std (252日ベース)
    vz_gate = clip(1 - k_vz × max(vz_gold,0), gate_min/gate_max, 1.0)
    L_t = clip(L_raw × vz_gate, gate_min, gate_max)

    r_t = L_t * r_gold - (L_t-1) * (sofr_d + spread/252)
    """
    n = len(gold_1x_prices)
    spread_d = spread_annual / TRADING_DAYS

    r_g = np.zeros(n)
    for i in range(1, n):
        if gold_1x_prices[i-1] > 0:
            r_g[i] = gold_1x_prices[i] / gold_1x_prices[i-1] - 1

    rg_s = pd.Series(r_g)
    sigma = rg_s.ewm(halflife=vz_hl, min_periods=vz_hl).std() * np.sqrt(TRADING_DAYS)
    sigma = sigma.replace(0, np.nan).ffill().fillna(0.16)

    L_raw = (target_vol / sigma).clip(lower=0.0)
    L_raw = (L_raw / 0.5).round() * 0.5

    vma = sigma.rolling(252, min_periods=60).mean()
    vstd = sigma.rolling(252, min_periods=60).std().replace(0, 0.001)
    vz_gold = ((sigma - vma) / vstd).fillna(0.0)

    vz_pos = vz_gold.clip(lower=0.0)
    vz_gate = (1.0 - k_vz * vz_pos).clip(gate_min/gate_max, 1.0)

    L_t = (L_raw * vz_gate).clip(gate_min, gate_max)
    L_arr = L_t.values

    nav = np.ones(n)
    for i in range(1, n):
        L = L_arr[i]
        r_t = L * r_g[i] - (L - 1.0) * (sofr_daily[i] + spread_d)
        nav[i] = nav[i-1] * (1 + r_t)
    nav_attrs = {'L_mean': float(np.nanmean(L_arr)),
                 'L_median': float(np.nanmedian(L_arr)),
                 'L_max': float(np.nanmax(L_arr))}
    return nav, nav_attrs


def build_gold_hybrid(gold_1x_prices, sofr_daily,
                       w_etf=0.5, L_etf=2.0, L_cfd=4.0,
                       etf_cost_annual=0.0324, cfd_spread_annual=0.012):
    """ハイブリッド (1540信用2x + CFD4x、重み加重平均)

    実効レバ = w_etf*L_etf + (1-w_etf)*L_cfd = 0.5*2 + 0.5*4 = 3.0 (実効3x相当)

    ETF側: vol drag なし (1540は現物ETF、信用買いコストは固定金利)
      r_etf = L_etf*r_gold - (L_etf-1)*margin_rate - etf_ter
    CFD側: vol drag なし、SOFR連動
      r_cfd = L_cfd*r_gold - (L_cfd-1)*(sofr_d + spread/252)

    r_t = w_etf*r_etf + (1-w_etf)*r_cfd
    """
    n = len(gold_1x_prices)
    etf_cost_d = etf_cost_annual / TRADING_DAYS
    spread_d = cfd_spread_annual / TRADING_DAYS

    nav = np.ones(n)
    for i in range(1, n):
        r_g = (gold_1x_prices[i] / gold_1x_prices[i-1] - 1
               if gold_1x_prices[i-1] > 0 else 0.0)
        r_etf = L_etf * r_g - etf_cost_d
        r_cfd = L_cfd * r_g - (L_cfd - 1.0) * (sofr_daily[i] + spread_d)
        r_t = w_etf * r_etf + (1.0 - w_etf) * r_cfd
        nav[i] = nav[i-1] * (1 + r_t)
    return nav


def build_bond_3x_with_drag(bond_1x_nav, sofr_daily,
                              swap_spread=SWAP_SPREAD,
                              sigma_bond=0.125):
    """TMF型 (vol drag 込み、deep-research準拠)

    既存 build_bond_3x との差分: vol_drag = 0.5*L*(L-1)*sigma^2 を加算減算

    Args:
        sigma_bond: 基礎資産(TLT)の年率σ (default 12.5%)
    """
    n = len(bond_1x_nav)
    swap_d = swap_spread / TRADING_DAYS
    cost_d = BOND_3X_COST / TRADING_DAYS
    L = 3.0
    drag_annual = 0.5 * L * (L - 1.0) * sigma_bond ** 2
    drag_d = drag_annual / TRADING_DAYS

    nav = np.ones(n)
    for i in range(1, n):
        br = (bond_1x_nav[i] / bond_1x_nav[i-1] - 1
              if bond_1x_nav[i-1] > 0 else 0.0)
        r_t = L * br - 2.0 * (sofr_daily[i] + swap_d) - cost_d - drag_d
        nav[i] = nav[i-1] * (1 + r_t)
    return nav
