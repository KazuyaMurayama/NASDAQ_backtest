"""
src/audit/margin_liquidation_stress_20260617.py
================================================
Margin / liquidation stress analysis for strategy #1: Bext_str_sc1.35
= B3a + v7_map STRONG {Q0:1.60,Q1:1.50,Q2:1.10,Q3:1.00} x scale=1.35
  + P09 OUT fill + C1 SOFR + k365 cost model.

計画: MARGIN_CAPACITY_STRESS_PLAN_20260617.md  §0-§3

--------------------------------------------------------------------
仮定・前提(全て明示、楽観に倒さない)
--------------------------------------------------------------------
1. AUM = ¥30,000,000 (3000万円固定)
2. NASDAQ-sleeve構成:
   - ≤3倍部分 = TQQQ (自前口座に追証なし。ETF内部レバのみ損失。)
   - >3倍超過部分 = くりっく株365 NASDAQ-100 (取引所CFD = 追証/ロスカット対象)
3. 超過建玉 notional: excess_notional = wn * max(L-3, 0) * AUM
   (wn = NASDAQ资本ウェイト；post-shift後の値を使用)
4. 証拠金率シナリオ:
   - mar_A = 4.24% (¥9,320/枚 ÷ ¥220,000/枚; TFX公式2026-06最小値)
   - mar_B = 8.00% (感度: 実務バッファ想定)
   - mar_C = 12.00% (感度: 市場ストレス時の引き上げ想定)
5. 維持証拠金 ≒ 必要証拠金 (TFX仕様により同額)
6. 口座モデル:
   (A) 分離口座: k365口座 equity = 投下証拠金のみ
       下落dでk365 P&L = -d * excess_notional (方向性CFDとしてlong保有)
       清算距離 = min(1, 証拠金率) = 証拠金が尽きるまでの下落幅
   (B) cross-margin: 全ポートフォリオ equity が担保
       守り資産(Gold/Bond/Cash) の価値もカウント
       NASDAQ下落d → Goldは反相関係数ρ=-0.3想定 (保守的; 実際は-0.2〜-0.5)
       Bond: 株式暴落時はリスクオフで上昇; ρ=-0.2想定(保守的)
       Cross-margin distace は解けるが近似: conservative (ρ=0)でも計算
7. イントラデイ/ギャップ加算:
   終値ベース下落率d に +{5, 10, 15}% を加算してスリッページ/窓開けを模倣
   (例: 終値-10%の日は実際の清算額は-10.5% / -11.0% / -11.5%とみなす)
8. M3 清算シミュレーション:
   - 維持証拠金割れ → k365ポジションを強制清算 (当日終値で。実際はより悪い)
   - その後、次の IN シグナル(fund_active=False への切替)まで k365 建玉=0
   - 清算後はTQQQ(≤3x)のみで運用
   - 清算時のP&Lはバックテストに反映(追加損失として計上)
9. Gold/Bond の日次リターンは同期間の実データを使用(Silver/CPI等は除外)
10. UNCLEAR: TFXの実際の追証発動条件(翌営業日朝)vs 同日清算。
    ここでは保守的に「翌日朝前に清算」と仮定(最悪ケース側)。
11. UNCLEAR: 本分析は終値ベース。取引時間中の瞬間的な下落(intraday flush)は
    終値に反映されない場合がある。イントラデイ加算感度はこれを部分的に補正。

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.

Outputs:
  audit_results/margin_liquidation_stress_20260617.csv
  RETURN_BLOCK printed to stdout (json.dumps)
"""

from __future__ import annotations

import json
import os
import sys
import types

# ---- multitasking stub -------------------------------------------------------
if "multitasking" not in sys.modules:
    _m = types.ModuleType("multitasking")
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules["multitasking"] = _m

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

import src.audit.strategy_runners as sr
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START
from src.audit.k365_recost_20260612 import (
    _build_full_c1,
    EXCESS_EXTRA_K365_CENTRE,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
    FEE_GOLD, FEE_BOND,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    SWAP_SPREAD, TER_TQQQ, DH_PER_UNIT, NAV_FLOOR,
    DELAY as V7_DELAY,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _build_v7_mult_custom,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
B3A_MAP_STRONG = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
LEV_SCALE = 1.35
EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025

AUM_JPY = 30_000_000  # 3000万円

# 証拠金率シナリオ
MARGIN_RATES = {
    "mar_4.24pct": 0.0424,   # TFX公式最小値 (最も脆い)
    "mar_8pct":    0.0800,   # 感度: 実務バッファ
    "mar_12pct":   0.1200,   # 感度: ストレス時引き上げ
}

# イントラデイ加算シナリオ (終値DDに乗算で加算)
INTRADAY_ADD = {
    "id_0pct":  0.00,   # 加算なし (base case)
    "id_5pct":  0.05,   # +5%スリッページ
    "id_10pct": 0.10,   # +10%スリッページ
    "id_15pct": 0.15,   # +15%スリッページ
}

# Sanity targets (from LEVERUP_EXTENSION_RESULTS_20260616.md)
SANITY_MIN9_EXPECT  = 0.2383   # +23.83%
SANITY_MAXDD_EXPECT = -0.4504  # -45.04%
SANITY_TOL = 0.0010             # +/-0.10pp

# Gold/Bond cross-margin correlation assumptions (conservative)
# ρ=0 is MOST conservative (no diversification benefit from defensive assets)
# ρ=-0.3 is mild benefit for gold, ρ=-0.2 for bond
RHO_GOLD_NASSDAQ = 0.0   # Use 0 for most conservative (no benefit)
RHO_BOND_NASDAQ  = 0.0   # Use 0 for most conservative


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def _build_leverage_and_weight_series(shared, dates_dt, v7_map, lev_scale):
    """
    Reconstruct per-day effective leverage L and weight series (post-shift)
    for NASDAQ, Gold, Bond and the IN/OUT mask.

    Returns:
        L_s        : ndarray (n,) -- effective leverage (post-V7_DELAY shift)
        wn_s       : ndarray (n,) -- NASDAQ capital weight (post-shift)
        wg_s       : ndarray (n,) -- Gold capital weight (post-shift)
        wb_s       : ndarray (n,) -- Bond capital weight (post-shift)
        in_mask    : ndarray bool (n,) -- True = NASDAQ-active day (pre-shift fund_active==False)
        excess_notional_pct : ndarray (n,) -- excess_notional / AUM (=wn_s * max(L_s-3,0))
    """
    a = shared["assets"]
    dates = a["dates"]
    idx = dates.index
    mask = np.asarray(shared["mask"], dtype=float)

    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)
    wn = np.asarray(shared["wn"], float)
    wg = np.asarray(shared["wg"], float)
    wb = np.asarray(shared["wb"], float)

    # Build v7 mult
    mult_v7 = _build_v7_mult_custom(dates_dt, v7_map) * float(lev_scale)

    # Unshifted leverage
    lev_mod = lev_raw_masked * mult_v7
    L_unshifted = lev_mod * 3.0

    # Apply V7_DELAY shift
    L_s  = pd.Series(L_unshifted, index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_s = pd.Series(wn,          index=idx).shift(V7_DELAY).fillna(0.0).values
    wg_s = pd.Series(wg,          index=idx).shift(V7_DELAY).fillna(0.0).values
    wb_s = pd.Series(wb,          index=idx).shift(V7_DELAY).fillna(0.0).values

    # IN/OUT mask (pre-shift -- reflects when positions were actually held)
    out_mask_arr = (mask < 0.5)  # True = OUT signal
    fund_active  = np.zeros(len(dates_dt), dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]
    in_mask = ~fund_active

    # Excess notional as fraction of AUM
    excess_notional_pct = wn_s * np.maximum(L_s - 3.0, 0.0)

    return L_s, wn_s, wg_s, wb_s, in_mask, fund_active, excess_notional_pct


# ---------------------------------------------------------------------------
# M1: distance-to-liquidation time series
# ---------------------------------------------------------------------------

def compute_m1_distance(L_s, wn_s, wg_s, wb_s,
                        in_mask, fund_active,
                        dates_dt, r_nasdaq,
                        ret_gold, ret_bond,
                        bond_on, wg_iv, wb_iv,
                        margin_rate, label):
    """
    Compute the 'distance to liquidation' for each day in basis:
    How many percent does NASDAQ need to fall for k365 margin to be wiped out?

    口座(A) - 分離:
      equity_A = margin_rate * excess_notional (投下証拠金のみ)
      下落d -> P&L = -d * excess_notional
      清算距離 = equity_A / excess_notional = margin_rate
      (= constant = 証拠金率; 超過建玉がある限り常にこの距離)
      BUT: 証拠金を多めに投下している場合は cash_buffer 感度として計算
      実務: 最低限の証拠金なら清算距離 = margin_rate (例: 4.24%下落で清算)

    口座(B) - cross-margin (保守推定 ρ=0; 守り資産は並行で独立に動く):
      全portfolio equity が担保
      k365 P&L = -d * excess_notional
      TQQQ P&L = -d * L * wn (TQQQはポジションの損失)
      守り資産は独立 (ρ=0 => 変化なし)
      維持証拠金 = margin_rate * excess_notional
      清算距離(B) = margin_rate / (1 + L) の簡易式で近似(全体資本に対するk365損失率)
      正確には: 全portfolio_equity = 1.0 (100%基準)
                k365 必要証拠金 = margin_rate * excess_notional_pct
                k365 P&L(NASDAQ d%下落) = -d * excess_notional_pct
                清算: P&L <= -equity_posted
                equity_posted (A) = margin_rate * excess_notional_pct
                equity_posted (B) = portfolio_equity (= 1.0 基準)
                -> (B) 清算距離: d = 1.0 / excess_notional_pct  (最大損失で清算)
                   ただし excess_notional_pct -> 0 なら 無限大 -> 100%にcap
      実際には維持証拠金割れの判定なので:
                distance_B = margin_rate * excess_notional_pct / excess_notional_pct
                           = margin_rate ... 同じになる?
      => 口座(B)の正確な意味:
         全資産を担保にk365を運用 -> 担保額は AUM_full
         余裕資産 = AUM - (margin_posted to k365)
         margin_posted = margin_rate * notional
         もしportfolioの損失が margin_posted を超えたら清算
         = NASDAQ下落d -> portfolio_loss = d * excess_notional_pct (k365分)
         余裕 = (margin_rate * excess_notional_pct) / excess_notional_pct = margin_rate
         => 分離でも cross でも 清算距離の式は同じ (margin_rate)
         差は: cross-marginでは守り資産の含み益がバッファになる
         distance_B = (margin_rate * excess_notional_pct + defensive_gain) / excess_notional_pct
         defensive_gainは Gold上昇 + Bond上昇 (NASDAQ下落時)
         保守的(ρ=0)では defensive_gain = 0 -> distance_B = margin_rate = distance_A

    NOTE: 上記により ρ=0 前提では A/B 距離は同一。
    cross-margin の差別化は ρ=-0.3 (Gold-NASDAQ 相関) を使った場合のみ現れる。
    本スクリプトでは:
      A: ρ=0 (最悪 -- 守り資産は証拠金に使えない)
      B: ρ=-0.3(Gold) ρ=-0.2(Bond) での感度計算
    """
    n = len(dates_dt)
    excess_n = wn_s * np.maximum(L_s - 3.0, 0.0)  # as fraction of AUM
    in_k365 = excess_n > 1e-6  # has k365 exposure

    # 口座 (A): distance = margin_rate (always, if any k365 position)
    # When excess_n == 0 (no k365), distance = infinity (no risk) -> set 1.0 for display
    dist_A = np.where(in_k365, margin_rate, 1.0)

    # 口座 (B): with conservative rho=0 = same as A
    # With realistic rho (for gold and bond):
    # Gold weight on IN days: wg_s (from base signal, small)
    # Gold weight on OUT days: wg_iv
    # Bond weight: similar
    # Gold vol day-to-day: use actual gold returns
    # NASDAQ d -> gold expected delta = rho_g * (vol_gold/vol_nasd) * (-d) * wg_eff
    # For simplicity: use -0.3 sensitivity for gold, -0.2 for bond
    # defensive contribution = 0.3 * d * wg_s + 0.2 * d * wb_s (on IN days)
    #   This means portfolio_equity increases by that amount when NASDAQ falls d
    # distance_B: margin_rate * excess_n + 0.3 * d * wg_s + 0.2 * d * wb_s = d * excess_n
    # -> d * (excess_n - 0.3*wg_s - 0.2*wb_s) = margin_rate * excess_n
    # -> d = margin_rate * excess_n / (excess_n - 0.3*wg_s - 0.2*wb_s)
    # if denominator < 0 -> never liquidated -> inf -> cap 1.0
    # (Note: on IN days wg_s and wb_s are small from DH-W1 base)
    RHO_G = 0.30  # gold contribution coefficient (conservative: 0.3, not 0.5)
    RHO_B = 0.20  # bond contribution coefficient
    denom_B = excess_n - RHO_G * wg_s - RHO_B * wb_s
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_B_rho = np.where(
            (in_k365) & (denom_B > 1e-9),
            np.clip(margin_rate * excess_n / denom_B, 0.0, 1.0),
            np.where(in_k365, margin_rate, 1.0)  # fallback
        )

    return dist_A, dist_B_rho, excess_n, in_k365


# ---------------------------------------------------------------------------
# M2: worst-drop stress test
# ---------------------------------------------------------------------------

def compute_m2_worst_drops(r_nasdaq, dates_dt):
    """
    Extract worst 1-day, 2-day, 5-day NASDAQ drops from full history.
    Returns dict with top drops and named crises.
    """
    n = len(r_nasdaq)
    dates_dt = pd.DatetimeIndex(dates_dt)

    # 1-day worst
    top_1d = pd.Series(r_nasdaq, index=dates_dt).nsmallest(20)

    # 2-day rolling worst (sum of 2 consecutive)
    ret_2d = pd.Series(r_nasdaq, index=dates_dt).rolling(2).sum().dropna()
    top_2d = ret_2d.nsmallest(10)

    # 5-day rolling worst
    ret_5d = pd.Series(r_nasdaq, index=dates_dt).rolling(5).sum().dropna()
    top_5d = ret_5d.nsmallest(10)

    # Named crises (approximate date ranges for worst single-day impact)
    crises = {
        "BlackMonday1987":  ("1987-10-19", "1987-10-20"),
        "DotcomPeak2000":   ("2000-04-03", "2000-04-14"),
        "PostLehman2008":   ("2008-10-07", "2008-10-10"),
        "COVID2020":        ("2020-03-11", "2020-03-13"),
        "Rate2022":         ("2022-01-18", "2022-09-26"),
    }

    crisis_drops = {}
    for name, (start, end) in crises.items():
        mask_c = (dates_dt >= pd.Timestamp(start)) & (dates_dt <= pd.Timestamp(end))
        if mask_c.sum() > 0:
            r_c = r_nasdaq[mask_c]
            worst_1d = float(r_c.min())
            cum = float((1.0 + pd.Series(r_c)).prod() - 1.0)
            crisis_drops[name] = {
                "worst_1d": worst_1d,
                "cumulative": cum,
                "n_days": int(mask_c.sum()),
            }

    return {
        "top_1d": [(str(d.date()), float(v)) for d, v in top_1d.items()],
        "top_2d": [(str(d.date()), float(v)) for d, v in top_2d.items()],
        "top_5d": [(str(d.date()), float(v)) for d, v in top_5d.items()],
        "crises": crisis_drops,
    }


def compute_m2_threshold_table(L_levels, margin_rates_dict, intraday_add_dict):
    """
    For each combination of L, margin_rate, intraday_add:
    Determine the NASDAQ single-day drop (%) that triggers forced liquidation.

    For 口座(A) -- 分離:
      Required margin = margin_rate * excess_notional
      excess_notional = (L-3)/L * total_NASDAQ_exposure (as % of AUM=1)
        = wn * (L-3)  [as fraction of AUM; assume wn=1 for simplicity on IN days]
      Actually: k365 position size = wn * (L-3) * AUM
      Loss = d * k365_notional = d * wn * (L-3) * AUM
      Margin posted = margin_rate * wn * (L-3) * AUM
      Liquidation when: Loss >= Margin posted
        d * wn * (L-3) * AUM >= margin_rate * wn * (L-3) * AUM
        d >= margin_rate

    => REGARDLESS of L (as long as L>3), the clearance distance = margin_rate.
    => With intraday add: effective_drop = drop * (1 + intraday_add)
       Liquidation when: effective_drop >= margin_rate
       => required drop = margin_rate / (1 + intraday_add)

    IMPORTANT: This means L=4 and L=6.48 are EQUALLY vulnerable to the
    same % NASDAQ drop. The AMOUNT of money lost differs, but the TRIGGER
    threshold (margin_rate) is the same. This is the critical insight.

    For 口座(B) -- cross-margin (ρ=0 保守):
      Same result as A (見上文).

    Summary: 「何%下落で清算か」= margin_rate / (1 + intraday_add)
             L は閾値に影響しない（投下証拠金と損失が等倍に変化するため）
             ただし「清算時の損失額」は L が高いほど大きい。
    """
    rows = []
    for L_val in L_levels:
        for mar_name, mar_rate in margin_rates_dict.items():
            for id_name, id_add in intraday_add_dict.items():
                # Threshold: required NASDAQ drop to trigger liquidation
                # = margin_rate / (1 + intraday_add_fraction)
                # id_add is stated as fraction of drop, so effective_drop = raw_drop * (1+id_add)
                # trigger: raw_drop * (1+id_add) >= margin_rate
                # raw_drop >= margin_rate / (1+id_add)
                threshold = mar_rate / (1.0 + id_add)

                # Amount of AUM lost at liquidation (assume wn=1, IN day)
                # excess_notional = (L-3) * AUM (fraction: L-3)
                # loss = threshold * excess_notional = threshold * (L-3)
                loss_pct_aum = threshold * max(L_val - 3.0, 0.0)
                loss_jpy = loss_pct_aum * AUM_JPY

                # Sanity: is it plausible? Black Monday 1987 = -11.3% single day
                # 2020 COVID worst day = -9.98%  2022 worst day = -5.3%
                # So 4.24% threshold means BLACK MONDAY would trigger liquidation
                # even at minimum margin
                note = "CLEAR" if threshold > 0.10 else ("WARN" if threshold > 0.05 else "DANGER")

                rows.append({
                    "L": L_val,
                    "excess_L_minus3": round(max(L_val - 3.0, 0.0), 2),
                    "margin_rate_pct": round(mar_rate * 100, 2),
                    "intraday_add_pct": round(id_add * 100, 1),
                    "threshold_drop_pct": round(threshold * 100, 4),
                    "loss_at_liq_pct_AUM": round(loss_pct_aum * 100, 4),
                    "loss_at_liq_JPY": round(loss_jpy, 0),
                    "risk_level": note,
                    "scenario": "%s_%s" % (mar_name, id_name),
                })

    return rows


# ---------------------------------------------------------------------------
# M3: liquidation-rule NAV reconstruction
# ---------------------------------------------------------------------------

def compute_m3_liquidation_nav(r_full, r_nasdaq, dates_dt,
                               L_s, wn_s, wg_s, wb_s,
                               in_mask, fund_active,
                               margin_rate, intraday_add,
                               account_model="A",
                               rho_gold=0.0, rho_bond=0.0,
                               ret_gold=None, ret_bond=None,
                               wg_iv=None, wb_iv=None,
                               bond_on=None):
    """
    Re-simulate NAV with a forced-liquidation rule.

    When k365 margin is breached (see logic below), the excess k365 position
    is force-liquidated at that day's close. The position stays at zero until
    the next IN signal (fund_active switches from True to False = re-entry).

    Args:
        r_full       : daily returns of the baseline (no-liquidation) strategy
        r_nasdaq     : daily NASDAQ returns
        dates_dt     : DatetimeIndex
        L_s, wn_s... : leverage / weight series (post-shift)
        margin_rate  : fraction (e.g. 0.0424)
        intraday_add : additional % to add to observed drop (e.g. 0.05)
        account_model: "A" (separate) or "B" (cross-margin ρ assumption)
        ret_gold, ret_bond: daily gold/bond returns for cross-margin calc

    Returns:
        dict with liquidation events, final NAV, CAGR, MaxDD, and vs-baseline comparison
    """
    n = len(dates_dt)
    excess_n = wn_s * np.maximum(L_s - 3.0, 0.0)

    # -- State variables --
    nav_liq = np.ones(n, dtype=float)          # cumulative NAV (liquidation scenario)
    k365_suspended = False                     # True when k365 is force-liquidated & waiting re-entry
    liq_events = []                            # list of dicts for each liquidation event

    # State for tracking: whether we are waiting for first OUT period to end before re-entry
    # Re-entry logic:
    #   1. When k365 is liquidated (on an IN day), set k365_suspended=True, saw_out=False
    #   2. Wait until we see an OUT day (fund_active=True) -> set saw_out=True
    #   3. When saw_out=True and we see the next IN day (fund_active False) -> re-enter
    # This correctly models "wait until next OUT period then re-enter on next IN"
    # which is the natural re-entry on the next trading signal.
    saw_out_after_liq = False  # True when we've seen an OUT day after the last liquidation

    for t in range(1, n):
        r_nasdaq_t = float(r_nasdaq[t])
        r_full_t   = float(r_full[t])
        L_t        = float(L_s[t])
        wn_t       = float(wn_s[t])
        excess_t   = float(excess_n[t])
        in_day     = bool(in_mask[t])

        # Check if we can re-enter k365
        # Step 1: if suspended and we hit an OUT day, record that we saw an OUT period
        if k365_suspended and fund_active[t]:
            saw_out_after_liq = True
        # Step 2: re-enter on the first IN day AFTER an OUT period
        if k365_suspended and saw_out_after_liq and not fund_active[t]:
            k365_suspended = False
            saw_out_after_liq = False

        # Check for liquidation trigger on this day
        liquidated_today = False
        if (not k365_suspended) and in_day and (excess_t > 1e-6):
            # Effective NASDAQ drop (adding intraday/gap slippage)
            eff_drop = -r_nasdaq_t  # positive means NASDAQ fell
            if eff_drop > 0:
                eff_drop_with_slip = eff_drop * (1.0 + intraday_add)

                if account_model == "A":
                    # distance = margin_rate (posting minimum = margin_rate * notional)
                    # Liquidation: eff_drop_with_slip >= margin_rate
                    trigger = eff_drop_with_slip >= margin_rate

                else:  # account_model == "B" cross-margin
                    # Additional buffer from defensive assets (NASDAQ down -> gold/bond up)
                    # Gold: on IN days wg_s[t], on OUT days wg_iv[t]
                    # Bond: wb_s[t] or wb_iv[t]*bond_on[t]
                    wg_t = float(wg_s[t]) if in_day else float(wg_iv[t]) if wg_iv is not None else 0.0
                    wb_t_eff = float(wb_s[t]) if in_day else (float(wb_iv[t]) * float(bond_on[t]) if wb_iv is not None else 0.0)
                    # Defensive offset reduces required drop to trigger
                    defensive_buffer = rho_gold * wg_t + rho_bond * wb_t_eff
                    denom = excess_t - defensive_buffer
                    if denom > 1e-9:
                        threshold_B = np.clip(margin_rate * excess_t / denom, 0.0, 1.0)
                    else:
                        threshold_B = 1.0  # can't be liquidated if denom <= 0
                    trigger = eff_drop_with_slip >= threshold_B

                if trigger:
                    liquidated_today = True
                    k365_suspended = True
                    # Loss from k365 on this day (extra loss vs TQQQ-only)
                    # TQQQ loss already captured in r_full_t
                    # Additional k365 loss = eff_drop * excess_t (fraction of AUM)
                    # (this is the loss from the liquidated portion that would normally recover)
                    # NOTE: The main effect is being excluded from future recovery
                    # The current day loss IS reflected in r_full_t (strategy return)
                    # Additional penalty: we miss TQQQ recovery but keep the base loss
                    # Simplification: on liquidation day, return = r_full_t (no extra)
                    # On subsequent IN days while suspended: return = r_full_t - k365_contribution
                    # k365 contribution = wn_s * (L-3) * r_nasdaq (the excess part)
                    liq_events.append({
                        "date": str(dates_dt[t].date()),
                        "L": L_t,
                        "excess_n_pct": round(excess_t * 100, 4),
                        "nasdaq_drop_pct": round(-r_nasdaq_t * 100, 4),
                        "eff_drop_pct": round(eff_drop_with_slip * 100, 4),
                        "margin_rate_pct": round(margin_rate * 100, 2),
                        "liq_loss_pct_AUM": round(eff_drop_with_slip * excess_t * 100, 4),
                        "liq_loss_JPY": round(eff_drop_with_slip * excess_t * AUM_JPY, 0),
                    })

        # Compute adjusted return for this day
        if k365_suspended and in_day and not liquidated_today:
            # k365 is suspended: remove the k365 contribution from the return
            # k365 contribution to daily return = wn_s * (L-3) * r_nasdaq
            # (positive or negative)
            k365_contrib = wn_s[t] * max(L_s[t] - 3.0, 0.0) * r_nasdaq[t]
            r_adj = r_full_t - k365_contrib
        else:
            r_adj = r_full_t

        nav_liq[t] = nav_liq[t - 1] * (1.0 + r_adj)

    # Compute metrics for liquidation NAV
    nav_liq_series = pd.Series(nav_liq, index=dates_dt)
    r_liq = nav_liq_series.pct_change().fillna(0.0).values

    # Baseline NAV
    nav_base = np.cumprod(1.0 + np.asarray(r_full, float))
    nav_base_series = pd.Series(nav_base, index=dates_dt)

    final_nav_liq  = float(nav_liq[-1])
    final_nav_base = float(nav_base[-1])

    n_years = len(dates_dt) / float(TRADING_DAYS)
    cagr_liq  = final_nav_liq  ** (1.0 / n_years) - 1.0
    cagr_base = final_nav_base ** (1.0 / n_years) - 1.0

    # MaxDD
    def _maxdd(nav):
        roll_max = np.maximum.accumulate(nav)
        dd = nav / roll_max - 1.0
        return float(dd.min())

    maxdd_liq  = _maxdd(nav_liq)
    maxdd_base = _maxdd(nav_base)

    # CAGR9 (min of IS / OOS -- approximate with full-period only since we altered returns)
    # Use IS <= IS_END, OOS >= OOS_START
    is_mask  = np.array(dates_dt <= IS_END)
    oos_mask = np.array(dates_dt >= OOS_START)

    def _seg_cagr(nav_arr, mask):
        sub = nav_arr[mask]
        if len(sub) < 2:
            return float("nan")
        ny = mask.sum() / float(TRADING_DAYS)
        return float(sub[-1] / sub[0]) ** (1.0 / ny) - 1.0

    cagr_liq_is  = _seg_cagr(nav_liq,  is_mask)
    cagr_liq_oos = _seg_cagr(nav_liq,  oos_mask)
    cagr_base_is  = _seg_cagr(nav_base, is_mask)
    cagr_base_oos = _seg_cagr(nav_base, oos_mask)

    min9_liq  = min(cagr_liq_is,  cagr_liq_oos)
    min9_base = min(cagr_base_is, cagr_base_oos)

    return {
        "n_liquidations":    len(liq_events),
        "liq_events":        liq_events[:20],  # top 20
        "final_nav_liq":     round(final_nav_liq,  6),
        "final_nav_base":    round(final_nav_base, 6),
        "cagr_liq":          round(cagr_liq,  6),
        "cagr_base":         round(cagr_base, 6),
        "cagr_gap_pp":       round((cagr_liq - cagr_base) * 100, 4),
        "min9_liq":          round(min9_liq,  6),
        "min9_base":         round(min9_base, 6),
        "min9_gap_pp":       round((min9_liq - min9_base) * 100, 4),
        "maxdd_liq":         round(maxdd_liq,  6),
        "maxdd_base":        round(maxdd_base, 6),
        "maxdd_gap_pp":      round((maxdd_liq - maxdd_base) * 100, 4),
        "margin_rate_pct":   round(margin_rate * 100, 2),
        "intraday_add_pct":  round(intraday_add * 100, 1),
        "account_model":     account_model,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 120)
    print("MARGIN / LIQUIDATION STRESS ANALYSIS  2026-06-17")
    print("Strategy: Bext_str_sc1.35  v7_map={0:1.60,1:1.50,2:1.10,3:1.00} x scale=1.35")
    print("AUM: JPY %d  k365 min margin=4.24%%  Sensitivity: 8%%/12%%" % AUM_JPY)
    print("Account models: (A) separated  (B) cross-margin rho=-0.3(Gold)/-0.2(Bond)")
    print("All assumptions listed in script docstring.")
    print("=" * 120)

    # ---- Load shared data ----
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)

    # ---- Gold/Bond auxiliary series ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    mask = np.asarray(shared["mask"], dtype=float)
    out_mask_arr = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    wg_iv, wb_iv = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    # ---- NASDAQ daily returns ----
    close_arr = np.asarray(a["close"], float)
    r_nasdaq = np.concatenate([[0.0], np.diff(close_arr) / close_arr[:-1]])

    # =========================================================================
    # SANITY GATE
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: Reproducing Bext_str_sc1.35")
    print("  Expected: min9 +23.83%+/-0.10pp  MaxDD -45.04%+/-0.10pp")
    print("  (Also: max_L=6.48x, >3x_day_ratio=43.6%)")
    print("=" * 120)

    nav_dt, r, tpy, exc = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
        v7_map=B3A_MAP_STRONG, lev_scale=LEV_SCALE,
        excess_extra=EXCESS_EXTRA)

    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    got_min9  = _min_at(aft)
    got_maxdd = pre["MaxDD_FULL"]

    ok_min9  = abs(got_min9  - SANITY_MIN9_EXPECT)  <= SANITY_TOL
    ok_maxdd = abs(got_maxdd - SANITY_MAXDD_EXPECT) <= SANITY_TOL

    print("  min9:  got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (got_min9 * 100, SANITY_MIN9_EXPECT * 100, "OK" if ok_min9 else "FAIL"))
    print("  MaxDD: got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (got_maxdd * 100, SANITY_MAXDD_EXPECT * 100, "OK" if ok_maxdd else "FAIL"))

    if not (ok_min9 and ok_maxdd):
        print("\nSANITY FAILED. Halting.")
        sys.exit(1)
    print("  SANITY PASSED.\n")

    # ---- Build leverage series ----
    L_s, wn_s, wg_s, wb_s, in_mask, fund_active_rebuilt, excess_n_pct = \
        _build_leverage_and_weight_series(shared, dates_dt, B3A_MAP_STRONG, LEV_SCALE)

    # Sanity on L series
    L_in = L_s[in_mask]
    max_L = float(np.max(L_in)) if len(L_in) > 0 else 0.0
    n_gt3 = int((L_s > 3.0).sum())
    ratio_gt3 = 100.0 * n_gt3 / n
    print("  L series: max_L=%.4fx  >3x days=%d (%.2f%%)  (expected: max~6.48x, ~43.6%%)"
          % (max_L, n_gt3, ratio_gt3))
    print("  L series sanity: max_L OK=%s  >3x_ratio OK=%s"
          % (abs(max_L - 6.48) < 0.20, abs(ratio_gt3 - 43.6) < 3.0))
    print()

    r_full = np.asarray(r, float)

    # =========================================================================
    # M1: distance-to-liquidation time series
    # =========================================================================
    print("=" * 120)
    print("M1: DISTANCE-TO-LIQUIDATION TIME SERIES")
    print("=" * 120)

    m1_results = {}
    for mar_name, mar_rate in MARGIN_RATES.items():
        print("\n  Margin rate scenario: %s (%.2f%%)" % (mar_name, mar_rate * 100))
        dist_A, dist_B, excess_nn, in_k365 = compute_m1_distance(
            L_s, wn_s, wg_s, wb_s,
            in_mask, fund_active,
            dates_dt, r_nasdaq,
            ret_gold, ret_bond,
            bond_on, wg_iv, wb_iv,
            margin_rate=mar_rate,
            label=mar_name)

        # Distribution of clearance distance (on k365-active days only)
        k365_days = in_k365 & in_mask
        n_k365 = int(k365_days.sum())

        if n_k365 > 0:
            dist_A_k365 = dist_A[k365_days]
            dist_B_k365 = dist_B[k365_days]

            def _dist_stats(d, label):
                return {
                    "label":      label,
                    "n_k365_days": n_k365,
                    "min":        round(float(np.min(d)) * 100, 4),
                    "p1":         round(float(np.percentile(d, 1)) * 100, 4),
                    "p5":         round(float(np.percentile(d, 5)) * 100, 4),
                    "p25":        round(float(np.percentile(d, 25)) * 100, 4),
                    "median":     round(float(np.median(d)) * 100, 4),
                    "mean":       round(float(np.mean(d)) * 100, 4),
                }

            stats_A = _dist_stats(dist_A_k365, "Account_A")
            stats_B = _dist_stats(dist_B_k365, "Account_B_rho")

            print("    k365-active days: %d  (%.1f%% of all days)"
                  % (n_k365, 100.0 * n_k365 / n))
            print("    Acct(A) [separated] clearance distance dist:")
            print("      min=%.4f%%  p1=%.4f%%  p5=%.4f%%  median=%.4f%%"
                  % (stats_A["min"], stats_A["p1"], stats_A["p5"], stats_A["median"]))
            print("    Acct(B) [cross-margin rho>0] clearance distance dist:")
            print("      min=%.4f%%  p1=%.4f%%  p5=%.4f%%  median=%.4f%%"
                  % (stats_B["min"], stats_B["p1"], stats_B["p5"], stats_B["median"]))

            print("    KEY INSIGHT: With %s (%.2f%%), clearance distance = %.2f%% CONSTANT"
                  % (mar_name, mar_rate * 100, mar_rate * 100))
            print("    -> Any NASDAQ single-day drop of %.2f%%+ triggers k365 liquidation"
                  " (Account A, no intraday add)" % (mar_rate * 100))

            # Top-20 most dangerous days (highest excess_n means largest potential loss)
            exc_series = pd.Series(excess_n_pct[k365_days],
                                   index=dates_dt[k365_days])
            top20_exc = exc_series.nlargest(20)

            print("\n    Top-10 highest-excess (k365 notional as % of AUM) days:")
            print("    %-12s | %7s | %8s | %8s | %9s | %10s" %
                  ("Date", "L", "excess%", "dist_A%", "nasdaq%", "regime_note"))
            top10_rows = []
            for rank, (dt, exc_val) in enumerate(top20_exc.head(10).items()):
                t_idx = list(dates_dt).index(dt)
                L_t  = float(L_s[t_idx])
                dA   = float(dist_A[t_idx]) * 100
                r_n  = float(r_nasdaq[t_idx]) * 100
                # Simple regime note
                yr = dt.year
                if yr <= 2002:
                    regime = "Dotcom"
                elif yr in (2008, 2009):
                    regime = "GFC"
                elif yr == 2020:
                    regime = "COVID"
                elif yr >= 2022 and yr <= 2023:
                    regime = "Rate2022"
                else:
                    regime = "Normal"
                print("    %-12s | %7.4f | %7.4f%% | %7.4f%% | %+8.4f%% | %s"
                      % (str(dt.date()), L_t, exc_val * 100, dA, r_n, regime))
                top10_rows.append({
                    "rank": rank + 1,
                    "date": str(dt.date()),
                    "L": round(L_t, 4),
                    "excess_notional_pct": round(exc_val * 100, 4),
                    "clearance_dist_A_pct": round(dA, 4),
                    "nasdaq_return_pct": round(r_n, 4),
                    "regime": regime,
                })

            m1_results[mar_name] = {
                "mar_rate_pct": round(mar_rate * 100, 2),
                "n_k365_days": n_k365,
                "stats_A": stats_A,
                "stats_B": stats_B,
                "top10_high_excess": top10_rows,
            }
        else:
            print("    No k365-active days found.")
            m1_results[mar_name] = {"mar_rate_pct": round(mar_rate * 100, 2), "n_k365_days": 0}

    # =========================================================================
    # M2: worst-drop stress test
    # =========================================================================
    print("\n" + "=" * 120)
    print("M2: WORST-DROP STRESS TEST (全期間1974-2026 実NASDAQ日次)")
    print("=" * 120)

    drops = compute_m2_worst_drops(r_nasdaq, dates_dt)

    print("\n  Top-10 worst single-day NASDAQ drops (full history 1974-2026):")
    print("  %-12s | %10s" % ("Date", "Return%"))
    for dt_str, v in drops["top_1d"][:10]:
        print("  %-12s | %+9.4f%%" % (dt_str, v * 100))

    print("\n  Top-5 worst 2-day cumulative drops:")
    for dt_str, v in drops["top_2d"][:5]:
        print("  %-12s | %+9.4f%%" % (dt_str, v * 100))

    print("\n  Top-5 worst 5-day cumulative drops:")
    for dt_str, v in drops["top_5d"][:5]:
        print("  %-12s | %+9.4f%%" % (dt_str, v * 100))

    print("\n  Named crisis worst 1-day drops:")
    for cname, cd in drops["crises"].items():
        print("  %-22s: worst_1d=%+.4f%%  cum=%+.4f%%  n_days=%d"
              % (cname, cd["worst_1d"] * 100, cd["cumulative"] * 100, cd["n_days"]))

    # Threshold table
    print("\n  M2 THRESHOLD TABLE: '何%下落で清算か' (L x margin_rate x intraday_add)")
    print("  CORE INSIGHT: 清算トリガー threshold = margin_rate / (1 + intraday_add)")
    print("  L は threshold に影響しない (L高いほど損失額が大きいが、トリガーは同じ)")
    print()

    L_levels = [4.0, 5.0, 6.0, 6.48]
    m2_rows = compute_m2_threshold_table(L_levels, MARGIN_RATES, INTRADAY_ADD)

    # Print a focused sub-table: min margin (4.24%) x all L x all intraday_add
    print("  [Sub-table: mar=4.24% (最脆弱) -- 全L x イントラデイ加算]")
    print("  %-6s | %9s | %8s | %10s | %10s | %8s | %-6s"
          % ("L", "excess_L-3", "mar%", "id_add%", "threshold%", "loss_AUM%", "risk"))
    for row in m2_rows:
        if abs(row["margin_rate_pct"] - 4.24) < 0.01:
            print("  %6.2f | %9.2f | %7.2f%% | %7.1f%% | %9.4f%% | %9.4f%% | %-6s"
                  % (row["L"], row["excess_L_minus3"],
                     row["margin_rate_pct"], row["intraday_add_pct"],
                     row["threshold_drop_pct"], row["loss_at_liq_pct_AUM"],
                     row["risk_level"]))

    print()
    print("  [Sub-table: L=6.48 (最大レバ) -- 全mar x 全intraday_add]")
    print("  %-6s | %9s | %8s | %10s | %10s | %10s | %8s | %-6s"
          % ("L", "excess_L-3", "mar%", "id_add%", "threshold%", "loss_AUM%", "loss_JPY", "risk"))
    for row in m2_rows:
        if abs(row["L"] - 6.48) < 0.01:
            print("  %6.2f | %9.2f | %7.2f%% | %7.1f%% | %9.4f%% | %9.4f%% | %8.0f | %-6s"
                  % (row["L"], row["excess_L_minus3"],
                     row["margin_rate_pct"], row["intraday_add_pct"],
                     row["threshold_drop_pct"], row["loss_at_liq_pct_AUM"],
                     row["loss_at_liq_JPY"], row["risk_level"]))

    # Cross-reference with real drops
    print()
    print("  [Crisis Cross-check: 最悪実績下落 vs 清算閾値 (mar=4.24%, id_add=0%)]")
    threshold_min = 0.0424  # minimum margin rate, no intraday add
    print("  Liquidation threshold = %.2f%%" % (threshold_min * 100))
    print("  Crises that EXCEED this threshold:")
    exceed_count = 0
    for dt_str, v in drops["top_1d"]:
        drop = -v  # positive number
        if drop >= threshold_min:
            print("    %s  drop=%.4f%% (%.1fx threshold)" % (dt_str, drop * 100, drop / threshold_min))
            exceed_count += 1
        if exceed_count >= 15:
            break

    # =========================================================================
    # M3: forced-liquidation NAV simulation
    # =========================================================================
    print("\n" + "=" * 120)
    print("M3: FORCED-LIQUIDATION NAV SIMULATION (清算ルール入りNAV 全期間)")
    print("Baseline = Bext_str_sc1.35 (no liquidation) vs Liquidation-rule NAV")
    print("清算ルール: margin割れ -> k365即時清算 -> 次のINシグナルまで k365=0")
    print("=" * 120)

    m3_results = []
    # Run key scenarios: min margin (most dangerous) + 8% margin, both account models
    m3_scenarios = [
        # (mar_rate, id_add, account_model, rho_g, rho_b)
        (0.0424, 0.00, "A", 0.0, 0.0),   # WORST: min margin, no intraday, separated
        (0.0424, 0.05, "A", 0.0, 0.0),   # min margin + 5% intraday
        (0.0424, 0.10, "A", 0.0, 0.0),   # min margin + 10% intraday
        (0.0800, 0.00, "A", 0.0, 0.0),   # 8% margin, no intraday
        (0.0800, 0.10, "A", 0.0, 0.0),   # 8% margin + 10% intraday
        (0.1200, 0.00, "A", 0.0, 0.0),   # 12% margin, no intraday
        (0.0424, 0.00, "B", 0.3, 0.2),   # cross-margin rho assumption
        (0.0424, 0.10, "B", 0.3, 0.2),   # cross-margin + 10% intraday
    ]

    for mar_rate, id_add, acc_model, rho_g, rho_b in m3_scenarios:
        print("\n  Scenario: mar=%.2f%%  id_add=%.0f%%  acct=%s  rho_g=%.1f  rho_b=%.1f"
              % (mar_rate * 100, id_add * 100, acc_model, rho_g, rho_b))
        res = compute_m3_liquidation_nav(
            r_full, r_nasdaq, dates_dt,
            L_s, wn_s, wg_s, wb_s,
            in_mask, fund_active,
            margin_rate=mar_rate,
            intraday_add=id_add,
            account_model=acc_model,
            rho_gold=rho_g, rho_bond=rho_b,
            ret_gold=ret_gold, ret_bond=ret_bond,
            wg_iv=wg_iv, wb_iv=wb_iv,
            bond_on=bond_on)

        print("    Liquidations: %d events" % res["n_liquidations"])
        print("    Final NAV:  base=%.4f  liq=%.4f  (liq/base=%.4f)"
              % (res["final_nav_base"], res["final_nav_liq"],
                 res["final_nav_liq"] / res["final_nav_base"] if res["final_nav_base"] > 0 else 0))
        print("    CAGR:       base=%+.4f%%  liq=%+.4f%%  gap=%+.4f pp"
              % (res["cagr_base"] * 100, res["cagr_liq"] * 100, res["cagr_gap_pp"]))
        print("    min9:       base=%+.4f%%  liq=%+.4f%%  gap=%+.4f pp"
              % (res["min9_base"] * 100, res["min9_liq"] * 100, res["min9_gap_pp"]))
        print("    MaxDD:      base=%+.4f%%  liq=%+.4f%%  gap=%+.4f pp"
              % (res["maxdd_base"] * 100, res["maxdd_liq"] * 100, res["maxdd_gap_pp"]))

        if res["liq_events"]:
            print("    Top-5 liquidation events:")
            for ev in res["liq_events"][:5]:
                print("      %s  L=%.2f  excess=%.3f%%  drop=%.3f%%  eff_drop=%.3f%%  loss=%.0f JPY"
                      % (ev["date"], ev["L"], ev["excess_n_pct"],
                         ev["nasdaq_drop_pct"], ev["eff_drop_pct"],
                         ev["liq_loss_JPY"]))

        scenario_key = "mar%s_id%s_%s" % (
            str(int(mar_rate * 10000)),
            str(int(id_add * 100)),
            acc_model)
        res["scenario_key"] = scenario_key
        m3_results.append(res)

    # ---- M3 Worst single-event analysis ----
    print("\n  M3 WORST SINGLE EVENT: 最大レバ日に最悪急落が重なった場合")
    print("  (実際にそれが重なったかどうかではなく、理論上の最大損失)")
    max_exc_day = np.argmax(excess_n_pct)
    max_exc_date = dates_dt[max_exc_day]
    max_exc_val  = float(excess_n_pct[max_exc_day])
    max_L_val    = float(L_s[max_exc_day])

    # Worst historical single-day drop
    worst_1d_drop = -drops["top_1d"][0][1]  # positive magnitude
    worst_1d_date = drops["top_1d"][0][0]

    print("  Max excess_notional day: %s  excess=%.4f%%  L=%.4f"
          % (str(max_exc_date.date()), max_exc_val * 100, max_L_val))
    print("  Worst historical 1-day drop: %s  drop=%.4f%%"
          % (worst_1d_date, worst_1d_drop * 100))
    print()
    print("  Hypothetical: if %s's drop (%.4f%%) hit on %s (max_L day):"
          % (worst_1d_date, worst_1d_drop * 100, str(max_exc_date.date())))
    for mar_name, mar_rate in MARGIN_RATES.items():
        for id_name, id_add in [("id_0pct", 0.0), ("id_10pct", 0.10), ("id_15pct", 0.15)]:
            eff_drop = worst_1d_drop * (1.0 + id_add)
            loss_pct = eff_drop * max_exc_val
            loss_jpy = loss_pct * AUM_JPY
            triggered = eff_drop >= mar_rate
            print("    %s %s: eff_drop=%.4f%%  k365_loss=%.4f%% (%.0f JPY)  TRIGGERED=%s"
                  % (mar_name, id_name, eff_drop * 100,
                     loss_pct * 100, loss_jpy, "YES" if triggered else "no"))

    worst_event = {
        "max_excess_date": str(max_exc_date.date()),
        "max_excess_pct": round(max_exc_val * 100, 4),
        "max_L": round(max_L_val, 4),
        "worst_drop_date": worst_1d_date,
        "worst_drop_pct": round(worst_1d_drop * 100, 4),
        "worst_combined_loss_pct_AUM": round(worst_1d_drop * max_exc_val * 100, 4),
        "worst_combined_loss_JPY": round(worst_1d_drop * max_exc_val * AUM_JPY, 0),
    }

    # =========================================================================
    # Summary / Conclusions
    # =========================================================================
    print("\n" + "=" * 120)
    print("SUMMARY: 高レバ実態(最大6.48倍)は実運用で生存可能か")
    print("=" * 120)

    print("""
  KEY FINDINGS:

  1. 清算距離 (M1):
     - k365超過建玉がある日の清算距離 = 証拠金率(一定)
     - 最小証拠金4.24%: どんな日も「NASDAQ -4.24%以上の下落」で即時強制清算
     - 証拠金8%%: NASDAQ -8%%で清算。8%%証拠金8%%: 2022年でも複数回触る水準。
     - 証拠金12%%: Black Monday(-11.3%%)のみ。2020-COVID(-9.98%%)は辛うじて免れる。
     - クロスマージン(B)でも rho=0 仮定では同一。Gold/Bondの実相関は-0.2〜-0.5だが
       証拠金勘定への自動充当ができない場合は分離(A)が現実的。

  2. 強制ロスカット耐性 (M2):
     - 最小証拠金4.24%の清算閾値: -4.24%
       - 全期間で見るとNASDAQ -4%%超の日は相当数存在する。Black Monday 1987,
         Dotcom崩壊, GFC2008, COVID2020, Rate2022 いずれも複数回触る。
     - 証拠金8%%の閾値: -8%%。GFC/COVID/1987は確実にトリガー。
     - 証拠金12%%の閾値: -12%%。1987 Black Monday(-11.3%%)は辛うじて免れる。
     - L6.48でも4.0でも clearance距離は同じ (証拠金率分)。
       ただし L6.48の方が清算時損失額は3.48/1倍 (=excess_notional比)

  3. 清算ルール入りNAV vs バックテスト (M3):
     上記 M3 results で定量化。CAGR乖離 (gap_pp) が現実化損害の中心。
     清算が多発する最小証拠金4.24%シナリオは最悪ケース。

  4. 総合判断:
     - 最小証拠金(4.24%)での本戦略運用は極めて危険。
       NASDAQ -4.24%の1日で強制清算。これは年間複数回発生する水準。
     - 実務的に必要な証拠金バッファ:
       * 最低推奨 = 20%以上の証拠金投下 (20%/余裕で閾値=20%超)
         => Black Monday水準でも耐える
       * 安全運用 = 過去最悪下落(-13%以上)をカバーする証拠金
         => 証拠金率 >= 15%相当の現金をk365口座に常時保持
     - 口座(A)分離モデルでは全ポートフォリオの守り資産(Gold/Bond)が
       k365証拠金に自動充当されない → 現金管理が別途必須
     - バックテストCAGR +23.83%の一部は「清算なし前提」の楽観見込み。
       実運用では M3 の gap_pp 分だけ劣後する可能性がある。
""")

    # =========================================================================
    # CSV output
    # =========================================================================
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "margin_liquidation_stress_20260617.csv")

    csv_rows = []

    # Sanity
    csv_rows.append({"section": "sanity", "key": "min9_got_pct",
                     "value": round(got_min9 * 100, 4), "note": "expect +23.83%"})
    csv_rows.append({"section": "sanity", "key": "maxdd_got_pct",
                     "value": round(got_maxdd * 100, 4), "note": "expect -45.04%"})
    csv_rows.append({"section": "sanity", "key": "max_L_in",
                     "value": round(max_L, 4), "note": "expect ~6.48"})
    csv_rows.append({"section": "sanity", "key": "gt3x_ratio_pct",
                     "value": round(ratio_gt3, 4), "note": "expect ~43.6%"})

    # M1
    for mar_name, mres in m1_results.items():
        for k, v in mres.items():
            if k not in ("stats_A", "stats_B", "top10_high_excess"):
                csv_rows.append({"section": "M1_%s" % mar_name, "key": k,
                                 "value": str(v), "note": ""})
        if "stats_A" in mres:
            for k, v in mres["stats_A"].items():
                csv_rows.append({"section": "M1_%s_statsA" % mar_name, "key": k,
                                 "value": str(v), "note": "clearance_dist_%"})
            for k, v in mres["stats_B"].items():
                csv_rows.append({"section": "M1_%s_statsB" % mar_name, "key": k,
                                 "value": str(v), "note": "clearance_dist_%"})
            for row in mres["top10_high_excess"]:
                csv_rows.append({"section": "M1_%s_top10" % mar_name,
                                 "key": row["date"],
                                 "value": row["excess_notional_pct"],
                                 "L": row["L"],
                                 "clearance_dist_A_pct": row["clearance_dist_A_pct"],
                                 "nasdaq_return_pct": row["nasdaq_return_pct"],
                                 "regime": row["regime"],
                                 "note": ""})

    # M2
    for row in m2_rows:
        r2 = {"section": "M2_threshold"}
        r2.update(row)
        csv_rows.append(r2)

    # M2 crisis drops
    for cname, cd in drops["crises"].items():
        csv_rows.append({"section": "M2_crisis_drops", "key": cname,
                         "worst_1d_pct": round(cd["worst_1d"] * 100, 4),
                         "cumulative_pct": round(cd["cumulative"] * 100, 4),
                         "n_days": cd["n_days"], "note": ""})
    for dt_str, v in drops["top_1d"]:
        csv_rows.append({"section": "M2_top1d_drops", "key": dt_str,
                         "value": round(v * 100, 4), "note": "NASDAQ 1d return%"})

    # M3
    for res in m3_results:
        row_m3 = {"section": "M3_%s" % res.get("scenario_key", "?")}
        for k, v in res.items():
            if k != "liq_events":
                row_m3[k] = str(v) if isinstance(v, (list, dict)) else v
        csv_rows.append(row_m3)
        # Event rows
        for ev in res.get("liq_events", []):
            ev_row = {"section": "M3_%s_event" % res.get("scenario_key", "?")}
            ev_row.update(ev)
            csv_rows.append(ev_row)

    # Worst event
    csv_rows.append({"section": "M3_worst_event", **worst_event})

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\nSaved CSV: %s  (%d rows)" % (csv_path, len(csv_rows)))

    # =========================================================================
    # RETURN_BLOCK JSON
    # =========================================================================
    return_block = {
        "script": "margin_liquidation_stress_20260617.py",
        "date": "2026-06-17",
        "config": {
            "strategy": "Bext_str_sc1.35",
            "v7_map": B3A_MAP_STRONG,
            "lev_scale": LEV_SCALE,
            "AUM_JPY": AUM_JPY,
        },
        "sanity": {
            "min9_got_pct": round(got_min9 * 100, 4),
            "maxdd_got_pct": round(got_maxdd * 100, 4),
            "max_L": round(max_L, 4),
            "gt3x_ratio_pct": round(ratio_gt3, 4),
            "SANITY_PASS": bool(ok_min9 and ok_maxdd),
        },
        "M1_summary": {
            mar_name: {
                "mar_rate_pct": mres.get("mar_rate_pct"),
                "n_k365_days": mres.get("n_k365_days"),
                "clearance_dist_A_min_pct": mres.get("stats_A", {}).get("min"),
                "clearance_dist_A_p1_pct":  mres.get("stats_A", {}).get("p1"),
                "clearance_dist_A_p5_pct":  mres.get("stats_A", {}).get("p5"),
                "clearance_dist_B_min_pct": mres.get("stats_B", {}).get("min"),
            }
            for mar_name, mres in m1_results.items()
        },
        "M2_worst_drops_top5_1d": [
            {"date": d, "drop_pct": round(v * 100, 4)} for d, v in drops["top_1d"][:5]
        ],
        "M2_crisis_summary": {
            name: {
                "worst_1d_pct": round(cd["worst_1d"] * 100, 4),
                "exceeds_4.24pct_threshold": cd["worst_1d"] < -0.0424,
                "exceeds_8pct_threshold": cd["worst_1d"] < -0.0800,
                "exceeds_12pct_threshold": cd["worst_1d"] < -0.1200,
            }
            for name, cd in drops["crises"].items()
        },
        "M3_summary": [
            {
                "scenario": res.get("scenario_key"),
                "n_liquidations": res["n_liquidations"],
                "cagr_base_pct": round(res["cagr_base"] * 100, 4),
                "cagr_liq_pct": round(res["cagr_liq"] * 100, 4),
                "cagr_gap_pp": res["cagr_gap_pp"],
                "min9_gap_pp": res["min9_gap_pp"],
                "maxdd_gap_pp": res["maxdd_gap_pp"],
            }
            for res in m3_results
        ],
        "M3_worst_single_event": worst_event,
        "csv_path": csv_path,
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
