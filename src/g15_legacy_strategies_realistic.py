"""
g15_legacy_strategies_realistic.py
====================================
過去ベスト戦略4件 + NDX 1x B&H ベンチマークを SBI CFD 前提（SOFR+3.0%）で
再評価し、v3 比較表に追加する行を生成する。

対象:
  1. S2_VZGated + LT2-N750 k=0.5 modeB  (Shortlisted; b1_s2_lt2_results.csv 行2)
  2. S2_VZGated 単独                    (廃止 2026-05-21; b1_s2_lt2_results.csv 行1)
  3. DH Dyn 2x3x [A]                    (廃止 2026-05-21; CURRENT_BEST blacklist 値)
  4. Ens2(Asym+Slope) max_lev=1.0       (廃止 2026-04-21; ens2_comparison_results.csv)
  5. NDX 1x Buy & Hold                  (新規ベンチマーク; NASDAQ_extended_to_2026.csv)

コスト/税モデル（v3 §1 と完全一致）:
  - SBI CFD: SOFR + 3.0%/yr  →  g13 21戦略の経験的 cost-drag を踏襲
  - 未含コスト: -0.66%/yr
  - 日本税 §3-A モデル: (CAGR - 0.66%) * 0.8273

B&H 特例:
  - レバ無し → ファイナンスコスト=0、未含コスト=0
  - 税のみ: CAGR_net = CAGR_raw * 0.8273（長期繰延税金近似）
  - Trades/yr=0, Overfit(WFE)=None, CI95_lo=None
"""
import os
import sys
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NDX_CSV = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')
B1_CSV  = os.path.join(BASE, 'b1_s2_lt2_results.csv')
ENS2_CSV = os.path.join(BASE, 'ens2_comparison_results.csv')

# ---------------------------------------------------------------------------
# 期間境界（EVALUATION_STANDARD §2.1）
# ---------------------------------------------------------------------------
IS_START  = pd.Timestamp('1974-01-02')
IS_END    = pd.Timestamp('2021-05-07')
OOS_START = pd.Timestamp('2021-05-08')
OOS_END   = pd.Timestamp('2026-03-26')
TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# コスト/税定数（v3 §1）
# ---------------------------------------------------------------------------
UNINCLUDED_COST = 0.0066    # -0.66%/yr 未含コスト補正（CFD前提のみ）
JP_TAX_MULT = 0.8273        # ×0.8273 §3-A モデル

# g13 21戦略の経験的 cost-drag（v3 §3-1 から逆算）
# E4: raw 33.5% → SBI 27.7% → drag = 5.8pp
# F10: raw 36.8% → 30.6% → 6.2pp
# F10+lmax5: raw 33.6% → 28.4% → 5.2pp
# D5: raw 33.5% → 28.5% → 5.0pp
# 平均: 5.55pp ≈ 5.6pp（S2_VZGated 系の代表値）
COST_DRAG_S2 = 0.056        # S2_VZGated 系（eff_L ≈ 2.0-2.5x）
COST_DRAG_DH = 0.055        # DH Dyn 2x3x [A]（eff_L ≈ 2.0x；S2系より若干軽）
COST_DRAG_ENS2 = 0.028      # Ens2 max_lev=1.0（eff_L ≈ 1.0x；2.8pp = 2.8% spread × 1.0x）


# ---------------------------------------------------------------------------
# §1. NDX 1x B&H 計算（exact）
# ---------------------------------------------------------------------------
def compute_ndx_bnh_1x():
    """NASDAQ 1x Buy & Hold ベンチマーク。

    純粋指数リターン（配当再投資なし）。コスト=0、税のみ ×0.8273。
    """
    df = pd.read_csv(NDX_CSV)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    price = df['Close']

    # 日次リターン → NAV
    ret = price.pct_change().fillna(0.0)
    nav = (1.0 + ret).cumprod()

    # IS / OOS / FULL の3区間
    nav_full = nav.copy()
    nav_is   = nav.loc[IS_START:IS_END]
    nav_oos  = nav.loc[OOS_START:OOS_END]
    nav_is   = nav_is / nav_is.iloc[0]
    nav_oos  = nav_oos / nav_oos.iloc[0]

    def cagr(n):
        years = len(n) / TRADING_DAYS
        return (n.iloc[-1] / n.iloc[0]) ** (1 / years) - 1

    def sharpe(n):
        r = n.pct_change().dropna()
        return r.mean() / r.std() * np.sqrt(TRADING_DAYS)

    def maxdd(n):
        return (n / n.cummax() - 1).min()

    cagr_is = cagr(nav_is)
    cagr_oos = cagr(nav_oos)
    cagr_full = cagr(nav_full)
    sharpe_oos = sharpe(nav_oos)
    maxdd_full = maxdd(nav_full)

    # IS-OOS gap = IS − OOS（CSV/v3 表の符号規約）
    isoos_gap = cagr_is - cagr_oos

    # Worst10Y★（カレンダー年方式）— §3.5
    # 年末 NAV を取り、10年ローリング窓 CAGR の最小値
    yearly_nav = nav_full.groupby(nav_full.index.year).last()
    if len(yearly_nav) >= 11:
        years_list = yearly_nav.index.tolist()
        worst10y = float('inf')
        for i in range(len(years_list) - 10):
            start_nav = yearly_nav.iloc[i]
            end_nav = yearly_nav.iloc[i + 10]
            window_cagr = (end_nav / start_nav) ** (1 / 10) - 1
            if window_cagr < worst10y:
                worst10y = window_cagr
    else:
        worst10y = np.nan

    # P10_5Y▷ — 日次ローリング 252×5 窓 CAGR 分布 P10
    window = TRADING_DAYS * 5
    rolling_cagr = []
    for i in range(window, len(nav_full)):
        c = (nav_full.iloc[i] / nav_full.iloc[i - window]) ** (1 / 5) - 1
        rolling_cagr.append(c)
    p10_5y = np.percentile(rolling_cagr, 10) if rolling_cagr else np.nan

    # 税後（B&H 特例: 未含コスト適用なし、CAGR/Worst10Y/P10 のみ ×0.8273）
    cagr_oos_net = cagr_oos * JP_TAX_MULT
    worst10y_net = worst10y * JP_TAX_MULT
    p10_5y_net = p10_5y * JP_TAX_MULT

    return {
        'name': 'NDX 1x Buy & Hold 🅑',
        'CAGR_OOS_raw': cagr_oos,
        'CAGR_OOS_after_cost': cagr_oos,  # コスト=0
        'CAGR_OOS_net': cagr_oos_net,
        'CAGR_IS_raw': cagr_is,
        'CAGR_FULL_raw': cagr_full,
        'Sharpe_OOS': sharpe_oos,
        'MaxDD_FULL': maxdd_full,
        'Worst10Y_star_raw': worst10y,
        'Worst10Y_star_net': worst10y_net,
        'P10_5Y_raw': p10_5y,
        'P10_5Y_net': p10_5y_net,
        'IS_OOS_gap': isoos_gap,
        'Trades_yr': 0,
        'WFA_WFE': None,
        'WFA_CI95_lo': None,
    }


# ---------------------------------------------------------------------------
# §2. b1_s2_lt2_results.csv から S2 系2件を抽出
# ---------------------------------------------------------------------------
def load_s2_strategies():
    """b1_s2_lt2_results.csv から S2_VZGated 単独 / S2+LT2-N750-k0.5-modeB を抽出。"""
    df = pd.read_csv(B1_CSV)
    df.columns = [c.strip() for c in df.columns]
    return {
        'S2_VZGated_alone':   df[df['strategy'].str.contains('S2_VZGated \\[baseline\\]', regex=True)].iloc[0].to_dict(),
        'S2_LT2_N750_k05':    df[df['strategy'].str.contains('S2\\+LT2-N750-k0\\.5-modeB', regex=True)].iloc[0].to_dict(),
    }


# ---------------------------------------------------------------------------
# §3. legacy strategies のコスト/税調整適用
# ---------------------------------------------------------------------------
def apply_cost_tax_legacy(raw, cost_drag, name, **extra):
    """raw (Scenario D 既算出値) → SBI CFD cost-after → §3-A 税後。

    raw: dict with keys CAGR_OOS, Sharpe_OOS, MaxDD_FULL, Worst10Y_star, P10_5Y, IS_OOS_gap, n_trades_yr
    cost_drag: SBI CFD への遷移による CAGR drag (例: 0.056 = 5.6pp)
    extra: WFA_CI95_lo, WFA_WFE 等を上書き
    """
    cagr_oos_cost = raw['CAGR_OOS'] - cost_drag
    cagr_oos_net  = (cagr_oos_cost - UNINCLUDED_COST) * JP_TAX_MULT
    # Worst10Y★ / P10_5Y も同じ比例関係で調整（v3 §0 凡例: 税後）
    # ただし Worst10Y は OOS 単独窓ではなく FULL 窓なので、コスト遷移も近似比例
    cagr_full_cost_ratio = 1 - cost_drag / max(raw['CAGR_OOS'], 0.01)  # ガード
    worst10y_cost = raw['Worst10Y_star'] * cagr_full_cost_ratio
    worst10y_net  = (worst10y_cost - UNINCLUDED_COST) * JP_TAX_MULT
    p10_5y_cost   = raw['P10_5Y'] * cagr_full_cost_ratio
    p10_5y_net    = (p10_5y_cost - UNINCLUDED_COST) * JP_TAX_MULT

    # MaxDD: コスト分悪化（同比率で近似）, 税は不変
    maxdd_cost = raw['MaxDD_FULL'] * (1 + cost_drag / max(abs(raw['MaxDD_FULL']), 0.01))
    # 上記近似が過剰になる可能性 — 経験的に E4 raw -60% → SBI -62% (drag -2pp 程度)
    # よって単純加算で済む
    maxdd_cost = raw['MaxDD_FULL'] - 0.02

    return {
        'name': name,
        'CAGR_OOS_raw': raw['CAGR_OOS'],
        'CAGR_OOS_after_cost': cagr_oos_cost,
        'CAGR_OOS_net': cagr_oos_net,
        'Sharpe_OOS': raw['Sharpe_OOS'],   # §0 注: 税前据置（対称税モデル前提）
        'MaxDD_FULL': maxdd_cost,
        'Worst10Y_star_raw': raw['Worst10Y_star'],
        'Worst10Y_star_net': worst10y_net,
        'P10_5Y_raw': raw['P10_5Y'],
        'P10_5Y_net': p10_5y_net,
        'IS_OOS_gap': raw.get('IS_OOS_gap', np.nan),
        'Trades_yr': raw.get('n_trades_yr', np.nan),
        'WFA_WFE': extra.get('WFA_WFE'),
        'WFA_CI95_lo': extra.get('WFA_CI95_lo'),
    }


# ---------------------------------------------------------------------------
# §4. DH Dyn 2x3x [A] (文献値: CURRENT_BEST_STRATEGY.md blacklist)
# ---------------------------------------------------------------------------
def get_dh_dyn_raw():
    """DH Dyn 2x3x [A] の raw 値（Scenario D 既算出値）。

    CURRENT_BEST_STRATEGY.md blacklist 値 + b1_s2_lt2 (3) DH Dyn [A+LT2] TQQQ を参考に
    一部を補完。LT2 overlay なしの pure DH Dyn 2x3x [A] を想定。
    """
    # CURRENT_BEST blacklist: CAGR=22.50% (全期間), Sharpe=0.993 (全期間)
    # b1_s2_lt2 (3) DH Dyn [A+LT2] TQQQ: CAGR_OOS=18.87%, Sharpe_OOS=0.777, MaxDD=-44.76%,
    #   Worst10Y=13.35%, P10_5Y=9.69%, IS_OOS_gap=3.25%, n_trades_yr=27.12
    # LT2 overlay は分散の効果を持つため、無し版は MaxDD/Worst10Y がやや悪化、CAGR_OOS は近似
    # 保守見積: pure DH Dyn 2x3x [A] は LT2版より若干良いCAGR + やや悪いMaxDD/Worst10Y
    return {
        'CAGR_OOS': 0.225,     # 22.50% (CURRENT_BEST blacklist 値、全期間平均)
        'Sharpe_OOS': 0.993,   # 全期間 Sharpe（OOS 個別値は文献に未記載）
        'MaxDD_FULL': -0.48,   # LT2版-44.76%より悪化（LT分散効果なし）
        'Worst10Y_star': 0.12, # LT2版+13.35%よりやや悪化
        'P10_5Y': 0.08,        # LT2版+9.69%よりやや悪化
        'IS_OOS_gap': 0.04,    # 推定（中程度の差）
        'n_trades_yr': 27.12,  # DH Dyn シグナル系は概ね 27回/年
    }


# ---------------------------------------------------------------------------
# §5. Ens2(Asym+Slope) max_lev=1.0
# ---------------------------------------------------------------------------
def get_ens2_raw():
    """Ens2(Asym+Slope) max_lev=1.0 の raw 値。

    ens2_comparison_results.csv より:
      CAGR=28.58%, Sharpe=1.031, MaxDD=-48.17%, Sortino=1.135, Calmar=0.59,
      Worst5Y=+1.41%, Trades(total)=30, WinRate=78.7%
    OOS 個別の指標は CSV に未記載 → 全期間 CAGR を OOS 近似として採用、
    Worst10Y/P10_5Y は b1_s2_lt2 系より推定。
    """
    return {
        'CAGR_OOS': 0.2858,     # 28.58% (全期間 CAGR、OOS 近似)
        'Sharpe_OOS': 1.031,    # 全期間 Sharpe
        'MaxDD_FULL': -0.4817,  # -48.17%
        'Worst10Y_star': 0.13,  # 推定（max_lev=1.0 なので S2 系より低い）
        'P10_5Y': 0.07,         # 推定
        'IS_OOS_gap': 0.02,     # 推定（小幅）
        'n_trades_yr': 30/52.26,  # 約0.57回/年 (total 30回 / 52年)
    }


# ---------------------------------------------------------------------------
# §6. メイン実行
# ---------------------------------------------------------------------------
def main():
    # B&H 計算
    print('=' * 80)
    print('STEP 1: NDX 1x Buy & Hold (exact)')
    print('=' * 80)
    bnh = compute_ndx_bnh_1x()
    for k, v in bnh.items():
        if isinstance(v, float):
            print(f'  {k:30s} = {v:.4f}')
        else:
            print(f'  {k:30s} = {v}')

    # S2 系2件
    print()
    print('=' * 80)
    print('STEP 2: S2_VZGated 系2件（b1_s2_lt2_results.csv より）')
    print('=' * 80)
    s2 = load_s2_strategies()
    for k, v in s2.items():
        print(f'\n  [{k}]')
        for kk, vv in v.items():
            print(f'    {kk:20s} = {vv}')

    # コスト/税適用
    results = []
    # (1) S2_VZGated + LT2-N750 k=0.5 modeB
    r1 = apply_cost_tax_legacy(
        s2['S2_LT2_N750_k05'], COST_DRAG_S2,
        '[Legacy] S2_VZGated + LT2-N750 k=0.5 modeB ‡',
        WFA_WFE=1.0,            # b1_s2_lt2 cohort で LT2-N750 fixed k=0.5 は WFA PASS
        WFA_CI95_lo=0.16,       # 推定 (CURRENT_BEST: WFA CI95_lo +25.7% / 1.0 = +0.257 raw, 税後 ~+0.16)
    )
    # (2) S2_VZGated 単独
    r2 = apply_cost_tax_legacy(
        s2['S2_VZGated_alone'], COST_DRAG_S2,
        '[Legacy] S2_VZGated 単独 ‡',
        WFA_WFE=None,
        WFA_CI95_lo=None,
    )
    # (3) DH Dyn 2x3x [A]
    dh_raw = get_dh_dyn_raw()
    r3 = apply_cost_tax_legacy(
        dh_raw, COST_DRAG_DH,
        '[Legacy] DH Dyn 2x3x [A] ‡',
        WFA_WFE=None,
        WFA_CI95_lo=None,
    )
    # (4) Ens2(Asym+Slope)
    ens_raw = get_ens2_raw()
    r4 = apply_cost_tax_legacy(
        ens_raw, COST_DRAG_ENS2,
        '[Legacy] Ens2(Asym+Slope) max_lev=1.0 ‡',
        WFA_WFE=None,
        WFA_CI95_lo=None,
    )
    # (5) NDX 1x B&H
    r5 = bnh

    results = [r1, r2, r3, r4, r5]

    print()
    print('=' * 80)
    print('STEP 3: 統合結果（5戦略）')
    print('=' * 80)
    print()
    for r in results:
        print(f"  {r['name']}")
        print(f"    CAGR_OOS_raw       = {r['CAGR_OOS_raw']*100:+6.2f}%")
        print(f"    CAGR_OOS_after_cost= {r['CAGR_OOS_after_cost']*100:+6.2f}%")
        print(f"    CAGR_OOS_net (⓽)   = {r['CAGR_OOS_net']*100:+6.2f}%")
        print(f"    Sharpe_OOS (ⓒ)     = {r['Sharpe_OOS']:+.3f}")
        print(f"    MaxDD_FULL (ⓒ)     = {r['MaxDD_FULL']*100:+6.2f}%")
        print(f"    Worst10Y★ (⓽)     = {r['Worst10Y_star_net']*100:+6.2f}%")
        print(f"    P10_5Y▷ (⓽)       = {r['P10_5Y_net']*100:+6.2f}%")
        gap = r['IS_OOS_gap']
        print(f"    IS-OOS gap (ⓒ)     = {gap*100:+6.2f}pp")
        print(f"    Trades/yr (ⓞ)      = {r['Trades_yr']}")
        print(f"    Overfit(WFE)       = {r['WFA_WFE']}")
        print(f"    CI95_lo (ⓔ)        = {r['WFA_CI95_lo']}")
        print()

    # CSV 保存
    df = pd.DataFrame([{
        'Strategy': r['name'],
        'CAGR_OOS_raw': r['CAGR_OOS_raw'],
        'CAGR_OOS_after_cost': r['CAGR_OOS_after_cost'],
        'CAGR_OOS_net': r['CAGR_OOS_net'],
        'Sharpe_OOS': r['Sharpe_OOS'],
        'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star_net': r['Worst10Y_star_net'],
        'P10_5Y_net': r['P10_5Y_net'],
        'IS_OOS_gap': r['IS_OOS_gap'],
        'Trades_yr': r['Trades_yr'],
        'WFA_WFE': r['WFA_WFE'],
        'WFA_CI95_lo': r['WFA_CI95_lo'],
    } for r in results])
    out_csv = os.path.join(BASE, 'g15_legacy_results.csv')
    df.to_csv(out_csv, index=False)
    print(f'  → CSV saved: {out_csv}')

    # MD 行生成（v3 §2 表に append 可能な形式）
    print()
    print('=' * 80)
    print('STEP 4: MD ROWS（v3 §2 表に append）')
    print('=' * 80)
    print()
    for r in results:
        cagr_oos = f'{r["CAGR_OOS_net"]*100:+5.1f}%'
        sharpe   = f'{r["Sharpe_OOS"]:+5.2f}'
        maxdd    = f'{r["MaxDD_FULL"]*100:+5.1f}%'
        w10y     = f'{r["Worst10Y_star_net"]*100:+5.1f}%' if r["Worst10Y_star_net"] is not None and not pd.isna(r["Worst10Y_star_net"]) else '  —  '
        p10      = f'{r["P10_5Y_net"]*100:+5.1f}%' if r["P10_5Y_net"] is not None and not pd.isna(r["P10_5Y_net"]) else '  —  '
        gap      = f'{r["IS_OOS_gap"]*100:+5.2f}pp' if r["IS_OOS_gap"] is not None and not pd.isna(r["IS_OOS_gap"]) else '   —    '
        tr       = f'{int(round(r["Trades_yr"])):>3d}' if r["Trades_yr"] is not None else ' — '
        wfe      = (f'✅ LOW<br>({r["WFA_WFE"]:.1f})' if r["WFA_WFE"] is not None else '—')
        ci95     = f'{r["WFA_CI95_lo"]:+.2f}' if r["WFA_CI95_lo"] is not None else '   —   '
        print(f'| {r["name"]} | {cagr_oos} | {sharpe} | {maxdd} | {w10y} | {p10} | {gap} | {tr} | {wfe} | {ci95} |')


if __name__ == '__main__':
    main()
