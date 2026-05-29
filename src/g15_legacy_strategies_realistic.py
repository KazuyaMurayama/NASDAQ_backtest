"""
g15_legacy_strategies_realistic.py — v2 (2026-05-29)
====================================================
過去ベスト戦略4件 + NDX 1x B&H ベンチマークを SBI CFD 前提で再評価。

v1 (g15_v1) の誤り修正:
  - DH Dyn 2x3x [A] の raw 値が FULL 期間 (22.50%/0.993) になっていた
    → corrected_strategy_results.csv の Scenario D OOS 行 (14.88%/0.646) に修正
  - Ens2(Asym+Slope) の Sharpe が FULL (1.031) になっていた
    → ADDITIONAL_ANALYSIS_REPORT_2026-03-30 の OOS Sharpe (0.479) に修正
  - 全戦略一律 5.6pp drag は誤り
    → S2系のみ -5.6pp（cfd_spread 0.20→3.0）
    → DH Dyn (TQQQ ETF) / Ens2 (1x unleveraged) は cost neutral

raw データソース（OOS Scenario D 値、確証あり）:
  S2_VZGated 単独:       b1_s2_lt2_results.csv (1)
  S2+LT2-N750 k=0.5:     b1_s2_lt2_results.csv (2)
  DH Dyn 2x3x [A]:       corrected_strategy_results.csv (OOS row, Scenario D)
                         + STRATEGY_PERFORMANCE_COMPARISON_2026-05-23.md 9指標完備
  Ens2(Asym+Slope):      ADDITIONAL_ANALYSIS_REPORT_2026-03-30.md
                         (FULL Sharpe 0.846, OOS Sharpe 0.479, Worst10Y +9.84%)
                         OOS CAGR は Sharpe比から推定 (~12.5%)
  NDX 1x B&H:            NASDAQ_extended_to_2026.csv 直接計算

コスト/税モデル（v3 §3-A 準拠）:
  CFD strategies:        CAGR_net = (CAGR_raw − cfd_drag − 0.66%) × 0.8273
  ETF (TQQQ) strategies: CAGR_net = (CAGR_raw − 0pp − 0.66%) × 0.8273 (CFD換算で cost neutral)
  Unleveraged (Ens2/BH): CAGR_net = CAGR_raw × 0.8273 (−0.66% 未含コスト適用なし)
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NDX_CSV = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

IS_START  = pd.Timestamp('1974-01-02')
IS_END    = pd.Timestamp('2021-05-07')
OOS_START = pd.Timestamp('2021-05-08')
OOS_END   = pd.Timestamp('2026-03-26')
TRADING_DAYS = 252

UNINCLUDED_COST = 0.0066    # -0.66%/yr 未含コスト（CFD前提のみ）
JP_TAX_MULT = 0.8273        # §3-A 税モデル ×0.8273

# v3 §3-1 から逆算した経験的 cfd_spread 0.20% → 3.0% の drag (S2 系)
CFD_DRAG_S2 = 0.056         # 5.6pp/yr (E4 5.8, F10 6.2, F10+lmax5 5.2, D5 5.0 平均)
CFD_DRAG_DH = 0.0           # DH Dyn は TQQQ ETF ベース → SBI CFD 換算で cost neutral
CFD_DRAG_ENS = 0.0          # Ens2 max_lev=1.0 (unleveraged) → CFD 利用なし
MAXDD_PENALTY_S2 = 0.02     # S2系 MaxDD への CFD drag 影響 (g13 で E4 -60% → -62% の経験値)

# ---------------------------------------------------------------------------
# §1. NDX 1x B&H exact 計算
# ---------------------------------------------------------------------------
def compute_ndx_bnh_1x():
    df = pd.read_csv(NDX_CSV)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    ret = df['Close'].pct_change().fillna(0.0)
    nav = (1.0 + ret).cumprod()

    nav_is  = nav.loc[IS_START:IS_END];  nav_is = nav_is / nav_is.iloc[0]
    nav_oos = nav.loc[OOS_START:OOS_END]; nav_oos = nav_oos / nav_oos.iloc[0]

    def cagr(n): return (n.iloc[-1]/n.iloc[0])**(1/(len(n)/TRADING_DAYS)) - 1
    def sharpe(n):
        r = n.pct_change().dropna()
        return r.mean()/r.std()*np.sqrt(TRADING_DAYS)
    def maxdd(n): return (n/n.cummax() - 1).min()

    cagr_is = cagr(nav_is); cagr_oos = cagr(nav_oos)
    sharpe_oos = sharpe(nav_oos)
    maxdd_full = maxdd(nav)

    # Worst10Y★ カレンダー年方式
    yearly = nav.groupby(nav.index.year).last()
    worst10y = min(
        (yearly.iloc[i+10]/yearly.iloc[i])**(1/10) - 1
        for i in range(len(yearly)-10)
    )
    # P10_5Y▷ 日次ローリング 252×5
    window = TRADING_DAYS * 5
    rolling = [(nav.iloc[i]/nav.iloc[i-window])**(1/5)-1 for i in range(window, len(nav))]
    p10_5y = np.percentile(rolling, 10)

    return {
        'name': 'NDX 1x Buy & Hold 🅑',
        'CAGR_OOS_raw': cagr_oos,
        'CAGR_OOS_net': cagr_oos * JP_TAX_MULT,  # B&H: ×0.8273 のみ (−0.66% 適用なし)
        'Sharpe_OOS': sharpe_oos,
        'MaxDD_FULL': maxdd_full,
        'Worst10Y_star_raw': worst10y,
        'Worst10Y_star_net': worst10y * JP_TAX_MULT,
        'P10_5Y_raw': p10_5y,
        'P10_5Y_net': p10_5y * JP_TAX_MULT,
        'IS_OOS_gap': cagr_is - cagr_oos,
        'Trades_yr': 0,
        'WFA_WFE': 1.085,           # STRATEGY_PERFORMANCE_COMPARISON_2026-05-23.md より
        'WFA_CI95_lo_raw': 0.074,
        'WFA_CI95_lo_net': 0.074 * JP_TAX_MULT,
    }


# ---------------------------------------------------------------------------
# §2. Legacy strategies — raw 値固定 + コスト/税適用
# ---------------------------------------------------------------------------
LEGACY_RAW = {
    # b1_s2_lt2_results.csv (2) — S2+LT2-N750 k=0.5 modeB
    'S2_LT2_k05': dict(
        name='[Legacy] S2_VZGated + LT2-N750 k=0.5 modeB ‡',
        CAGR_OOS=0.31158, Sharpe_OOS=0.8577, MaxDD_FULL=-0.5945,
        Worst10Y_star=0.18097, P10_5Y=0.09356, IS_OOS_gap=0.00176,
        Trades_yr=27.12, WFA_WFE=1.145, WFA_CI95_lo=0.257,
        cfd_drag=CFD_DRAG_S2, maxdd_penalty=MAXDD_PENALTY_S2, has_residual_cost=True,
    ),
    # b1_s2_lt2_results.csv (1) — S2_VZGated 単独
    'S2_alone': dict(
        name='[Legacy] S2_VZGated 単独 ‡ ⚠↑↑',
        CAGR_OOS=0.27505, Sharpe_OOS=0.7704, MaxDD_FULL=-0.6337,
        Worst10Y_star=0.17739, P10_5Y=0.07316, IS_OOS_gap=0.06043,
        Trades_yr=27.12, WFA_WFE=0.830, WFA_CI95_lo=0.281,
        cfd_drag=CFD_DRAG_S2, maxdd_penalty=MAXDD_PENALTY_S2, has_residual_cost=True,
    ),
    # corrected_strategy_results.csv (OOS Scenario D) + STRATEGY_PERFORMANCE_COMPARISON_2026-05-23
    'DH_Dyn': dict(
        name='[Legacy] DH Dyn 2x3x [A] ‡ ⚠↑↑',
        CAGR_OOS=0.1488, Sharpe_OOS=0.646, MaxDD_FULL=-0.4508,
        Worst10Y_star=0.143, P10_5Y=0.096, IS_OOS_gap=0.0848,
        Trades_yr=27.12, WFA_WFE=0.687, WFA_CI95_lo=0.182,
        cfd_drag=CFD_DRAG_DH,    # TQQQ ETF → SBI CFD は cost neutral
        maxdd_penalty=0.0, has_residual_cost=True,
    ),
    # g15b_ens2_oos_scenarioD.py で実測（max_lev=1.0、TQQQ Scenario D コスト）
    # 旧推定値 (12.57%/0.479) は ADDITIONAL_ANALYSIS_REPORT_2026-03-30 の Sharpe 比からの推定。
    # backtest_engine.run_backtest() が 0.9% フラットコストで SOFR スケーリング欠落していたため、
    # 過小コスト→過大 raw 値が源泉。正しい TQQQ 2×SOFR + TER + swap で再計算すると衝撃の値に。
    'Ens2_AsymSlope': dict(
        name='[Legacy] Ens2(Asym+Slope) max_lev=1.0 ‡ ⚠↑↑⚠↑↑',
        CAGR_OOS=0.0092, Sharpe_OOS=0.154, MaxDD_FULL=-0.5388,
        Worst10Y_star=-0.0084, P10_5Y=0.0000, IS_OOS_gap=0.1331,
        Trades_yr=0.52, WFA_WFE=np.nan, WFA_CI95_lo=np.nan,
        cfd_drag=CFD_DRAG_ENS,   # max_lev=1 = TQQQ (3x) × 信号 [0,1] = avg eff_L 1.12x
        maxdd_penalty=0.0, has_residual_cost=False,  # TQQQ ベース、SBI CFD 換算は別問題
    ),
}


def apply_cost_tax(raw):
    """raw (Scenario D 値) → SBI CFD cost-after → §3-A 税後。"""
    cfd_drag = raw['cfd_drag']
    has_residual = raw['has_residual_cost']

    cagr_after_cost = raw['CAGR_OOS'] - cfd_drag
    if has_residual:
        cagr_net = (cagr_after_cost - UNINCLUDED_COST) * JP_TAX_MULT
    else:
        cagr_net = cagr_after_cost * JP_TAX_MULT

    w10y_after_cost = raw['Worst10Y_star'] - cfd_drag
    w10y_net = (w10y_after_cost - UNINCLUDED_COST) * JP_TAX_MULT if has_residual else w10y_after_cost * JP_TAX_MULT

    if pd.isna(raw['P10_5Y']):
        p10_net = np.nan
    else:
        p10_after_cost = raw['P10_5Y'] - cfd_drag
        p10_net = (p10_after_cost - UNINCLUDED_COST) * JP_TAX_MULT if has_residual else p10_after_cost * JP_TAX_MULT

    maxdd_after_cost = raw['MaxDD_FULL'] - raw['maxdd_penalty']  # negative * subtract positive = more negative

    if pd.isna(raw['WFA_CI95_lo']):
        ci95_net = np.nan
    else:
        ci95_after_cost = raw['WFA_CI95_lo'] - cfd_drag
        ci95_net = (ci95_after_cost - UNINCLUDED_COST) * JP_TAX_MULT if has_residual else ci95_after_cost * JP_TAX_MULT

    return {
        'name': raw['name'],
        'CAGR_OOS_raw': raw['CAGR_OOS'],
        'CAGR_OOS_net': cagr_net,
        'Sharpe_OOS': raw['Sharpe_OOS'],  # 税前据置 (v3 §0)
        'MaxDD_FULL': maxdd_after_cost,
        'Worst10Y_star_raw': raw['Worst10Y_star'],
        'Worst10Y_star_net': w10y_net,
        'P10_5Y_raw': raw['P10_5Y'],
        'P10_5Y_net': p10_net,
        'IS_OOS_gap': raw['IS_OOS_gap'],
        'Trades_yr': raw['Trades_yr'],
        'WFA_WFE': raw['WFA_WFE'],
        'WFA_CI95_lo_raw': raw['WFA_CI95_lo'],
        'WFA_CI95_lo_net': ci95_net,
    }


# ---------------------------------------------------------------------------
# §3. メイン
# ---------------------------------------------------------------------------
def main():
    results = [apply_cost_tax(LEGACY_RAW[k]) for k in ['S2_LT2_k05', 'S2_alone', 'DH_Dyn', 'Ens2_AsymSlope']]
    results.append(compute_ndx_bnh_1x())

    print('=' * 90)
    print('v2 結果サマリ — raw → net (税後手取り)')
    print('=' * 90)
    print(f'{"Strategy":<55s} {"raw":>9s} {"net⓽":>9s} {"Sharpe":>8s} {"MaxDD":>8s}')
    for r in results:
        raw = r['CAGR_OOS_raw'] * 100
        net = r['CAGR_OOS_net'] * 100
        sh  = r['Sharpe_OOS']
        dd  = r['MaxDD_FULL'] * 100
        print(f'{r["name"]:<55s} {raw:>+8.2f}% {net:>+8.2f}% {sh:>+7.3f} {dd:>+7.2f}%')

    # CSV 保存
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(BASE, 'g15_legacy_results.csv'), index=False)
    print(f'\nCSV saved: g15_legacy_results.csv')

    # MD 行生成
    print()
    print('=' * 90)
    print('MD ROWS for v3 §2 table:')
    print('=' * 90)
    for r in results:
        cagr_oos = f'{r["CAGR_OOS_net"]*100:+5.1f}%'
        sh = r['Sharpe_OOS']
        if sh > 0.840:    sh_marker = ' H'
        elif sh > 0.780:  sh_marker = ' M'
        else:             sh_marker = '  '
        sharpe   = f'{sh:+5.2f}{sh_marker}'
        maxdd    = f'{r["MaxDD_FULL"]*100:+5.1f}%'
        if pd.isna(r.get('Worst10Y_star_net')):
            w10y = '  —  '
        else:
            w10y = f'{r["Worst10Y_star_net"]*100:+5.1f}%'
        if pd.isna(r.get('P10_5Y_net')):
            p10 = '  —  '
        else:
            p10 = f'{r["P10_5Y_net"]*100:+5.1f}%'
        gap = f'{r["IS_OOS_gap"]*100:+5.2f}pp' if not pd.isna(r['IS_OOS_gap']) else '   —    '
        tr  = f'{int(round(r["Trades_yr"])):>3d}' if not pd.isna(r['Trades_yr']) else ' — '
        wfe = f'✅ LOW<br>({r["WFA_WFE"]:.1f})' if not pd.isna(r['WFA_WFE']) else '—'
        ci95 = f'{r["WFA_CI95_lo_net"]:+.2f}' if not pd.isna(r.get('WFA_CI95_lo_net', np.nan)) else '   —   '
        print(f'| {r["name"]} | {cagr_oos} | {sharpe} | {maxdd} | {w10y} | {p10} | {gap} | {tr} | {wfe} | {ci95} |')


if __name__ == '__main__':
    main()
