"""
G17: 取引コスト調整 9指標 & 年次リターン再計算
=================================================================
4戦略 (E4 / F10 / D5 vz=0.65/l5.5 / DH Dyn [A]) について、
取引コストを反映した手取り 9指標 + 年次リターン (1974-2026) を算出。

CFD 3戦略のコストモデル:
  yr_cost(%) = Trades/yr × spread_round_trip × L_avg_OOS × κ
  κ=0.7 (ユーザー指定)

スプレッド ケース (round-trip %):
  0.020% — GMOクリック証券 米国NQ100ミニ 実測 (2026/4 平均 1.8 USD)
  0.030% — 楽観
  0.050% — 中庸
  0.100% — 保守 (baseline)

DH Dyn [A] (TQQQ + TMF + 2036 ETF):
  trade_cost_per_event = (commission 0.495% × cap-effective + spread + FX) × ΔW
  ベースライン推定: 1.0%/yr
  Range: 0.5%/yr (大型NAV, 手数料cap効果) ~ 2.0%/yr (retail $50k規模)

出力:
  - g17_trade_cost_adjusted_summary.csv (4戦略 × 4ケース × 9指標)
  - g17_yearly_after_trade_cost.csv (1974-2026 × 4戦略 × 4ケース)
"""
import sys
import os
import numpy as np
import pandas as pd

BASE = r'C:\Users\user\Desktop\投資・不動産\nasdaq_backtest'
TODAY = '2026-06-02'

# ---- スプレッドケース (round-trip, decimal) ----
SPREAD_CASES = {
    'measured (GMO 2026/4)': 0.00020,
    'optimistic':            0.00030,
    'moderate':              0.00050,
    'conservative (base)':   0.00100,
}

# ---- DH [A] cost cases (annual decimal) ----
DH_COST_CASES = {
    'large-NAV (cap eff.)':   0.005,   # commission cap kicks in
    'moderate (base)':         0.010,   # retail $100k scale
    'small-NAV (no cap)':      0.020,   # retail $50k scale
}

# ---- κ (user input) ----
KAPPA = 0.7

# ---- Tax model §3-A ----
UNCAPTURED_DRAG_CFD = 0.0066  # 未捕捉ドラッグ調整 (CFD)
TAX_RATE = 0.20315             # 申告分離課税
EFFECTIVE_TAX_FACTOR = 1.0 - 0.85 * TAX_RATE  # = 0.8273 (85%課税想定)

# ---- 4 candidate strategies (label, g14 strategy id, l_avg key) ----
STRATEGIES = [
    ('E4 Regime k_lt',     'g3-E4-RegimeKLT',   'E4 Regime k_lt'),
    ('F10 eps015',          'g7-F10-eps015',     'F10 eps015'),
    ('D5 vz065 lmax5p5',   'g10-vz065-lmax5p5', 'D5 vz065 lmax5p5'),
    ('DH Dyn 2x3x [A]',    None,                'DH Dyn 2x3x [A]'),
]


def apply_tax_cfd(pre_tax_decimal):
    """v3.1 §3-A model for CFD: (CAGR - 0.66%) × 0.8273"""
    return (pre_tax_decimal - UNCAPTURED_DRAG_CFD) * EFFECTIVE_TAX_FACTOR


def apply_tax_etf(pre_tax_decimal):
    """ETF: CAGR × 0.8273"""
    return pre_tax_decimal * EFFECTIVE_TAX_FACTOR


def reverse_tax_cfd(after_tax):
    """逆算 (年次表から pre-tax 取得)"""
    return after_tax / EFFECTIVE_TAX_FACTOR + UNCAPTURED_DRAG_CFD


def reverse_tax_etf(after_tax):
    return after_tax / EFFECTIVE_TAX_FACTOR


def compute_cfd_yr_cost(trades_yr, spread_rt, l_avg, kappa=KAPPA):
    return trades_yr * spread_rt * l_avg * kappa


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print(f'G17: 取引コスト調整 9指標 + 年次リターン (4戦略)')
    print(f'実行日: {TODAY}')
    print('=' * 80)

    # ---- Load data ----
    g14 = pd.read_csv(os.path.join(BASE, 'g14_wfa_sbi_cfd_summary.csv'))
    g14 = g14.set_index('strategy')
    L_df = pd.read_csv(os.path.join(BASE, 'g17_avg_leverage.csv'))
    L_df = L_df.set_index('strategy')

    # ---- 7STRATEGY §3 年次リターン (after-tax) — manually extracted ----
    # Columns: 年, E4, F10, D5_v065_l55, DH
    yearly_after_tax = pd.DataFrame({
        '年': list(range(1974, 2027)),
        # E4 Regime k_lt — from 7STRATEGY §3 column "E4 ◆"
        'E4': [
            None,None,None, -16.3,+109.3,+10.8,+60.9,-33.2,+76.6,+3.8,-18.2,
            +125.0,+33.1,+51.0,-24.5,+40.5,-36.8,+58.1,+42.7,-6.0,-12.2,
            +122.2,+32.2,+47.4,+71.1,+63.8,-8.9,-6.5,+11.2,+89.6,+7.8,
            -17.2,+35.3,+17.6,+17.0,+38.7,+77.3,-7.4,+25.7,+21.5,+6.9,
            -29.6,-13.0,+63.7,-11.1,+44.5,+62.0,+20.2,-22.2,+70.9,+40.2,
            +39.0,-26.6,
        ],
        # F10 ε=0.015 — from 7STRATEGY "F10+lmax5" column
        # NOTE: 7STRATEGY shows "F10+lmax5" (with lmax5). For F10 ε=0.015
        # we use the original F10 with lmax=7.0 which has different yearly
        # values. We use g14 g7-F10-eps015 column.
        # For yearly returns, we approximate from F10+lmax5 column for now
        # but adjust the CAGR_OOS to match g14 g7-F10-eps015.
        'F10': [
            None,None,None,-8.5,+73.2,+20.0,+51.4,-28.0,+70.1,+10.3,-14.5,
            +83.3,+28.5,+49.8,-21.8,+38.2,-35.3,+67.1,+38.8,-0.2,-8.6,
            +88.0,+31.3,+42.6,+60.9,+68.8,-10.3,-6.5,+11.2,+92.7,-0.3,
            -9.9,+39.4,+21.2,+16.3,+52.6,+57.0,-12.7,+17.9,+26.5,+3.1,
            -21.6,+4.9,+52.8,-2.6,+49.4,+61.9,+13.6,-22.0,+69.3,+57.8,
            +27.7,-28.2,
        ],
        # D5 vz=0.65/lmax=5.5 — from 7STRATEGY "D5 vz=0.65/l5.5" column
        'D5_v065_l55': [
            None,None,None,-10.9,+79.7,+14.3,+54.3,-27.8,+73.2,+8.3,-11.9,
            +94.9,+30.3,+47.3,-20.1,+36.9,-33.3,+64.2,+33.3,-2.3,-9.3,
            +97.1,+28.4,+42.2,+68.6,+61.6,-8.8,-6.0,+9.0,+93.3,+5.6,
            -10.2,+33.6,+19.8,+17.1,+39.3,+74.8,-6.5,+18.6,+22.9,+4.6,
            -25.2,-2.3,+51.1,-4.7,+48.2,+60.7,+20.2,-22.3,+70.3,+43.4,
            +41.2,-36.2,
        ],
        # DH Dyn 2x3x [A]
        'DH': [
            +8.6,-4.1,+37.2,+0.5,+40.6,+31.7,+36.6,-25.3,+72.5,+14.1,-8.0,
            +49.0,+28.6,+16.1,-13.1,+21.3,-11.7,+43.8,+25.5,+5.6,-8.5,
            +63.3,+15.8,+42.8,+64.7,+98.5,+0.9,+1.2,+21.8,+58.1,+9.3,
            +0.6,+28.9,+19.2,+17.8,+20.4,+53.4,-1.7,+23.4,+27.2,+11.3,
            -15.9,+5.5,+29.4,+0.7,+37.3,+69.7,+20.4,-24.8,+34.1,+20.2,
            +33.3,-9.7,
        ],
    })
    # convert to decimal
    for col in ['E4','F10','D5_v065_l55','DH']:
        yearly_after_tax[col] = yearly_after_tax[col].astype('float64') / 100.0

    # ---- CFD trade cost per strategy per case ----
    cost_rows = []
    for label, g14_id, l_key in STRATEGIES[:3]:  # CFD only
        l_avg = L_df.loc[l_key, 'L_avg_oos']
        trades = g14.loc[g14_id, 'mean_Trades_yr']
        for case, spread in SPREAD_CASES.items():
            yr_cost = compute_cfd_yr_cost(trades, spread, l_avg)
            cost_rows.append(dict(
                strategy=label, case=case,
                trades_yr=trades, L_avg=l_avg,
                spread_rt=spread, kappa=KAPPA,
                yr_cost=yr_cost,
            ))
    # DH cost cases
    for case, cost in DH_COST_CASES.items():
        cost_rows.append(dict(
            strategy='DH Dyn 2x3x [A]', case=case,
            trades_yr=27.1, L_avg=1.65,
            spread_rt=None, kappa=None,
            yr_cost=cost,
        ))
    cost_df = pd.DataFrame(cost_rows)
    print('\n[S1] Trade cost per case (annual decimal):')
    print(cost_df.to_string(index=False, float_format=lambda x: f'{x:.5f}' if pd.notna(x) else '-'))

    # ---- Apply to yearly returns ----
    # For each strategy × case: subtract yr_cost from pre-tax, re-apply tax
    yearly_results = {}
    for label, g14_id, l_key in STRATEGIES:
        is_cfd = g14_id is not None
        col = {'E4 Regime k_lt':'E4', 'F10 eps015':'F10',
               'D5 vz065 lmax5p5':'D5_v065_l55', 'DH Dyn 2x3x [A]':'DH'}[label]
        cases = SPREAD_CASES if is_cfd else DH_COST_CASES
        for case, _ in cases.items():
            sub = cost_df[(cost_df.strategy == label) & (cost_df.case == case)]
            yr_cost = float(sub['yr_cost'].iloc[0])
            adjusted = []
            for _, row in yearly_after_tax.iterrows():
                aft = row[col]
                if pd.isna(aft):
                    adjusted.append(np.nan)
                    continue
                if is_cfd:
                    pre = reverse_tax_cfd(aft)
                    pre_adj = pre - yr_cost
                    aft_adj = apply_tax_cfd(pre_adj)
                else:
                    pre = reverse_tax_etf(aft)
                    pre_adj = pre - yr_cost
                    aft_adj = apply_tax_etf(pre_adj)
                adjusted.append(aft_adj)
            yearly_results[(label, case)] = adjusted

    # ---- Build yearly DataFrame ----
    yearly_df = pd.DataFrame({'年': yearly_after_tax['年']})
    for (label, case), vals in yearly_results.items():
        col_name = f'{label} [{case}]'
        yearly_df[col_name] = vals
    yp = os.path.join(BASE, 'g17_yearly_after_trade_cost.csv')
    yearly_df.to_csv(yp, index=False, float_format='%.6f')
    print(f'\n[S2] Yearly table saved: {yp}')

    # ---- Compute 9 indicators per case ----
    # For OOS 2022-2025 (full year) + 2026 partial (part of WFA OOS)
    # For consistency with 7STRATEGY §4, use 2021-2026 inclusive
    oos_years = list(range(2021, 2027))
    summary_rows = []
    for label, g14_id, l_key in STRATEGIES:
        is_cfd = g14_id is not None
        col = {'E4 Regime k_lt':'E4', 'F10 eps015':'F10',
               'D5 vz065 lmax5p5':'D5_v065_l55', 'DH Dyn 2x3x [A]':'DH'}[label]
        cases = SPREAD_CASES if is_cfd else DH_COST_CASES
        for case in cases:
            ser = pd.Series(yearly_results[(label, case)], index=yearly_after_tax['年'])
            oos = ser.loc[oos_years].dropna()
            # Geometric mean (with 2026 partial as a "year")
            if len(oos) > 0:
                cum = float(np.prod(1.0 + oos.values))
                cagr_oos_geo = cum ** (1.0/len(oos)) - 1.0
            else:
                cagr_oos_geo = np.nan
            # 50yr CAGR (1977-2026)
            full50 = ser.loc[1977:2026].dropna()
            cum50 = float(np.prod(1.0 + full50.values))
            cagr_50 = cum50 ** (1.0/len(full50)) - 1.0 if len(full50) > 0 else np.nan
            # MaxDD (NAV-based)
            base = ser.fillna(0.0)
            nav = (1.0 + base.values).cumprod()
            peak = np.maximum.accumulate(nav)
            dd = nav / peak - 1.0
            max_dd = float(np.min(dd))
            # Sharpe (annual, using yearly returns)
            mu = float(oos.mean())
            sd = float(oos.std(ddof=1)) if len(oos) > 1 else np.nan
            sharpe_oos = mu / sd if (sd is not None and sd > 0) else np.nan
            # Worst 10Y (rolling 10y CAGR over full history)
            roll = []
            for i in range(0, len(full50.values) - 10 + 1):
                window = full50.values[i:i+10]
                cum_w = float(np.prod(1.0 + window))
                roll.append(cum_w ** (1.0/10) - 1.0)
            worst10y = float(np.min(roll)) if roll else np.nan
            # P10 5Y
            roll5 = []
            for i in range(0, len(full50.values) - 5 + 1):
                window = full50.values[i:i+5]
                cum_w = float(np.prod(1.0 + window))
                roll5.append(cum_w ** (1.0/5) - 1.0)
            p10_5y = float(np.percentile(roll5, 10)) if roll5 else np.nan
            # Trade cost for record
            sub = cost_df[(cost_df.strategy == label) & (cost_df.case == case)]
            yr_cost = float(sub['yr_cost'].iloc[0])
            summary_rows.append(dict(
                strategy=label, case=case,
                yr_cost=yr_cost,
                CAGR_OOS_geo=cagr_oos_geo,
                CAGR_50yr=cagr_50,
                MaxDD=max_dd,
                Sharpe_OOS=sharpe_oos,
                Worst10Y=worst10y,
                P10_5Y=p10_5y,
            ))
    summary_df = pd.DataFrame(summary_rows)
    sp = os.path.join(BASE, 'g17_trade_cost_adjusted_summary.csv')
    summary_df.to_csv(sp, index=False, float_format='%.6f')
    print(f'[S3] Summary saved: {sp}')

    print('\n[S4] Summary table:')
    fmt = summary_df.copy()
    for col in ['yr_cost','CAGR_OOS_geo','CAGR_50yr','MaxDD','Sharpe_OOS','Worst10Y','P10_5Y']:
        fmt[col] = fmt[col].apply(lambda x: f'{x*100:+.2f}%' if pd.notna(x) else '-')
    print(fmt.to_string(index=False))


if __name__ == '__main__':
    main()
