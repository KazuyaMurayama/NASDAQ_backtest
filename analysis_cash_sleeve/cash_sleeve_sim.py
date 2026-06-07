"""DH-W1 キャッシュ期間を 1倍投信(NASDAQ/Gold/Bond)で置換するシミュレーション。
Part1: OUT(cash)日における各1倍資産の年率リターン/リスク (Q1回答)。
Part2: 6配分パターン × 5営業日ラグ + 信託報酬 + 約20%税 を EVALUATION_STANDARD
       §3.12 標準9指標 (CAGR_OOS / 累積CAGR OOS·IS / IS-OOS gap / Sharpe_OOS /
       MaxDD / Worst10Y★ / P10_5Y / Trades_yr / WFE / CI95_lo) で評価。
原資産は全てコード内に存在 (外部取得なし)。USD建て。
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import load_shared_assets, generate_windows
from g23a_dh_refinement_variants import build_W1
from corrected_strategy_backtest import build_bond_1x_nav_corrected
from compute_cfd_worst10y import prepare_gold_local
from g18_daily_trade_cost_wfa import (
    metrics_from_nav, apply_tax_etf_decimal, wfa_metrics,
)

LAG_DAYS = 5
TRADING_DAYS = 252
OUT_DIR = os.path.join(ROOT, 'analysis_cash_sleeve')

# 信託報酬 基本ケース (実SBI商品, 年率)
FEE_NDX  = 0.001958   # SBI NASDAQ100インデックス・ファンド
FEE_GOLD = 0.001838   # SBI・iシェアーズ・ゴールド(ヘッジなし)
FEE_BOND = 0.00154    # iシェアーズ 米国債20年超(2255, ヘッジなし)

# 6配分パターン: (NDX, Gold, Bond) ウェイト
PATTERNS = {
    'P1_NDX100':        (1.0, 0.0, 0.0),
    'P2_GOLD100':       (0.0, 1.0, 0.0),
    'P3_BOND100':       (0.0, 0.0, 1.0),
    'P4_NDX50_GOLD50':  (0.5, 0.5, 0.0),
    'P5_GOLD50_BOND50': (0.0, 0.5, 0.5),
    'P6_THIRDS':        (1/3, 1/3, 1/3),
    'P7_GOLD75_BOND25': (0.0, 0.75, 0.25),
}

# レポート対象の4戦略 (年次表もこの4本)
FOCUS = ['BASELINE_DH-W1', 'P2_GOLD100', 'P5_GOLD50_BOND50', 'P7_GOLD75_BOND25']


def ann_ret_vol_on_mask(ret, mask_bool):
    r = np.asarray(ret, dtype=float)[mask_bool]
    if len(r) < 5:
        return np.nan, np.nan, np.nan, 0
    n_days = len(r)
    cum = float(np.prod(1.0 + r))
    yrs = n_days / TRADING_DAYS
    cagr = cum ** (1.0 / yrs) - 1.0 if cum > 0 else -1.0
    vol = float(np.std(r, ddof=1)) * np.sqrt(TRADING_DAYS)
    return cagr, vol, cum, n_days


def _geo(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan
    c = float(np.prod(1.0 + x))
    return c ** (1.0 / len(x)) - 1.0 if c > 0 else -1.0


def worst10y_p10_from_yearly(yr_values):
    arr = np.asarray(yr_values, dtype=float)
    w10 = []
    for i in range(len(arr) - 10 + 1):
        c = float(np.prod(1.0 + arr[i:i+10]))
        w10.append(c ** (1.0/10) - 1.0 if c > 0 else -1.0)
    worst10y = float(np.min(w10)) if w10 else np.nan
    r5 = []
    for i in range(len(arr) - 5 + 1):
        c = float(np.prod(1.0 + arr[i:i+5]))
        r5.append(c ** (1.0/5) - 1.0 if c > 0 else -1.0)
    p10 = float(np.percentile(r5, 10)) if r5 else np.nan
    return worst10y, p10


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('DH-W1 キャッシュ・スリーブ 1倍投信置換シミュレーション (標準9指標)')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates'].reset_index(drop=True)
    n = len(dates)

    # --- DH-W1 baseline (OUT=cash 0%) ---
    nav_w1, cost_w1, mask, wn_w1, lev_w1 = build_W1(a)
    mask = np.asarray(mask, dtype=float)              # 1=HOLD, 0=OUT
    r_w1 = pd.Series(np.asarray(nav_w1, dtype=float)).pct_change().fillna(0).values
    out = (mask < 0.5)
    print(f'  HOLD={mask.mean()*100:.1f}%  OUT(cash)={100-mask.mean()*100:.1f}%  n={n}')

    # --- 1x underlyings (USD), positionally aligned to dates ---
    close = np.asarray(a['close'], dtype=float)
    ret_ndx = np.concatenate([[0.0], np.diff(close) / close[:-1]])
    gold_1x = np.asarray(prepare_gold_local(a['dates']), dtype=float)
    ret_gold = np.concatenate([[0.0], np.diff(gold_1x) / gold_1x[:-1]])
    bond_1x = np.asarray(build_bond_1x_nav_corrected(
        a['dates'], use_time_varying_duration=True, bond_maturity=22.0), dtype=float)
    ret_bond = np.concatenate([[0.0], np.diff(bond_1x) / bond_1x[:-1]])
    ret_ndx  = np.nan_to_num(ret_ndx,  nan=0.0)
    ret_gold = np.nan_to_num(ret_gold, nan=0.0)
    ret_bond = np.nan_to_num(ret_bond, nan=0.0)

    # ===== PART 1: OUT日の各資産 年率リターン/リスク =====
    print('\n[Part1] OUT(cash)日のみで各1倍資産の年率リターン/リスク')
    p1_rows = []
    for name, ret in [('NASDAQ_1x', ret_ndx), ('Gold_1x', ret_gold), ('Bond_1x', ret_bond)]:
        cagr, vol, cum, nd = ann_ret_vol_on_mask(ret, out)
        rpr = cagr / vol if vol and vol > 0 else np.nan
        p1_rows.append(dict(asset=name, out_days=nd,
                            ann_return_pct=round(cagr*100, 2),
                            ann_vol_pct=round(vol*100, 2),
                            ret_per_risk=round(rpr, 3) if not np.isnan(rpr) else np.nan,
                            cum_mult=round(cum, 3)))
        print(f'  {name:10s} 年率Ret={cagr*100:+6.2f}%  年率Vol={vol*100:5.2f}%  '
              f'累積×{cum:.3f}  (OUT {nd}日)')
    pd.DataFrame(p1_rows).to_csv(
        os.path.join(OUT_DIR, 'cash_sleeve_part1_asset_during_cash.csv'),
        index=False, float_format='%.4f')

    # ===== PART 2: 6パターン × ラグ + 手数料 + 税 + 標準9指標 =====
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out[:-LAG_DAYS]      # 5営業日前方シフト (両端ラグ)

    windows = generate_windows(dates)
    print(f'\n[Part2] WFA windows={len(windows)}  (5d lag + 信託報酬 + 税後 ×0.8273)')

    # 効果的ウェイト (WFA trade計数用): HOLD=DH動的, fund=パターン固定, cash=0
    wn_base = np.asarray(a['wn_A'], dtype=float) * mask
    wb_base = np.asarray(a['wb_A'], dtype=float) * mask
    lev_base = np.asarray(a['lev_raw'], dtype=float) * mask * 3.0

    def evaluate(label, r_daily, wn_eff, wb_eff, lev_eff):
        nav_pre = pd.Series(np.cumprod(1.0 + r_daily), index=dates.index)
        m = metrics_from_nav(nav_pre, dates, a['ret'])
        yr_aft = m['yearly'].apply(apply_tax_etf_decimal)
        is_sub  = yr_aft.loc[[y for y in yr_aft.index if 1974 <= y <= 2020]]
        oos_sub = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
        cagr_is, cagr_oos = _geo(is_sub.values), _geo(oos_sub.values)
        oos_mult = float(np.prod(1.0 + oos_sub.values))
        worst10y, p10 = worst10y_p10_from_yearly(yr_aft.values)
        try:
            wfa = wfa_metrics(nav_pre, dates, windows,
                              lev_arr=lev_eff, wn_arr=wn_eff, wb_arr=wb_eff)
            ci95 = wfa.get('WFA_CI95_lo', np.nan)
            wfe = wfa.get('WFA_WFE', np.nan)
            trades = wfa.get('mean_Trades_yr', np.nan)
        except Exception as e:
            print(f'    WFA failed {label}: {e}')
            ci95 = wfe = trades = np.nan
        row = dict(
            strategy=label,
            CAGR_OOS_pct=round(cagr_oos*100, 2),
            CAGR_IS_pct=round(cagr_is*100, 2),
            OOS_mult=round(oos_mult, 3),
            IS_OOS_gap_pp=round((cagr_is - cagr_oos)*100, 2),
            Sharpe_OOS=round(m['Sharpe_OOS'], 3),
            MaxDD_pct=round(m['MaxDD_FULL']*100, 2),
            Worst10Y_pct=round(worst10y*100, 2),
            P10_5Y_pct=round(p10*100, 2),
            Trades_yr=round(trades, 1) if not np.isnan(trades) else np.nan,
            WFE=round(wfe, 3) if not np.isnan(wfe) else np.nan,
            CI95_lo_pct=round(ci95*100, 2) if not np.isnan(ci95) else np.nan,
        )
        print(f"  {label:22s} OOS={row['CAGR_OOS_pct']:+6.2f}% IS={row['CAGR_IS_pct']:+6.2f}% "
              f"gap={row['IS_OOS_gap_pp']:+5.2f} Sh={row['Sharpe_OOS']:+.2f} "
              f"DD={row['MaxDD_pct']:+6.1f} W10={row['Worst10Y_pct']:+5.2f} "
              f"P10={row['P10_5Y_pct']:+5.2f} Tr={row['Trades_yr']} "
              f"WFE={row['WFE']} CI={row['CI95_lo_pct']}")
        return row, oos_mult, yr_aft

    rows = []
    yearly_focus = {}   # label -> after-tax yearly Series
    # Baseline DH-W1 (cash 0%)
    base_row, base_mult, base_yr = evaluate('BASELINE_DH-W1', r_w1, wn_base, wb_base, lev_base)
    rows.append(base_row)
    yearly_focus['BASELINE_DH-W1'] = base_yr

    for name, (wnp, wgp, wbp) in PATTERNS.items():
        r_blend = wnp*ret_ndx + wgp*ret_gold + wbp*ret_bond
        fee_daily = (wnp*FEE_NDX + wgp*FEE_GOLD + wbp*FEE_BOND) / TRADING_DAYS
        r_enh = np.where(fund_active, r_blend - fee_daily, r_w1)
        wn_eff = np.where(fund_active, wnp, wn_base)
        wb_eff = np.where(fund_active, wbp, wb_base)
        lev_eff = np.where(fund_active, 1.0, lev_base)
        row, oos_mult, yr_aft = evaluate(name, r_enh, wn_eff, wb_eff, lev_eff)
        row['vs_base_OOSmult_pct'] = round((oos_mult/base_mult - 1)*100, 1)
        rows.append(row)
        if name in FOCUS:
            yearly_focus[name] = yr_aft

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'cash_sleeve_7patterns_metrics.csv'),
              index=False, float_format='%.4f')

    # 4戦略のみの指標表
    df_focus = df[df['strategy'].isin(FOCUS)].copy()
    df_focus.to_csv(os.path.join(OUT_DIR, 'cash_sleeve_4strategies_metrics.csv'),
                    index=False, float_format='%.4f')

    # 年次リターン表 (4戦略, 税後 %)
    years = sorted(set(y for s in yearly_focus.values() for y in s.index))
    ycols = {'year': years}
    for label in FOCUS:
        ser = yearly_focus[label]
        ycols[label] = [round(float(ser.get(y, np.nan))*100, 2) if y in ser.index else np.nan
                        for y in years]
    pd.DataFrame(ycols).to_csv(
        os.path.join(OUT_DIR, 'cash_sleeve_4strategies_yearly.csv'),
        index=False, float_format='%.2f')

    print('\n[DONE] CSVs -> analysis_cash_sleeve/ (7patterns, 4strategies, yearly)')


if __name__ == '__main__':
    main()
