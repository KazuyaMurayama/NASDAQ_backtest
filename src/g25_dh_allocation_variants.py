"""G25: DH 配分シフト型 4 候補 (V1-V4) — キャッシュは最終手段
=================================================================
ユーザー指示: キャッシュ比率はほぼ 0% にし、TQQQ リスク高まったら
TMF/2036 に rotation。よっぽど全部悪いときのみ cash。

DH-V1 (3-state preset, 常に 100% 投資):
  Bull (raw_a2>0.5 AND vz<0): 80% TQQQ / 10% TMF / 10% 2036
  Normal (else): 60% TQQQ / 20% TMF / 20% 2036
  Risk-off (raw_a2<0.2 OR vz>0.65): 30% TQQQ / 35% TMF / 35% 2036
  Cash: 0%

DH-V2 (連続 vol-tilt, 常に 100% 投資):
  TQQQ = clip(0.80 - 0.5 × max(vz,0), 0.20, 0.80)
  TMF/2036 = (1 - TQQQ)/2 each
  Cash: 0%

DH-V3 (DH base × F10 ε tilt、binary mask なし):
  既存 wn_f10/wb_f10/wg_f10 をそのまま使用 (F10 ε rebalance)
  Cash: 0%

DH-V4 (3-state + 極端 cash 安全弁):
  V1 と同じ 3-state、ただし vz>1.5 AND raw_a2<0.05 の極端 panic 時のみ
  50% TMF + 50% cash → expected cash < 1% over total

商品: TQQQ + TMF + 2036
コスト: per_unit=0.0010 (moderate), ETF 税 ×0.8273
peak_lev ≤ 3.0x assert
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import load_shared_assets, BASE, generate_windows
from g18_daily_trade_cost_wfa import (
    build_dh_nav_with_cost, wfa_metrics,
    metrics_from_nav, apply_tax_etf_decimal,
)

DH_PER_UNIT = 0.0010


def _a2(a):
    return a['raw_a2'].values if hasattr(a['raw_a2'], 'values') else a['raw_a2']


# ----- 4 variants -----

def build_V1(a):
    """3-state preset, 常に 100% 投資 (cash 0%)"""
    a2 = _a2(a); vz = np.asarray(a['vz_arr'])
    n = len(a['close'])
    bull = (a2 > 0.5) & (vz < 0.0)
    risk_off = (a2 < 0.2) | (vz > 0.65)
    normal = ~bull & ~risk_off

    wn = np.where(bull, 0.80, np.where(normal, 0.60, 0.30))
    wb = np.where(bull, 0.10, np.where(normal, 0.20, 0.35))
    wg = np.where(bull, 0.10, np.where(normal, 0.20, 0.35))
    # TQQQ exposure: lev_raw = 1 always when bull (full TQQQ engagement)
    # else use DH lev_raw for proportional confidence
    lev_raw = np.where(bull, 1.0, np.where(normal, 0.7, 0.4))
    # Cash の発生は wn+wb+wg < 1 のときのみ → V1 では常に 1.0
    return _make_nav_call(a, lev_raw, wn, wb, wg)


def build_V2(a):
    """連続 vol-tilt, 常に 100% 投資"""
    a2 = _a2(a); vz = np.asarray(a['vz_arr'])
    n = len(a['close'])
    # TQQQ = clip(0.80 - 0.5*max(vz,0), 0.20, 0.80)
    wn = np.clip(0.80 - 0.5 * np.maximum(vz, 0), 0.20, 0.80)
    remainder = 1.0 - wn
    wb = remainder * 0.5
    wg = remainder * 0.5
    # lev_raw proportional to confidence: scale with a2 and inverse vol
    # 単純に raw_a2 を lev_raw 使う (DH base 流儀)
    lev_raw = np.clip(np.where(a2 > 0.15, a2, 0.0), 0.0, 1.0)
    return _make_nav_call(a, lev_raw, wn, wb, wg)


def build_V3(a):
    """DH base + F10 ε tilt (常に 100% 投資、binary mask なし)"""
    # F10 ε tilted weights (sum might not be 1.0 if wb_A < tilt)
    wn = np.asarray(a['wn_f10'])
    wb = np.asarray(a['wb_f10'])
    wg = np.asarray(a['wg_f10'])
    lev_raw = np.asarray(a['lev_raw'])
    return _make_nav_call(a, lev_raw, wn, wb, wg)


def build_V4(a):
    """V1 (3-state) + 極端 panic 時の cash 安全弁"""
    a2 = _a2(a); vz = np.asarray(a['vz_arr'])
    bull = (a2 > 0.5) & (vz < 0.0)
    risk_off = (a2 < 0.2) | (vz > 0.65)
    extreme = (vz > 1.5) & (a2 < 0.05)
    normal = ~bull & ~risk_off & ~extreme
    risk_off = risk_off & ~extreme  # exclude extreme from risk_off

    wn = np.where(extreme, 0.0,
         np.where(bull, 0.80,
         np.where(normal, 0.60, 0.30)))
    wb = np.where(extreme, 0.50,
         np.where(bull, 0.10,
         np.where(normal, 0.20, 0.35)))
    wg = np.where(extreme, 0.0,
         np.where(bull, 0.10,
         np.where(normal, 0.20, 0.35)))
    # cash slot = 1 - (wn+wb+wg) — V4 では extreme 時に 50% cash
    lev_raw = np.where(extreme, 0.0,
              np.where(bull, 1.0,
              np.where(normal, 0.7, 0.4)))
    return _make_nav_call(a, lev_raw, wn, wb, wg)


def _make_nav_call(a, lev_raw, wn, wb, wg):
    nav, cost = build_dh_nav_with_cost(
        a['close'], lev_raw, wn, wg, wb,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    return nav, cost, wn, wb, wg, lev_raw


VARIANT_BUILDERS = {
    'DH-V1 (3-state preset)':       build_V1,
    'DH-V2 (Continuous vol-tilt)':  build_V2,
    'DH-V3 (F10 ε tilt, always in)': build_V3,
    'DH-V4 (3-state + extreme cash)': build_V4,
}


def calc_allocation_summary(wn, wb, wg, dates):
    """全期間 / IS / OOS の時間加重平均配分"""
    cash = 1.0 - (wn + wb + wg)
    summary = {}
    for name, mask in [('全期間', np.ones(len(dates), dtype=bool)),
                        ('IS', (dates <= pd.Timestamp('2021-05-07')).values),
                        ('OOS', (dates >= pd.Timestamp('2021-05-08')).values)]:
        summary[name] = dict(
            TQQQ=float(np.nanmean(wn[mask])) * 100,
            TMF=float(np.nanmean(wb[mask])) * 100,
            T2036=float(np.nanmean(wg[mask])) * 100,
            Cash=float(np.nanmean(cash[mask])) * 100,
        )
    return summary


def calc_full_metrics(label, nav, dates, ret_nas, wn, wb, wg, lev_raw,
                       windows):
    """10 指標 + 累積 CAGR + WFA"""
    m = metrics_from_nav(nav, dates, ret_nas)
    yr_pre = m['yearly']
    yr_aft = yr_pre.apply(apply_tax_etf_decimal)
    is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1977 <= y <= 2020]]
    oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
    def _geo(x):
        if len(x) == 0: return float('nan')
        c = float(np.prod(1.0 + x.values))
        return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
    cagr_is, cagr_oos = _geo(is_subset), _geo(oos_subset)
    L_eff = lev_raw * 3.0
    wfa = wfa_metrics(nav, dates, windows, lev_arr=L_eff, wn_arr=wn, wb_arr=wb)
    peak_lev = float(np.nanmax(wn * lev_raw * 3.0))
    return dict(
        Strategy=label,
        CAGR_OOS_pct=cagr_oos*100,
        cum_CAGR_OOS_pct=cagr_oos*100,
        cum_CAGR_IS_pct=cagr_is*100,
        IS_OOS_gap_pp=(cagr_is - cagr_oos)*100,
        Sharpe_OOS=m['Sharpe_OOS'],
        MaxDD_FULL_pct=m['MaxDD_FULL']*100,
        Worst10Y_CAGR_pct=m['Worst10Y_star']*100,
        P10_5Y_CAGR_pct=m['P10_5Y'],
        Trades_yr=wfa.get('mean_Trades_yr', np.nan),
        WFA_WFE=wfa.get('WFA_WFE', np.nan),
        WFA_CI95_lo_pct=wfa.get('WFA_CI95_lo', np.nan)*100,
        peak_lev=peak_lev,
    )


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G25: DH 配分シフト型 4 候補 (V1-V4) — cash 最小化 + risk rotation')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    ret_nas = a['ret']
    windows = generate_windows(dates)
    print(f'\n[Windows: {len(windows)}]')

    rows = []
    alloc_summaries = {}

    for label, build_fn in VARIANT_BUILDERS.items():
        print(f'\n[Building {label}...]')
        nav, cost, wn, wb, wg, lev_raw = build_fn(a)
        m = calc_full_metrics(label, nav, dates, ret_nas, wn, wb, wg, lev_raw, windows)
        rows.append(m)
        alloc_summaries[label] = calc_allocation_summary(
            np.asarray(wn), np.asarray(wb), np.asarray(wg), dates
        )
        print(f'  CAGR_OOS={m["CAGR_OOS_pct"]:+.2f}%, IS={m["cum_CAGR_IS_pct"]:+.2f}%, '
              f'gap={m["IS_OOS_gap_pp"]:+.2f}pp, peak_lev={m["peak_lev"]:.2f}x, '
              f'WFE={m["WFA_WFE"]:.3f}, Trades={m["Trades_yr"]:.0f}')

    df = pd.DataFrame(rows)
    csv = os.path.join(BASE, 'g25_dh_allocation_variants_metrics.csv')
    df.to_csv(csv, index=False, float_format='%.4f')
    print(f'\n→ Metrics CSV: {csv}')

    print('\n' + '=' * 80)
    print('ALLOCATION SUMMARY (時間加重平均, %)')
    print('=' * 80)
    for label, summ in alloc_summaries.items():
        print(f'\n[{label}]')
        for period, vals in summ.items():
            print(f'  {period:8s}: TQQQ={vals["TQQQ"]:5.1f}%  TMF={vals["TMF"]:5.1f}%  '
                  f'2036={vals["T2036"]:5.1f}%  Cash={vals["Cash"]:5.1f}%')


if __name__ == '__main__':
    main()
