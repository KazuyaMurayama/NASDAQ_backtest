"""
G19C: DH Dyn 2x3x [A] 厳密 WFA + signal audit
=================================================================
v6.1 で DH [A] の IS-OOS gap CAGR = +10.29pp の過適合警告。
本 script は:
  1. DH [A] の 50-window WFA (g14 同等の窓設計)
  2. signal audit: build_a2_signal の IS/OOS 分布比較
  3. per-window CAGR の IS 期間 variance 確認

判定:
  - True overfit (signal が IS regime に依存) vs Spurious gap (OOS 偶発)
  - 採用見送り or 部分採用 or 修正版設計
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import (
    load_shared_assets, BASE,
    generate_windows, compute_window_metrics, compute_summary_stats,
    evaluate_criteria,
)
from g18_daily_trade_cost_wfa import (
    build_dh_nav_with_cost, metrics_from_nav,
    apply_tax_etf_decimal, COST_CASES_DH,
    IS_END_TS, OOS_START_TS, OOS_END_TS,
    wfa_metrics,
)
from corrected_strategy_backtest import TRADING_DAYS


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G19C: DH Dyn [A] 厳密 WFA + signal audit')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    close = a['close']
    windows = generate_windows(dates)
    print(f'  Windows: {len(windows)}')

    # 1. DH [A] WFA per window (3 cost cases)
    print('\n[S2] DH [A] WFA per window (3 cost cases)')
    all_windows = []
    summaries = []
    for case, per_unit_cost in COST_CASES_DH.items():
        nav_adj, _ = build_dh_nav_with_cost(
            close, a['lev_raw'], a['wn_A'], a['wg_A'], a['wb_A'], dates,
            a['gold_2x'], a['bond_3x'], a['sofr'], per_unit_cost,
        )
        L_eff = np.asarray(a['lev_raw']) * 3.0
        wfa = wfa_metrics(nav_adj, dates, windows,
                           lev_arr=L_eff, wn_arr=a['wn_A'], wb_arr=a['wb_A'])
        verdict, crits = evaluate_criteria(wfa)
        wfa['case'] = case
        wfa['verdict'] = verdict
        summaries.append(wfa)
        print(f'  [{case}] CI95_lo={wfa["WFA_CI95_lo"]:+.3f}, WFE={wfa["WFA_WFE"]:.3f}, '
              f'verdict={verdict}, mean_IS={wfa["mean_CAGR_IS"]*100:+.2f}%, '
              f'mean_postIS={wfa["mean_CAGR_postIS"]*100:+.2f}%')

    sdf = pd.DataFrame(summaries)
    summary_csv = os.path.join(BASE, 'g19c_dh_dyn_wfa_summary.csv')
    sdf.to_csv(summary_csv, index=False)
    print(f'\n→ Summary CSV: {summary_csv}')

    # 2. Signal audit: build_a2_signal の IS/OOS 分布
    print('\n[S3] Signal audit: raw_a2 distribution IS vs OOS')
    raw_a2 = a['raw_a2']
    raw_vals = raw_a2.values
    is_mask = dates <= IS_END_TS
    oos_mask = dates >= OOS_START_TS
    is_raw = raw_vals[is_mask.values]
    oos_raw = raw_vals[oos_mask.values]

    audit = pd.DataFrame({
        'metric': ['mean', 'median', 'std', 'p10', 'p25', 'p50', 'p75', 'p90',
                    '% > 0.15', '% > 0.30', '% > 0.50', '% > 0.70'],
        'IS (1974-2021)': [
            float(np.nanmean(is_raw)),
            float(np.nanmedian(is_raw)),
            float(np.nanstd(is_raw)),
            float(np.nanpercentile(is_raw, 10)),
            float(np.nanpercentile(is_raw, 25)),
            float(np.nanpercentile(is_raw, 50)),
            float(np.nanpercentile(is_raw, 75)),
            float(np.nanpercentile(is_raw, 90)),
            float((is_raw > 0.15).mean()*100),
            float((is_raw > 0.30).mean()*100),
            float((is_raw > 0.50).mean()*100),
            float((is_raw > 0.70).mean()*100),
        ],
        'OOS (2021-2026)': [
            float(np.nanmean(oos_raw)),
            float(np.nanmedian(oos_raw)),
            float(np.nanstd(oos_raw)),
            float(np.nanpercentile(oos_raw, 10)),
            float(np.nanpercentile(oos_raw, 25)),
            float(np.nanpercentile(oos_raw, 50)),
            float(np.nanpercentile(oos_raw, 75)),
            float(np.nanpercentile(oos_raw, 90)),
            float((oos_raw > 0.15).mean()*100),
            float((oos_raw > 0.30).mean()*100),
            float((oos_raw > 0.50).mean()*100),
            float((oos_raw > 0.70).mean()*100),
        ],
    })
    print(audit.round(4).to_string(index=False))

    # 3. Per-window CAGR variance check
    print('\n[S4] Per-window CAGR variance (moderate cost)')
    nav_adj, _ = build_dh_nav_with_cost(
        close, a['lev_raw'], a['wn_A'], a['wg_A'], a['wb_A'], dates,
        a['gold_2x'], a['bond_3x'], a['sofr'], 0.0010,  # moderate
    )
    L_eff = np.asarray(a['lev_raw']) * 3.0

    per_window = []
    for w in windows:
        m = compute_window_metrics(nav_adj, w, wn=a['wn_A'], wb=a['wb_A'], lev_arr=L_eff)
        m['window_id'] = w['window_id']
        m['start_date'] = w['start_date']
        per_window.append(m)
    pwdf = pd.DataFrame(per_window)
    pwdf['is_oos'] = pwdf['start_date'] >= OOS_START_TS
    pw_csv = os.path.join(BASE, 'g19c_dh_dyn_per_window.csv')
    pwdf.to_csv(pw_csv, index=False)
    print(f'→ Per-window CSV: {pw_csv}')

    # IS vs OOS CAGR distribution
    is_cagrs = pwdf.loc[~pwdf['is_oos'], 'CAGR'].dropna().values
    oos_cagrs = pwdf.loc[pwdf['is_oos'], 'CAGR'].dropna().values
    print(f'\n  IS windows ({len(is_cagrs)}):  mean={is_cagrs.mean()*100:+.2f}%, '
          f'std={is_cagrs.std()*100:.2f}%, '
          f'median={np.median(is_cagrs)*100:+.2f}%')
    print(f'  OOS windows ({len(oos_cagrs)}): mean={oos_cagrs.mean()*100:+.2f}%, '
          f'std={oos_cagrs.std()*100:.2f}%, '
          f'median={np.median(oos_cagrs)*100:+.2f}%')

    # signal audit CSV
    audit_csv = os.path.join(BASE, 'g19c_dh_dyn_signal_audit.csv')
    audit.to_csv(audit_csv, index=False)
    print(f'\n→ Audit CSV: {audit_csv}')

    # 結論
    print('\n[S5] 結論')
    gap_pct = sdf.loc[1, 'mean_CAGR_IS'] - sdf.loc[1, 'mean_CAGR_postIS']  # moderate
    print(f'  WFA mean CAGR IS - postIS gap: {gap_pct*100:+.2f}pp (moderate)')
    is_oos_dist_diff = abs(is_raw.mean() - oos_raw.mean())
    print(f'  raw_a2 mean diff IS vs OOS: {is_oos_dist_diff:+.4f}')
    print(f'  → signal 分布の差が大きい場合 = signal IS overfit の証拠')
    print(f'  → 分布差が小さい場合 = 信号は同等だが outcome が異なる (regime spurious)')


if __name__ == '__main__':
    main()
