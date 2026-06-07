"""Re-compute cross-strategy transfer WFE/CI95_lo with canonical split (2021-05-08).

Builds candidate NAVs for S1/S2/S3/E4-equivalent × nasdaq_mom63 × M6 defensive overlay,
then computes WFE_calendar + CI95_lo_annual against canonical IS/OOS split.

Note: S1/S2/S3 baselines come from baseline_navs_20260605.parquet.
For E4, we use the cached e4 nav_e4 (=baseline). The "E4 + overlay" candidate is built
on S2 (D5) infrastructure since E4 = S2_VZGated regime kLT; the session5_phase_d
audit reused build_candidate_nav('S2', ...). We follow that convention here.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import pickle
import pandas as pd
import numpy as np
from recompute_9metrics_for_decision import (  # type: ignore
    cagr, sharpe_annual, maxdd, worst10y_calendar, p10_5y_rolling,
    wfa_ci95_lo_annual, wfa_wfe_calendar, wfe_50w_sharpe,
    STANDARD_SPLIT, STANDARD_IS_END,
)


def main():
    from integration.build_strategy_with_signal import build_candidate_nav  # noqa: E402

    macro = pd.read_csv(
        ROOT / 'data' / 'macro_features.csv',
        parse_dates=['date'], index_col='date',
    ).sort_index()
    signal_raw = macro['nasdaq_mom63'].dropna()

    bp = pd.read_parquet(ROOT / 'data' / 'signals' / 'integration' / 'baseline_navs_20260605.parquet')
    rows = []

    for strat_label, strat_key in [
        ('S2_D5', 'S2'),
        ('S3_DH-W1', 'S3'),
        # E4 transfer follows session5 convention (overlay applied via S2 infrastructure)
        ('E4_via_S2', 'S2'),
    ]:
        baseline_nav = bp[strat_key].dropna()
        cand_nav = build_candidate_nav(strat_key, signal_raw, 'M6', 'defensive').dropna()

        for tag, nav in [('baseline', baseline_nav), ('candidate', cand_nav)]:
            rows.append({
                'transfer_target': strat_label,
                'nav_type': tag,
                'CAGR_IS': cagr(nav.loc[:STANDARD_IS_END]),
                'CAGR_OOS': cagr(nav.loc[STANDARD_SPLIT:]),
                'CAGR_FULL': cagr(nav),
                'Sharpe_OOS': sharpe_annual(nav.loc[STANDARD_SPLIT:]),
                'MaxDD': maxdd(nav),
                'Worst10Y': worst10y_calendar(nav),
                'P10_5Y': p10_5y_rolling(nav),
                'WFA_CI95_lo_annual': wfa_ci95_lo_annual(nav),
                'WFA_WFE_calendar': wfa_wfe_calendar(nav),
                'WFA_WFE_50w_sharpe': wfe_50w_sharpe(nav),
            })

    df = pd.DataFrame(rows)
    out = ROOT / 'data' / 'signals' / 'expansion' / 'decision_transfer_canonicalsplit_20260607.csv'
    df.to_csv(out, index=False, float_format='%.6f')
    print(f'Wrote {out}')

    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
