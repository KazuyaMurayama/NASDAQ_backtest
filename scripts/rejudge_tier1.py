"""Re-judge Tier 1 CSV with relaxed thresholds, write updated CSV + delta report.

Created: 2026-06-05
Purpose: Strict thresholds yielded 0 STRONG/STANDARD PASS in Tier 1. Re-judge with
2x looser severe-degradation thresholds to see whether viable adoption candidates
emerge (and whether Tier 2/3/4 is worth running).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
from integration.nine_metric_eval import judge_improvement_relaxed

CSV = ROOT / 'data' / 'signals' / 'integration' / 'tier1_results_20260605.csv'
OUT_CSV = ROOT / 'data' / 'signals' / 'integration' / 'tier1_results_relaxed_20260605.csv'


def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} patterns")

    # Re-judge each row using relaxed thresholds
    relaxed_results = []
    for _, row in df.iterrows():
        metrics_dict = row.to_dict()
        out = judge_improvement_relaxed(metrics_dict)
        relaxed_results.append(out)

    rel_df = pd.DataFrame(relaxed_results)
    merged = pd.concat([df.reset_index(drop=True), rel_df], axis=1)
    merged.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    # Print summary
    print("\nStrict vs Relaxed judgment counts:")
    print("\nStrict:")
    print(df['judgment'].value_counts().to_string())
    print("\nRelaxed:")
    print(rel_df['judgment_relaxed'].value_counts().to_string())

    # Per-strategy relaxed
    print("\nRelaxed per-strategy:")
    merged['_strat'] = merged['strategy']
    for strat in ['S1', 'S2', 'S3']:
        sub = merged[merged['_strat'] == strat]
        s = (sub['judgment_relaxed'] == 'STRONG_PASS_RELAXED').sum()
        st = (sub['judgment_relaxed'] == 'STANDARD_PASS_RELAXED').sum()
        m = (sub['judgment_relaxed'] == 'MARGINAL_RELAXED').sum()
        f = (sub['judgment_relaxed'] == 'FAIL_RELAXED').sum()
        print(f"  {strat}: STRONG={s} STANDARD={st} MARGINAL={m} FAIL={f}")

    # Top RELAXED PASS patterns
    pass_rel = merged[merged['judgment_relaxed'].isin(['STRONG_PASS_RELAXED', 'STANDARD_PASS_RELAXED'])]
    if not pass_rel.empty:
        print(f"\nTop relaxed PASS (showing pattern_id, improvements, degradations):")
        for _, r in pass_rel.head(30).iterrows():
            print(f"  {r['pattern_id']:50s} imp={r['improved_axes_relaxed']:40s} deg={r['degraded_axes_relaxed']}")
    else:
        print("\nNo STRONG/STANDARD relaxed PASS patterns.")

    # Notable upgrades: MARGINAL -> STANDARD_PASS_RELAXED / STRONG_PASS_RELAXED
    upgraded = merged[(merged['judgment'] == 'MARGINAL') & (merged['judgment_relaxed'].isin(['STRONG_PASS_RELAXED', 'STANDARD_PASS_RELAXED']))]
    print(f"\nUpgraded MARGINAL -> RELAXED PASS: {len(upgraded)} patterns")


if __name__ == '__main__':
    main()
