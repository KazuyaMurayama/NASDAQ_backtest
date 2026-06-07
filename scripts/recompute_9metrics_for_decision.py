"""Re-compute 9+1 metrics with standard IS/OOS split for SIGNAL_EXPANSION_FINAL_DECISION_20260607.md.

Evaluation Standard: v1.1
Cost Scenario: D (src/product_costs.py 2026-05-12 basis) — NAVs already include cost
Canonical IS/OOS split: 2021-05-07 / 2021-05-08 (per backtest_lt_strategies.py & docs/rules/08 §5)
Also reports 2021-01-01 split as a sensitivity check (per prompt request)
Metrics: 標準7 + WFA補助2 (docs/rules/08_evaluation-metrics.md)
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pickle
import pandas as pd
import numpy as np
from scipy.stats import t as student_t

# Canonical split per repo standard
STANDARD_SPLIT = '2021-05-08'   # OOS_START
STANDARD_IS_END = '2021-05-07'  # IS_END
# Sensitivity split (prompt request)
ALT_SPLIT = '2021-01-01'


# ---------------------------------------------------------------------------
# Metric primitives (match src/integration/nine_metric_eval.py conventions)
# ---------------------------------------------------------------------------

def cagr(nav: pd.Series) -> float:
    n = nav.dropna()
    if len(n) < 2 or float(n.iloc[0]) <= 0:
        return float('nan')
    years = (n.index[-1] - n.index[0]).days / 365.25
    if years <= 0:
        return float('nan')
    return float((n.iloc[-1] / n.iloc[0]) ** (1 / years) - 1)


def sharpe_annual(nav: pd.Series) -> float:
    r = nav.pct_change().dropna()
    if len(r) < 30 or float(r.std()) == 0:
        return float('nan')
    return float(r.mean() / r.std() * np.sqrt(252))


def maxdd(nav: pd.Series) -> float:
    n = nav.dropna()
    if len(n) < 2:
        return float('nan')
    return float((n / n.cummax() - 1).min())


def worst10y_calendar(nav: pd.Series) -> float:
    """Calendar-year end NAV, 10yr rolling worst CAGR (Worst10Y★)."""
    yearly = nav.resample('YE').last().dropna()
    if len(yearly) < 11:
        return float('nan')
    rolling = (yearly / yearly.shift(10)) ** (1 / 10) - 1
    rolling = rolling.dropna()
    if rolling.empty:
        return float('nan')
    return float(rolling.min())


def p10_5y_rolling(nav: pd.Series) -> float:
    """Daily rolling 5y CAGR, P10."""
    n = nav.dropna()
    window = 5 * 252
    if len(n) < window + 100:
        return float('nan')
    rolling_cagr = (n / n.shift(window)) ** (1 / 5) - 1
    rolling_cagr = rolling_cagr.dropna()
    if rolling_cagr.empty:
        return float('nan')
    return float(rolling_cagr.quantile(0.10))


def wfa_ci95_lo_annual(nav: pd.Series) -> float:
    """Non-overlapping 1yr (calendar) annual CAGR, t-dist 95% CI lower bound.

    Conforms to docs/rules/08 §1.2 § EVALUATION_STANDARD §3.9 conceptually
    (proxy: calendar-year returns serve as the "non-overlapping 1yr-window
    CAGR" sample for a quick NAV-only audit). Full G1 WFA uses 50 walk-forward
    windows; that requires a separate run.
    """
    yearly = nav.resample('YE').last().dropna()
    if len(yearly) < 5:
        return float('nan')
    annual_ret = yearly.pct_change().dropna()
    n = len(annual_ret)
    if n < 5:
        return float('nan')
    mean = float(annual_ret.mean())
    sd = float(annual_ret.std(ddof=1))
    t_crit = float(student_t.ppf(0.975, n - 1))
    return float(mean - t_crit * sd / np.sqrt(n))


def wfa_wfe_calendar(nav: pd.Series,
                     is_end: str = STANDARD_IS_END,
                     oos_start: str = STANDARD_SPLIT) -> float:
    """Walk-Forward Efficiency (proxy): mean(IS calendar-yr CAGR) vs mean(OOS calendar-yr CAGR).

    Definition: WFE = mean_post / mean_is per src/g1_wfa.py compute_summary_stats.
    For consistency with the prompt's suggestion, we also expose a 50-window
    Sharpe-based variant via wfe_50w_sharpe(...).
    """
    yearly = nav.resample('YE').last().dropna()
    if len(yearly) < 6:
        return float('nan')
    annual_ret = yearly.pct_change().dropna()
    if annual_ret.empty:
        return float('nan')
    is_mask = annual_ret.index <= pd.Timestamp(is_end)
    oos_mask = annual_ret.index >= pd.Timestamp(oos_start)
    is_rets = annual_ret[is_mask]
    oos_rets = annual_ret[oos_mask]
    if len(is_rets) == 0 or len(oos_rets) == 0:
        return float('nan')
    mean_is = float(is_rets.mean())
    mean_oos = float(oos_rets.mean())
    if mean_is == 0:
        return float('nan')
    return float(mean_oos / mean_is)


def wfe_50w_sharpe(nav: pd.Series, n_windows: int = 50) -> float:
    """50-window Sharpe-based WFE proxy (matches src/integration/nine_metric_eval._wfe)."""
    n = nav.dropna()
    if len(n) < n_windows * 60:
        return float('nan')
    full_sharpe = sharpe_annual(n)
    if np.isnan(full_sharpe) or full_sharpe == 0:
        return float('nan')
    win = len(n) // n_windows
    sharpes = []
    for i in range(n_windows):
        s = n.iloc[i * win : (i + 1) * win if i < n_windows - 1 else len(n)]
        sh = sharpe_annual(s)
        if not np.isnan(sh):
            sharpes.append(sh)
    if not sharpes:
        return float('nan')
    return float(np.mean(sharpes) / full_sharpe)


def trades_per_yr_from_nav(nav: pd.Series, baseline_nav: pd.Series | None = None) -> float:
    """Sign-flip proxy. If baseline given, use diff-return sign flips."""
    n = nav.dropna()
    if len(n) < 252:
        return float('nan')
    r = n.pct_change().dropna()
    if baseline_nav is not None:
        b = baseline_nav.pct_change().reindex(r.index).fillna(0)
        diff = r - b
        sign = np.sign(diff.where(diff.abs() > 1e-4, 0.0))
    else:
        sign = np.sign(r)
    flips = int((sign != sign.shift(1)).fillna(False).sum())
    years = (r.index[-1] - r.index[0]).days / 365.25
    return float(flips / max(years, 1e-9))


def all_metrics(nav: pd.Series, label: str,
                split: str = STANDARD_SPLIT,
                is_end: str = STANDARD_IS_END,
                trades_yr_exact: float | None = None,
                baseline_nav_for_trades: pd.Series | None = None) -> dict:
    is_nav = nav.loc[:is_end]
    oos_nav = nav.loc[split:]
    c_is = cagr(is_nav)
    c_oos = cagr(oos_nav)
    return {
        'label': label,
        'period_start': str(nav.index[0].date()),
        'period_end': str(nav.index[-1].date()),
        'n_obs': len(nav),
        'CAGR_IS': c_is,
        'CAGR_OOS': c_oos,
        'CAGR_FULL': cagr(nav),
        'Sharpe_OOS': sharpe_annual(oos_nav),
        'MaxDD_FULL': maxdd(nav),
        'Worst10Y_calendar': worst10y_calendar(nav),
        'P10_5Y': p10_5y_rolling(nav),
        'IS_OOS_gap_pp': (c_is - c_oos) * 100 if not (np.isnan(c_is) or np.isnan(c_oos)) else float('nan'),
        'Trades_per_yr_exact': trades_yr_exact if trades_yr_exact is not None else float('nan'),
        'Trades_per_yr_navproxy': trades_per_yr_from_nav(nav, baseline_nav_for_trades),
        'WFA_CI95_lo_annual': wfa_ci95_lo_annual(nav),
        'WFA_WFE_calendar': wfa_wfe_calendar(nav, is_end=is_end, oos_start=split),
        'WFA_WFE_50w_sharpe': wfe_50w_sharpe(nav),
    }


# ---------------------------------------------------------------------------
# NAV loaders
# ---------------------------------------------------------------------------

def load_baseline_navs() -> dict[str, pd.Series]:
    bp = pd.read_parquet(ROOT / 'data' / 'signals' / 'integration' / 'baseline_navs_20260605.parquet')
    return {
        'S1': bp['S1'].dropna(),
        'S2': bp['S2'].dropna(),
        'S3': bp['S3'].dropna(),
    }


def load_e4_nav() -> tuple[pd.Series, float]:
    """Return (nav, exact trades/yr)."""
    c = pickle.load(open(ROOT / 'audit_results' / '_cache' / 'e4_nav_cache.pkl', 'rb'))
    dates = pd.to_datetime(c['dates'])
    nav_raw = c['nav_e4']
    nav = pd.Series(np.asarray(nav_raw), index=dates).dropna()
    trades_yr = float(c['n_tr']) / float(c['n_years'])
    return nav, trades_yr


def build_adopt_candidate_nav() -> pd.Series:
    """S3 (DH-W1) + nasdaq_mom63 × M6 defensive overlay."""
    from integration.build_strategy_with_signal import build_candidate_nav  # noqa: E402

    macro = pd.read_csv(
        ROOT / 'data' / 'macro_features.csv',
        parse_dates=['date'], index_col='date',
    ).sort_index()
    signal_raw = macro['nasdaq_mom63'].dropna()
    cand = build_candidate_nav('S3', signal_raw, 'M6', 'defensive').dropna()
    return cand


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    print('=== Loading baseline NAVs ===')
    bn = load_baseline_navs()
    s1_nav, s2_nav, s3_nav = bn['S1'], bn['S2'], bn['S3']
    print(f'  S1: {len(s1_nav)} obs  [{s1_nav.index.min().date()} → {s1_nav.index.max().date()}]')
    print(f'  S2: {len(s2_nav)} obs  [{s2_nav.index.min().date()} → {s2_nav.index.max().date()}]')
    print(f'  S3: {len(s3_nav)} obs  [{s3_nav.index.min().date()} → {s3_nav.index.max().date()}]')

    print('=== Loading E4 NAV (current §1 Active) ===')
    e4_nav, e4_trades_yr = load_e4_nav()
    print(f'  E4: {len(e4_nav)} obs  [{e4_nav.index.min().date()} → {e4_nav.index.max().date()}]  trades/yr_exact={e4_trades_yr:.3f}')

    print('=== Building ADOPT candidate (S3 DH-W1 + nasdaq_mom63 × M6 defensive) ===')
    try:
        cand_s3 = build_adopt_candidate_nav()
    except Exception as e:
        print(f'  ERROR: failed to build ADOPT candidate: {e}')
        cand_s3 = None

    print('\n=== Canonical split (IS_END=2021-05-07 / OOS_START=2021-05-08) ===')
    rows_canonical = [
        all_metrics(s1_nav, 'S1_F10_baseline'),
        all_metrics(s2_nav, 'S2_D5_baseline'),
        all_metrics(s3_nav, 'S3_DH-W1_baseline'),
        all_metrics(e4_nav, 'E4_Active_CFD',
                    trades_yr_exact=e4_trades_yr),
    ]
    if cand_s3 is not None:
        rows_canonical.append(
            all_metrics(cand_s3, 'S3_DH-W1_+nasdaq_mom63_M6_def',
                        baseline_nav_for_trades=s3_nav)
        )
    df_canonical = pd.DataFrame(rows_canonical)
    df_canonical['split_used'] = STANDARD_SPLIT

    print('\n=== Alt split (2021-01-01) for sensitivity ===')
    rows_alt = [
        all_metrics(s1_nav, 'S1_F10_baseline', split=ALT_SPLIT, is_end='2020-12-31'),
        all_metrics(s2_nav, 'S2_D5_baseline', split=ALT_SPLIT, is_end='2020-12-31'),
        all_metrics(s3_nav, 'S3_DH-W1_baseline', split=ALT_SPLIT, is_end='2020-12-31'),
        all_metrics(e4_nav, 'E4_Active_CFD', split=ALT_SPLIT, is_end='2020-12-31',
                    trades_yr_exact=e4_trades_yr),
    ]
    if cand_s3 is not None:
        rows_alt.append(
            all_metrics(cand_s3, 'S3_DH-W1_+nasdaq_mom63_M6_def',
                        split=ALT_SPLIT, is_end='2020-12-31',
                        baseline_nav_for_trades=s3_nav)
        )
    df_alt = pd.DataFrame(rows_alt)
    df_alt['split_used'] = ALT_SPLIT

    df = pd.concat([df_canonical, df_alt], ignore_index=True)
    out_csv = ROOT / 'data' / 'signals' / 'expansion' / 'decision_9metrics_20260607.csv'
    df.to_csv(out_csv, index=False, float_format='%.6f')
    print(f'\nWrote {out_csv}')

    # Pretty print
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)
    print('\n--- Canonical split (2021-05-08) ---')
    print(df_canonical.drop(columns=['period_start', 'period_end', 'n_obs']).to_string(index=False))
    print('\n--- Alt split (2021-01-01) ---')
    print(df_alt.drop(columns=['period_start', 'period_end', 'n_obs']).to_string(index=False))
    return df


if __name__ == '__main__':
    main()
