"""Phase B screening: scorecard + report.

For each (signal, asset, horizon) triple:
  - rolling Spearman IC (window=252) + Newey-West HAC t-stat + p-value
  - conditional hit rate at signal == max_level
  - half-sample + decade IC stability
  - BH-FDR across all triples
  - PASS/FAIL judgment

PASS criteria (per plan §5.5):
  PASS if (
    headline 20d horizon: BH-FDR p < 0.10 AND |mean_ic| > 0.05
    AND hit_rate Wilson_lower > base_rate + 3pp (at max level)
    AND half-sample same_sign True
  )
  OR PASS if (
    20d AND 60d both have |mean_ic| > 0.04 (stable across horizons)
    AND decade same_sign True
    AND 20d and 60d agree in sign
  )

Note on hit-rate units
----------------------
``signals.hit_rate.hit_rate_lift`` returns ``lift_pp`` already scaled
to percentage points (multiplied by 100). The "Wilson_lower >
base_rate + 3pp" check is done in raw proportion space
(Wilson_lower and base_rate are both in [0, 1]), so the comparison
uses ``+ 0.03`` not ``+ 3``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .ic import compute_ic, ic_tstat_newey_west
from .hit_rate import hit_rate_lift
from .stability import half_sample_ic_check, decade_ic_check
from .multiplicity import fdr_bh


@dataclass
class ScorecardRow:
    signal_id: int
    signal_name: str
    asset: str
    horizon: int
    n_obs: int
    mean_ic: float
    t_stat: float
    p_value: float
    p_value_bh: Optional[float]  # filled in batch BH step
    hit_rate_max_level: float
    hit_wilson_lower: float
    base_rate: float
    hit_lift_pp: float
    half_sample_same_sign: bool
    decade_same_sign: bool
    pass_flag: Optional[bool]  # filled after BH


def _p_value_from_t(t_stat: float) -> float:
    """Two-sided p-value via normal large-sample approximation."""
    if t_stat is None or np.isnan(t_stat):
        return float('nan')
    return float(2 * (1 - scipy_stats.norm.cdf(abs(t_stat))))


def evaluate_triple(
    signal_id: int,
    signal_name: str,
    signal: pd.Series,
    asset: str,
    horizon: int,
    forward_returns: pd.Series,
    ic_window: int = 252,
) -> ScorecardRow:
    """Compute all metrics for one (signal, asset, horizon)."""
    # IC + Newey-West
    ic = compute_ic(signal, forward_returns, window=ic_window)
    ic_clean = ic.dropna()
    mean_ic = float(ic_clean.mean()) if len(ic_clean) else float('nan')
    t_stat = ic_tstat_newey_west(ic) if len(ic_clean) >= 30 else float('nan')
    p_val = _p_value_from_t(t_stat)

    # Hit rate at maximum signal level (quantized signals: max bucket =
    # most "extreme" condition - bottom-tail or top-tail depending on direction).
    sig_clean = signal.dropna()
    if len(sig_clean):
        try:
            max_level = int(sig_clean.max())
        except (ValueError, TypeError):
            max_level = 0
    else:
        max_level = 0
    hr = hit_rate_lift(signal, forward_returns, signal_value=max_level)

    # Stability
    half = half_sample_ic_check(signal, forward_returns)
    dec = decade_ic_check(signal, forward_returns)

    return ScorecardRow(
        signal_id=signal_id,
        signal_name=signal_name,
        asset=asset,
        horizon=int(horizon),
        n_obs=int(len(ic_clean)),
        mean_ic=mean_ic,
        t_stat=float(t_stat) if not np.isnan(t_stat) else float('nan'),
        p_value=p_val,
        p_value_bh=None,
        hit_rate_max_level=float(hr['hit_rate']) if hr['n_conditional'] else float('nan'),
        hit_wilson_lower=float(hr['wilson_lower_95']) if hr['n_conditional'] else float('nan'),
        base_rate=float(hr['base_rate']),
        hit_lift_pp=float(hr['lift_pp']) if hr['n_conditional'] else float('nan'),
        half_sample_same_sign=bool(half['same_sign']),
        decade_same_sign=bool(dec['same_sign']),
        pass_flag=None,
    )


def batch_evaluate(
    triples: List[Tuple],  # (signal_id, signal_name, signal_series, asset, horizon, fwd_ret_series)
    ic_window: int = 252,
) -> pd.DataFrame:
    """Run evaluate_triple on a list of triples and return scorecard DataFrame."""
    rows = []
    for tup in triples:
        sid, sname, sig, asset, h, fr = tup
        r = evaluate_triple(sid, sname, sig, asset, h, fr, ic_window=ic_window)
        rows.append(asdict(r))
    if not rows:
        return pd.DataFrame(columns=[f.name for f in ScorecardRow.__dataclass_fields__.values()])
    return pd.DataFrame(rows)


def apply_fdr_and_judgment(scorecard: pd.DataFrame, alpha: float = 0.10) -> pd.DataFrame:
    """Apply BH-FDR across all rows + PASS/FAIL judgment per plan §5.5."""
    scorecard = scorecard.copy()
    if 'p_value_bh' not in scorecard.columns:
        scorecard['p_value_bh'] = np.nan
    scorecard['p_value_bh'] = scorecard['p_value_bh'].astype('float64')

    # BH-FDR across all valid p-values
    valid_mask = scorecard['p_value'].notna()
    valid = scorecard.loc[valid_mask].copy()
    if not valid.empty:
        fdr = fdr_bh(valid['p_value'], alpha=alpha)
        # fdr_bh is indexed by the position labels of the input series.
        scorecard.loc[valid.index, 'p_value_bh'] = fdr['p_bh'].values

    def _judge_primary(row) -> bool:
        if row['horizon'] != 20:
            return False
        if pd.isna(row['p_value_bh']) or row['p_value_bh'] >= alpha:
            return False
        if pd.isna(row['mean_ic']) or abs(row['mean_ic']) <= 0.05:
            return False
        if pd.isna(row['hit_wilson_lower']) or pd.isna(row['base_rate']):
            return False
        if row['hit_wilson_lower'] <= row['base_rate'] + 0.03:
            return False
        if not row['half_sample_same_sign']:
            return False
        return True

    scorecard['pass_flag'] = scorecard.apply(_judge_primary, axis=1)

    # Secondary: 20d AND 60d both have |IC|>0.04, decade same-sign, agreeing sign.
    secondary_passes = set()
    for (sid, asset), grp in scorecard.groupby(['signal_id', 'asset']):
        h20 = grp[grp['horizon'] == 20]
        h60 = grp[grp['horizon'] == 60]
        if h20.empty or h60.empty:
            continue
        r20 = h20.iloc[0]
        r60 = h60.iloc[0]
        if pd.isna(r20['mean_ic']) or pd.isna(r60['mean_ic']):
            continue
        if (abs(r20['mean_ic']) > 0.04 and abs(r60['mean_ic']) > 0.04
                and r20['decade_same_sign'] and r60['decade_same_sign']
                and np.sign(r20['mean_ic']) == np.sign(r60['mean_ic'])
                and np.sign(r20['mean_ic']) != 0):
            secondary_passes.add((sid, asset, 20))
            secondary_passes.add((sid, asset, 60))

    for idx, row in scorecard.iterrows():
        key = (row['signal_id'], row['asset'], int(row['horizon']))
        if key in secondary_passes:
            scorecard.at[idx, 'pass_flag'] = True

    # Ensure pass_flag is boolean
    scorecard['pass_flag'] = scorecard['pass_flag'].astype(bool)
    return scorecard


def _fail_reasons(row, alpha: float = 0.10) -> str:
    reasons = []
    if pd.isna(row['p_value_bh']) or row['p_value_bh'] >= alpha:
        reasons.append('FDR>=10%')
    if pd.isna(row['mean_ic']) or abs(row['mean_ic']) <= 0.05:
        reasons.append('|IC|<=0.05')
    if (pd.isna(row['hit_wilson_lower']) or pd.isna(row['base_rate'])
            or row['hit_wilson_lower'] <= row['base_rate'] + 0.03):
        reasons.append('Wilson_lower<=base+3pp')
    if not row['half_sample_same_sign']:
        reasons.append('half-sample sign flip')
    if not row['decade_same_sign']:
        reasons.append('decade sign mixed')
    return ', '.join(reasons) if reasons else '(complex)'


def generate_report_markdown(
    scorecard: pd.DataFrame,
    out_path: str,
    title: str = "Phase B Screening Report",
    created_date: str = "2026-06-04",
) -> str:
    """Generate human-readable markdown report."""
    pass_rows = (
        scorecard[scorecard['pass_flag'] == True]
        .assign(_abs_ic=lambda d: d['mean_ic'].abs())
        .sort_values('_abs_ic', ascending=False)
        .drop(columns='_abs_ic')
    )
    fail_rows = scorecard[scorecard['pass_flag'] != True]

    n_total = len(scorecard)
    n_pass = len(pass_rows)
    n_fail = len(fail_rows)
    n_signals = int(scorecard['signal_id'].nunique())
    n_signals_pass = int(pass_rows['signal_id'].nunique()) if not pass_rows.empty else 0

    lines = [
        f"# {title}",
        "",
        f"作成日: {created_date}",
        f"最終更新日: {created_date}",
        "",
        "## サマリ",
        "",
        f"- 評価triple数: {n_total}",
        f"- PASS: **{n_pass}** triples ({n_signals_pass} unique signals)",
        f"- FAIL: {n_fail} triples",
        f"- ユニーク信号数 (全体): {n_signals}",
        f"- BH-FDR alpha: 0.10",
        "",
        "## 採用 (PASS) 信号",
        "",
    ]

    if pass_rows.empty:
        lines.append(
            "**該当なし**。Phase B 閾値での通過信号はゼロ。"
            "フォールバック (IC>0.03, FDR<15%) を §5.5 の事前合意に従い検討要。"
        )
    else:
        lines.extend([
            "| signal_id | name | asset | horizon | mean_IC | p_BH | hit_rate | lift_pp | direction |",
            "|---|---|---|---|---|---|---|---|---|",
        ])
        for _, r in pass_rows.iterrows():
            if pd.isna(r['mean_ic']):
                direction = 'n/a'
            elif r['mean_ic'] > 0:
                direction = 'positive'
            else:
                direction = 'negative (inverse)'
            p_bh_str = (
                f"{r['p_value_bh']:.4f}" if pd.notna(r['p_value_bh']) else 'NaN'
            )
            hr_str = (
                f"{r['hit_rate_max_level']:.3f}" if pd.notna(r['hit_rate_max_level']) else 'NaN'
            )
            lift_str = (
                f"{r['hit_lift_pp']:+.2f}pp" if pd.notna(r['hit_lift_pp']) else 'NaN'
            )
            lines.append(
                f"| {int(r['signal_id'])} | {r['signal_name']} | {r['asset']} | {int(r['horizon'])}d "
                f"| {r['mean_ic']:+.4f} | {p_bh_str} | {hr_str} | {lift_str} | {direction} |"
            )
        lines.extend([
            "",
            "### 解釈ガイド",
            "",
            "- **positive IC**: 信号値↑ (より高いバケット) → forward return↑ (順方向先行指標)",
            "- **negative IC**: 信号値↑ → forward return↓ (逆張り先行指標)",
            "- **lift_pp**: max信号レベル条件下のヒット率 - base rate (パーセントポイント)",
            "- **p_BH**: BH-FDR 補正済 p 値 (`alpha=0.10`)",
        ])

    lines.extend(["", "## 棄却 (FAIL) 信号 — 主要因", ""])
    if fail_rows.empty:
        lines.append("**該当なし**。全 triple が PASS。")
    else:
        lines.append("| signal_id | name | asset | horizon | mean_IC | p_BH | 主要因 |")
        lines.append("|---|---|---|---|---|---|---|")
        # Sort by signal_id, asset, horizon for readability.
        fail_sorted = fail_rows.sort_values(['signal_id', 'asset', 'horizon'])
        for _, r in fail_sorted.iterrows():
            p_bh_str = (
                f"{r['p_value_bh']:.4f}" if pd.notna(r['p_value_bh']) else 'NaN'
            )
            ic_str = f"{r['mean_ic']:+.4f}" if pd.notna(r['mean_ic']) else 'NaN'
            lines.append(
                f"| {int(r['signal_id'])} | {r['signal_name']} | {r['asset']} | {int(r['horizon'])}d "
                f"| {ic_str} | {p_bh_str} | {_fail_reasons(r)} |"
            )

    # Summary stats: FDR distribution
    lines.extend([
        "",
        "## 統計サマリ",
        "",
    ])
    if scorecard['p_value'].notna().any():
        n_raw_sig = int((scorecard['p_value'] < 0.10).sum())
        n_bh_sig = int((scorecard['p_value_bh'] < 0.10).sum())
        ic_abs = scorecard['mean_ic'].abs()
        lines.extend([
            f"- raw p<0.10: {n_raw_sig} / {n_total}",
            f"- BH p<0.10: {n_bh_sig} / {n_total}",
            f"- |IC| 中央値: {ic_abs.median():.4f}",
            f"- |IC| 95%分位: {ic_abs.quantile(0.95):.4f}",
            f"- |IC| max: {ic_abs.max():.4f}",
            "",
        ])

    lines.extend([
        "## 採用判定ルール",
        "",
        "**Primary (20d horizon):** BH-FDR<0.10 AND |IC|>0.05 AND Wilson_lower > base+3pp AND 半分割同符号",
        "",
        "**Secondary (20d AND 60d):** |IC|>0.04 両水準 AND decade 同符号 AND 20d/60d 同符号",
        "",
        "## 次工程 (Phase C)",
        "",
        "PASS 信号を `phase_b_selection_<date>.csv` に出力。"
        "Phase C で overlay/standalone モードでの WFA 評価へ。",
        "",
    ])

    out_text = "\n".join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(out_text)
    return out_text
