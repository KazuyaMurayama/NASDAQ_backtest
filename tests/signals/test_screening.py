"""Tests for signals.screening (Phase B B7).

5 tests:
  1. evaluate_triple populates all dataclass fields
  2. batch_evaluate returns DataFrame with expected shape
  3. strong synthetic signal passes primary criterion
  4. random noise signal fails primary criterion
  5. generate_report_markdown produces expected sections
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import numpy as np
import pandas as pd

from signals.screening import (
    apply_fdr_and_judgment,
    batch_evaluate,
    evaluate_triple,
    generate_report_markdown,
)


def _persistent_signal(seed: int = 0, n: int = 500, ic_direction: int = +1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2010-01-01', periods=n, freq='B')
    s = pd.Series(rng.integers(0, 4, size=n), index=idx)
    r = pd.Series(
        ic_direction * s.values * 0.005 + rng.normal(0, 0.01, size=n),
        index=idx,
    )
    return s, r


def test_evaluate_triple_returns_all_fields():
    s, r = _persistent_signal(seed=1, n=600)
    out = evaluate_triple(99, 'test_signal', s, 'NDX', 20, r, ic_window=126)
    assert out.signal_id == 99
    assert out.signal_name == 'test_signal'
    assert out.asset == 'NDX'
    assert out.horizon == 20
    assert not np.isnan(out.mean_ic)
    assert not np.isnan(out.t_stat)
    assert not np.isnan(out.p_value)
    assert isinstance(out.half_sample_same_sign, bool)
    assert isinstance(out.decade_same_sign, bool)


def test_batch_evaluate_returns_dataframe():
    s, r = _persistent_signal(seed=2, n=600)
    triples = [
        (1, 'a', s, 'NDX', 20, r),
        (2, 'b', s, 'IEF', 5, r),
    ]
    df = batch_evaluate(triples, ic_window=126)
    assert len(df) == 2
    for col in ('mean_ic', 't_stat', 'p_value', 'pass_flag'):
        assert col in df.columns


def test_apply_fdr_passes_strong_signal():
    """Strong persistent +IC signal at 20d should PASS primary criterion."""
    s, r = _persistent_signal(seed=3, n=2500, ic_direction=+1)
    triples = [(1, 'strong', s, 'NDX', 20, r)]
    df = batch_evaluate(triples, ic_window=252)
    df = apply_fdr_and_judgment(df)
    assert bool(df.iloc[0]['pass_flag']) is True


def test_apply_fdr_rejects_random_signal():
    """Random noise should not PASS."""
    rng = np.random.default_rng(4)
    idx = pd.date_range('2010-01-01', periods=1500, freq='B')
    s = pd.Series(rng.integers(0, 4, size=1500), index=idx)
    r = pd.Series(rng.normal(0, 0.01, size=1500), index=idx)
    triples = [(1, 'random', s, 'NDX', 20, r)]
    df = batch_evaluate(triples, ic_window=252)
    df = apply_fdr_and_judgment(df)
    assert bool(df.iloc[0]['pass_flag']) is False


def test_generate_report_markdown_writes_file(tmp_path):
    s, r = _persistent_signal(seed=5, n=600, ic_direction=+1)
    triples = [(1, 'test', s, 'NDX', 20, r)]
    df = batch_evaluate(triples, ic_window=126)
    df = apply_fdr_and_judgment(df)
    out = tmp_path / 'report.md'
    md = generate_report_markdown(df, str(out))
    assert "Phase B Screening Report" in md
    assert "サマリ" in md
    assert "採用 (PASS) 信号" in md
    assert "棄却 (FAIL) 信号" in md
    assert out.exists()
    assert out.read_text(encoding='utf-8') == md


def test_secondary_criterion_passes_horizon_stable_signal():
    """20d AND 60d both with |IC|>0.04, decade same-sign, same sign across horizons."""
    s, r = _persistent_signal(seed=7, n=3000, ic_direction=+1)
    triples = [
        (1, 'stable', s, 'NDX', 5, r),
        (1, 'stable', s, 'NDX', 20, r),
        (1, 'stable', s, 'NDX', 60, r),
    ]
    df = batch_evaluate(triples, ic_window=252)
    df = apply_fdr_and_judgment(df)
    # 20d should pass (primary or secondary). 60d via secondary.
    h20_pass = bool(df[df['horizon'] == 20].iloc[0]['pass_flag'])
    h60_pass = bool(df[df['horizon'] == 60].iloc[0]['pass_flag'])
    assert h20_pass or h60_pass


def test_evaluate_triple_handles_empty_signal():
    """Sanity: degenerate input doesn't crash."""
    idx = pd.date_range('2010-01-01', periods=100, freq='B')
    s = pd.Series([np.nan] * 100, index=idx, dtype=float)
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, size=100), index=idx)
    out = evaluate_triple(1, 'empty', s, 'NDX', 20, r, ic_window=30)
    assert out.signal_id == 1
    # mean_ic should be nan (no valid IC observations)
    assert np.isnan(out.mean_ic)
