import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from signals.adoption import pareto_judge, hard_requirements_check


def test_pareto_pass_two_improvements():
    base = {'cagr': 0.10, 'sharpe': 0.5, 'maxdd': -0.30, 'trades_yr': 50}
    cand = {'cagr': 0.13, 'sharpe': 0.7, 'maxdd': -0.28, 'trades_yr': 60}
    out = pareto_judge(cand, base)
    assert out['pareto_pass'] is True
    assert 'cagr' in out['improved_axes']
    assert 'sharpe' in out['improved_axes']


def test_pareto_fail_one_degradation():
    base = {'cagr': 0.10, 'sharpe': 0.5, 'maxdd': -0.30, 'trades_yr': 50}
    cand = {'cagr': 0.13, 'sharpe': 0.7, 'maxdd': -0.45, 'trades_yr': 60}  # MaxDD worse by 15pp
    out = pareto_judge(cand, base)
    assert out['pareto_pass'] is False
    assert 'maxdd' in out['degraded_axes']


def test_pareto_fail_trades_cap_exceeded():
    base = {'cagr': 0.10, 'sharpe': 0.5, 'maxdd': -0.30, 'trades_yr': 50}
    cand = {'cagr': 0.13, 'sharpe': 0.7, 'maxdd': -0.28, 'trades_yr': 300}  # excess trades
    out = pareto_judge(cand, base)
    assert out['pareto_pass'] is False


def test_hard_requirements_pass():
    g = {
        'G3': {'ci95_lo': 0.1, 'wfe': 1.1},
        'G7': {'p_cand_gt_bench': 0.95},
        'G9': {'p_value': 0.05},
    }
    out = hard_requirements_check(g)
    assert out['pass'] is True


def test_hard_requirements_fail_g3():
    g = {
        'G3': {'ci95_lo': -0.1, 'wfe': 1.1},
        'G7': {'p_cand_gt_bench': 0.95},
        'G9': {'p_value': 0.05},
    }
    out = hard_requirements_check(g)
    assert out['pass'] is False
    assert any('G3' in f for f in out['failures'])
