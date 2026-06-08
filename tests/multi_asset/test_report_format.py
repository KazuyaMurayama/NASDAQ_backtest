import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from multi_asset.report_format import fmt_metric_table


def test_bolds_best_per_column_and_recommended_row():
    rows = [
        {'name': 'A', 'sharpe': 1.0, 'maxdd': -0.30, 'trades': 50},
        {'name': 'B', 'sharpe': 1.5, 'maxdd': -0.10, 'trades': 20},
        {'name': 'C', 'sharpe': 1.2, 'maxdd': -0.20, 'trades': 12},
    ]
    cols = [
        {'key': 'name', 'label': '構成'},
        {'key': 'sharpe', 'label': 'Sharpe', 'fmt': lambda v: f'{v:.2f}', 'better': 'max'},
        {'key': 'maxdd', 'label': 'MaxDD', 'fmt': lambda v: f'{v*100:.1f}%', 'better': 'max'},
        {'key': 'trades', 'label': 'Trades', 'fmt': lambda v: f'{v:.0f}', 'better': 'min'},
    ]
    md = fmt_metric_table(rows, cols, name_key='name', recommended='C')
    lines = md.splitlines()
    # best sharpe (B=1.5) bolded
    assert '**1.50**' in md
    # best maxdd (least negative = -0.10 = B) bolded
    assert '**-10.0%**' in md
    # fewest trades (C=12) bolded
    assert '**12**' in md
    # recommended row name C bolded
    assert '**C**' in md
    # non-best values not bolded
    assert '| 1.00 |' in md or ' 1.00 ' in md


def test_handles_nan_gracefully():
    rows = [{'name': 'A', 'x': float('nan')}, {'name': 'B', 'x': 2.0}]
    cols = [{'key': 'name', 'label': 'n'},
            {'key': 'x', 'label': 'X', 'fmt': lambda v: f'{v:.1f}', 'better': 'max'}]
    md = fmt_metric_table(rows, cols, name_key='name', recommended='B')
    assert '**2.0**' in md
