import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import pytest
from multi_asset.report_format import fmt_metric_table, validate_markdown_tables


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


def test_escapes_pipe_in_labels_and_cells():
    # a literal '|' in a label or value must be escaped so it doesn't break
    # the markdown table structure
    rows = [{'name': 'A|B', 'x': 1.0}]
    cols = [{'key': 'name', 'label': 'n|m'},
            {'key': 'x', 'label': 'Calmar(税後/|DD|)', 'fmt': lambda v: f'{v:.1f}', 'better': 'max'}]
    md = fmt_metric_table(rows, cols, name_key='name', recommended='A|B')
    header = md.splitlines()[0]
    # every row must have the same number of UNescaped column separators
    def n_seps(line):
        return len(line.replace('\\|', '').split('|')) - 1
    counts = {n_seps(l) for l in md.splitlines()}
    assert len(counts) == 1            # header, sep, and data rows all align
    assert '\\|' in md                 # the literal pipes got escaped


def test_validator_accepts_good_table_rejects_broken():
    good = "| a | b |\n|---|---|\n| 1 | 2 |\n\n| x | y | z |\n|---|---|---|\n| 1 | 2 | 3 |"
    assert validate_markdown_tables(good) is True
    broken = "| a | b |\n|---|---|\n| 1 | 2 | 3 |"   # data row has extra column
    with pytest.raises(ValueError):
        validate_markdown_tables(broken)


def test_fmt_metric_table_output_is_always_valid():
    # even with pipe-laden labels, the emitted table must self-validate
    rows = [{'name': 'A', 'x': 1.0}, {'name': 'B', 'x': 2.0}]
    cols = [{'key': 'name', 'label': 'n|m'},
            {'key': 'x', 'label': 'Calmar(後/|DD|)', 'fmt': lambda v: f'{v:.1f}', 'better': 'max'}]
    md = fmt_metric_table(rows, cols, name_key='name', recommended='B')
    assert validate_markdown_tables(md) is True


def test_handles_nan_gracefully():
    rows = [{'name': 'A', 'x': float('nan')}, {'name': 'B', 'x': 2.0}]
    cols = [{'key': 'name', 'label': 'n'},
            {'key': 'x', 'label': 'X', 'fmt': lambda v: f'{v:.1f}', 'better': 'max'}]
    md = fmt_metric_table(rows, cols, name_key='name', recommended='B')
    assert '**2.0**' in md
