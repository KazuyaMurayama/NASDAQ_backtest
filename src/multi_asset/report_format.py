"""Markdown metric tables that bold the recommended row and the best value
per column, so the reader can see at a glance what matters.
"""
from __future__ import annotations

import math
from typing import Optional


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


def _best_index(rows, key, better) -> Optional[int]:
    vals = [(i, r.get(key)) for i, r in enumerate(rows) if _is_num(r.get(key))]
    if not vals:
        return None
    if better == 'max':
        return max(vals, key=lambda t: t[1])[0]
    if better == 'min':
        return min(vals, key=lambda t: t[1])[0]
    return None


def fmt_metric_table(rows, cols, name_key: str, recommended=None) -> str:
    """Render a Markdown table.

    cols: list of {key, label, fmt(callable), better in {'max','min',None}}.
    The best cell per 'better' column is bolded; the recommended row's name
    cell is bolded.
    """
    best = {c['key']: _best_index(rows, c['key'], c.get('better'))
            for c in cols if c.get('better')}

    header = '| ' + ' | '.join(c['label'] for c in cols) + ' |'
    sep = '|' + '|'.join('---' for _ in cols) + '|'
    out = [header, sep]
    for i, r in enumerate(rows):
        cells = []
        for c in cols:
            key = c['key']
            v = r.get(key)
            if key == name_key:
                txt = str(v)
                if recommended is not None and v == recommended:
                    txt = f'**{txt}**'
            else:
                fmt = c.get('fmt', str)
                txt = fmt(v) if _is_num(v) else (str(v) if v is not None else '—')
                if best.get(key) == i and _is_num(v):
                    txt = f'**{txt}**'
            cells.append(txt)
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)
