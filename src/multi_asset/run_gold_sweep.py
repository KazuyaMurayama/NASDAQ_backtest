"""Phase 2.3 — Gold standalone hold-vs-cash signal sweep (real data).

Reuses the same tested building blocks as the Bond sweep
(single_asset_sweep, bond_signals causal builders, sweep_report).

Data:
  - data/base_dataset.csv     → gold_ret (LBMA, 1974+ on NYSE calendar)
  - data/dtb3_daily.csv       → cash (3M T-bill) leg
  - data/dgs10_daily.csv + data/cpiaucsl_monthly.csv → real yield, CPI YoY
  - data/dxy_daily.csv        → dollar index (2006+; pre-2006 → cash)

Outputs (repo root, house naming):
  - GOLD_SINGLE_ASSET_SWEEP_20260608.md
  - gold_single_asset_sweep_results.csv

Run: PYTHONPATH=src python -m multi_asset.run_gold_sweep
"""
from __future__ import annotations

import os
import sys

import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from multi_asset.single_asset_sweep import run_single_asset_sweep, CANONICAL_SPLIT
from multi_asset.bond_signals import (
    ma_cross_position, momentum_position, zscore_position,
)
from multi_asset.sweep_report import render_sweep_md

ROOT = os.path.dirname(_SRC)
DATA = os.path.join(ROOT, 'data')
DATE = '2026-06-08'
DATESTAMP = '20260608'


def _reindex(s, cal):
    s = s.sort_index()
    return s.reindex(cal.union(s.index)).ffill().reindex(cal)


def _load():
    base = pd.read_csv(os.path.join(DATA, 'base_dataset.csv'),
                       parse_dates=['date'], index_col='date').sort_index()
    gold_ret = base['gold_ret'].dropna()
    gold_price = (1.0 + gold_ret).cumprod()
    cal = gold_ret.index

    dtb3 = pd.read_csv(os.path.join(DATA, 'dtb3_daily.csv'),
                       parse_dates=['Date'], index_col='Date')['yield_pct']
    cash_ret = (dtb3 / 100.0 / 252.0).reindex(cal).ffill().fillna(0.0)

    dgs10 = pd.read_csv(os.path.join(DATA, 'dgs10_daily.csv'),
                        parse_dates=['Date'], index_col='Date')['yield_pct']
    cpi = pd.read_csv(os.path.join(DATA, 'cpiaucsl_monthly.csv'),
                      parse_dates=['DATE'], index_col='DATE')['CPIAUCSL']
    cpi_yoy = (cpi / cpi.shift(12) - 1.0) * 100.0  # % YoY

    dgs10_d = _reindex(dgs10, cal)
    cpi_yoy_d = _reindex(cpi_yoy, cal)
    real_yield = dgs10_d - cpi_yoy_d  # nominal 10y minus trailing inflation
    dxy = _reindex(pd.read_csv(os.path.join(DATA, 'dxy_daily.csv'),
                               parse_dates=['Date'], index_col='Date')['dxy'], cal)
    return gold_ret, gold_price, cash_ret, real_yield, cpi_yoy_d, dxy


_HYPOTHESIS = {
    'BUY_AND_HOLD': 'baseline: always invested',
    'ALL_CASH': 'baseline: always in T-bill cash',
    'gold_ma200': 'trend: hold when price > 200d MA',
    'gold_ma100': 'trend: hold when price > 100d MA',
    'gold_mom252': 'momentum: hold when 12m return > 0',
    'gold_mom126': 'momentum: hold when 6m return > 0',
    'gold_realyield_lo': 'hold when real 10y yield z <= 0 (low real yield favors gold)',
    'gold_dxy_lo': 'hold when DXY z <= 0 (weak dollar favors gold; DXY 2006+)',
    'gold_infl_hi': 'hold when CPI YoY z >= 0.5 (accelerating inflation)',
}


def main():
    gold_ret, gold_price, cash_ret, real_yield, cpi_yoy, dxy = _load()
    signals = {
        'BUY_AND_HOLD':  pd.Series(1.0, index=gold_price.index),
        'ALL_CASH':      pd.Series(0.0, index=gold_price.index),
        'gold_ma200':    ma_cross_position(gold_price, 200),
        'gold_ma100':    ma_cross_position(gold_price, 100),
        'gold_mom252':   momentum_position(gold_price, 252),
        'gold_mom126':   momentum_position(gold_price, 126),
        'gold_realyield_lo': zscore_position(real_yield, 252, enter=0.0, invert=True),
        'gold_dxy_lo':   zscore_position(dxy, 252, enter=0.0, invert=True),
        'gold_infl_hi':  zscore_position(cpi_yoy, 252, enter=0.5),
    }
    df = run_single_asset_sweep(gold_ret, cash_ret, signals,
                                split_date=CANONICAL_SPLIT, baseline='bh')
    df['hypothesis'] = df['signal'].map(_HYPOTHESIS)

    csv_path = os.path.join(ROOT, 'gold_single_asset_sweep_results.csv')
    df.to_csv(csv_path, index=False)

    data_note = ('Goldリターン: `base_dataset.csv`（LBMA 1974+）、キャッシュ: DTB3、'
                 '実質金利: DGS10−CPI YoY、DXY: 2006+（以前はキャッシュ扱い）。')
    md = render_sweep_md('Gold', df, CANONICAL_SPLIT, DATE, data_note, _HYPOTHESIS)
    with open(os.path.join(ROOT, f'GOLD_SINGLE_ASSET_SWEEP_{DATESTAMP}.md'),
              'w', encoding='utf-8') as f:
        f.write(md)

    print('wrote', csv_path)
    print(df[['signal', 'cand_cagr_full', 'cand_sharpe_full', 'cand_cagr_oos',
              'cand_maxdd', 'cand_worst10y', 'judgment']].to_string(index=False))


if __name__ == '__main__':
    main()
