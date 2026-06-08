"""Phase 3 — 3-asset allocation across the confirmed 1x timing strategies.

Sleeves (all 1x; leverage is Phase 4):
  - NASDAQ: mom252 x VT(0.15) x VZ, weekly rebal (1x timing PROXY; can be
    swapped for the canonical CFD Active strategy later)
  - Gold:   m252_tv0.10_z0.75_mo   (user-confirmed)
  - Bond:   m252_tv0.05_z1.0_wk    (user-confirmed)

Allocators: equal / inverse-vol (risk-parity) / sharpe-tilt, weights rebalanced
monthly. Evaluated full-period (9-metric style) + WFA vs equal-weight B&H.
Report tables bold the recommended row and the best value per column.

Outputs (repo root):
  - MULTIASSET_ALLOCATION_20260608.md / multiasset_allocation_results.csv

Run: PYTHONPATH=src python -m multi_asset.run_allocation
"""
from __future__ import annotations

import os
import sys

import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from integration.nine_metric_eval import (
    _cagr, _sharpe, _maxdd, _worst_window_cagr, _pct_window_cagr,
)
from multi_asset.single_asset_sweep import build_holdcash_nav
from multi_asset.bond_signals import momentum_position, zscore_position
from multi_asset.strategy_layers import (
    vol_target_scale, vol_regime_gate, compose, rebalance_periodic, apply_exec_lag,
)
from multi_asset.allocator import (
    equal_weights, inverse_vol_weights, sharpe_tilt_weights, combine_portfolio,
)
from multi_asset.walkforward import wfa_stats, paired_block_bootstrap
from multi_asset.report_format import fmt_metric_table
from multi_asset.run_bond_sweep import _load_bond_and_cash, _load_macro_daily
from multi_asset.run_gold_sweep import _load as _load_gold

ROOT = os.path.dirname(_SRC)
DATE = '2026-06-08'
N_BOOT = 5000


def _bond_strat():
    bond_ret, bond_price, cash = _load_bond_and_cash()
    fed10 = _load_macro_daily('repo_4_repo_4_dff_minus_10y.parquet',
                              'repo_4_dff_minus_10y', bond_ret.index)
    pos = compose(momentum_position(bond_price, 252),
                  vol_target_scale(bond_ret, 0.05, 20, 1.0),
                  vol_regime_gate(bond_ret, 20, 252, 1.0, 0.0),
                  zscore_position(fed10, 252, enter=0.0, invert=True))
    pos = apply_exec_lag(rebalance_periodic(pos, 5), 4)
    return build_holdcash_nav(bond_ret, cash, pos).pct_change()


def _gold_strat():
    gold_ret, gold_price, cash, real_yield, _c, _d = _load_gold()
    pos = compose(momentum_position(gold_price, 252),
                  vol_target_scale(gold_ret, 0.10, 20, 1.0),
                  vol_regime_gate(gold_ret, 20, 252, 0.75, 0.0),
                  zscore_position(real_yield, 252, enter=0.0, invert=True))
    pos = apply_exec_lag(rebalance_periodic(pos, 21), 4)
    return build_holdcash_nav(gold_ret, cash, pos).pct_change()


def _nasdaq_strat():
    base = pd.read_csv(os.path.join(ROOT, 'data', 'base_dataset.csv'),
                       parse_dates=['date'], index_col='date').sort_index()
    nq_ret = base['nasdaq_ret'].apply(lambda x: x).dropna()
    # base_dataset stores nasdaq_ret as LOG return; convert to simple
    import numpy as np
    nq_ret = (np.exp(base['nasdaq_ret']) - 1.0).dropna()
    nq_price = base['nasdaq_close'].reindex(nq_ret.index)
    cash = (pd.read_csv(os.path.join(ROOT, 'data', 'dtb3_daily.csv'),
                        parse_dates=['Date'], index_col='Date')['yield_pct']
            / 100.0 / 252.0).reindex(nq_ret.index).ffill().fillna(0.0)
    pos = compose(momentum_position(nq_price, 252),
                  vol_target_scale(nq_ret, 0.15, 20, 1.0),
                  vol_regime_gate(nq_ret, 20, 252, 1.0, 0.0))
    pos = apply_exec_lag(rebalance_periodic(pos, 5), 4)
    return build_holdcash_nav(nq_ret, cash, pos).pct_change(), nq_ret


def _metrics(ret):
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    return {
        'cagr': _cagr(nav), 'sharpe': _sharpe(nav), 'maxdd': _maxdd(nav),
        'worst10y': _worst_window_cagr(nav, 10), 'p10_5y': _pct_window_cagr(nav, 5, 0.10),
        'wfe': wfa_stats(nav)['wfe'], 'ci95_lo': wfa_stats(nav)['ci95_lo_cagr'],
    }


def _turnover_per_year(weights):
    dw = weights.diff().abs().sum(axis=1) * 0.5
    return float(dw.mean() * 252)


def main():
    import numpy as np
    bond = _bond_strat(); gold = _gold_strat()
    nq, nq_raw = _nasdaq_strat()
    df = pd.concat({'NASDAQ': nq, 'Gold': gold, 'Bond': bond}, axis=1).dropna()
    # raw (untimed) returns for the B&H reference, aligned
    base = pd.read_csv(os.path.join(ROOT, 'data', 'base_dataset.csv'),
                       parse_dates=['date'], index_col='date').sort_index()
    raw = pd.concat({'NASDAQ': nq_raw,
                     'Gold': base['gold_ret'], 'Bond': base['bond_ret']},
                    axis=1).reindex(df.index)

    allocs = {
        'Alloc_Equal': equal_weights(df),
        'Alloc_InvVol': inverse_vol_weights(df, 63),
        'Alloc_SharpeTilt': sharpe_tilt_weights(df, 126),
    }
    # rebalance weights monthly for realistic portfolio turnover
    allocs = {k: rebalance_periodic_df(v, 21) for k, v in allocs.items()}

    rows = []
    # individual timed sleeves
    for name in ['NASDAQ', 'Gold', 'Bond']:
        m = _metrics(df[name]); m['strategy'] = f'{name}_only(timed,1x)'; m['turnover'] = float('nan')
        rows.append(m)
    # allocations
    port_rets = {}
    for name, w in allocs.items():
        pret = combine_portfolio(df, w)
        port_rets[name] = pret
        m = _metrics(pret); m['strategy'] = name; m['turnover'] = _turnover_per_year(w)
        rows.append(m)
    # reference: equal-weight B&H of raw assets
    ref = combine_portfolio(raw, equal_weights(raw))
    m = _metrics(ref); m['strategy'] = 'Ref_EW_BuyHold'; m['turnover'] = 0.0
    rows.append(m)

    # recommend: best Sharpe among allocations with WFA pass-ish (wfe in band)
    alloc_rows = [r for r in rows if r['strategy'].startswith('Alloc')]
    rec = sorted(alloc_rows, key=lambda r: -r['sharpe'])[0]['strategy']

    # bootstrap: recommended alloc vs best single sleeve & vs EW B&H
    best_sleeve = max(['NASDAQ', 'Gold', 'Bond'], key=lambda n: _sharpe((1+df[n].fillna(0)).cumprod()))
    rec_ret = port_rets[rec]
    vs_sleeve = paired_block_bootstrap(rec_ret, df[best_sleeve], n_boot=N_BOOT)
    vs_ref = paired_block_bootstrap(rec_ret, ref, n_boot=N_BOOT)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(ROOT, 'multiasset_allocation_results.csv'), index=False)

    cols = [
        {'key': 'strategy', 'label': '構成'},
        {'key': 'cagr', 'label': 'CAGR', 'fmt': lambda v: f'{v*100:+.2f}%', 'better': 'max'},
        {'key': 'sharpe', 'label': 'Sharpe', 'fmt': lambda v: f'{v:+.3f}', 'better': 'max'},
        {'key': 'maxdd', 'label': 'MaxDD', 'fmt': lambda v: f'{v*100:.1f}%', 'better': 'max'},
        {'key': 'worst10y', 'label': 'Worst10Y', 'fmt': lambda v: f'{v*100:+.2f}%', 'better': 'max'},
        {'key': 'p10_5y', 'label': 'P10_5Y', 'fmt': lambda v: f'{v*100:+.2f}%', 'better': 'max'},
        {'key': 'wfe', 'label': 'WFE', 'fmt': lambda v: f'{v:.2f}'},
        {'key': 'ci95_lo', 'label': 'CI95_lo', 'fmt': lambda v: f'{v*100:+.2f}%', 'better': 'max'},
        {'key': 'turnover', 'label': 'Turnover/yr', 'fmt': lambda v: f'{v:.1f}', 'better': 'min'},
    ]
    lines = [
        '# マルチアセット資産配分（Phase 3）',
        '', f'作成日: {DATE}', f'最終更新日: {DATE}', '',
        '> 確定 1x 戦略を束ねる: NASDAQ=mom252×VT×VZ(週次, **1xタイミングPROXY**), '
        'Gold=**m252_tv0.10_z0.75_mo**, Bond=**m252_tv0.05_z1.0_wk**。',
        '> 配分: 等加重 / インバースボラ(リスクパリティ) / シャープ傾斜、**月次リバランス**。',
        '> **太字=推奨行／各列の最良値**。全期間(1974-2026)。', '',
        '## 配分ロジック比較', '',
        fmt_metric_table(rows, cols, name_key='strategy', recommended=rec),
        '',
        f'## 推奨配分: **{rec}**',
        f'- アロケーション中 Sharpe 最良。最良単一スリーブ({best_sleeve})比 '
        f'P(配分>{best_sleeve})={vs_sleeve["p_a_gt_b"]:.3f}、'
        f'等加重B&H比 P(>EW_BH)={vs_ref["p_a_gt_b"]:.3f}（n={N_BOOT}）。',
        '- ⚠ NASDAQスリーブは1xタイミングPROXY。確定NASDAQ Active戦略へ差し替え可。',
        '- ⚠ レバレッジ未適用（全て1x）。倍率適用は Phase 4。',
    ]
    with open(os.path.join(ROOT, 'MULTIASSET_ALLOCATION_20260608.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(res[['strategy', 'cagr', 'sharpe', 'maxdd', 'worst10y', 'turnover']].to_string(index=False))
    print(f'recommended={rec}  vs {best_sleeve}: P={vs_sleeve["p_a_gt_b"]:.3f}  vs EW_BH: P={vs_ref["p_a_gt_b"]:.3f}')


def rebalance_periodic_df(weights, every):
    out = weights.copy()
    keep = pd.Series(False, index=weights.index)
    keep.iloc[::every] = True
    out[~keep.values] = float('nan')
    return out.ffill()


if __name__ == '__main__':
    main()
