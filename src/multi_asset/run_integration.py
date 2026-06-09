"""Phase 6 — final 3-asset portfolio integration.

Combines the confirmed per-asset products (net-of-cost-after-tax) into the
final portfolio across representative risk configurations, so the user can
pick NASDAQ leverage x allocation method from one comparison.

Fixed: Gold = 1x fund (T+5), Bond = TMF@1x (T+2).
Varies: NASDAQ = TQQQ@{1,2,3}x (T+2); allocation = Equal / InverseVol (monthly).

Outputs (repo root):
  - MULTIASSET_INTEGRATED_20260608.md / multiasset_integrated_results.csv

Run: PYTHONPATH=src python -m multi_asset.run_integration
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import product_costs as pc
from integration.nine_metric_eval import _cagr, _sharpe, _maxdd, _worst_window_cagr
from multi_asset.leverage_eval import strategy_net_returns, after_tax_cagr
from multi_asset.allocator import equal_weights, inverse_vol_weights, combine_portfolio
from multi_asset.walkforward import wfa_stats, paired_block_bootstrap
from multi_asset.report_format import fmt_metric_table, validate_markdown_tables
from multi_asset.run_leverage_decision import _assets, _sofr_annual

ROOT = os.path.dirname(_SRC)
DATE = '2026-06-08'
N_BOOT = 5000

CONFIGS = [
    ('保守 (NQ@1x × InvVol)',    1.0, 'invvol'),
    ('やや保守 (NQ@1x × Equal)', 1.0, 'equal'),
    ('中庸 (NQ@2x × Equal)',     2.0, 'equal'),
    ('やや攻め (NQ@2x × InvVol)', 2.0, 'invvol'),
    ('攻め (NQ@3x × Equal)',      3.0, 'equal'),
]


def _rebal_df(weights, every=21):
    out = weights.copy()
    keep = pd.Series(False, index=weights.index)
    keep.iloc[::every] = True
    out[~keep.values] = np.nan
    return out.ffill()


def _asset_net_returns():
    a = _assets()
    nq_ret, nq_pos, nq_cash, _ = a['NASDAQ']
    g_ret, g_pos, g_cash, _ = a['Gold']
    b_ret, b_pos, b_cash, _ = a['Bond']
    sofr_nq = _sofr_annual(nq_ret.dropna().index)
    sofr_g = _sofr_annual(g_ret.dropna().index)
    sofr_b = _sofr_annual(b_ret.dropna().index)

    nq = {k: strategy_net_returns(nq_ret, nq_pos, nq_cash, sofr_nq, k, pc.TQQQ,
                                  exec_lag=pc.EXEC_LAG_ETF) for k in (1.0, 2.0, 3.0)}
    gold = strategy_net_returns(g_ret, g_pos, g_cash, sofr_g, 1.0, pc.GOLD1X,
                                exec_lag=pc.EXEC_LAG_FUND_1X)
    bond = strategy_net_returns(b_ret, b_pos, b_cash, sofr_b, 1.0, pc.TMF,
                                exec_lag=pc.EXEC_LAG_ETF)
    raw = {'NASDAQ': nq_ret, 'Gold': g_ret, 'Bond': b_ret}
    return nq, gold, bond, raw


def _metrics(ret):
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    at = after_tax_cagr(ret)
    dd = _maxdd(nav)
    return {
        'cagr_at': at, 'sharpe': _sharpe(nav), 'maxdd': dd,
        'worst10y': _worst_window_cagr(nav, 10),
        'calmar': (at / abs(dd) if dd and not np.isnan(dd) else float('nan')),
        'wfe': wfa_stats(nav)['wfe'],
    }


def _portfolio(nq_k, gold, bond, method):
    df = pd.concat({'NASDAQ': nq_k, 'Gold': gold, 'Bond': bond}, axis=1).dropna()
    w = equal_weights(df) if method == 'equal' else inverse_vol_weights(df, 63)
    w = _rebal_df(w, 21)
    return combine_portfolio(df, w), df


def main():
    nq, gold, bond, raw = _asset_net_returns()
    rows, port_rets = [], {}
    for label, k, method in CONFIGS:
        port, df = _portfolio(nq[k], gold, bond, method)
        port_rets[label] = port
        m = _metrics(port); m['config'] = label
        rows.append(m)
    # reference: equal-weight buy&hold (untimed, 1x), aligned to a portfolio index
    ref_idx = port_rets[CONFIGS[0][0]].index
    ref_df = pd.concat(raw, axis=1).reindex(ref_idx)
    ref = combine_portfolio(ref_df, equal_weights(ref_df))
    m = _metrics(ref); m['config'] = '参考: 無タイミング等加重B&H'
    rows.append(m)

    rec = sorted([r for r in rows if r['config'] != rows[-1]['config']],
                 key=lambda r: -r['calmar'])[0]['config']
    rec_vs_ref = paired_block_bootstrap(port_rets[rec], ref, n_boot=N_BOOT)

    pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'multiasset_integrated_results.csv'), index=False)
    cols = [
        {'key': 'config', 'label': '構成(NASDAQ倍率×配分)'},
        {'key': 'cagr_at', 'label': 'CAGR(税後)', 'fmt': lambda v: f'{v*100:+.2f}%', 'better': 'max'},
        {'key': 'sharpe', 'label': 'Sharpe', 'fmt': lambda v: f'{v:+.3f}', 'better': 'max'},
        {'key': 'maxdd', 'label': 'MaxDD', 'fmt': lambda v: f'{v*100:.1f}%', 'better': 'max'},
        {'key': 'worst10y', 'label': 'Worst10Y', 'fmt': lambda v: f'{v*100:+.2f}%', 'better': 'max'},
        {'key': 'calmar', 'label': 'Calmar(税後CAGR÷MaxDD)', 'fmt': lambda v: f'{v:.3f}', 'better': 'max'},
        {'key': 'wfe', 'label': 'WFE', 'fmt': lambda v: f'{v:.2f}'},
    ]
    lines = [
        '# マルチアセット 最終ポートフォリオ統合（Phase 6）',
        '', f'作成日: {DATE}', f'最終更新日: {DATE}', '',
        '> 確定スリーブを束ねた最終比較。**Gold=1倍投信(T+5)／Bond=TMF@1x(T+2) は固定**、'
        '**NASDAQ倍率(TQQQ@1/2/3x)×配分(等加重/InvVol)** を可変。',
        '> 全期間(1974-2026)・純コスト後・**税20.315%後**・配分は月次リバランス。'
        '**太字=推奨行／各列最良値**。推奨=Calmar最大。', '',
        '## 構成比較', '',
        fmt_metric_table(rows, cols, name_key='config', recommended=rec),
        '',
        f'## 推奨(リスク調整): **{rec}**',
        f'- Calmar(税後CAGR÷|MaxDD|)最大。無タイミング等加重B&H比 '
        f'P(>B&H)={rec_vs_ref["p_a_gt_b"]:.3f}（n={N_BOOT}）。',
        '- ★ **NASDAQ倍率と配分はリスク選好で選択**：上表で攻め(NQ@3x)はCAGR↑/MaxDD↑、保守(InvVol)はSharpe↑/DD↓。',
        '- 構成商品: NASDAQ=TQQQ(実効倍率) / Gold=SBI純金1倍投信 / Bond=TMF@1x。'
        'シグナル: NASDAQ=mom252×VT×VZ / Gold=m252_tv0.10_z0.75_mo / Bond=m252_tv0.05_z1.0_wk。',
    ]
    doc = '\n'.join(lines) + '\n'
    validate_markdown_tables(doc)
    with open(os.path.join(ROOT, 'MULTIASSET_INTEGRATED_20260608.md'), 'w', encoding='utf-8') as f:
        f.write(doc)

    for r in rows:
        print(f"{r['config']}: CAGR_at={r['cagr_at']*100:+.2f}% Sharpe={r['sharpe']:.2f} "
              f"MaxDD={r['maxdd']*100:.0f}% Calmar={r['calmar']:.3f} WFE={r['wfe']:.2f}")
    print('recommended=', rec, 'P(>B&H)=', round(rec_vs_ref['p_a_gt_b'], 3))
    return rec, rows


if __name__ == '__main__':
    main()
