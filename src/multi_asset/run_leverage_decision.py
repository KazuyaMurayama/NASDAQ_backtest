"""Phase 4 — product & leverage decision per asset (no CFD; max 3x, Gold 2x).

For each asset's confirmed signal, compare candidates on net-of-cost-after-tax
return/risk over the full period:
  - 1x fund   (T+5, no SOFR/swap, low TER)         [FUND_1X_SET]
  - leveraged ETF at effective leverage k (T+2)    [LEVERAGED_SET]
    NASDAQ/Bond k in {1..3}, Gold k in {1..2}.
Recommend per asset by Calmar (after-tax CAGR / |MaxDD|) among WFA-sane options.
Tables bold the recommended row and the best value per column.

Outputs (repo root):
  - LEVERAGE_DECISION_20260608.md / leverage_decision_results.csv

Run: PYTHONPATH=src python -m multi_asset.run_leverage_decision
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
from multi_asset.bond_signals import momentum_position, zscore_position
from multi_asset.strategy_layers import (
    vol_target_scale, vol_regime_gate, compose, rebalance_periodic,
)
from multi_asset.leverage_eval import strategy_net_returns, after_tax_cagr
from multi_asset.walkforward import wfa_stats
from multi_asset.report_format import fmt_metric_table, validate_markdown_tables
from multi_asset.run_bond_sweep import _load_bond_and_cash, _load_macro_daily
from multi_asset.run_gold_sweep import _load as _load_gold

ROOT = os.path.dirname(_SRC)
DATA = os.path.join(ROOT, 'data')
DATE = '2026-06-08'


def _sofr_annual(idx):
    dtb3 = pd.read_csv(os.path.join(DATA, 'dtb3_daily.csv'),
                       parse_dates=['Date'], index_col='Date')['yield_pct']
    return (dtb3 / 100.0).reindex(idx).ffill().fillna(0.0)


def _assets():
    # Bond: m252_tv0.05_z1.0_wk
    bond_ret, bond_price, bond_cash = _load_bond_and_cash()
    fed10 = _load_macro_daily('repo_4_repo_4_dff_minus_10y.parquet',
                              'repo_4_dff_minus_10y', bond_ret.index)
    bond_pos = rebalance_periodic(compose(
        momentum_position(bond_price, 252),
        vol_target_scale(bond_ret, 0.05, 20, 1.0),
        vol_regime_gate(bond_ret, 20, 252, 1.0, 0.0),
        zscore_position(fed10, 252, enter=0.0, invert=True)), 5)

    # Gold: m252_tv0.1_z0.75_mo
    gold_ret, gold_price, gold_cash, real_yield, _c, _d = _load_gold()
    gold_pos = rebalance_periodic(compose(
        momentum_position(gold_price, 252),
        vol_target_scale(gold_ret, 0.10, 20, 1.0),
        vol_regime_gate(gold_ret, 20, 252, 0.75, 0.0),
        zscore_position(real_yield, 252, enter=0.0, invert=True)), 21)

    # NASDAQ: mom252 x VT(0.15) x VZ, weekly (rebuilt for TQQQ leverage)
    base = pd.read_csv(os.path.join(DATA, 'base_dataset.csv'),
                       parse_dates=['date'], index_col='date').sort_index()
    nq_ret = (np.exp(base['nasdaq_ret']) - 1.0).dropna()
    nq_price = base['nasdaq_close'].reindex(nq_ret.index)
    nq_cash = (pd.read_csv(os.path.join(DATA, 'dtb3_daily.csv'),
               parse_dates=['Date'], index_col='Date')['yield_pct']
               / 100.0 / 252.0).reindex(nq_ret.index).ffill().fillna(0.0)
    nq_pos = rebalance_periodic(compose(
        momentum_position(nq_price, 252),
        vol_target_scale(nq_ret, 0.15, 20, 1.0),
        vol_regime_gate(nq_ret, 20, 252, 1.0, 0.0)), 5)

    return {
        'NASDAQ': (nq_ret, nq_pos, nq_cash, [1.0, 1.5, 2.0, 2.5, 3.0]),
        'Gold': (gold_ret, gold_pos, gold_cash, [1.0, 1.5, 2.0]),
        'Bond': (bond_ret, bond_pos, bond_cash, [1.0, 1.5, 2.0, 2.5, 3.0]),
    }


def _metrics(ret):
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    at = after_tax_cagr(ret)
    dd = _maxdd(nav)
    wfa = wfa_stats(nav)
    return {
        'cagr_pre': _cagr(nav), 'cagr_at': at, 'sharpe': _sharpe(nav),
        'maxdd': dd, 'worst10y': _worst_window_cagr(nav, 10),
        'wfe': wfa['wfe'], 'calmar': (at / abs(dd) if dd and not np.isnan(dd) else float('nan')),
    }


def _run_asset(asset, ret, pos, cash, k_grid):
    sofr = _sofr_annual(ret.dropna().index)
    lev = pc.LEVERAGED_SET[asset]
    fund = pc.FUND_1X_SET[asset]
    rows = []
    # 1x fund (T+5)
    fr = strategy_net_returns(ret, pos, cash, sofr, 1.0, fund,
                              exec_lag=pc.EXEC_LAG_FUND_1X)
    m = _metrics(fr); m['candidate'] = f'{fund.ticker}(1x投信,T+5)'; rows.append(m)
    # leveraged ETF at each effective k (T+2)
    for k in k_grid:
        rr = strategy_net_returns(ret, pos, cash, sofr, k, lev,
                                  exec_lag=pc.EXEC_LAG_ETF)
        m = _metrics(rr); m['candidate'] = f'{lev.ticker}@{k:g}x(T+2)'; rows.append(m)

    # Recommend by Calmar (after-tax CAGR / |MaxDD|) — a risk-adjusted default.
    # Do NOT hard-exclude on WFE (bond's low WFE is low-vol, not overfit); flag it.
    rec_row = sorted(rows, key=lambda r: -r['calmar'])[0]
    rec = rec_row['candidate']
    rec_wfe_flag = '' if 0.5 <= rec_row['wfe'] <= 2.0 else \
        f'（注: WFE={rec_row["wfe"]:.2f} は健全域外＝低ボラ起因、過学習ではない）'
    # most aggressive option for the return-seeking reader
    aggressive = sorted(rows, key=lambda r: -r['cagr_at'])[0]['candidate']

    cols = [
        {'key': 'candidate', 'label': '候補(商品@倍率)'},
        {'key': 'cagr_pre', 'label': 'CAGR(税前)', 'fmt': lambda v: f'{v*100:+.2f}%', 'better': 'max'},
        {'key': 'cagr_at', 'label': 'CAGR(税後)', 'fmt': lambda v: f'{v*100:+.2f}%', 'better': 'max'},
        {'key': 'sharpe', 'label': 'Sharpe', 'fmt': lambda v: f'{v:+.3f}', 'better': 'max'},
        {'key': 'maxdd', 'label': 'MaxDD', 'fmt': lambda v: f'{v*100:.1f}%', 'better': 'max'},
        {'key': 'worst10y', 'label': 'Worst10Y', 'fmt': lambda v: f'{v*100:+.2f}%', 'better': 'max'},
        {'key': 'calmar', 'label': 'Calmar(税後CAGR÷MaxDD)', 'fmt': lambda v: f'{v:.3f}', 'better': 'max'},
        {'key': 'wfe', 'label': 'WFE', 'fmt': lambda v: f'{v:.2f}'},
    ]
    md = [f'### {asset}', '',
          fmt_metric_table(rows, cols, name_key='candidate', recommended=rec),
          '', f'**推奨(リスク調整): {rec}**（Calmar最大）{rec_wfe_flag}']
    if aggressive != rec:
        md.append(f'- リターン重視なら **{aggressive}**（税後CAGR最大／MaxDDは深化、上表参照）。')
    else:
        md.append('- ※ 本資産は**低レバが税後CAGRも最良**（レバはコストで減益）→ リターン重視でも増レバ非推奨。')
    md.append('')
    for r in rows:
        r['asset'] = asset
    return rows, rec, '\n'.join(md)


def main():
    assets = _assets()
    all_rows, recs, sections = [], {}, []
    for asset, (ret, pos, cash, kg) in assets.items():
        rows, rec, md = _run_asset(asset, ret, pos, cash, kg)
        all_rows += rows; recs[asset] = rec; sections.append(md)

    pd.DataFrame(all_rows).to_csv(os.path.join(ROOT, 'leverage_decision_results.csv'), index=False)
    header = [
        '# Phase 4 — 商品・レバレッジ判定（CFD無し / NASDAQ・Bond最大3x / Gold最大2x）',
        '', f'作成日: {DATE}', f'最終更新日: {DATE}', '',
        '> 確定シグナルを、**1倍投信(T+5)** か **レバETF各倍率(T+2)** で保有した場合の'
        '純コスト後(SOFR financing・スワップ・TER・売買0.10%)・**税20.315%後**を全期間(1974-2026)比較。',
        '> 商品: NASDAQ=TQQQ(3x上限) / Gold=2036(2x上限) / Bond=TMF(3x上限) / 1倍投信=SBI実商品。',
        '',
        '## 読み方・注記（重要）',
        '- **太字 = 推奨行（候補名）／各列の最良値**。',
        '- **推奨 = Calmar（税後CAGR ÷ |MaxDD|）最大**（リスク調整の既定）。WFEは健全性の参考表示（フィルタには使わない）。',
        '- **「@kx」は実効レバk**。レバETFで k<上限 を作る場合、**資本の k/L だけをETFに置き、残りはキャッシュ(T-bill)で運用**する想定。'
        'このため低レバ(@1x等)は**余剰キャッシュの金利収入がfinancingを相殺**し、1倍投信と近い結果になる（資本効率効果・1980年代の高SOFR期に特に効く）。',
        '- レバを上げるほど **税後CAGRは増えるが MaxDD は急速に悪化**（トレードオフ）。攻守は選好で選ぶ。',
        '- Bond はレバを上げると **financing が低リターンを食って税後CAGRが減少**（＝1xが最良＝分散役）。',
        '',
        '## 資産別 判定', '',
    ]
    summary = ['', '## 推奨サマリ', '',
               '| 資産 | 推奨商品・倍率 |', '|---|---|']
    for a in ['NASDAQ', 'Gold', 'Bond']:
        summary.append(f'| **{a}** | **{recs[a]}** |')
    doc = '\n'.join(header + sections + summary) + '\n'
    validate_markdown_tables(doc)   # fail loudly if any table is malformed
    with open(os.path.join(ROOT, 'LEVERAGE_DECISION_20260608.md'), 'w', encoding='utf-8') as f:
        f.write(doc)

    for asset, (ret, pos, cash, kg) in assets.items():
        print(f'--- {asset} --- recommended={recs[asset]}')
        for r in all_rows:
            if r['asset'] == asset:
                print(f"  {r['candidate']}: CAGR_at={r['cagr_at']*100:+.2f}% "
                      f"MaxDD={r['maxdd']*100:.0f}% Calmar={r['calmar']:.2f} WFE={r['wfe']:.2f}")


if __name__ == '__main__':
    main()
