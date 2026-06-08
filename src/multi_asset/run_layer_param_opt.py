"""Phase 2.6 — Layer parameter optimization (realistic execution) for Gold/Bond.

Structure: base(momentum) x VT(vol-target) x VZ(vol-regime) x MACRO, then
REBALANCE periodically (weekly/monthly) and apply a ~5-business-day execution
lag, so trade frequency is realistic. Reports the REAL trades/yr (actual
position changes, not the NAV proxy). Tables bold the recommended row and the
best value per column. Top stacks validated with WFA + paired bootstrap vs cash.

Outputs (repo root):
  - BOND_LAYER_PARAM_OPT_20260608.md / bond_layer_param_opt_results.csv
  - GOLD_LAYER_PARAM_OPT_20260608.md / gold_layer_param_opt_results.csv

Run: PYTHONPATH=src python -m multi_asset.run_layer_param_opt
"""
from __future__ import annotations

import os
import sys

import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from multi_asset.single_asset_sweep import (
    run_single_asset_sweep, build_holdcash_nav, buy_and_hold_nav, CANONICAL_SPLIT,
)
from multi_asset.bond_signals import momentum_position, zscore_position
from multi_asset.strategy_layers import (
    vol_target_scale, vol_regime_gate, compose, rebalance_periodic, apply_exec_lag,
)
from multi_asset.walkforward import wfa_stats, paired_block_bootstrap
from multi_asset.report_format import fmt_metric_table
from multi_asset.run_bond_sweep import _load_bond_and_cash, _load_macro_daily
from multi_asset.run_gold_sweep import _load as _load_gold

ROOT = os.path.dirname(_SRC)
DATE = '2026-06-08'
N_BOOT = 5000

MOMS = [126, 252]
ZTHRESH = [0.75, 1.0]
MACRO_ENTER = [0.0]
REBAL = [5, 21]        # weekly / monthly rebalance
EXEC_LAG = 4           # + signal's internal 1-day shift ≈ 5 business days
REBAL_LABEL = {5: 'wk', 21: 'mo'}


def _build_grid(price, asset_ret, macro_raw, macro_invert, target_vols):
    strat = {}
    for m in MOMS:
        base = momentum_position(price, m)
        for tv in target_vols:
            VT = vol_target_scale(asset_ret, target_vol=tv, vol_window=20, max_leverage=1.0)
            for zt in ZTHRESH:
                VZ = vol_regime_gate(asset_ret, vol_window=20, z_window=252,
                                     z_thresh=zt, gate_min=0.0)
                for me in MACRO_ENTER:
                    MA = zscore_position(macro_raw, 252, enter=me, invert=macro_invert)
                    for rb in REBAL:
                        stack = compose(base, VT, VZ, MA)
                        stack = rebalance_periodic(stack, every=rb)
                        stack = apply_exec_lag(stack, EXEC_LAG)
                        name = f"m{m}_tv{tv}_z{zt}_{REBAL_LABEL[rb]}"
                        strat[name] = stack
    return strat


def _validate(df, asset_ret, cash_ret, lookup, top_n=6):
    cand = df[~df['signal'].isin(['BUY_AND_HOLD', 'ALL_CASH'])].copy()
    cand = cand.sort_values('cand_sharpe_full', ascending=False).head(top_n)
    bh_ret = buy_and_hold_nav(asset_ret).pct_change().dropna()
    rows = []
    for _, c in cand.iterrows():
        name = c['signal']
        nav = build_holdcash_nav(asset_ret, cash_ret, lookup[name])
        cret = nav.pct_change().dropna()
        cash_al = cash_ret.reindex(cret.index).fillna(0.0)
        wfa = wfa_stats(nav)
        vc = paired_block_bootstrap(cret, cash_al, n_boot=N_BOOT)
        vb = paired_block_bootstrap(cret, bh_ret, n_boot=N_BOOT)
        rows.append({
            'name': name, 'wfe': round(wfa['wfe'], 3),
            'ci95_lo': wfa['ci95_lo_cagr'], 'p_cash': vc['p_a_gt_b'],
            'p_bh': vb['p_a_gt_b'], 'trades': c['cand_trades_yr_real'],
            'PASS': bool(wfa['passed'] and vc['p_a_gt_b'] > 0.90),
        })
    return rows


def _recommend(val_rows):
    passed = [r for r in val_rows if r['PASS']]
    pool = passed if passed else val_rows
    # prefer fewest trades, then highest P(>cash)
    key = (lambda r: (r['trades'], -r['p_cash'])) if passed else (lambda r: -r['p_cash'])
    return sorted(pool, key=key)[0]['name']


def _run(asset, price, asset_ret, cash_ret, macro_raw, macro_invert, target_vols):
    strat = _build_grid(price, asset_ret, macro_raw, macro_invert, target_vols)
    strat['BUY_AND_HOLD'] = pd.Series(1.0, index=price.index)
    strat['ALL_CASH'] = pd.Series(0.0, index=price.index)
    df = run_single_asset_sweep(asset_ret, cash_ret, strat,
                                split_date=CANONICAL_SPLIT, baseline='bh',
                                sort_by='cand_sharpe_full')
    df.to_csv(os.path.join(ROOT, f'{asset.lower()}_layer_param_opt_results.csv'), index=False)
    val_rows = _validate(df, asset_ret, cash_ret, strat, top_n=6)
    rec = _recommend(val_rows)

    top = df[~df['signal'].isin(['BUY_AND_HOLD', 'ALL_CASH'])].head(10)
    top_rows = top.to_dict('records')
    top_cols = [
        {'key': 'signal', 'label': '構成'},
        {'key': 'cand_cagr_full', 'label': 'CAGR_full', 'fmt': lambda v: f'{v*100:+.2f}%', 'better': 'max'},
        {'key': 'cand_sharpe_full', 'label': 'Sharpe_full', 'fmt': lambda v: f'{v:+.3f}', 'better': 'max'},
        {'key': 'cand_maxdd', 'label': 'MaxDD', 'fmt': lambda v: f'{v*100:.1f}%', 'better': 'max'},
        {'key': 'cand_worst10y', 'label': 'Worst10Y', 'fmt': lambda v: f'{v*100:+.2f}%', 'better': 'max'},
        {'key': 'cand_is_oos_gap', 'label': 'IS-OOSgap', 'fmt': lambda v: f'{v*100:+.2f}pp', 'better': 'min'},
        {'key': 'cand_trades_yr_real', 'label': 'Trades/yr', 'fmt': lambda v: f'{v:.0f}', 'better': 'min'},
    ]
    val_cols = [
        {'key': 'name', 'label': '構成'},
        {'key': 'wfe', 'label': 'WFE', 'fmt': lambda v: f'{v:.3f}'},
        {'key': 'ci95_lo', 'label': 'CI95_lo', 'fmt': lambda v: f'{v*100:+.2f}%', 'better': 'max'},
        {'key': 'p_cash', 'label': 'P(>cash)', 'fmt': lambda v: f'{v:.3f}', 'better': 'max'},
        {'key': 'p_bh', 'label': 'P(>B&H)', 'fmt': lambda v: f'{v:.3f}', 'better': 'max'},
        {'key': 'trades', 'label': 'Trades/yr', 'fmt': lambda v: f'{v:.0f}', 'better': 'min'},
        {'key': 'PASS', 'label': 'PASS', 'fmt': lambda v: '✅' if v else '❌'},
    ]
    lines = [
        f'# {asset} レイヤー・パラメータ最適化（Phase 2.6, 現実的実行）',
        '', f'作成日: {DATE}', f'最終更新日: {DATE}', '',
        f'> base=momentum×VT(vol-target)×VZ(vol-regime)×MACRO、'
        f'**定期リバランス(週次wk/月次mo)＋実行ラグ≈5営業日**を適用。',
        f'> グリッド: mom={MOMS}, tv={target_vols}, z_thresh={ZTHRESH}, '
        f'rebalance={list(REBAL_LABEL.values())}（{len(strat)-2}構成）。',
        '> **Trades/yr は実建玉変更回数**（NAV代理ではない）。**太字=推奨行／各列の最良値**。', '',
        '## 上位構成（全期間, Sharpe_full降順）', '',
        fmt_metric_table(top_rows, top_cols, name_key='signal', recommended=rec),
        '', f'## 上位構成の正式検証（WFA + bootstrap vs cash, n={N_BOOT}）', '',
        fmt_metric_table(val_rows, val_cols, name_key='name', recommended=rec),
        '', f'## 推奨構成: **{rec}**',
        '- 検証 PASS の中で **Trades/yr 最小**（次点 P(>cash) 最大）を推奨。PASS無しなら P(>cash) 最大。',
    ]
    with open(os.path.join(ROOT, f'{asset.upper()}_LAYER_PARAM_OPT_20260608.md'),
              'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'=== {asset} top (sharpe) ===')
    print(top[['signal', 'cand_cagr_full', 'cand_sharpe_full', 'cand_maxdd',
               'cand_trades_yr_real', 'cand_is_oos_gap']].head(6).to_string(index=False))
    print(f'--- {asset} validation --- recommended={rec}')
    for r in val_rows:
        print(f"  {r['name']}: trades={r['trades']:.0f} P(>cash)={r['p_cash']:.2f} "
              f"WFE={r['wfe']:.2f} PASS={r['PASS']}")
    return df, val_rows, rec


def main():
    bond_ret, bond_price, bond_cash = _load_bond_and_cash()
    fed10 = _load_macro_daily('repo_4_repo_4_dff_minus_10y.parquet',
                              'repo_4_dff_minus_10y', bond_ret.index)
    _run('Bond', bond_price, bond_ret, bond_cash, fed10, True, [0.05, 0.07])

    gold_ret, gold_price, gold_cash, real_yield, _cpi, _dxy = _load_gold()
    _run('Gold', gold_price, gold_ret, gold_cash, real_yield, True, [0.10, 0.13])


if __name__ == '__main__':
    main()
