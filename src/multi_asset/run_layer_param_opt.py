"""Phase 2.6 — Layer parameter optimization for the Gold/Bond layered stacks.

Sweeps the key layer parameters (momentum lookback, vol-target level, VZ
z-threshold, macro-gate threshold, deadband eps) over the winning
base+VT+VZ+MACRO(+DB) structure, evaluates full-period 9-metrics, watches
IS-OOS gap for overfitting, and validates the top stacks with WFA + paired
block bootstrap vs cash. Mirrors NASDAQ's A1-A6/B6 parameter sweeps.

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
from multi_asset.strategy_layers import vol_target_scale, vol_regime_gate, deadband, compose
from multi_asset.walkforward import wfa_stats, paired_block_bootstrap
from multi_asset.run_bond_sweep import _load_bond_and_cash, _load_macro_daily
from multi_asset.run_gold_sweep import _load as _load_gold

ROOT = os.path.dirname(_SRC)
DATE = '2026-06-08'
N_BOOT = 5000

MOMS = [126, 252]
ZTHRESH = [0.75, 1.0, 1.5]
MACRO_ENTER = [0.0, 0.5]
DEADBANDS = [0.0, 0.10]


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
                    for eps in DEADBANDS:
                        stack = compose(base, VT, VZ, MA)
                        if eps > 0:
                            stack = deadband(stack, eps)
                        name = f"m{m}_tv{tv}_z{zt}_me{me}_db{eps}"
                        strat[name] = stack
    return strat


def _validate(df, asset_ret, cash_ret, lookup, top_n=6):
    """WFA + paired bootstrap vs cash/B&H for top stacks by Sharpe_full,
    preferring low IS-OOS gap on ties."""
    cand = df[~df['signal'].isin(['BUY_AND_HOLD', 'ALL_CASH'])].copy()
    cand = cand.sort_values('cand_sharpe_full', ascending=False).head(top_n)
    bh_ret = buy_and_hold_nav(asset_ret).pct_change().dropna()
    rows = []
    for name in cand['signal']:
        nav = build_holdcash_nav(asset_ret, cash_ret, lookup[name])
        cret = nav.pct_change().dropna()
        cash_al = cash_ret.reindex(cret.index).fillna(0.0)
        wfa = wfa_stats(nav)
        vc = paired_block_bootstrap(cret, cash_al, n_boot=N_BOOT)
        vb = paired_block_bootstrap(cret, bh_ret, n_boot=N_BOOT)
        rows.append({
            'signal': name, 'wfe': round(wfa['wfe'], 3),
            'ci95_lo': round(wfa['ci95_lo_cagr'], 4),
            'p_beat_cash': round(vc['p_a_gt_b'], 3),
            'p_beat_bh': round(vb['p_a_gt_b'], 3),
            'PASS': bool(wfa['passed'] and vc['p_a_gt_b'] > 0.90),
        })
    return pd.DataFrame(rows)


def _run(asset, price, asset_ret, cash_ret, macro_raw, macro_invert, target_vols):
    strat = _build_grid(price, asset_ret, macro_raw, macro_invert, target_vols)
    strat['BUY_AND_HOLD'] = pd.Series(1.0, index=price.index)
    strat['ALL_CASH'] = pd.Series(0.0, index=price.index)
    df = run_single_asset_sweep(asset_ret, cash_ret, strat,
                                split_date=CANONICAL_SPLIT, baseline='bh',
                                sort_by='cand_sharpe_full')
    df.to_csv(os.path.join(ROOT, f'{asset.lower()}_layer_param_opt_results.csv'), index=False)
    val = _validate(df, asset_ret, cash_ret, strat, top_n=6)

    top = df[~df['signal'].isin(['BUY_AND_HOLD', 'ALL_CASH'])].head(12)
    lines = [
        f'# {asset} レイヤー・パラメータ最適化（Phase 2.6）',
        '', f'作成日: {DATE}', f'最終更新日: {DATE}', '',
        f'> base=momentum×VT(vol-target)×VZ(vol-regime)×MACRO×DB(deadband) の'
        f'パラメータ総当たり（{len(strat)-2} 構成）。Sharpe_full 降順。',
        f'> グリッド: mom={MOMS}, tv={target_vols}, z_thresh={ZTHRESH}, '
        f'macro_enter={MACRO_ENTER}, deadband={DEADBANDS}。', '',
        '## 上位12構成（全期間）', '',
        '| 構成 | CAGR_full | Sharpe_full | MaxDD | Worst10Y | IS-OOSgap | Trades/yr |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {r['signal']} | {r['cand_cagr_full']*100:+.2f}% | "
            f"{r['cand_sharpe_full']:+.3f} | {r['cand_maxdd']*100:.1f}% | "
            f"{r['cand_worst10y']*100:+.2f}% | {r['cand_is_oos_gap']*100:+.2f}pp | "
            f"{r['cand_trades_yr']:.0f} |")
    lines += ['', '## 上位構成の正式検証（WFA + bootstrap vs cash, n=%d）' % N_BOOT, '',
              '| 構成 | WFE | CI95_lo | P(>cash) | P(>B&H) | PASS |',
              '|---|---:|---:|---:|---:|:--:|']
    for _, r in val.iterrows():
        lines.append(
            f"| {r['signal']} | {r['wfe']} | {r['ci95_lo']:+.4f} | "
            f"{r['p_beat_cash']} | {r['p_beat_bh']} | {'✅' if r['PASS'] else '❌'} |")
    # recommended = best PASS by Sharpe_full; else best by P(>cash)
    passed = val[val['PASS']]
    rec = passed.iloc[0]['signal'] if len(passed) else (
        val.sort_values('p_beat_cash', ascending=False).iloc[0]['signal'])
    lines += ['', f'## 推奨構成: **{rec}**',
              '- 上表で PASS かつ Sharpe_full 上位の構成。PASS無しの場合は P(>cash) 最大を暫定推奨。']
    with open(os.path.join(ROOT, f'{asset.upper()}_LAYER_PARAM_OPT_20260608.md'),
              'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'=== {asset} top6 ===')
    print(top.head(6)[['signal', 'cand_cagr_full', 'cand_sharpe_full',
                       'cand_maxdd', 'cand_is_oos_gap']].to_string(index=False))
    print(f'--- {asset} validation ---'); print(val.to_string(index=False))
    print(f'>>> {asset} recommended: {rec}')
    return df, val, rec


def main():
    bond_ret, bond_price, bond_cash = _load_bond_and_cash()
    fed10 = _load_macro_daily('repo_4_repo_4_dff_minus_10y.parquet',
                              'repo_4_dff_minus_10y', bond_ret.index)
    _run('Bond', bond_price, bond_ret, bond_cash, fed10, True, [0.05, 0.07, 0.10])

    gold_ret, gold_price, gold_cash, real_yield, _cpi, _dxy = _load_gold()
    _run('Gold', gold_price, gold_ret, gold_cash, real_yield, True, [0.10, 0.13, 0.16])


if __name__ == '__main__':
    main()
