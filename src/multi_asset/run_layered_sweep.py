"""Phase 2.5 — Layered (1-5 layer) timing-strategy sweep for Bond & Gold.

Ports the NASDAQ archetypes (docs/multiasset/NASDAQ_STRATEGY_ARCHETYPES.md):
base directional signal x vol-targeting x VZ vol-regime gate x macro/regime
gate x turnover smoothing (deadband). Evaluates full-period 9-metrics, then
validates the top multi-layer stacks with WFA + paired block bootstrap vs cash.

Outputs (repo root):
  - BOND_LAYERED_SWEEP_20260608.md / bond_layered_sweep_results.csv
  - GOLD_LAYERED_SWEEP_20260608.md / gold_layered_sweep_results.csv

Run: PYTHONPATH=src python -m multi_asset.run_layered_sweep
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
from multi_asset.bond_signals import ma_cross_position, momentum_position, zscore_position
from multi_asset.strategy_layers import (
    vol_target_scale, vol_regime_gate, dual_ma_position,
    donchian_breakout_position, deadband, compose,
)
from multi_asset.sweep_report import render_sweep_md
from multi_asset.walkforward import wfa_stats, paired_block_bootstrap
from multi_asset.run_bond_sweep import _load_bond_and_cash, build_signals as _bond_signals
from multi_asset.run_gold_sweep import _load as _load_gold

ROOT = os.path.dirname(_SRC)
DATE = '2026-06-08'
N_BOOT = 3000


def build_layered(price, asset_ret, macro_gate, target_vol, base_key):
    """Return {name: position} for 1-5 layer stacks plus baselines."""
    bases = {
        'mom252': momentum_position(price, 252),
        'mom126': momentum_position(price, 126),
        'ma200': ma_cross_position(price, 200),
        'dualma50_200': dual_ma_position(price, 50, 200),
        'donch100_50': donchian_breakout_position(price, 100, 50),
    }
    VT = vol_target_scale(asset_ret, target_vol=target_vol, vol_window=20, max_leverage=1.0)
    VZ = vol_regime_gate(asset_ret, vol_window=20, z_window=252, z_thresh=1.0, gate_min=0.0)
    MA = macro_gate
    base = bases[base_key]

    strat = {
        'BUY_AND_HOLD': pd.Series(1.0, index=price.index),
        'ALL_CASH': pd.Series(0.0, index=price.index),
    }
    # 1 layer
    for k, v in bases.items():
        strat[f'1L:{k}'] = v
    strat['1L:VT'] = VT
    strat['1L:VZ'] = VZ
    strat['1L:MACRO'] = MA
    # 2 layers
    strat[f'2L:{base_key}+VT'] = compose(base, VT)
    strat[f'2L:{base_key}+VZ'] = compose(base, VZ)
    strat[f'2L:{base_key}+MACRO'] = compose(base, MA)
    strat['2L:VT+VZ'] = compose(VT, VZ)
    strat['2L:VT+MACRO'] = compose(VT, MA)
    strat['2L:VZ+MACRO'] = compose(VZ, MA)
    # 3 layers
    strat[f'3L:{base_key}+VT+VZ'] = compose(base, VT, VZ)
    strat[f'3L:{base_key}+VT+MACRO'] = compose(base, VT, MA)
    strat[f'3L:{base_key}+VZ+MACRO'] = compose(base, VZ, MA)
    strat['3L:VT+VZ+MACRO'] = compose(VT, VZ, MA)
    # 4 layers
    strat[f'4L:{base_key}+VT+VZ+MACRO'] = compose(base, VT, VZ, MA)
    # 5 layers (+ deadband turnover control)
    strat[f'5L:{base_key}+VT+VZ+MACRO+DB'] = deadband(compose(base, VT, VZ, MA), eps=0.10)
    return strat


def _validate_top(df, asset_ret, cash_ret, builder_lookup, top_n=4):
    """WFA + paired bootstrap vs cash for top multi-layer stacks by Sharpe_full."""
    cand = df[df['signal'].str.startswith(('2L', '3L', '4L', '5L'))]
    cand = cand.sort_values('cand_sharpe_full', ascending=False).head(top_n)
    bh_ret = buy_and_hold_nav(asset_ret).pct_change().dropna()
    rows = []
    for name in cand['signal']:
        pos = builder_lookup[name]
        nav = build_holdcash_nav(asset_ret, cash_ret, pos)
        cret = nav.pct_change().dropna()
        cash_al = cash_ret.reindex(cret.index).fillna(0.0)
        wfa = wfa_stats(nav)
        vc = paired_block_bootstrap(cret, cash_al, n_boot=N_BOOT)
        vb = paired_block_bootstrap(cret, bh_ret, n_boot=N_BOOT)
        passed = wfa['passed'] and vc['p_a_gt_b'] > 0.90
        rows.append({'signal': name, 'wfe': round(wfa['wfe'], 3),
                     'ci95_lo': round(wfa['ci95_lo_cagr'], 4),
                     'p_beat_cash': round(vc['p_a_gt_b'], 3),
                     'p_beat_bh': round(vb['p_a_gt_b'], 3),
                     'PASS': passed})
    return pd.DataFrame(rows)


def _hyps(strat):
    return {k: k for k in strat}


def _run_asset(asset, price, asset_ret, cash_ret, macro_gate, target_vol, base_key):
    strat = build_layered(price, asset_ret, macro_gate, target_vol, base_key)
    df = run_single_asset_sweep(asset_ret, cash_ret, strat,
                                split_date=CANONICAL_SPLIT, baseline='bh')
    df.to_csv(os.path.join(ROOT, f'{asset.lower()}_layered_sweep_results.csv'), index=False)

    val = _validate_top(df, asset_ret, cash_ret, strat, top_n=4)
    note = (f'多層合成（base={base_key}, vol_target={target_vol}）。'
            f'VT=ボラターゲ, VZ=ボラレジームgate, MACRO=資産別マクロgate, DB=デッドバンド。')
    md = render_sweep_md(f'{asset}(多層)', df, CANONICAL_SPLIT, DATE, note, _hyps(strat))
    md += '\n## 上位多層スタックの正式検証（WFA + bootstrap vs cash, n=%d）\n\n' % N_BOOT
    md += '| stack | WFE | CI95_lo | P(>cash) | P(>B&H) | PASS |\n|---|---:|---:|---:|---:|:--:|\n'
    for _, r in val.iterrows():
        md += (f"| {r['signal']} | {r['wfe']} | {r['ci95_lo']:+.4f} | "
               f"{r['p_beat_cash']} | {r['p_beat_bh']} | {'✅' if r['PASS'] else '❌'} |\n")
    with open(os.path.join(ROOT, f'{asset.upper()}_LAYERED_SWEEP_20260608.md'),
              'w', encoding='utf-8') as f:
        f.write(md)
    print(f'\n=== {asset} top by CAGR_full ===')
    print(df[['signal', 'cand_cagr_full', 'cand_sharpe_full', 'cand_maxdd',
              'cand_worst10y']].head(8).to_string(index=False))
    print(f'--- {asset} validation ---')
    print(val.to_string(index=False))
    return df, val


def main():
    # Bond
    bond_ret, bond_price, bond_cash = _load_bond_and_cash()
    bsig = _bond_signals(bond_price, bond_ret.index)
    _run_asset('Bond', bond_price, bond_ret, bond_cash,
               macro_gate=bsig['fed_easy'], target_vol=0.07, base_key='mom252')
    # Gold
    gold_ret, gold_price, gold_cash, real_yield, cpi_yoy, dxy = _load_gold()
    gold_macro = zscore_position(real_yield, 252, enter=0.0, invert=True)
    _run_asset('Gold', gold_price, gold_ret, gold_cash,
               macro_gate=gold_macro, target_vol=0.12, base_key='mom126')


if __name__ == '__main__':
    main()
