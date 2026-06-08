"""Phase 2.4 — Formal WFA + block bootstrap on promising single-asset candidates.

Replaces the NAV-proxy screening values (WFE/CI95 from nine_metric_eval) with
proper walk-forward (calendar-year windows) and stationary block bootstrap
(block=60, paired vs cash and vs B&H), per house Phase D conventions.

Candidates (from Phase 2.2/2.3 full-period results):
  - bond_mom252       (12m momentum on bonds)
  - gold_realyield_lo (hold gold when real-yield z <= 0)
  - gold_mom126       (6m momentum on gold)

Outputs (repo root):
  - PHASE_D_VALIDATION_20260608.md
  - phase_d_validation_results.csv

Run: PYTHONPATH=src python -m multi_asset.run_phase_d
"""
from __future__ import annotations

import os
import sys

import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from multi_asset.single_asset_sweep import build_holdcash_nav, buy_and_hold_nav
from multi_asset.bond_signals import momentum_position, zscore_position
from multi_asset.walkforward import (
    wfa_stats, block_bootstrap, paired_block_bootstrap, N_BOOTSTRAP, BLOCK_SIZE,
)
from multi_asset.run_bond_sweep import _load_bond_and_cash
from multi_asset.run_gold_sweep import _load as _load_gold

ROOT = os.path.dirname(_SRC)
DATE = '2026-06-08'
DATESTAMP = '20260608'
N_BOOT = 5000  # below house 10000 for runtime; rerun at 10000 to finalize


def _candidates():
    bond_ret, bond_price, bond_cash = _load_bond_and_cash()
    gold_ret, gold_price, gold_cash, real_yield, _cpi, _dxy = _load_gold()
    return [
        ('bond_mom252', bond_ret, bond_cash,
         momentum_position(bond_price, 252)),
        ('gold_realyield_lo', gold_ret, gold_cash,
         zscore_position(real_yield, 252, enter=0.0, invert=True)),
        ('gold_mom126', gold_ret, gold_cash,
         momentum_position(gold_price, 126)),
    ]


def main():
    rows = []
    for name, asset_ret, cash_ret, pos in _candidates():
        cand_nav = build_holdcash_nav(asset_ret, cash_ret, pos)
        cand_ret = cand_nav.pct_change().dropna()
        bh_ret = buy_and_hold_nav(asset_ret).pct_change().dropna()
        cash_aligned = cash_ret.reindex(cand_ret.index).fillna(0.0)

        wfa = wfa_stats(cand_nav)
        boot = block_bootstrap(cand_ret, n_boot=N_BOOT, block=BLOCK_SIZE)
        vs_cash = paired_block_bootstrap(cand_ret, cash_aligned, n_boot=N_BOOT)
        vs_bh = paired_block_bootstrap(cand_ret, bh_ret, n_boot=N_BOOT)

        # PASS: WFA gates AND bootstrap beats cash with >90% probability
        passed = wfa['passed'] and vs_cash['p_a_gt_b'] > 0.90
        rows.append({
            'candidate': name,
            'wfa_n_windows': wfa['n_windows'],
            'wfa_wfe': round(wfa['wfe'], 3),
            'wfa_ci95_lo_cagr': round(wfa['ci95_lo_cagr'], 4),
            'wfa_pct_pos_windows': round(wfa['pct_pos_windows'], 3),
            'wfa_pass': wfa['passed'],
            'boot_cagr_median': round(boot['cagr_median'], 4),
            'boot_cagr_ci': f"[{boot['cagr_lo']:.4f}, {boot['cagr_hi']:.4f}]",
            'boot_p_cagr_pos': round(boot['p_pos'], 3),
            'p_beat_cash': round(vs_cash['p_a_gt_b'], 3),
            'p_beat_bh': round(vs_bh['p_a_gt_b'], 3),
            'PASS': passed,
        })
        print(f"{name}: WFE={wfa['wfe']:.2f} CI95lo={wfa['ci95_lo_cagr']:+.4f} "
              f"P(>cash)={vs_cash['p_a_gt_b']:.3f} P(>BH)={vs_bh['p_a_gt_b']:.3f} "
              f"PASS={passed}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(ROOT, 'phase_d_validation_results.csv')
    df.to_csv(csv_path, index=False)

    lines = [
        '# Phase 2.4 — Bond/Gold 候補の正式WFA・ブートストラップ検証',
        '', f'作成日: {DATE}', f'最終更新日: {DATE}', '',
        '> `MULTIASSET_GOLD_BOND_SIGNAL_PLAN_20260608.md` Phase 2.4。',
        f'> WFA=暦年窓・t分布CI95・WFE。Bootstrap=定常ブロック(block={BLOCK_SIZE}日, '
        f'n={N_BOOT}, 対キャッシュ/対B&Hはペア化)。',
        '> 進格ゲート: **WFA CI95_lo>0 かつ 0.5≤WFE≤2.0**、かつ **P(対キャッシュ>0)>0.90**。',
        f'> ※ n_boot={N_BOOT}（house標準10000未満。確定時は10000で再実行）。', '',
        '| 候補 | WFA窓 | WFE | CI95_lo(CAGR) | 勝率窓 | WFA可 | Boot CAGR中央 | Boot95%CI | P(CAGR>0) | P(>cash) | P(>B&H) | 総合PASS |',
        '|---|---:|---:|---:|---:|:--:|---:|---|---:|---:|---:|:--:|',
    ]
    for r in rows:
        lines.append(
            f"| {r['candidate']} | {r['wfa_n_windows']} | {r['wfa_wfe']} | "
            f"{r['wfa_ci95_lo_cagr']:+.4f} | {r['wfa_pct_pos_windows']} | "
            f"{'✅' if r['wfa_pass'] else '❌'} | {r['boot_cagr_median']:+.4f} | "
            f"{r['boot_cagr_ci']} | {r['boot_p_cagr_pos']} | {r['p_beat_cash']} | "
            f"{r['p_beat_bh']} | {'✅' if r['PASS'] else '❌'} |")
    lines += [
        '', '## 注記',
        '- WFE はNAV代理(50等分)ではなく**暦年窓ベースの正式値**に置換済み。',
        '- 「P(>cash)」が hold/cash 判断の本質的ゲート（B&Hでなくキャッシュ超を要求）。',
        '- ⚠ DXY系は2006+のため本検証では実質金利・モメンタムを対象に選定。',
    ]
    with open(os.path.join(ROOT, f'PHASE_D_VALIDATION_{DATESTAMP}.md'),
              'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', csv_path)


if __name__ == '__main__':
    main()
