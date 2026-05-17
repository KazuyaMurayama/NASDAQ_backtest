"""
H1〜H5 仮説バックテスト (2026-05-17)
======================================
GOLD_BOND_STRATEGY_PLAN_2026-05-17.md に基づき、Gold/Bondスリーブの
代替商品を組み込んだ5仮説をバックテストする。

H1: TQQQ 3x × Gold CFD 3x × TMF 3x (対称3x強化)
H2: S2 CFD × Gold CFD 5x × TMF 3x (高レバ・対称)
H3: S2 CFD × Gold動的(S2_Gold) × TMF 3x (二重S2)
H4: TQQQ 3x × TOCOM 3x × TMF軽減 (低コスト・Gold傾斜)
H5: TQQQ 3x × Gold ハイブリッド(1540+CFD) × TMF 3x (実効3x)

ベースライン:
  BL_A: TQQQ 3x × Gold 2x × Bond 3x (現行ベスト DH Dyn [A])
  BL_S2: S2 CFD × Gold 2x × Bond 3x (S2 DH統合版)

出力:
  - コンソール: 7メトリクス比較表
  - ファイル: H1_H5_SUMMARY_2026-05-17.md
"""

import sys, os, types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr, build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_a2_signal, simulate_rebalance_A,
    TRADING_DAYS, THRESHOLD,
)
from test_portfolio_diversification import prepare_gold_data
from cfd_leverage_backtest import (
    build_nav_strategy, calc_7metrics, CFD_SPREAD_LOW,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from sleeves_extended import (
    build_gold_cfd, build_gold_tocom, build_gold_s2_dynamic,
    build_gold_hybrid, build_bond_3x_with_drag,
)

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

S2_PARAMS = dict(target_vol=0.80, k_vz=0.30, gate_min=0.50)


def _fp(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v*100:+.{d}f}%'

def _ff(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:.{d}f}'


def run_scenario(name, close, lev_A, wn, wg, wb, dates,
                 gold_nav, bond_nav, sofr, nas_mode, cfd_lev):
    nav = build_nav_strategy(
        close, lev_A, wn, wg, wb, dates,
        gold_nav, bond_nav, sofr, nas_mode, cfd_lev, CFD_SPREAD_LOW,
    )
    m = calc_7metrics(nav, dates)
    m['name'] = name
    return m


def make_h4_weights(wn_A):
    rest = 1.0 - wn_A
    return wn_A, rest * 0.65, rest * 0.35


def make_h2_weights(wn_A):
    wn_h2 = np.clip(wn_A, 0.30, 0.80)
    rest = 1.0 - wn_h2
    return wn_h2, rest * 0.5, rest * 0.5


def generate_report(rows, baseline_a, baseline_s2, dates_info, leverages) -> str:
    lines = []
    lines.append('# H1〜H5 Gold/Bond仮説 バックテスト結果')
    lines.append('')
    lines.append('作成日: 2026-05-17')
    lines.append('最終更新日: 2026-05-17')
    lines.append('')
    lines.append(f'データ期間: {dates_info["start"]} 〜 {dates_info["end"]}')
    lines.append('')
    lines.append('参照: [GOLD_BOND_STRATEGY_PLAN_2026-05-17.md](GOLD_BOND_STRATEGY_PLAN_2026-05-17.md)')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## ベースライン')
    lines.append('')
    lines.append('| シナリオ | 構造 | CAGR_FULL | CAGR_OOS | Sharpe_OOS | MaxDD | Worst5Y | IS-OOS Gap |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for m, label, struct in [
        (baseline_a, 'BL_A (現行)', 'TQQQ 3x × Gold 2x × Bond 3x'),
        (baseline_s2, 'BL_S2', 'S2 CFD × Gold 2x × Bond 3x'),
    ]:
        gap = abs(m.get('CAGR_IS', float('nan')) - m.get('CAGR_OOS', float('nan'))) * 100
        lines.append(
            f'| {label} | {struct} | {_fp(m.get("CAGR_FULL"))} | {_fp(m.get("CAGR_OOS"))}'
            f' | {_ff(m.get("Sharpe_OOS"))} | {_fp(m.get("MaxDD_FULL"))}'
            f' | {_fp(m.get("Worst5Y"))} | {gap:.1f}pp |'
        )
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 仮説結果')
    lines.append('')
    lines.append('| 仮説 | 構造 | CAGR_FULL | CAGR_OOS | Sharpe_OOS | MaxDD | Worst5Y | IS-OOS Gap | vs BL_A | vs BL_S2 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        m = r['metrics']
        gap = abs(m.get('CAGR_IS', float('nan')) - m.get('CAGR_OOS', float('nan'))) * 100
        sh = m.get('Sharpe_OOS', float('nan'))
        w5 = m.get('Worst5Y', float('nan'))
        pass_a = '✅' if (not np.isnan(sh) and sh >= 0.65 and not np.isnan(w5) and w5 >= 0.0 and gap <= 8.0) else '❌'
        pass_s2 = '✅' if (not np.isnan(sh) and sh >= 0.769) else '❌'
        lines.append(
            f'| {r["id"]} | {r["struct"]} | {_fp(m.get("CAGR_FULL"))} | {_fp(m.get("CAGR_OOS"))}'
            f' | {_ff(m.get("Sharpe_OOS"))} | {_fp(m.get("MaxDD_FULL"))}'
            f' | {_fp(m.get("Worst5Y"))} | {gap:.1f}pp | {pass_a} | {pass_s2} |'
        )
    lines.append('')
    lines.append('**判定ルール**:')
    lines.append('- **vs BL_A**: Sharpe_OOS≥0.65 AND Worst5Y≥0% AND IS-OOS Gap≤8pp')
    lines.append('- **vs BL_S2**: Sharpe_OOS≥0.769 (S2版超え)')
    lines.append('')

    if leverages:
        lines.append('## H3 動的レバレッジ統計')
        lines.append('')
        lines.append('| 統計 | 値 |')
        lines.append('|---|---|')
        for k, v in leverages.items():
            lines.append(f'| {k} | {v:.2f}x |')
        lines.append('')

    lines.append('## 仮説詳細')
    lines.append('')
    for r in rows:
        lines.append(f'### {r["id"]}: {r["title"]}')
        lines.append('')
        lines.append(f'- **構造**: {r["struct"]}')
        lines.append(f'- **重み方針**: {r["weights"]}')
        lines.append(f'- **コスト前提**: {r["costs"]}')
        m = r['metrics']
        lines.append(f'- **CAGR**: FULL {_fp(m.get("CAGR_FULL"))}, IS {_fp(m.get("CAGR_IS"))}, OOS {_fp(m.get("CAGR_OOS"))}')
        lines.append(f'- **Sharpe**: FULL {_ff(m.get("Sharpe_FULL"))}, OOS {_ff(m.get("Sharpe_OOS"))}')
        lines.append(f'- **リスク**: MaxDD {_fp(m.get("MaxDD_FULL"))}, Worst5Y {_fp(m.get("Worst5Y"))}')
        lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('## 結論')
    lines.append('')
    passes_a = [r for r in rows
                if r['metrics'].get('Sharpe_OOS', -99) >= 0.65
                and r['metrics'].get('Worst5Y', -99) >= 0.0
                and abs(r['metrics'].get('CAGR_IS', 0) - r['metrics'].get('CAGR_OOS', 0)) * 100 <= 8.0]
    passes_s2 = [r for r in rows if r['metrics'].get('Sharpe_OOS', -99) >= 0.769]

    if passes_s2:
        lines.append(f'### S2版超え (Sharpe_OOS ≥ 0.769): {len(passes_s2)}件')
        for r in passes_s2:
            lines.append(f'- **{r["id"]}**: Sharpe {_ff(r["metrics"].get("Sharpe_OOS"))}, CAGR_OOS {_fp(r["metrics"].get("CAGR_OOS"))}, MaxDD {_fp(r["metrics"].get("MaxDD_FULL"))}')
        lines.append('')
    if passes_a:
        lines.append(f'### BL_A基準パス (Sharpe≥0.65, Worst5Y≥0%, Gap≤8pp): {len(passes_a)}件')
        for r in passes_a:
            lines.append(f'- **{r["id"]}**: Sharpe {_ff(r["metrics"].get("Sharpe_OOS"))}, Worst5Y {_fp(r["metrics"].get("Worst5Y"))}')
        lines.append('')
    if not passes_a and not passes_s2:
        lines.append('**全仮説で採用基準未達。**')
        lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/run_hypotheses_H1_H5.py`, `src/sleeves_extended.py`*')
    lines.append('*計画書: [GOLD_BOND_STRATEGY_PLAN_2026-05-17.md](GOLD_BOND_STRATEGY_PLAN_2026-05-17.md)*')
    return '\n'.join(lines)


def main():
    print('=' * 70)
    print('H1〜H5 Gold/Bond 仮説バックテスト')
    print('=' * 70)

    df      = load_data(DATA_PATH)
    close   = df['Close']
    dates   = df['Date']
    returns = close.pct_change().fillna(0)
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df):,} days)')

    print('Loading SOFR...')
    sofr = load_sofr(dates)

    print('Building base sleeves...')
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x_base = build_bond_3x(bond_1x, sofr, True)
    gold_1x = prepare_gold_data(dates)
    gold_2x_base = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)

    print('Building extended sleeves...')
    gold_1x_arr = np.asarray(gold_1x, dtype=float)
    gold_cfd_3x = build_gold_cfd(gold_1x_arr, 3.0, sofr, spread_annual=0.012)
    gold_cfd_5x = build_gold_cfd(gold_1x_arr, 5.0, sofr, spread_annual=0.012)
    gold_tocom_3x = build_gold_tocom(gold_1x_arr, 3.0, sofr, roll_cost_annual=0.02)
    gold_s2_dyn, gold_s2_attrs = build_gold_s2_dynamic(
        gold_1x_arr, sofr, target_vol=0.30, k_vz=0.20,
        gate_min=1.0, gate_max=5.0, spread_annual=0.012,
    )
    gold_hybrid = build_gold_hybrid(
        gold_1x_arr, sofr, w_etf=0.5, L_etf=2.0, L_cfd=4.0,
        etf_cost_annual=0.0324, cfd_spread_annual=0.012,
    )

    print('Building A2 signal...')
    raw_a2, vz, _ = build_a2_signal(close, returns, return_components=True)
    lev_A, wn_A, wg_A, wb_A, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)

    print('Computing S2 leverage...')
    L_s2 = compute_L_s2_vz_gated(returns, vz, **S2_PARAMS)

    wn_h2, wg_h2, wb_h2 = make_h2_weights(wn_A)
    wn_h4, wg_h4, wb_h4 = make_h4_weights(wn_A)

    print('\nRunning baselines and 5 hypotheses...')
    print('-' * 70)

    print('  BL_A: TQQQ 3x × Gold 2x × Bond 3x')
    bl_a = run_scenario('BL_A', close, lev_A, wn_A, wg_A, wb_A, dates,
                         gold_2x_base, bond_3x_base, sofr, 'TQQQ', 3.0)
    print('  BL_S2: S2 CFD × Gold 2x × Bond 3x')
    bl_s2 = run_scenario('BL_S2', close, lev_A, wn_A, wg_A, wb_A, dates,
                          gold_2x_base, bond_3x_base, sofr, 'CFD', np.asarray(L_s2))

    print('  H1: TQQQ 3x × Gold CFD 3x × TMF 3x')
    h1 = run_scenario('H1', close, lev_A, wn_A, wg_A, wb_A, dates,
                       gold_cfd_3x, bond_3x_base, sofr, 'TQQQ', 3.0)

    print('  H2: S2 CFD × Gold CFD 5x × TMF 3x (wn≤0.80)')
    h2 = run_scenario('H2', close, lev_A, wn_h2, wg_h2, wb_h2, dates,
                       gold_cfd_5x, bond_3x_base, sofr, 'CFD', np.asarray(L_s2))

    print('  H3: S2 CFD × Gold S2動的(1-5x) × TMF 3x')
    h3 = run_scenario('H3', close, lev_A, wn_A, wg_A, wb_A, dates,
                       gold_s2_dyn, bond_3x_base, sofr, 'CFD', np.asarray(L_s2))

    print('  H4: TQQQ 3x × TOCOM 3x × TMF軽減(wg=65%, wb=35%)')
    h4 = run_scenario('H4', close, lev_A, wn_h4, wg_h4, wb_h4, dates,
                       gold_tocom_3x, bond_3x_base, sofr, 'TQQQ', 3.0)

    print('  H5: TQQQ 3x × Gold ハイブリッド(1540 2x + CFD 4x) × TMF 3x')
    h5 = run_scenario('H5', close, lev_A, wn_A, wg_A, wb_A, dates,
                       gold_hybrid, bond_3x_base, sofr, 'TQQQ', 3.0)

    rows = [
        {'id': 'H1', 'title': '対称3x強化',
         'struct': 'TQQQ 3x × GoldCFD 3x × TMF 3x',
         'weights': 'wn動的(0.30-0.90), wg=wb=(1-wn)/2',
         'costs': 'Gold CFD: spread 1.2%/yr + (L-1)×SOFR; TMF: 0.95%+SOFR×2',
         'metrics': h1},
        {'id': 'H2', 'title': '高レバ対称(S2 + Gold5x)',
         'struct': 'S2 CFD × GoldCFD 5x × TMF 3x',
         'weights': 'wn∈[0.30,0.80] (上限引下), wg=wb=(1-wn)/2',
         'costs': 'Gold CFD 5x: spread 1.2% + 4×SOFR; TMF: 0.95%+SOFR×2',
         'metrics': h2},
        {'id': 'H3', 'title': '二重S2 (NASDAQ S2 + Gold S2)',
         'struct': 'S2 CFD × Gold S2動的(1-5x) × TMF 3x',
         'weights': 'wn動的, wg=wb=(1-wn)/2',
         'costs': 'Gold S2: target_vol=0.30, k_vz=0.20, gate∈[1,5]',
         'metrics': h3},
        {'id': 'H4', 'title': '低コスト・Gold傾斜',
         'struct': 'TQQQ 3x × TOCOM 3x × TMF軽減',
         'weights': 'wg=(1-wn)×0.65, wb=(1-wn)×0.35',
         'costs': 'TOCOM: roll 2%/yr + (L-1)×SOFR; TMF: 0.95%+SOFR×2',
         'metrics': h4},
        {'id': 'H5', 'title': 'ハイブリッド(1540+CFD)',
         'struct': 'TQQQ 3x × Gold(0.5×1540 2x + 0.5×CFD 4x) × TMF 3x',
         'weights': '現行wn動的, wg=wb=(1-wn)/2',
         'costs': '1540: 3.24%/yr固定 (信用2.80%+TER0.44%); CFD 4x: spread 1.2%+3×SOFR',
         'metrics': h5},
    ]

    print('\n' + '=' * 70)
    print(f'{"Scenario":<8} {"CAGR_FULL":>10} {"CAGR_OOS":>10} {"Sharpe_OOS":>11} {"MaxDD":>9} {"Worst5Y":>9} {"Gap":>7}')
    print('-' * 70)
    for label, m in [('BL_A', bl_a), ('BL_S2', bl_s2)] + [(r['id'], r['metrics']) for r in rows]:
        gap = abs(m.get('CAGR_IS', 0) - m.get('CAGR_OOS', 0)) * 100
        print(f'{label:<8} {m.get("CAGR_FULL",0)*100:>+9.2f}% '
              f'{m.get("CAGR_OOS",0)*100:>+9.2f}% '
              f'{m.get("Sharpe_OOS",0):>11.3f} '
              f'{m.get("MaxDD_FULL",0)*100:>+8.1f}% '
              f'{m.get("Worst5Y",0)*100:>+8.2f}% '
              f'{gap:>6.1f}pp')

    print(f'\nGold S2動的 統計: {gold_s2_attrs}')

    print('\nGenerating report...')
    dates_info = {'start': str(dates.iloc[0].date()), 'end': str(dates.iloc[-1].date())}
    md = generate_report(rows, bl_a, bl_s2, dates_info, gold_s2_attrs)
    out = os.path.join(BASE, 'H1_H5_SUMMARY_2026-05-17.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
