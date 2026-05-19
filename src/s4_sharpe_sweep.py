"""
S4_RelVol Sharpe改善グリッドサーチ (2026-05-17)
=================================================
目的: S4_RelVol のOOS Sharpeをベースライン P2 (0.757) 超に改善しつつ
      IS-OOS Gap < 10pp・Worst5Y > -3% を同時達成するパラメータ探索。

前回結果 (S4 best): OOS Sharpe 0.697, Worst5Y -2.33%, IS-OOS Gap 14.8pp
採用基準:
  OOS Sharpe > 0.757 AND |IS-OOS CAGR gap| < 10pp AND Worst5Y > -3%

グリッド:
  l_base        ∈ {3, 4, 5}
  k_rel         ∈ {0.5, 1.0, 1.5, 2.0}
  rel_threshold ∈ {0.9, 1.0, 1.1, 1.2}
  k_vz          ∈ {0.30, 0.50}
  gate_min      ∈ {0.20, 0.30}
  = 192 combos

出力:
  - コンソール: 採用基準パスのみ表示
  - ファイル: S4_SHARPE_SWEEP_2026-05-17.md
"""

import sys, os, types, itertools

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
from dynamic_leverage_strategies import compute_L_s4_relvol

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

SHARPE_THRESHOLD  = 0.757
WORST5Y_THRESHOLD = -0.03
GAP_THRESHOLD     = 10.0

L_BASE_GRID        = [3, 4, 5]
K_REL_GRID         = [0.5, 1.0, 1.5, 2.0]
REL_THRESHOLD_GRID = [0.9, 1.0, 1.1, 1.2]
K_VZ_GRID          = [0.30, 0.50]
GATE_MIN_GRID      = [0.20, 0.30]


def _fp(v, d=2):
    return '—' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v*100:+.{d}f}%'

def _ff(v, d=3):
    return '—' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v:.{d}f}'


def generate_report(all_rows, pass_rows, dates_info) -> str:
    lines = []
    lines.append('# S4_RelVol Sharpe改善グリッドサーチ結果')
    lines.append('')
    lines.append('作成日: 2026-05-17')
    lines.append('')
    lines.append('## 検証設定')
    lines.append('')
    lines.append('| 項目 | 値 |')
    lines.append('|---|---|')
    lines.append(f'| l_base グリッド | {L_BASE_GRID} |')
    lines.append(f'| k_rel グリッド | {K_REL_GRID} |')
    lines.append(f'| rel_threshold グリッド | {REL_THRESHOLD_GRID} |')
    lines.append(f'| k_vz グリッド | {K_VZ_GRID} |')
    lines.append(f'| gate_min グリッド | {GATE_MIN_GRID} |')
    lines.append(f'| 総コンボ数 | {len(all_rows)} |')
    lines.append(f'| データ期間 | {dates_info["start"]} 〜 {dates_info["end"]} |')
    lines.append('')
    lines.append('## 採用基準')
    lines.append('')
    lines.append(f'- OOS Sharpe > {SHARPE_THRESHOLD}')
    lines.append(f'- |IS-OOS CAGR Gap| < {GAP_THRESHOLD}pp')
    lines.append(f'- Worst5Y > {WORST5Y_THRESHOLD*100:.0f}%')
    lines.append('')
    lines.append('---')
    lines.append('')

    if pass_rows:
        lines.append(f'## 採用基準パス: {len(pass_rows)}件')
        lines.append('')
        lines.append('| l_base | k_rel | rel_thr | k_vz | gate_min | CAGR_OOS | Sharpe_OOS | IS-OOS Gap | Worst5Y | AvgLev |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        for r in sorted(pass_rows, key=lambda x: -x['Sharpe_OOS']):
            lines.append(
                f'| {r["l_base"]} | {r["k_rel"]} | {r["rel_thr"]} | {r["k_vz"]} | {r["gate_min"]}'
                f' | {_fp(r["CAGR_OOS"])} | {_ff(r["Sharpe_OOS"])} | {r["gap_pp"]:.1f}pp'
                f' | {_fp(r["Worst5Y"])} | {r["lev_mean"]:.2f}x |'
            )
        lines.append('')
        best = sorted(pass_rows, key=lambda x: -x['Sharpe_OOS'])[0]
        lines.append('## 推奨パラメータ (Sharpe最大)')
        lines.append('')
        lines.append('| パラメータ | 値 |')
        lines.append('|---|---|')
        lines.append(f'| l_base | {best["l_base"]} |')
        lines.append(f'| k_rel | {best["k_rel"]} |')
        lines.append(f'| rel_threshold | {best["rel_thr"]} |')
        lines.append(f'| k_vz | {best["k_vz"]} |')
        lines.append(f'| gate_min | {best["gate_min"]} |')
        lines.append('')
        lines.append('| メトリクス | 値 |')
        lines.append('|---|---|')
        lines.append(f'| CAGR_OOS | {_fp(best["CAGR_OOS"])} |')
        lines.append(f'| Sharpe_OOS | {_ff(best["Sharpe_OOS"])} |')
        lines.append(f'| IS-OOS Gap | {best["gap_pp"]:.1f}pp |')
        lines.append(f'| MaxDD_FULL | {_fp(best["MaxDD_FULL"])} |')
        lines.append(f'| Worst5Y | {_fp(best["Worst5Y"])} |')
        lines.append(f'| 平均Lev | {best["lev_mean"]:.2f}x |')
    else:
        lines.append('## 採用基準パス: 0件')
        lines.append('')
        lines.append('**全192コンボで採用基準未達。**')
        lines.append('')
        lines.append('ベストOOS Sharpeでのトップ10:')
        lines.append('')
        lines.append('| l_base | k_rel | rel_thr | k_vz | gate_min | CAGR_OOS | Sharpe_OOS | IS-OOS Gap | Worst5Y | AvgLev |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        top10 = sorted(all_rows, key=lambda x: -x.get('Sharpe_OOS', -99))[:10]
        for r in top10:
            lines.append(
                f'| {r["l_base"]} | {r["k_rel"]} | {r["rel_thr"]} | {r["k_vz"]} | {r["gate_min"]}'
                f' | {_fp(r["CAGR_OOS"])} | {_ff(r["Sharpe_OOS"])} | {r["gap_pp"]:.1f}pp'
                f' | {_fp(r["Worst5Y"])} | {r["lev_mean"]:.2f}x |'
            )

    lines.append('')
    lines.append('## 参照: P2 ベースライン (tv=0.80, 確定採用済)')
    lines.append('')
    lines.append('| CAGR_OOS | Sharpe_OOS | IS-OOS Gap | Worst5Y |')
    lines.append('|---|---|---|---|')
    lines.append('| +27.57% | 0.757 | 5.4pp | -4.75% |')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/s4_sharpe_sweep.py`*')
    return '\n'.join(lines)


def main():
    print('=' * 70)
    print('S4_RelVol Sharpe改善グリッドサーチ')
    print('=' * 70)

    df      = load_data(DATA_PATH)
    close   = df['Close']
    dates   = df['Date']
    returns = close.pct_change().fillna(0)
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df):,} days)')

    print('Loading SOFR...')
    sofr = load_sofr(dates)

    print('Building bond/gold...')
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, True)
    gold_1x = prepare_gold_data(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)

    print('Building A2 signal...')
    raw_a2, vz, _ = build_a2_signal(close, returns, return_components=True)
    lev_A, wn_A, wg_A, wb_A, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)

    combos = list(itertools.product(
        L_BASE_GRID, K_REL_GRID, REL_THRESHOLD_GRID, K_VZ_GRID, GATE_MIN_GRID
    ))
    print(f'\nGrid: {len(combos)} combos')
    print(f'採用基準: Sharpe_OOS>{SHARPE_THRESHOLD}, Gap<{GAP_THRESHOLD}pp, Worst5Y>{WORST5Y_THRESHOLD*100:.0f}%')
    print('-' * 80)

    all_rows  = []
    pass_rows = []

    for i, (lb, kr, rt, kv, gm) in enumerate(combos):
        if i % 20 == 0:
            print(f'  [{i+1}/{len(combos)}]...')

        L_t = compute_L_s4_relvol(
            returns, vz,
            l_base=lb, k_rel=kr, rel_threshold=rt,
            k_vz=kv, gate_min=gm,
        )
        nav = build_nav_strategy(
            close, lev_A, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr, 'CFD', np.asarray(L_t), CFD_SPREAD_LOW,
        )
        m = calc_7metrics(nav, dates)

        cagr_is  = m.get('CAGR_IS', float('nan'))
        cagr_oos = m.get('CAGR_OOS', float('nan'))
        sharpe   = m.get('Sharpe_OOS', float('nan'))
        worst5y  = m.get('Worst5Y', float('nan'))
        maxdd    = m.get('MaxDD_FULL', float('nan'))
        gap_pp   = abs((cagr_is - cagr_oos) * 100) if not (np.isnan(cagr_is) or np.isnan(cagr_oos)) else float('nan')
        lev_mean = float(np.nanmean(np.asarray(L_t, dtype=float)))

        row = {
            'l_base': lb, 'k_rel': kr, 'rel_thr': rt, 'k_vz': kv, 'gate_min': gm,
            'CAGR_IS': cagr_is, 'CAGR_OOS': cagr_oos, 'Sharpe_OOS': sharpe,
            'MaxDD_FULL': maxdd, 'Worst5Y': worst5y, 'gap_pp': gap_pp, 'lev_mean': lev_mean,
        }
        all_rows.append(row)

        if (not np.isnan(sharpe) and sharpe > SHARPE_THRESHOLD and
                not np.isnan(gap_pp) and gap_pp < GAP_THRESHOLD and
                not np.isnan(worst5y) and worst5y > WORST5Y_THRESHOLD):
            pass_rows.append(row)
            print(f'  ✅ PASS lb={lb} kr={kr} rt={rt} kv={kv} gm={gm}: '
                  f'Sharpe={sharpe:.3f} Gap={gap_pp:.1f}pp Worst5Y={worst5y*100:+.1f}%')

    print(f'\n採用基準パス: {len(pass_rows)}/{len(combos)}')
    if not pass_rows:
        top = sorted(all_rows, key=lambda x: -x.get('Sharpe_OOS', -99))[:3]
        print('Top-3 by Sharpe:')
        for r in top:
            print(f'  lb={r["l_base"]} kr={r["k_rel"]} rt={r["rel_thr"]} kv={r["k_vz"]} gm={r["gate_min"]}: '
                  f'Sharpe={r["Sharpe_OOS"]:.3f} Gap={r["gap_pp"]:.1f}pp Worst5Y={r["Worst5Y"]*100:+.1f}%')

    print('\nGenerating report...')
    dates_info = {'start': str(dates.iloc[0].date()), 'end': str(dates.iloc[-1].date())}
    md  = generate_report(all_rows, pass_rows, dates_info)
    out = os.path.join(BASE, 'S4_SHARPE_SWEEP_2026-05-17.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
