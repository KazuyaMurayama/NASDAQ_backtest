"""
P2*/S2* 再設計: target_vol ∈ {0.10〜0.25} グリッドサーチ (2026-05-17)
======================================================================
目的: target_vol が「実際に機能する」パラメータレンジを特定。

P2* (k_vz=0.0): compute_L_vol_target のみ
S2* (k_vz=0.30, gate_min=0.50): compute_L_s2_vz_gated（確定パラメータ）

clip_rate (L_t ≥ l_max の割合) が 50% 未満であれば target_vol が機能していると判定。

出力:
  - コンソール: 全グリッド結果 + clip_rate
  - ファイル: S2_LOW_VOL_SWEEP_2026-05-17.md
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
from dynamic_leverage_strategies import (
    compute_L_vol_target,
    compute_L_s2_vz_gated,
)

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

TARGET_VOLS = [0.10, 0.13, 0.16, 0.20, 0.25]
K_VZ_S2     = 0.30
GATE_MIN_S2 = 0.50
L_MAX       = 7.0


def clip_rate(L_t: pd.Series) -> float:
    return float((np.asarray(L_t) >= L_MAX - 0.05).mean() * 100)


def lev_stats(L_t: pd.Series) -> dict:
    v = np.asarray(L_t, dtype=float)
    return {'mean': float(np.nanmean(v)), 'median': float(np.nanmedian(v)),
            'p25': float(np.nanpercentile(v, 25)), 'p75': float(np.nanpercentile(v, 75))}


def _fp(v, d=2):
    return '—' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v*100:+.{d}f}%'

def _ff(v, d=3):
    return '—' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v:.{d}f}'


def generate_report(rows, dates_info) -> str:
    lines = []
    lines.append('# P2*/S2* 低target_vol グリッドサーチ結果')
    lines.append('')
    lines.append('作成日: 2026-05-17')
    lines.append('')
    lines.append('## 検証設定')
    lines.append('')
    lines.append('| 項目 | 値 |')
    lines.append('|---|---|')
    lines.append(f'| target_vol グリッド | {TARGET_VOLS} |')
    lines.append('| P2* (baseline) | k_vz=0.0 |')
    lines.append(f'| S2* | k_vz={K_VZ_S2}, gate_min={GATE_MIN_S2} |')
    lines.append(f'| データ期間 | {dates_info["start"]} 〜 {dates_info["end"]} |')
    lines.append('')
    lines.append('**判定基準**: clip_rate < 50% → target_vol が実際にデレバ機能として稼働')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 全グリッド結果')
    lines.append('')
    lines.append('| 戦略 | target_vol | CAGR(FULL) | CAGR(IS) | CAGR(OOS) | Sharpe(OOS) | MaxDD | Worst5Y | 平均Lev | clip_rate | 機能判定 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        func = '✅ 機能' if r['clip_rate'] < 50 else '⚠️ ノイズ'
        lines.append(
            f'| {r["name"]} | {r["tv"]}'
            f' | {_fp(r["CAGR_FULL"])} | {_fp(r["CAGR_IS"])} | {_fp(r["CAGR_OOS"])}'
            f' | {_ff(r["Sharpe_OOS"])} | {_fp(r["MaxDD_FULL"])}'
            f' | {_fp(r["Worst5Y"])} | {r["lev_mean"]:.2f}x | {r["clip_rate"]:.1f}% | {func} |'
        )
    lines.append('')

    lines.append('## 考察')
    lines.append('')
    func_rows = [r for r in rows if r['clip_rate'] < 50]
    if func_rows:
        lines.append(f'**target_vol が実機能する閾値**: {min(r["tv"] for r in func_rows):.2f} 以下')
        lines.append('')
        lines.append('clip_rate < 50% の組み合わせ:')
        for r in func_rows:
            lines.append(f'- {r["name"]} (tv={r["tv"]}): clip_rate={r["clip_rate"]:.1f}%, 平均Lev={r["lev_mean"]:.2f}x, Sharpe_OOS={_ff(r["Sharpe_OOS"])}')
    else:
        lines.append('**全 target_vol でclip_rate≥50%**: {0.10〜0.25} レンジ内でも target_vol は実機能しない。')
        lines.append('NASDAQの実現ボラ中央値(≈13.6%)に対してtarget_vol/σが常に7倍以上となるため。')
    lines.append('')

    lines.append('## S2確定パラメータ（tv=0.80）との比較参考')
    lines.append('')
    lines.append('| 戦略 | CAGR(OOS) | Sharpe(OOS) | MaxDD | Worst5Y |')
    lines.append('|---|---|---|---|---|')
    lines.append('| S2確定 (tv=0.80) | +27.57% | 0.769 | -62.36% | -4.75% | (前回セッション値) |')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/s2_low_vol_sweep.py`*')
    lines.append('*関連正典: [CFD_DYNAMIC_LEVERAGE_GUIDE.md](CFD_DYNAMIC_LEVERAGE_GUIDE.md)*')
    return '\n'.join(lines)


def main():
    print('=' * 70)
    print('P2*/S2* Low target_vol Sweep')
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

    print(f'\n{"Strategy":<22} {"tv":>5} {"CAGR_OOS":>10} {"Sharpe_OOS":>11} {"MaxDD":>8} {"AvgLev":>8} {"Clip%":>7}')
    print('-' * 80)

    rows = []
    for tv in TARGET_VOLS:
        for name, L_t in [
            (f'P2* tv={tv}', compute_L_vol_target(returns, target_vol=tv)),
            (f'S2* tv={tv}', compute_L_s2_vz_gated(returns, vz, target_vol=tv,
                                                      k_vz=K_VZ_S2, gate_min=GATE_MIN_S2)),
        ]:
            nav = build_nav_strategy(
                close, lev_A, wn_A, wg_A, wb_A, dates,
                gold_2x, bond_3x, sofr, 'CFD', np.asarray(L_t), CFD_SPREAD_LOW,
            )
            m   = calc_7metrics(nav, dates)
            cr  = clip_rate(L_t)
            ls  = lev_stats(L_t)
            rows.append({
                'name': name, 'tv': tv,
                'CAGR_FULL': m.get('CAGR_FULL'), 'CAGR_IS': m.get('CAGR_IS'),
                'CAGR_OOS': m.get('CAGR_OOS'), 'Sharpe_OOS': m.get('Sharpe_OOS'),
                'MaxDD_FULL': m.get('MaxDD_FULL'), 'Worst5Y': m.get('Worst5Y'),
                'lev_mean': ls['mean'], 'clip_rate': cr,
            })
            func = '✅' if cr < 50 else '⚠️'
            print(f'{name:<22} {tv:>5.2f} {m.get("CAGR_OOS",float("nan"))*100:>+10.2f}%'
                  f' {m.get("Sharpe_OOS",float("nan")):>11.3f}'
                  f' {m.get("MaxDD_FULL",float("nan"))*100:>+8.1f}%'
                  f' {ls["mean"]:>8.2f}x {cr:>6.1f}% {func}')

    print('\nGenerating report...')
    dates_info = {'start': str(dates.iloc[0].date()), 'end': str(dates.iloc[-1].date())}
    md  = generate_report(rows, dates_info)
    out = os.path.join(BASE, 'S2_LOW_VOL_SWEEP_2026-05-17.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
