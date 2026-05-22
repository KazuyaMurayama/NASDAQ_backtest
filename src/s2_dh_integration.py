"""
S2 を DH Dynポートフォリオ正式統合シナリオ試算 (2026-05-17)
=============================================================
目的: S2_VZGated (CFD動的レバレッジ) を DH Dyn [A] ポートフォリオの
      NASDAQスリーブに統合した場合の効果を検証する。

シナリオ:
  A: DH Dyn [A] + TQQQ 3x (現行ベスト, ベースライン)
  B: DH Dyn [A] + S2 CFD (L_t ∈ [1,7], 動的, tv=0.80, k_vz=0.30, gate_min=0.50)

主な論点:
  - S2 avg 5x vs TQQQ effective 3x → リスク密度が異なる
  - A2タイミング信号は同一 (Approach A, 閾値0.15)
  - Gold2x + Bond3x スリーブは同一
  - CFD はvol drag なし (TQQQ は vol drag あり)

出力:
  - コンソール: 全期間指標 + 年次リターン比較
  - ファイル: S2_DH_INTEGRATION_2026-05-17.md
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
    IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

S2_PARAMS = dict(target_vol=0.80, k_vz=0.30, gate_min=0.50)

FULL_START = '1974-01-02'


def yearly_returns(nav: pd.Series, dates: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(dates.values)
    nav_s = pd.Series(nav.values, index=dt)
    years = sorted(set(dt.year))
    rows = []
    for yr in years:
        mask = nav_s.index.year == yr
        sub = nav_s[mask]
        if len(sub) < 2:
            continue
        ret = sub.iloc[-1] / sub.iloc[0] - 1
        rows.append({'year': yr, 'return': ret})
    return pd.DataFrame(rows).set_index('year')


def _fp(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v*100:+.{d}f}%'

def _ff(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:.{d}f}'


def generate_report(m_tqqq, m_cfd, yr_tqqq, yr_cfd, lev_s2, dates_info) -> str:
    lines = []
    lines.append('# S2 DH Dyn統合シナリオ試算')
    lines.append('')
    lines.append('作成日: 2026-05-17')
    lines.append('')
    lines.append('## 比較シナリオ')
    lines.append('')
    lines.append('| シナリオ | NASDAQスリーブ | レバレッジ | vol drag |')
    lines.append('|---|---|---|---|')
    lines.append('| A: 現行ベスト | TQQQ 3x ETF | 固定 3x | あり |')
    lines.append(f'| B: S2 CFD統合 | CFD 動的 | {np.nanmean(np.asarray(lev_s2, dtype=float)):.2f}x (平均) | なし |')
    lines.append('')
    lines.append('**S2確定パラメータ**: target_vol=0.80, k_vz=0.30, gate_min=0.50, l_min=1.0, l_max=7.0')
    lines.append('')
    lines.append(f'データ期間: {dates_info["start"]} 〜 {dates_info["end"]}')
    lines.append(f'IS: 〜{IS_END} / OOS: {OOS_START}〜')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## パフォーマンス指標比較')
    lines.append('')
    lines.append('| 指標 | A: TQQQ 3x | B: S2 CFD | 差 (B-A) |')
    lines.append('|---|---|---|---|')

    metrics = [
        ('CAGR_FULL', 'CAGR (全期間)'),
        ('CAGR_IS',   'CAGR (IS)'),
        ('CAGR_OOS',  'CAGR (OOS)'),
        ('Sharpe_FULL','Sharpe (全期間)'),
        ('Sharpe_IS',  'Sharpe (IS)'),
        ('Sharpe_OOS', 'Sharpe (OOS)'),
        ('MaxDD_FULL', 'MaxDD (全期間)'),
        ('Worst5Y',    'Worst5Y'),
    ]
    for key, label in metrics:
        va = m_tqqq.get(key)
        vb = m_cfd.get(key)
        if key in ('Sharpe_FULL', 'Sharpe_IS', 'Sharpe_OOS'):
            fa = _ff(va); fb = _ff(vb)
            diff = f'{vb - va:+.3f}' if (va is not None and vb is not None and
                                           not np.isnan(va) and not np.isnan(vb)) else '—'
        else:
            fa = _fp(va); fb = _fp(vb)
            diff = (_fp(vb - va) if (va is not None and vb is not None and
                                      not np.isnan(va) and not np.isnan(vb)) else '—')
        lines.append(f'| {label} | {fa} | {fb} | {diff} |')

    lines.append('')
    lines.append('## IS-OOS Gap (オーバーフィッティング指標)')
    lines.append('')
    for name, m in [('A: TQQQ', m_tqqq), ('B: S2 CFD', m_cfd)]:
        ci = m.get('CAGR_IS', float('nan'))
        co = m.get('CAGR_OOS', float('nan'))
        if not (np.isnan(ci) or np.isnan(co)):
            lines.append(f'- **{name}**: IS {_fp(ci)}, OOS {_fp(co)}, Gap {abs(ci-co)*100:.1f}pp')
        else:
            lines.append(f'- **{name}**: データ不足')
    lines.append('')

    lines.append('## 年次リターン比較 (FULL期間)')
    lines.append('')
    lines.append('| 年 | A: TQQQ 3x | B: S2 CFD | 差 (B-A) | OOS |')
    lines.append('|---|---|---|---|---|')
    all_years = sorted(set(list(yr_tqqq.index) + list(yr_cfd.index)))
    for yr in all_years:
        ra = yr_tqqq.loc[yr, 'return'] if yr in yr_tqqq.index else float('nan')
        rb = yr_cfd.loc[yr, 'return'] if yr in yr_cfd.index else float('nan')
        diff = rb - ra if not (np.isnan(ra) or np.isnan(rb)) else float('nan')
        oos_tag = '✅OOS' if yr >= int(OOS_START[:4]) else ''
        fa = _fp(ra) if not np.isnan(ra) else '—'
        fb = _fp(rb) if not np.isnan(rb) else '—'
        fd = _fp(diff) if not np.isnan(diff) else '—'
        lines.append(f'| {yr} | {fa} | {fb} | {fd} | {oos_tag} |')

    lines.append('')
    lines.append('## 考察')
    lines.append('')
    ca = m_tqqq.get('CAGR_FULL', float('nan'))
    cb = m_cfd.get('CAGR_FULL', float('nan'))
    sa = m_tqqq.get('Sharpe_OOS', float('nan'))
    sb = m_cfd.get('Sharpe_OOS', float('nan'))
    w5a = m_tqqq.get('Worst5Y', float('nan'))
    w5b = m_cfd.get('Worst5Y', float('nan'))
    avg_lev = float(np.nanmean(np.asarray(lev_s2, dtype=float)))

    lines.append(f'- S2 CFDの平均レバレッジ: **{avg_lev:.2f}x** (TQQQの実効3xより{"高い" if avg_lev > 3.0 else "低い"})')
    if not (np.isnan(ca) or np.isnan(cb)):
        if cb > ca:
            lines.append(f'- CAGR: S2 CFD が {(cb-ca)*100:+.2f}pp 上回る (+{cb*100:.2f}% vs +{ca*100:.2f}%)')
        else:
            lines.append(f'- CAGR: TQQQ が {(ca-cb)*100:+.2f}pp 上回る (+{ca*100:.2f}% vs +{cb*100:.2f}%)')
    if not (np.isnan(sa) or np.isnan(sb)):
        if sb > sa:
            lines.append(f'- OOS Sharpe: S2 CFD が {sb-sa:+.3f} 上回る ({sb:.3f} vs {sa:.3f})')
        else:
            lines.append(f'- OOS Sharpe: TQQQ が {sa-sb:+.3f} 上回る ({sa:.3f} vs {sb:.3f})')
    if not (np.isnan(w5a) or np.isnan(w5b)):
        if w5b > w5a:
            lines.append(f'- Worst5Y: S2 CFD が {(w5b-w5a)*100:+.2f}pp 良い ({w5b*100:+.2f}% vs {w5a*100:+.2f}%)')
        else:
            lines.append(f'- Worst5Y: TQQQ が {(w5a-w5b)*100:+.2f}pp 良い ({w5a*100:+.2f}% vs {w5b*100:+.2f}%)')
    lines.append('')
    lines.append('### vol drag 効果')
    lines.append('CFD は vol drag なし。TQQQ は日次リバランス型で高ボラ時にvol dragが発生する。')
    lines.append('S2の動的デレバ（高ボラ時にレバレッジ削減）はvol drag軽減と方向性が一致する。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/s2_dh_integration.py`*')
    return '\n'.join(lines)


def main():
    print('=' * 70)
    print('S2 DH Dyn 統合シナリオ試算')
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

    print('Building A2 signal (Approach A, threshold=0.15)...')
    raw_a2, vz, _ = build_a2_signal(close, returns, return_components=True)
    lev_A, wn_A, wg_A, wb_A, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)

    print('Computing S2 leverage...')
    L_s2 = compute_L_s2_vz_gated(returns, vz, **S2_PARAMS)

    print('\nBuilding NAVs...')
    nav_tqqq = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr, 'TQQQ', 3.0, CFD_SPREAD_LOW,
    )
    nav_cfd = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr, 'CFD', np.asarray(L_s2), CFD_SPREAD_LOW,
    )

    m_tqqq = calc_7metrics(nav_tqqq, dates)
    m_cfd  = calc_7metrics(nav_cfd, dates)

    print(f'\n{"指標":<22} {"A: TQQQ 3x":>14} {"B: S2 CFD":>14} {"差(B-A)":>10}')
    print('-' * 65)
    for key, label in [
        ('CAGR_FULL', 'CAGR全期間'), ('CAGR_IS', 'CAGR IS'), ('CAGR_OOS', 'CAGR OOS'),
        ('Sharpe_OOS', 'Sharpe OOS'), ('MaxDD_FULL', 'MaxDD'), ('Worst5Y', 'Worst5Y'),
    ]:
        va = m_tqqq.get(key, float('nan'))
        vb = m_cfd.get(key, float('nan'))
        if 'Sharpe' in key:
            fa = f'{va:.3f}'; fb = f'{vb:.3f}'
            fd = f'{vb-va:+.3f}' if not (np.isnan(va) or np.isnan(vb)) else '—'
        else:
            fa = f'{va*100:+.2f}%'; fb = f'{vb*100:+.2f}%'
            fd = f'{(vb-va)*100:+.2f}pp' if not (np.isnan(va) or np.isnan(vb)) else '—'
        print(f'{label:<22} {fa:>14} {fb:>14} {fd:>10}')

    avg_lev = float(np.nanmean(np.asarray(L_s2, dtype=float)))
    print(f'\nS2 平均レバレッジ: {avg_lev:.2f}x')
    print(f'IS-OOS Gap TQQQ: {abs(m_tqqq.get("CAGR_IS",0)-m_tqqq.get("CAGR_OOS",0))*100:.1f}pp')
    print(f'IS-OOS Gap S2  : {abs(m_cfd.get("CAGR_IS",0)-m_cfd.get("CAGR_OOS",0))*100:.1f}pp')

    yr_tqqq = yearly_returns(nav_tqqq, dates)
    yr_cfd  = yearly_returns(nav_cfd, dates)

    print('\nGenerating report...')
    dates_info = {'start': str(dates.iloc[0].date()), 'end': str(dates.iloc[-1].date())}
    md  = generate_report(m_tqqq, m_cfd, yr_tqqq, yr_cfd, L_s2, dates_info)
    out = os.path.join(BASE, 'S2_DH_INTEGRATION_2026-05-17.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
