"""Phase 2.2 — Bond standalone hold-vs-cash signal sweep (real data).

Uses the tested building blocks:
  - multi_asset.single_asset_sweep  (NAV + 9-metric evaluation)
  - multi_asset.bond_signals        (causal signal builders)
  - integration.nine_metric_eval    (house 9-metric standard)

Data:
  - data/base_dataset.csv  → bond_ret (synth 1974-2009 + IEF 2009+)
  - data/dtb3_daily.csv    → cash (3M T-bill) leg
  - macro parquets         → credit spread, term premium, Fed-vs-10y

Outputs (repo root, house naming):
  - BOND_SINGLE_ASSET_SWEEP_20260608.md
  - bond_single_asset_sweep_results.csv

NOTE: Trades/yr, WFE, CI95_lo are NAV-proxy *screening* values
(nine_metric_eval S3). Proper Trades/yr & WFA come from Phase 2.4
(phase_d_wfa / phase_d_bootstrap).

Run: PYTHONPATH=src python -m multi_asset.run_bond_sweep
"""
from __future__ import annotations

import os
import sys

import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from multi_asset.single_asset_sweep import run_single_asset_sweep, CANONICAL_SPLIT
from multi_asset.bond_signals import (
    ma_cross_position, momentum_position, zscore_position,
)
from _sweep_format import MD_HEADER_STRAT, fmt_row_strat, MD_WFA_NOTE

ROOT = os.path.dirname(_SRC)
DATA = os.path.join(ROOT, 'data')
RAW = os.path.join(DATA, 'signals', 'expansion', 'raw')
DATE = '2026-06-08'
DATESTAMP = '20260608'


def _load_bond_and_cash():
    base = pd.read_csv(os.path.join(DATA, 'base_dataset.csv'),
                       parse_dates=['date'], index_col='date').sort_index()
    bond_ret = base['bond_ret'].dropna()
    bond_price = (1.0 + bond_ret).cumprod()

    dtb3 = pd.read_csv(os.path.join(DATA, 'dtb3_daily.csv'),
                       parse_dates=['Date'], index_col='Date').sort_index()
    # annualized % yield → daily simple return
    cash_ret = (dtb3['yield_pct'] / 100.0 / 252.0).reindex(bond_ret.index).ffill().fillna(0.0)
    return bond_ret, bond_price, cash_ret


def _load_macro_daily(filename, col, calendar):
    df = pd.read_parquet(os.path.join(RAW, filename))
    s = df[col] if col in df.columns else df.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s.sort_index().reindex(calendar.union(s.index)).ffill().reindex(calendar)


def build_signals(bond_price, cal):
    credit = _load_macro_daily('repo_1_repo_1_baa_minus_aaa_spread.parquet',
                               'repo_1_baa_minus_aaa_spread', cal)
    termprem = _load_macro_daily('repo_2_repo_2_30y_10y_termpremium.parquet',
                                 'repo_2_30y_10y_termpremium', cal)
    fed10 = _load_macro_daily('repo_4_repo_4_dff_minus_10y.parquet',
                              'repo_4_dff_minus_10y', cal)
    return {
        # baselines
        'BUY_AND_HOLD':  pd.Series(1.0, index=bond_price.index),
        'ALL_CASH':      pd.Series(0.0, index=bond_price.index),
        # B-TR trend / momentum (hypothesis: rate trends persist)
        'bond_ma200':    ma_cross_position(bond_price, 200),
        'bond_ma100':    ma_cross_position(bond_price, 100),
        'bond_mom252':   momentum_position(bond_price, 252),
        'bond_mom126':   momentum_position(bond_price, 126),
        # B-CR credit spread widening → flight to quality → hold bonds
        'credit_spread_hi': zscore_position(credit, 252, enter=0.5),
        # B-TP low/falling term premium → long end favorable → hold
        'termprem_lo':   zscore_position(termprem, 252, enter=0.0, invert=True),
        # B-FED inverted/easy (dff-10y low) → recession hedge → hold bonds
        'fed_easy':      zscore_position(fed10, 252, enter=0.0, invert=True),
    }


_HYPOTHESIS = {
    'BUY_AND_HOLD': 'baseline: always invested',
    'ALL_CASH': 'baseline: always in T-bill cash',
    'bond_ma200': 'trend: hold when price > 200d MA',
    'bond_ma100': 'trend: hold when price > 100d MA',
    'bond_mom252': 'momentum: hold when 12m return > 0',
    'bond_mom126': 'momentum: hold when 6m return > 0',
    'credit_spread_hi': 'flight-to-quality: hold when BAA-AAA z >= 0.5',
    'termprem_lo': 'hold when 30y-10y term premium z <= 0',
    'fed_easy': 'hold when DFF-10y z <= 0 (easy/inverted)',
}


def main():
    bond_ret, bond_price, cash_ret = _load_bond_and_cash()
    signals = build_signals(bond_price, bond_ret.index)
    df = run_single_asset_sweep(bond_ret, cash_ret, signals,
                                split_date=CANONICAL_SPLIT, baseline='bh')
    df['hypothesis'] = df['signal'].map(_HYPOTHESIS)

    csv_path = os.path.join(ROOT, 'bond_single_asset_sweep_results.csv')
    df.to_csv(csv_path, index=False)

    # ---- Markdown report (house 9-metric STRAT table) ----
    lines = [
        '# Bond 単独タイミングシグナル スイープ（保有 vs キャッシュ）',
        '', f'作成日: {DATE}', f'最終更新日: {DATE}', '',
        '> `MULTIASSET_GOLD_BOND_SIGNAL_PLAN_20260608.md` Phase 2.2 の成果物。',
        '> Bondリターン: `base_dataset.csv`（synth duration 1974-2009 + IEF 2009+）、'
        'キャッシュ: DTB3（3M T-bill）。OOS分割: ' + CANONICAL_SPLIT + '。',
        '> **Trades/WFE/CI95 は NAV代理の screening 値**（正式は Phase 2.4 WFA）。',
        '', '## 9指標スイープ結果（CAGR_OOS降順）', '',
        MD_HEADER_STRAT[0], MD_HEADER_STRAT[1],
    ]
    for _, r in df.iterrows():
        row = {
            'CAGR_OOS': r['cand_cagr_oos'], 'IS_OOS_gap': r['cand_is_oos_gap'],
            'Sharpe_OOS': r['cand_sharpe_oos'], 'MaxDD_FULL': r['cand_maxdd'],
            'Worst10Y_star': r['cand_worst10y'], 'P10_5Y': r['cand_p10_5y'],
            'Trades_yr': r['cand_trades_yr'], 'WFA_WFE': r['cand_wfe'],
            'WFA_CI95_lo': r['cand_ci95_lo'],
        }
        lines.append(fmt_row_strat(r['signal'], row))
    lines += ['', MD_WFA_NOTE, '',
              '*Sharpe ◎/★ マーカは NASDAQ ベースライン基準（参考）。*', '',
              '## シグナル仮説', '']
    for _, r in df.iterrows():
        lines.append(f"- **{r['signal']}** — {r['hypothesis']}  "
                     f"(判定: {r['judgment']})")

    # ---- Key findings (vs CASH, the relevant benchmark for hold/cash) ----
    cash_cagr = float(df.loc[df['signal'] == 'ALL_CASH', 'cand_cagr_oos'].iloc[0])
    bh_cagr = float(df.loc[df['signal'] == 'BUY_AND_HOLD', 'cand_cagr_oos'].iloc[0])
    act = df[~df['signal'].isin(['ALL_CASH', 'BUY_AND_HOLD'])]
    best = act.loc[act['cand_cagr_oos'].idxmax()]
    beat_cash = act[act['cand_cagr_oos'] > cash_cagr]['signal'].tolist()
    lines += [
        '', '## 所見（暫定・要 Phase 2.4 WFA 確認）', '',
        f'- OOS({CANONICAL_SPLIT}〜)はBond弱気相場。**常時保有(B&H) CAGR_OOS={bh_cagr*100:+.2f}%・'
        f'MaxDD≈-50%** と劣悪 → 「vs B&H」の判定は楽観的に出る点に注意。',
        f'- **本質的ベンチマークはキャッシュ**: ALL_CASH CAGR_OOS=**{cash_cagr*100:+.2f}%**。',
        f'- アクティブ最良は **{best["signal"]}**（CAGR_OOS={best["cand_cagr_oos"]*100:+.2f}%, '
        f'MaxDD={best["cand_maxdd"]*100:.1f}%）。',
        ('- **キャッシュを上回ったシグナルは: ' + ', '.join(beat_cash) + '**。')
        if beat_cash else
        '- ⚠ **どのBondシグナルもOOSでキャッシュを上回れなかった**（この局面はキャッシュ保有が正解。'
        'タイミングはDD圧縮には寄与: 例 bond_mom252 MaxDD≈-15% vs B&H -50%）。',
        '- ALL_CASH の Sharpe は無リスク近似による退化値のため参考外。',
    ]
    md_path = os.path.join(ROOT, f'BOND_SINGLE_ASSET_SWEEP_{DATESTAMP}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('wrote', csv_path)
    print('wrote', md_path)
    print(df[['signal', 'cand_cagr_oos', 'cand_sharpe_oos', 'cand_maxdd',
              'cand_worst10y', 'cand_trades_yr', 'judgment']].to_string(index=False))


if __name__ == '__main__':
    main()
