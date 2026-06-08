"""Shared Markdown renderer for single-asset hold-vs-cash sweeps.

Uses the house standard formatter (_sweep_format.MD_HEADER_STRAT /
fmt_row_strat) for the 9-metric table. Renders TWO tables:
  1. 全期間(1974-2026) — CAGR/Sharpe over the full sample (integrated-report
     §6 basis): regime-neutral, the primary view.
  2. OOS(canonical split) — single-split CAGR/Sharpe: secondary cross-ref.
MaxDD / Worst10Y / P10 / WFE / CI95 are full-period in both tables.
"""
from __future__ import annotations

import pandas as pd

from _sweep_format import MD_HEADER_STRAT, fmt_row_strat, MD_WFA_NOTE


def _strat_table(df: pd.DataFrame, cagr_col: str, sharpe_col: str) -> list:
    lines = [MD_HEADER_STRAT[0], MD_HEADER_STRAT[1]]
    for _, r in df.iterrows():
        row = {
            'CAGR_OOS': r[cagr_col], 'IS_OOS_gap': r['cand_is_oos_gap'],
            'Sharpe_OOS': r[sharpe_col], 'MaxDD_FULL': r['cand_maxdd'],
            'Worst10Y_star': r['cand_worst10y'], 'P10_5Y': r['cand_p10_5y'],
            'Trades_yr': r['cand_trades_yr'], 'WFA_WFE': r['cand_wfe'],
            'WFA_CI95_lo': r['cand_ci95_lo'],
        }
        lines.append(fmt_row_strat(r['signal'], row))
    return lines


def _findings(df: pd.DataFrame, asset: str, cagr_col: str) -> list:
    def _g(sig):
        s = df.loc[df['signal'] == sig, cagr_col]
        return float(s.iloc[0]) if len(s) else float('nan')
    cash = _g('ALL_CASH')
    bh = _g('BUY_AND_HOLD')
    act = df[~df['signal'].isin(['ALL_CASH', 'BUY_AND_HOLD'])]
    best = act.loc[act[cagr_col].idxmax()]
    beat_cash = act[act[cagr_col] > cash]['signal'].tolist()
    out = [
        '', '## 所見（全期間ベース・暫定／要 Phase 2.4 WFA・bootstrap）', '',
        f'- 全期間(1974-2026) 常時保有(B&H) CAGR=**{bh*100:+.2f}%** / '
        f'常時キャッシュ CAGR=**{cash*100:+.2f}%**。',
        f'- アクティブ最良 **{best["signal"]}**: CAGR={best[cagr_col]*100:+.2f}%, '
        f'Sharpe_full={best["cand_sharpe_full"]:+.3f}, MaxDD={best["cand_maxdd"]*100:.1f}%, '
        f'Worst10Y={best["cand_worst10y"]*100:+.2f}%, WFE={best["cand_wfe"]:.2f}, '
        f'CI95_lo={best["cand_ci95_lo"]:+.3f}。',
    ]
    if beat_cash:
        out.append(f'- **全期間でキャッシュ超のシグナル**: {", ".join(beat_cash)}。')
    else:
        out.append('- ⚠ 全期間でもキャッシュを上回るシグナルなし。')
    out.append('- ⚠ Trades/WFE/CI95 は NAV代理の screening 値（正式は Phase 2.4）。'
               'ALL_CASH の Sharpe は無リスク近似による退化値で参考外。')
    return out


def render_sweep_md(asset: str, df: pd.DataFrame, split: str,
                    date: str, data_note: str,
                    hypotheses: dict) -> str:
    lines = [
        f'# {asset} 単独タイミングシグナル スイープ（保有 vs キャッシュ）',
        '', f'作成日: {date}', f'最終更新日: {date}', '',
        f'> `MULTIASSET_GOLD_BOND_SIGNAL_PLAN_20260608.md` Phase 2 の成果物。{data_note}',
        '> **全期間(1974-2026)を主・単一OOS分割(' + split + ')を従**として併記'
        '（統合レポート§6=全期間 と WFA=WFE/CI95 で手法整合）。',
        '',
        '## ① 全期間(1974-2026) 9指標（CAGR_full 降順・主表）',
        '> CAGR/Sharpe列=全期間。MaxDD/Worst10Y/P10/WFE/CI95も全期間。',
        '',
    ]
    lines += _strat_table(df, 'cand_cagr_full', 'cand_sharpe_full')
    lines += ['', '## ② 参考: OOS(' + split + ') 単一分割 CAGR/Sharpe（従表）', '']
    df_oos = df.sort_values('cand_cagr_oos', ascending=False, na_position='last')
    lines += _strat_table(df_oos, 'cand_cagr_oos', 'cand_sharpe_oos')
    lines += ['', MD_WFA_NOTE, '',
              '*Sharpe ◎/★ マーカは NASDAQ ベースライン基準（参考）。*']
    lines += _findings(df, asset, 'cand_cagr_full')
    lines += ['', '## シグナル仮説', '']
    for _, r in df.iterrows():
        h = hypotheses.get(r['signal'], '')
        lines.append(f"- **{r['signal']}** — {h} (判定vs B&H: {r['judgment']})")
    return '\n'.join(lines) + '\n'
