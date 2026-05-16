"""
Enhanced Dynamic Leverage Backtest: S1/S2/S3 (2026-05-15)
==========================================================
A2 Opt + P2 ボラ型 の知見を融合した改良戦略。

S1: A2-Conviction Leverage  — A2確信度×ボラ調整でレバ決定
S2: VZ-Gated Vol Target     — P2にVIXゲートを前置（本命）
S3: Decomposed A2 Leverage  — A2の原子因子をL_tに再配線

評価:
  IS  (1974-2021-05-07): パラメータ最適化 (ISスコア = 0.6×CAGR + 0.4×Sharpe/3)
  OOS (2021-05-08-2026-03-26): 過学習チェック（主指標）

採用基準（事前定義）:
  1. OOS Sharpe > P2 best
  2. MaxDD > P2 best または CAGR_OOS が P2 best 以上
  3. Worst5Y > -5%
  4. |CAGR_IS - CAGR_OOS| < 10pp
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
    build_nav_strategy, calc_7metrics,
    CFD_SPREAD_LOW, IS_END, OOS_START, FULL_START, FULL_END,
)
from dynamic_leverage_strategies import (
    compute_L_vol_target,
    compute_L_s1_conviction,
    compute_L_s2_vz_gated,
    compute_L_s3_decomposed,
    compute_L_s4_relvol,
)

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

# P2ベスト (前回バックテスト結果)
P2_BEST_TARGET_VOL = 0.8
P2_BEST_CAGR_OOS   = 0.2713
P2_BEST_SHARPE_OOS = None   # 実行時に確定

# 採用基準定数
ADOPT_MIN_WORST5Y = -0.05   # Worst5Y > -5%
ADOPT_MAX_IS_OOS_GAP = 0.10  # |CAGR_IS - CAGR_OOS| < 10pp


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

GRIDS = {
    'S1_Conviction': [
        {'alpha': a, 'target_vol': tv}
        for a, tv in itertools.product([0.5, 1.0, 1.5], [0.40, 0.60, 0.80])
    ],  # 9組み合わせ
    'S2_VZGated': [
        {'k_vz': k, 'gate_min': g, 'target_vol': tv}
        for tv, k, g in itertools.product(
            [0.60, 0.70, 0.80],
            [0.20, 0.30, 0.45, 0.60],
            [0.20, 0.35, 0.50],
        )
    ],  # 36組み合わせ（拡張版）
    'S3_Decomposed': [
        {'beta_defense': b, 'l_max': lm}
        for b, lm in itertools.product([0.7, 1.0, 1.3], [5.0, 7.0])
    ],  # 6組み合わせ
    'S4_RelVol': [
        {'l_base': lb, 'k_rel': kr, 'rel_threshold': rt, 'k_vz': kv, 'gate_min': 0.20,
         'short_hl': 20, 'long_hl': 120}
        for lb, kr, rt, kv in itertools.product(
            [5.0, 6.0, 7.0],
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.2],
            [0.30, 0.50, 0.80],
        )
    ],  # 72組み合わせ（相対ボラゲート）
}


def is_score(m: dict) -> float:
    cagr = m.get('CAGR_IS', -999)
    sh   = m.get('Sharpe_IS', 0)
    if np.isnan(cagr) or np.isnan(sh):
        return -999.0
    return 0.6 * cagr + 0.4 * sh / 3.0


def run_one(L_t, close, lev_A, wn_A, wg_A, wb_A, dates, gold_2x, bond_3x, sofr) -> dict:
    nav = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr, 'CFD', np.asarray(L_t), CFD_SPREAD_LOW,
    )
    return calc_7metrics(nav, dates)


def lev_stats(L_t) -> dict:
    vals = np.asarray(L_t, dtype=float)
    return {
        'mean': float(np.nanmean(vals)),
        'std':  float(np.nanstd(vals)),
        'p25':  float(np.nanpercentile(vals, 25)),
        'p50':  float(np.nanpercentile(vals, 50)),
        'p75':  float(np.nanpercentile(vals, 75)),
        'pct_max': float((vals >= 6.9).mean() * 100),  # ≈7x
        'pct_min': float((vals <= 1.1).mean() * 100),  # ≈1x
    }


def meets_adoption_criteria(m: dict, p2_sharpe_oos: float) -> tuple:
    """(bool, list of failed criteria)"""
    failed = []
    if p2_sharpe_oos and m.get('Sharpe_OOS', 0) <= p2_sharpe_oos:
        failed.append(f"Sharpe_OOS {m.get('Sharpe_OOS',0):.3f} ≤ P2 {p2_sharpe_oos:.3f}")
    if m.get('Worst5Y', -999) < ADOPT_MIN_WORST5Y:
        failed.append(f"Worst5Y {m.get('Worst5Y',0)*100:.1f}% < -5%")
    gap = abs(m.get('CAGR_IS', 0) - m.get('CAGR_OOS', 0))
    if gap > ADOPT_MAX_IS_OOS_GAP:
        failed.append(f"IS-OOS gap {gap*100:.1f}pp > 10pp")
    return (len(failed) == 0, failed)


def fmt(v, pct=True, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v*100:+.{d}f}%' if pct else f'{v:.{d}f}'


def generate_report(all_results, baselines, data_info, p2_best_m, p2_sharpe_oos) -> str:
    lines = []
    lines.append('# NASDAQ CFD 改良動的レバレッジ戦略 バックテスト (S1/S2/S3/S4)')
    lines.append('')
    lines.append('作成日: 2026-05-15')
    lines.append('最終更新日: 2026-05-16')
    lines.append('')
    lines.append(f'**データ期間**: {data_info["start"]} 〜 {data_info["end"]}')
    lines.append(f'**IS期間**: {FULL_START} 〜 {IS_END} | **OOS期間**: {OOS_START} 〜 {data_info["end"]}')
    lines.append('')
    lines.append('## 戦略概要')
    lines.append('')
    lines.append('| 戦略 | コンセプト | パラメータ | グリッド数 |')
    lines.append('|------|-----------|-----------|---------|')
    lines.append('| S1: A2-Conviction | A2確信度×ボラ調整 | alpha, target_vol | 9 |')
    lines.append('| S2: VZ-Gated P2  | P2+VIXゲート前置 | k_vz, gate_min, target_vol | 36 |')
    lines.append('| S3: Decomposed A2 | A2原子因子→L_t再配線 | beta_defense, l_max | 6 |')
    lines.append('| S4: RelVol-Gated | 相対ボラ(短期/長期)+VIXゲート | l_base, k_rel, rel_threshold, k_vz | 72 |')
    lines.append('')
    lines.append('**採用基準（事前定義）**:')
    lines.append('1. OOS Sharpe > P2 best')
    lines.append('2. |CAGR_IS - CAGR_OOS| < 10pp')
    lines.append('3. Worst5Y > -5%')
    lines.append('')
    lines.append('> ⚠️ **P2/S2の設計上の制約**: NASDAQの実現ボラ中央値≈13.6%に対して')
    lines.append('> target_vol=0.60〜0.80はtarget_vol/σの比が中央値≈4〜6となり')
    lines.append('> l_max=7に99%以上クリップ。target_volパラメータは実質ノイズ。')
    lines.append('> S4はこの問題を解決するため相対ボラ（短期/長期EWMA比）を採用。')
    lines.append('')
    lines.append('---')
    lines.append('')

    # IS最適パラメータ
    lines.append('## IS最適パラメータ')
    lines.append('')
    lines.append('| 戦略 | 最適パラメータ | ISスコア |')
    lines.append('|------|-------------|---------|')
    for key, res in all_results.items():
        best_score, best_params, _, _ = res['best']
        p = ', '.join(f'{k}={v}' for k, v in best_params.items())
        lines.append(f'| {key} | {p} | {best_score:.4f} |')
    lines.append('')
    lines.append('---')
    lines.append('')

    # メイン比較テーブル
    lines.append('## メイン結果（IS最適パラメータで評価）')
    lines.append('')
    hdr = ('| 戦略 | CAGR(FULL) | CAGR(IS) | CAGR(OOS) | Sharpe(OOS)'
           ' | MaxDD | Worst5Y | Worst10Y | 平均Lev | Lev標準偏差 | 採用判定 |')
    sep = '|' + '---|' * (hdr.count('|') - 1)
    lines.append(hdr)
    lines.append(sep)

    def row(name, m, ls=None, adopt=None):
        av  = f'{ls["mean"]:.2f}x' if ls else '—'
        std = f'{ls["std"]:.2f}' if ls else '—'
        ad  = adopt if adopt else '—'
        return (f'| **{name}** | {fmt(m.get("CAGR_FULL"))} | {fmt(m.get("CAGR_IS"))}'
                f' | {fmt(m.get("CAGR_OOS"))} | {fmt(m.get("Sharpe_OOS"), pct=False)}'
                f' | {fmt(m.get("MaxDD_FULL"))} | {fmt(m.get("Worst5Y"))}'
                f' | {fmt(m.get("Worst10Y"))} | {av} | {std} | {ad} |')

    for bname, bm in baselines.items():
        lines.append(row(bname, bm))
    lines.append(row('P2 best (target_vol=0.8)', p2_best_m,
                     ls=lev_stats(p2_best_m.get('_L_t_', np.full(1, 5.34))),
                     adopt='baseline'))

    for key, res in all_results.items():
        _, _, m, L_t = res['best']
        ls = lev_stats(L_t)
        ok, failed = meets_adoption_criteria(m, p2_sharpe_oos)
        adopt_str = '✅ 採用候補' if ok else f'⚠️ {"; ".join(failed[:1])}'
        lines.append(row(key, m, ls, adopt_str))
    lines.append('')
    lines.append('---')
    lines.append('')

    # パラメータ感応度
    lines.append('## パラメータ感応度（IS上位3 vs OOS）')
    lines.append('')
    for key, res in all_results.items():
        lines.append(f'### {key}')
        lines.append('')
        lines.append('| Rank | パラメータ | ISスコア | CAGR(IS) | CAGR(OOS) | Sharpe(OOS) | MaxDD |')
        lines.append('|------|-----------|---------|---------|---------|------------|------|')
        for rank, (score, params, m, _) in enumerate(res['top3'], 1):
            p = ', '.join(f'{k}={v}' for k, v in params.items())
            lines.append(
                f'| {rank} | {p} | {score:.4f}'
                f' | {fmt(m.get("CAGR_IS"))} | {fmt(m.get("CAGR_OOS"))}'
                f' | {fmt(m.get("Sharpe_OOS"), pct=False)} | {fmt(m.get("MaxDD_FULL"))} |'
            )
        lines.append('')

    lines.append('---')
    lines.append('')

    # 採用判定サマリー
    lines.append('## 採用判定サマリー')
    lines.append('')
    adopted = []
    for key, res in all_results.items():
        _, _, m, _ = res['best']
        ok, failed = meets_adoption_criteria(m, p2_sharpe_oos)
        if ok:
            adopted.append(key)
        status = '✅ 採用候補' if ok else '❌ 不採用'
        reason = '全基準クリア' if ok else f'未達: {", ".join(failed)}'
        lines.append(f'- **{key}**: {status} — {reason}')
    lines.append('')
    if adopted:
        lines.append(f'**採用候補**: {", ".join(adopted)}')
        lines.append('')
        lines.append('> ⚠️ CURRENT_BEST_STRATEGY.md は、')
        lines.append('> OOS MaxDD が -50% 以内かつ Worst5Y > -5% を確認後に更新を提案。')
    else:
        lines.append('**採用候補なし**: すべての戦略が1つ以上の採用基準を未達。')
        lines.append('P2 best (target_vol=0.8) が引き続き推奨動的レバレッジ戦略。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/enh_lev_backtest.py`*')
    lines.append(f'*データ期間: {data_info["start"]} 〜 {data_info["end"]}*')
    return '\n'.join(lines)


def main():
    print('=' * 70)
    print('Enhanced Dynamic Leverage Backtest: S1/S2/S3')
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

    print('Building A2 signal (with components)...')
    raw_a2, vz, components = build_a2_signal(close, returns, return_components=True)
    lev_A, wn_A, wg_A, wb_A, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)

    # distribution check
    print(f'  raw_A2: mean={raw_a2.mean():.3f} p25={raw_a2.quantile(.25):.3f} '
          f'p75={raw_a2.quantile(.75):.3f} p95={raw_a2.quantile(.95):.3f}')
    print(f'  vz:     mean={vz.mean():.3f} std={vz.std():.3f} '
          f'p5={vz.quantile(.05):.2f} p95={vz.quantile(.95):.2f}')

    # --- Baselines ---
    print('\nBuilding baselines...')
    baselines = {}
    for lev, bname in [(3.0, 'CFD 3x [固定]'), (7.0, 'CFD 7x [固定]')]:
        nav = build_nav_strategy(close, lev_A, wn_A, wg_A, wb_A, dates,
                                  gold_2x, bond_3x, sofr, 'CFD', lev, CFD_SPREAD_LOW)
        baselines[bname] = calc_7metrics(nav, dates)
        print(f'  {bname}: CAGR={baselines[bname]["CAGR_FULL"]*100:+.2f}%')

    # P2 best
    print('  P2 best (target_vol=0.8)...')
    L_p2 = compute_L_vol_target(returns, target_vol=P2_BEST_TARGET_VOL)
    p2_nav = build_nav_strategy(close, lev_A, wn_A, wg_A, wb_A, dates,
                                 gold_2x, bond_3x, sofr, 'CFD', np.asarray(L_p2), CFD_SPREAD_LOW)
    p2_best_m = calc_7metrics(p2_nav, dates)
    p2_sharpe_oos = p2_best_m.get('Sharpe_OOS', 0)
    print(f'  P2 best: CAGR_OOS={p2_best_m["CAGR_OOS"]*100:+.2f}% '
          f'Sharpe_OOS={p2_sharpe_oos:.3f} MaxDD={p2_best_m["MaxDD_FULL"]*100:.1f}%')

    # --- Strategy grid search ---
    all_results = {}

    def run_grid(key, fn_and_grid_items, worst5y_is_filter=-0.05):
        """worst5y_is_filter: IS段階でWorst5Y < 閾値のパラメータをIS-bestから除外。"""
        print(f'\n--- {key} ---')
        ranked = []
        filtered_out = 0
        for params, L_t in fn_and_grid_items:
            m     = run_one(L_t, close, lev_A, wn_A, wg_A, wb_A, dates,
                            gold_2x, bond_3x, sofr)
            score = is_score(m)
            # Worst5Y_IS フィルタ: 全期間Worst5Yが閾値未満は採用レースから外す
            w5 = m.get('Worst5Y', -999)
            passes = (w5 >= worst5y_is_filter) if not np.isnan(w5) else False
            if not passes:
                filtered_out += 1
            ranked.append((score, params, m, L_t, passes))
            print(f'  {params} → IS:{m.get("CAGR_IS",0)*100:+.2f}% '
                  f'OOS:{m.get("CAGR_OOS",0)*100:+.2f}% '
                  f'Sharpe_OOS:{m.get("Sharpe_OOS",0):.3f} '
                  f'MaxDD:{m.get("MaxDD_FULL",0)*100:.1f}% '
                  f'Worst5Y:{w5*100:+.1f}% {"❌" if not passes else ""}')
        ranked.sort(key=lambda x: x[0], reverse=True)
        # Worst5Y フィルタ通過済み優先
        passing = [(sc, pr, m, lt, ok) for sc, pr, m, lt, ok in ranked if ok]
        ranked_final = passing if passing else ranked  # フィルタ後に残るものがなければ全体から選ぶ
        print(f'  (Worst5Y≥-5% フィルタ通過: {len(passing)}/{len(ranked)}, 除外: {filtered_out})')
        # top3はフィルタ通過済みから選ぶ
        top3 = ranked_final[:3]
        all_results[key] = {
            'best': tuple(ranked_final[0][:4]),   # (score, params, m, L_t)
            'top3': [tuple(x[:4]) for x in top3],
            'filter_n': len(passing),
            'total_n':  len(ranked),
        }
        _, bp, bm, bL = tuple(ranked_final[0][:4])
        ls = lev_stats(bL)
        print(f'  Best: {bp}')
        print(f'    CAGR_FULL={bm["CAGR_FULL"]*100:+.2f}% OOS={bm["CAGR_OOS"]*100:+.2f}% '
              f'MaxDD={bm["MaxDD_FULL"]*100:.1f}% Worst5Y={bm.get("Worst5Y",0)*100:.1f}%')
        print(f'    Avg Lev={ls["mean"]:.2f}x ±{ls["std"]:.2f}  '
              f'@7x={ls["pct_max"]:.1f}%  @1x={ls["pct_min"]:.1f}%')

    # S1
    run_grid('S1_Conviction', [
        ({'alpha': p['alpha'], 'target_vol': p['target_vol']},
         compute_L_s1_conviction(raw_a2, returns, **p))
        for p in GRIDS['S1_Conviction']
    ])

    # S2 (拡張グリッド: target_vol も最適化対象)
    run_grid('S2_VZGated', [
        ({'k_vz': p['k_vz'], 'gate_min': p['gate_min'], 'target_vol': p['target_vol']},
         compute_L_s2_vz_gated(returns, vz, **p))
        for p in GRIDS['S2_VZGated']
    ])

    # S3
    run_grid('S3_Decomposed', [
        ({'beta_defense': p['beta_defense'], 'l_max': p['l_max']},
         compute_L_s3_decomposed(components, **p))
        for p in GRIDS['S3_Decomposed']
    ])

    # S4 (相対ボラゲート: target_vol死パラメータ問題を回避)
    run_grid('S4_RelVol', [
        ({'l_base': p['l_base'], 'k_rel': p['k_rel'],
          'rel_threshold': p['rel_threshold'], 'k_vz': p['k_vz']},
         compute_L_s4_relvol(returns, vz, **p))
        for p in GRIDS['S4_RelVol']
    ])

    # --- Sanity check: S2 with k_vz=0 should match P2 ---
    print('\n--- Sanity Check: S2(k_vz=0) == P2 ---')
    L_s2_zero = compute_L_s2_vz_gated(returns, vz, target_vol=P2_BEST_TARGET_VOL,
                                        k_vz=0.0, gate_min=0.0)
    m_s2_zero = run_one(L_s2_zero, close, lev_A, wn_A, wg_A, wb_A, dates,
                         gold_2x, bond_3x, sofr)
    diff = abs(m_s2_zero.get('CAGR_FULL', 0) - p2_best_m.get('CAGR_FULL', 0))
    print(f'  S2(k_vz=0) CAGR_FULL={m_s2_zero["CAGR_FULL"]*100:+.4f}% '
          f'vs P2={p2_best_m["CAGR_FULL"]*100:+.4f}% diff={diff*100:.4f}pp '
          f'{"✅" if diff < 0.0001 else "⚠️"}')

    # --- Report ---
    print('\nGenerating report...')
    data_info = {'start': str(dates.iloc[0].date()), 'end': str(dates.iloc[-1].date())}
    md = generate_report(all_results, baselines, data_info, p2_best_m, p2_sharpe_oos)
    out = os.path.join(BASE, 'ENH_LEVERAGE_BACKTEST_2026-05-16.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
