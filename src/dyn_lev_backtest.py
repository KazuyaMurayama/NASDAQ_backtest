"""
Dynamic Leverage Backtest: P1-P5 (2026-05-15)
==============================================
NASDAQのCFD取引で1〜7倍の動的レバレッジ戦略をバックテスト。

戦略:
  P1: SOFR適応型        — SOFRが低いときだけ高レバ
  P2: ボラ・ターゲティング型 — 実現ボラに反比例
  P3: モメンタム比例型   — 短期トレンド強度に比例
  P4: 複合スコア型      — SOFR×ボラ×モメンタム乗算（本命）
  P5: Kelly近似型       — L* = safety × μ_excess / σ²

評価:
  - IS (1974-2021-05-07): パラメータ最適化
  - OOS (2021-05-08-2026-03-26): 過学習チェック
  - 評価関数: 0.6×CAGR_IS + 0.4×Sharpe_IS/3.0
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
    compute_L_sofr_adaptive,
    compute_L_vol_target,
    compute_L_momentum,
    compute_L_composite,
    compute_L_kelly,
)

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

L_MIN = 1.0
L_MAX = 7.0
STEP  = 0.5


# ---------------------------------------------------------------------------
# Parameter grids (IS only; ≤12 combinations per strategy)
# ---------------------------------------------------------------------------

GRIDS = {
    'P1_SOFR': [
        {'sofr_high': sh} for sh in [0.04, 0.06, 0.08, 0.10, 0.15]
    ],
    'P2_Vol': [
        {'target_vol': tv} for tv in [0.50, 0.60, 0.70, 0.80]
    ],
    'P3_Mom': [
        {'m': m, 'k': k}
        for m, k in itertools.product([10, 20, 60], [0.5, 1.0, 1.5])
    ],
    'P4_Composite': [
        {'sofr_high': sh, 'target_vol': tv}
        for sh, tv in itertools.product([0.06, 0.08, 0.10], [0.50, 0.60, 0.70])
    ],
    'P5_Kelly': [
        {'safety': s} for s in [0.3, 0.5, 0.7]
    ],
}


def is_score(metrics: dict) -> float:
    cagr = metrics.get('CAGR_IS', -999)
    sh   = metrics.get('Sharpe_IS', 0)
    if np.isnan(cagr) or np.isnan(sh):
        return -999.0
    return 0.6 * cagr + 0.4 * sh / 3.0


def run_one(L_t, close, lev_A, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr) -> dict:
    nav = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr, 'CFD', np.asarray(L_t), CFD_SPREAD_LOW,
    )
    return calc_7metrics(nav, dates)


def grid_search(strategy_fn, param_grid,
                close, returns, sofr, lev_A, wn_A, wg_A, wb_A,
                dates, gold_2x, bond_3x) -> list:
    """IS最適化。結果を (IS score, params, metrics) のリストで返す。"""
    results = []
    for params in param_grid:
        try:
            L_t = strategy_fn(close=close, returns=returns, sofr_daily=sofr, **params)
        except TypeError:
            # P1はclose/returnsを取らない等、引数が異なるため個別対応
            L_t = _call_strategy(strategy_fn, params, close, returns, sofr)
        m = run_one(L_t, close, lev_A, wn_A, wg_A, wb_A, dates,
                    gold_2x, bond_3x, sofr)
        score = is_score(m)
        results.append((score, params, m, L_t))
    results.sort(key=lambda x: x[0], reverse=True)
    return results


def _call_strategy(fn, params, close, returns, sofr):
    """戦略関数ごとに引数が違うため振り分け。"""
    name = fn.__name__
    if name == 'compute_L_sofr_adaptive':
        return fn(sofr_daily=sofr, **params)
    elif name == 'compute_L_vol_target':
        return fn(returns=returns, **params)
    elif name == 'compute_L_momentum':
        return fn(close=close, **params)
    elif name == 'compute_L_composite':
        return fn(close=close, returns=returns, sofr_daily=sofr, **params)
    elif name == 'compute_L_kelly':
        return fn(returns=returns, sofr_daily=sofr, **params)
    raise ValueError(f"Unknown strategy: {name}")


def lev_stats(L_t, dates: pd.Series) -> dict:
    """平均・パーセンタイル・最大/最低滞在比率"""
    vals = np.asarray(L_t)
    return {
        'mean':  float(np.mean(vals)),
        'p25':   float(np.percentile(vals, 25)),
        'p50':   float(np.percentile(vals, 50)),
        'p75':   float(np.percentile(vals, 75)),
        'pct_max': float((vals >= L_MAX).mean() * 100),
        'pct_min': float((vals <= L_MIN).mean() * 100),
    }


def decade_avg_lev(L_t, dates: pd.Series) -> dict:
    """年代別平均レバレッジ"""
    df = pd.DataFrame({'L': np.asarray(L_t), 'dt': dates.values})
    df['decade'] = (pd.to_datetime(df['dt']).dt.year // 10 * 10).astype(str) + 's'
    return df.groupby('decade')['L'].mean().round(2).to_dict()


def fmt(v, pct=True, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    if pct:
        return f'{v*100:+.{decimals}f}%'
    return f'{v:.{decimals}f}'


def generate_report(all_results: dict, baselines: dict,
                    data_info: dict, all_lev_stats: dict,
                    all_decade: dict) -> str:
    lines = []
    lines.append('# NASDAQ CFD 動的レバレッジ戦略 バックテスト (1x-7x)')
    lines.append('')
    lines.append('作成日: 2026-05-15')
    lines.append('最終更新日: 2026-05-15')
    lines.append('')
    lines.append(f'**データ期間**: {data_info["start"]} 〜 {data_info["end"]}')
    lines.append(f'**IS期間**: {FULL_START} 〜 {IS_END}')
    lines.append(f'**OOS期間**: {OOS_START} 〜 {data_info["end"]}')
    lines.append('')
    lines.append('---')
    lines.append('')

    # 戦略説明
    lines.append('## 1. 戦略概要')
    lines.append('')
    lines.append('| 戦略 | 説明 | パラメータ |')
    lines.append('|------|------|-----------|')
    descs = {
        'P1_SOFR':     ('SOFR適応型',        'sofr_high (SOFR閾値)'),
        'P2_Vol':      ('ボラ・ターゲティング型', 'target_vol (年率ボラ目標)'),
        'P3_Mom':      ('モメンタム比例型',    'm (ルックバック日数), k (シグモイド傾き)'),
        'P4_Composite':('複合スコア型',        'sofr_high, target_vol'),
        'P5_Kelly':    ('Kelly近似型',         'safety (Kelly分率)'),
    }
    for key, (name, params) in descs.items():
        lines.append(f'| {key}: {name} | — | {params} |')
    lines.append('')
    lines.append('---')
    lines.append('')

    # IS最適パラメータ
    lines.append('## 2. IS最適パラメータ')
    lines.append('')
    lines.append('| 戦略 | 最適パラメータ | ISスコア (0.6×CAGR+0.4×Sharpe/3) |')
    lines.append('|------|-------------|----------------------------------|')
    for key, res in all_results.items():
        best_score, best_params, best_m, _ = res['best']
        params_str = ', '.join(f'{k}={v}' for k, v in best_params.items())
        lines.append(f'| {key} | {params_str} | {best_score:.4f} |')
    lines.append('')
    lines.append('---')
    lines.append('')

    # メイン結果テーブル
    lines.append('## 3. メイン結果（IS最適パラメータで評価）')
    lines.append('')
    hdr = ('| 戦略 | CAGR(FULL) | CAGR(IS) | CAGR(OOS) '
           '| Sharpe(FULL) | Sharpe(OOS) | MaxDD | Worst5Y | Worst10Y '
           '| 平均Lev | Lev=7x% | Lev=1x% |')
    sep = '|' + '---|' * (hdr.count('|') - 1)
    lines.append(hdr)
    lines.append(sep)

    # baselines
    for bname, bm in baselines.items():
        row = (f'| **{bname}** '
               f'| {fmt(bm.get("CAGR_FULL"))} '
               f'| {fmt(bm.get("CAGR_IS"))} '
               f'| {fmt(bm.get("CAGR_OOS"))} '
               f'| {fmt(bm.get("Sharpe_FULL"), pct=False)} '
               f'| {fmt(bm.get("Sharpe_OOS"), pct=False)} '
               f'| {fmt(bm.get("MaxDD_FULL"))} '
               f'| {fmt(bm.get("Worst5Y"))} '
               f'| {fmt(bm.get("Worst10Y"))} '
               f'| — | — | — |')
        lines.append(row)

    for key, res in all_results.items():
        _, _, m, L_t = res['best']
        ls = all_lev_stats[key]
        row = (f'| **{key}** '
               f'| {fmt(m.get("CAGR_FULL"))} '
               f'| {fmt(m.get("CAGR_IS"))} '
               f'| {fmt(m.get("CAGR_OOS"))} '
               f'| {fmt(m.get("Sharpe_FULL"), pct=False)} '
               f'| {fmt(m.get("Sharpe_OOS"), pct=False)} '
               f'| {fmt(m.get("MaxDD_FULL"))} '
               f'| {fmt(m.get("Worst5Y"))} '
               f'| {fmt(m.get("Worst10Y"))} '
               f'| {ls["mean"]:.2f}x '
               f'| {ls["pct_max"]:.1f}% '
               f'| {ls["pct_min"]:.1f}% |')
        lines.append(row)
    lines.append('')
    lines.append('---')
    lines.append('')

    # パラメータ感応度
    lines.append('## 4. パラメータ感応度（IS上位3パターン vs OOS）')
    lines.append('')
    for key, res in all_results.items():
        lines.append(f'### {key}')
        lines.append('')
        lines.append('| ランク | パラメータ | ISスコア | CAGR(IS) | CAGR(OOS) | Sharpe(OOS) |')
        lines.append('|--------|-----------|---------|---------|---------|------------|')
        for rank, (score, params, m, _) in enumerate(res['top3'], 1):
            params_str = ', '.join(f'{k}={v}' for k, v in params.items())
            lines.append(
                f'| {rank} | {params_str} '
                f'| {score:.4f} '
                f'| {fmt(m.get("CAGR_IS"))} '
                f'| {fmt(m.get("CAGR_OOS"))} '
                f'| {fmt(m.get("Sharpe_OOS"), pct=False)} |'
            )
        lines.append('')

    lines.append('---')
    lines.append('')

    # 年代別平均レバレッジ
    lines.append('## 5. 年代別平均レバレッジ（経済的妥当性確認）')
    lines.append('')
    decades = sorted(set().union(*[set(d.keys()) for d in all_decade.values()]))
    hdr2 = '| 年代 | ' + ' | '.join(all_results.keys()) + ' |'
    sep2 = '|-----|' + ':---:|' * len(all_results)
    lines.append(hdr2)
    lines.append(sep2)
    for dec in decades:
        cells = [f'{all_decade[k].get(dec, "—")}x' if dec in all_decade[k] else '—'
                 for k in all_results]
        lines.append(f'| {dec} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('> 1970-1980年代（高金利期）に低レバ傾向があれば経済的に妥当。')
    lines.append('')
    lines.append('---')
    lines.append('')

    # リスク注意事項
    lines.append('## 6. リスク注意事項')
    lines.append('')
    lines.append('- **動的レバレッジの実運用**: ポジションサイズ変更のたびに証拠金調整が必要。')
    lines.append('  CFDの最大個人レバレッジは10倍のため L_max=7x は規制内。')
    lines.append('- **SOFR proxy の限界**: DTB3 (3M T-bill) は SOFR の近似値。')
    lines.append('  1980年代のFFレートとの乖離により、P1/P4/P5のIS期間性能に楽観バイアスがある可能性。')
    lines.append('- **IS-OOS乖離チェック**: OOS CAGR が IS CAGR の 60% を下回る戦略は過学習の疑い。')
    lines.append('- **CURRENT_BEST_STRATEGY.md**: OOS CAGR が固定 CFD 3x を上回り、かつ')
    lines.append('  OOS MaxDD が -50% 以内、パラメータ感応度 ±2pp 以内の場合のみ更新を提案。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/dyn_lev_backtest.py`*')
    lines.append(f'*データ期間: {data_info["start"]} 〜 {data_info["end"]}*')

    return '\n'.join(lines)


def main():
    print('=' * 70)
    print('Dynamic Leverage Backtest: P1-P5 (1x-7x NASDAQ CFD)')
    print('=' * 70)

    df    = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    returns = close.pct_change().fillna(0)
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df):,} days)')

    print('Loading SOFR...')
    sofr = load_sofr(dates)

    print('Building bond/gold...')
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, True)
    gold_1x = prepare_gold_data(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)

    print('Building DH Dyn signal...')
    raw, vz = build_a2_signal(close, returns)
    lev_A, wn_A, wg_A, wb_A, _ = simulate_rebalance_A(raw, vz, THRESHOLD)

    # --- Baselines ---
    print('\nBuilding baselines...')
    baselines = {}
    for lev, bname in [(3.0, 'CFD 3x [固定]'), (5.0, 'CFD 5x [固定]'), (7.0, 'CFD 7x [固定]')]:
        nav = build_nav_strategy(close, lev_A, wn_A, wg_A, wb_A, dates,
                                  gold_2x, bond_3x, sofr, 'CFD', lev, CFD_SPREAD_LOW)
        baselines[bname] = calc_7metrics(nav, dates)
        print(f'  {bname}: CAGR={baselines[bname]["CAGR_FULL"]*100:+.2f}%')

    # --- Strategy functions ---
    strategy_fns = {
        'P1_SOFR':      compute_L_sofr_adaptive,
        'P2_Vol':       compute_L_vol_target,
        'P3_Mom':       compute_L_momentum,
        'P4_Composite': compute_L_composite,
        'P5_Kelly':     compute_L_kelly,
    }

    all_results   = {}
    all_lev_stats = {}
    all_decade    = {}

    for key, fn in strategy_fns.items():
        print(f'\n--- {key} ---')
        param_grid = GRIDS[key]
        print(f'  Grid: {len(param_grid)} combinations')

        ranked = []
        for params in param_grid:
            L_t = _call_strategy(fn, params, close, returns, sofr)
            m   = run_one(L_t, close, lev_A, wn_A, wg_A, wb_A, dates,
                          gold_2x, bond_3x, sofr)
            score = is_score(m)
            ranked.append((score, params, m, L_t))
            print(f'  params={params} → IS score={score:.4f} '
                  f'CAGR_IS={m.get("CAGR_IS",float("nan"))*100:+.2f}% '
                  f'Sharpe_IS={m.get("Sharpe_IS",float("nan")):.3f}')

        ranked.sort(key=lambda x: x[0], reverse=True)
        best = ranked[0]
        top3 = ranked[:3]

        _, bp, bm, bL = best
        print(f'  Best: {bp} → CAGR_FULL={bm.get("CAGR_FULL",0)*100:+.2f}% '
              f'CAGR_OOS={bm.get("CAGR_OOS",0)*100:+.2f}% '
              f'MaxDD={bm.get("MaxDD_FULL",0)*100:.1f}%')

        all_results[key]   = {'best': best, 'top3': top3}
        all_lev_stats[key] = lev_stats(bL, dates)
        all_decade[key]    = decade_avg_lev(bL, dates)

        ls = all_lev_stats[key]
        print(f'  Avg Lev: {ls["mean"]:.2f}x  '
              f'p25={ls["p25"]:.1f}x p50={ls["p50"]:.1f}x p75={ls["p75"]:.1f}x  '
              f'@7x: {ls["pct_max"]:.1f}%  @1x: {ls["pct_min"]:.1f}%')

    # --- Sanity check ---
    print('\n--- Sanity Check (固定3x再現) ---')
    L_const3 = pd.Series(np.full(len(close), 3.0), index=close.index)
    m_const3 = run_one(L_const3, close, lev_A, wn_A, wg_A, wb_A, dates,
                        gold_2x, bond_3x, sofr)
    expected_3x = baselines['CFD 3x [固定]']['CAGR_FULL']
    actual_3x   = m_const3.get('CAGR_FULL', float('nan'))
    ok = abs(actual_3x - expected_3x) < 0.0001
    print(f'  固定L=3.0 via dynamic path: {actual_3x*100:+.4f}% '
          f'(期待: {expected_3x*100:+.4f}%) {"✅" if ok else "⚠️ 要確認"}')

    # --- Report ---
    print('\nGenerating report...')
    data_info = {'start': str(dates.iloc[0].date()), 'end': str(dates.iloc[-1].date())}
    md = generate_report(all_results, baselines, data_info, all_lev_stats, all_decade)
    out = os.path.join(BASE, 'DYN_LEVERAGE_BACKTEST_2026-05-15.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
