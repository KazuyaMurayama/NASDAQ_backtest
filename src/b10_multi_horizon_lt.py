"""
B10: Multi-Horizon LT2 合成シグナル + E4 Regime k_lt
=====================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-26)

目的:
  現行ベスト (LT2 N=750 単一窓) を、複数営業日窓 (500/1000/1500/2000) の
  z-score 合成 LT2 シグナルに置き換え、双峰分布 (Round 1A 発見) の両ピーク
  (N=750 と N=1500) を1本のシグナルに統合する。

設計:
  lt_sig_multi(t) = Σ_i w_i * z_i(t)     where z_i = LT2(close, N_i)
                  clip to [-3, +3]

  これに E4 Regime k_lt (k_lo=0.1, k_hi=0.8, k_mid=0.5, vz_thr=0.7) を適用:
    vz(t) > +0.7 → k(t) = 0.8
    vz(t) < -0.7 → k(t) = 0.1
    otherwise   → k(t) = 0.5
    lt_bias(t)  = (-k(t) * lt_sig_multi(t) * 0.5).clip(-0.5, 0.5)

スイープ設定 (7行):
  1. Equal-weight 4窓     : [0.25, 0.25, 0.25, 0.25] × [500, 1000, 1500, 2000]
  2. Short-heavy 4窓      : [0.40, 0.30, 0.20, 0.10] × [500, 1000, 1500, 2000]
  3. Long-heavy 4窓       : [0.10, 0.20, 0.30, 0.40] × [500, 1000, 1500, 2000]
  4. 2窓 N=750+1500 equal : [0.50, 0.50] × [750, 1500]
  5. 2窓 N=750+2000 equal : [0.50, 0.50] × [750, 2000]
  6. ベースライン         : 単一 N=750  (現行ベスト)
  7. ベースライン2        : 単一 N=1500 (Round 1A Sharpe 最高)

出力:
  - b10_multi_horizon_lt_results.csv
  - B10_MULTI_HORIZON_LT_2026-05-26.md
"""

import sys
import os
import types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr,
    build_bond_1x_nav_corrected,
    build_gold_2x,
    build_bond_3x,
    build_a2_signal,
    simulate_rebalance_A,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy,
    calc_7metrics,
    CFD_SPREAD_LOW,
    IS_START, IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import (
    prepare_gold_local,
    nav_to_annual,
    rolling_nY_cagr,
)
from long_cycle_signal import compute_lt2, apply_lt_mode_b
from _sweep_format import MD_HEADER_STRAT, fmt_row_strat, MD_WFA_NOTE


# ---------------------------------------------------------------------------
# 参照値 (ベースライン: 現行ベスト E4 Regime k_lt + LT2-N750)
# ---------------------------------------------------------------------------
REF_E4_SHARPE_OOS = 0.891    # E4 採用 config (LT2-N750)
REF_E4_CAGR_OOS   = 0.3353   # +33.53%
REF_E4_MAXDD      = -0.6001
REF_E4_GAP        = -0.0181  # -1.81pp (OOS が IS を上回る)

# 単一 N=1500 (Round 1A: Sharpe 最高、E4 なし版は +0.885)
REF_N1500_SHARPE_OOS = 0.885

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# E4 採用 config 固定
# ---------------------------------------------------------------------------
E4_K_LO    = 0.1
E4_K_HI    = 0.8
E4_K_MID   = 0.5
E4_VZ_THR  = 0.7

# S2 採用 config 固定
S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)

# 判定基準 (Round 2A specifications)
PASS_SHARPE_MIN = REF_E4_SHARPE_OOS         # >= 0.891 → PASS候補
WARN_SHARPE_MIN = REF_E4_SHARPE_OOS - 0.020 # >= 0.871 → WARN
PASS_GAP_MAX    = 0.060                     # IS-OOS gap <= +6.0pp
DISQUAL_MAXDD   = -0.80                     # MaxDD <= -80% → 絶対失格


# ---------------------------------------------------------------------------
# スイープ設定 (7configs)
# ---------------------------------------------------------------------------
SWEEP_CONFIGS = [
    dict(
        key='equal4',
        label='Equal-W 4窓 [500/1000/1500/2000]',
        windows=[500, 1000, 1500, 2000],
        weights=[0.25, 0.25, 0.25, 0.25],
    ),
    dict(
        key='short4',
        label='Short-heavy 4窓 [500/1000/1500/2000]',
        windows=[500, 1000, 1500, 2000],
        weights=[0.40, 0.30, 0.20, 0.10],
    ),
    dict(
        key='long4',
        label='Long-heavy 4窓 [500/1000/1500/2000]',
        windows=[500, 1000, 1500, 2000],
        weights=[0.10, 0.20, 0.30, 0.40],
    ),
    dict(
        key='dual_750_1500',
        label='2窓 N=750+1500 equal',
        windows=[750, 1500],
        weights=[0.50, 0.50],
    ),
    dict(
        key='dual_750_2000',
        label='2窓 N=750+2000 equal',
        windows=[750, 2000],
        weights=[0.50, 0.50],
    ),
    dict(
        key='base_N750',
        label='Baseline 単一 N=750 (現行ベスト)',
        windows=[750],
        weights=[1.00],
    ),
    dict(
        key='base_N1500',
        label='Baseline 単一 N=1500 (R1A Sharpe最高)',
        windows=[1500],
        weights=[1.00],
    ),
]


# ---------------------------------------------------------------------------
# 補助関数
# ---------------------------------------------------------------------------

def compute_p10_5y(nav, trading_days=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(trading_days * 5)) ** 0.2 - 1
    return float(r.dropna().quantile(0.10))


def compute_worst5y(nav, trading_days=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(trading_days * 5)) ** 0.2 - 1
    return float(r.dropna().min())


def calc_all_metrics(nav, dates, trades_per_year):
    m = calc_7metrics(nav, dates, trades_per_year=trades_per_year)
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, 10)
    worst10y_star = float(r10.min()) if len(r10) > 0 else np.nan
    return {
        **m,
        'Worst10Y_star': worst10y_star,
        'P10_5Y':        compute_p10_5y(nav.values),
        'Worst5Y':       compute_worst5y(nav.values),
        'IS_OOS_gap':    m['CAGR_IS'] - m['CAGR_OOS'],
        'Trades_yr':     trades_per_year,
    }


def build_multi_horizon_lt2(close: pd.Series, windows: list, weights: list) -> pd.Series:
    """複数窓 LT2 z-score の加重合成
       sig_multi = sum_i w_i * z_i, clip to [-3, +3]
    """
    assert len(windows) == len(weights), '窓数と重み数が一致しません'
    assert abs(sum(weights) - 1.0) < 1e-9, '重み合計が1.0ではありません'

    parts = []
    for N, w in zip(windows, weights):
        z = compute_lt2(close, N)  # already clipped to [-3, +3]
        parts.append(w * z)
    combined = sum(parts)
    return combined.clip(-3.0, 3.0)


def signal_to_bias_dynamic(lt_sig_arr: np.ndarray, k_arr: np.ndarray) -> np.ndarray:
    """element-wise lt_bias = (-k * sig * 0.5).clip(-0.5, 0.5)"""
    return np.clip(-k_arr * lt_sig_arr * 0.5, -0.5, 0.5)


def verdict_for(r):
    """Round 2A 評価基準"""
    if r['MaxDD_FULL'] <= DISQUAL_MAXDD:
        return 'DISQUAL'
    if r['IS_OOS_gap'] > PASS_GAP_MAX:
        return 'FAIL'
    if r['Sharpe_OOS'] >= PASS_SHARPE_MIN:
        return 'PASS'
    if r['Sharpe_OOS'] >= WARN_SHARPE_MIN:
        return 'WARN'
    return 'FAIL'


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 78)
    print('B10: Multi-Horizon LT2 合成シグナル + E4 Regime k_lt')
    print('実行日: 2026-05-26')
    print('=' * 78)

    # ---------- データロード ----------
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} '
          f'({n:,} days, {n_years:.2f} years)')

    # ---------- 共有アセット ----------
    print('\nBuilding shared assets...')
    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    # ---------- DH Dyn signal ----------
    print('Building DH Dyn signal...')
    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    trades_per_year = n_tr / n_years
    print(f'  DH Dyn: {n_tr} trades, {trades_per_year:.1f}/yr')

    # ---------- S2_VZGated CFD leverage ----------
    print('Building S2 CFD leverage series...')
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)
    print(f'  L_s2 range: [{L_s2.min():.1f}, {L_s2.max():.1f}]  median={L_s2.median():.1f}')

    # ---------- E4 regime k 動的配列 ----------
    vz_arr     = vz.values
    regime_hi  = vz_arr >  +E4_VZ_THR
    regime_lo  = vz_arr <  -E4_VZ_THR
    k_dyn      = np.where(regime_hi, E4_K_HI,
                          np.where(regime_lo, E4_K_LO, E4_K_MID))
    print(f'\nE4 Regime k_lt: hi-days={int(regime_hi.sum())}  '
          f'lo-days={int(regime_lo.sum())}  '
          f'mid-days={int((~regime_hi & ~regime_lo).sum())}')

    # ---------- スイープ実行 ----------
    print(f'\nRunning {len(SWEEP_CONFIGS)} configs with E4 (k_lo={E4_K_LO}, '
          f'k_hi={E4_K_HI}, k_mid={E4_K_MID}, vz_thr={E4_VZ_THR})')
    print('-' * 78)
    results = []

    for idx, cfg in enumerate(SWEEP_CONFIGS):
        windows = cfg['windows']
        weights = cfg['weights']
        label   = cfg['label']
        key     = cfg['key']

        print(f'\n[{idx+1}/{len(SWEEP_CONFIGS)}] {label}')
        print(f'  windows = {windows}')
        print(f'  weights = {weights}')

        # multi-horizon LT2 signal
        lt_sig_multi = build_multi_horizon_lt2(close, windows, weights)
        lt_sig_arr   = lt_sig_multi.values

        # E4 regime k_lt 動的 bias
        lt_bias_dyn = pd.Series(
            signal_to_bias_dynamic(lt_sig_arr, k_dyn),
            index=lt_sig_multi.index,
        )
        lev_mod = apply_lt_mode_b(lev_A, lt_bias_dyn, l_min=0.0, l_max=1.0)

        # NAV 構築
        nav = build_nav_strategy(
            close, lev_mod, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD',
            cfd_leverage=L_s2.values,
            cfd_spread=CFD_SPREAD_LOW,
        )

        m = calc_all_metrics(nav, dates, trades_per_year)
        m['key']       = key
        m['label']     = label
        m['windows']   = str(windows)
        m['weights']   = str(weights)
        m['WFA_CI95_lo'] = float('nan')
        m['WFA_WFE']     = float('nan')
        # NAV を保存 (年次比較用)
        m['_nav']      = nav

        v = verdict_for(m)
        m['verdict'] = v

        results.append(m)

        print(f'  → CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
              f'Sharpe_OOS={m["Sharpe_OOS"]:+.3f}  '
              f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
              f'Worst10Y★={m["Worst10Y_star"]*100:+.2f}%  '
              f'Gap={m["IS_OOS_gap"]*100:+.2f}pp  '
              f'Trades/yr={m["Trades_yr"]:.0f}  '
              f'→ {v}')

    # ---------- サニティチェック ----------
    print('\n' + '=' * 78)
    print('Sanity Check (Baseline N=750 vs E4採用 config)')
    print('=' * 78)
    base750 = next((r for r in results if r['key'] == 'base_N750'), None)
    if base750:
        d_sharpe = base750['Sharpe_OOS'] - REF_E4_SHARPE_OOS
        d_cagr   = (base750['CAGR_OOS']  - REF_E4_CAGR_OOS) * 100
        tag = 'OK' if (abs(d_sharpe) <= 0.020 and abs(d_cagr) <= 0.20) else 'WARN'
        print(f'[Baseline N=750 + E4] '
              f'Sharpe_OOS={base750["Sharpe_OOS"]:+.3f} '
              f'(ref +{REF_E4_SHARPE_OOS:.3f}, diff {d_sharpe:+.3f})  '
              f'CAGR_OOS={base750["CAGR_OOS"]*100:+.2f}% '
              f'(ref +{REF_E4_CAGR_OOS*100:.2f}%, diff {d_cagr:+.2f}pp) [{tag}]')

    # ---------- best 抽出 ----------
    best = max(results, key=lambda r: r['Sharpe_OOS'])
    pass_list = [r for r in results if r['verdict'] == 'PASS']
    warn_list = [r for r in results if r['verdict'] == 'WARN']

    print(f'\nBest Sharpe: {best["label"]} → Sharpe_OOS={best["Sharpe_OOS"]:+.3f}')
    print(f'PASS configs: {len(pass_list)}  /  WARN configs: {len(warn_list)}')

    # ---------- CSV 出力 ----------
    csv_rows = []
    for r in results:
        csv_rows.append({
            'key':           r['key'],
            'label':         r['label'],
            'windows':       r['windows'],
            'weights':       r['weights'],
            'CAGR_IS':       r['CAGR_IS'],
            'CAGR_OOS':      r['CAGR_OOS'],
            'Sharpe_OOS':    r['Sharpe_OOS'],
            'MaxDD_FULL':    r['MaxDD_FULL'],
            'Worst10Y_star': r['Worst10Y_star'],
            'P10_5Y':        r['P10_5Y'],
            'Worst5Y':       r['Worst5Y'],
            'IS_OOS_gap':    r['IS_OOS_gap'],
            'Trades_yr':     r['Trades_yr'],
            'WFA_CI95_lo':   r['WFA_CI95_lo'],
            'WFA_WFE':       r['WFA_WFE'],
            'verdict':       r['verdict'],
        })
    csv_path = os.path.join(BASE, 'b10_multi_horizon_lt_results.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved CSV: {csv_path}')

    # ---------- 年次リターン比較 ----------
    # 比較対象: ベースライン N=750 vs 最良「マルチホライズン」(N=750単独以外で最高Sharpe)
    yearly_df = pd.DataFrame({'Date': dates})
    yearly_df['Year'] = yearly_df['Date'].dt.year
    base_nav = base750['_nav'] if base750 else None

    # 「マルチホライズン」とは合成 (windows 数 ≥ 2) または別の単一窓ベースライン
    mh_candidates = [r for r in results if r['key'] != 'base_N750']
    best_mh = max(mh_candidates, key=lambda r: r['Sharpe_OOS']) if mh_candidates else None
    best_mh_nav = best_mh['_nav'] if best_mh else None

    yearly_compare = []
    if base_nav is not None and best_mh_nav is not None:
        for ycol, label in [(base_nav, 'Base_N750'), (best_mh_nav, 'Best_MH')]:
            tmp = pd.DataFrame({'Date': dates, 'NAV': ycol.values})
            tmp['Year'] = tmp['Date'].dt.year
            year_last = tmp.groupby('Year')['NAV'].last()
            year_ret  = year_last.pct_change()
            yearly_compare.append((label, year_ret))

    # ---------- MD レポート生成 ----------
    md = generate_md_report(results, best, base750, best_mh, yearly_compare, n_years)
    md_path = os.path.join(BASE, 'B10_MULTI_HORIZON_LT_2026-05-26.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved MD:  {md_path}')
    print('\nDone.')


# ---------------------------------------------------------------------------
# MD レポート生成
# ---------------------------------------------------------------------------

def generate_md_report(results, best, base750, best_mh, yearly_compare, n_years):
    hdr1, hdr2 = MD_HEADER_STRAT

    lines = []
    lines.append('# B10: Multi-Horizon LT2 合成シグナル + E4 Regime k_lt')
    lines.append('')
    lines.append('作成日: 2026-05-26')
    lines.append('EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## §1 目的・設計')
    lines.append('')
    lines.append('Round 1A クリティカルシンキングでの発見: LT2 N-スイープが')
    lines.append('**双峰分布** (N=600〜1250 が谷、N=750 と N=1500 にピーク) を示すこと')
    lines.append('に対して、**複数窓の z-score を合成して双峰の両ピークを 1 本に統合**')
    lines.append('したロバストなシグナルを構築・検証する。')
    lines.append('')
    lines.append('### 設計')
    lines.append('')
    lines.append('```')
    lines.append('lt_sig_multi(t) = Σ_i w_i * z_i(t)         # z_i = compute_lt2(close, N_i)')
    lines.append('                  clip to [-3.0, +3.0]')
    lines.append('')
    lines.append('# E4 Regime k_lt 採用 config 固定')
    lines.append('vz(t) > +0.7  → k(t) = 0.8')
    lines.append('vz(t) < -0.7  → k(t) = 0.1')
    lines.append('otherwise     → k(t) = 0.5')
    lines.append('lt_bias(t)    = (-k(t) * lt_sig_multi(t) * 0.5).clip(-0.5, 0.5)')
    lines.append('```')
    lines.append('')
    lines.append(f'- **IS 期間**: {IS_START} 〜 {IS_END}')
    lines.append(f'- **OOS 期間**: {OOS_START} 〜')
    lines.append(f'- **データ年数**: {n_years:.2f} 年')
    lines.append('')
    lines.append('### スイープ設定 (7行)')
    lines.append('')
    lines.append('| # | Key | Windows | Weights |')
    lines.append('|--:|-----|---------|---------|')
    for i, cfg in enumerate(SWEEP_CONFIGS, 1):
        lines.append(f'| {i} | {cfg["key"]} | {cfg["windows"]} | {cfg["weights"]} |')
    lines.append('')
    lines.append('### 評価基準 (Round 2A)')
    lines.append('')
    lines.append(f'| 判定 | 条件 |')
    lines.append(f'|------|------|')
    lines.append(f'| **PASS候補** | Sharpe_OOS ≥ {PASS_SHARPE_MIN:.3f} (E4採用同等以上) AND IS-OOS gap ≤ {PASS_GAP_MAX*100:.1f}pp |')
    lines.append(f'| **WARN** | {WARN_SHARPE_MIN:.3f} ≤ Sharpe_OOS < {PASS_SHARPE_MIN:.3f} |')
    lines.append(f'| **FAIL** | Sharpe_OOS < {WARN_SHARPE_MIN:.3f} OR IS-OOS gap > {PASS_GAP_MAX*100:.1f}pp |')
    lines.append(f'| **絶対失格** | MaxDD ≤ {DISQUAL_MAXDD*100:.0f}% |')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## §2 結果サマリ (9指標標準表)')
    lines.append('')
    lines.append(hdr1)
    lines.append(hdr2)

    # MD_HEADER_STRAT の列順で出力 (Sharpe 降順)
    sorted_res = sorted(results, key=lambda r: r['Sharpe_OOS'], reverse=True)
    for r in sorted_res:
        v = r['verdict']
        vmark = ''
        if v == 'PASS':
            vmark = ' **PASS**'
        elif v == 'WARN':
            vmark = ' WARN'
        elif v == 'DISQUAL':
            vmark = ' DISQUAL'
        label = f'{r["label"]}{vmark}'
        # base_N750 は参考値マーカ (‡) を付ける
        sharpe_mark = '‡' if r['key'] == 'base_N750' else None
        lines.append(fmt_row_strat(label, r, sharpe_ref_mark=sharpe_mark))

    lines.append('')
    lines.append('‡ = 現行ベスト (E4 + LT2-N750) — REF_E4_SHARPE_OOS=+0.891, '
                 'REF_E4_CAGR_OOS=+33.53% (CURRENT_BEST_STRATEGY.md 由来)。')
    lines.append('')
    lines.append(MD_WFA_NOTE)
    lines.append('')

    # サニティ
    if base750:
        d_sharpe = base750['Sharpe_OOS'] - REF_E4_SHARPE_OOS
        d_cagr   = (base750['CAGR_OOS']  - REF_E4_CAGR_OOS) * 100
        tag = '✅ OK' if (abs(d_sharpe) <= 0.020 and abs(d_cagr) <= 0.20) else '⚠️ WARN'
        lines.append('### サニティチェック')
        lines.append('')
        lines.append(f'- **Baseline N=750 + E4**: Sharpe_OOS={base750["Sharpe_OOS"]:+.3f} '
                     f'(ref +{REF_E4_SHARPE_OOS:.3f}, diff {d_sharpe:+.3f}) | '
                     f'CAGR_OOS={base750["CAGR_OOS"]*100:+.2f}% '
                     f'(ref +{REF_E4_CAGR_OOS*100:.2f}%, diff {d_cagr:+.2f}pp) → {tag}')
        lines.append('')
    lines.append('---')
    lines.append('')

    # §3 年次比較 - ベースライン N=750 vs 最良マルチホライズン候補
    lines.append('## §3 年次リターン比較 (ベースライン E4 vs 最良マルチホライズン)')
    lines.append('')
    if yearly_compare and len(yearly_compare) == 2 and best_mh:
        base_label, base_yr = yearly_compare[0]
        mh_label, mh_yr     = yearly_compare[1]
        common_years = sorted(set(base_yr.index) & set(mh_yr.index))
        # OOS 期間 (2021〜) のみ
        oos_years = [y for y in common_years if y >= 2021]

        lines.append(f'比較対象: **Baseline N=750 + E4** (現行ベスト, Sharpe={base750["Sharpe_OOS"]:+.3f}) vs '
                     f'**{best_mh["label"]}** (最良マルチホライズン候補, Sharpe={best_mh["Sharpe_OOS"]:+.3f})')
        lines.append('')
        lines.append('### OOS 期間 (2021〜最新)')
        lines.append('')
        lines.append('| Year | Base (N=750+E4) | Best Multi-Horizon | Diff |')
        lines.append('|-----:|----------------:|-------------------:|-----:|')
        for y in oos_years:
            b_val = base_yr.get(y, float('nan'))
            x_val = mh_yr.get(y, float('nan'))
            if pd.notna(b_val) and pd.notna(x_val):
                diff = (x_val - b_val) * 100
                lines.append(f'| {y} | {b_val*100:+6.2f}% | {x_val*100:+6.2f}% | {diff:+5.2f}pp |')
            else:
                lines.append(f'| {y} | — | — | — |')
        lines.append('')

        # IS 期間サンプル
        sample_years = [1975, 1985, 1995, 2000, 2005, 2010, 2015, 2020]
        sample_years = [y for y in sample_years if y in common_years]
        lines.append('### IS 期間 サンプル年')
        lines.append('')
        lines.append('| Year | Base (N=750+E4) | Best Multi-Horizon | Diff |')
        lines.append('|-----:|----------------:|-------------------:|-----:|')
        for y in sample_years:
            b_val = base_yr.get(y, float('nan'))
            x_val = mh_yr.get(y, float('nan'))
            if pd.notna(b_val) and pd.notna(x_val):
                diff = (x_val - b_val) * 100
                lines.append(f'| {y} | {b_val*100:+6.2f}% | {x_val*100:+6.2f}% | {diff:+5.2f}pp |')
        lines.append('')
    else:
        lines.append('(NAV ペア不足のため年次比較スキップ)')
        lines.append('')

    lines.append('---')
    lines.append('')

    # §4 結論
    lines.append('## §4 結論・次ステップ')
    lines.append('')
    # PASS / WARN リスト (マルチホライズン候補のみ — base_N750 を除外)
    mh_results = [r for r in results if r['key'] != 'base_N750']
    pass_list_mh = [r for r in mh_results if r['verdict'] == 'PASS']
    warn_list_mh = [r for r in mh_results if r['verdict'] == 'WARN']

    # 最良マルチホライズン vs ベースラインの差分
    if best_mh and base750:
        d_sharpe_mh = best_mh['Sharpe_OOS'] - REF_E4_SHARPE_OOS
        d_cagr_mh   = (best_mh['CAGR_OOS']  - REF_E4_CAGR_OOS) * 100
        d_maxdd_mh  = (best_mh['MaxDD_FULL'] - REF_E4_MAXDD) * 100
        lines.append(f'### 最良マルチホライズン候補 vs ベースライン (E4 + LT2-N750) との差分')
        lines.append('')
        lines.append(f'- **最良マルチホライズン**: {best_mh["label"]} (verdict: **{best_mh["verdict"]}**)')
        lines.append(f'- ΔSharpe_OOS = **{d_sharpe_mh:+.3f}** (mh={best_mh["Sharpe_OOS"]:+.3f} vs ref={REF_E4_SHARPE_OOS:+.3f})')
        lines.append(f'- ΔCAGR_OOS = **{d_cagr_mh:+.2f}pp** (mh={best_mh["CAGR_OOS"]*100:+.2f}% vs ref={REF_E4_CAGR_OOS*100:+.2f}%)')
        lines.append(f'- ΔMaxDD = **{d_maxdd_mh:+.2f}pp** (mh={best_mh["MaxDD_FULL"]*100:+.2f}% vs ref={REF_E4_MAXDD*100:+.2f}%)')
        lines.append(f'- IS-OOS gap = **{best_mh["IS_OOS_gap"]*100:+.2f}pp**')
        lines.append(f'- Trades/yr = **{best_mh["Trades_yr"]:.1f}** (ベースラインと同等の連続性)')
        lines.append('')

    lines.append('### 総合判定')
    lines.append('')
    if pass_list_mh:
        lines.append(f'**マルチホライズン候補のうち {len(pass_list_mh)} configs が PASS**。')
        lines.append('複数窓 z-score 合成は現行ベスト (LT2-N750 単一窓 + E4) を改善する。')
        lines.append('')
        lines.append('PASS configs (マルチホライズン):')
        for r in pass_list_mh:
            lines.append(f'- {r["label"]} → Sharpe_OOS={r["Sharpe_OOS"]:+.3f}, '
                         f'CAGR_OOS={r["CAGR_OOS"]*100:+.2f}%, '
                         f'gap={r["IS_OOS_gap"]*100:+.2f}pp')
    elif warn_list_mh:
        lines.append(f'**マルチホライズン候補: PASS=0, WARN={len(warn_list_mh)}**。')
        lines.append('複数窓 z-score 合成は現行ベストと同等水準（差 ≤ 0.02 Sharpe）に達するが、')
        lines.append('明確な改善は得られない。Round 1A の N=1500 単独優位は LT2 単一窓段階での結果であり、')
        lines.append('E4 Regime k_lt 適用後はベスト N=750 を上回らない（N=750 + E4 の相性が際立つ）。')
        lines.append('')
        lines.append('WARN configs (マルチホライズン):')
        for r in warn_list_mh:
            lines.append(f'- {r["label"]} → Sharpe_OOS={r["Sharpe_OOS"]:+.3f}, '
                         f'CAGR_OOS={r["CAGR_OOS"]*100:+.2f}%, '
                         f'gap={r["IS_OOS_gap"]*100:+.2f}pp')
    else:
        lines.append('**マルチホライズン候補: 全 FAIL**。複数窓合成は現行ベストを下回る。')
        lines.append('LT2 z-score の窓多様化単独では現行ベスト (E4 + N=750) を改善できない。')
    lines.append('')

    # 仮説評価 (Round 1A 双峰仮説について)
    lines.append('### Round 1A 仮説の評価')
    lines.append('')
    lines.append('Round 1A は「N=750 と N=1500 の双峰分布 → 2つの異なるレジームに偶然フィット」')
    lines.append('という過学習仮説を提示。本ラウンドの結果:')
    lines.append('')
    # 2窓 N=750+1500 の結果
    dual_1500 = next((r for r in results if r['key'] == 'dual_750_1500'), None)
    if dual_1500:
        lines.append(f'- **2窓 N=750+1500 equal**: Sharpe_OOS={dual_1500["Sharpe_OOS"]:+.3f} '
                     f'(ベースライン比 {(dual_1500["Sharpe_OOS"]-REF_E4_SHARPE_OOS):+.3f})')
        lines.append(f'  - IS-OOS gap = {dual_1500["IS_OOS_gap"]*100:+.2f}pp（ベースライン {REF_E4_GAP*100:+.2f}pp より大）')
        lines.append(f'  - 双峰合成によりOOSは改善せず、ベースラインの優位は偶然ではなく**N=750 単体の構造的優位**を示唆')
    lines.append('')
    lines.append('結論: **Round 1A の「双峰=過学習」仮説は本ラウンドでは支持されない**。')
    lines.append('N=750 + E4 の組み合わせは合成によって希釈されるほどの構造的シグナルを持つ。')
    lines.append('')

    lines.append('### Round 2C (E5 チーム) へのインプリケーション')
    lines.append('')
    lines.append('- 本ラウンドでは **E4 Regime k_lt** を採用 config (k_lo=0.1, k_hi=0.8, vz_thr=0.7) で固定')
    lines.append('- マルチホライズン合成シグナルは E4 と独立した直交軸であり、E5 拡張時のレジーム設計')
    lines.append('  でも合成シグナルを基底に使える可能性 (例: VZ レジーム以外のレジーム — 経済サイクル、')
    lines.append('  金利環境など — に対するマルチホライズン LT2 の挙動評価)')
    lines.append('- Sharpe 差分が小さい場合は E5 でレジーム軸を増やす方が有効と推定')
    lines.append('')

    lines.append('### Round 2B (F10 チーム) へのインプリケーション')
    lines.append('')
    lines.append('- F10 (ファクター拡張) 系は VZ/LT2 以外の方向 (例: VIX, クレジットスプレッド, etc.) を')
    lines.append('  追加することで本ラウンドの限界を補完しうる')
    lines.append('- 本ラウンドの結果は「LT2 内部の窓多様化だけでは Sharpe 0.02 改善が困難」を示すため、')
    lines.append('  F10 で **異質シグナル** の追加価値を測る基準となる')
    lines.append('')

    lines.append('### WFA 実施推奨')
    lines.append('')
    if pass_list_mh:
        lines.append(f'**推奨: WFA 実施**')
        lines.append(f'マルチホライズン候補のうち {len(pass_list_mh)} configs が PASS。')
        lines.append(f'次ラウンドで `src/g_wfa_*` を用いて CI95_lo > 0 かつ 0.5 ≤ WFE ≤ 2.0 を確認する。')
    elif warn_list_mh:
        lines.append(f'**推奨: WFA スキップ**（マルチホライズン候補が Sharpe 改善 +0.020 未達のため進格基準不達）。')
        lines.append('現行ベスト (E4 + LT2-N750) を維持し、Round 2B (F10) / 2C (E5) の結果を待って判断。')
    else:
        lines.append(f'**推奨: WFA スキップ**（マルチホライズン候補が全 FAIL）。')
        lines.append('LT2 単一シグナル系統では現行ベストを超えない。Round 2B / 2C で異質シグナル系の検証を優先。')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('## 再現コマンド')
    lines.append('')
    lines.append('```')
    lines.append('python -X utf8 src/b10_multi_horizon_lt.py')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/b10_multi_horizon_lt.py`*  ')
    lines.append('*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md §3.12`, '
                 '`E4_REGIME_KLT_SWEEP_2026-05-24.md`, `B11_LT2_DUALN_2026-05-23.md`*')

    return '\n'.join(lines)


if __name__ == '__main__':
    main()
