"""
E2: Bond-tilt 拡張スイープ（低 gold_frac 領域）
================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

目的: F1a/B9 で未探索の gold_frac ≤ 0.40（bond 増量）領域 × wn_min 2D を検証。
     Worst10Y★ と MaxDD の改善余地を探る。

gold_frac グリッド : [0.20, 0.25, 0.30, 0.35, 0.40, 0.50 REF]
wn_min グリッド   : [0.20, 0.25, 0.30 REF, 0.35, 0.40]
合計: 6 × 5 = 30 configs

出力:
  - e2_bond_sweep_results.csv
  - E2_BOND_SWEEP_2026-05-24.md
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr, build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_a2_signal,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy, calc_7metrics,
    CFD_SPREAD_LOW, IS_START, IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import build_lt_signal, signal_to_bias, apply_lt_mode_b
from _sweep_format import MD_HEADER_1P, fmt_row_1p, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# グリッド & 定数
# ---------------------------------------------------------------------------
GOLD_FRAC_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]   # 0.50 = REF
WN_MIN_GRID    = [0.20, 0.25, 0.30, 0.35, 0.40]          # 0.30 = REF
REF_GOLD_FRAC  = 0.50
REF_WN_MIN     = 0.30

REF_CAGR_OOS   = 0.3116
REF_SHARPE_OOS = 0.858
REF_MAXDD      = -0.5945
REF_WORST10Y   = 0.1810

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)

# E2 専用 PASS 基準（Worst10Y★改善にフォーカス）
PASS_WORST10Y_IMPROVE = 0.190    # REF +18.10% → 目標 +19.0% 以上
PASS_SHARPE_MIN       = 0.770    # Sharpe は S2 ベースライン以上を維持
PASS_CAGR_OOS         = 0.2700   # CAGR は +27%以上 (REF の -4pp 許容)
PASS_MAXDD            = -0.6445  # MaxDD guardrail
PASS_GAP_MAX          = 0.060    # IS-OOS gap ≤ +6.0pp


def simulate_rebalance_A_wmin(raw, vz, threshold=THRESHOLD,
                               wn_min=0.30, wn_max=0.90, gold_frac=0.50):
    n = len(raw); raw_v = raw.values; vz_v = vz.values
    lev = np.zeros(n); wn = np.zeros(n); wg = np.zeros(n); wb = np.zeros(n)
    cur_lev = raw_v[0]
    cur_wn = float(np.clip(0.55 + 0.25 * cur_lev - 0.10 * max(vz_v[0], 0), wn_min, wn_max))
    cur_wg = (1 - cur_wn) * gold_frac; cur_wb = (1 - cur_wn) * (1 - gold_frac)
    lev[0] = cur_lev; wn[0] = cur_wn; wg[0] = cur_wg; wb[0] = cur_wb
    n_trades = 0
    for i in range(1, n):
        t = raw_v[i]
        if (t == 0 and cur_lev > 0) or (cur_lev == 0 and t > 0) or abs(t - cur_lev) > threshold:
            cur_lev = t
            cur_wn = float(np.clip(0.55 + 0.25 * cur_lev - 0.10 * max(vz_v[i], 0), wn_min, wn_max))
            cur_wg = (1 - cur_wn) * gold_frac; cur_wb = (1 - cur_wn) * (1 - gold_frac)
            n_trades += 1
        lev[i] = cur_lev; wn[i] = cur_wn; wg[i] = cur_wg; wb[i] = cur_wb
    return lev, wn, wg, wb, n_trades


def compute_p10_5y(nav, td=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    return float(((s / s.shift(td * 5)) ** 0.2 - 1).dropna().quantile(0.10))


def calc_all_metrics(nav, dates, trades_yr):
    m = calc_7metrics(nav, dates, trades_per_year=trades_yr)
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, 10)
    return {**m,
            'Worst10Y_star': float(r10.min()) if len(r10) > 0 else float('nan'),
            'P10_5Y':        compute_p10_5y(nav.values),
            'IS_OOS_gap':    m['CAGR_IS'] - m['CAGR_OOS']}


def passes_e2(r):
    return (r['Worst10Y_star']  >= PASS_WORST10Y_IMPROVE and
            r['Sharpe_OOS']     >= PASS_SHARPE_MIN and
            r['CAGR_OOS']       >= PASS_CAGR_OOS and
            r['MaxDD_FULL']     > PASS_MAXDD and
            r['IS_OOS_gap']     <= PASS_GAP_MAX)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('E2: Bond-tilt 拡張スイープ（低 gold_frac 領域）')
    print('=' * 70)
    total = len(GOLD_FRAC_GRID) * len(WN_MIN_GRID)
    print(f'グリッド: gold_frac={GOLD_FRAC_GRID} × wn_min={WN_MIN_GRID}  ({total} configs)')

    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({n:,} days)')

    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)

    lt_sig  = build_lt_signal(close, 'LT2', 750)
    lt_bias = signal_to_bias(lt_sig, 0.5)
    print('Assets and signals built.')

    results = []
    idx = 0
    for gf in GOLD_FRAC_GRID:
        for wn_min in WN_MIN_GRID:
            idx += 1
            is_ref = abs(gf - REF_GOLD_FRAC) < 1e-9 and abs(wn_min - REF_WN_MIN) < 1e-9
            lev_b, wn_b, wg_b, wb_b, n_tr_b = simulate_rebalance_A_wmin(
                raw_a2, vz, THRESHOLD, wn_min=wn_min, wn_max=0.90, gold_frac=gf)
            n_trades_yr = n_tr_b / n_years
            lev_mod = apply_lt_mode_b(lev_b, lt_bias, l_min=0.0, l_max=1.0)
            nav = build_nav_strategy(
                close, lev_mod, wn_b, wg_b, wb_b, dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
            )
            m = calc_all_metrics(nav, dates, n_trades_yr)
            m.update({'gold_frac': gf, 'wn_min': wn_min,
                      'Trades_yr': n_trades_yr,
                      'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
            results.append(m)
            ref_tag = ' [REF]' if is_ref else ''
            print(f'  [{idx:>2d}/{total}] gf={gf:.2f} wn={wn_min:.2f}{ref_tag}: '
                  f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  Sharpe={m["Sharpe_OOS"]:+.3f}  '
                  f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  Worst10Y★={m["Worst10Y_star"]*100:+.2f}%  '
                  f'IS-OOS={m["IS_OOS_gap"]*100:+.2f}pp')

    # Sanity
    r_ref = next(r for r in results
                 if abs(r['gold_frac'] - REF_GOLD_FRAC) < 1e-9
                 and abs(r['wn_min'] - REF_WN_MIN) < 1e-9)
    diff = (r_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    sanity_ok = abs(diff) <= 0.15
    print(f'\n[SANITY] REF CAGR_OOS={r_ref["CAGR_OOS"]*100:+.2f}% (ref +31.16%, diff {diff:+.2f}pp) '
          f'→ {"OK" if sanity_ok else "WARN"}')

    pass_list = [r for r in results
                 if passes_e2(r) and not (abs(r['gold_frac'] - REF_GOLD_FRAC) < 1e-9
                                          and abs(r['wn_min'] - REF_WN_MIN) < 1e-9)]
    best_worst10y = max(results, key=lambda r: r['Worst10Y_star'])
    best_sharpe   = max(results, key=lambda r: r['Sharpe_OOS'])
    verdict = 'PASS' if pass_list else 'WARN' if best_worst10y['Worst10Y_star'] > REF_WORST10Y else 'FAIL'
    print(f'PASS configs: {len(pass_list)}  Best Worst10Y★: gf={best_worst10y["gold_frac"]:.2f} '
          f'wn={best_worst10y["wn_min"]:.2f} → {best_worst10y["Worst10Y_star"]*100:+.2f}%')
    print(f'Best Sharpe:   gf={best_sharpe["gold_frac"]:.2f} wn={best_sharpe["wn_min"]:.2f} '
          f'→ Sharpe {best_sharpe["Sharpe_OOS"]:+.3f}')
    print(f'総合判定: {verdict}')

    # CSV
    pd.DataFrame([{
        'gold_frac': r['gold_frac'], 'wn_min': r['wn_min'],
        'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'], 'Trades_yr': r['Trades_yr'],
    } for r in results]).to_csv(os.path.join(BASE, 'e2_bond_sweep_results.csv'),
                                index=False, float_format='%.6f')

    # MD
    hdr1, hdr2 = MD_HEADER_1P
    rows_md = '\n'.join(
        fmt_row_1p(
            f'gf={r["gold_frac"]:.2f}/wn={r["wn_min"]:.2f}{"  ← REF" if abs(r["gold_frac"]-REF_GOLD_FRAC)<1e-9 and abs(r["wn_min"]-REF_WN_MIN)<1e-9 else ""}',
            r
        ) for r in results
    )
    report = f"""\
# E2: Bond-tilt 拡張スイープ（低 gold_frac 領域）

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

| 項目 | 定義 |
|------|------|
| **gold_frac グリッド** | {GOLD_FRAC_GRID}（REF=0.50） |
| **wn_min グリッド** | {WN_MIN_GRID}（REF=0.30） |
| **合計 configs** | {total}（6×5） |
| **目的** | Bond 増量（低 gold_frac）による Worst10Y★ / MaxDD 改善検証 |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**F1a/B9 との違い**: F1a は gold_frac ∈ [0.20, 0.80] を固定 wn で検証済み。
本スイープは低 gold_frac（bond 重視）× wn_min の 2D 組合せを初めて検証。
Gold_frac ≤ 0.40 では bond_frac ≥ 0.60 となり、Bond 3x の防御機能が増強される。

**サニティ**: REF (gf=0.50, wn=0.30) CAGR_OOS={r_ref["CAGR_OOS"]*100:+.2f}% (diff {diff:+.2f}pp) → {"OK" if sanity_ok else "WARN"}

---

## §2 9指標テーブル（{total} configs）

> gold_frac 降順 = bond 比重が大きい順（bond_frac = 1 − gold_frac）

{hdr1}
{hdr2}
{rows_md}

{MD_WFA_NOTE}

---

## §3 判定

**E2 PASS 基準**（Worst10Y★ 改善フォーカス）:
- Worst10Y★ ≥ {PASS_WORST10Y_IMPROVE*100:.1f}%（REF +18.10% より +{(PASS_WORST10Y_IMPROVE-REF_WORST10Y)*100:.1f}pp 改善）
- Sharpe_OOS ≥ {PASS_SHARPE_MIN:.3f}（S2 ベースライン維持）
- CAGR_OOS ≥ {PASS_CAGR_OOS*100:.1f}%（REF -4pp 許容）
- MaxDD > {PASS_MAXDD*100:.1f}%（guardrail）
- IS-OOS gap ≤ {PASS_GAP_MAX*100:.1f}pp

- **PASS configs**: {len(pass_list)} / {total-1}（REF除く）
- **Worst10Y★ 最高**: gf={best_worst10y["gold_frac"]:.2f}, wn={best_worst10y["wn_min"]:.2f} → {best_worst10y["Worst10Y_star"]*100:+.2f}%（REF比 {(best_worst10y["Worst10Y_star"]-REF_WORST10Y)*100:+.2f}pp）
- **最高 Sharpe**: gf={best_sharpe["gold_frac"]:.2f}, wn={best_sharpe["wn_min"]:.2f} → {best_sharpe["Sharpe_OOS"]:+.3f}
- **総合判定: {verdict}**

---

## §4 考察

gold_frac ≤ 0.40（bond_frac ≥ 0.60）の領域では:
- Bond 3x はデフレ/リスクオフ局面（2000-2002, 2008-2009, 2020年初）で防御的に機能
- ただし 1970s-80s の高金利期・2022年の利上げ局面では Bond がドラッグになる
- Worst10Y★ は「最悪の10年」を示すため、Bond が機能した era と機能しなかった era の両方に影響される

---

*生成スクリプト: `src/e2_bond_sweep.py`*
*参照: `F1_ALLOC_SWEEP_2026-05-21.md`, `B9_S2LT2_GOLDFRAC_SWEEP_2026-05-23.md`, `CURRENT_BEST_STRATEGY.md`*
"""
    md_path = os.path.join(BASE, 'E2_BOND_SWEEP_2026-05-24.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "e2_bond_sweep_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
