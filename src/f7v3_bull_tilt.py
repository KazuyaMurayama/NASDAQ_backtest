"""
F7-v3: Bull-Conviction wn Tilt 定式再設計スイープ
==================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

問題診断:
  F7/F7-v2 の現行定式 `tilt × (raw_a2 - 0.15) × (1 - raw_a2)` は
  raw_a2 = 0.575 で peak、ピーク値は `tilt × 0.18`。tilt=0.5 でも
  ピークは 0.09 → cap=0.10 にすら届かず、F7-v2 で cap を 0.15/0.20 に
  上げても結果が変わらなかった真因。

設計:
  定式 A (Large-Tilt Bell): 既存定式のまま、tilt を大きくして cap を飽和
    tilt_amount = clip(tilt × (raw_a2 - THR) × (1 - raw_a2), 0, 0.10)
    tilt ∈ [0.6, 1.0, 2.0, 5.0, 10.0]  cap=0.10 固定

  定式 B (Linear): モノトーン増加で raw_a2 高いほど大きく
    tilt_amount = clip(tilt × (raw_a2 - THR), 0, TILT_CAP_B)   (bull mask のみ)
    tilt × cap = [(0.10, 0.10), (0.10, 0.20), (0.20, 0.10), (0.20, 0.20)]

ベース: E4 採用 config (k_lo=0.1, k_hi=0.8, vz_thr=0.7, LT2-N750, mode B)
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
    build_a2_signal, simulate_rebalance_A,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy, calc_7metrics,
    CFD_SPREAD_LOW, IS_START, IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import build_lt_signal, apply_lt_mode_b
from _sweep_format import MD_HEADER_1P, fmt_row_1p, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# グリッド・定数
# ---------------------------------------------------------------------------
# 定式 A: Large-Tilt Bell (既存定式 + tilt 大値, cap=0.10 固定)
TILT_GRID_A  = [0.6, 1.0, 2.0, 5.0, 10.0]
TILT_CAP_A   = 0.10

# 定式 B: Linear (モノトーン増加)
GRID_B = [
    (0.10, 0.10),
    (0.10, 0.20),
    (0.20, 0.10),
    (0.20, 0.20),
]

# E4 採用 config (k_lo=0.1, k_hi=0.8, vz_thr=0.7) — F7-v3 base
K_LO, K_HI, VZ_THR, K_MID = 0.1, 0.8, 0.7, 0.5

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)

# REF 値 (E4 current best)
REF_CAGR_OOS   = 0.3353
REF_SHARPE_OOS = 0.8915
REF_MAXDD      = -0.6001

# PASS 基準
PASS_SHARPE_DELTA = 0.020
PASS_CAGR_OOS     = 0.295
PASS_GAP          = 0.060
PASS_MAXDD        = -0.6501
PASS_WORST10Y     = 0.150


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


def passes_all(r):
    return (r['Sharpe_OOS'] - REF_SHARPE_OOS >= PASS_SHARPE_DELTA and
            r['CAGR_OOS'] >= PASS_CAGR_OOS and
            r['IS_OOS_gap'] <= PASS_GAP and
            r['MaxDD_FULL'] > PASS_MAXDD and
            r['Worst10Y_star'] >= PASS_WORST10Y)


def count_trades_tilted(wn_tilted, wb_tilted, lev_arr):
    """wn / wb / lev のいずれかが変化した日をリバランス日として数える。"""
    n = len(wn_tilted)
    n_tr = 0
    for i in range(1, n):
        if (wn_tilted[i] != wn_tilted[i-1] or
            wb_tilted[i] != wb_tilted[i-1] or
            lev_arr[i]    != lev_arr[i-1]):
            n_tr += 1
    return n_tr


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('F7-v3: Bull-Conviction wn Tilt 定式再設計スイープ')
    print('=' * 70)
    total_a = len(TILT_GRID_A)
    total_b = len(GRID_B)
    total   = total_a + total_b
    print(f'定式 A (Large-Tilt Bell, cap={TILT_CAP_A}): tilt={TILT_GRID_A}  ({total_a} configs)')
    print(f'定式 B (Linear): (tilt, cap)={GRID_B}  ({total_b} configs)')
    print(f'合計: {total} configs + REF')
    print(f'Base: E4 採用 config (k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, LT2-N750, mode B)')

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
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_trades_yr_ref = n_tr / n_years
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)

    # E4 base lev_mod (全 config 共通)
    lt_sig_raw = build_lt_signal(close, 'LT2', 750)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    k_dyn = np.where(vz_arr > VZ_THR, K_HI,
                     np.where(vz_arr < -VZ_THR, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=lt_sig_raw.index)
    lev_mod = apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)
    lev_mod_arr = np.asarray(lev_mod)

    # raw_a2 を numpy 配列に
    raw_a2_vals = raw_a2.values if hasattr(raw_a2, 'values') else np.asarray(raw_a2)
    bull_mask   = raw_a2_vals > THRESHOLD

    print('Assets and signals built. Starting sweep...')

    # ---------------- REF (tilt = 0.0) ----------------
    nav_ref = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_ref, dates, n_trades_yr_ref)
    m_ref.update({'formula': 'REF', 'tilt': 0.0, 'cap': float('nan'),
                  'Trades_yr': n_trades_yr_ref,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
    print(f'[REF tilt=0.00] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  '
          f'Sharpe={m_ref["Sharpe_OOS"]:+.3f}  MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%  '
          f'Tr/yr={n_trades_yr_ref:.1f}')
    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    print(f'[SANITY] diff vs E4 best CAGR={diff_ref:+.2f}pp → '
          f'{"OK" if abs(diff_ref) <= 0.15 else "WARN"}')

    results = [m_ref]

    # ---------------- 定式 A: Large-Tilt Bell ----------------
    print('\n--- 定式 A: Large-Tilt Bell (cap=0.10 固定) ---')
    idx = 0
    for tilt in TILT_GRID_A:
        idx += 1
        tilt_amount_raw = tilt * (raw_a2_vals - THRESHOLD) * (1.0 - raw_a2_vals)
        tilt_amount = np.where(bull_mask, np.clip(tilt_amount_raw, 0.0, TILT_CAP_A), 0.0)

        wn_tilted = wn_A + tilt_amount
        wb_tilted = np.clip(wb_A - tilt_amount, 0.0, wb_A)

        n_tr_tilt = count_trades_tilted(wn_tilted, wb_tilted, lev_mod_arr)
        trades_yr = n_tr_tilt / n_years

        nav = build_nav_strategy(
            close, lev_mod, wn_tilted, wg_A, wb_tilted, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, trades_yr)
        m.update({'formula': 'A', 'tilt': tilt, 'cap': TILT_CAP_A,
                  'Trades_yr': trades_yr,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
        results.append(m)
        # peak は raw_a2=0.575 で tilt × 0.18 だが、cap=0.10 で頭打ち
        peak_unclipped = tilt * 0.18
        print(f'  A[{idx}/{total_a}] tilt={tilt:>5.1f} (peak_raw={peak_unclipped:.3f}): '
              f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
              f'Sharpe={m["Sharpe_OOS"]:+.3f}  '
              f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
              f'W10Y={m["Worst10Y_star"]*100:+.2f}%  '
              f'Tr/yr={trades_yr:.1f}')

    # ---------------- 定式 B: Linear ----------------
    print('\n--- 定式 B: Linear (モノトーン増加) ---')
    idx = 0
    for tilt, cap in GRID_B:
        idx += 1
        tilt_amount_raw = tilt * (raw_a2_vals - THRESHOLD)
        tilt_amount = np.where(bull_mask, np.clip(tilt_amount_raw, 0.0, cap), 0.0)

        wn_tilted = wn_A + tilt_amount
        wb_tilted = np.clip(wb_A - tilt_amount, 0.0, wb_A)

        n_tr_tilt = count_trades_tilted(wn_tilted, wb_tilted, lev_mod_arr)
        trades_yr = n_tr_tilt / n_years

        nav = build_nav_strategy(
            close, lev_mod, wn_tilted, wg_A, wb_tilted, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, trades_yr)
        m.update({'formula': 'B', 'tilt': tilt, 'cap': cap,
                  'Trades_yr': trades_yr,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
        results.append(m)
        # peak は raw_a2=1.0 で tilt × 0.85, cap で頭打ち
        peak_unclipped = tilt * (1.0 - THRESHOLD)
        print(f'  B[{idx}/{total_b}] tilt={tilt:.2f}/cap={cap:.2f} '
              f'(peak_raw={peak_unclipped:.3f}): '
              f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
              f'Sharpe={m["Sharpe_OOS"]:+.3f}  '
              f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
              f'W10Y={m["Worst10Y_star"]*100:+.2f}%  '
              f'Tr/yr={trades_yr:.1f}')

    print(f'\n全 {len(results)-1} configs 完了。')

    pass_list = [r for r in results[1:] if passes_all(r)]
    best_sharpe = max(results[1:], key=lambda r: r['Sharpe_OOS'])
    verdict = ('PASS' if pass_list else
               'WARN' if best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')
    print(f'PASS configs: {len(pass_list)}')
    if best_sharpe['formula'] == 'A':
        bs_label = f'A:tilt={best_sharpe["tilt"]:.1f}'
    else:
        bs_label = f'B:tilt={best_sharpe["tilt"]:.2f}/cap={best_sharpe["cap"]:.2f}'
    print(f'Best Sharpe: {bs_label} → '
          f'Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  '
          f'CAGR_OOS={best_sharpe["CAGR_OOS"]*100:+.2f}%  '
          f'MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%  '
          f'IS-OOS={best_sharpe["IS_OOS_gap"]*100:+.2f}pp  '
          f'Tr/yr={best_sharpe["Trades_yr"]:.1f}')
    print(f'総合判定: {verdict}')

    # ---------------- CSV ----------------
    pd.DataFrame([{
        'formula': r['formula'],
        'tilt': r['tilt'],
        'cap':  r.get('cap', float('nan')),
        'CAGR_IS':   r['CAGR_IS'],
        'CAGR_OOS':  r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'],
        'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'],
        'P10_5Y':    r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'],
        'Trades_yr': r['Trades_yr'],
    } for r in results]).to_csv(os.path.join(BASE, 'f7v3_bull_tilt_results.csv'),
                                index=False, float_format='%.6f')

    # ---------------- MD ----------------
    hdr1, hdr2 = MD_HEADER_1P
    ref_row = fmt_row_1p('REF (tilt=0.00)', m_ref)
    rows_a = '\n'.join(
        fmt_row_1p(f'A:tilt={r["tilt"]:.1f}', r)
        for r in results[1:1+total_a]
    )
    rows_b = '\n'.join(
        fmt_row_1p(f'B:tilt={r["tilt"]:.2f}/cap={r["cap"]:.2f}', r)
        for r in results[1+total_a:]
    )

    # 考察用に主要数値を抽出
    best_a = max(results[1:1+total_a], key=lambda r: r['Sharpe_OOS'])
    best_b = max(results[1+total_a:], key=lambda r: r['Sharpe_OOS'])
    tryr_ref = m_ref['Trades_yr']
    tryr_a_max = max(r['Trades_yr'] for r in results[1:1+total_a])
    tryr_b_max = max(r['Trades_yr'] for r in results[1+total_a:])

    report = f"""\
# F7-v3: Bull-Conviction wn Tilt 定式再設計スイープ

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

### 問題診断（F7/F7-v2 ふりかえり）
F7/F7-v2 の現行定式 `tilt × (raw_a2 - 0.15) × (1 - raw_a2)` は
raw_a2 = 0.575 で peak、ピーク値は `tilt × 0.18`。tilt=0.5 でも
ピークは 0.09 → cap=0.10 にすら届かない。これが F7-v2 で cap を 0.15/0.20
に上げても結果が完全同一だった真因。

### 定式 A: Large-Tilt Bell（既存定式 + tilt 大値）
```
bull_mask    = raw_a2 > THRESHOLD                                # 0.15
tilt_amount  = tilt × (raw_a2 - 0.15) × (1 - raw_a2)
tilt_amount  = clip(tilt_amount, 0, 0.10)                         # cap 固定
```
| tilt | peak (raw_a2=0.575) | 挙動 |
|---:|---:|:---|
| 0.6 | 0.108 | cap にわずかに届く |
| 1.0 | 0.180 | cap に頻繁に飽和 |
| 2.0 | 0.360 | 大部分の bull 日が cap に張り付く |
| 5.0 | 0.900 | ほぼ全 bull 日が cap |
| 10.0| 1.800 | 完全ステップ関数（bull → +0.10） |

### 定式 B: Linear（モノトーン増加）
```
bull_mask    = raw_a2 > THRESHOLD                                # 0.15
tilt_amount  = tilt × (raw_a2 - 0.15)                            # raw_a2 高いほど大
tilt_amount  = clip(tilt_amount, 0, TILT_CAP_B)
```
| (tilt, cap) | peak (raw_a2=1.0) | 挙動 |
|:---|---:|:---|
| (0.10, 0.10) | 0.085 | cap に届かず実質無 cap |
| (0.10, 0.20) | 0.085 | cap に届かず実質無 cap (= 上と同一) |
| (0.20, 0.10) | 0.170 | raw_a2 > 0.65 で cap |
| (0.20, 0.20) | 0.170 | cap に届かず実質無 cap |

### 共通設定
| 項目 | 定義 |
|------|------|
| **bull mask** | raw_a2 > THRESHOLD ({THRESHOLD}) |
| **wn 調整** | wn_tilted = wn_A + tilt_amount |
| **wb 調整** | wb_tilted = max(wb_A - tilt_amount, 0) |
| **Base config** | E4 採用: k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, LT2-N750, mode B |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**サニティ**: REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (diff vs E4 best CAGR={diff_ref:+.2f}pp)

---

## §2 9指標テーブル

{hdr1}
{hdr2}
{ref_row}
{rows_a}
{rows_b}

{MD_WFA_NOTE}

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ REF+0.020 (≥ {REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}) |
| (ii) CAGR_OOS | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap | ≤ {PASS_GAP*100:.1f}pp |
| (iv) MaxDD | > {PASS_MAXDD*100:.1f}% |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% |

- **PASS configs**: {len(pass_list)} / {total}
- **最高 Sharpe**: {bs_label} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}, CAGR_OOS={best_sharpe["CAGR_OOS"]*100:+.2f}%, Tr/yr={best_sharpe["Trades_yr"]:.1f}
- **定式 A 最高**: tilt={best_a["tilt"]:.1f} → Sharpe={best_a["Sharpe_OOS"]:+.3f}, CAGR_OOS={best_a["CAGR_OOS"]*100:+.2f}%, Tr/yr={best_a["Trades_yr"]:.1f}
- **定式 B 最高**: tilt={best_b["tilt"]:.2f}/cap={best_b["cap"]:.2f} → Sharpe={best_b["Sharpe_OOS"]:+.3f}, CAGR_OOS={best_b["CAGR_OOS"]*100:+.2f}%, Tr/yr={best_b["Trades_yr"]:.1f}
- **総合判定: {verdict}**

---

## §4 考察

### 定式 A vs 定式 B の構造的違い
- **定式 A** はピークが raw_a2=0.575 にある「ベル型」。tilt を大きくすると
  大部分の bull 日が cap=0.10 に張り付き、極端な tilt（例: tilt=10）では
  「bull 日なら一律 +0.10」のステップ関数に収束する。
- **定式 B** は raw_a2 にモノトーン線形。確信度が高い日ほど tilt が大きく
  なる素直な設計だが、(tilt=0.10, cap≥0.10) と (tilt=0.20, cap=0.20) は
  cap に届かない領域があり、実質「無 cap 線形」。

### Trades/yr 比較
| 設計 | Tr/yr (max) | REF 比 |
|:---|---:|---:|
| REF | {tryr_ref:.1f} | 1.00× |
| 定式 A | {tryr_a_max:.1f} | {tryr_a_max/tryr_ref:.2f}× |
| 定式 B | {tryr_b_max:.1f} | {tryr_b_max/tryr_ref:.2f}× |

定式 A（tilt 大）はステップ関数に近づくため、bull → bull 内の連続変化が
減って Trades/yr が REF に近づく可能性。定式 B はモノトーン線形のため
raw_a2 が変動するたびに wn が連続変化し、Trades/yr が増える傾向。

### cap 飽和挙動
- 定式 A: tilt ≥ 1.0 で cap=0.10 が支配的になり、tilt をさらに上げても
  同一挙動に収束。tilt=2.0 と tilt=10.0 がほぼ同一結果なら「cap 飽和」
  の証左。
- 定式 B: (tilt=0.20, cap=0.10) のみ cap が効く。それ以外は実質無 cap。

### F7/F7-v2 との比較
F7-v2 で cap を 0.10→0.15→0.20 と上げても結果が同一だったのは、ピーク
tilt_amount = tilt × 0.18 ≤ tilt × 0.18 < 0.10 (tilt ≤ 0.5) で常に cap に
届かなかったため。F7-v3 定式 A の tilt ≥ 0.6 で初めて cap が制約として効く。

---

*生成スクリプト: `src/f7v3_bull_tilt.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/f7_bull_tilt.py`, `src/f7v2_bull_tilt.py`, `src/e4_regime_klt.py`*
"""
    md_path = os.path.join(BASE, 'F7V3_BULL_TILT_2026-05-24.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "f7v3_bull_tilt_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
