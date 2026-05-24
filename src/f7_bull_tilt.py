"""
F7: Bull-Conviction wn Tilt スイープ
====================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

仮説:
  現行 simulate_rebalance_A は raw_a2 > THRESHOLD の全 "bull" 日に対して
  同じ wn (NASDAQ 比率) を割り当てる。確信度の高い bull 日 (raw_a2 ~0.8+) と
  閾値ギリギリの bull 日 (raw_a2 ~0.16) を同じ扱いにしている。
  bull バケット内で raw_a2 に応じて連続的に wn を tilt させることで、
  強い確信期のアップサイドを取りに行く。

メカニズム:
  bull mask:  raw_a2 > THRESHOLD (= 0.15)
  tilt_amount = tilt * (raw_a2 - THRESHOLD) * (1 - raw_a2)     # [THR, 1] の中央でピーク
  tilt_amount = clip(tilt_amount, 0, 0.10)                     # 最大 +10pp に制限
  wn_tilted = wn_A + tilt_amount
  wb_tilted = max(wb_A - tilt_amount, 0)                       # 債券から減らす（マイナス禁止）

グリッド:
  tilt = [0.10, 0.20, 0.30, 0.50] + REF (tilt=0.0)

ベース:
  E4 採用 config (k_lo=0.1, k_hi=0.8, vz_thr=0.7, LT2-N750, mode B)
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
TILT_GRID = [0.10, 0.20, 0.30, 0.50]   # 非REF configs
TILT_CAP  = 0.10                        # 最大 tilt 量

# E4 採用 config (k_lo=0.1, k_hi=0.8, vz_thr=0.7) — F7 base
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
    print('F7: Bull-Conviction wn Tilt スイープ')
    print('=' * 70)
    total = len(TILT_GRID)
    print(f'グリッド: tilt={TILT_GRID}  ({total} configs + REF)')
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
    m_ref.update({'tilt': 0.0,
                  'Trades_yr': n_trades_yr_ref,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
    print(f'[REF tilt=0.00] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  '
          f'Sharpe={m_ref["Sharpe_OOS"]:+.3f}  MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%')
    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    print(f'[SANITY] diff vs E4 best CAGR={diff_ref:+.2f}pp → '
          f'{"OK" if abs(diff_ref) <= 0.15 else "WARN"}')

    results = [m_ref]

    # ---------------- Tilt configs ----------------
    for idx, tilt in enumerate(TILT_GRID, start=1):
        tilt_amount = tilt * (raw_a2_vals - THRESHOLD) * (1.0 - raw_a2_vals)
        tilt_amount = np.where(bull_mask, np.clip(tilt_amount, 0.0, TILT_CAP), 0.0)

        wn_tilted = wn_A + tilt_amount
        wb_tilted = np.clip(wb_A - tilt_amount, 0.0, wb_A)
        # 実際に債券で吸収できなかった分はそのまま wb=0 になる（wn の増加は維持）

        n_tr_tilt = count_trades_tilted(wn_tilted, wb_tilted, lev_mod_arr)
        trades_yr = n_tr_tilt / n_years

        nav = build_nav_strategy(
            close, lev_mod, wn_tilted, wg_A, wb_tilted, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, trades_yr)
        m.update({'tilt': tilt,
                  'Trades_yr': trades_yr,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
        results.append(m)
        print(f'  [{idx}/{total}] tilt={tilt:.2f}: '
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
    print(f'Best Sharpe: tilt={best_sharpe["tilt"]:.2f} → '
          f'Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  '
          f'CAGR_OOS={best_sharpe["CAGR_OOS"]*100:+.2f}%  '
          f'MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%  '
          f'IS-OOS={best_sharpe["IS_OOS_gap"]*100:+.2f}pp')
    print(f'総合判定: {verdict}')

    # ---------------- CSV ----------------
    pd.DataFrame([{
        'tilt': r['tilt'],
        'CAGR_IS':   r['CAGR_IS'],
        'CAGR_OOS':  r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'],
        'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'],
        'P10_5Y':    r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'],
        'Trades_yr': r['Trades_yr'],
    } for r in results]).to_csv(os.path.join(BASE, 'f7_bull_tilt_results.csv'),
                                index=False, float_format='%.6f')

    # ---------------- MD ----------------
    hdr1, hdr2 = MD_HEADER_1P
    ref_row = fmt_row_1p('tilt=0.00 (REF)', m_ref)
    tilt_rows = '\n'.join(
        fmt_row_1p(f'tilt={r["tilt"]:.2f}', r)
        for r in results[1:]
    )

    report = f"""\
# F7: Bull-Conviction wn Tilt スイープ

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

| 項目 | 定義 |
|------|------|
| **tilt グリッド** | {TILT_GRID}（+ REF=0.0） |
| **tilt cap** | +{TILT_CAP:.2f}（wn に加える最大量） |
| **bull mask** | raw_a2 > THRESHOLD ({THRESHOLD}) |
| **Base config** | E4 採用: k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, LT2-N750, mode B |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**メカニズム**:
```
bull_mask    = raw_a2 > THRESHOLD                                   # 0.15
tilt_amount  = tilt × (raw_a2 - THRESHOLD) × (1 - raw_a2)
tilt_amount  = clip(tilt_amount, 0, +0.10)                          # cap
wn_tilted    = wn_A + tilt_amount         (bull 日のみ)
wb_tilted    = max(wb_A - tilt_amount, 0) (bond からの振替, 負禁止)
```
tilt は確信度 (raw_a2 - THRESHOLD) と余力 (1 - raw_a2) の積で、bull バケット
（[0.15, 1.0]）の中央付近 ≈ 0.575 でピーク。

**サニティ**: REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (diff vs E4 best CAGR={diff_ref:+.2f}pp)

---

## §2 9指標テーブル

{hdr1}
{hdr2}
{ref_row}
{tilt_rows}

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
- **最高 Sharpe**: tilt={best_sharpe["tilt"]:.2f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}
- **総合判定: {verdict}**

---

## §4 考察

bull 日内の確信度差分（raw_a2 = 0.16 と raw_a2 = 0.8 を同列に扱わない）を取りに行く設計:
- アップサイドが取れれば CAGR_OOS は上昇、ただし bond クッションを削るため MaxDD は悪化方向
- tilt が大きいほど bull 強気時のレバが効くが、ピークでも +10pp に制限することで暴走を抑制
- 自由度 1（tilt のみ）。過学習リスクは低め
- Trades/yr は wn の連続変化により REF より増加する点に注意

---

*生成スクリプト: `src/f7_bull_tilt.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/e4_regime_klt.py`*
"""
    md_path = os.path.join(BASE, 'F7_BULL_TILT_2026-05-24.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "f7_bull_tilt_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
