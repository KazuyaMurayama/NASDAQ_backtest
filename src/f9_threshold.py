"""
F9: Bull-Tilt THRESHOLD 最適化スイープ
=====================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

目的:
  F7-v3 で PASS した「E4 + step-func bull-tilt」(tilt=10.0, cap=0.10) の
  raw_a2 閾値 (THRESHOLD=0.15) を系統的に最適化する。

設計:
  Tilt 定式 (定式 A の極端版 = step function):
    bull_mask    = raw_a2 > THRESHOLD
    tilt_amount  = clip(TILT_STEP × (raw_a2 - THRESHOLD) × (1 - raw_a2), 0, TILT_CAP)
                   (TILT_STEP = 10.0  →  ほぼ完全ステップ関数)
    wn_tilted    = wn_A + tilt_amount
    wb_tilted    = clip(wb_A - tilt_amount, 0, wb_A)

  グリッド: THRESHOLD ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40}
    - T015 (0.15) が F7-v3 A:tilt=10 と同一（ベースライン）
    - <0.15 はより多くの日に tilt 適用（積極的）
    - >0.15 は高確信日のみ tilt 適用（選択的）

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
    DATA_PATH, TRADING_DAYS, THRESHOLD as BASE_THRESHOLD,
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
# F7-v3 で PASS した step-func 定式 (A:tilt=10.0, cap=0.10) を継承
TILT_STEP = 10.0
TILT_CAP  = 0.10

# THRESHOLD グリッド
THR_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

# config_id ラベル
def thr_label(thr):
    return f'T{int(round(thr * 100)):03d}'

# E4 採用 config (k_lo=0.1, k_hi=0.8, vz_thr=0.7) — F7-v3 base 継承
K_LO, K_HI, VZ_THR, K_MID = 0.1, 0.8, 0.7, 0.5

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)

# REF 値 (E4 current best, tilt なし)
REF_CAGR_OOS   = 0.3353
REF_SHARPE_OOS = 0.8915
REF_MAXDD      = -0.6001

# F7-v3 A:tilt=10 (T015 と同一) ベースライン Sharpe (参考値)
T015_SHARPE_REF = 0.9293

# PASS 基準
PASS_SHARPE_DELTA = 0.020   # REF + 0.020 = 0.9115
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
    print('F9: Bull-Tilt THRESHOLD 最適化スイープ')
    print('=' * 70)
    print(f'Tilt 定式: step-func (TILT_STEP={TILT_STEP}, TILT_CAP={TILT_CAP})')
    print(f'THRESHOLD grid: {THR_GRID}  ({len(THR_GRID)} configs + REF)')
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

    # raw_a2 / vz / lev / wn_A / wb_A は signal 構築 (THRESHOLD=BASE_THRESHOLD=0.15 を使う)
    # NOTE: simulate_rebalance_A の THRESHOLD は wn/wg/wb 構築の bull/bear 判定に使用される。
    #       F9 で変化させるのは「tilt 適用の bull mask」のみ。元の wn_A/wg_A/wb_A は固定。
    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, BASE_THRESHOLD)
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
    n_all = len(raw_a2_vals)

    print('Assets and signals built. Starting sweep...')

    # ---------------- REF (tilt なし) ----------------
    nav_ref = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_ref, dates, n_trades_yr_ref)
    m_ref.update({'config_id': 'REF', 'threshold': float('nan'),
                  'bull_days': int(np.nan_to_num((raw_a2_vals > BASE_THRESHOLD).sum())),
                  'bull_ratio': float((raw_a2_vals > BASE_THRESHOLD).sum() / n_all),
                  'Trades_yr': n_trades_yr_ref,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
    print(f'[REF (no tilt)] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  '
          f'Sharpe={m_ref["Sharpe_OOS"]:+.3f}  MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%  '
          f'Tr/yr={n_trades_yr_ref:.1f}')
    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    print(f'[SANITY] diff vs E4 best CAGR={diff_ref:+.2f}pp → '
          f'{"OK" if abs(diff_ref) <= 0.15 else "WARN"}')

    results = [m_ref]

    # ---------------- THRESHOLD グリッド (step-func tilt) ----------------
    print('\n--- step-func bull-tilt (TILT_STEP=10.0, TILT_CAP=0.10) ---')
    idx = 0
    for thr in THR_GRID:
        idx += 1
        bull_mask = raw_a2_vals > thr
        n_bull = int(bull_mask.sum())
        bull_ratio = n_bull / n_all

        tilt_amount_raw = TILT_STEP * (raw_a2_vals - thr) * (1.0 - raw_a2_vals)
        tilt_amount = np.where(bull_mask, np.clip(tilt_amount_raw, 0.0, TILT_CAP), 0.0)

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
        cid = thr_label(thr)
        m.update({'config_id': cid, 'threshold': thr,
                  'bull_days': n_bull, 'bull_ratio': bull_ratio,
                  'Trades_yr': trades_yr,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
        results.append(m)
        flag_base = ' (baseline)' if abs(thr - 0.15) < 1e-9 else ''
        print(f'  [{idx}/{len(THR_GRID)}] {cid} thr={thr:.2f} '
              f'bull={n_bull:>5d}d ({bull_ratio*100:.1f}%): '
              f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
              f'Sharpe={m["Sharpe_OOS"]:+.3f}  '
              f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
              f'W10Y={m["Worst10Y_star"]*100:+.2f}%  '
              f'Tr/yr={trades_yr:.1f}{flag_base}')

    print(f'\n全 {len(results)-1} configs 完了。')

    # ---------------- 判定 ----------------
    pass_list = [r for r in results[1:] if passes_all(r)]
    best_sharpe = max(results[1:], key=lambda r: r['Sharpe_OOS'])
    verdict = ('PASS' if pass_list else
               'WARN' if best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')
    bs_label = f'{best_sharpe["config_id"]} (thr={best_sharpe["threshold"]:.2f})'
    print(f'PASS configs: {len(pass_list)}')
    print(f'Best Sharpe: {bs_label} → '
          f'Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  '
          f'CAGR_OOS={best_sharpe["CAGR_OOS"]*100:+.2f}%  '
          f'MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%  '
          f'IS-OOS={best_sharpe["IS_OOS_gap"]*100:+.2f}pp  '
          f'Tr/yr={best_sharpe["Trades_yr"]:.1f}')
    print(f'総合判定: {verdict}')

    # T015 baseline 比較
    t015 = next((r for r in results[1:] if r['config_id'] == 'T015'), None)
    if t015 is not None:
        d_sharpe = best_sharpe['Sharpe_OOS'] - t015['Sharpe_OOS']
        print(f'T015 baseline Sharpe={t015["Sharpe_OOS"]:+.3f}, '
              f'best - T015 = {d_sharpe:+.4f}')

    # ---------------- CSV ----------------
    csv_path = os.path.join(BASE, 'f9_threshold_results.csv')
    pd.DataFrame([{
        'config_id': r['config_id'],
        'threshold': r['threshold'],
        'bull_days': r['bull_days'],
        'bull_ratio': r['bull_ratio'],
        'CAGR_IS':       r['CAGR_IS'],
        'CAGR_OOS':      r['CAGR_OOS'],
        'Sharpe_OOS':    r['Sharpe_OOS'],
        'MaxDD_FULL':    r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'],
        'P10_5Y':        r['P10_5Y'],
        'IS_OOS_gap':    r['IS_OOS_gap'],
        'Trades_yr':     r['Trades_yr'],
    } for r in results]).to_csv(csv_path, index=False, float_format='%.6f')

    # ---------------- MD ----------------
    hdr1, hdr2 = MD_HEADER_1P
    ref_row = fmt_row_1p('REF (no tilt)', m_ref)
    grid_rows = '\n'.join(
        fmt_row_1p(
            f'{r["config_id"]} (thr={r["threshold"]:.2f})'
            + (' ←baseline' if abs(r["threshold"] - 0.15) < 1e-9 else ''),
            r
        )
        for r in results[1:]
    )

    # bull 日数テーブル
    bull_rows = []
    for r in results[1:]:
        bull_rows.append(
            f'| {r["config_id"]} | {r["threshold"]:.2f} | {r["bull_days"]:,} '
            f'| {r["bull_ratio"]*100:.1f}% | {r["Trades_yr"]:.1f} |'
        )
    bull_table = '\n'.join(bull_rows)

    # 考察用 抽出
    sorted_by_sharpe = sorted(results[1:], key=lambda r: r['Sharpe_OOS'], reverse=True)
    best = sorted_by_sharpe[0]
    worst = sorted_by_sharpe[-1]
    t015_row = next((r for r in results[1:] if r['config_id'] == 'T015'), None)
    d_best_vs_t015 = best['Sharpe_OOS'] - t015_row['Sharpe_OOS'] if t015_row else 0.0

    # 単調 / U字 判定
    sharpes_in_grid_order = [r['Sharpe_OOS'] for r in results[1:]]
    is_monotone_inc = all(sharpes_in_grid_order[i] <= sharpes_in_grid_order[i+1]
                          for i in range(len(sharpes_in_grid_order)-1))
    is_monotone_dec = all(sharpes_in_grid_order[i] >= sharpes_in_grid_order[i+1]
                          for i in range(len(sharpes_in_grid_order)-1))
    if is_monotone_inc:
        pattern = '単調増加（THRESHOLDが高いほど Sharpe 高）'
    elif is_monotone_dec:
        pattern = '単調減少（THRESHOLDが低いほど Sharpe 高）'
    else:
        # 最良位置で型判定
        best_idx = sharpes_in_grid_order.index(max(sharpes_in_grid_order))
        if best_idx == 0 or best_idx == len(sharpes_in_grid_order) - 1:
            pattern = f'端点最良型（最良は {best["config_id"]}: thr={best["threshold"]:.2f}）'
        else:
            pattern = f'内側ピーク型（最良は {best["config_id"]}: thr={best["threshold"]:.2f}）'

    report = f"""\
# F9: Bull-Tilt THRESHOLD 最適化スイープ

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

### 背景
F7-v3（2026-05-24）で「E4 + step-func bull-tilt (tilt=10.0, cap=0.10)」が
PASS 基準 (Sharpe_OOS ≥ {REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}, CAGR_OOS ≥ {PASS_CAGR_OOS*100:.1f}%,
gap ≤ {PASS_GAP*100:.1f}pp, MaxDD > {PASS_MAXDD*100:.1f}%, W10Y★ ≥ {PASS_WORST10Y*100:.1f}%) を達成。
ただし THRESHOLD は signal 構築の bull/bear 判定値 (BASE_THRESHOLD={BASE_THRESHOLD}) を
継承して固定していた。

### F9 の動機
「tilt を適用する bull 日」の閾値を変えると Sharpe / CAGR / Trades/yr に
どう影響するかを系統的に検証する。
- 低 THRESHOLD（例 0.05）→ ほぼ全 bull 日に tilt（積極的）
- 高 THRESHOLD（例 0.40）→ 強確信日のみ tilt（選択的）

### Tilt 定式（F7-v3 step-func を継承）
```
bull_mask    = raw_a2 > THRESHOLD       # ← F9 で可変
tilt_amount  = TILT_STEP × (raw_a2 - THRESHOLD) × (1 - raw_a2)
tilt_amount  = clip(tilt_amount, 0, TILT_CAP)
wn_tilted    = wn_A + tilt_amount
wb_tilted    = clip(wb_A - tilt_amount, 0, wb_A)
```
- TILT_STEP = 10.0（極端値 → 実質ステップ関数）
- TILT_CAP  = 0.10（cap 固定）

### 共通設定
| 項目 | 定義 |
|------|------|
| **Base config** | E4 採用: k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, LT2-N750, mode B |
| **wn/wg/wb 構築** | simulate_rebalance_A の THRESHOLD は BASE_THRESHOLD={BASE_THRESHOLD} 固定 |
| **F9 可変要素** | tilt 適用 bull_mask の閾値のみ |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**サニティ**: REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (diff vs E4 best CAGR={diff_ref:+.2f}pp)

### グリッド
| config_id | THRESHOLD | 解釈 |
|:---|---:|:---|
| REF | — | E4 (tilt なし) |
| T005 | 0.05 | ほぼ全 bull 日に tilt（積極的） |
| T010 | 0.10 | 中程度の積極度 |
| T015 | 0.15 | F7-v3 A:tilt=10 の再現（ベースライン） |
| T020 | 0.20 | やや選択的 |
| T025 | 0.25 | 選択的 |
| T030 | 0.30 | 高確信日のみ |
| T040 | 0.40 | 非常に選択的 |

---

## §2 9指標テーブル

{hdr1}
{hdr2}
{ref_row}
{grid_rows}

{MD_WFA_NOTE}

### Bull 日数 / Trades_yr 詳細
| config_id | THR | bull_days | bull_ratio | Trades_yr |
|:---|---:|---:|---:|---:|
{bull_table}

---

## §3 考察

### パターン
**観察パターン**: {pattern}

### 最良 vs ベースライン (T015)
- **最良 config**: {best["config_id"]} (thr={best["threshold"]:.2f})
  - Sharpe_OOS = {best["Sharpe_OOS"]:+.4f}
  - CAGR_OOS = {best["CAGR_OOS"]*100:+.2f}%
  - MaxDD = {best["MaxDD_FULL"]*100:+.2f}%
  - Trades_yr = {best["Trades_yr"]:.1f}
- **T015 (baseline)**: Sharpe = {t015_row["Sharpe_OOS"]:+.4f}, CAGR_OOS = {t015_row["CAGR_OOS"]*100:+.2f}%, Trades_yr = {t015_row["Trades_yr"]:.1f}
- **Sharpe 差**: 最良 − T015 = {d_best_vs_t015:+.4f}

### Bull 日数 vs Sharpe トレードオフ
THRESHOLD を上げると tilt 適用日数が減り、Trades_yr も減少する。
- 最低 Trades_yr: {min(r["Trades_yr"] for r in results[1:]):.1f} ({min(results[1:], key=lambda r: r["Trades_yr"])["config_id"]})
- 最高 Trades_yr: {max(r["Trades_yr"] for r in results[1:]):.1f} ({max(results[1:], key=lambda r: r["Trades_yr"])["config_id"]})

### Sharpe レンジ
- 最高 Sharpe: {best["Sharpe_OOS"]:+.4f} ({best["config_id"]})
- 最低 Sharpe: {worst["Sharpe_OOS"]:+.4f} ({worst["config_id"]})
- レンジ幅: {(best["Sharpe_OOS"] - worst["Sharpe_OOS"]):.4f}

---

## §4 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ REF+0.020 (≥ {REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}) |
| (ii) CAGR_OOS | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap | ≤ {PASS_GAP*100:.1f}pp |
| (iv) MaxDD | > {PASS_MAXDD*100:.1f}% |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% |

- **PASS configs**: {len(pass_list)} / {len(THR_GRID)}
- **PASS リスト**: {', '.join(r['config_id'] for r in pass_list) if pass_list else '(なし)'}
- **最高 Sharpe**: {bs_label} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}, CAGR_OOS={best_sharpe["CAGR_OOS"]*100:+.2f}%, Tr/yr={best_sharpe["Trades_yr"]:.1f}
- **総合判定: {verdict}**

---

*生成スクリプト: `src/f9_threshold.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/f7v3_bull_tilt.py`, `src/e4_regime_klt.py`*
"""
    md_path = os.path.join(BASE, 'F9_THRESHOLD_2026-05-24.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {csv_path}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
