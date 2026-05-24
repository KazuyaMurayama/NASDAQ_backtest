"""
G5: Real-yield LT Overlay スイープ
==================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

仮説:
  現行ベスト (E4: Volatility Regime-Conditional k_lt + LT2 + S2_VZGated) は
  価格・リターン由来のシグナル (LT2 = 価格モメンタム, vz = ボラ z-score) のみで構成。
  ここに **マクロ非相関** な情報源として「実質10年金利 z-score」を LT overlay として
  追加することで、独立情報の寄与により Sharpe 改善 / IS-OOS gap 抑制が期待される。

メカニズム:
  ry        = DGS10_t - infl_5y_t              (実質金利, 年率%)
  infl_5y_t = log(CPI_t / CPI_{t-1260}) / 5     (5年トレーリングインフレ, 年率)
  ry_z      = standardize(ry, window=2520, min_periods=252)   (10年ローリング)
  lt_bias_macro = (-k_macro × ry_z × 0.5).clip(-0.5, 0.5)
    → 実質金利↑ (リスクオフ) → NASDAQ 配分縮小 (contrarian for stocks)

  lt_bias_combined = (1 - alpha) × lt_bias_E4 + alpha × lt_bias_macro

グリッド:
  K_MACRO_GRID = [0.3, 0.5, 0.7]
  ALPHA_GRID   = [0.25, 0.50]
  → 6 configs + REF (k_macro=0, alpha=0 = pure E4)

出力:
  - g5_real_yield_results.csv
  - G5_REAL_YIELD_2026-05-24.md
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
# グリッド
# ---------------------------------------------------------------------------
K_MACRO_GRID = [0.3, 0.5, 0.7]
ALPHA_GRID   = [0.25, 0.50]

# E4 base (regime-conditional k_lt, current best)
K_LO, K_HI, VZ_THR, K_MID = 0.1, 0.8, 0.7, 0.5

# REF = E4 current best (k_macro=0, alpha=0)
REF_CAGR_OOS   = 0.3353
REF_SHARPE_OOS = 0.8915
REF_MAXDD      = -0.6001

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)

PASS_SHARPE_DELTA = 0.020   # ≥ REF + 0.020 = 0.9115
PASS_CAGR_OOS     = 0.295
PASS_GAP          = 0.060
PASS_MAXDD        = -0.6501
PASS_WORST10Y     = 0.150


# ---------------------------------------------------------------------------
# Macro signal: real-yield z-score
# ---------------------------------------------------------------------------

def _load_dgs10():
    df = pd.read_csv(os.path.join(BASE, 'data', 'dgs10_daily.csv'), parse_dates=['Date'])
    return df.set_index('Date')['yield_pct']


def _load_cpi():
    df = pd.read_csv(os.path.join(BASE, 'data', 'cpiaucsl_monthly.csv'), parse_dates=['DATE'])
    return df.set_index('DATE')['CPIAUCSL']


def build_real_yield_z(dates_series):
    """Compute lag-2 real-yield z-score aligned to dates_series.

    Returns pd.Series indexed by dates_series.values.
    NaN → 0 (neutral) for the early history (~7 yrs) before enough data.
    """
    dgs10_series = _load_dgs10()
    cpi_series   = _load_cpi()
    idx = dates_series.values

    yield_daily = dgs10_series.reindex(idx, method='ffill').ffill().bfill()
    cpi_daily   = cpi_series.reindex(idx, method='ffill').ffill().bfill()

    cpi_arr = cpi_daily.values
    n = len(cpi_arr)
    infl_5y = np.full(n, np.nan)
    for i in range(1260, n):
        if cpi_arr[i - 1260] > 0:
            infl_5y[i] = (np.log(cpi_arr[i] / cpi_arr[i - 1260]) / 5) * 100  # annualized %

    ry = yield_daily.values - infl_5y
    ry_series = pd.Series(ry, index=idx)

    ry_mean = ry_series.rolling(2520, min_periods=252).mean()
    ry_std  = ry_series.rolling(2520, min_periods=252).std()
    ry_z = (ry_series - ry_mean) / ry_std.replace(0, np.nan)
    ry_z = ry_z.fillna(0.0)

    # 2-day lag (lookahead prevention)
    ry_z_lag = ry_z.shift(2).fillna(0.0)
    return ry_z_lag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            r['CAGR_OOS']     >= PASS_CAGR_OOS and
            r['IS_OOS_gap']   <= PASS_GAP and
            r['MaxDD_FULL']    > PASS_MAXDD and
            r['Worst10Y_star']>= PASS_WORST10Y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('G5: Real-yield LT Overlay スイープ')
    print('=' * 70)
    total = len(K_MACRO_GRID) * len(ALPHA_GRID)
    print(f'グリッド: k_macro={K_MACRO_GRID} × alpha={ALPHA_GRID}  ({total} configs + REF)')

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
    n_trades_yr = n_tr / n_years
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)

    lt_sig_raw = build_lt_signal(close, 'LT2', 750)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values

    # E4 base bias (compute ONCE)
    k_dyn = np.where(vz_arr > VZ_THR, K_HI, np.where(vz_arr < -VZ_THR, K_LO, K_MID))
    lt_bias_E4 = pd.Series(
        np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
        index=lt_sig_raw.index,
    )

    # Real-yield z-score (compute ONCE)
    ry_z = build_real_yield_z(dates)
    # Sanity check on ry_z
    n_nonzero = int((ry_z != 0).sum())
    print(f'ry_z: non-zero days = {n_nonzero:,} / {n:,} (first ~7yr neutral by design)')
    print(f'ry_z stats (excl. zeros): mean={ry_z[ry_z!=0].mean():+.3f} '
          f'std={ry_z[ry_z!=0].std():.3f}  min={ry_z.min():+.2f}  max={ry_z.max():+.2f}')

    # ----- REF (E4 only, no macro overlay) -----
    lev_mod_ref = apply_lt_mode_b(lev_raw, lt_bias_E4, l_min=0.0, l_max=1.0)
    nav_ref = build_nav_strategy(
        close, lev_mod_ref, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_ref, dates, n_trades_yr)
    m_ref.update({'k_macro': 0.0, 'alpha': 0.0,
                  'Trades_yr': n_trades_yr,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
    print(f'[REF E4 only] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  '
          f'Sharpe={m_ref["Sharpe_OOS"]:+.3f}  MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%')
    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    print(f'[SANITY] CAGR_OOS diff vs E4 best = {diff_ref:+.2f}pp '
          f'→ {"OK" if abs(diff_ref) <= 0.20 else "WARN"}')

    results = [m_ref]
    idx = 0
    ry_z_arr = ry_z.values

    for k_macro in K_MACRO_GRID:
        # Macro overlay bias (depends only on k_macro)
        lt_bias_macro_arr = np.clip(-k_macro * ry_z_arr * 0.5, -0.5, 0.5)
        for alpha in ALPHA_GRID:
            idx += 1
            lt_bias_combined_arr = (1 - alpha) * lt_bias_E4.values + alpha * lt_bias_macro_arr
            lt_bias_combined = pd.Series(lt_bias_combined_arr, index=lt_sig_raw.index)
            lev_mod_combined = apply_lt_mode_b(lev_raw, lt_bias_combined, l_min=0.0, l_max=1.0)
            nav = build_nav_strategy(
                close, lev_mod_combined, wn_A, wg_A, wb_A, dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
            )
            m = calc_all_metrics(nav, dates, n_trades_yr)
            m.update({'k_macro': k_macro, 'alpha': alpha,
                      'Trades_yr': n_trades_yr,
                      'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
            results.append(m)
            print(f'  [{idx}/{total}] k={k_macro:.1f}/a={alpha:.2f}: '
                  f'CAGR={m["CAGR_OOS"]*100:+.2f}%  Sharpe={m["Sharpe_OOS"]:+.3f}  '
                  f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  gap={m["IS_OOS_gap"]*100:+.2f}pp')

    print(f'\n全 {len(results)-1} configs 完了。')

    pass_list   = [r for r in results[1:] if passes_all(r)]
    best_sharpe = max(results[1:], key=lambda r: r['Sharpe_OOS'])
    verdict     = ('PASS' if pass_list else
                   'WARN' if best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')
    print(f'PASS configs: {len(pass_list)}')
    print(f'Best Sharpe: k_macro={best_sharpe["k_macro"]:.1f} alpha={best_sharpe["alpha"]:.2f} '
          f'→ Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  '
          f'CAGR={best_sharpe["CAGR_OOS"]*100:+.2f}%  '
          f'MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%  '
          f'IS-OOS={best_sharpe["IS_OOS_gap"]*100:+.2f}pp')
    print(f'総合判定: {verdict}')

    # ----- CSV -----
    pd.DataFrame([{
        'k_macro': r['k_macro'], 'alpha': r['alpha'],
        'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'], 'Trades_yr': r['Trades_yr'],
    } for r in results]).to_csv(os.path.join(BASE, 'g5_real_yield_results.csv'),
                                index=False, float_format='%.6f')

    # ----- MD -----
    hdr1, hdr2 = MD_HEADER_1P
    ref_row = fmt_row_1p('REF (E4 only)', m_ref)
    # 残り configs を Sharpe 降順で表示
    other = sorted(results[1:], key=lambda r: r['Sharpe_OOS'], reverse=True)
    rows_md = '\n'.join(
        fmt_row_1p(f'k={r["k_macro"]:.1f}/a={r["alpha"]:.2f}', r) for r in other
    )

    report = f"""\
# G5: Real-yield LT Overlay スイープ

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

| 項目 | 定義 |
|------|------|
| **base** | E4 (k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}) + LT2-N750-modeB + S2_VZGated |
| **macro signal** | 実質10年金利 z-score (DGS10 − 5yr CPI inflation, 10yr rolling z) |
| **lag** | 2 営業日 (lookahead 防止) |
| **k_macro グリッド** | {K_MACRO_GRID} |
| **alpha グリッド** | {ALPHA_GRID} (= macro 寄与の重み) |
| **合計 configs** | {total} + REF |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**合成式**:
```
ry        = DGS10_t − (log(CPI_t / CPI_{{t−1260}}) / 5) × 100
ry_z      = (ry − rolling_mean(ry, 2520)) / rolling_std(ry, 2520)
lt_bias_macro    = (−k_macro × ry_z × 0.5).clip(−0.5, 0.5)
lt_bias_E4       = (−k_dyn(vz) × lt_sig × 0.5).clip(−0.5, 0.5)   # 現行ベスト
lt_bias_combined = (1 − alpha) × lt_bias_E4 + alpha × lt_bias_macro
```

**サニティ**: REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (diff vs E4 best {diff_ref:+.2f}pp)

---

## §2 9指標テーブル（REF + 全 {total} configs, Sharpe 降順）

{hdr1}
{hdr2}
{ref_row}
{rows_md}

{MD_WFA_NOTE}

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ REF+{PASS_SHARPE_DELTA:.3f} (≥{REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}) |
| (ii) CAGR_OOS | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap | ≤ {PASS_GAP*100:.1f}pp |
| (iv) MaxDD | > {PASS_MAXDD*100:.2f}% |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% |

- **PASS configs**: {len(pass_list)} / {total}
- **最高 Sharpe**: k_macro={best_sharpe["k_macro"]:.1f}, alpha={best_sharpe["alpha"]:.2f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}
- **総合判定: {verdict}**

---

## §4 考察

実質金利 z-score は LT2 (価格モメンタム) や vz (ボラ z-score) と独立な
マクロシグナルとして overlay を加算する設計:
- alpha=0.25 は macro を弱く混ぜる (E4 を保ち補助)
- alpha=0.50 は均等 (LT2 と macro を等重)
- contrarian: 実質金利高騰局面 → リスク資産抑制（NASDAQ 配分↓）
- 注意点: 初期 ~7 年 (1962-1969) は CPI/DGS10 ローリング窓不足のため ry_z=0 (neutral)。
  OOS は 1980 以降に位置するため判定への影響は限定的。
- 同方向のシグナル重複に注意: VIX レジーム / vz_gated と相関する可能性。

---

*生成スクリプト: `src/g5_real_yield.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`*
"""
    md_path = os.path.join(BASE, 'G5_REAL_YIELD_2026-05-24.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "g5_real_yield_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
