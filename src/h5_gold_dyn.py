"""
H5: Gold Sleeve Dynamic 1x/2x Switch (real-yield regime conditional)
=====================================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

仮説:
  Gold 2x (UGL proxy) は実質金利が低い局面で有効に機能するが、
  実質金利が高い局面ではドラッグとなる。実質金利 z-score が高い時に
  gold 1x へ部分的（または完全）にスイッチすることで、Worst10Y★ / MaxDD を
  改善しつつ CAGR コストを最小化できるか検証する。

仕組み:
  - DGS10 と CPI から実質金利 z-score `ry_z` を計算（G5 と同様）
  - ry_z_lag > ry_thr の日: blended = alpha_low * gold_2x_ret + (1-alpha_low) * gold_1x_ret
  - ry_z_lag <= ry_thr の日: 通常の gold_2x (alpha = 1.0)
  - blended NAV を build_nav_strategy の gold_2x 引数に渡す

E4 ベースは固定: K_LO=0.1, K_HI=0.8, VZ_THR=0.7, K_MID=0.5, LT2-N750-modeB

グリッド:
  RY_THR_GRID    = [0.5, 1.0, 1.5]   # real yield z-score 切替閾値
  ALPHA_LOW_GRID = [0.0, 0.5]        # 高実質金利時の gold_2x ウェイト (0=1x only, 0.5=blend)
  合計: 3 × 2 = 6 configs + REF (alpha=1.0 always = 通常 E4)
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
RY_THR_GRID    = [0.5, 1.0, 1.5]
ALPHA_LOW_GRID = [0.0, 0.5]

# E4 base (固定)
K_LO, K_HI, VZ_THR, K_MID = 0.1, 0.8, 0.7, 0.5

# REF (E4 current best, 2026-05-24 暫定昇格)
REF_CAGR_OOS   = 0.3353
REF_SHARPE_OOS = 0.8915
REF_MAXDD      = -0.6001
REF_WORST10Y   = 0.1867

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)

PASS_SHARPE_DELTA = 0.020
PASS_CAGR_OOS     = 0.295
PASS_GAP          = 0.060
PASS_MAXDD        = -0.5901
PASS_WORST10Y     = 0.190


# ---------------------------------------------------------------------------
# Real yield z-score builder
# ---------------------------------------------------------------------------
dgs10_df = pd.read_csv(os.path.join(BASE, 'data', 'dgs10_daily.csv'), parse_dates=['Date'])
dgs10_series = dgs10_df.set_index('Date')['yield_pct']

cpi_df = pd.read_csv(os.path.join(BASE, 'data', 'cpiaucsl_monthly.csv'), parse_dates=['DATE'])
cpi_series = cpi_df.set_index('DATE')['CPIAUCSL']


def build_ry_z(dates_series):
    """Compute lagged (2-day) real-yield z-score aligned to dates_series."""
    yield_daily = dgs10_series.reindex(dates_series.values, method='ffill').ffill().bfill()
    cpi_daily   = cpi_series.reindex(dates_series.values, method='ffill').ffill().bfill()
    cpi_arr = cpi_daily.values
    infl_5y = np.full(len(cpi_arr), np.nan)
    for i in range(1260, len(cpi_arr)):
        if cpi_arr[i - 1260] > 0:
            infl_5y[i] = (np.log(cpi_arr[i] / cpi_arr[i - 1260]) / 5) * 100
    ry = yield_daily.values - infl_5y
    ry_series = pd.Series(ry, index=dates_series.values)
    ry_mean = ry_series.rolling(2520, min_periods=252).mean()
    ry_std  = ry_series.rolling(2520, min_periods=252).std()
    ry_z = ((ry_series - ry_mean) / ry_std.replace(0, np.nan)).fillna(0.0)
    return ry_z.shift(2).fillna(0.0)


# ---------------------------------------------------------------------------
# Metrics helpers (E4 と同一)
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
            r['CAGR_OOS'] >= PASS_CAGR_OOS and
            r['IS_OOS_gap'] <= PASS_GAP and
            r['MaxDD_FULL'] > PASS_MAXDD and
            r['Worst10Y_star'] >= PASS_WORST10Y)


def build_blended_gold_nav(gold_1x_arr, gold_2x_arr, ry_z_arr, ry_thr, alpha_low, index):
    """Build blended gold NAV from daily-return blend of 1x and 2x golds.

    alpha = 1.0   when ry_z_arr <= ry_thr   (full 2x)
    alpha = alpha_low when ry_z_arr  > ry_thr (high real-yield regime)
    Returns: pd.Series of NAV with same starting value as gold_2x_arr[0].
    """
    g1 = pd.Series(gold_1x_arr, index=index)
    g2 = pd.Series(gold_2x_arr, index=index)
    r_g1 = g1.pct_change().fillna(0).values
    r_g2 = g2.pct_change().fillna(0).values
    alpha_arr = np.where(ry_z_arr > ry_thr, alpha_low, 1.0)
    r_blend = alpha_arr * r_g2 + (1.0 - alpha_arr) * r_g1
    # at t=0 set return to 0 (sanity)
    r_blend[0] = 0.0
    nav_factor = (1.0 + r_blend).cumprod()
    start_val = float(gold_2x_arr[0]) if gold_2x_arr[0] != 0 else 1.0
    return pd.Series(start_val * nav_factor, index=index)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('H5: Gold Sleeve Dynamic 1x/2x Switch')
    print('=' * 70)
    total = len(RY_THR_GRID) * len(ALPHA_LOW_GRID)
    print(f'グリッド: ry_thr={RY_THR_GRID} × alpha_low={ALPHA_LOW_GRID}  ({total} configs + REF)')

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

    # E4 lev_mod (固定: K_LO=0.1, K_HI=0.8, VZ_THR=0.7, K_MID=0.5)
    lt_sig_raw = build_lt_signal(close, 'LT2', 750)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    k_dyn = np.where(vz_arr > VZ_THR, K_HI,
                     np.where(vz_arr < -VZ_THR, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=lt_sig_raw.index)
    lev_mod = apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)

    # Real yield z-score (前計算)
    ry_z = build_ry_z(dates)
    ry_z_arr = ry_z.values

    print(f'ry_z range: min={ry_z_arr.min():+.2f}, max={ry_z_arr.max():+.2f}, '
          f'mean={ry_z_arr.mean():+.2f}, std={ry_z_arr.std():.2f}')
    for thr in RY_THR_GRID:
        share = float((ry_z_arr > thr).mean())
        print(f'  share(ry_z > {thr:.1f}) = {share*100:.1f}% of days')

    print('Assets and signals built. Starting sweep...')

    idx_dates = dates.index

    # REF (gold_2x 固定 = alpha=1.0 always)
    nav_ref = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_ref, dates, n_trades_yr)
    m_ref.update({'ry_thr': float('nan'), 'alpha_low': 1.0,
                  'Trades_yr': n_trades_yr,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
    print(f'[REF gold_2x fixed] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  '
          f'Sharpe={m_ref["Sharpe_OOS"]:+.3f}  MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%  '
          f'Worst10Y★={m_ref["Worst10Y_star"]*100:+.2f}%')
    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    print(f'[SANITY] diff vs E4-best CAGR_OOS={diff_ref:+.2f}pp → '
          f'{"OK" if abs(diff_ref) <= 0.15 else "WARN"}')

    results = [m_ref]
    idx = 0
    for ry_thr in RY_THR_GRID:
        for alpha_low in ALPHA_LOW_GRID:
            idx += 1
            gold_blended_nav = build_blended_gold_nav(
                gold_1x, gold_2x, ry_z_arr, ry_thr, alpha_low, idx_dates,
            )
            nav = build_nav_strategy(
                close, lev_mod, wn_A, wg_A, wb_A, dates,
                gold_blended_nav, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
            )
            m = calc_all_metrics(nav, dates, n_trades_yr)
            m.update({'ry_thr': ry_thr, 'alpha_low': alpha_low,
                      'Trades_yr': n_trades_yr,
                      'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
            results.append(m)
            print(f'[{idx}/{total}] ry={ry_thr:.1f}/a={alpha_low:.1f}: '
                  f'CAGR={m["CAGR_OOS"]*100:+.2f}%  Sharpe={m["Sharpe_OOS"]:+.3f}  '
                  f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
                  f'Worst10Y★={m["Worst10Y_star"]*100:+.2f}%')

    print(f'\n全 {len(results)-1} configs 完了。')

    pass_list   = [r for r in results[1:] if passes_all(r)]
    best_sharpe = max(results[1:], key=lambda r: r['Sharpe_OOS'])
    verdict     = ('PASS' if pass_list else
                   'WARN' if best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')
    print(f'PASS configs: {len(pass_list)}')
    print(f'Best Sharpe: ry_thr={best_sharpe["ry_thr"]:.1f} alpha_low={best_sharpe["alpha_low"]:.1f} '
          f'→ Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%  '
          f'Worst10Y★={best_sharpe["Worst10Y_star"]*100:+.2f}%  '
          f'IS-OOS={best_sharpe["IS_OOS_gap"]*100:+.2f}pp')
    print(f'総合判定: {verdict}')

    # CSV
    csv_rows = []
    for r in results:
        csv_rows.append({
            'ry_thr':    r['ry_thr'],
            'alpha_low': r['alpha_low'],
            'CAGR_IS':       r['CAGR_IS'],
            'CAGR_OOS':      r['CAGR_OOS'],
            'Sharpe_OOS':    r['Sharpe_OOS'],
            'MaxDD_FULL':    r['MaxDD_FULL'],
            'Worst10Y_star': r['Worst10Y_star'],
            'P10_5Y':        r['P10_5Y'],
            'IS_OOS_gap':    r['IS_OOS_gap'],
            'Trades_yr':     r['Trades_yr'],
        })
    csv_path = os.path.join(BASE, 'h5_gold_dyn_results.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format='%.6f')

    # MD: 9指標標準 (Sharpe 降順, REF 先頭)
    hdr1, hdr2 = MD_HEADER_1P
    sweep_rows = sorted(results[1:], key=lambda r: r['Sharpe_OOS'], reverse=True)
    rows_md = '\n'.join(
        fmt_row_1p(f'ry={r["ry_thr"]:.1f}/a={r["alpha_low"]:.1f}', r)
        for r in sweep_rows
    )
    ref_row = fmt_row_1p('REF (gold 2x fixed)', m_ref)

    report = f"""\
# H5: Gold Sleeve Dynamic 1x/2x Switch（実質金利レジーム条件付き）

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

| 項目 | 定義 |
|------|------|
| **仮説** | Gold 2x は低実質金利期に有効・高実質金利期にドラッグ。高ry_z期に gold_1x へ部分／完全スイッチして Worst10Y★ / MaxDD を改善する |
| **ry_thr グリッド** | {RY_THR_GRID}（実質金利 z-score 切替閾値） |
| **alpha_low グリッド** | {ALPHA_LOW_GRID}（高実質金利期の gold_2x ウェイト、0=1x only, 0.5=blend） |
| **E4 base** | K_LO={K_LO}, K_HI={K_HI}, VZ_THR={VZ_THR}, K_MID={K_MID}, LT2-N750-modeB |
| **合計 configs** | {total} + REF |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**ブレンド規則**:
```
ry_z_lag > ry_thr  → blended_ret = alpha_low * r_gold_2x + (1 - alpha_low) * r_gold_1x
ry_z_lag <= ry_thr → blended_ret = r_gold_2x  (alpha = 1.0)
```
ry_z は DGS10 - 5Y 実質インフレ率を 10年ロール平均/標準偏差で z-score 化（2日ラグ）。

**サニティ**: REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (vs E4-best {REF_CAGR_OOS*100:+.2f}%, diff {diff_ref:+.2f}pp)

---

## §2 9指標テーブル（Sharpe 降順）

{hdr1}
{hdr2}
{ref_row}
{rows_md}

{MD_WFA_NOTE}

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ REF+{PASS_SHARPE_DELTA:.3f} (≥{REF_SHARPE_OOS+PASS_SHARPE_DELTA:.4f}) |
| (ii) CAGR_OOS | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap | ≤ {PASS_GAP*100:.1f}pp |
| (iv) MaxDD | > {PASS_MAXDD*100:.2f}% |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% |

- **PASS configs**: {len(pass_list)} / {total}
- **最高 Sharpe**: ry_thr={best_sharpe["ry_thr"]:.1f}, alpha_low={best_sharpe["alpha_low"]:.1f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}
- **総合判定: {verdict}**

---

## §4 考察

- gold_2x は実質金利上昇局面（インフレ後退・FRB tightening 後の正常化期）でドラッグ。
  高 ry_z 期に 1x/blend にスイッチすることで MaxDD と Worst10Y★ の改善を狙う。
- 過学習リスク: 自由度2 (ry_thr, alpha_low)。少自由度のため WFE は良好と期待。
- ry_z は S2_VZGated や E4 内部の vz とは独立シグナル（マクロ系）のため、二重使用の懸念は低い。
- 副作用: 高 ry_z 期の CAGR_OOS が低下する可能性（gold 2x の上昇分を取り逃がす）。

---

*生成スクリプト: `src/h5_gold_dyn.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `e4_regime_klt.py`*
"""
    md_path = os.path.join(BASE, 'H5_GOLD_DYN_2026-05-24.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {csv_path}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
