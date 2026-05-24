"""
F5: Bond-Regime Gated Sleeve スイープ
=====================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

仮説:
  TMF（3xロング国債）は金利上昇局面で構造的にパフォーマンスが悪化する。
  10Y利回りのYoY変化（slope_10y = DGS10_t - DGS10_{t-252}, pp単位）が
  閾値 slope_thr を超える期間は、Bondスリーブ比率 wb を gamma_b 倍に縮小し、
  不足分を frac_gold で Gold へ、残りは Cash（暗黙的に sum<1）に退避する。

  これにより MaxDD と Worst10Y★ の改善を目指す。

メカニズム:
  1. data/dgs10_daily.csv 読込 → 日次にalign（ffill）
  2. slope_10y = yield_t - yield_{t-252}
  3. 2-day lag を適用（ルックアヘッド回避）
  4. bond_gated = slope_10y_lag > slope_thr のとき wb_new = wb_A * gamma_b
  5. wb_deficit = wb_A - wb_new を frac_gold で Gold に、残りは Cash

グリッド:
  slope_thr_grid = [0.50, 0.75, 1.00]   (pp YoY change)
  gamma_b_grid   = [0.00, 0.25, 0.50]   (bond keep fraction when gated)
  frac_gold_grid = [0.50, 1.00]         (fraction of deficit → gold)

  合計: 3 × 3 × 2 = 18 configs + REF (no gating)

ベース戦略: E4 採用 config (k_lo=0.1, k_hi=0.8, vz_thr=0.7, k_mid=0.5)
           + LT2-N750-modeB + S2_VZGated + CFD3x

出力:
  - f5_bond_regime_results.csv
  - F5_BOND_REGIME_2026-05-24.md
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
SLOPE_THR_GRID = [0.50, 0.75, 1.00]
GAMMA_B_GRID   = [0.00, 0.25, 0.50]
FRAC_GOLD_GRID = [0.50, 1.00]

# E4 採用 config (CURRENT_BEST_STRATEGY 暫定昇格 2026-05-24)
K_LO, K_HI, VZ_THR, K_MID = 0.1, 0.8, 0.7, 0.5

# REF: E4 採用 config の単独実行値
REF_CAGR_OOS   = 0.3353
REF_SHARPE_OOS = 0.8915
REF_MAXDD      = -0.6001
REF_WORST10Y   = 0.1867

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)

# PASS 基準（F5 は MaxDD / Worst10Y★ の改善を厳格に要求）
PASS_SHARPE_DELTA = 0.020   # ≥ REF + 0.020 = 0.9115
PASS_CAGR_OOS     = 0.295
PASS_GAP          = 0.060
PASS_MAXDD        = -0.5901   # MaxDD > -59.01%（厳格化）
PASS_WORST10Y     = 0.190     # ≥ 19.0%（厳格化）


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


def align_dgs10(dates_series, dgs10_series):
    """Align DGS10 daily yield to backtest dates via ffill, with bfill fallback."""
    dgs10_aligned = dgs10_series.reindex(dates_series.values, method='ffill')
    # 古い時期の欠損は bfill で埋める
    return dgs10_aligned.fillna(method='bfill').values


def build_slope_10y(dates):
    """DGS10 から slope_10y (YoY pp変化) を計算し、2-day lag を適用して返す。"""
    dgs10_path = os.path.join(BASE, 'data', 'dgs10_daily.csv')
    dgs10_df = pd.read_csv(dgs10_path, parse_dates=['Date'])
    dgs10_ser = dgs10_df.set_index('Date')['yield_pct']
    yield_arr = align_dgs10(dates, dgs10_ser)
    slope_10y = yield_arr - np.roll(yield_arr, 252)
    slope_10y[:252] = 0.0
    slope_10y_lag = np.roll(slope_10y, 2)
    slope_10y_lag[:2] = 0.0
    return yield_arr, slope_10y_lag


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('F5: Bond-Regime Gated Sleeve スイープ')
    print('=' * 70)
    total = len(SLOPE_THR_GRID) * len(GAMMA_B_GRID) * len(FRAC_GOLD_GRID)
    print(f'グリッド: slope_thr={SLOPE_THR_GRID} × gamma_b={GAMMA_B_GRID} × frac_gold={FRAC_GOLD_GRID}  ({total} configs + REF)')

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

    # E4 採用 config の lev_mod を構築（全 config 共通）
    lt_sig_raw = build_lt_signal(close, 'LT2', 750)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    k_dyn = np.where(vz_arr > VZ_THR, K_HI, np.where(vz_arr < -VZ_THR, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5), index=lt_sig_raw.index)
    lev_mod = apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)

    # DGS10 slope_10y 構築
    yield_arr, slope_10y_lag = build_slope_10y(dates)
    print(f'DGS10: range=[{np.nanmin(yield_arr):.2f}%, {np.nanmax(yield_arr):.2f}%]')
    for thr in SLOPE_THR_GRID:
        pct = (slope_10y_lag > thr).sum() / len(slope_10y_lag) * 100
        print(f'  slope_thr={thr:.2f}pp: gate active {pct:.1f}% of days')

    # wn_A, wg_A, wb_A は numpy 配列（simulate_rebalance_A の戻り値）
    print(f'wn_A type: {type(wn_A).__name__}, shape: {np.asarray(wn_A).shape}')
    print('Assets and signals built. Starting sweep...')

    # REF: bond gate なし (wn_A, wg_A, wb_A をそのまま使用)
    nav_ref = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_ref, dates, n_trades_yr)
    m_ref.update({'slope_thr': float('nan'), 'gamma_b': float('nan'), 'frac_gold': float('nan'),
                  'Trades_yr': n_trades_yr,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
    print(f'[REF no gate] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  Sharpe={m_ref["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%  Worst10Y★={m_ref["Worst10Y_star"]*100:+.2f}%')
    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    print(f'[SANITY] CAGR diff vs E4 baseline = {diff_ref:+.2f}pp → {"OK" if abs(diff_ref) <= 0.15 else "WARN"}')

    results = [m_ref]
    idx = 0

    for slope_thr in SLOPE_THR_GRID:
        bond_gated = slope_10y_lag > slope_thr  # bool 配列（全期間）

        for gamma_b in GAMMA_B_GRID:
            # wb_new = wb_A * gamma_b (gated 日のみ), それ以外は wb_A
            wb_new = np.where(bond_gated, wb_A * gamma_b, wb_A)
            wb_deficit = wb_A - wb_new

            for frac_gold in FRAC_GOLD_GRID:
                idx += 1
                wg_new = wg_A + wb_deficit * frac_gold
                # 残り (1-frac_gold) は cash として暗黙的にレジ外 (sum < 1)

                nav = build_nav_strategy(
                    close, lev_mod, wn_A, wg_new, wb_new, dates,
                    gold_2x, bond_3x, sofr,
                    nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
                )
                m = calc_all_metrics(nav, dates, n_trades_yr)
                m.update({'slope_thr': slope_thr, 'gamma_b': gamma_b, 'frac_gold': frac_gold,
                          'Trades_yr': n_trades_yr,
                          'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
                results.append(m)
                print(f'  [{idx:>2d}/{total}] thr={slope_thr:.2f}/gb={gamma_b:.2f}/fg={frac_gold:.2f}: '
                      f'CAGR={m["CAGR_OOS"]*100:+.2f}%  Sharpe={m["Sharpe_OOS"]:+.3f}  '
                      f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  Worst10Y★={m["Worst10Y_star"]*100:+.2f}%')

    print(f'\n全 {len(results)-1} configs 完了。')

    pass_list  = [r for r in results[1:] if passes_all(r)]
    best_sharpe= max(results[1:], key=lambda r: r['Sharpe_OOS'])
    best_dd    = max(results[1:], key=lambda r: r['MaxDD_FULL'])  # MaxDD は負値なので max が小さいDD
    best_w10   = max(results[1:], key=lambda r: r['Worst10Y_star'])
    verdict    = ('PASS' if pass_list else
                  'WARN' if (best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS or
                             best_dd['MaxDD_FULL'] > REF_MAXDD or
                             best_w10['Worst10Y_star'] > REF_WORST10Y) else 'FAIL')
    print(f'PASS configs: {len(pass_list)}')
    print(f'Best Sharpe: thr={best_sharpe["slope_thr"]:.2f}/gb={best_sharpe["gamma_b"]:.2f}/'
          f'fg={best_sharpe["frac_gold"]:.2f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%')
    print(f'Best MaxDD:  thr={best_dd["slope_thr"]:.2f}/gb={best_dd["gamma_b"]:.2f}/'
          f'fg={best_dd["frac_gold"]:.2f} → MaxDD={best_dd["MaxDD_FULL"]*100:+.2f}%')
    print(f'Best W10Y★:  thr={best_w10["slope_thr"]:.2f}/gb={best_w10["gamma_b"]:.2f}/'
          f'fg={best_w10["frac_gold"]:.2f} → Worst10Y★={best_w10["Worst10Y_star"]*100:+.2f}%')
    print(f'総合判定: {verdict}')

    # CSV
    pd.DataFrame([{
        'slope_thr': r['slope_thr'], 'gamma_b': r['gamma_b'], 'frac_gold': r['frac_gold'],
        'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'], 'Trades_yr': r['Trades_yr'],
    } for r in results]).to_csv(os.path.join(BASE, 'f5_bond_regime_results.csv'),
                                index=False, float_format='%.6f')

    # MD: 上位20行表示（Sharpe降順）+ REF
    hdr1, hdr2 = MD_HEADER_1P
    top_results = sorted(results[1:], key=lambda r: r['Sharpe_OOS'], reverse=True)[:20]
    rows_md = '\n'.join(
        fmt_row_1p(f'thr={r["slope_thr"]:.2f}/gb={r["gamma_b"]:.2f}/fg={r["frac_gold"]:.2f}', r)
        for r in top_results
    )
    ref_row = fmt_row_1p('REF (no bond gate)', m_ref)

    report = f"""\
# F5: Bond-Regime Gated Sleeve スイープ

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

| 項目 | 定義 |
|------|------|
| **slope_thr グリッド** | {SLOPE_THR_GRID}（pp、10Y YoY変化の閾値） |
| **gamma_b グリッド** | {GAMMA_B_GRID}（gated時のBond維持率） |
| **frac_gold グリッド** | {FRAC_GOLD_GRID}（不足分のうちGoldに退避する比率） |
| **ベース戦略** | E4 採用 (k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}) + LT2-N750-modeB + S2_VZGated + CFD3x |
| **合計 configs** | {total} + REF |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**メカニズム**:
```
slope_10y_t  = DGS10_t - DGS10_{{t-252}}           # YoY変化 (pp)
slope_lag_t  = slope_10y_{{t-2}}                   # 2-day lag (ルックアヘッド回避)
bond_gated_t = slope_lag_t > slope_thr
wb_new_t     = wb_A_t * gamma_b   if bond_gated_t else wb_A_t
wg_new_t     = wg_A_t + (wb_A_t - wb_new_t) * frac_gold
# 残り (1-frac_gold) は cash として暗黙的にレジ外 (sum < 1)
```

**サニティ**: REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (vs E4 baseline {REF_CAGR_OOS*100:+.2f}%, diff {diff_ref:+.2f}pp)

---

## §2 9指標テーブル（Sharpe 上位20件 + REF）

> 列 `Param` = `thr / gb / fg` の3つ組

{hdr1}
{hdr2}
{ref_row}
{rows_md}

{MD_WFA_NOTE}

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ REF+0.020 (≥{REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}) |
| (ii) CAGR_OOS | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap | ≤ {PASS_GAP*100:.1f}pp |
| (iv) MaxDD | > {PASS_MAXDD*100:.2f}% （E4より厳格化） |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% （E4より厳格化） |

- **PASS configs**: {len(pass_list)} / {total}
- **最高 Sharpe**: thr={best_sharpe["slope_thr"]:.2f}/gb={best_sharpe["gamma_b"]:.2f}/fg={best_sharpe["frac_gold"]:.2f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}
- **最良 MaxDD**: thr={best_dd["slope_thr"]:.2f}/gb={best_dd["gamma_b"]:.2f}/fg={best_dd["frac_gold"]:.2f} → MaxDD={best_dd["MaxDD_FULL"]*100:+.2f}%
- **最良 Worst10Y★**: thr={best_w10["slope_thr"]:.2f}/gb={best_w10["gamma_b"]:.2f}/fg={best_w10["frac_gold"]:.2f} → Worst10Y★={best_w10["Worst10Y_star"]*100:+.2f}%
- **総合判定: {verdict}**

---

## §4 考察

仮説: TMF（3x長期国債）は金利上昇局面で構造的に劣化する。10Y利回りYoY変化が大きい期間は Bondスリーブを縮小すべき。
- gamma_b=0 は「gated期間中は完全に Bond を外す」極端ケース
- frac_gold=1.0 は「不足分を全て Gold へ」、0.5 は「半分 Gold / 半分 Cash」
- 主目的: MaxDD と Worst10Y★ の改善（CAGR を犠牲にしてでも）
- リスク: bond gateが頻発する期間 (1970s/1980s) でレジーム判定の精度が結果を左右

---

*生成スクリプト: `src/f5_bond_regime.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `data/dgs10_daily.csv`*
"""
    md_path = os.path.join(BASE, 'F5_BOND_REGIME_2026-05-24.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "f5_bond_regime_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
