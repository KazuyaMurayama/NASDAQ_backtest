"""
A3: Regime Asset Tilt (VoV + vz による NASDAQ → Gold/Bond 動的シフト)
=====================================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-27)

概念: E4ベース（k_lo=0.1, k_hi=0.8, vz_thr=0.7, l_max=7.0）を完全維持。
  VoV(Vol-of-Vol) > VOV_THR かつ vz > vz_thr の高ストレス日に
  NASDAQ → Gold/Bond に α 分（半々）シフトしてMaxDD/Worst10Yを改善する。

E4採用値固定: k_lo=0.1, k_hi=0.8, vz_thr=0.7, l_max=7.0

グリッド:
  VOV_THR_GRID   = [1.3, 1.5, 1.7]     # vov z-score閾値
  ALPHA_MAX_GRID = [0.20, 0.30, 0.40]  # NASDAQ削減上限
  合計: 3×3 = 9 configs + REF（α=0）

出力:
  - a3_regime_asset_tilt_results.csv
  - A3_REGIME_ASSET_TILT_2026-05-27.md
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
from compute_cfd_worst10y import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import build_lt_signal, apply_lt_mode_b
from a2_dyn_lmax import (
    compute_L_s2_dyn_lmax, signal_to_bias_dynamic,
    compute_p10_5y, calc_all_metrics,
    S2_BASE, K_LO, K_HI, K_MID, VZ_THR,
    REF_CAGR_OOS, REF_SHARPE_OOS, REF_MAXDD,
    PASS_SHARPE_DELTA, PASS_CAGR_OOS, PASS_GAP, PASS_MAXDD, PASS_WORST10Y,
)
from _sweep_format import MD_HEADER_1P, fmt_row_1p, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# A3 グリッド
# ---------------------------------------------------------------------------
VOV_THR_GRID   = [1.3, 1.5, 1.7]
ALPHA_MAX_GRID = [0.20, 0.30, 0.40]

# l_max=7.0 固定（E4ベスト）
L_MAX_FIXED = 7.0
LT_N = 750  # LT2-N750 (E4採用値)


def compute_vov_zscore(ret: pd.Series, win_short=20, win_long=60, win_norm=252) -> pd.Series:
    """VoV (Vol-of-Vol) z-score: 短期ボラのローリング標準偏差を時系列正規化"""
    vol_short = ret.rolling(win_short, min_periods=win_short // 2).std() * np.sqrt(TRADING_DAYS)
    vov       = vol_short.rolling(win_long, min_periods=win_long // 2).std()
    vov_mean  = vov.expanding(min_periods=win_norm).mean()
    vov_std   = vov.expanding(min_periods=win_norm).std().replace(0, np.nan)
    vov_z     = (vov - vov_mean) / vov_std
    return vov_z.fillna(0.0)


def apply_regime_tilt(wn: np.ndarray, wg: np.ndarray, wb: np.ndarray,
                      stress: np.ndarray, alpha_max: float):
    """
    高ストレス日に NASDAQ → Gold/Bond に α 分（半々）シフト。
    wn+wg+wb の合計は保存される。
    """
    alpha = np.where(stress, alpha_max, 0.0)
    wn_p  = wn * (1.0 - alpha)
    shift = wn * alpha
    wg_p  = wg + shift * 0.5
    wb_p  = wb + shift * 0.5
    return wn_p, wg_p, wb_p


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('A3: Regime Asset Tilt (VoV + vz)')
    print('=' * 70)
    total = len(VOV_THR_GRID) * len(ALPHA_MAX_GRID)
    print(f'グリッド: VOV_THR={VOV_THR_GRID} × ALPHA_MAX={ALPHA_MAX_GRID} '
          f'({total} configs + REF)')
    print(f'E4固定値: K_LO={K_LO}, K_HI={K_HI}, VZ_THR={VZ_THR}, l_max={L_MAX_FIXED}')

    # サニティ: ウェイト合計保存テスト
    _test_s = np.array([True, False, True, False, True])
    _wn = np.full(5, 0.6); _wg = np.full(5, 0.2); _wb = np.full(5, 0.2)
    _wp, _gp, _bp = apply_regime_tilt(_wn, _wg, _wb, _test_s, 0.3)
    assert np.allclose(_wp + _gp + _bp, _wn + _wg + _wb), 'weight sum not conserved'
    assert np.allclose(_wp[_test_s], 0.6 * 0.7), 'stress alpha mismatch'
    assert np.allclose(_wp[~_test_s], 0.6), 'non-stress alpha leaked'
    print('[SANITY] apply_regime_tilt: weight conservation OK')

    # --- データロード ---
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
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                          bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_trades_yr = n_tr / n_years

    # E4 レジーム（vz_thr=0.7, k_lo=0.1, k_hi=0.8）
    lt_sig_raw = build_lt_signal(close, 'LT2', LT_N)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    regime_hi  = vz_arr > +VZ_THR
    regime_lo  = vz_arr < -VZ_THR
    k_dyn_e4   = np.where(regime_hi, K_HI, np.where(regime_lo, K_LO, K_MID))
    lt_bias_e4 = pd.Series(signal_to_bias_dynamic(lt_sig_arr, k_dyn_e4),
                           index=lt_sig_raw.index)
    lev_mod_e4 = apply_lt_mode_b(lev_raw, lt_bias_e4, l_min=0.0, l_max=1.0)

    # S2 動的レバ（l_max=7.0 固定 = E4ベスト）
    l_max_fixed_series = pd.Series(L_MAX_FIXED, index=ret.index)
    L_s2 = compute_L_s2_dyn_lmax(ret, vz, l_max_fixed_series, **S2_BASE)

    # VoV z-score 事前計算（全configで共通）
    vov_z = compute_vov_zscore(ret)
    print(f'VoV z-score stats: mean={vov_z.mean():.3f}, '
          f'min={vov_z.min():.3f}, max={vov_z.max():.3f}, '
          f'>1.5割合={(vov_z > 1.5).mean()*100:.1f}%')

    print('Starting sweep...')

    # --- REF: α=0（E4そのもの） ---
    nav_ref = build_nav_strategy(
        close, lev_mod_e4, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_ref, dates, n_trades_yr)
    m_ref.update({'vov_thr': float('nan'), 'alpha_max': 0.0,
                  'Trades_yr': n_trades_yr, 'Stress_pct': 0.0,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
    print(f'[REF α=0] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  '
          f'Sharpe={m_ref["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%')
    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    print(f'[SANITY] REF CAGR_OOS diff={diff_ref:+.2f}pp → '
          f'{"OK" if abs(diff_ref) <= 0.30 else "WARN (>0.30pp)"}')

    # --- グリッドスイープ ---
    results = [m_ref]
    idx = 0
    vz_pos_arr = (vz.fillna(0).values > VZ_THR)
    vov_arr    = vov_z.values

    for vov_thr in VOV_THR_GRID:
        for alpha_max in ALPHA_MAX_GRID:
            idx += 1
            stress = (vov_arr > vov_thr) & vz_pos_arr  # 二重ゲート
            wn_p, wg_p, wb_p = apply_regime_tilt(wn_A, wg_A, wb_A, stress, alpha_max)

            nav_dyn = build_nav_strategy(
                close, lev_mod_e4, wn_p, wg_p, wb_p, dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
            )
            m = calc_all_metrics(nav_dyn, dates, n_trades_yr)
            stress_pct = stress.mean() * 100
            m.update({'vov_thr': vov_thr, 'alpha_max': alpha_max,
                      'Trades_yr': n_trades_yr, 'Stress_pct': stress_pct,
                      'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
            results.append(m)
            print(f'  [{idx:>2d}/{total}] vov_thr={vov_thr:.1f} α={alpha_max:.2f}: '
                  f'CAGR={m["CAGR_OOS"]*100:+.2f}%  Sharpe={m["Sharpe_OOS"]:+.3f}  '
                  f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  stress={stress_pct:.1f}%')

    print(f'\n全 {len(results)-1} configs 完了。')

    # --- 判定 ---
    def passes_all(r):
        return (r['Sharpe_OOS'] - REF_SHARPE_OOS >= PASS_SHARPE_DELTA and
                r['CAGR_OOS'] >= PASS_CAGR_OOS and
                r['IS_OOS_gap'] <= PASS_GAP and
                r['MaxDD_FULL'] > PASS_MAXDD and
                r['Worst10Y_star'] >= PASS_WORST10Y)

    pass_list   = [r for r in results[1:] if passes_all(r)]
    best_sharpe = max(results[1:], key=lambda r: r['Sharpe_OOS'])
    best_maxdd  = max(results[1:], key=lambda r: r['MaxDD_FULL'])
    improved_dd = [r for r in results[1:] if r['MaxDD_FULL'] > m_ref['MaxDD_FULL']]
    print(f'MaxDD改善 configs: {len(improved_dd)} / {total}')
    print(f'PASS configs: {len(pass_list)}')
    print(f'Best Sharpe: vov_thr={best_sharpe["vov_thr"]:.1f} α={best_sharpe["alpha_max"]:.2f} '
          f'→ Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%')
    print(f'Best MaxDD: vov_thr={best_maxdd["vov_thr"]:.1f} α={best_maxdd["alpha_max"]:.2f} '
          f'→ MaxDD={best_maxdd["MaxDD_FULL"]*100:+.2f}%  CAGR={best_maxdd["CAGR_OOS"]*100:+.2f}%')

    verdict = ('PASS' if pass_list else
               'WARN' if best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')
    print(f'総合判定: {verdict}')

    # --- CSV出力 ---
    pd.DataFrame([{
        'vov_thr': r.get('vov_thr', np.nan),
        'alpha_max': r['alpha_max'],
        'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'], 'Trades_yr': r['Trades_yr'],
        'Stress_pct': r.get('Stress_pct', np.nan),
    } for r in results]).to_csv(
        os.path.join(BASE, 'a3_regime_asset_tilt_results.csv'),
        index=False, float_format='%.6f')

    # --- MD出力 (MD_HEADER_1P 使用) ---
    hdr1, hdr2 = MD_HEADER_1P
    top_results = sorted(results[1:], key=lambda r: r['Sharpe_OOS'], reverse=True)
    rows_md = '\n'.join(
        fmt_row_1p(f'vov={r["vov_thr"]:.1f}/α={r["alpha_max"]:.2f}', r)
        for r in top_results
    )
    ref_row = fmt_row_1p('α=0 (REF=E4)', m_ref)

    dd_delta   = (best_maxdd['MaxDD_FULL'] - m_ref['MaxDD_FULL']) * 100
    cagr_delta = (best_maxdd['CAGR_OOS']   - m_ref['CAGR_OOS'])   * 100

    report = f"""\
# A3: Regime Asset Tilt (VoV + vz) — 高ストレス時 NASDAQ→Gold/Bond シフト

作成日: 2026-05-27
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

E4ベース（k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, l_max={L_MAX_FIXED}）完全維持。
追加で VoV(Vol-of-Vol) z-score と vz の二重ゲートで高ストレス日を検出し、
NASDAQ ウェイトを α 分削減して Gold/Bond に半々で再配分する。

| 項目 | 定義 |
|------|------|
| **E4固定値** | k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, l_max={L_MAX_FIXED}（変更なし） |
| **VOV_THR グリッド** | {VOV_THR_GRID} |
| **ALPHA_MAX グリッド** | {ALPHA_MAX_GRID} |
| **合計 configs** | {total} + REF |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**ストレス判定 / Tilt 式**:
```
vol20_t  = 20日実現ボラ（年率）
vov60_t  = vol20 の 60日ローリング std
vov_z_t  = (vov60 - expanding_mean) / expanding_std
stress_t = (vov_z_t > VOV_THR) AND (vz_t > {VZ_THR})

α_t   = ALPHA_MAX × stress_t
wn_t' = wn_A × (1 - α_t)
wg_t' = wg_A + wn_A × α_t × 0.5
wb_t' = wb_A + wn_A × α_t × 0.5
```

合計保存: wn'+wg'+wb' = wn_A+wg_A+wb_A

**サニティ**: REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (E4={REF_CAGR_OOS*100:+.2f}%, diff {diff_ref:+.2f}pp)

---

## §2 9指標テーブル（Sharpe 降順 + REF）

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
| (iv) MaxDD | > {PASS_MAXDD*100:.1f}% |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% |

- **PASS configs**: {len(pass_list)} / {total}
- **MaxDD改善 configs**: {len(improved_dd)} / {total}
- **最高 Sharpe**: vov={best_sharpe["vov_thr"]:.1f}, α={best_sharpe["alpha_max"]:.2f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}
- **最良 MaxDD**: vov={best_maxdd["vov_thr"]:.1f}, α={best_maxdd["alpha_max"]:.2f} → MaxDD={best_maxdd["MaxDD_FULL"]*100:+.2f}% (E4比 {dd_delta:+.2f}pp), CAGR_OOS={best_maxdd["CAGR_OOS"]*100:+.2f}% (E4比 {cagr_delta:+.2f}pp)
- **総合判定: {verdict}**

---

## §4 考察

VoV（Vol-of-Vol）は「ボラ自体のばらつき」を捉える二階指標で、レジーム変化に先行する性質がある。
vz の即時シグナルと組み合わせることで、構造的高ストレス局面のみピンポイントで検出可能。

- α=0.2〜0.4 の控えめなティルトでも MaxDD/Worst10Y 改善が見込まれる
- 自由度2（VOV_THR, ALPHA_MAX）。WFE 検証を推奨
- E4本体（k_dyn, l_max, S2）は不変なので、A3 が FAIL しても元の E4 に戻せる
- stress_pct（高ストレス日比率）が 1〜10% 程度が健全。過多なら VOV_THR を上げる方向

---

*生成スクリプト: `src/a3_regime_asset_tilt.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/e4_regime_klt.py`, `src/a2_dyn_lmax.py`*
"""
    md_path = os.path.join(BASE, 'A3_REGIME_ASSET_TILT_2026-05-27.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "a3_regime_asset_tilt_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
