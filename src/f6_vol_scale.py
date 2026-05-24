"""
F6: Portfolio Vol-Scaled Sleeve Weights スイープ
================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

仮説:
  実現ポートフォリオボラ (20D) が target_port_vol を超えたら、
  全スリーブ重み (wn/wg/wb) を同時に scale down する。
  Sharpe を直接ターゲットにするリスク低減策。

機構:
  - r_port_est = wn*lev_mod*L_s2_lag*ret + wg*r_gold2x + wb*r_bond3x
  - sigma_port = rolling(20).std() * sqrt(252)
  - port_scale = clip(target_vol / sigma_port_lag1, scale_min, 1.0)
  - 全 sleeve weight に乗算（1日ラグ適用済）

E4 ベース (lev_mod 計算):
  K_LO=0.1, K_HI=0.8, VZ_THR=0.7, K_MID=0.5, LT2 N=750 (現行 Active)

グリッド:
  TARGET_VOL_GRID = [0.12, 0.15, 0.18, 0.22]
  SCALE_MIN_GRID  = [0.50, 0.70]
  合計: 4×2=8 configs + REF
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
TARGET_VOL_GRID = [0.12, 0.15, 0.18, 0.22]
SCALE_MIN_GRID  = [0.50, 0.70]

# E4 base (現行 Active)
K_LO, K_HI, VZ_THR, K_MID = 0.1, 0.8, 0.7, 0.5
LT2_N = 750

# REF (E4 current best)
REF_CAGR_OOS   = 0.3353
REF_SHARPE_OOS = 0.8915
REF_MAXDD      = -0.6001

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)

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


def _bfill(s: pd.Series) -> pd.Series:
    """pandas 互換性: 新旧両対応の bfill"""
    try:
        return s.bfill()
    except AttributeError:
        return s.fillna(method='bfill')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('F6: Portfolio Vol-Scaled Sleeve Weights スイープ')
    print('=' * 70)
    total = len(TARGET_VOL_GRID) * len(SCALE_MIN_GRID)
    print(f'グリッド: target_vol={TARGET_VOL_GRID} × scale_min={SCALE_MIN_GRID}  ({total} configs + REF)')
    print(f'E4 base: K_LO={K_LO}, K_HI={K_HI}, VZ_THR={VZ_THR}, K_MID={K_MID}, LT2 N={LT2_N}')

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

    # E4 base lev_mod
    lt_sig_raw = build_lt_signal(close, 'LT2', LT2_N)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    k_dyn      = np.where(vz_arr > VZ_THR, K_HI,
                          np.where(vz_arr < -VZ_THR, K_LO, K_MID))
    lt_bias    = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                           index=lt_sig_raw.index)
    lev_mod    = apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)

    # ポートフォリオ実現ボラ計算用構成要素 (lookahead 回避)
    r_gold2x = pd.Series(gold_2x).pct_change().fillna(0).values
    r_bond3x = pd.Series(bond_3x).pct_change().fillna(0).values
    L_s2_lag = np.roll(L_s2.values, 1); L_s2_lag[0] = L_s2.values[0]
    r_port_est = (wn_A * np.asarray(lev_mod) * L_s2_lag * ret.values
                  + wg_A * r_gold2x
                  + wb_A * r_bond3x)
    sigma_port_raw = pd.Series(r_port_est).rolling(20).std()
    sigma_port = _bfill(sigma_port_raw) * np.sqrt(252)

    print('Assets and signals built. Starting sweep...')

    # REF (no scaling — port_scale = 1.0 ones)
    nav_ref = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_ref, dates, n_trades_yr)
    m_ref.update({'target_vol': float('nan'), 'scale_min': float('nan'),
                  'Trades_yr': n_trades_yr,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
    print(f'[REF no-scale] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  '
          f'Sharpe={m_ref["Sharpe_OOS"]:+.3f}  MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%')
    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    print(f'[SANITY] diff vs E4 ref={diff_ref:+.2f}pp → {"OK" if abs(diff_ref) <= 0.15 else "WARN"}')

    results = [m_ref]
    idx = 0
    sigma_lag = sigma_port.shift(1)
    sigma_lag.iloc[0] = sigma_port.iloc[0]

    for target_vol in TARGET_VOL_GRID:
        for scale_min in SCALE_MIN_GRID:
            idx += 1
            port_scale = np.clip(target_vol / sigma_lag.values, scale_min, 1.0)
            wn_scaled = wn_A * port_scale
            wg_scaled = wg_A * port_scale
            wb_scaled = wb_A * port_scale

            nav = build_nav_strategy(
                close, lev_mod, wn_scaled, wg_scaled, wb_scaled, dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
            )
            m = calc_all_metrics(nav, dates, n_trades_yr)
            m.update({'target_vol': target_vol, 'scale_min': scale_min,
                      'Trades_yr': n_trades_yr,
                      'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
            results.append(m)
            tv_pct = target_vol * 100
            print(f'  [{idx}/{total}] tv={tv_pct:.0f}%/smin={scale_min:.2f}: '
                  f'CAGR={m["CAGR_OOS"]*100:+.2f}%  Sharpe={m["Sharpe_OOS"]:+.3f}  '
                  f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  W10Y={m["Worst10Y_star"]*100:+.2f}%')

    print(f'\n全 {len(results)-1} configs 完了。')

    pass_list  = [r for r in results[1:] if passes_all(r)]
    best_sharpe= max(results[1:], key=lambda r: r['Sharpe_OOS'])
    verdict    = ('PASS' if pass_list else
                  'WARN' if best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')
    print(f'PASS configs: {len(pass_list)}')
    print(f'Best Sharpe: tv={best_sharpe["target_vol"]*100:.0f}% / smin={best_sharpe["scale_min"]:.2f} '
          f'→ Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  CAGR_OOS={best_sharpe["CAGR_OOS"]*100:+.2f}%  '
          f'MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%  IS-OOS={best_sharpe["IS_OOS_gap"]*100:+.2f}pp')
    print(f'総合判定: {verdict}')

    # CSV
    pd.DataFrame([{
        'target_vol': r['target_vol'], 'scale_min': r['scale_min'],
        'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'], 'Trades_yr': r['Trades_yr'],
    } for r in results]).to_csv(os.path.join(BASE, 'f6_vol_scale_results.csv'),
                                index=False, float_format='%.6f')

    # MD: 全 configs（8件）を Sharpe 降順 + REF
    hdr1, hdr2 = MD_HEADER_1P
    sorted_results = sorted(results[1:], key=lambda r: r['Sharpe_OOS'], reverse=True)
    rows_md = '\n'.join(
        fmt_row_1p(f'tv={r["target_vol"]*100:.0f}%/smin={r["scale_min"]:.2f}', r)
        for r in sorted_results
    )
    ref_row = fmt_row_1p('REF (no-scale)', m_ref)

    report = f"""\
# F6: Portfolio Vol-Scaled Sleeve Weights スイープ

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

| 項目 | 定義 |
|------|------|
| **target_vol グリッド** | {TARGET_VOL_GRID}（年率ボラ目標） |
| **scale_min グリッド** | {SCALE_MIN_GRID}（縮小下限） |
| **E4 base** | K_LO={K_LO}, K_HI={K_HI}, VZ_THR={VZ_THR}, K_MID={K_MID}, LT2 N={LT2_N} |
| **合計 configs** | {total} + REF |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**機構**:
```
r_port_est_t = wn*lev_mod*L_s2_lag*ret + wg*r_gold2x + wb*r_bond3x
sigma_port_t = rolling(20).std(r_port_est) * sqrt(252)
port_scale_t = clip(target_vol / sigma_port_{{t-1}}, scale_min, 1.0)
wn'_t = wn*port_scale_t, wg'_t = wg*port_scale_t, wb'_t = wb*port_scale_t
```
1日ラグ + L_s2 ラグで lookahead 回避。常に scale down のみ（≤1.0）。

**サニティ**: REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (diff {diff_ref:+.2f}pp vs E4-ref {REF_CAGR_OOS*100:+.2f}%)

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
| (i) Sharpe_OOS | ≥ REF+0.020 (≥{REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}) |
| (ii) CAGR_OOS | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap | ≤ {PASS_GAP*100:.1f}pp |
| (iv) MaxDD | > {PASS_MAXDD*100:.2f}% |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% |

- **PASS configs**: {len(pass_list)} / {total}
- **最高 Sharpe**: tv={best_sharpe["target_vol"]*100:.0f}% / smin={best_sharpe["scale_min"]:.2f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}
- **総合判定: {verdict}**

---

## §4 考察

ポートフォリオ実現ボラを直接ターゲットにする手法:
- 全 sleeve 同時 scale down → CFD レバレッジ (L_s2) は据え置きで、エクスポージャ全体だけを縮小
- target_vol が低いほど縮小頻度↑ → MaxDD ↓ / CAGR ↓ のトレードオフ
- scale_min が小さいほど深い縮小可能 → 危機時の保護↑ / 通常時のドラッグ↑
- 1日ラグ + L_s2 1日ラグの二重ラグで lookahead 完全回避
- E4 と直交（k_lt は LT2 バイアス、port_scale は全 sleeve エクスポージャ）

---

*生成スクリプト: `src/f6_vol_scale.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `E4_REGIME_KLT_SWEEP_2026-05-24.md`*
"""
    md_path = os.path.join(BASE, 'F6_VOL_SCALE_2026-05-24.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "f6_vol_scale_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
