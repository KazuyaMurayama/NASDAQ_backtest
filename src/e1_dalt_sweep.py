"""
E1: DALT — Drawdown-Aware Leverage Throttle スイープ
=====================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

概念: ポートフォリオの実現 DrawDown に応じて lev_mod を抑制するブレーキ機構。
  dd_brake_t = clip(1 + dd_t / dd_ref, dd_floor, 1.0)   (1日ラグ適用で look-ahead 回避)
  lev_mod_dalt = lev_mod * dd_brake_t

  dd_t < 0 → dd_brake < 1.0 → レバレッジ抑制 → MaxDD 改善期待
  dd_ref  : ブレーキが 0 になる DD 深さ（例: -0.30 = 30% DD で floor まで縮小）
  dd_floor: ブレーキの最小値（例: 0.40 = 最大40%のレバに圧縮）

  ベース NAV（ブレーキなし）から DD を計算し、1日ラグで次日の lev_mod に適用。
  これにより循環依存を回避し look-ahead バイアスゼロを保証。

グリッド:
  dd_ref_grid   = [0.20, 0.25, 0.30, 0.35, 0.40]  (5値)
  dd_floor_grid = [0.30, 0.40, 0.50, 0.60]         (4値)
  合計: 5×4=20 configs + REF (DALT なし)

出力:
  - e1_dalt_results.csv
  - E1_DALT_SWEEP_2026-05-24.md
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
from long_cycle_signal import build_lt_signal, signal_to_bias, apply_lt_mode_b
from _sweep_format import MD_HEADER_1P, fmt_row_1p, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# グリッド
# ---------------------------------------------------------------------------
DD_REF_GRID   = [0.20, 0.25, 0.30, 0.35, 0.40]
DD_FLOOR_GRID = [0.30, 0.40, 0.50, 0.60]

REF_CAGR_OOS   = 0.3116
REF_SHARPE_OOS = 0.858
REF_MAXDD      = -0.5945

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)

# DALT PASS 基準（MaxDD 改善フォーカス）
PASS_MAXDD_IMPROVE = -0.5500   # REF -59.45% から -55.00% 以浅に
PASS_CAGR_OOS      = 0.2600   # CAGR +26%以上（最大-5.16pp 許容）
PASS_SHARPE_MIN    = 0.720    # Sharpe ≥ 0.720（ある程度許容）
PASS_WORST10Y      = 0.150    # Worst10Y★ guardrail
PASS_GAP_MAX       = 0.060    # IS-OOS gap ≤ 6.0pp


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


def apply_dalt(nav_base_values: np.ndarray,
               lev_mod: np.ndarray,
               dd_ref: float,
               dd_floor: float) -> np.ndarray:
    """ベース NAV から DD を計算し、1日ラグで lev_mod にブレーキを適用。"""
    peak = np.maximum.accumulate(nav_base_values)
    dd   = nav_base_values / peak - 1.0          # ≤ 0
    # dd_brake: DD = 0 → brake=1.0; DD = -dd_ref → brake=dd_floor
    dd_brake = np.clip(1.0 + dd / dd_ref, dd_floor, 1.0)   # dd<0, dd_ref>0 → dd/dd_ref<0 → brake<1
    # 1日ラグ: 昨日の DD → 今日のブレーキ
    dd_brake_lag = np.roll(dd_brake, 1)
    dd_brake_lag[0] = 1.0
    return np.clip(lev_mod * dd_brake_lag, 0.0, 1.0)


def passes_dalt(r):
    return (r['MaxDD_FULL']     > PASS_MAXDD_IMPROVE and
            r['CAGR_OOS']       >= PASS_CAGR_OOS and
            r['Sharpe_OOS']     >= PASS_SHARPE_MIN and
            r['Worst10Y_star']  >= PASS_WORST10Y and
            r['IS_OOS_gap']     <= PASS_GAP_MAX)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('E1: DALT — Drawdown-Aware Leverage Throttle スイープ')
    print('=' * 70)
    total = len(DD_REF_GRID) * len(DD_FLOOR_GRID)
    print(f'グリッド: dd_ref={DD_REF_GRID} × dd_floor={DD_FLOOR_GRID}  ({total} configs + REF)')

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

    lt_sig  = build_lt_signal(close, 'LT2', 750)
    lt_bias = signal_to_bias(lt_sig, 0.5)
    lev_mod_base = apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)

    # ベース NAV（DALT なし = REF）
    nav_base = build_nav_strategy(
        close, lev_mod_base, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_base, dates, n_trades_yr)
    m_ref.update({'dd_ref': 0.0, 'dd_floor': 1.0,
                  'Trades_yr': n_trades_yr,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan'),
                  'label': 'REF (DALT なし)'})
    print(f'\n[REF] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  Sharpe={m_ref["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%  Worst10Y★={m_ref["Worst10Y_star"]*100:+.2f}%')

    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    sanity_ok = abs(diff_ref) <= 0.15
    print(f'[SANITY] CAGR_OOS diff={diff_ref:+.2f}pp → {"OK" if sanity_ok else "WARN"}')

    nav_base_arr = np.array(nav_base)
    results = [m_ref]
    idx = 0

    for dd_ref in DD_REF_GRID:
        for dd_floor in DD_FLOOR_GRID:
            idx += 1
            lev_mod_dalt = apply_dalt(nav_base_arr, lev_mod_base, dd_ref, dd_floor)
            nav_dalt = build_nav_strategy(
                close, lev_mod_dalt, wn_A, wg_A, wb_A, dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
            )
            m = calc_all_metrics(nav_dalt, dates, n_trades_yr)
            m.update({'dd_ref': dd_ref, 'dd_floor': dd_floor,
                      'Trades_yr': n_trades_yr,
                      'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan'),
                      'label': f'dd_ref={dd_ref:.2f} / floor={dd_floor:.2f}'})
            results.append(m)
            print(f'  [{idx:>2d}/{total}] dd_ref={dd_ref:.2f} floor={dd_floor:.2f}: '
                  f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  Sharpe={m["Sharpe_OOS"]:+.3f}  '
                  f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  Worst10Y★={m["Worst10Y_star"]*100:+.2f}%  '
                  f'IS-OOS={m["IS_OOS_gap"]*100:+.2f}pp')

    pass_list = [r for r in results[1:] if passes_dalt(r)]
    best_maxdd  = min(results[1:], key=lambda r: r['MaxDD_FULL'])   # most negative = worst
    best_maxdd  = max(results[1:], key=lambda r: -r['MaxDD_FULL'])  # least negative = best
    best_sharpe = max(results[1:], key=lambda r: r['Sharpe_OOS'])
    verdict = 'PASS' if pass_list else 'WARN' if best_maxdd['MaxDD_FULL'] > REF_MAXDD else 'FAIL'

    print(f'\nPASS configs: {len(pass_list)}')
    print(f'Best MaxDD: dd_ref={best_maxdd["dd_ref"]:.2f} floor={best_maxdd["dd_floor"]:.2f} '
          f'→ {best_maxdd["MaxDD_FULL"]*100:+.2f}%  Sharpe={best_maxdd["Sharpe_OOS"]:+.3f}')
    print(f'Best Sharpe: dd_ref={best_sharpe["dd_ref"]:.2f} floor={best_sharpe["dd_floor"]:.2f} '
          f'→ {best_sharpe["Sharpe_OOS"]:+.3f}  MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%')
    print(f'総合判定: {verdict}')

    # CSV
    pd.DataFrame([{
        'dd_ref': r['dd_ref'], 'dd_floor': r['dd_floor'],
        'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'], 'Trades_yr': r['Trades_yr'],
    } for r in results]).to_csv(os.path.join(BASE, 'e1_dalt_results.csv'),
                                index=False, float_format='%.6f')

    # MD レポート
    hdr1, hdr2 = MD_HEADER_1P
    rows_md = '\n'.join(
        fmt_row_1p(
            'REF (DALT なし)' if r['dd_ref'] == 0.0
            else f'ref={r["dd_ref"]:.2f}/floor={r["dd_floor"]:.2f}',
            r
        ) for r in results
    )

    report = f"""\
# E1: DALT — Drawdown-Aware Leverage Throttle スイープ

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

| 項目 | 定義 |
|------|------|
| **dd_ref グリッド** | {DD_REF_GRID}（ブレーキが効き始めるDD深さ） |
| **dd_floor グリッド** | {DD_FLOOR_GRID}（ブレーキの最小係数） |
| **合計 configs** | {total} + REF（DALT なし） |
| **ベース戦略** | S2+LT2-N750-k0.5-modeB（REF） |
| **目的** | DrawDown 時にレバ自動縮小 → MaxDD 改善 |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**DALT ロジック**:
```
dd_t       = nav_t / peak_t - 1         # 現在 DD（≤ 0）
dd_brake_t = clip(1 + dd_t / dd_ref, dd_floor, 1.0)
             ↑ dd=0 → brake=1.0  / dd=-dd_ref → brake=dd_floor
lev_mod_dalt_t = lev_mod_t × dd_brake_lag(t-1)  # 1日ラグ
```

**サニティ**: REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (ref +31.16%, diff {diff_ref:+.2f}pp) → {"OK" if sanity_ok else "WARN"}

---

## §2 9指標テーブル（dd_ref / dd_floor 組合せ）

{hdr1}
{hdr2}
{rows_md}

{MD_WFA_NOTE}

---

## §3 判定

**DALT PASS 基準**（MaxDD 改善フォーカス）:
- MaxDD > {PASS_MAXDD_IMPROVE*100:.1f}%（REF -59.45% から -55.00% 以浅）
- CAGR_OOS ≥ {PASS_CAGR_OOS*100:.1f}%（REF -5pp 許容）
- Sharpe_OOS ≥ {PASS_SHARPE_MIN:.3f}
- Worst10Y★ ≥ {PASS_WORST10Y*100:.1f}%
- IS-OOS gap ≤ {PASS_GAP_MAX*100:.1f}pp

- **PASS configs**: {len(pass_list)} / {total}（REF除く）
- **MaxDD 最良**: dd_ref={best_maxdd["dd_ref"]:.2f}, floor={best_maxdd["dd_floor"]:.2f} → MaxDD={best_maxdd["MaxDD_FULL"]*100:+.2f}%, Sharpe={best_maxdd["Sharpe_OOS"]:+.3f}, CAGR={best_maxdd["CAGR_OOS"]*100:+.2f}%
- **Sharpe 最良**: dd_ref={best_sharpe["dd_ref"]:.2f}, floor={best_sharpe["dd_floor"]:.2f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}, MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%
- **総合判定: {verdict}**

---

## §4 考察

DALT のトレードオフ:
- `dd_ref` 小 → ブレーキが早期に効く → MaxDD 早期抑制、ただし V字回復を逃す
- `dd_ref` 大 → ブレーキが遅い → MaxDD 改善小、CAGR 喪失少
- `dd_floor` 小 → 深 DD 時のレバを大幅縮小 → MaxDD 大幅改善、CAGR 犠牲大
- `dd_floor` 大 → レバ縮小が緩やか → MaxDD 改善小、CAGR 維持

ベース NAV から DD を計算している点に注意: DALT 適用後の NAV とは DD が異なる（conservative 近似）。
厳密な実装には反復計算が必要だが、実用上の影響は小さい。

---

*生成スクリプト: `src/e1_dalt_sweep.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`*
"""
    md_path = os.path.join(BASE, 'E1_DALT_SWEEP_2026-05-24.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "e1_dalt_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
