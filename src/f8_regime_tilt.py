"""
F8: Regime-Conditional Bull-Tilt スイープ
==========================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

目的:
  F7-v3 で PASS した「定式 A: step-func bull-tilt (tilt=10, cap=0.10)」を
  ベースに、tilt の適用条件を VZ レジームで絞り込み、MaxDD 改善・
  Trades/yr 削減・Sharpe 維持/向上を狙う。

固定:
  - Base: E4 採用 config (k_lo=0.1, k_hi=0.8, vz_thr=0.7, k_mid=0.5, LT2-N750, mode B)
  - Tilt: step function (tilt=10.0 → ほぼ完全ステップ), TILT_CAP=0.10, THRESHOLD=0.15

グリッド (8 configs incl. REF):
  REF             — E4 (tilt なし)
  F7V3_BASE       — bull_mask = raw_a2 > 0.15
  R1_CALM         — bull_mask AND abs(vz) < 0.7
  R2_NO_BEAR_VZ   — bull_mask AND vz > -0.7
  R3_BULL_VZ      — bull_mask AND vz > +0.7
  R4_VZ_SCALED    — cap_eff = 0.10 * max(0, 1 - abs(vz))（VZ-scaled cap）
  R5_CALM_BOOST   — abs(vz)<0.7: cap=0.15 / vz>+0.7: cap=0.10 / vz<-0.7: cap=0.05
  R6_DOUBLE       — bull_mask AND vz > 0.0

PASS 基準:
  Sharpe_OOS ≥ REF + 0.020 = 0.9115
  CAGR_OOS   ≥ 0.295
  IS-OOS gap ≤ 6.0pp
  MaxDD      > -65.01%
  Worst10Y★  ≥ 15.0%
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
# 固定パラメータ
# ---------------------------------------------------------------------------
TILT      = 10.0          # 定式 A (step function) — F7-v3 採用値
TILT_CAP  = 0.10
VZ_REG    = 0.7           # VZ レジーム閾値 (calm / stressed の境界)

# E4 採用 config (k_lo=0.1, k_hi=0.8, vz_thr=0.7) — F8 base
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

# F7V3_BASE 参照値（F7-v3 A:tilt=10.0 から）— 比較用
F7V3_BASE_REF = dict(
    CAGR_OOS=0.365182, Sharpe_OOS=0.929319, MaxDD=-0.620361,
    Worst10Y=0.181124, Trades_yr=179.188093,
)


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


def compute_tilt_amount(config_id, raw_a2, vz, bull_mask):
    """config に応じて tilt_amount (配列) を返す。
    全 config 共通: step-func base (tilt=10, cap=0.10) を bull_mask に適用。
    各 config は (i) 追加マスク, または (ii) cap を VZ に応じて変える。
    """
    # step-func base: tilt=10 で raw_a2 > 0.15 ならほぼ +cap、それ以外 0
    tilt_raw = TILT * (raw_a2 - THRESHOLD) * (1.0 - raw_a2)
    tilt_base = np.clip(tilt_raw, 0.0, TILT_CAP)

    if config_id == 'F7V3_BASE':
        # bull_mask のみ (F7-v3 A:tilt=10 と同一)
        return np.where(bull_mask, tilt_base, 0.0)

    if config_id == 'R1_CALM':
        # calm regime: abs(vz) < VZ_REG
        mask = bull_mask & (np.abs(vz) < VZ_REG)
        return np.where(mask, tilt_base, 0.0)

    if config_id == 'R2_NO_BEAR_VZ':
        # bear VZ 除外: vz > -VZ_REG
        mask = bull_mask & (vz > -VZ_REG)
        return np.where(mask, tilt_base, 0.0)

    if config_id == 'R3_BULL_VZ':
        # bull VZ 限定: vz > +VZ_REG
        mask = bull_mask & (vz > VZ_REG)
        return np.where(mask, tilt_base, 0.0)

    if config_id == 'R4_VZ_SCALED':
        # VZ-scaled cap: cap_eff = 0.10 * max(0, 1 - |vz|)
        cap_eff = TILT_CAP * np.maximum(0.0, 1.0 - np.abs(vz))
        ta = np.clip(tilt_raw, 0.0, cap_eff)
        return np.where(bull_mask, ta, 0.0)

    if config_id == 'R5_CALM_BOOST':
        # calm: cap=0.15, bull-VZ: cap=0.10, bear-VZ: cap=0.05
        cap_eff = np.where(np.abs(vz) < VZ_REG, 0.15,
                  np.where(vz > VZ_REG, 0.10, 0.05))
        ta = np.minimum(np.maximum(tilt_raw, 0.0), cap_eff)
        return np.where(bull_mask, ta, 0.0)

    if config_id == 'R6_DOUBLE':
        # raw_a2 > 0.15 AND vz > 0.0 の複合
        mask = bull_mask & (vz > 0.0)
        return np.where(mask, tilt_base, 0.0)

    raise ValueError(f'Unknown config_id: {config_id}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 72)
    print('F8: Regime-Conditional Bull-Tilt スイープ')
    print('=' * 72)

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

    # VZ 分布のサニティ情報（あとで MD 用に）
    bull_count   = int(bull_mask.sum())
    bull_calm    = int((bull_mask & (np.abs(vz_arr) < VZ_REG)).sum())
    bull_no_bear = int((bull_mask & (vz_arr > -VZ_REG)).sum())
    bull_only    = int((bull_mask & (vz_arr > VZ_REG)).sum())
    bull_pos_vz  = int((bull_mask & (vz_arr > 0.0)).sum())
    print(f'Bull days total                : {bull_count:,} ({bull_count/n*100:5.1f}%)')
    print(f'  & calm (|vz|<0.7)            : {bull_calm:,} ({bull_calm/max(bull_count,1)*100:5.1f}% of bull)')
    print(f'  & no-bear-vz (vz>-0.7)       : {bull_no_bear:,} ({bull_no_bear/max(bull_count,1)*100:5.1f}% of bull)')
    print(f'  & bull-vz only (vz>+0.7)     : {bull_only:,} ({bull_only/max(bull_count,1)*100:5.1f}% of bull)')
    print(f'  & vz>0                       : {bull_pos_vz:,} ({bull_pos_vz/max(bull_count,1)*100:5.1f}% of bull)')

    print('Assets and signals built. Starting sweep...')

    # ---------------- REF (tilt = 0.0) ----------------
    nav_ref = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_ref, dates, n_trades_yr_ref)
    m_ref.update({'config_id': 'REF', 'Trades_yr': n_trades_yr_ref,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
    print(f'[REF tilt=0.00] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  '
          f'Sharpe={m_ref["Sharpe_OOS"]:+.3f}  MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%  '
          f'Tr/yr={n_trades_yr_ref:.1f}')
    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    print(f'[SANITY] diff vs E4 best CAGR={diff_ref:+.2f}pp → '
          f'{"OK" if abs(diff_ref) <= 0.15 else "WARN"}')

    results = [m_ref]

    # ---------------- F8 各 config ----------------
    config_ids = ['F7V3_BASE', 'R1_CALM', 'R2_NO_BEAR_VZ', 'R3_BULL_VZ',
                  'R4_VZ_SCALED', 'R5_CALM_BOOST', 'R6_DOUBLE']
    print(f'\n--- F8 configs ({len(config_ids)}) ---')
    idx = 0
    for cid in config_ids:
        idx += 1
        tilt_amount = compute_tilt_amount(cid, raw_a2_vals, vz_arr, bull_mask)

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
        m.update({'config_id': cid, 'Trades_yr': trades_yr,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
        # tilt 発火日数（diagnostic）
        m['tilt_days'] = int((tilt_amount > 0).sum())
        results.append(m)

        print(f'  [{idx}/{len(config_ids)}] {cid:<14s}  '
              f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
              f'Sharpe={m["Sharpe_OOS"]:+.3f}  '
              f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
              f'W10Y={m["Worst10Y_star"]*100:+.2f}%  '
              f'Gap={m["IS_OOS_gap"]*100:+.2f}pp  '
              f'Tr/yr={trades_yr:.1f}  '
              f'tilt_days={m["tilt_days"]:,}')

    print(f'\n全 {len(results)-1} configs 完了。')

    pass_list = [r for r in results[1:] if passes_all(r)]
    # WARN 基準: Sharpe > REF だが PASS 基準に1つでも届かない
    warn_list = [r for r in results[1:]
                 if (r['Sharpe_OOS'] > REF_SHARPE_OOS) and not passes_all(r)]
    fail_list = [r for r in results[1:]
                 if r['Sharpe_OOS'] <= REF_SHARPE_OOS]
    best_sharpe = max(results[1:], key=lambda r: r['Sharpe_OOS'])

    verdict = ('PASS' if pass_list else
               'WARN' if best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')

    print(f'\nPASS configs: {len(pass_list)} ({[r["config_id"] for r in pass_list]})')
    print(f'WARN configs: {len(warn_list)} ({[r["config_id"] for r in warn_list]})')
    print(f'FAIL configs: {len(fail_list)} ({[r["config_id"] for r in fail_list]})')
    print(f'Best Sharpe : {best_sharpe["config_id"]} → '
          f'Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  '
          f'CAGR_OOS={best_sharpe["CAGR_OOS"]*100:+.2f}%  '
          f'MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%  '
          f'IS-OOS={best_sharpe["IS_OOS_gap"]*100:+.2f}pp  '
          f'Tr/yr={best_sharpe["Trades_yr"]:.1f}')
    print(f'総合判定: {verdict}')

    # ---------------- CSV ----------------
    pd.DataFrame([{
        'config_id': r['config_id'],
        'CAGR_IS':   r['CAGR_IS'],
        'CAGR_OOS':  r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'],
        'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'],
        'P10_5Y':    r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'],
        'Trades_yr': r['Trades_yr'],
        'tilt_days': r.get('tilt_days', 0),
    } for r in results]).to_csv(os.path.join(BASE, 'f8_regime_tilt_results.csv'),
                                index=False, float_format='%.6f')

    # ---------------- MD ----------------
    hdr1, hdr2 = MD_HEADER_1P
    rows = []
    rows.append(fmt_row_1p('REF (E4, tilt=0)',    m_ref))
    rows.append(fmt_row_1p('F7V3_BASE',           results[1]))
    rows.append(fmt_row_1p('R1_CALM',             results[2]))
    rows.append(fmt_row_1p('R2_NO_BEAR_VZ',       results[3]))
    rows.append(fmt_row_1p('R3_BULL_VZ',          results[4]))
    rows.append(fmt_row_1p('R4_VZ_SCALED',        results[5]))
    rows.append(fmt_row_1p('R5_CALM_BOOST',       results[6]))
    rows.append(fmt_row_1p('R6_DOUBLE',           results[7]))
    table_body = '\n'.join(rows)

    # F7V3_BASE 比の差分計算
    f7v3 = results[1]
    delta_lines = []
    label_map = {
        'R1_CALM':       'R1 calm only',
        'R2_NO_BEAR_VZ': 'R2 no-bear-vz',
        'R3_BULL_VZ':    'R3 bull-vz only',
        'R4_VZ_SCALED':  'R4 vz-scaled cap',
        'R5_CALM_BOOST': 'R5 calm boost',
        'R6_DOUBLE':     'R6 vz>0',
    }
    for r in results[2:]:
        cid = r['config_id']
        dS  = r['Sharpe_OOS'] - f7v3['Sharpe_OOS']
        dC  = (r['CAGR_OOS'] - f7v3['CAGR_OOS']) * 100
        dM  = (r['MaxDD_FULL'] - f7v3['MaxDD_FULL']) * 100
        dT  = r['Trades_yr'] - f7v3['Trades_yr']
        delta_lines.append(
            f'| {label_map[cid]} '
            f'| {dS:+.3f} | {dC:+.2f}pp | {dM:+.2f}pp | {dT:+.1f} |'
        )
    delta_table = '\n'.join(delta_lines)

    # config 説明
    desc_table = """| config_id | tilt 適用条件 | 直感 |
|:----------|:--------------|:-----|
| REF | tilt なし | E4 採用 config そのまま |
| F7V3_BASE | raw_a2 > 0.15 | F7-v3 A:tilt=10（step function）と完全同一 |
| R1_CALM | raw_a2 > 0.15 AND \\|vz\\| < 0.7 | calm regime 限定。stressed 時の上振せ抑制 |
| R2_NO_BEAR_VZ | raw_a2 > 0.15 AND vz > -0.7 | bear VZ 除外（高ボラ売り局面では tilt しない） |
| R3_BULL_VZ | raw_a2 > 0.15 AND vz > +0.7 | bull VZ 限定（最も保守的・極端ケース） |
| R4_VZ_SCALED | cap_eff = 0.10 * max(0, 1 - \\|vz\\|) | VZ 大きいほど cap 縮小 |
| R5_CALM_BOOST | \\|vz\\|<0.7→cap=0.15 / vz>+0.7→cap=0.10 / vz<-0.7→cap=0.05 | calm 時 +50% 増量 |
| R6_DOUBLE | raw_a2 > 0.15 AND vz > 0.0 | vz > 0 = ボラ平均超え |"""

    # 判定理由（PASS / WARN / FAIL の内訳）
    if pass_list:
        verdict_detail = (
            f'**PASS**: 次の config が全基準クリア — '
            f'{", ".join(r["config_id"] for r in pass_list)}'
        )
    elif warn_list:
        verdict_detail = (
            f'**WARN**: Sharpe_OOS は REF 超え（>+{REF_SHARPE_OOS:.3f}）も、'
            f'PASS 基準のいずれかに未達 — {", ".join(r["config_id"] for r in warn_list)}'
        )
    else:
        verdict_detail = '**FAIL**: 全 config が REF Sharpe を超えず'

    pass_basis = ('Sharpe_OOS ≥ +0.9115、CAGR_OOS ≥ +29.5%、'
                  'IS-OOS gap ≤ +6.0pp、MaxDD > -65.01%、Worst10Y★ ≥ +15.0%')

    report = f"""\
# F8: Regime-Conditional Bull-Tilt スイープ

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

### 問題意識（F7-v3 からの継承）
F7-v3 で「定式 A: step-function tilt (tilt=10.0, cap=0.10)」が PASS 候補となった。
- F7V3_BASE: CAGR_OOS=+{F7V3_BASE_REF['CAGR_OOS']*100:.2f}%, Sharpe={F7V3_BASE_REF['Sharpe_OOS']:+.3f},
  MaxDD={F7V3_BASE_REF['MaxDD']*100:+.2f}%, Worst10Y★={F7V3_BASE_REF['Worst10Y']*100:+.2f}%,
  Trades/yr={F7V3_BASE_REF['Trades_yr']:.1f}

しかし副作用として:
1. **Trades/yr が REF 比 6.6× に急増**（27 → 179）→ 取引コスト感応度悪化
2. **MaxDD がやや拡大**（-60.0% → -62.0%）
3. **IS-OOS gap が拡大**（-1.8pp → -4.5pp）

### F8 設計の動機
tilt が**全 bull 日**に発火するのが過剰。VZ レジームで絞ることで以下を狙う:
- **calm 限定**: 高ボラ局面の上振せを止める → MaxDD 改善
- **bull-VZ 限定**: tilt 発火日を激減 → Trades/yr 削減
- **scaled cap**: 段階的縮小で滑らかな移行
- **calm boost**: calm 時のみ cap を引き上げる「逆張り」設計

### 共通設定
| 項目 | 定義 |
|------|------|
| Base config | E4 採用: k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, k_mid={K_MID}, LT2-N750, mode B |
| Tilt 定式 | step-function: clip(tilt × (raw_a2-0.15) × (1-raw_a2), 0, cap), tilt={TILT} |
| TILT_CAP (base) | {TILT_CAP} |
| VZ_REG 閾値 | ±{VZ_REG} （calm / stressed の境界） |
| THRESHOLD | {THRESHOLD} (raw_a2 bull 判定) |
| wn 調整 | wn_tilted = wn_A + tilt_amount |
| wb 調整 | wb_tilted = max(wb_A - tilt_amount, 0) |
| IS  | {IS_START} 〜 {IS_END} |
| OOS | {OOS_START} 〜 |

**サニティ**: REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (diff vs E4 best CAGR={diff_ref:+.2f}pp → {"OK" if abs(diff_ref) <= 0.15 else "WARN"})

### config 一覧

{desc_table}

### Bull-day VZ 分布（diagnostic）
- Bull days total (raw_a2 > 0.15): **{bull_count:,}** / {n:,} ({bull_count/n*100:.1f}%)
- ∩ calm (|vz|<{VZ_REG}): {bull_calm:,} ({bull_calm/max(bull_count,1)*100:.1f}% of bull)
- ∩ no-bear-vz (vz>-{VZ_REG}): {bull_no_bear:,} ({bull_no_bear/max(bull_count,1)*100:.1f}% of bull)
- ∩ bull-vz only (vz>+{VZ_REG}): {bull_only:,} ({bull_only/max(bull_count,1)*100:.1f}% of bull)
- ∩ vz>0: {bull_pos_vz:,} ({bull_pos_vz/max(bull_count,1)*100:.1f}% of bull)

---

## §2 9指標テーブル

{hdr1}
{hdr2}
{table_body}

{MD_WFA_NOTE}

---

## §3 考察

### F7V3_BASE 比 差分（同 base, レジーム条件のみ追加）

| config | ΔSharpe | ΔCAGR_OOS | ΔMaxDD | ΔTrades/yr |
|:-------|--------:|----------:|-------:|-----------:|
{delta_table}

### 最良 config
**{best_sharpe["config_id"]}** が Sharpe_OOS={best_sharpe["Sharpe_OOS"]:+.3f} で最大。
- vs REF: ΔSharpe={best_sharpe["Sharpe_OOS"]-REF_SHARPE_OOS:+.3f}, ΔCAGR={(best_sharpe["CAGR_OOS"]-REF_CAGR_OOS)*100:+.2f}pp,
  ΔMaxDD={(best_sharpe["MaxDD_FULL"]-REF_MAXDD)*100:+.2f}pp, ΔTr/yr={best_sharpe["Trades_yr"]-n_trades_yr_ref:+.1f}
- vs F7V3_BASE: ΔSharpe={best_sharpe["Sharpe_OOS"]-f7v3["Sharpe_OOS"]:+.3f},
  ΔCAGR={(best_sharpe["CAGR_OOS"]-f7v3["CAGR_OOS"])*100:+.2f}pp,
  ΔMaxDD={(best_sharpe["MaxDD_FULL"]-f7v3["MaxDD_FULL"])*100:+.2f}pp,
  ΔTr/yr={best_sharpe["Trades_yr"]-f7v3["Trades_yr"]:+.1f}

### レジーム条件の効果（観察ポイント）
- **Trades/yr 削減**: tilt 発火日が減るほど Trades/yr が落ちる。
  R3_BULL_VZ（最少発火）と R1_CALM, R6_DOUBLE で比較。
- **MaxDD 改善**: bear-vz 局面で tilt しない R2/R3 で MaxDD 改善が期待。
- **Sharpe 維持**: F7V3_BASE の Sharpe={f7v3["Sharpe_OOS"]:+.3f} に対し、レジーム
  限定で Sharpe を維持/向上できるかが採用判定の鍵。
- **R4_VZ_SCALED / R5_CALM_BOOST**: 段階的設計が単純マスクより優位かをチェック。

---

## §4 判定

| 基準 | 条件 |
|------|------|
| (i)   Sharpe_OOS  | ≥ REF + 0.020 (≥ {REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}) |
| (ii)  CAGR_OOS    | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap  | ≤ {PASS_GAP*100:.1f}pp |
| (iv)  MaxDD       | > {PASS_MAXDD*100:.2f}% |
| (v)   Worst10Y★   | ≥ {PASS_WORST10Y*100:.1f}% |

- PASS configs: **{len(pass_list)} / 7** {[r['config_id'] for r in pass_list] if pass_list else ''}
- WARN configs: {len(warn_list)} {[r['config_id'] for r in warn_list] if warn_list else ''}
- FAIL configs: {len(fail_list)} {[r['config_id'] for r in fail_list] if fail_list else ''}
- 最高 Sharpe: **{best_sharpe['config_id']}** (Sharpe={best_sharpe['Sharpe_OOS']:+.3f}, CAGR_OOS={best_sharpe['CAGR_OOS']*100:+.2f}%, Tr/yr={best_sharpe['Trades_yr']:.1f})

### 総合判定: {verdict}

{verdict_detail}

PASS 基準（§3.12 v1.1 準拠）: {pass_basis}

---

*生成スクリプト: `src/f8_regime_tilt.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/f7v3_bull_tilt.py`, `src/e4_regime_klt.py`*
"""
    md_path = os.path.join(BASE, 'F8_REGIME_TILT_2026-05-24.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "f8_regime_tilt_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
