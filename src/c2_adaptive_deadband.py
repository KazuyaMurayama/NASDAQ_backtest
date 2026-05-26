"""
C2: Adaptive Deadband（ε_t=σ連動）
====================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-27)

概念: F10系のε=0.015固定deadbandを、市場ボラティリティに連動して動的変更。
  ε_t = ε_0 × (σ_t / σ̄)
  - 高ボラ時: εが大きくなり不要な取引を抑制
  - 低ボラ時: εが小さくなりシグナルに素早く反応
  CFAR（Constant False Alarm Rate）原理の適用。

ベース: F10 ε-Deadband (f10_epsilon_deadband.py) の構造を踏襲。
REF行: F10 ε=0.015固定deadband

グリッド:
  EPS_BASE_GRID = [0.010, 0.015, 0.020]  # ε_0基準値
  SIGMA_LOOKBACK = 250  # σ̄の計算窓
  vol_ratio = σ_20d / σ̄_250d (clip 0.3~3.0)
  ε_t = ε_0 × vol_ratio

採用条件 (優先順):
  (1) Trades/yr ≤ 70
  (2) Sharpe_OOS ≥ 0.891 (現行ベスト E4 以上)
  (3) IS-OOS gap ≤ 6.0pp
  (4) MaxDD > -65% (望ましい), > -80% (絶対)

E4 現行ベスト基準値 (REF):
  Sharpe=+0.891, CAGR_OOS=+33.53%, MaxDD=-60.01%, Trades/yr=27,
  Worst10Y★=+18.67%, IS-OOS gap=-1.81pp
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
# 固定パラメータ (F8-R5 CALM_BOOST / F10踏襲)
# ---------------------------------------------------------------------------
TILT      = 10.0
VZ_REG    = 0.7
CAP_CALM  = 0.15
CAP_BULL  = 0.10
CAP_BEAR  = 0.05

# E4 採用 config
K_LO, K_HI, VZ_THR, K_MID = 0.1, 0.8, 0.7, 0.5

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)

# C2 グリッド
EPS_BASE_GRID = [0.010, 0.015, 0.020]   # ε_0基準値
SIGMA_LOOKBACK = 250                     # σ̄の計算窓 (営業日)
VOL_WINDOW = 20                          # 実現ボラ計算窓

# REF: F10固定ε (ε=0.015)
EPS_FIXED_REF = 0.015

# 現行ベスト E4
REF_CAGR_OOS    = 0.3353
REF_SHARPE_OOS  = 0.8915
REF_MAXDD       = -0.6001
REF_TRADES_YR   = 27.1
REF_WORST10Y    = 0.1867
REF_GAP         = -0.0181

# F10 ε=0.015固定 (REF比較用プレースホルダ、実行時に算出)
F10_REF_CAGR_OOS   = None
F10_REF_SHARPE     = None
F10_REF_TRADES_YR  = None

# 採用判定
ADOPT_TRADES_YR = 70.0
ADOPT_SHARPE    = REF_SHARPE_OOS
ADOPT_GAP       = 0.060
ADOPT_MAXDD     = -0.65


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


def compute_tilt_with_deadband(raw_a2, vz, bull_mask, eps):
    """F10固定ε版 (REF用)。

    各日:
      raw_tilt = clip(TILT * (raw_a2 - 0.15) * (1 - raw_a2), 0, cap_eff)
      bull_mask が False の日 → raw_tilt = 0
      |raw_tilt - cur_tilt| >= eps の時のみ cur_tilt を更新
    """
    n = len(raw_a2)
    cap_eff = np.where(np.abs(vz) < VZ_REG, CAP_CALM,
              np.where(vz > VZ_REG, CAP_BULL, CAP_BEAR))

    tilt_raw = TILT * (raw_a2 - THRESHOLD) * (1.0 - raw_a2)
    tilt_target = np.minimum(np.maximum(tilt_raw, 0.0), cap_eff)
    tilt_target = np.where(bull_mask, tilt_target, 0.0)

    confirmed = np.zeros(n, dtype=float)
    cur = 0.0
    n_updates = 0
    for i in range(n):
        if i == 0 or abs(tilt_target[i] - cur) >= eps:
            cur = tilt_target[i]
            n_updates += 1
        confirmed[i] = cur
    return confirmed, n_updates


def compute_tilt_with_adaptive_deadband(raw_a2, vz, bull_mask, eps_series):
    """C2 Adaptive Deadband: ε_t=σ連動の時変εでtiltを確定。

    Parameters
    ----------
    raw_a2 : np.ndarray  — A2シグナル
    vz     : np.ndarray  — ボラティリティZスコア
    bull_mask : np.ndarray (bool) — bull判定マスク
    eps_series : np.ndarray  — 時変ε (values配列、raw_a2と同長)

    Returns
    -------
    confirmed : np.ndarray — 確定済みtilt系列
    n_updates : int       — tilt更新回数
    """
    n = len(raw_a2)
    cap_eff = np.where(np.abs(vz) < VZ_REG, CAP_CALM,
              np.where(vz > VZ_REG, CAP_BULL, CAP_BEAR))

    tilt_raw = TILT * (raw_a2 - THRESHOLD) * (1.0 - raw_a2)
    tilt_target = np.minimum(np.maximum(tilt_raw, 0.0), cap_eff)
    tilt_target = np.where(bull_mask, tilt_target, 0.0)

    confirmed = np.zeros(n, dtype=float)
    cur = 0.0
    n_updates = 0
    for i in range(n):
        eps_i = float(eps_series[i])          # 時変ε（当日のσ連動値）
        if i == 0 or abs(tilt_target[i] - cur) >= eps_i:
            cur = tilt_target[i]
            n_updates += 1
        confirmed[i] = cur
    return confirmed, n_updates


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 72)
    print('C2: Adaptive Deadband（ε_t=σ連動）')
    print('=' * 72)
    print(f'ε_0 グリッド: {EPS_BASE_GRID}  (vol_ratio窓: {VOL_WINDOW}d/{SIGMA_LOOKBACK}d)')

    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({n:,} days, {n_years:.2f} yr)')

    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)

    # E4 base
    lt_sig_raw = build_lt_signal(close, 'LT2', 750)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    k_dyn = np.where(vz_arr > VZ_THR, K_HI,
                     np.where(vz_arr < -VZ_THR, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=lt_sig_raw.index)
    lev_mod = apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)
    lev_mod_arr = np.asarray(lev_mod)
    lev_raw_arr = np.asarray(lev_raw)

    raw_a2_vals = raw_a2.values if hasattr(raw_a2, 'values') else np.asarray(raw_a2)
    bull_mask   = raw_a2_vals > THRESHOLD

    # ---------------------------------------------------------------------------
    # σ連動ε計算
    # ---------------------------------------------------------------------------
    # 20日実現ボラ（年率）
    vol_20d  = ret.rolling(VOL_WINDOW).std() * np.sqrt(252)
    # 250日平均ボラ
    vol_mean = vol_20d.rolling(SIGMA_LOOKBACK).mean()
    # vol_ratio: clip 0.3~3.0、NaN→1.0
    vol_ratio = (vol_20d / vol_mean).fillna(1.0).clip(0.3, 3.0)
    vol_ratio_vals = vol_ratio.values

    print(f'vol_ratio: mean={vol_ratio.mean():.3f}, std={vol_ratio.std():.3f}, '
          f'min={vol_ratio.min():.3f}, max={vol_ratio.max():.3f}')
    print(f'Bull days: {int(bull_mask.sum()):,} / {n:,} ({bull_mask.sum()/n*100:.1f}%)')

    print('Assets and signals built. Starting sweep...')

    results = []

    # ---------------------------------------------------------------------------
    # REF: F10 ε=0.015固定
    # ---------------------------------------------------------------------------
    eps_ref = EPS_FIXED_REF
    tilt_ref, n_updates_ref = compute_tilt_with_deadband(
        raw_a2_vals, vz_arr, bull_mask, eps_ref
    )
    wn_ref = wn_A + tilt_ref
    wb_ref = np.clip(wb_A - tilt_ref, 0.0, wb_A)
    n_tr_ref   = count_trades_tilted(wn_ref, wb_ref, lev_raw_arr)
    trades_yr_ref = n_tr_ref / n_years
    nav_ref = build_nav_strategy(
        close, lev_mod, wn_ref, wg_A, wb_ref, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_ref, dates, trades_yr_ref)
    m_ref.update({
        'label': f'ε=0.015/fixed',
        'eps_0': eps_ref,
        'mode':  'fixed',
        'Trades_yr':    trades_yr_ref,
        'tilt_updates': n_updates_ref,
        'WFA_CI95_lo':  float('nan'),
        'WFA_WFE':      float('nan'),
    })
    results.append(m_ref)

    print(f'  [REF] ε={eps_ref:.3f}/fixed  '
          f'CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  '
          f'Sharpe={m_ref["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%  '
          f'W10Y={m_ref["Worst10Y_star"]*100:+.2f}%  '
          f'Gap={m_ref["IS_OOS_gap"]*100:+.2f}pp  '
          f'Tr/yr={trades_yr_ref:>5.1f}')

    # ---------------------------------------------------------------------------
    # C2 Adaptive Deadband スイープ
    # ---------------------------------------------------------------------------
    for idx, eps_0 in enumerate(EPS_BASE_GRID, 1):
        # 時変ε系列
        eps_t = (eps_0 * vol_ratio_vals)   # ndarray

        tilt_confirmed, n_updates = compute_tilt_with_adaptive_deadband(
            raw_a2_vals, vz_arr, bull_mask, eps_t
        )

        wn_tilted = wn_A + tilt_confirmed
        wb_tilted = np.clip(wb_A - tilt_confirmed, 0.0, wb_A)

        n_tr_total = count_trades_tilted(wn_tilted, wb_tilted, lev_raw_arr)
        trades_yr  = n_tr_total / n_years

        nav = build_nav_strategy(
            close, lev_mod, wn_tilted, wg_A, wb_tilted, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, trades_yr)

        # ε_t 統計
        eps_t_mean = float(eps_t.mean())
        eps_t_std  = float(eps_t.std())
        eps_t_min  = float(eps_t.min())
        eps_t_max  = float(eps_t.max())

        m.update({
            'label':        f'ε₀={eps_0:.3f}/adaptive',
            'eps_0':        eps_0,
            'mode':         'adaptive',
            'Trades_yr':    trades_yr,
            'tilt_updates': n_updates,
            'eps_t_mean':   eps_t_mean,
            'eps_t_std':    eps_t_std,
            'eps_t_min':    eps_t_min,
            'eps_t_max':    eps_t_max,
            'WFA_CI95_lo':  float('nan'),
            'WFA_WFE':      float('nan'),
        })
        results.append(m)

        print(f'  [{idx}/{len(EPS_BASE_GRID)}] ε₀={eps_0:.3f}/adaptive  '
              f'ε_t∈[{eps_t_min:.4f},{eps_t_max:.4f}] mean={eps_t_mean:.4f}  '
              f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
              f'Sharpe={m["Sharpe_OOS"]:+.3f}  '
              f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
              f'W10Y={m["Worst10Y_star"]*100:+.2f}%  '
              f'Gap={m["IS_OOS_gap"]*100:+.2f}pp  '
              f'Tr/yr={trades_yr:>5.1f}  '
              f'updates={n_updates:>4d}')

    # ---------------------------------------------------------------------------
    # REFとの差分サマリ
    # ---------------------------------------------------------------------------
    ref_m = results[0]
    print(f'\n[REFとの比較 (F10 ε=0.015/fixed)]')
    print(f'  REF: Sharpe={ref_m["Sharpe_OOS"]:+.3f}  CAGR_OOS={ref_m["CAGR_OOS"]*100:+.2f}%  '
          f'Tr/yr={ref_m["Trades_yr"]:.1f}')
    for r in results[1:]:
        dS = r['Sharpe_OOS'] - ref_m['Sharpe_OOS']
        dC = (r['CAGR_OOS'] - ref_m['CAGR_OOS']) * 100
        dT = r['Trades_yr'] - ref_m['Trades_yr']
        print(f'  {r["label"]}: ΔSharpe={dS:+.4f}  ΔCAGR={dC:+.2f}pp  ΔTr/yr={dT:+.1f}')

    # ---------------------------------------------------------------------------
    # 採用判定
    # ---------------------------------------------------------------------------
    def adopt_status(r):
        cond_tr  = r['Trades_yr'] <= ADOPT_TRADES_YR
        cond_sh  = r['Sharpe_OOS'] >= ADOPT_SHARPE
        cond_gap = r['IS_OOS_gap'] <= ADOPT_GAP
        cond_dd  = r['MaxDD_FULL'] > ADOPT_MAXDD
        if cond_tr and cond_sh and cond_gap and cond_dd:
            return 'PASS'
        if cond_sh and cond_gap and r['MaxDD_FULL'] > -0.80:
            return 'WARN'
        return 'FAIL'

    for r in results:
        r['verdict'] = adopt_status(r)

    adaptive_results = results[1:]  # REF除く
    pass_list = [r for r in adaptive_results if r['verdict'] == 'PASS']
    warn_list = [r for r in adaptive_results if r['verdict'] == 'WARN']

    print(f'\nPASS ε₀: {[r["label"] for r in pass_list]}')
    print(f'WARN ε₀: {[r["label"] for r in warn_list]}')

    if pass_list:
        best = max(pass_list, key=lambda r: r['Sharpe_OOS'])
        verdict_overall = 'PASS'
        print(f'採用候補: {best["label"]} → Sharpe={best["Sharpe_OOS"]:+.3f}, '
              f'CAGR_OOS={best["CAGR_OOS"]*100:+.2f}%, Tr/yr={best["Trades_yr"]:.1f}')
    elif warn_list:
        best = max(warn_list, key=lambda r: r['Sharpe_OOS'])
        verdict_overall = 'WARN'
    else:
        best = max(adaptive_results, key=lambda r: r['Sharpe_OOS'])
        verdict_overall = 'FAIL'

    print(f'\n総合判定: {verdict_overall}')

    # ---------------------------------------------------------------------------
    # CSV
    # ---------------------------------------------------------------------------
    csv_path = os.path.join(BASE, 'c2_adaptive_deadband_results.csv')
    pd.DataFrame([{
        'label':         r.get('label', ''),
        'eps_0':         r['eps_0'],
        'mode':          r.get('mode', ''),
        'CAGR_IS':       r['CAGR_IS'],
        'CAGR_OOS':      r['CAGR_OOS'],
        'Sharpe_OOS':    r['Sharpe_OOS'],
        'MaxDD_FULL':    r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'],
        'P10_5Y':        r['P10_5Y'],
        'IS_OOS_gap':    r['IS_OOS_gap'],
        'Trades_yr':     r['Trades_yr'],
        'tilt_updates':  r['tilt_updates'],
        'eps_t_mean':    r.get('eps_t_mean', r['eps_0']),
        'eps_t_std':     r.get('eps_t_std', 0.0),
        'verdict':       r['verdict'],
    } for r in results]).to_csv(csv_path, index=False, float_format='%.6f')

    # ---------------------------------------------------------------------------
    # MD
    # ---------------------------------------------------------------------------
    hdr1, hdr2 = MD_HEADER_1P
    rows = [fmt_row_1p(r['label'], r) for r in results]
    table_body = '\n'.join(rows)

    # 採用候補 block
    adaptive_candidates = [r for r in adaptive_results
                           if r['Trades_yr'] <= ADOPT_TRADES_YR
                           and r['Sharpe_OOS'] >= ADOPT_SHARPE]
    if adaptive_candidates:
        best_cand = max(adaptive_candidates, key=lambda r: r['Sharpe_OOS'])
        adopt_block = (
            f'**採用候補: {best_cand["label"]}** → '
            f'Sharpe={best_cand["Sharpe_OOS"]:+.3f}, '
            f'CAGR_OOS={best_cand["CAGR_OOS"]*100:+.2f}%, '
            f'MaxDD={best_cand["MaxDD_FULL"]*100:+.2f}%, '
            f'Trades/yr={best_cand["Trades_yr"]:.1f}, '
            f'IS-OOS gap={best_cand["IS_OOS_gap"]*100:+.2f}pp'
        )
    else:
        best_cand = best
        adopt_block = (
            '**採用候補なし**: Trades/yr ≤ 70 かつ Sharpe ≥ +0.8915 を同時に満たす adaptive ε₀ が存在しない。'
        )

    # REFとの比較表
    r_ref = results[0]
    comparison_rows = []
    for r in results:
        dS = r['Sharpe_OOS'] - r_ref['Sharpe_OOS']
        dC = (r['CAGR_OOS']  - r_ref['CAGR_OOS']) * 100
        dT =  r['Trades_yr'] - r_ref['Trades_yr']
        comparison_rows.append(
            f'| {r["label"]} '
            f'| {r["Trades_yr"]:>5.1f} '
            f'| {r["Sharpe_OOS"]:+.3f} '
            f'| {r["CAGR_OOS"]*100:+.2f}% '
            f'| {r["MaxDD_FULL"]*100:+.2f}% '
            f'| {r["IS_OOS_gap"]*100:+.2f}pp '
            f'| {dS:+.4f} '
            f'| {dC:+.2f}pp '
            f'| {dT:+.1f} '
            f'| {r["verdict"]} |'
        )
    comparison_body = '\n'.join(comparison_rows)

    report = f"""\
# C2: Adaptive Deadband（ε_t=σ連動）

作成日: 2026-05-27
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 目的

C2: Adaptive Deadband（ε_t=σ連動）- CFAR原理でノイズ耐性向上

F10系のε=0.015固定deadbandを、市場ボラティリティに連動して動的変更。

```
ε_t = ε_0 × (σ_t / σ̄)
  σ_t  = 20日実現ボラ（年率）
  σ̄   = 250日平均ボラ
  clip: vol_ratio ∈ [0.3, 3.0]
```

- **高ボラ時**: εが大きくなり不要な取引を抑制
- **低ボラ時**: εが小さくなりシグナルに素早く反応
- **CFAR（Constant False Alarm Rate）原理**: ボラ環境に関わらず「誤検出率」を一定に保つ

| 項目 | 定義 |
|------|------|
| **Base config** | E4 採用: k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, k_mid={K_MID}, LT2-N750, mode B |
| **Tilt 定式** | step-function: clip(tilt × (raw_a2-0.15) × (1-raw_a2), 0, cap_eff), tilt={TILT} |
| **cap_eff** | calm: {CAP_CALM}, bull-VZ: {CAP_BULL}, bear-VZ: {CAP_BEAR} |
| **ε₀ グリッド** | {EPS_BASE_GRID} |
| **vol窓** | σ_{VOL_WINDOW}d (実現ボラ) / σ̄_{SIGMA_LOOKBACK}d (平均) |
| **vol_ratio clip** | [0.3, 3.0] |
| **REF** | F10 ε=0.015固定（固定deadband比較） |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

---

## §2 9指標テーブル

{hdr1}
{hdr2}
{table_body}

{MD_WFA_NOTE}

---

## §3 REF（F10 ε=0.015固定）との比較

| config | Tr/yr | Sharpe<br>_OOS | CAGR<br>_OOS | MaxDD | IS-OOS<br>gap | ΔSharpe<br>vs REF | ΔCAGR<br>vs REF | ΔTr/yr<br>vs REF | 判定 |
|:-------|------:|---------------:|-------------:|------:|--------------:|------------------:|----------------:|-----------------:|:-----|
{comparison_body}

---

## §4 採用判断

### 採用条件 (優先順)
| 順位 | 条件 |
|:----:|:-----|
| (1) | Trades/yr ≤ {ADOPT_TRADES_YR:.0f}（ユーザー許容） |
| (2) | Sharpe_OOS ≥ +{ADOPT_SHARPE:.4f}（現行 E4 以上） |
| (3) | IS-OOS gap ≤ +{ADOPT_GAP*100:.1f}pp |
| (4) | MaxDD > {ADOPT_MAXDD*100:+.2f}% (望ましい), > -80% (絶対) |

### 判定: **{verdict_overall}**

{adopt_block}

### E4 (現行ベスト) との比較
| 指標 | E4 (現行ベスト) | F10 REF (ε=0.015固定) | C2 最良 ({best_cand["label"]}) |
|:-----|----------------:|----------------------:|-------------------------------:|
| CAGR_OOS | +{REF_CAGR_OOS*100:.2f}% | {r_ref["CAGR_OOS"]*100:+.2f}% | {best_cand["CAGR_OOS"]*100:+.2f}% |
| Sharpe_OOS | +{REF_SHARPE_OOS:.3f} | {r_ref["Sharpe_OOS"]:+.3f} | {best_cand["Sharpe_OOS"]:+.3f} |
| MaxDD | {REF_MAXDD*100:+.2f}% | {r_ref["MaxDD_FULL"]*100:+.2f}% | {best_cand["MaxDD_FULL"]*100:+.2f}% |
| Worst10Y★ | +{REF_WORST10Y*100:.2f}% | {r_ref["Worst10Y_star"]*100:+.2f}% | {best_cand["Worst10Y_star"]*100:+.2f}% |
| IS-OOS gap | {REF_GAP*100:+.2f}pp | {r_ref["IS_OOS_gap"]*100:+.2f}pp | {best_cand["IS_OOS_gap"]*100:+.2f}pp |
| Trades/yr | {REF_TRADES_YR:.1f} | {r_ref["Trades_yr"]:.1f} | {best_cand["Trades_yr"]:.1f} |
| WFA CI95_lo | +0.265‡ | — | (実施推奨) |
| WFA WFE | +1.131‡ | — | (実施推奨) |

‡ E4 の G3 WFA は CI95_lo=+26.51%, WFE=+1.131（CURRENT_BEST_STRATEGY.md より）。

---

## §5 再現コマンド

```bash
cd "C:\\Users\\user\\Desktop\\投資・不動産\\nasdaq_backtest"
python -X utf8 src/c2_adaptive_deadband.py
```

出力:
- `c2_adaptive_deadband_results.csv` — 9指標 + verdict
- `C2_ADAPTIVE_DEADBAND_2026-05-27.md` — 本レポート

参照:
- `src/f10_epsilon_deadband.py` — F10 固定ε実装（ベース）
- `src/e4_regime_klt.py` — E4 base 実装
- `CURRENT_BEST_STRATEGY.md` — E4 現行ベスト
- `EVALUATION_STANDARD.md` §3.12 — 9指標標準

---

*生成スクリプト: `src/c2_adaptive_deadband.py`*
"""

    md_path = os.path.join(BASE, 'C2_ADAPTIVE_DEADBAND_2026-05-27.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'\nSaved: {csv_path}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
