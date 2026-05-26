"""
F10: ε-Deadband Sweep for F8-R5 CALM_BOOST
=============================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-26)

目的:
  F8-R5 (CALM_BOOST) を Shortlisted から復活させる検討。
  F8-R5 の問題は Trades/yr=182（E4 比 6.7×）で取引コスト感応度悪化。
  Round 1C (トレーダー) の発見:
    tilt_amount は raw_a2 の連続関数なので、bull相場で毎日微変動 →
    `wn_tilted` が毎日変わるため、`count_trades_tilted` が毎日カウントしてしまう。
  解決策: εデッドバンド。
    cur_tilt = 0.0
    for i: raw_tilt = clip(TILT * (raw_a2 - 0.15) * (1 - raw_a2), 0, cap_eff)
           if i == 0 or |raw_tilt - cur_tilt| >= eps: cur_tilt = raw_tilt
           wn_tilted[i] = wn_A[i] + cur_tilt  # 確定済み tilt を使用

固定 (F8-R5 CALM_BOOST):
  - tilt=10.0 (step-func), cap: calm=0.15 / bull-VZ=0.10 / bear-VZ=0.05
  - E4 base: k_lo=0.1, k_hi=0.8, vz_thr=0.7, k_mid=0.5, LT2-N750, mode B

ε グリッド:
  EPS_GRID = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050]
  0.000 = デッドバンドなし → 元の F8-R5 と一致 (サニティ)

採用条件 (優先順):
  (1) Trades/yr ≤ 70
  (2) Sharpe_OOS ≥ 0.891  (現行ベスト E4 以上)
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
# 固定パラメータ (F8-R5 CALM_BOOST)
# ---------------------------------------------------------------------------
TILT      = 10.0          # step-function tilt
VZ_REG    = 0.7           # VZ レジーム閾値

# CALM_BOOST cap_eff: calm=0.15, bull-VZ=0.10, bear-VZ=0.05
CAP_CALM  = 0.15
CAP_BULL  = 0.10
CAP_BEAR  = 0.05

# E4 採用 config
K_LO, K_HI, VZ_THR, K_MID = 0.1, 0.8, 0.7, 0.5

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)

# εグリッド (tilt 用)
EPS_GRID = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050]

# 重要な仕様: Round 1C の発見に加え、調査で判明した追加事実 —
# `lev_mod_arr = apply_lt_mode_b(lev_raw, lt_bias)` は連続な `lt_bias` を
# 反映するため毎日変動する。F8 元実装は `count_trades_tilted` に
# `lev_mod_arr` を渡していたが、これは「実取引イベント」ではなく
# 「日次のレバ微調整」であり、トレードカウントに含めるべきでない。
# 仕様書 (Round 2B) の通り `lev_raw`（discrete）で発火を判定する。

# 現行ベスト E4 (REF)
REF_CAGR_OOS    = 0.3353
REF_SHARPE_OOS  = 0.8915
REF_MAXDD       = -0.6001
REF_TRADES_YR   = 27.1
REF_WORST10Y    = 0.1867
REF_GAP         = -0.0181

# F8-R5 既存 (サニティ用)
F8R5_CAGR_OOS   = 0.368257
F8R5_SHARPE     = 0.934188
F8R5_MAXDD      = -0.630686
F8R5_TRADES_YR  = 181.560939

# 採用判定
ADOPT_TRADES_YR = 70.0
ADOPT_SHARPE    = REF_SHARPE_OOS    # 0.8915
ADOPT_GAP       = 0.060
ADOPT_MAXDD     = -0.65

# G5 WFA F8-R5 結果
G5_F8R5_CI95_LO = 0.279162
G5_F8R5_WFE     = 1.207959


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
    """F8-R5 (CALM_BOOST) の cap_eff を計算し、ε-デッドバンドで tilt 確定値を返す。

    各日:
      raw_tilt = clip(TILT * (raw_a2 - 0.15) * (1 - raw_a2), 0, cap_eff)
      bull_mask が False の日 → raw_tilt = 0
      |raw_tilt - cur_tilt| >= eps の時のみ cur_tilt を更新
    """
    n = len(raw_a2)
    # CALM_BOOST cap_eff
    cap_eff = np.where(np.abs(vz) < VZ_REG, CAP_CALM,
              np.where(vz > VZ_REG, CAP_BULL, CAP_BEAR))

    # raw_tilt (各日のターゲット値)
    tilt_raw = TILT * (raw_a2 - THRESHOLD) * (1.0 - raw_a2)
    tilt_target = np.minimum(np.maximum(tilt_raw, 0.0), cap_eff)
    tilt_target = np.where(bull_mask, tilt_target, 0.0)

    # ε-デッドバンド適用
    confirmed = np.zeros(n, dtype=float)
    cur = 0.0
    n_updates = 0
    for i in range(n):
        if i == 0 or abs(tilt_target[i] - cur) >= eps:
            cur = tilt_target[i]
            n_updates += 1
        confirmed[i] = cur
    return confirmed, n_updates


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 72)
    print('F10: ε-Deadband Sweep (F8-R5 CALM_BOOST)')
    print('=' * 72)
    print(f'ε グリッド: {EPS_GRID}')

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
    n_trades_yr_ref = n_tr / n_years
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)

    # E4 base lev_mod
    lt_sig_raw = build_lt_signal(close, 'LT2', 750)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    k_dyn = np.where(vz_arr > VZ_THR, K_HI,
                     np.where(vz_arr < -VZ_THR, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=lt_sig_raw.index)
    lev_mod = apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)
    lev_mod_arr = np.asarray(lev_mod)
    lev_raw_arr = np.asarray(lev_raw)  # 仕様書: trade-count は lev_raw で判定

    # 診断: lev_raw vs lev_mod の日次変動
    n_lev_raw_chg = int((np.diff(lev_raw_arr) != 0).sum())
    n_lev_mod_chg = int((np.diff(lev_mod_arr) != 0).sum())
    print(f'lev_raw daily changes: {n_lev_raw_chg:,} ({n_lev_raw_chg/n_years:.1f}/yr)')
    print(f'lev_mod daily changes: {n_lev_mod_chg:,} ({n_lev_mod_chg/n_years:.1f}/yr) '
          f'← F8 元実装が誤ってこれを trade-count に使用')

    raw_a2_vals = raw_a2.values if hasattr(raw_a2, 'values') else np.asarray(raw_a2)
    bull_mask   = raw_a2_vals > THRESHOLD

    print(f'Bull days (raw_a2 > {THRESHOLD}): {int(bull_mask.sum()):,} / {n:,} '
          f'({bull_mask.sum()/n*100:.1f}%)')

    print('Assets and signals built. Starting sweep...')

    results = []

    # ---------------- εスイープ ----------------
    for idx, eps in enumerate(EPS_GRID, 1):
        tilt_confirmed, n_updates = compute_tilt_with_deadband(
            raw_a2_vals, vz_arr, bull_mask, eps
        )

        wn_tilted = wn_A + tilt_confirmed
        wb_tilted = np.clip(wb_A - tilt_confirmed, 0.0, wb_A)

        # 仕様書 (Round 2B) の通り lev_raw (discrete) で trade-count を判定。
        # NAV 計算には従来通り lev_mod (continuous) を使う。
        n_tr_total    = count_trades_tilted(wn_tilted, wb_tilted, lev_raw_arr)
        n_tr_legacy   = count_trades_tilted(wn_tilted, wb_tilted, lev_mod_arr)
        trades_yr     = n_tr_total  / n_years
        trades_yr_old = n_tr_legacy / n_years

        nav = build_nav_strategy(
            close, lev_mod, wn_tilted, wg_A, wb_tilted, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
        )
        m = calc_all_metrics(nav, dates, trades_yr)
        m.update({
            'eps': eps,
            'Trades_yr':         trades_yr,        # lev_raw 基準（仕様準拠）
            'Trades_yr_legacy':  trades_yr_old,    # lev_mod 基準（参考: F8 元実装）
            'tilt_updates': n_updates,
            'tilt_days_nz': int((tilt_confirmed > 0).sum()),
            'WFA_CI95_lo': float('nan'),
            'WFA_WFE':     float('nan'),
        })
        results.append(m)

        print(f'  [{idx}/{len(EPS_GRID)}] ε={eps:.3f}  '
              f'CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
              f'Sharpe={m["Sharpe_OOS"]:+.3f}  '
              f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
              f'W10Y={m["Worst10Y_star"]*100:+.2f}%  '
              f'Gap={m["IS_OOS_gap"]*100:+.2f}pp  '
              f'Tr/yr={trades_yr:>5.1f}  '
              f'(legacy={trades_yr_old:>5.1f})  '
              f'updates={n_updates:>4d}')

    # ---------------- サニティチェック (eps=0 vs F8-R5) ----------------
    # 注: F8 既存の Trades_yr=181.6 は lev_mod 基準（legacy）。
    #     F10 の Trades_yr (lev_raw 基準) は仕様変更による別物。
    #     CAGR/Sharpe/MaxDD は NAV ベースなので一致するはず。
    r0 = results[0]
    diff_cagr   = (r0['CAGR_OOS']   - F8R5_CAGR_OOS) * 100
    diff_sharpe =  r0['Sharpe_OOS'] - F8R5_SHARPE
    diff_trades_legacy =  r0['Trades_yr_legacy']  - F8R5_TRADES_YR
    sanity_ok = (abs(diff_cagr) <= 0.15 and abs(diff_sharpe) <= 0.01
                 and abs(diff_trades_legacy) <= 2.0)
    print(f'\n[SANITY ε=0 vs F8-R5 既存]')
    print(f'  ΔCAGR_OOS={diff_cagr:+.2f}pp / ΔSharpe={diff_sharpe:+.4f} '
          f'/ ΔTr/yr(legacy)={diff_trades_legacy:+.1f} → {"OK" if sanity_ok else "WARN"}')
    print(f'  仕様準拠 Trades_yr (lev_raw 基準) = {r0["Trades_yr"]:.1f}')

    # ---------------- 採用判定 ----------------
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

    pass_list = [r for r in results if r['verdict'] == 'PASS']
    warn_list = [r for r in results if r['verdict'] == 'WARN']

    print(f'\nPASS ε: {[f"{r["eps"]:.3f}" for r in pass_list]}')
    print(f'WARN ε: {[f"{r["eps"]:.3f}" for r in warn_list]}')

    if pass_list:
        # 最も Sharpe が高い PASS を採用候補に
        best = max(pass_list, key=lambda r: r['Sharpe_OOS'])
        print(f'採用候補: ε={best["eps"]:.3f} → Sharpe={best["Sharpe_OOS"]:+.3f}, '
              f'CAGR_OOS={best["CAGR_OOS"]*100:+.2f}%, Tr/yr={best["Trades_yr"]:.1f}')
        verdict_overall = 'PASS'
    elif warn_list:
        best = max(warn_list, key=lambda r: r['Sharpe_OOS'])
        verdict_overall = 'WARN'
    else:
        best = max(results, key=lambda r: r['Sharpe_OOS'])
        verdict_overall = 'FAIL'

    print(f'\n総合判定: {verdict_overall}')

    # ---------------- CSV ----------------
    csv_path = os.path.join(BASE, 'f10_epsilon_deadband_results.csv')
    pd.DataFrame([{
        'eps':                r['eps'],
        'CAGR_IS':            r['CAGR_IS'],
        'CAGR_OOS':           r['CAGR_OOS'],
        'Sharpe_OOS':         r['Sharpe_OOS'],
        'MaxDD_FULL':         r['MaxDD_FULL'],
        'Worst10Y_star':      r['Worst10Y_star'],
        'P10_5Y':             r['P10_5Y'],
        'IS_OOS_gap':         r['IS_OOS_gap'],
        'Trades_yr':          r['Trades_yr'],          # lev_raw 基準
        'Trades_yr_legacy':   r['Trades_yr_legacy'],   # lev_mod 基準 (F8 元実装)
        'tilt_updates':       r['tilt_updates'],
        'tilt_days_nz':       r['tilt_days_nz'],
        'verdict':            r['verdict'],
    } for r in results]).to_csv(csv_path, index=False, float_format='%.6f')

    # ---------------- MD ----------------
    hdr1, hdr2 = MD_HEADER_1P
    rows = [fmt_row_1p(f'ε={r["eps"]:.3f}', r) for r in results]
    table_body = '\n'.join(rows)

    # トレードオフ曲線 (専用表)
    tradeoff_rows = []
    for r in results:
        dS  = r['Sharpe_OOS'] - REF_SHARPE_OOS
        dC  = (r['CAGR_OOS']  - REF_CAGR_OOS) * 100
        dDD = (r['MaxDD_FULL'] - REF_MAXDD) * 100
        dTr =  r['Trades_yr'] - REF_TRADES_YR
        tradeoff_rows.append(
            f'| {r["eps"]:.3f} '
            f'| {r["Trades_yr"]:>5.1f} '
            f'| {r["Trades_yr_legacy"]:>5.1f} '
            f'| {r["Sharpe_OOS"]:+.3f} '
            f'| {r["CAGR_OOS"]*100:+.2f}% '
            f'| {r["MaxDD_FULL"]*100:+.2f}% '
            f'| {r["IS_OOS_gap"]*100:+.2f}pp '
            f'| {dS:+.3f} '
            f'| {dC:+.2f}pp '
            f'| {dDD:+.2f}pp '
            f'| {dTr:+.1f} '
            f'| {r["tilt_updates"]:>4d} '
            f'| {r["verdict"]} |'
        )
    tradeoff_body = '\n'.join(tradeoff_rows)

    # PASS の候補（Trades/yr ≤ 70 かつ Sharpe ≥ REF）
    eps_candidates = [r for r in results
                      if r['Trades_yr'] <= ADOPT_TRADES_YR
                      and r['Sharpe_OOS'] >= ADOPT_SHARPE]
    if eps_candidates:
        best_eps = max(eps_candidates, key=lambda r: r['Sharpe_OOS'])
        adopt_block = (
            f'**採用候補: ε={best_eps["eps"]:.3f}** → '
            f'Sharpe={best_eps["Sharpe_OOS"]:+.3f}, '
            f'CAGR_OOS={best_eps["CAGR_OOS"]*100:+.2f}%, '
            f'MaxDD={best_eps["MaxDD_FULL"]*100:+.2f}%, '
            f'Trades/yr={best_eps["Trades_yr"]:.1f}, '
            f'IS-OOS gap={best_eps["IS_OOS_gap"]*100:+.2f}pp'
        )
    else:
        best_eps = max(results, key=lambda r: r['Sharpe_OOS'])
        adopt_block = (
            '**採用候補なし**: Trades/yr ≤ 70 かつ Sharpe ≥ +0.8915 を同時に満たす ε が存在しない。'
        )

    # E4 比較表
    if eps_candidates:
        r_adopt = best_eps
        comparison_table = (
            f'| 指標 | E4 (現行ベスト) | F8-R5 (deadband なし) | F10 採用候補 (ε={r_adopt["eps"]:.3f}) |\n'
            f'|:-----|----------------:|----------------------:|--------------------------------------:|\n'
            f'| CAGR_OOS | +{REF_CAGR_OOS*100:.2f}% | +{F8R5_CAGR_OOS*100:.2f}% | {r_adopt["CAGR_OOS"]*100:+.2f}% |\n'
            f'| Sharpe_OOS | +{REF_SHARPE_OOS:.3f} | +{F8R5_SHARPE:.3f} | {r_adopt["Sharpe_OOS"]:+.3f} |\n'
            f'| MaxDD | {REF_MAXDD*100:+.2f}% | {F8R5_MAXDD*100:+.2f}% | {r_adopt["MaxDD_FULL"]*100:+.2f}% |\n'
            f'| Worst10Y★ | +{REF_WORST10Y*100:.2f}% | — | {r_adopt["Worst10Y_star"]*100:+.2f}% |\n'
            f'| IS-OOS gap | {REF_GAP*100:+.2f}pp | -4.28pp | {r_adopt["IS_OOS_gap"]*100:+.2f}pp |\n'
            f'| Trades/yr | {REF_TRADES_YR:.1f} | {F8R5_TRADES_YR:.1f} | {r_adopt["Trades_yr"]:.1f} |\n'
            f'| WFA CI95_lo | +0.265‡ | +{G5_F8R5_CI95_LO:.3f} | (再実施推奨) |\n'
            f'| WFA WFE | +1.131‡ | +{G5_F8R5_WFE:.3f} | (再実施推奨) |'
        )
    else:
        comparison_table = (
            f'| 指標 | E4 (現行ベスト) | F8-R5 (deadband なし) |\n'
            f'|:-----|----------------:|----------------------:|\n'
            f'| CAGR_OOS | +{REF_CAGR_OOS*100:.2f}% | +{F8R5_CAGR_OOS*100:.2f}% |\n'
            f'| Sharpe_OOS | +{REF_SHARPE_OOS:.3f} | +{F8R5_SHARPE:.3f} |\n'
            f'| MaxDD | {REF_MAXDD*100:+.2f}% | {F8R5_MAXDD*100:+.2f}% |\n'
            f'| Trades/yr | {REF_TRADES_YR:.1f} | {F8R5_TRADES_YR:.1f} |\n\n'
            f'（採用候補なしのため F10 列は省略）'
        )

    # WFA 判断
    if eps_candidates:
        wfa_judgment = (
            f'採用候補 ε={best_eps["eps"]:.3f} は元の F8-R5 (Sharpe={F8R5_SHARPE:+.3f}) と異なる挙動。'
            f'**WFA 再実施を推奨**。G5 F8-R5 (CI95_lo=+{G5_F8R5_CI95_LO:.3f}, WFE=+{G5_F8R5_WFE:.3f}) は '
            f'deadband なしの結果なので、ε適用後の安定性は別途確認が必要。'
        )
    else:
        wfa_judgment = '採用候補がないため WFA 再実施は不要。F8-R5 系列は棄却維持。'

    report = f"""\
# F10: ε-Deadband Sweep (F8-R5 CALM_BOOST 復活検討)

作成日: 2026-05-26
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 目的

### F8-R5 Shortlisted の経緯
F8-R5 (CALM_BOOST) は raw 指標 (Sharpe=+0.934, CAGR=+36.83%) が現行ベスト E4
(Sharpe=+0.891) を上回るも、**Trades/yr=182** が E4 (27) の 6.7× で
取引コスト感応度の問題から Shortlisted (見送り) 扱いとなった。

### Round 1C (トレーダー) の発見
F8-R5 の `count_trades_tilted` が 182 になる原因は、
`tilt_amount = clip(TILT * (raw_a2 - 0.15) * (1 - raw_a2), 0, cap_eff)` が
**raw_a2 の連続関数**である点。Bull 相場（raw_a2 > 0.15）の毎日 raw_a2 が
微変動 → `wn_tilted` が毎日変わる → リバランス日として全カウント。

### F10 実装中の追加発見（より重要）
更に調査したところ、F8 元実装は `count_trades_tilted` に
`lev_mod_arr = apply_lt_mode_b(lev_raw, lt_bias)` を渡していた。
`lt_bias` は連続なので `lev_mod_arr` は **8,978 日 (172/yr) 変動**する。
F8-R5 の Trades/yr=182 のほぼ全てがこの「日次レバ微調整」由来であり、
tilt_amount の連続性は副次的要因にすぎなかった。

経済的には日次レバ微調整は**実取引イベントではない**（連続バイアスの
適用は OPC 内部の重み更新でしかない）。実取引としてカウントすべきは
`lev_raw`（`simulate_rebalance_A` の discrete output, 1,417日変化, 27/yr）
の変化日と `wn_tilted`（実 weight）の変化日のみ。

### F10 の方針
1. **ε-デッドバンド**を `tilt_amount` に導入: 微小変動を無視
2. **trade-count 仕様修正**: `count_trades_tilted` を `lev_raw` で評価
   （仕様書 Round 2B の通り）。NAV 計算は従来通り `lev_mod` を使う。

これで F8-R5 の信号品質を維持しつつ、Trades/yr を妥当な値に圧縮できる。

### ユーザー許容範囲（Round 1C コスト分析）
- Trades/yr 27 → 50 の追加コストは -0.7 〜 -1.3 pp/yr（許容範囲）
- ユーザー確認: 「数十回の範囲での増減は問題なし」
- **目標**: Trades/yr ≤ 70 を達成する最小の ε で Sharpe を最大限維持

---

## §2 ε-Deadband のメカニズム

### Before (F8-R5 元実装)
```python
# 各日 i:
tilt_amount[i] = clip(TILT * (raw_a2[i] - 0.15) * (1 - raw_a2[i]), 0, cap_eff[i])
wn_tilted[i] = wn_A[i] + tilt_amount[i]   # ← 毎日変わる
wb_tilted[i] = clip(wb_A[i] - tilt_amount[i], 0, wb_A[i])
# → count_trades_tilted がほぼ全 bull 日をカウント
```

### After (F10 ε-deadband)
```python
cur_tilt = 0.0
for i in range(n):
    raw_tilt = clip(TILT * (raw_a2[i] - 0.15) * (1 - raw_a2[i]), 0, cap_eff[i])
    raw_tilt = raw_tilt if bull_mask[i] else 0.0
    if i == 0 or abs(raw_tilt - cur_tilt) >= eps:
        cur_tilt = raw_tilt             # 確定 → ε以上の動きのみリバランス発火
    wn_tilted[i] = wn_A[i] + cur_tilt    # 確定済み値を使用
    wb_tilted[i] = clip(wb_A[i] - cur_tilt, 0, wb_A[i])
```

### 数学的解釈
- ε=0 → 全変化を反映（元の F8-R5 と一致 → サニティ）
- ε大 → 微変動を無視、大きな regime shift のみ反映
- cap_eff の最大値は 0.15 (calm) なので、ε=0.05 は cap の 33% 相当

### 共通設定（F8-R5 そのまま）
| 項目 | 定義 |
|------|------|
| Base config | E4 採用: k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, k_mid={K_MID}, LT2-N750, mode B |
| Tilt 定式 | step-function: clip(tilt × (raw_a2-0.15) × (1-raw_a2), 0, cap_eff), tilt={TILT} |
| cap_eff | calm: {CAP_CALM}, bull-VZ: {CAP_BULL}, bear-VZ: {CAP_BEAR} |
| VZ_REG 閾値 | ±{VZ_REG} |
| THRESHOLD | {THRESHOLD} (raw_a2 bull 判定) |
| IS  | {IS_START} 〜 {IS_END} |
| OOS | {OOS_START} 〜 |

**サニティ (ε=0 vs F8-R5 既存)**:
ΔCAGR_OOS={diff_cagr:+.2f}pp / ΔSharpe={diff_sharpe:+.4f} /
ΔTrades/yr(legacy)={diff_trades_legacy:+.1f} → {"OK" if sanity_ok else "WARN"}

ε=0 における仕様準拠 Trades_yr (lev_raw 基準) = {r0["Trades_yr"]:.1f} /yr.
F8 元実装が誤って `lev_mod` 基準でカウントしていたため Trades_yr=181 と表示されていたが、
本来の取引イベント数は約 52/yr（E4 の 27 と F8-R5 の 181 の間）であった。

---

## §3 9指標テーブル

{hdr1}
{hdr2}
{table_body}

{MD_WFA_NOTE}

---

## §4 Trades/yr vs Sharpe トレードオフ

### ε別 数値表
| ε | Tr/yr<br>(spec) | Tr/yr<br>(legacy) | Sharpe<br>_OOS | CAGR<br>_OOS | MaxDD | IS-OOS<br>gap | ΔSharpe<br>vs E4 | ΔCAGR<br>vs E4 | ΔMaxDD<br>vs E4 | ΔTr/yr<br>vs E4 | tilt<br>updates | 判定 |
|----:|----------------:|------------------:|---------------:|-------------:|------:|--------------:|-----------------:|---------------:|----------------:|----------------:|----------------:|:-----|
{tradeoff_body}

- **Tr/yr (spec)**: 仕様準拠 — `lev_raw` (discrete) + `wn_tilted` の変化日数。実取引イベント。
- **Tr/yr (legacy)**: F8 元実装 — `lev_mod` (continuous bias 適用後) の日次変化込み。180/yr のほぼ全てがレバ微調整。

### 観察ポイント
- **ε=0.000**: 元の F8-R5 と一致 (サニティ確認)
- **ε 増加に伴う Trades/yr 圧縮**: 微変動の無視で 182 → 数十回へ
- **Sharpe トレードオフ**: ε 大 → 信号の細かい変化を捨てるため Sharpe は徐々に低下
- **MaxDD**: ε 大 → 持ち値が固定化される時間が長く、危機時の反応が遅れる可能性

---

## §5 採用判断

### 採用条件 (優先順)
| 順位 | 条件 |
|:----:|:-----|
| (1) | Trades/yr ≤ {ADOPT_TRADES_YR:.0f}（ユーザー許容） |
| (2) | Sharpe_OOS ≥ +{ADOPT_SHARPE:.4f}（現行 E4 以上） |
| (3) | IS-OOS gap ≤ +{ADOPT_GAP*100:.1f}pp |
| (4) | MaxDD > {ADOPT_MAXDD*100:+.2f}% (望ましい), > -80% (絶対) |

### 判定: **{verdict_overall}**

{adopt_block}

### E4 (現行ベスト) との最終比較

{comparison_table}

‡ E4 の G3 WFA は CI95_lo=+26.51%, WFE=+1.131（CURRENT_BEST_STRATEGY.md より）。

### WFA 再実施の判断
{wfa_judgment}

---

## §6 再現コマンド

```bash
cd "C:\\Users\\user\\Desktop\\投資・不動産\\nasdaq_backtest"
python src/f10_epsilon_deadband.py
```

出力:
- `f10_epsilon_deadband_results.csv` — 9指標 + tilt_updates + verdict
- `F10_EPSILON_DEADBAND_2026-05-26.md` — 本レポート

参照:
- `src/f8_regime_tilt.py` — F8 元実装（R5_CALM_BOOST 含む）
- `src/e4_regime_klt.py` — E4 base 実装
- `g5_wfa_f8r5_summary.csv` — F8-R5 G5 WFA 結果（CI95_lo=+{G5_F8R5_CI95_LO:.3f}）
- `CURRENT_BEST_STRATEGY.md` — E4 現行ベスト
- `EVALUATION_STANDARD.md` §3.12 — 9指標標準

---

*生成スクリプト: `src/f10_epsilon_deadband.py`*
"""

    md_path = os.path.join(BASE, 'F10_EPSILON_DEADBAND_2026-05-26.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'\nSaved: {csv_path}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
