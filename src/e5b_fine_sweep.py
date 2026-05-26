"""
E5b: vz_thr 細密スイープ + vz_thr × l_max 交差グリッド
========================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-26)

背景:
  E5 スイープで vz_thr=0.65 が突出した Sharpe=+0.945 を記録したが、
  隣接値（0.60, 0.625, 0.675）が未評価で「山の形」（プラトー or スパイク）が不明。
  また l_max=5.0 が MaxDD の強力なレバーとわかったため、
  vz_thr と l_max の最適組み合わせを探索する。

Part A (vz_thr 細密スイープ):
  VZT_FINE = [0.575, 0.600, 0.625, 0.650, 0.675, 0.700, 0.725]
  tv=0.80, l_max=7.0 固定 → 7 configs

Part B (vz_thr × l_max 交差グリッド):
  VZT_CROSS  = [0.60, 0.625, 0.650, 0.675, 0.70]
  LMAX_CROSS = [5.0, 6.0, 7.0]
  tv=0.80 固定 → 5 × 3 = 15 configs

固定:
  k_lo=0.1, k_hi=0.8, k_mid=0.5  (E4 現行採用値)

合計: 7 + 15 = 22 configs（一部 Part A と Part B で重複するが両方記録）

出力:
  - e5b_fine_sweep_results.csv (22行)
  - E5B_FINE_SWEEP_2026-05-26.md

WFA は本ラウンド省略（CI95_lo / WFE = N/A）。
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
from _sweep_format import MD_HEADER_1P, MD_HEADER_2P, fmt_row_1p, fmt_row_2p, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# グリッド
# ---------------------------------------------------------------------------

# Part A: vz_thr 細密スイープ（tv=0.8, l_max=7.0 固定）
VZT_FINE = [0.575, 0.600, 0.625, 0.650, 0.675, 0.700, 0.725]

# Part B: vz_thr × l_max 交差グリッド（tv=0.8 固定）
VZT_CROSS  = [0.60, 0.625, 0.650, 0.675, 0.70]
LMAX_CROSS = [5.0, 6.0, 7.0]

# E4 採用固定値
K_LO  = 0.1
K_HI  = 0.8
K_MID = 0.5

# ベースライン参考値（E4 現行ベスト, 2026-05-24 確定）
BASE_VZT, BASE_TV, BASE_LMAX = 0.70, 0.80, 7.0

# S2 の他パラメータ（E4 と同一）
S2_BASE = dict(k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)


def signal_to_bias_dynamic(lt_sig_arr: np.ndarray, k_arr: np.ndarray) -> np.ndarray:
    return np.clip(-k_arr * lt_sig_arr * 0.5, -0.5, 0.5)


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


def run_config(close, ret, dates, vz, lev_arr, lt_sig_arr, lt_sig_raw,
               wn_A, wg_A, wb_A, gold_2x, bond_3x, sofr, n_trades_yr,
               vz_thr, tv, l_max):
    """1 config 評価。E5 と同一ロジック。"""
    L_s2 = compute_L_s2_vz_gated(
        ret, vz,
        target_vol=tv, l_max=l_max,
        **S2_BASE,
    )

    regime_hi  = vz.values > +vz_thr
    regime_lo  = vz.values < -vz_thr
    k_dyn = np.where(regime_hi, K_HI, np.where(regime_lo, K_LO, K_MID))

    lt_bias_dyn = pd.Series(
        signal_to_bias_dynamic(lt_sig_arr, k_dyn),
        index=lt_sig_raw.index,
    )
    lev_mod = apply_lt_mode_b(lev_arr, lt_bias_dyn, l_min=0.0, l_max=1.0)
    nav = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m = calc_all_metrics(nav, dates, n_trades_yr)
    m.update({
        'vz_thr': vz_thr, 'tv': tv, 'l_max': l_max,
        'k_lo': K_LO, 'k_hi': K_HI, 'k_mid': K_MID,
        'Trades_yr': n_trades_yr,
        'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan'),
    })
    return m


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 72)
    print('E5b: vz_thr 細密スイープ + vz_thr × l_max 交差グリッド')
    print('=' * 72)
    print(f'Part A vz_thr fine:   {VZT_FINE}  (tv={BASE_TV}, l_max={BASE_LMAX} fixed)')
    print(f'Part B vz_thr×l_max:  vz_thr={VZT_CROSS} × l_max={LMAX_CROSS}  '
          f'(tv={BASE_TV} fixed)')
    print(f'固定: k_lo={K_LO}, k_hi={K_HI}, k_mid={K_MID}')

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

    lt_sig_raw = build_lt_signal(close, 'LT2', 750)
    lt_sig_arr = lt_sig_raw.values

    print('Assets / signals built. Starting sweeps...')

    common_kwargs = dict(
        close=close, ret=ret, dates=dates, vz=vz, lev_arr=lev_raw,
        lt_sig_arr=lt_sig_arr, lt_sig_raw=lt_sig_raw,
        wn_A=wn_A, wg_A=wg_A, wb_A=wb_A,
        gold_2x=gold_2x, bond_3x=bond_3x, sofr=sofr,
        n_trades_yr=n_trades_yr,
    )

    # -----------------------------------------------------------------------
    # Part A: vz_thr 細密スイープ
    # -----------------------------------------------------------------------
    print('\n--- Part A: vz_thr 細密スイープ (tv=0.80, l_max=7.0 固定) ---')
    part_a = []
    for i, vzt in enumerate(VZT_FINE, 1):
        m = run_config(vz_thr=vzt, tv=BASE_TV, l_max=BASE_LMAX, **common_kwargs)
        m['part'] = 'A'
        part_a.append(m)
        print(f'  [A{i}/{len(VZT_FINE)}] vz_thr={vzt:.3f}: '
              f'CAGR_OOS={m["CAGR_OOS"]*100:+6.2f}%  '
              f'Sharpe={m["Sharpe_OOS"]:+.3f}  '
              f'MaxDD={m["MaxDD_FULL"]*100:+6.2f}%  '
              f'W10Y★={m["Worst10Y_star"]*100:+6.2f}%')

    # -----------------------------------------------------------------------
    # Part B: vz_thr × l_max 交差グリッド
    # -----------------------------------------------------------------------
    print('\n--- Part B: vz_thr × l_max 交差グリッド (tv=0.80 固定) ---')
    part_b = []
    idx = 0
    total_b = len(VZT_CROSS) * len(LMAX_CROSS)
    for vzt in VZT_CROSS:
        for lm in LMAX_CROSS:
            idx += 1
            m = run_config(vz_thr=vzt, tv=BASE_TV, l_max=lm, **common_kwargs)
            m['part'] = 'B'
            part_b.append(m)
            print(f'  [B{idx:>2d}/{total_b}] vz_thr={vzt:.3f} l_max={lm:.1f}: '
                  f'CAGR_OOS={m["CAGR_OOS"]*100:+6.2f}%  '
                  f'Sharpe={m["Sharpe_OOS"]:+.3f}  '
                  f'MaxDD={m["MaxDD_FULL"]*100:+6.2f}%  '
                  f'W10Y★={m["Worst10Y_star"]*100:+6.2f}%')

    # -----------------------------------------------------------------------
    # CSV 保存
    # -----------------------------------------------------------------------
    all_results = part_a + part_b
    csv_rows = [{
        'part': r['part'],
        'vz_thr': r['vz_thr'], 'tv': r['tv'], 'l_max': r['l_max'],
        'k_lo': r['k_lo'], 'k_hi': r['k_hi'], 'k_mid': r['k_mid'],
        'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'], 'Trades_yr': r['Trades_yr'],
    } for r in all_results]
    csv_path = os.path.join(BASE, 'e5b_fine_sweep_results.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nCSV saved: {csv_path}')

    # -----------------------------------------------------------------------
    # 分析サマリ
    # -----------------------------------------------------------------------
    # Part A — vz_thr=0.65 周辺の山の形状
    a_by_vzt = {round(r['vz_thr'], 3): r for r in part_a}
    r_065 = a_by_vzt[0.650]
    r_0625 = a_by_vzt[0.625]
    r_0675 = a_by_vzt[0.675]
    r_060 = a_by_vzt[0.600]
    r_070 = a_by_vzt[0.700]

    sharpe_peak = r_065['Sharpe_OOS']
    sharpe_left = r_0625['Sharpe_OOS']
    sharpe_right = r_0675['Sharpe_OOS']
    delta_left  = sharpe_peak - sharpe_left
    delta_right = sharpe_peak - sharpe_right
    max_delta   = max(delta_left, delta_right)
    # 判定基準: max diff <= 0.02 → プラトー / >= 0.03 → スパイク / 中間 → 緩やかな山
    if max_delta <= 0.02:
        a_shape_verdict = 'プラトー（隣接2点との差 ≤ 0.02）'
        a_shape_tag = 'plateau'
    elif max_delta >= 0.03:
        a_shape_verdict = f'スパイク（隣接2点との差 ≥ 0.03、最大 {max_delta:+.4f}）'
        a_shape_tag = 'spike'
    else:
        a_shape_verdict = f'緩やかな山（隣接2点との差 0.02 < {max_delta:+.4f} < 0.03）'
        a_shape_tag = 'soft_peak'

    a_best = max(part_a, key=lambda r: r['Sharpe_OOS'])

    # Part B — Sharpe ≥ 0.930 AND MaxDD ≤ -57% 達成 config
    target_sharpe = 0.930
    target_maxdd = -0.57   # MaxDD ≤ -57% i.e. MaxDD > -0.57 (浅い)
    b_pass = [r for r in part_b
              if r['Sharpe_OOS'] >= target_sharpe and r['MaxDD_FULL'] > target_maxdd]

    # vz_thr=0.65, l_max=5.0 の組み合わせ
    target_065_5 = next((r for r in part_b
                         if abs(r['vz_thr'] - 0.65) < 1e-9 and abs(r['l_max'] - 5.0) < 1e-9), None)

    b_max_sharpe = max(part_b, key=lambda r: r['Sharpe_OOS'])
    b_min_dd     = max(part_b, key=lambda r: r['MaxDD_FULL'])

    print(f'\n[Part A 形状判定]')
    print(f'  vz_thr=0.65 Sharpe = {sharpe_peak:+.4f}')
    print(f'  vz_thr=0.625 Sharpe = {sharpe_left:+.4f}  (差: {-delta_left:+.4f})')
    print(f'  vz_thr=0.675 Sharpe = {sharpe_right:+.4f}  (差: {-delta_right:+.4f})')
    print(f'  判定: {a_shape_verdict}')

    print(f'\n[Part B 結論]')
    print(f'  Sharpe≥{target_sharpe} & MaxDD≤{target_maxdd*100:.0f}% 達成 configs: {len(b_pass)}/{total_b}')
    for r in b_pass:
        print(f'    vz_thr={r["vz_thr"]:.3f} l_max={r["l_max"]:.1f}: '
              f'CAGR={r["CAGR_OOS"]*100:+.2f}% MaxDD={r["MaxDD_FULL"]*100:+.2f}% '
              f'Sharpe={r["Sharpe_OOS"]:+.3f}')
    if target_065_5:
        print(f'  vz_thr=0.65 × l_max=5.0: '
              f'CAGR={target_065_5["CAGR_OOS"]*100:+.2f}% '
              f'MaxDD={target_065_5["MaxDD_FULL"]*100:+.2f}% '
              f'Sharpe={target_065_5["Sharpe_OOS"]:+.3f}')

    # -----------------------------------------------------------------------
    # MD レポート生成
    # -----------------------------------------------------------------------
    hdr1_1p, hdr2_1p = MD_HEADER_1P

    # Part A table（vz_thr 昇順）
    rows_a = '\n'.join(
        fmt_row_1p(f'vz_thr={r["vz_thr"]:.3f}', r) for r in part_a
    )

    # Part B 9指標フル表（vz_thr × l_max 並び）— 2P フォーマッタ流用は不可なので 1P 拡張
    def fmt_row_2param(vzt, lm, r, ref_s2=0.770, ref_lt2=0.885):
        from _sweep_format import _fp1, _ff2, _gap_pp, _tr, _wfa
        mark = ' ★' if r['Sharpe_OOS'] > ref_lt2 else (' ◎' if r['Sharpe_OOS'] > ref_s2 else '')
        return (
            f'| vz={vzt:.3f}/lmax={lm:.1f} '
            f'| {_fp1(r["CAGR_OOS"])} '
            f'| {_ff2(r["Sharpe_OOS"])}{mark} '
            f'| {_fp1(r["MaxDD_FULL"])} '
            f'| {_fp1(r["Worst10Y_star"])} '
            f'| {_fp1(r["P10_5Y"])} '
            f'| {_gap_pp(r["IS_OOS_gap"])} '
            f'| {_tr(r.get("Trades_yr"))} '
            f'| {_wfa(r.get("WFA_CI95_lo"))} '
            f'| {_wfa(r.get("WFA_WFE"))} |'
        )

    rows_b = '\n'.join(
        fmt_row_2param(r['vz_thr'], r['l_max'], r) for r in part_b
    )

    # vz_thr × l_max クロス表（CAGR_OOS / MaxDD / Sharpe / W10Y★）
    def cross_table(metric_key, fmt='pct'):
        lines = ['| vz_thr \\ l_max | ' + ' | '.join(f'{lm:.1f}' for lm in LMAX_CROSS) + ' |']
        lines.append('|---:|' + '---:|' * len(LMAX_CROSS))
        for vzt in VZT_CROSS:
            cells = []
            for lm in LMAX_CROSS:
                r = next(x for x in part_b
                         if abs(x['vz_thr'] - vzt) < 1e-9 and abs(x['l_max'] - lm) < 1e-9)
                v = r[metric_key]
                if fmt == 'pct':
                    cells.append(f'{v*100:+5.2f}%')
                else:
                    cells.append(f'{v:+5.3f}')
            lines.append(f'| **{vzt:.3f}** | ' + ' | '.join(cells) + ' |')
        return '\n'.join(lines)

    cross_cagr   = cross_table('CAGR_OOS')
    cross_maxdd  = cross_table('MaxDD_FULL')
    cross_sharpe = cross_table('Sharpe_OOS', fmt='float')
    cross_w10y   = cross_table('Worst10Y_star')

    # WFA 推奨設定 Top3（Sharpe降順、ただし MaxDD -65% 未満は除外）
    cand = [r for r in (part_a + part_b) if r['MaxDD_FULL'] > -0.65]
    cand_uniq = {}
    for r in cand:
        key = (round(r['vz_thr'], 3), round(r['tv'], 3), round(r['l_max'], 2))
        if key not in cand_uniq or r['Sharpe_OOS'] > cand_uniq[key]['Sharpe_OOS']:
            cand_uniq[key] = r
    top3 = sorted(cand_uniq.values(), key=lambda r: r['Sharpe_OOS'], reverse=True)[:3]

    top3_lines = []
    for i, r in enumerate(top3, 1):
        top3_lines.append(
            f'  {i}. vz_thr={r["vz_thr"]:.3f}, tv={r["tv"]:.2f}, l_max={r["l_max"]:.1f} → '
            f'Sharpe={r["Sharpe_OOS"]:+.3f}, CAGR={r["CAGR_OOS"]*100:+.2f}%, '
            f'MaxDD={r["MaxDD_FULL"]*100:+.2f}%, W10Y★={r["Worst10Y_star"]*100:+.2f}%'
        )
    top3_block = '\n'.join(top3_lines)

    # WFA 推奨追加スイープ点
    if a_shape_tag == 'spike':
        wfa_additional = (
            '- **vz_thr=0.65 はスパイク**: WFA G6 では追加点 vz_thr=0.6375 / 0.6625 を加え、'
            '計5点で WFA fold ごとの最適点が 0.65 ± 0.025 内で一貫するか確認すべき。'
        )
    elif a_shape_tag == 'plateau':
        wfa_additional = (
            '- **vz_thr=0.65 はプラトー**: 隣接点 0.625 / 0.675 がほぼ同性能のため、'
            'WFA G6 では vz_thr=0.65 1点で代表させ、過学習リスクは低い。'
        )
    else:
        wfa_additional = (
            '- **vz_thr=0.65 は緩やかな山**: WFA G6 では 0.625 / 0.65 / 0.675 の3点で fold 一貫性を確認推奨。'
        )

    # Part B 採用判断
    if b_pass:
        b_best = max(b_pass, key=lambda r: r['CAGR_OOS'])
        b_recommend = (
            f'**Part B 採用判断**: Sharpe ≥ {target_sharpe} & MaxDD ≤ {target_maxdd*100:.0f}% を満たす '
            f'config {len(b_pass)} 個 → 最も CAGR が高い vz_thr={b_best["vz_thr"]:.3f}, '
            f'l_max={b_best["l_max"]:.1f} '
            f'(CAGR={b_best["CAGR_OOS"]*100:+.2f}%, MaxDD={b_best["MaxDD_FULL"]*100:+.2f}%, '
            f'Sharpe={b_best["Sharpe_OOS"]:+.3f}) を WFA shortlist 入り推奨。'
        )
    else:
        # 緩和条件: Sharpe ≥ 0.920 & MaxDD ≤ -57%
        soft_pass = [r for r in part_b
                     if r['Sharpe_OOS'] >= 0.920 and r['MaxDD_FULL'] > -0.57]
        if soft_pass:
            b_best = max(soft_pass, key=lambda r: r['CAGR_OOS'])
            b_recommend = (
                f'**Part B 採用判断**: 厳格条件 Sharpe ≥ {target_sharpe} & MaxDD ≤ {target_maxdd*100:.0f}% '
                f'の達成なし。緩和条件 Sharpe ≥ 0.920 では {len(soft_pass)} 個達成 → '
                f'vz_thr={b_best["vz_thr"]:.3f}, l_max={b_best["l_max"]:.1f} '
                f'(CAGR={b_best["CAGR_OOS"]*100:+.2f}%, MaxDD={b_best["MaxDD_FULL"]*100:+.2f}%, '
                f'Sharpe={b_best["Sharpe_OOS"]:+.3f}) を候補とする。'
            )
        else:
            b_recommend = (
                f'**Part B 採用判断**: Sharpe ≥ {target_sharpe} & MaxDD ≤ {target_maxdd*100:.0f}% を満たす '
                f'config なし、緩和条件 (Sharpe ≥ 0.920) でも該当なし。'
                f'vz_thr × l_max の同時最適化では現行 (vz_thr=0.70, l_max=7.0) からの大幅改善は困難。'
            )

    md = f"""\
# E5b: vz_thr 細密スイープ + vz_thr × l_max 交差グリッド

作成日: 2026-05-26
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**
ベースライン: vz_thr={BASE_VZT}, tv={BASE_TV}, l_max={BASE_LMAX} (現行 E4 採用)

## §1 目的（E5 follow-up）

E5 スイープで **vz_thr=0.65 が突出した Sharpe=+0.945** を記録したが、
0.05 刻みの隣接値（0.60, 0.70）と比べた山の形状（スパイクかプラトーか）が
未判定だった。本ラウンドでは:

- **Part A**: vz_thr を **0.025 刻み** で再スイープ（7点）し、0.65 のピーク形状を確認
- **Part B**: vz_thr × l_max **交差グリッド**（5×3=15）で MaxDD と Sharpe の同時最適点を探索

特に「**Sharpe ≥ 0.930 かつ MaxDD ≤ -57%**」を満たす設定の有無を判定する。

## §2 Part A: vz_thr 細密スイープ結果

**設定**: tv={BASE_TV}, l_max={BASE_LMAX}, k_lo={K_LO}, k_hi={K_HI}, k_mid={K_MID} 固定。
vz_thr のみを **0.575〜0.725 で 0.025 刻み** スイープ。

{hdr1_1p}
{hdr2_1p}
{rows_a}

{MD_WFA_NOTE}

### §2.1 山の形状判定

- vz_thr=0.625 → Sharpe={sharpe_left:+.4f}
- **vz_thr=0.650 → Sharpe={sharpe_peak:+.4f}** （E5 ピーク）
- vz_thr=0.675 → Sharpe={sharpe_right:+.4f}
- 左差 (0.65 - 0.625): {delta_left:+.4f}
- 右差 (0.65 - 0.675): {delta_right:+.4f}
- 最大差: {max_delta:+.4f}

**形状判定**: **{a_shape_verdict}**

- 判定基準: 隣接2点との差 ≤ 0.02 → プラトー（良好）/ ≥ 0.03 → スパイク（過学習リスク）/ 中間 → 緩やかな山

## §3 Part B: vz_thr × l_max 交差グリッド

**設定**: tv={BASE_TV}, k_lo={K_LO}, k_hi={K_HI}, k_mid={K_MID} 固定。
vz_thr ∈ {VZT_CROSS}, l_max ∈ {LMAX_CROSS} の 5×3=15 configs。

### §3.1 9指標フル表

{hdr1_1p}
{hdr2_1p}
{rows_b}

{MD_WFA_NOTE}

### §3.2 クロス表: CAGR_OOS（vz_thr × l_max）

{cross_cagr}

### §3.3 クロス表: MaxDD（vz_thr × l_max）

{cross_maxdd}

### §3.4 クロス表: Sharpe_OOS（vz_thr × l_max）

{cross_sharpe}

### §3.5 クロス表: Worst10Y★ CAGR（vz_thr × l_max）

{cross_w10y}

## §4 重要発見

### §4.1 vz_thr=0.65 はスパイクかプラトーか

**結論**: **{a_shape_verdict}**

- vz_thr=0.625 / 0.650 / 0.675 の Sharpe 差は左 {delta_left:+.4f}, 右 {delta_right:+.4f}
- {'隣接2点との差が 0.02 以下のため、0.65 は安定したプラトーで偶然の勝者リスクは低い。' if a_shape_tag == 'plateau' else 'WFA fold 別の最適点が 0.65 から大きくずれる可能性を WFA G6 で要確認。' if a_shape_tag == 'spike' else '緩やかな山だが、WFA で 0.625-0.675 の3点で fold 一貫性確認推奨。'}

### §4.2 vz_thr=0.65 × l_max=5.0 の組み合わせ

{
f'**達成指標**: CAGR_OOS={target_065_5["CAGR_OOS"]*100:+.2f}%, '
f'Sharpe={target_065_5["Sharpe_OOS"]:+.3f}, '
f'MaxDD={target_065_5["MaxDD_FULL"]*100:+.2f}%, '
f'W10Y★={target_065_5["Worst10Y_star"]*100:+.2f}%, '
f'P10_5Y={target_065_5["P10_5Y"]*100:+.2f}%, '
f'IS-OOS gap={target_065_5["IS_OOS_gap"]*100:+.2f}pp, '
f'Trades/yr={int(round(target_065_5["Trades_yr"]))}'
if target_065_5 else 'データなし（計算ミス）'
}

- 比較ベースライン (vz_thr=0.70, l_max=7.0): CAGR={r_070["CAGR_OOS"]*100:+.2f}%, Sharpe={r_070["Sharpe_OOS"]:+.3f}, MaxDD={r_070["MaxDD_FULL"]*100:+.2f}%
- 比較 (vz_thr=0.65, l_max=7.0): CAGR={r_065["CAGR_OOS"]*100:+.2f}%, Sharpe={r_065["Sharpe_OOS"]:+.3f}, MaxDD={r_065["MaxDD_FULL"]*100:+.2f}%

### §4.3 「Sharpe ≥ 0.930 かつ MaxDD ≤ -57%」設定の有無

- 該当 configs: **{len(b_pass)} / {total_b}**
{chr(10).join(f'  - vz_thr={r["vz_thr"]:.3f}, l_max={r["l_max"]:.1f}: CAGR={r["CAGR_OOS"]*100:+.2f}%, MaxDD={r["MaxDD_FULL"]*100:+.2f}%, Sharpe={r["Sharpe_OOS"]:+.3f}' for r in b_pass) if b_pass else '  - 該当なし'}

- 最高 Sharpe (Part B): vz_thr={b_max_sharpe["vz_thr"]:.3f}, l_max={b_max_sharpe["l_max"]:.1f} → Sharpe={b_max_sharpe["Sharpe_OOS"]:+.3f}, MaxDD={b_max_sharpe["MaxDD_FULL"]*100:+.2f}%
- 最浅 MaxDD (Part B): vz_thr={b_min_dd["vz_thr"]:.3f}, l_max={b_min_dd["l_max"]:.1f} → MaxDD={b_min_dd["MaxDD_FULL"]*100:+.2f}%, Sharpe={b_min_dd["Sharpe_OOS"]:+.3f}

## §5 WFA 推奨設定リスト（上位3候補）

Sharpe_OOS 降順（MaxDD ≤ -65% フィルタ適用後）:

{top3_block}

{b_recommend}

### §5.1 G6 (WFA for vz_thr=0.65) 追加スイープ提案

{wfa_additional}

### §5.2 Round 3C (G7 WFA F10) への知見

- **vz_thr=0.65 のピーク形状**: {a_shape_verdict.split('（')[0]} → F10 で複数パラメータと組み合わせる際の基準点として {'安心して採用可能' if a_shape_tag == 'plateau' else '慎重に WFA fold 検証必要' if a_shape_tag == 'spike' else '3点比較で確認推奨'}
- **l_max=5.0 の MaxDD 改善効果**: l_max=7.0 → 5.0 で MaxDD 約 {(r_065["MaxDD_FULL"] - (target_065_5["MaxDD_FULL"] if target_065_5 else 0))*100:+.2f}pp 改善（vz_thr=0.65 比較時）、CAGR 犠牲は約 {(r_065["CAGR_OOS"] - (target_065_5["CAGR_OOS"] if target_065_5 else 0))*100:+.2f}pp
- **次の探索方向**: vz_thr × l_max 軸では Sharpe ≥ 0.93 & MaxDD ≤ -57% の同時達成{'は確認済み（F10 へ昇格）' if b_pass else 'は困難。別軸（k_mid 細密、DH signal multiplier、bond/gold スリーブ比率など）を検討'}

## §6 再現コマンド

```bash
cd C:\\Users\\user\\Desktop\\投資・不動産\\nasdaq_backtest
python src/e5b_fine_sweep.py
```

実行時間: 22 configs × 1 NAV 構築 ≈ 数分 (E5 と同等)。

出力:
- `e5b_fine_sweep_results.csv` — 22 行 × 9指標
- `E5B_FINE_SWEEP_2026-05-26.md` — 本レポート

---

*生成スクリプト: `src/e5b_fine_sweep.py`*
*参照: `E5_VZT_TV_SWEEP_2026-05-26.md`, `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`*
"""
    md_path = os.path.join(BASE, 'E5B_FINE_SWEEP_2026-05-26.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'MD saved:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
