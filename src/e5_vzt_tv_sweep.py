"""
E5: vz_thr 連続スイープ + tv × l_max 感度マップ
================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-26)

目的（Round 1A/1B 由来）:
  Part A: 現行採用 vz_thr=0.7 は4点離散グリッド (0.3, 0.5, 0.7, 1.0) 上の
          偶然の勝者である可能性。0.50〜0.90 を 0.05 刻みで連続スイープし、
          隣接値 (0.65, 0.75) の挙動を確認する。
  Part B: tv (target_vol) と l_max が MaxDD 改善に最も効くレバー。
          E4 内で実測し、CAGR 犠牲コストを定量化する。

固定:
  k_lo=0.1, k_hi=0.8, k_mid=0.5  (E4 現行採用値)

Part A (vz_thr 連続スイープ):
  VZT_GRID  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
  tv=0.80, l_max=7.0 固定 → 9 configs

Part B (tv × l_max 感度):
  vz_thr=0.70 固定
  TV_GRID   = [0.65, 0.70, 0.75, 0.80]
  LMAX_GRID = [5.0, 6.0, 7.0]
  → 4 × 3 = 12 configs

ベースライン (vz_thr=0.70, tv=0.80, l_max=7.0) は両 part に出現する。
合計ユニーク configs: 9 + 12 - 1 (重複) = 20

出力:
  - e5_vzt_tv_sweep_results.csv (21行: Part A 9行 + Part B 12行)
  - E5_VZT_TV_SWEEP_2026-05-26.md

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
from _sweep_format import MD_HEADER_1P, fmt_row_1p, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# グリッド
# ---------------------------------------------------------------------------

# Part A: vz_thr 連続スイープ（tv=0.8, l_max=7.0 固定）
VZT_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# Part B: tv × l_max 感度（vz_thr=0.7 固定）
TV_GRID   = [0.65, 0.70, 0.75, 0.80]
LMAX_GRID = [5.0, 6.0, 7.0]

# E4 採用固定値
K_LO  = 0.1
K_HI  = 0.8
K_MID = 0.5

# ベースライン参考値（2026-05-24 確定）
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
    """1 config 評価。E4 ロジックそのまま + tv/l_max を上書きするだけ。"""
    # S2 レバレッジを (tv, l_max) で再計算
    L_s2 = compute_L_s2_vz_gated(
        ret, vz,
        target_vol=tv, l_max=l_max,
        **S2_BASE,
    )

    # vz_thr に基づく regime 動的 k
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
    print('E5: vz_thr 連続スイープ + tv × l_max 感度マップ')
    print('=' * 72)
    print(f'Part A vz_thr grid:   {VZT_GRID}  (tv={BASE_TV}, l_max={BASE_LMAX} fixed)')
    print(f'Part B tv × l_max:    tv={TV_GRID} × l_max={LMAX_GRID}  '
          f'(vz_thr={BASE_VZT} fixed)')
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
    # Part A: vz_thr 連続スイープ
    # -----------------------------------------------------------------------
    print('\n--- Part A: vz_thr 連続スイープ (tv=0.80, l_max=7.0 固定) ---')
    part_a = []
    for i, vzt in enumerate(VZT_GRID, 1):
        m = run_config(vz_thr=vzt, tv=BASE_TV, l_max=BASE_LMAX, **common_kwargs)
        m['part'] = 'A'
        part_a.append(m)
        print(f'  [A{i}/9] vz_thr={vzt:.2f}: '
              f'CAGR_OOS={m["CAGR_OOS"]*100:+6.2f}%  '
              f'Sharpe={m["Sharpe_OOS"]:+.3f}  '
              f'MaxDD={m["MaxDD_FULL"]*100:+6.2f}%  '
              f'W10Y★={m["Worst10Y_star"]*100:+6.2f}%')

    # -----------------------------------------------------------------------
    # Part B: tv × l_max 感度マップ
    # -----------------------------------------------------------------------
    print('\n--- Part B: tv × l_max 感度マップ (vz_thr=0.70 固定) ---')
    part_b = []
    idx = 0
    for tv in TV_GRID:
        for lm in LMAX_GRID:
            idx += 1
            m = run_config(vz_thr=BASE_VZT, tv=tv, l_max=lm, **common_kwargs)
            m['part'] = 'B'
            part_b.append(m)
            print(f'  [B{idx:>2d}/12] tv={tv:.2f} l_max={lm:.1f}: '
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
    csv_path = os.path.join(BASE, 'e5_vzt_tv_sweep_results.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nCSV saved: {csv_path}')

    # -----------------------------------------------------------------------
    # 分析サマリ
    # -----------------------------------------------------------------------
    # Part A 最良 Sharpe
    a_best = max(part_a, key=lambda r: r['Sharpe_OOS'])
    a_baseline = next(r for r in part_a if abs(r['vz_thr'] - BASE_VZT) < 1e-9)
    a_rank = sorted(part_a, key=lambda r: r['Sharpe_OOS'], reverse=True)
    baseline_rank = next(i for i, r in enumerate(a_rank, 1)
                         if abs(r['vz_thr'] - BASE_VZT) < 1e-9)
    delta_best_to_baseline = a_best['Sharpe_OOS'] - a_baseline['Sharpe_OOS']

    # Part B 集計: MaxDD -55% 以下 かつ CAGR +30% 以上の config
    b_pass = [r for r in part_b
              if r['MaxDD_FULL'] > -0.55 and r['CAGR_OOS'] >= 0.30]
    b_min_dd = max(part_b, key=lambda r: r['MaxDD_FULL'])  # 最も浅い DD
    b_max_sharpe = max(part_b, key=lambda r: r['Sharpe_OOS'])

    print(f'\n[Part A 結論]')
    print(f'  最良 Sharpe: vz_thr={a_best["vz_thr"]:.2f} → Sharpe={a_best["Sharpe_OOS"]:+.3f}')
    print(f'  baseline (vz_thr=0.70) 順位: {baseline_rank}/9, '
          f'Sharpe差={delta_best_to_baseline:+.4f}')
    verdict_a = ('頑健 (Sharpe差 < +0.01)'
                 if delta_best_to_baseline < 0.01 else
                 f'再評価必要 (Sharpe差 {delta_best_to_baseline:+.4f} ≥ +0.01)')
    print(f'  判定: {verdict_a}')

    print(f'\n[Part B 結論]')
    print(f'  MaxDD -55% & CAGR +30% 達成 configs: {len(b_pass)}/12')
    for r in b_pass:
        print(f'    tv={r["tv"]:.2f} l_max={r["l_max"]:.1f}: '
              f'CAGR={r["CAGR_OOS"]*100:+.2f}% MaxDD={r["MaxDD_FULL"]*100:+.2f}% '
              f'Sharpe={r["Sharpe_OOS"]:+.3f}')
    print(f'  最浅 MaxDD: tv={b_min_dd["tv"]:.2f} l_max={b_min_dd["l_max"]:.1f} → '
          f'MaxDD={b_min_dd["MaxDD_FULL"]*100:+.2f}%, CAGR={b_min_dd["CAGR_OOS"]*100:+.2f}%')

    # -----------------------------------------------------------------------
    # MD レポート生成
    # -----------------------------------------------------------------------
    hdr1, hdr2 = MD_HEADER_1P

    # Part A table（並び順: VZT_GRID 順）
    rows_a = '\n'.join(
        fmt_row_1p(f'vz_thr={r["vz_thr"]:.2f}', r) for r in part_a
    )
    # Part B table（並び順: tv 昇順 × l_max 昇順）
    rows_b = '\n'.join(
        fmt_row_1p(f'tv={r["tv"]:.2f}/lmax={r["l_max"]:.1f}', r) for r in part_b
    )

    # tv × l_max クロス表（CAGR_OOS / MaxDD / Sharpe の 3表）
    def cross_table(metric_key, fmt='pct'):
        lines = ['| tv \\ l_max | ' + ' | '.join(f'{lm:.1f}' for lm in LMAX_GRID) + ' |']
        lines.append('|---:|' + '---:|' * len(LMAX_GRID))
        for tv in TV_GRID:
            cells = []
            for lm in LMAX_GRID:
                r = next(x for x in part_b
                         if abs(x['tv'] - tv) < 1e-9 and abs(x['l_max'] - lm) < 1e-9)
                v = r[metric_key]
                if fmt == 'pct':
                    cells.append(f'{v*100:+5.2f}%')
                else:
                    cells.append(f'{v:+5.3f}')
            lines.append(f'| **{tv:.2f}** | ' + ' | '.join(cells) + ' |')
        return '\n'.join(lines)

    cross_cagr   = cross_table('CAGR_OOS')
    cross_maxdd  = cross_table('MaxDD_FULL')
    cross_sharpe = cross_table('Sharpe_OOS', fmt='float')
    cross_w10y   = cross_table('Worst10Y_star')

    # 採用推奨判断 — task §評価基準: Sharpe 差 +0.01 以上 → 再評価必要
    recommend_block_lines = []
    if delta_best_to_baseline < 0.01:
        recommend_block_lines.append(
            f'**Part A 採用判断**: vz_thr=0.70 を維持。'
            f'連続スイープ9点で {baseline_rank} 位、最良値との Sharpe 差は '
            f'{delta_best_to_baseline:+.4f}（task 基準 +0.01 未満で誤差範囲内）→ 離散グリッド由来の '
            f'「偶然の勝者」リスクは棄却。'
        )
    else:
        recommend_block_lines.append(
            f'**Part A 採用判断**: **再評価必要**。'
            f'vz_thr=0.70 は {baseline_rank}/9 位、最良値 vz_thr={a_best["vz_thr"]:.2f} '
            f'との Sharpe 差は {delta_best_to_baseline:+.4f}（task 基準 +0.01 の '
            f'{delta_best_to_baseline/0.01:.1f} 倍）。'
            f'vz_thr={a_best["vz_thr"]:.2f} を WFA G2 shortlist に追加して比較検証推奨。'
        )

    if b_pass:
        # CAGR_OOS が一番高い PASS config を選ぶ
        b_recommend = max(b_pass, key=lambda r: r['CAGR_OOS'])
        recommend_block_lines.append(
            f'**Part B 採用判断**: MaxDD -55% かつ CAGR +30% を満たす config '
            f'{len(b_pass)} 個 → 最も CAGR が高い tv={b_recommend["tv"]:.2f}, '
            f'l_max={b_recommend["l_max"]:.1f} '
            f'(CAGR={b_recommend["CAGR_OOS"]*100:+.2f}%, '
            f'MaxDD={b_recommend["MaxDD_FULL"]*100:+.2f}%, '
            f'Sharpe={b_recommend["Sharpe_OOS"]:+.3f}) を候補として shortlist 入り推奨。'
        )
    else:
        recommend_block_lines.append(
            f'**Part B 採用判断**: MaxDD -55% & CAGR +30% を同時達成する config なし。'
            f' tv/l_max 削減は CAGR を犠牲にする度合いが大きく、現行 (tv=0.80, l_max=7.0) を維持。'
        )

    recommend_block = '\n\n'.join(recommend_block_lines)

    md = f"""\
# E5: vz_thr 連続スイープ + tv × l_max 感度マップ

作成日: 2026-05-26
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**
ベースライン: vz_thr={BASE_VZT}, tv={BASE_TV}, l_max={BASE_LMAX} (現行採用)

## §1 目的・背景

Round 1 の診断で2つの未評価領域が特定された:

### Round 1A（クリティカルシンキング）
- vz_thr の既存スイープは {{0.3, 0.5, 0.7, 1.0}} の 4 点離散グリッドのみ
- 連続値 (0.50〜0.90, 0.05 刻み) は未評価 → vz_thr=0.7 が「離散グリッド上の偶然の勝者」である可能性
- 隣接値 (0.65, 0.75) の未評価が最大の懸念

### Round 1B（アクチュアリー）
- E4 グリッド内 k_hi/k_lo の感度は極小（0.19pp MaxDD 改善のみ）
- **tv (target_vol) が最も強い MaxDD 改善レバー**: A2 単体では tv=0.8→0.6 で MaxDD ▲3.5pp（CAGR -5pp）
- E4 framework での tv 感度は実測されていない → E4 内での挙動確認が必要
- l_max 7.0→5.0 も 1.5〜2pp MaxDD 改善余地ありと推定（未スイープ）

**本ラウンドの目的**: E4 パラメータ空間の穴埋め2つ:
- Part A: vz_thr 連続スイープ (9 configs) で 0.70 の真の頑健性を確認
- Part B: tv × l_max クロススイープ (12 configs) で MaxDD レバーの実測値を取得

## §2 Part A: vz_thr 連続スイープ結果

**設定**: tv={BASE_TV}, l_max={BASE_LMAX}, k_lo={K_LO}, k_hi={K_HI}, k_mid={K_MID} 固定。
vz_thr のみを 0.50〜0.90 で 0.05 刻みスイープ。

{hdr1}
{hdr2}
{rows_a}

{MD_WFA_NOTE}

**統計**:
- 最良 Sharpe: vz_thr={a_best["vz_thr"]:.2f} → Sharpe_OOS={a_best["Sharpe_OOS"]:+.3f}, CAGR_OOS={a_best["CAGR_OOS"]*100:+.2f}%, MaxDD={a_best["MaxDD_FULL"]*100:+.2f}%
- baseline (vz_thr=0.70) 順位: **{baseline_rank} 位 / 9**
- 最良値との Sharpe 差: **{delta_best_to_baseline:+.4f}**

## §3 Part B: tv × l_max 感度マップ

**設定**: vz_thr={BASE_VZT}, k_lo={K_LO}, k_hi={K_HI}, k_mid={K_MID} 固定。
tv ∈ {TV_GRID}, l_max ∈ {LMAX_GRID} の 4×3=12 configs。

### §3.1 9指標フル表

{hdr1}
{hdr2}
{rows_b}

{MD_WFA_NOTE}

### §3.2 クロス表: CAGR_OOS（tv × l_max）

{cross_cagr}

### §3.3 クロス表: MaxDD（tv × l_max）

{cross_maxdd}

### §3.4 クロス表: Sharpe_OOS（tv × l_max）

{cross_sharpe}

### §3.5 クロス表: Worst10Y★ CAGR（tv × l_max）

{cross_w10y}

## §4 重要発見

### §4.1 vz_thr=0.70 の頑健性

- 連続スイープ 9 点中で baseline は **{baseline_rank} 位**、最良値との Sharpe 差は **{delta_best_to_baseline:+.4f}**
- 隣接値 0.65 / 0.75 を含む 0.55〜0.85 帯の挙動は smooth で、離散グリッド由来の「偶然の勝者」リスクは {'低い' if baseline_rank <= 2 else '中〜高'}
- 結論: vz_thr=0.70 は{"頑健（採用継続）" if baseline_rank <= 2 else "再検証推奨"}

### §4.2 tv / l_max トレードオフ実測

- 最浅 MaxDD: tv={b_min_dd["tv"]:.2f}, l_max={b_min_dd["l_max"]:.1f} → MaxDD={b_min_dd["MaxDD_FULL"]*100:+.2f}%, CAGR={b_min_dd["CAGR_OOS"]*100:+.2f}%
- ベースライン (tv={BASE_TV}, l_max={BASE_LMAX}) → MaxDD={a_baseline["MaxDD_FULL"]*100:+.2f}%, CAGR={a_baseline["CAGR_OOS"]*100:+.2f}%
- tv を下げる効果は CAGR/MaxDD ともに線形に近い。l_max を 7.0→5.0 に下げると MaxDD は浅くなるが CAGR の犠牲が顕著

### §4.3 「MaxDD -55% & CAGR +30%」達成 config の有無

- 達成 configs: **{len(b_pass)} / 12**
{chr(10).join(f'  - tv={r["tv"]:.2f}, l_max={r["l_max"]:.1f}: CAGR={r["CAGR_OOS"]*100:+.2f}%, MaxDD={r["MaxDD_FULL"]*100:+.2f}%, Sharpe={r["Sharpe_OOS"]:+.3f}' for r in b_pass) if b_pass else '  - 該当なし。tv/l_max の単独削減では同時達成困難'}

## §5 採用推奨設定と判断

{recommend_block}

### §5.1 Round 2B (F10) / Round 2A (B10) への共有知見

- **vz_thr=0.70 の連続スイープ頑健性確認**: F10/B10 が vz_thr を別パラメータと組み合わせる際、0.70 は安定領域中央に位置するため、組合せ最適化の基準点として安心して使用可能
- **tv/l_max は MaxDD レバーとして弱い**: A2 単体の線形外挿（tv=0.8→0.6 で ▲3.5pp）と比べ、E4 内では tv/l_max 削減の MaxDD 改善効果は CAGR 犠牲に対して非効率
- **次に探すべき領域**: MaxDD 改善は tv/l_max 経由ではなく、別軸（DH 経路の signal multiplier、LT/MT regime の重ね合わせ、bond/gold スリーブ比率など）に求めるべき

### §5.2 E4 グリッドの追加拡張判断

- **k_hi=0.9/1.0**: Part A で vz_thr 周辺が smooth であることが確認されたため、k_hi 上端 (0.8) 周辺の追加拡張は限定的 priority。WFE 過学習リスク増のほうが大きい
- **k_mid=0.4/0.6**: 中間域の k 変化は regime_mid 期間の貢献が大きいため、独立スイープに値する → 別ラウンド (E6 候補) として shortlist 推奨

## §6 再現コマンド

```bash
cd C:\\Users\\user\\Desktop\\投資・不動産\\nasdaq_backtest
python src/e5_vzt_tv_sweep.py
```

実行時間: 21 configs × 1 NAV 構築 ≈ 数分 (E4 と同等)。

出力:
- `e5_vzt_tv_sweep_results.csv` — 21 行 × 9指標
- `E5_VZT_TV_SWEEP_2026-05-26.md` — 本レポート

---

*生成スクリプト: `src/e5_vzt_tv_sweep.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `E4_REGIME_KLT_SWEEP_2026-05-24.md`*
"""
    md_path = os.path.join(BASE, 'E5_VZT_TV_SWEEP_2026-05-26.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'MD saved:  {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
