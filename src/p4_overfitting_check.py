"""
P4 過学習確認 (Overfitting Check)
====================================
作成日: 2026-05-18

目的:
  P2 (38コンボ) + P3 (34コンボ) = 72試行に対して
  1. DSR (Deflated Sharpe Ratio) 多重検定補正
  2. 5-Fold Walk-Forward CV
  を実施し、最良候補が統計的に有意か確認する。

対象コンボ (top5 + baseline):
  Baseline:    DH Dyn [A] ゲートなし
  P01_Dyn×HY: bond_gate=Dyn_Corr(w=60,mg=0.2) + nas_gate=HY(z=1.0,slope=0.5)
  P02_Dyn×CPI: bond_gate=Dyn_Corr(w=60,mg=0.2) + nas_gate=CPI(thresh=5.0,r=0.3)
  P03_Dyn×MA:  bond_gate=Dyn_Corr(w=60,mg=0.2) + nas_gate=MA(w=200,h=0.6)
  P05_HY×CPI:  nas_gate=HY(z=1.0,slope=0.5) × CPI(thresh=5.0,r=0.3) (multiplicative)
  P06_HY×MA:   nas_gate=HY(z=1.0,slope=0.5) × MA(w=200,h=0.6)       (multiplicative)
"""

import sys
import os
import types

# multitasking stub (yfinance dependency)
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis as kurt

from corrected_strategy_backtest import (
    load_sofr,
    build_bond_1x_nav_corrected,
    build_bond_3x,
    build_a2_signal,
    simulate_rebalance_A,
    TRADING_DAYS,
    DELAY,
    THRESHOLD,
)
from test_portfolio_diversification import prepare_gold_data
from sleeves_extended import build_gold_tocom
from cfd_leverage_backtest import build_nav_strategy, calc_7metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')
SIGNALS_PATH = os.path.join(BASE, 'data', 'timing_signals_raw.csv')
DATE_STR = '2026-05-18'

# ---------------------------------------------------------------------------
# P2+P3 OOS Sharpe values (hardcoded, 72 total)
# ---------------------------------------------------------------------------
P2_SHARPES = [
    # HY (8)
    0.603, 0.611, 0.639, 0.633, 0.629, 0.637, 0.627, 0.642,
    # YC_FF (6)
    0.592, 0.604, 0.530, 0.552, 0.604, 0.552,
    # Dyn_Corr (6)
    0.818, 0.823, 0.814, 0.805, 0.827, 0.808,
    # CPI (9)
    0.644, 0.631, 0.614, 0.637, 0.619, 0.595, 0.629, 0.606, 0.574,
    # MA (9)
    0.589, 0.621, 0.608, 0.595, 0.570, 0.590, 0.551, 0.572, 0.554,
]  # total 38

P3_SHARPES = [
    # Pairs mult (10)
    0.829, 0.833, 0.798, 0.815, 0.667, 0.616, 0.616, 0.626, 0.622, 0.570,
    # Min sensitivity (2)
    0.829, 0.833,
    # L1 regrid Dyn×HY (9)
    0.812, 0.814, 0.814, 0.825, 0.827, 0.829, 0.814, 0.830, 0.831,
    # L1 regrid Dyn×CPI (9)
    0.813, 0.817, 0.818, 0.822, 0.833, 0.836, 0.824, 0.838, 0.840,
    # Triplets (3)
    0.803, 0.789, 0.812,
    # All-5 (1)
    0.798,
]  # total 34

ALL_SHARPES = P2_SHARPES + P3_SHARPES
assert len(ALL_SHARPES) == 72, f"Expected 72, got {len(ALL_SHARPES)}"

# ---------------------------------------------------------------------------
# Part 1: DSR (Deflated Sharpe Ratio)
# ---------------------------------------------------------------------------

def compute_dsr(sr_hat_annual, all_sharpes_annual, T, n_eff):
    """
    DSR = P(SR_true > E[max SR] | data)
    Bailey & Lopez de Prado, 2014

    Args:
        sr_hat_annual: 候補の年率Sharpe
        all_sharpes_annual: 72試行の年率Sharpe配列
        T: OOS観測日数 (≈1250)
        n_eff: 実質独立試行数

    Returns: (psr_deflated, e_max_sr_annual)
    """
    em = 0.5772156649  # Euler-Mascheroni constant

    # 年率→日率変換
    sr_daily = sr_hat_annual / np.sqrt(252)
    all_daily = np.array(all_sharpes_annual) / np.sqrt(252)

    # E[max SR] under null
    var_sr = np.var(all_daily, ddof=1)
    e_max_sr_daily = np.sqrt(var_sr) * (
        (1 - em) * norm.ppf(1 - 1.0 / n_eff)
        + em * norm.ppf(1 - 1.0 / (n_eff * np.e))
    )

    # PSR (Probabilistic SR): P(SR_true > 0) with moment correction
    g3 = skew(all_daily, bias=False)
    g4 = kurt(all_daily, fisher=False, bias=False)

    num = (sr_daily - e_max_sr_daily) * np.sqrt(T - 1)
    den_sq = 1 - g3 * sr_daily + ((g4 - 1) / 4) * sr_daily**2
    den = np.sqrt(max(den_sq, 1e-10))

    psr_z = num / den
    psr = float(norm.cdf(psr_z))

    return psr, float(e_max_sr_daily * np.sqrt(252))


def run_dsr_analysis(combo_sharpes_annual, all_sharpes, T_oos=1250):
    """DSR感度分析を実行する"""
    n_effs = [5, 8, 10, 15, 20]
    results = []

    for combo_name, sr_hat in combo_sharpes_annual.items():
        for n_eff in n_effs:
            psr, e_max = compute_dsr(sr_hat, all_sharpes, T_oos, n_eff)

            if n_eff == 8:
                if psr >= 0.95:
                    dsr_pass = 'ADOPT'
                elif psr >= 0.90:
                    dsr_pass = 'GRAY'
                else:
                    dsr_pass = 'REJECT'
            else:
                dsr_pass = 'ADOPT' if psr >= 0.95 else ('GRAY' if psr >= 0.90 else 'REJECT')

            results.append({
                'combo_name': combo_name,
                'sharpe_oos': sr_hat,
                'n_trials_total': len(all_sharpes),
                'n_eff': n_eff,
                'e_max_sr_annual': round(e_max, 4),
                'psr_deflated': round(psr, 4),
                'dsr_pass': dsr_pass,
            })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Part 2: Gate builders
# ---------------------------------------------------------------------------

def build_hy_gate(hy, z_thresh, slope):
    mu = hy.rolling(252, min_periods=126).mean()
    sd = hy.rolling(252, min_periods=126).std().clip(lower=0.01)
    z = (hy - mu) / sd
    g = (1.0 - np.maximum(0.0, z - z_thresh) * slope).clip(0.2, 1.0)
    return g.fillna(1.0)


def build_cpi_gate(cpi_yoy, cpi_accel, cpi_thresh, reduce_factor):
    infl_regime = ((cpi_yoy - cpi_thresh) / 5.0).clip(0.0, 1.0)
    accel_norm = (cpi_accel / 2.0).clip(0.0, 1.0)
    g = (1.0 - reduce_factor * np.maximum(infl_regime, accel_norm)).clip(
        1.0 - reduce_factor, 1.0
    )
    return g.fillna(1.0)


def build_ma_gate(close, ma_window, half_lev):
    ma = close.rolling(ma_window).mean()
    gate_vals = np.where(close.values >= ma.values, 1.0, half_lev)
    g = pd.Series(
        np.where(np.isnan(ma.values), 1.0, gate_vals), index=close.index
    )
    return g


def build_corr_gate(close, bond_3x, gold_2x, window, min_gate):
    ret = pd.Series(close.pct_change().fillna(0).values, index=close.index)
    bond_ret = pd.Series(bond_3x, index=close.index).pct_change().fillna(0)
    gold_ret = pd.Series(gold_2x, index=close.index).pct_change().fillna(0)
    rho_nb = ret.rolling(window).corr(bond_ret)
    rho_ng = ret.rolling(window).corr(gold_ret)
    hedge_health = (-rho_nb).clip(lower=0.0) + (-rho_ng).clip(lower=0.0)
    g = hedge_health.clip(lower=min_gate, upper=1.0)
    return g.fillna(1.0)


def apply_gates(wn_A_arr, nas_gate=None, bond_gate=None):
    ones = np.ones(len(wn_A_arr))
    g_nas = np.where(np.isnan(nas_gate), 1.0, nas_gate) if nas_gate is not None else ones
    g_bond = np.where(np.isnan(bond_gate), 1.0, bond_gate) if bond_gate is not None else ones
    wn = np.clip(wn_A_arr * g_nas, 0.0, 1.0)
    rest = 1.0 - wn
    wg = rest * 0.5
    wb = rest * 0.5 * g_bond
    return wn, wg, wb


# ---------------------------------------------------------------------------
# Part 3: Fold metrics
# ---------------------------------------------------------------------------

def fold_metrics(nav_series, dates_series, fold_start, fold_end):
    """フォールド内メトリクスを計算する"""
    mask = (dates_series >= pd.Timestamp(fold_start)) & (
        dates_series <= pd.Timestamp(fold_end)
    )
    if mask.sum() < 100:
        return None
    ns = nav_series[mask].values.copy()
    ns = ns / ns[0]  # 初期NAVを1に正規化
    r = pd.Series(ns).pct_change().fillna(0).values
    n = len(ns)
    yrs = n / 252.0
    if yrs <= 0 or ns[-1] <= 0:
        return None
    cagr = ns[-1] ** (1 / yrs) - 1
    sharpe = (r.mean() * 252) / (r.std() * np.sqrt(252) + 1e-10)
    maxdd = float(np.min(ns / np.maximum.accumulate(ns) - 1))
    # Worst1Y within fold
    ns_s = pd.Series(ns)
    r1y = (ns_s / ns_s.shift(252)) ** 1.0 - 1
    worst1y = float(r1y.min()) if len(ns_s) >= 252 else np.nan
    return {
        'cagr': cagr,
        'sharpe': sharpe,
        'maxdd': maxdd,
        'worst1y': worst1y,
        'n_days': n,
    }


def compute_cv_folds(nav, dates_dt):
    """5フォールドWalk-Forward CV用フォールド定義と計算"""
    # OOS期間: 2021-05-08 〜 2026-03-26
    # 取引日ベースでフォールド境界を計算
    oos_mask = dates_dt >= pd.Timestamp('2021-05-08')
    oos_dates = dates_dt[oos_mask]
    n_oos = len(oos_dates)
    print(f"  OOS期間の取引日数: {n_oos} (2021-05-08 〜 {oos_dates.max().strftime('%Y-%m-%d')})")

    # 各フォールドを250日ベースで区切る（embago: 10日）
    embargo = 10
    fold_size = 250

    fold_boundaries = []
    start_idx = 0
    for i in range(5):
        end_idx = min(start_idx + fold_size - 1, n_oos - 1)
        fold_start = oos_dates.iloc[start_idx]
        fold_end = oos_dates.iloc[end_idx]
        fold_boundaries.append((f'Fold-{i+1}', fold_start, fold_end))
        # 次フォールドは embargo 日後から
        start_idx = end_idx + 1 + embargo
        if start_idx >= n_oos:
            break

    # Fold-5が不足している場合は残り期間
    if len(fold_boundaries) == 4:
        last_end_idx = n_oos - 1 - fold_boundaries[-1][2].isin(oos_dates).sum()
        # シンプルに最後のフォールド終端から残り全部
        prev_end = fold_boundaries[-1][2]
        prev_end_pos = oos_dates.searchsorted(prev_end)
        next_start_pos = min(prev_end_pos + 1 + embargo, n_oos - 1)
        if next_start_pos < n_oos:
            fold5_start = oos_dates.iloc[next_start_pos]
            fold5_end = oos_dates.iloc[-1]
            if (fold5_end - fold5_start).days > 100:
                fold_boundaries.append(('Fold-5', fold5_start, fold5_end))

    for fname, fs, fe in fold_boundaries:
        n_days = len(oos_dates[(oos_dates >= fs) & (oos_dates <= fe)])
        print(f"  {fname}: {fs.strftime('%Y-%m-%d')} ~ {fe.strftime('%Y-%m-%d')} ({n_days}日)")

    return fold_boundaries


# ---------------------------------------------------------------------------
# Main data setup
# ---------------------------------------------------------------------------

def setup_data():
    """データのロードと前処理"""
    print("[データロード]")
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    close = pd.Series(df['Close'].values, dtype=float)
    dates_str = pd.Series(df['Date'].dt.strftime('%Y-%m-%d').values)
    dates_dt = pd.to_datetime(dates_str)
    dates_idx = pd.DatetimeIndex(dates_dt.values)

    print("  NASDAQ: {} rows ({} ~ {})".format(
        len(close),
        dates_dt.min().strftime('%Y-%m-%d'),
        dates_dt.max().strftime('%Y-%m-%d')
    ))

    # Build financial components
    sofr = load_sofr(pd.Series(dates_str.values))
    bond_1x = build_bond_1x_nav_corrected(pd.Series(dates_str.values))
    gold_1x = prepare_gold_data(dates_dt)
    gold_2x = build_gold_tocom(gold_1x, 2.0, sofr)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=False)

    # A2 baseline signal
    raw_a2, vz_a2 = build_a2_signal(close, close.pct_change())
    lev, wn_A, wg_A, wb_A, n_trades = simulate_rebalance_A(raw_a2, vz_a2, THRESHOLD)

    # Load timing signals with DatetimeIndex alignment
    sig_raw = pd.read_csv(SIGNALS_PATH, index_col=0, parse_dates=True)
    sig = sig_raw.reindex(dates_idx)
    hy_s = pd.Series(sig['hy_spread'].values, index=close.index)
    dff_s = pd.Series(sig['dff'].fillna(0.0).values, index=close.index)
    cpi_yoy = pd.Series(sig['cpi_yoy'].fillna(0.0).values, index=close.index)
    cpi_acc = pd.Series(sig['cpi_accel'].fillna(0.0).values, index=close.index)

    print("  シグナルロード完了")

    return {
        'close': close,
        'dates_dt': dates_dt,
        'sofr': sofr,
        'gold_2x': gold_2x,
        'bond_3x': bond_3x,
        'lev': lev,
        'wn_A': wn_A,
        'hy_s': hy_s,
        'dff_s': dff_s,
        'cpi_yoy': cpi_yoy,
        'cpi_acc': cpi_acc,
    }


# ---------------------------------------------------------------------------
# NAV builder per combo
# ---------------------------------------------------------------------------

def build_combo_nav(combo_name, data):
    """コンボ名からNAVを構築する"""
    close = data['close']
    dates_dt = data['dates_dt']
    sofr = data['sofr']
    gold_2x = data['gold_2x']
    bond_3x = data['bond_3x']
    lev = data['lev']
    wn_A = data['wn_A']
    hy_s = data['hy_s']
    cpi_yoy = data['cpi_yoy']
    cpi_acc = data['cpi_acc']

    if combo_name == 'Baseline':
        wn = wn_A.copy()
        rest = 1.0 - wn
        wg = rest * 0.5
        wb = rest * 0.5
        nav = build_nav_strategy(
            close, lev, wn, wg, wb, dates_dt,
            gold_2x, bond_3x, sofr,
            nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002
        )

    elif combo_name == 'P01_Dyn×HY':
        # bond_gate=Dyn_Corr(w=60,mg=0.2), nas_gate=HY(z=1.0,slope=0.5)
        g_corr = build_corr_gate(close, bond_3x, gold_2x, window=60, min_gate=0.2)
        g_hy = build_hy_gate(hy_s, z_thresh=1.0, slope=0.5)
        wn, wg, wb = apply_gates(wn_A, nas_gate=g_hy.values, bond_gate=g_corr.values)
        nav = build_nav_strategy(
            close, lev, wn, wg, wb, dates_dt,
            gold_2x, bond_3x, sofr,
            nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002
        )

    elif combo_name == 'P02_Dyn×CPI':
        # bond_gate=Dyn_Corr(w=60,mg=0.2), nas_gate=CPI(thresh=5.0,r=0.3)
        g_corr = build_corr_gate(close, bond_3x, gold_2x, window=60, min_gate=0.2)
        g_cpi = build_cpi_gate(cpi_yoy, cpi_acc, cpi_thresh=5.0, reduce_factor=0.3)
        wn, wg, wb = apply_gates(wn_A, nas_gate=g_cpi.values, bond_gate=g_corr.values)
        nav = build_nav_strategy(
            close, lev, wn, wg, wb, dates_dt,
            gold_2x, bond_3x, sofr,
            nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002
        )

    elif combo_name == 'P03_Dyn×MA':
        # bond_gate=Dyn_Corr(w=60,mg=0.2), nas_gate=MA(w=200,h=0.6)
        g_corr = build_corr_gate(close, bond_3x, gold_2x, window=60, min_gate=0.2)
        g_ma = build_ma_gate(close, ma_window=200, half_lev=0.6)
        wn, wg, wb = apply_gates(wn_A, nas_gate=g_ma.values, bond_gate=g_corr.values)
        nav = build_nav_strategy(
            close, lev, wn, wg, wb, dates_dt,
            gold_2x, bond_3x, sofr,
            nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002
        )

    elif combo_name == 'P05_HY×CPI':
        # nas_gate=HY(z=1.0,slope=0.5) × CPI(thresh=5.0,r=0.3) multiplicative
        g_hy = build_hy_gate(hy_s, z_thresh=1.0, slope=0.5)
        g_cpi = build_cpi_gate(cpi_yoy, cpi_acc, cpi_thresh=5.0, reduce_factor=0.3)
        g_nas = g_hy.values * g_cpi.values
        g_nas = np.clip(g_nas, 0.2, 1.0)
        wn, wg, wb = apply_gates(wn_A, nas_gate=g_nas, bond_gate=None)
        nav = build_nav_strategy(
            close, lev, wn, wg, wb, dates_dt,
            gold_2x, bond_3x, sofr,
            nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002
        )

    elif combo_name == 'P06_HY×MA':
        # nas_gate=HY(z=1.0,slope=0.5) × MA(w=200,h=0.6) multiplicative
        g_hy = build_hy_gate(hy_s, z_thresh=1.0, slope=0.5)
        g_ma = build_ma_gate(close, ma_window=200, half_lev=0.6)
        g_nas = g_hy.values * g_ma.values
        g_nas = np.clip(g_nas, 0.2, 1.0)
        wn, wg, wb = apply_gates(wn_A, nas_gate=g_nas, bond_gate=None)
        nav = build_nav_strategy(
            close, lev, wn, wg, wb, dates_dt,
            gold_2x, bond_3x, sofr,
            nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002
        )

    else:
        raise ValueError(f"Unknown combo: {combo_name}")

    return nav


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_md_report(dsr_df, cv_df, final_judgments, combo_sharpes):
    """Markdownレポートを生成する"""
    lines = []
    lines.append(f"# P4 過学習確認レポート (Overfitting Check)")
    lines.append("")
    lines.append(f"作成日: {DATE_STR}")
    lines.append(f"最終更新日: {DATE_STR}")
    lines.append("")
    lines.append("## 概要")
    lines.append("")
    lines.append("P2 (38コンボ単独シグナル) + P3 (34コンボ組合せ) = 72試行に対して")
    lines.append("DSR (Deflated Sharpe Ratio) と 5-Fold Walk-Forward CV により過学習を確認。")
    lines.append("")
    lines.append("- IS:  1974-01-02 〜 2021-05-07")
    lines.append("- OOS: 2021-05-08 〜 2026-03-26 (≈ 1250日)")
    lines.append("- N_trials: 72 (P2: 38, P3: 34)")
    lines.append("- N_eff (保守的推定): 8 (信号間相関を考慮)")
    lines.append("")

    # DSR Table
    lines.append("## Part 1: DSR (Deflated Sharpe Ratio)")
    lines.append("")
    lines.append("### N_eff=8 での結果")
    lines.append("")
    lines.append("| コンボ | Sharpe_OOS | N_total | N_eff | E[maxSR] | PSR_deflated | 判定 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    dsr_8 = dsr_df[dsr_df['n_eff'] == 8].copy()
    for _, row in dsr_8.iterrows():
        verdict = row['dsr_pass']
        lines.append(f"| {row['combo_name']} | {row['sharpe_oos']:.3f} | "
                     f"{row['n_trials_total']} | {row['n_eff']} | "
                     f"{row['e_max_sr_annual']:.4f} | {row['psr_deflated']:.4f} | "
                     f"**{verdict}** |")
    lines.append("")

    # DSR Sensitivity Table
    lines.append("### N_eff 感度分析")
    lines.append("")
    combos_order = list(combo_sharpes.keys())
    n_eff_vals = sorted(dsr_df['n_eff'].unique())
    header = "| コンボ | " + " | ".join([f"N_eff={n}" for n in n_eff_vals]) + " |"
    separator = "|---|" + "|".join(["---:"] * len(n_eff_vals)) + "|"
    lines.append(header)
    lines.append(separator)
    for combo in combos_order:
        row_vals = []
        for n in n_eff_vals:
            sub = dsr_df[(dsr_df['combo_name'] == combo) & (dsr_df['n_eff'] == n)]
            if len(sub) > 0:
                psr = sub.iloc[0]['psr_deflated']
                row_vals.append(f"{psr:.3f}")
            else:
                row_vals.append("N/A")
        lines.append(f"| {combo} | " + " | ".join(row_vals) + " |")
    lines.append("")

    # CV Results Table
    lines.append("## Part 2: 5-Fold Walk-Forward CV")
    lines.append("")
    lines.append("### フォールド別 Sharpe")
    lines.append("")
    folds = sorted(cv_df['fold_id'].unique())
    combo_order = list(combo_sharpes.keys())
    header = "| コンボ | " + " | ".join(folds) + " | Median | Positive率 |"
    separator = "|---|" + "|".join(["---:"] * (len(folds) + 2)) + "|"
    lines.append(header)
    lines.append(separator)
    for combo in combo_order:
        sub = cv_df[cv_df['combo_name'] == combo]
        sharpes = []
        for f in folds:
            row = sub[sub['fold_id'] == f]
            if len(row) > 0 and not np.isnan(row.iloc[0]['sharpe']):
                sharpes.append(row.iloc[0]['sharpe'])
            else:
                sharpes.append(np.nan)
        sharpe_strs = [f"{s:.3f}" if not np.isnan(s) else "N/A" for s in sharpes]
        valid = [s for s in sharpes if not np.isnan(s)]
        median_s = np.median(valid) if valid else np.nan
        pos_rate = sum(1 for s in valid if s > 0) / len(folds)
        lines.append(f"| {combo} | " + " | ".join(sharpe_strs) +
                     f" | {median_s:.3f} | {pos_rate:.0%} |")
    lines.append("")

    lines.append("### フォールド別 CAGR")
    lines.append("")
    header = "| コンボ | " + " | ".join(folds) + " | Median |"
    separator = "|---|" + "|".join(["---:"] * (len(folds) + 1)) + "|"
    lines.append(header)
    lines.append(separator)
    for combo in combo_order:
        sub = cv_df[cv_df['combo_name'] == combo]
        cagrs = []
        for f in folds:
            row = sub[sub['fold_id'] == f]
            if len(row) > 0 and not np.isnan(row.iloc[0]['cagr']):
                cagrs.append(row.iloc[0]['cagr'])
            else:
                cagrs.append(np.nan)
        cagr_strs = [f"{c*100:.1f}%" if not np.isnan(c) else "N/A" for c in cagrs]
        valid = [c for c in cagrs if not np.isnan(c)]
        median_c = np.median(valid) if valid else np.nan
        lines.append(f"| {combo} | " + " | ".join(cagr_strs) +
                     f" | {median_c*100:.1f}% |")
    lines.append("")

    lines.append("### フォールド別 MaxDD")
    lines.append("")
    header = "| コンボ | " + " | ".join(folds) + " | Median |"
    separator = "|---|" + "|".join(["---:"] * (len(folds) + 1)) + "|"
    lines.append(header)
    lines.append(separator)
    for combo in combo_order:
        sub = cv_df[cv_df['combo_name'] == combo]
        mdds = []
        for f in folds:
            row = sub[sub['fold_id'] == f]
            if len(row) > 0 and not np.isnan(row.iloc[0]['maxdd']):
                mdds.append(row.iloc[0]['maxdd'])
            else:
                mdds.append(np.nan)
        mdd_strs = [f"{m*100:.1f}%" if not np.isnan(m) else "N/A" for m in mdds]
        valid = [m for m in mdds if not np.isnan(m)]
        median_m = np.median(valid) if valid else np.nan
        lines.append(f"| {combo} | " + " | ".join(mdd_strs) +
                     f" | {median_m*100:.1f}% |")
    lines.append("")

    # Final Judgment Table
    lines.append("## 最終判定")
    lines.append("")
    lines.append("### 判定基準")
    lines.append("")
    lines.append("| 指標 | 採用閾値 |")
    lines.append("|---|---|")
    lines.append("| DSR (N_eff=8) | ≥ 0.95 → ADOPT / 0.90-0.95 → GRAY / < 0.90 → REJECT |")
    lines.append("| CV Sharpe>0 フォールド比率 | ≥ 80% (5フォールド中4以上) |")
    lines.append("| CV median Sharpe | ≥ 0.5 |")
    lines.append("| CV median CAGR | ≥ 15% |")
    lines.append("")
    lines.append("### 最終判定表")
    lines.append("")
    lines.append("| コンボ | DSR_pass | DSR_PSR | CV_pos_rate | CV_med_Sharpe | CV_med_CAGR | 総合判定 |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for combo, jud in final_judgments.items():
        lines.append(
            f"| {combo} | {jud['dsr_pass']} | {jud['dsr_psr']:.3f} | "
            f"{jud['cv_pos_rate']:.0%} | {jud['cv_med_sharpe']:.3f} | "
            f"{jud['cv_med_cagr']*100:.1f}% | **{jud['final']}** |"
        )
    lines.append("")

    # Investment Summary
    lines.append("## 投資判断サマリー")
    lines.append("")
    adopt_list = [c for c, j in final_judgments.items() if j['final'] == 'ADOPT']
    gray_list = [c for c, j in final_judgments.items() if j['final'] == 'GRAY']
    reject_list = [c for c, j in final_judgments.items() if j['final'] == 'REJECT']

    if adopt_list:
        lines.append(
            f"ADOPT候補 ({', '.join(adopt_list)}) はDSR・CV双方の基準を満たし、"
            "統計的有意性が確認された。ただし、CAGR_OOS が1次採用基準 (≥ 20%) に未達のため、"
            "実運用への組み込みは追加検討が必要。"
        )
    elif gray_list:
        lines.append(
            f"GRAYゾーン候補 ({', '.join(gray_list)}) は一部の基準を満たすが、"
            "統計的有意性が不十分。過学習リスクを否定できず、実運用への採用は推奨しない。"
        )
    else:
        lines.append(
            "全候補がREJECT判定。72試行の多重検定補正後、統計的に有意なシグナルは確認されなかった。"
            "現行のDH Dyn [A] Baselineを維持し、新たなシグナル開発またはゲート設計の見直しを推奨する。"
        )
    lines.append("")
    lines.append("P2+P3の結果が示す通り、CAGR_OOS ≥ 20% の1次採用基準を満たすコンボは存在せず、"
                 "OOS期間 (2021-05-08〜2026-03-26) における市場環境 (高金利・CPI上昇) "
                 "がゲート戦略に不利に作用した可能性がある。")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("P4 過学習確認 (Overfitting Check)")
    print(f"実行日: {DATE_STR}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Data setup
    # -----------------------------------------------------------------------
    data = setup_data()

    # -----------------------------------------------------------------------
    # Target combos with their OOS Sharpe (from P2/P3 results)
    # -----------------------------------------------------------------------
    COMBO_SHARPES = {
        'Baseline':   0.697,   # P3 report baseline Sharpe_OOS (DH Dyn [A])
        'P01_Dyn×HY': 0.829,
        'P02_Dyn×CPI': 0.833,
        'P03_Dyn×MA':  0.798,
        'P05_HY×CPI':  0.667,
        'P06_HY×MA':   0.616,
    }

    # -----------------------------------------------------------------------
    # Part 1: DSR analysis
    # -----------------------------------------------------------------------
    print("\n[Part 1: DSR Analysis]")
    T_oos = 1250  # OOS観測日数

    dsr_df = run_dsr_analysis(COMBO_SHARPES, ALL_SHARPES, T_oos)

    print("\n  DSR結果 (N_eff=8):")
    print(f"  {'コンボ':20s} {'Sharpe_OOS':>12s} {'E[maxSR]':>10s} {'PSR':>8s} {'判定':>8s}")
    print("  " + "-" * 65)
    dsr_8 = dsr_df[dsr_df['n_eff'] == 8]
    for _, row in dsr_8.iterrows():
        print(f"  {row['combo_name']:20s} {row['sharpe_oos']:>12.3f} "
              f"{row['e_max_sr_annual']:>10.4f} {row['psr_deflated']:>8.4f} "
              f"{row['dsr_pass']:>8s}")

    # -----------------------------------------------------------------------
    # Part 2: Build NAVs and run CV
    # -----------------------------------------------------------------------
    print("\n[Part 2: Walk-Forward CV]")
    dates_dt = data['dates_dt']

    # Compute fold boundaries
    fold_boundaries = compute_cv_folds(None, dates_dt)

    cv_records = []
    final_judgments = {}

    for combo_name in COMBO_SHARPES.keys():
        print(f"\n  {combo_name}:")
        nav = build_combo_nav(combo_name, data)

        fold_sharpes = []
        fold_cagrs = []

        for fold_name, fold_start, fold_end in fold_boundaries:
            metrics = fold_metrics(
                nav,
                dates_dt,
                fold_start.strftime('%Y-%m-%d'),
                fold_end.strftime('%Y-%m-%d')
            )

            if metrics is None:
                print(f"    {fold_name}: データ不足 (スキップ)")
                cv_records.append({
                    'combo_name': combo_name,
                    'fold_id': fold_name,
                    'fold_start': fold_start.strftime('%Y-%m-%d'),
                    'fold_end': fold_end.strftime('%Y-%m-%d'),
                    'n_days': 0,
                    'cagr': np.nan,
                    'sharpe': np.nan,
                    'maxdd': np.nan,
                    'worst1y': np.nan,
                })
                fold_sharpes.append(np.nan)
                fold_cagrs.append(np.nan)
                continue

            print(f"    {fold_name} ({metrics['n_days']}日): "
                  f"Sharpe={metrics['sharpe']:.3f}, CAGR={metrics['cagr']*100:.1f}%, "
                  f"MaxDD={metrics['maxdd']*100:.1f}%")

            cv_records.append({
                'combo_name': combo_name,
                'fold_id': fold_name,
                'fold_start': fold_start.strftime('%Y-%m-%d'),
                'fold_end': fold_end.strftime('%Y-%m-%d'),
                'n_days': metrics['n_days'],
                'cagr': metrics['cagr'],
                'sharpe': metrics['sharpe'],
                'maxdd': metrics['maxdd'],
                'worst1y': metrics['worst1y'],
            })
            fold_sharpes.append(metrics['sharpe'])
            fold_cagrs.append(metrics['cagr'])

        # Compute CV summary
        valid_sharpes = [s for s in fold_sharpes if not np.isnan(s)]
        valid_cagrs = [c for c in fold_cagrs if not np.isnan(c)]
        n_total_folds = len(fold_boundaries)

        cv_med_sharpe = np.median(valid_sharpes) if valid_sharpes else np.nan
        cv_med_cagr = np.median(valid_cagrs) if valid_cagrs else np.nan
        cv_pos_rate = sum(1 for s in valid_sharpes if s > 0) / n_total_folds if n_total_folds > 0 else 0.0

        # DSR pass for this combo (N_eff=8)
        dsr_row = dsr_8[dsr_8['combo_name'] == combo_name]
        dsr_pass = dsr_row.iloc[0]['dsr_pass'] if len(dsr_row) > 0 else 'REJECT'
        dsr_psr = dsr_row.iloc[0]['psr_deflated'] if len(dsr_row) > 0 else 0.0

        # Final judgment
        cv_ok = (cv_pos_rate >= 0.80) and (not np.isnan(cv_med_sharpe) and cv_med_sharpe >= 0.5) and \
                (not np.isnan(cv_med_cagr) and cv_med_cagr >= 0.15)

        if dsr_pass == 'ADOPT' and cv_ok:
            final = 'ADOPT'
        elif dsr_pass == 'REJECT' and not cv_ok:
            final = 'REJECT'
        else:
            final = 'GRAY'

        final_judgments[combo_name] = {
            'dsr_pass': dsr_pass,
            'dsr_psr': dsr_psr,
            'cv_pos_rate': cv_pos_rate,
            'cv_med_sharpe': cv_med_sharpe if not np.isnan(cv_med_sharpe) else 0.0,
            'cv_med_cagr': cv_med_cagr if not np.isnan(cv_med_cagr) else 0.0,
            'final': final,
        }

    cv_df = pd.DataFrame(cv_records)

    # -----------------------------------------------------------------------
    # Print final judgments
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("最終判定:")
    print(f"  {'コンボ':20s} {'DSR':>6s} {'PSR':>8s} {'CV_pos':>8s} {'CV_SR':>8s} {'CV_CAGR':>9s} {'総合':>8s}")
    print("  " + "-" * 70)
    for combo, jud in final_judgments.items():
        print(f"  {combo:20s} {jud['dsr_pass']:>6s} {jud['dsr_psr']:>8.3f} "
              f"{jud['cv_pos_rate']:>8.0%} {jud['cv_med_sharpe']:>8.3f} "
              f"{jud['cv_med_cagr']*100:>8.1f}% {jud['final']:>8s}")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    print("\n[出力ファイル生成]")

    # 1. DSR CSV
    dsr_out = os.path.join(BASE, f'P4_DSR_RESULTS_{DATE_STR}.csv')
    dsr_df.to_csv(dsr_out, index=False)
    print(f"  保存: {dsr_out}")

    # 2. CV CSV
    cv_out = os.path.join(BASE, f'P4_CV_RESULTS_{DATE_STR}.csv')
    cv_df.to_csv(cv_out, index=False)
    print(f"  保存: {cv_out}")

    # 3. MD report
    md_content = generate_md_report(dsr_df, cv_df, final_judgments, COMBO_SHARPES)
    md_out = os.path.join(BASE, f'P4_OVERFITTING_CHECK_{DATE_STR}.md')
    with open(md_out, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  保存: {md_out}")

    print("\n完了!")
    return dsr_df, cv_df, final_judgments


if __name__ == '__main__':
    main()
