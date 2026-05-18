"""
P5 ブロックブートストラップ・ストレステスト
=============================================
作成日: 2026-05-18

目的:
  P4でGRAY判定を受けたDyn系3コンボ (P01/P02/P03) に対して
  Stationary Block Bootstrap (Politis-Romano 1994) により
  サンプル感度を評価し、ADOPT / GRAY / REJECT を確定する。

手法:
  - Stationary Block Bootstrap (ブロック平均長 L=21日)
  - 反復数 B=2000
  - 対象期間: OOS (2021-05-08〜2026-03-26, ≈1250日)
  - Paired Bootstrap vs Baseline (同一インデックス系列)
  - シード: 42
"""

import sys
import os
import types
import time

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

# P4から再利用
from p4_overfitting_check import setup_data, build_combo_nav

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATE_STR = '2026-05-18'

COMBO_NAMES = [
    'Baseline',
    'P01_Dyn×HY',
    'P02_Dyn×CPI',
    'P03_Dyn×MA',
    'P05_HY×CPI',
    'P06_HY×MA',
]

P4_STATUS = {
    'Baseline':    'REJECT',
    'P01_Dyn×HY':  'GRAY',
    'P02_Dyn×CPI': 'GRAY',
    'P03_Dyn×MA':  'GRAY',
    'P05_HY×CPI':  'REJECT',
    'P06_HY×MA':   'REJECT',
}

OBS_SHARPE = {
    'Baseline':    0.697,
    'P01_Dyn×HY':  0.829,
    'P02_Dyn×CPI': 0.833,
    'P03_Dyn×MA':  0.798,
    'P05_HY×CPI':  0.667,
    'P06_HY×MA':   0.616,
}

N_BOOT = 2000
BLOCK_LEN = 21.0
SEED = 42
OOS_START = pd.Timestamp('2021-05-08')

# ---------------------------------------------------------------------------
# Stationary Block Bootstrap (Politis-Romano 1994)
# ---------------------------------------------------------------------------

def stationary_bootstrap_indices(T: int, L: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano Stationary Block Bootstrap のインデックス生成。"""
    p = 1.0 / L
    idx = np.empty(T, dtype=np.int64)
    idx[0] = rng.integers(0, T)
    u = rng.random(T)
    for t in range(1, T):
        if u[t] < p:
            idx[t] = rng.integers(0, T)        # 新ブロック開始
        else:
            idx[t] = (idx[t - 1] + 1) % T      # 既ブロック継続 (circular wrap)
    return idx


# ---------------------------------------------------------------------------
# Bootstrap for a single combo
# ---------------------------------------------------------------------------

def bootstrap_combo(daily_returns, n_boot=N_BOOT, block_len=BLOCK_LEN, seed=SEED):
    """1コンボのブートストラップを実行し、(n_boot × metrics) DataFrame を返す"""
    T = len(daily_returns)
    rng = np.random.default_rng(seed)
    records = []
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(T, block_len, rng)
        r_b = daily_returns[idx]
        nav = np.cumprod(1.0 + r_b)
        # メトリクス
        yrs = T / 252.0
        cagr = nav[-1] ** (1.0 / yrs) - 1.0
        sharpe = (r_b.mean() * 252.0) / (r_b.std() * np.sqrt(252.0) + 1e-10)
        peak = np.maximum.accumulate(nav)
        maxdd = float(np.min(nav / peak - 1.0))
        # Worst1Y
        if len(nav) >= 252:
            nav_s = pd.Series(nav)
            r1y = (nav_s / nav_s.shift(252)) - 1.0
            worst1y = float(r1y.min())
        else:
            worst1y = np.nan
        calmar = cagr / max(abs(maxdd), 1e-10)
        records.append({
            'iter': b, 'cagr': cagr, 'sharpe': sharpe,
            'maxdd': maxdd, 'worst1y': worst1y, 'calmar': calmar
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Paired bootstrap vs Baseline
# ---------------------------------------------------------------------------

def paired_bootstrap_vs_baseline(combo_returns, baseline_returns, n_boot=N_BOOT,
                                  block_len=BLOCK_LEN, seed=SEED):
    """同一インデックス系列で両系列をリサンプリングし ΔSharpe, ΔCAGR 分布を返す"""
    T = len(combo_returns)
    assert len(baseline_returns) == T
    rng = np.random.default_rng(seed)
    records = []
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(T, block_len, rng)
        r_c = combo_returns[idx]
        r_b = baseline_returns[idx]
        nav_c = np.cumprod(1.0 + r_c)
        nav_b = np.cumprod(1.0 + r_b)
        yrs = T / 252.0
        cagr_c = nav_c[-1] ** (1.0 / yrs) - 1.0
        cagr_b = nav_b[-1] ** (1.0 / yrs) - 1.0
        sr_c = (r_c.mean() * 252.0) / (r_c.std() * np.sqrt(252.0) + 1e-10)
        sr_b = (r_b.mean() * 252.0) / (r_b.std() * np.sqrt(252.0) + 1e-10)
        records.append({
            'iter': b,
            'delta_sharpe': sr_c - sr_b,
            'delta_cagr': cagr_c - cagr_b,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Judgment functions
# ---------------------------------------------------------------------------

def judge_combo(summary, p4_status):
    """P5総合判定 (p4_statusを考慮)。

    Args:
        summary: compute_summary + compute_criteria の結果辞書 (c8_pass含む)
        p4_status: P4での判定

    Returns:
        'ROBUST', 'MARGINAL', 'FRAGILE'
    """
    c8 = summary.get('c8_pass', False)
    c2 = summary['cagr_median'] >= 0.15
    c4 = summary['sharpe_median'] >= 0.5
    c5 = summary['pct_cagr_above_15'] >= 0.50
    c6 = summary['pct_sharpe_above_05'] >= 0.60

    abs_pass = sum([c2, c4, c5, c6])

    if c8 and abs_pass >= 3:
        return 'ROBUST'
    elif c8 and abs_pass < 3:
        return 'MARGINAL'
    elif (not c8) and (c5 or c6):
        return 'MARGINAL'
    else:
        return 'FRAGILE'


def final_verdict(p5_verdict, p4_status):
    """最終判定: p4_statusを継承してFINAL判定を決める"""
    if p4_status == 'GRAY':
        if p5_verdict == 'ROBUST':
            return 'ADOPT'
        elif p5_verdict == 'MARGINAL':
            return 'GRAY'
        else:
            return 'REJECT'
    elif p4_status == 'REJECT':
        return 'REJECT'   # 維持
    else:
        return p5_verdict


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(boot_df, paired_df, combo_name):
    """ブートストラップ結果から要約統計を計算する"""
    cagr = boot_df['cagr'].values
    sharpe = boot_df['sharpe'].values
    maxdd = boot_df['maxdd'].values

    n_boot = len(boot_df)

    summary = {
        'combo_name': combo_name,
        'block_len': BLOCK_LEN,
        'seed': SEED,
        'n_boot': n_boot,
        # CAGR
        'cagr_median': float(np.median(cagr)),
        'cagr_p05': float(np.percentile(cagr, 5)),
        'cagr_p95': float(np.percentile(cagr, 95)),
        # Sharpe
        'sharpe_median': float(np.median(sharpe)),
        'sharpe_p05': float(np.percentile(sharpe, 5)),
        'sharpe_p95': float(np.percentile(sharpe, 95)),
        # MaxDD
        'maxdd_median': float(np.median(maxdd)),
        'maxdd_p05': float(np.percentile(maxdd, 5)),
        'maxdd_p95': float(np.percentile(maxdd, 95)),
        # Percentage above thresholds
        'pct_cagr_above_15': float(np.mean(cagr >= 0.15)),
        'pct_sharpe_above_05': float(np.mean(sharpe >= 0.5)),
        'pct_maxdd_above_neg50': float(np.mean(maxdd >= -0.50)),
    }

    # Paired bootstrap metrics
    if paired_df is not None and len(paired_df) > 0:
        ds = paired_df['delta_sharpe'].values
        summary['delta_sharpe_p05'] = float(np.percentile(ds, 5))
        summary['delta_sharpe_median'] = float(np.median(ds))
        summary['pval_delta_sharpe'] = float(np.mean(ds <= 0))
    else:
        summary['delta_sharpe_p05'] = np.nan
        summary['delta_sharpe_median'] = np.nan
        summary['pval_delta_sharpe'] = np.nan

    summary['obs_sharpe_oos'] = OBS_SHARPE.get(combo_name, np.nan)

    return summary


def compute_criteria(summary):
    """9基準のpass/failを計算する"""
    c1 = summary['cagr_p05'] >= 0.10
    c2 = summary['cagr_median'] >= 0.15
    c3 = summary['sharpe_p05'] >= 0.3
    c4 = summary['sharpe_median'] >= 0.5
    c5 = summary['pct_cagr_above_15'] >= 0.50
    c6 = summary['pct_sharpe_above_05'] >= 0.60
    c7 = summary['maxdd_p95'] >= -0.60  # 95%ile (worst-tail) >= -60%
    c8 = (not np.isnan(summary['delta_sharpe_p05'])) and (summary['delta_sharpe_p05'] > 0)
    c9 = (not np.isnan(summary['pval_delta_sharpe'])) and (summary['pval_delta_sharpe'] < 0.10)
    return {
        'c1_pass': bool(c1), 'c2_pass': bool(c2), 'c3_pass': bool(c3),
        'c4_pass': bool(c4), 'c5_pass': bool(c5), 'c6_pass': bool(c6),
        'c7_pass': bool(c7), 'c8_pass': bool(c8), 'c9_pass': bool(c9),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_md_report(summary_df):
    """Markdownレポートを生成する"""
    lines = []
    lines.append("# P5 ブロックブートストラップ・ストレステスト")
    lines.append("")
    lines.append(f"作成日: {DATE_STR}")
    lines.append(f"最終更新日: {DATE_STR}")
    lines.append("")

    lines.append("## 概要")
    lines.append("")
    lines.append("P4でGRAY判定を受けたDyn系3コンボ (P01/P02/P03) に対して、")
    lines.append("Stationary Block Bootstrap により OOS期間のサンプル感度を評価。")
    lines.append("ADOPT / GRAY / REJECT を確定する。")
    lines.append("")
    lines.append("- OOS期間: 2021-05-08 〜 2026-03-26 (≈1250日)")
    lines.append("- P4 REJECT コンボ (Baseline/P05/P06) はREJECT維持を確認")
    lines.append("")

    lines.append("## 手法")
    lines.append("")
    lines.append("| 項目 | 設定 |")
    lines.append("|---|---|")
    lines.append("| 手法 | Stationary Block Bootstrap (Politis-Romano 1994) |")
    lines.append(f"| ブロック平均長 | L = {int(BLOCK_LEN)}日 (月次) |")
    lines.append(f"| 反復数 | B = {N_BOOT} |")
    lines.append("| 対象期間 | OOS (2021-05-08〜2026-03-26) |")
    lines.append(f"| シード | {SEED} |")
    lines.append("| Paired Bootstrap | 同一インデックス系列でBaseline比較 |")
    lines.append("")

    lines.append("## 判定基準")
    lines.append("")
    lines.append("| 基準 | 閾値 |")
    lines.append("|---|---|")
    lines.append("| C1: CAGR 5%ile | ≥ 10% |")
    lines.append("| C2: CAGR median | ≥ 15% |")
    lines.append("| C3: Sharpe 5%ile | ≥ 0.3 |")
    lines.append("| C4: Sharpe median | ≥ 0.5 |")
    lines.append("| C5: pct_above (CAGR ≥ 15%) | ≥ 50% |")
    lines.append("| C6: pct_above (Sharpe ≥ 0.5) | ≥ 60% |")
    lines.append("| C7: MaxDD 95%ile | ≥ -60% |")
    lines.append("| C8: ΔSharpe vs Baseline 5%ile | > 0 |")
    lines.append("| C9: p_value (Δ ≤ 0) | < 0.10 |")
    lines.append("")

    lines.append("## 結果テーブル — 絶対基準")
    lines.append("")
    lines.append("| コンボ | CAGR_med | CAGR_p05 | SR_med | SR_p05 | MaxDD_p95 | "
                 "C1 | C2 | C3 | C4 | C5 | C6 | C7 |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for _, row in summary_df.iterrows():
        def ok(v): return "✓" if v else "✗"
        lines.append(
            f"| {row['combo_name']} "
            f"| {row['cagr_median']*100:.1f}% "
            f"| {row['cagr_p05']*100:.1f}% "
            f"| {row['sharpe_median']:.3f} "
            f"| {row['sharpe_p05']:.3f} "
            f"| {row['maxdd_p95']*100:.1f}% "
            f"| {ok(row['c1_pass'])} "
            f"| {ok(row['c2_pass'])} "
            f"| {ok(row['c3_pass'])} "
            f"| {ok(row['c4_pass'])} "
            f"| {ok(row['c5_pass'])} "
            f"| {ok(row['c6_pass'])} "
            f"| {ok(row['c7_pass'])} |"
        )
    lines.append("")

    lines.append("## 結果テーブル — 相対基準 (Baseline比)")
    lines.append("")
    lines.append("| コンボ | ΔSR_p05 | ΔSR_med | p_val(Δ≤0) | C8 | C9 |")
    lines.append("|---|---:|---:|---:|:---:|:---:|")
    for _, row in summary_df.iterrows():
        def ok(v): return "✓" if v else "✗"
        dsr_p05 = row['delta_sharpe_p05']
        dsr_med = row['delta_sharpe_median']
        pval = row['pval_delta_sharpe']
        dsr_p05_str = f"{dsr_p05:.3f}" if not (isinstance(dsr_p05, float) and np.isnan(dsr_p05)) else "N/A"
        dsr_med_str = f"{dsr_med:.3f}" if not (isinstance(dsr_med, float) and np.isnan(dsr_med)) else "N/A"
        pval_str = f"{pval:.3f}" if not (isinstance(pval, float) and np.isnan(pval)) else "N/A"
        lines.append(
            f"| {row['combo_name']} "
            f"| {dsr_p05_str} "
            f"| {dsr_med_str} "
            f"| {pval_str} "
            f"| {ok(row['c8_pass'])} "
            f"| {ok(row['c9_pass'])} |"
        )
    lines.append("")

    lines.append("## 最終判定")
    lines.append("")
    lines.append("| コンボ | P4_status | P5_verdict | 最終判定 |")
    lines.append("|---|---|---|---|")
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['combo_name']} "
            f"| {row['p4_status']} "
            f"| {row['p5_verdict']} "
            f"| **{row['final_verdict']}** |"
        )
    lines.append("")

    lines.append("## 投資判断サマリー")
    lines.append("")
    adopt_list = summary_df[summary_df['final_verdict'] == 'ADOPT']['combo_name'].tolist()
    gray_list  = summary_df[summary_df['final_verdict'] == 'GRAY']['combo_name'].tolist()
    reject_list = summary_df[summary_df['final_verdict'] == 'REJECT']['combo_name'].tolist()

    if adopt_list:
        lines.append(
            f"**ADOPT確定候補**: {', '.join(adopt_list)}"
        )
        lines.append("")
        lines.append(
            "ブロックブートストラップによりサンプル感度が高く、"
            "Baseline比でのSharpe優位性が5%ile水準で確認された。"
            "P4のGRAY判定をADOPTに格上げする。"
        )
    if gray_list:
        lines.append(
            f"**GRAY維持候補**: {', '.join(gray_list)}"
        )
        lines.append("")
        lines.append(
            "P5でMARGINAL判定。絶対水準は一定の安定性を示すが、"
            "Baseline比優位性が統計的に十分でなく、実運用採用は慎重に判断すること。"
        )
    if reject_list:
        lines.append(
            f"**REJECT確定候補**: {', '.join(reject_list)}"
        )
        lines.append("")
        lines.append(
            "P4 REJECT維持またはP5でFRAGILE。"
            "ブートストラップ分布が採用基準を満たさず、現行DH Dyn [A] Baselineを維持する。"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("P5 ブロックブートストラップ・ストレステスト")
    print(f"実行日: {DATE_STR}  /  B={N_BOOT}, L={int(BLOCK_LEN)}, seed={SEED}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Data setup
    # ------------------------------------------------------------------
    data = setup_data()
    dates_dt = data['dates_dt']
    oos_mask = dates_dt >= OOS_START

    print(f"\nOOS期間: {OOS_START.strftime('%Y-%m-%d')} 〜 {dates_dt[oos_mask].max().strftime('%Y-%m-%d')}")
    print(f"OOS取引日数: {oos_mask.sum()}")

    # ------------------------------------------------------------------
    # Extract OOS daily returns for each combo
    # ------------------------------------------------------------------
    print("\n[OOSリターン抽出]")
    daily_returns_dict = {}
    for combo_name in COMBO_NAMES:
        nav = build_combo_nav(combo_name, data)
        nav_oos = nav[oos_mask].values
        r_oos = pd.Series(nav_oos).pct_change().dropna().values
        daily_returns_dict[combo_name] = r_oos
        print(f"  {combo_name}: T={len(r_oos)}日")

    baseline_returns = daily_returns_dict['Baseline']

    # ------------------------------------------------------------------
    # Run bootstrap for each combo
    # ------------------------------------------------------------------
    print("\n[ブートストラップ実行]")
    raw_records = []
    summary_records = []

    for combo_name in COMBO_NAMES:
        print(f"\n  --- {combo_name} ---")
        dr = daily_returns_dict[combo_name]

        # Individual bootstrap
        t0 = time.time()
        boot_df = bootstrap_combo(dr, n_boot=N_BOOT, block_len=BLOCK_LEN, seed=SEED)
        elapsed_ind = time.time() - t0
        print(f"  bootstrap_combo: {elapsed_ind:.1f}秒")

        # Paired bootstrap vs Baseline
        if combo_name == 'Baseline':
            paired_df = None
        else:
            t0 = time.time()
            paired_df = paired_bootstrap_vs_baseline(
                dr, baseline_returns, n_boot=N_BOOT, block_len=BLOCK_LEN, seed=SEED
            )
            elapsed_pair = time.time() - t0
            print(f"  paired_bootstrap: {elapsed_pair:.1f}秒")

        # Summary
        summary = compute_summary(boot_df, paired_df, combo_name)
        criteria = compute_criteria(summary)
        summary.update(criteria)

        # Paired delta columns for raw records
        if paired_df is not None:
            delta_sharpe_col = paired_df['delta_sharpe'].values
            delta_cagr_col   = paired_df['delta_cagr'].values
        else:
            delta_sharpe_col = np.full(N_BOOT, np.nan)
            delta_cagr_col   = np.full(N_BOOT, np.nan)

        # Accumulate raw records
        for i, row in enumerate(boot_df.itertuples(index=False)):
            raw_records.append({
                'combo_name': combo_name,
                'block_len': BLOCK_LEN,
                'seed': SEED,
                'iter': int(row.iter),
                'cagr': row.cagr,
                'sharpe': row.sharpe,
                'maxdd': row.maxdd,
                'worst1y': row.worst1y,
                'calmar': row.calmar,
                'delta_sharpe_vs_baseline': delta_sharpe_col[i],
                'delta_cagr_vs_baseline': delta_cagr_col[i],
            })

        # P5 verdict
        p5_v = judge_combo(summary, P4_STATUS[combo_name])
        fv = final_verdict(p5_v, P4_STATUS[combo_name])
        summary['p5_verdict'] = p5_v
        summary['p4_status'] = P4_STATUS[combo_name]
        summary['final_verdict'] = fv

        summary_records.append(summary)

        # Print stats
        print(f"  Sharpe  : median={summary['sharpe_median']:.3f}, "
              f"p05={summary['sharpe_p05']:.3f}, p95={summary['sharpe_p95']:.3f}")
        print(f"  CAGR    : median={summary['cagr_median']*100:.1f}%, "
              f"p05={summary['cagr_p05']*100:.1f}%, p95={summary['cagr_p95']*100:.1f}%")
        print(f"  MaxDD   : median={summary['maxdd_median']*100:.1f}%, "
              f"p95(worst)={summary['maxdd_p05']*100:.1f}%")

        # Criteria
        c_labels = ['C1','C2','C3','C4','C5','C6','C7','C8','C9']
        c_vals = [criteria[f'c{i}_pass'] for i in range(1, 10)]
        pass_str = " | ".join([f"{l}:{'OK' if v else '--'}" for l, v in zip(c_labels, c_vals)])
        print(f"  基準    : {pass_str}")
        print(f"  >> P5_verdict={p5_v}  /  final_verdict={fv}")

    # ------------------------------------------------------------------
    # Build DataFrames
    # ------------------------------------------------------------------
    raw_df = pd.DataFrame(raw_records)
    summary_df = pd.DataFrame(summary_records)

    # Reorder summary columns
    summary_cols = [
        'combo_name', 'block_len', 'seed', 'n_boot',
        'cagr_median', 'cagr_p05', 'cagr_p95',
        'sharpe_median', 'sharpe_p05', 'sharpe_p95',
        'maxdd_median', 'maxdd_p05', 'maxdd_p95',
        'pct_cagr_above_15', 'pct_sharpe_above_05', 'pct_maxdd_above_neg50',
        'delta_sharpe_p05', 'delta_sharpe_median', 'pval_delta_sharpe',
        'obs_sharpe_oos',
        'c1_pass', 'c2_pass', 'c3_pass', 'c4_pass', 'c5_pass',
        'c6_pass', 'c7_pass', 'c8_pass', 'c9_pass',
        'p5_verdict', 'p4_status', 'final_verdict',
    ]
    summary_df = summary_df[summary_cols]

    # ------------------------------------------------------------------
    # Final judgment table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("最終判定テーブル:")
    print(f"  {'コンボ':20s} {'P4':>7s} {'P5':>9s} {'FINAL':>8s}")
    print("  " + "-" * 50)
    for _, row in summary_df.iterrows():
        print(f"  {row['combo_name']:20s} {row['p4_status']:>7s} "
              f"{row['p5_verdict']:>9s} {row['final_verdict']:>8s}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print("\n[出力ファイル保存]")

    # 1. RAW CSV
    raw_out = os.path.join(BASE, f'P5_BOOTSTRAP_RAW_{DATE_STR}.csv')
    raw_df.to_csv(raw_out, index=False)
    print(f"  保存: {raw_out}  ({len(raw_df)}行)")

    # 2. SUMMARY CSV
    sum_out = os.path.join(BASE, f'P5_BOOTSTRAP_SUMMARY_{DATE_STR}.csv')
    summary_df.to_csv(sum_out, index=False)
    print(f"  保存: {sum_out}  ({len(summary_df)}行)")

    # 3. MD report
    md_content = generate_md_report(summary_df)
    md_out = os.path.join(BASE, f'P5_BOOTSTRAP_STRESS_{DATE_STR}.md')
    with open(md_out, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  保存: {md_out}")

    print("\n完了!")
    return raw_df, summary_df


if __name__ == '__main__':
    main()
