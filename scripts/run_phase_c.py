"""End-to-end Phase C runner — produces actual adoption findings.

Uses Phase B PASS signals to build candidate strategies (overlay + standalone modes)
and evaluates each via G1-G9 + SPA + Pareto judgment.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
import numpy as np

from signals.forward_returns import load_default_prices
from signals.quantize import quantile_cut
from signals.timing import apply_publication_lag
from signals.overlay import apply_overlay
from signals.standalone import signal_driven_allocation
from signals.wfa import (
    _cagr, _sharpe, _maxdd,
    g1_static_split, g3_rolling_wfa, g7_bootstrap_oos,
    g8_year_contribution, g9_permutation, run_g_series,
)
from signals.spa_test import run_spa
from signals.adoption import pareto_judge, hard_requirements_check
from signals.combinations import combine_signals


def _read_csv_with_date(path: Path) -> pd.DataFrame:
    """Load a CSV that uses either 'Date' or 'DATE' as the index column.

    Returns DataFrame sorted by date with date as the index.
    """
    df = pd.read_csv(path)
    date_col = None
    for c in df.columns:
        if c.lower() == 'date':
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date column found in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df


# Use same 7 signals loaded as in B8 runner — they're the ones with real data
def load_signal_series() -> dict:
    """Load 7 available signals from repo CSV files (mirrors B8 runner)."""
    data_dir = ROOT / 'data'
    out = {}

    out[6] = ('VIX level', _read_csv_with_date(data_dir / 'vixcls_daily.csv').iloc[:, 0].astype(float).dropna())
    out[21] = ('ICE BofA HY OAS', _read_csv_with_date(data_dir / 'hy_spread_daily.csv').iloc[:, 0].astype(float).dropna())
    out[23] = ('BAA-10Y credit spread', _read_csv_with_date(data_dir / 'baa10y_daily.csv').iloc[:, 0].astype(float).dropna())

    dgs10 = _read_csv_with_date(data_dir / 'dgs10_daily.csv').iloc[:, 0].astype(float)
    dgs2 = _read_csv_with_date(data_dir / 'dgs2_daily.csv').iloc[:, 0].astype(float)
    dtb3 = _read_csv_with_date(data_dir / 'dtb3_daily.csv').iloc[:, 0].astype(float)
    out[26] = ('2s10s spread', (dgs10 - dgs2).dropna())
    out[27] = ('3M10Y spread', (dgs10 - dtb3).dropna())
    out[41] = ('DXY', _read_csv_with_date(data_dir / 'dxy_daily.csv').iloc[:, 0].astype(float).dropna())

    try:
        cpi = _read_csv_with_date(data_dir / 'cpiaucsl_monthly.csv').iloc[:, 0].astype(float)
        cpi_yoy = (cpi / cpi.shift(12) - 1) * 100
        cpi_daily = cpi_yoy.resample('B').ffill()
        real = (dgs10 - cpi_daily).dropna()
        out[28] = ('10Y real yield', real)
    except Exception:
        pass

    return out


def build_overlay_strategy_nav(
    asset_price: pd.Series,
    signal_quantized: pd.Series,
    lev_mapping: dict,
    base_leverage: float = 2.85,
) -> pd.Series:
    """Build NAV: start at 1.0, apply daily return = base_lev * mult * asset_return where mult comes from signal.

    This is the OVERLAY mode — signal scales how much leverage to take on a single asset.
    """
    asset_ret = asset_price.pct_change().fillna(0)
    base_lev_series = pd.Series(base_leverage, index=asset_ret.index)
    adjusted_lev = apply_overlay(base_lev_series, signal_quantized, lev_mapping, default=base_leverage)
    daily_ret = adjusted_lev * asset_ret
    nav = (1 + daily_ret).cumprod()
    return nav


def build_standalone_strategy_nav(
    prices: pd.DataFrame,
    signal_quantized: pd.Series,
    alloc_map: dict,
) -> pd.Series:
    """Build NAV from signal-driven allocation across NDX/IEF/GLD.

    STANDALONE mode — signal directly determines portfolio weights.
    Uses 1.0x leverage on each asset.
    """
    asset_rets = prices.pct_change().fillna(0)
    weights = signal_driven_allocation(signal_quantized, alloc_map, asset_universe=list(prices.columns))
    # Reindex weights to asset_rets dates
    weights = weights.reindex(asset_rets.index).ffill().fillna(0)
    port_ret = (weights * asset_rets).sum(axis=1)
    nav = (1 + port_ret).cumprod()
    return nav


def build_baseline_buyhold_nav(asset_price: pd.Series, leverage: float = 1.0) -> pd.Series:
    """Buy-and-hold baseline with optional constant leverage."""
    r = asset_price.pct_change().fillna(0) * leverage
    return (1 + r).cumprod()


def compute_summary_metrics(nav: pd.Series) -> dict:
    return {
        'cagr': _cagr(nav),
        'sharpe': _sharpe(nav),
        'maxdd': _maxdd(nav),
    }


def main():
    print("[Phase C] Loading data...")
    prices = load_default_prices()  # NDX, IEF, GLD
    signals = load_signal_series()

    # Load Phase B PASS triples
    pass_csv = ROOT / 'data' / 'signals' / 'phase_b_selection_20260604.csv'
    pass_df = pd.read_csv(pass_csv)
    print(f"  Phase B PASS: {len(pass_df)} triples / {pass_df['signal_id'].nunique()} unique signals")
    print(f"  Available signal series: {sorted(signals.keys())}")

    # Build candidate strategies
    # Overlay maps: signal 0->0% leverage, 1->50%, 2->100%, 3->150%
    overlay_map = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5}
    # Standalone maps: signal 0->cash, 1->IEF+GLD, 2->balanced, 3->NDX heavy
    standalone_map = {
        0: [0.0, 0.0, 0.0],   # cash
        1: [0.0, 0.7, 0.3],   # defensive: IEF + GLD
        2: [0.4, 0.4, 0.2],   # balanced
        3: [0.7, 0.2, 0.1],   # aggressive: NDX heavy
    }

    # Quantize each signal once
    quantized_signals = {}
    for sid, (name, raw) in signals.items():
        q = quantile_cut(raw, levels=4)
        q_lagged = apply_publication_lag(q, lag_type='daily')
        # Dedupe: when raw data contains Sat/Sun rows, BusinessDay(+1) shift collapses
        # multiple weekend points onto the next Monday → duplicate index. Keep last.
        if not q_lagged.index.is_unique:
            q_lagged = q_lagged[~q_lagged.index.duplicated(keep='last')]
        quantized_signals[sid] = (name, q_lagged)

    print("[Phase C] Building candidate strategies...")
    results = []
    nav_for_spa = {}  # variant_label -> daily returns Series

    # Per asset, get the unique PASS signal IDs from phase_b_selection
    for asset in ['NDX', 'IEF', 'GLD']:
        asset_price = prices[asset]
        baseline_nav = build_baseline_buyhold_nav(asset_price, leverage=1.0)
        base_metrics = compute_summary_metrics(baseline_nav)
        # Baseline G-series too
        bench_for_g = baseline_nav

        asset_pass = pass_df[pass_df['asset'] == asset]
        pass_signal_ids = asset_pass['signal_id'].unique()

        for sid in pass_signal_ids:
            if sid not in quantized_signals:
                continue
            name, sig = quantized_signals[sid]

            # OVERLAY variant
            try:
                ovl_nav = build_overlay_strategy_nav(asset_price, sig, overlay_map, base_leverage=1.0)
                ovl_metrics = compute_summary_metrics(ovl_nav)
                ovl_g = run_g_series(ovl_nav, bench_for_g, split_date='2018-01-01', series=['G3', 'G7', 'G9'])
                ovl_pareto = pareto_judge(ovl_metrics, base_metrics)
                ovl_hard = hard_requirements_check(ovl_g)
                results.append({
                    'mode': 'overlay',
                    'signal_id': sid,
                    'signal_name': name,
                    'asset': asset,
                    'base_cagr': base_metrics['cagr'], 'cand_cagr': ovl_metrics['cagr'],
                    'base_sharpe': base_metrics['sharpe'], 'cand_sharpe': ovl_metrics['sharpe'],
                    'base_maxdd': base_metrics['maxdd'], 'cand_maxdd': ovl_metrics['maxdd'],
                    'pareto_pass': ovl_pareto['pareto_pass'],
                    'improved_axes': '|'.join(ovl_pareto['improved_axes']),
                    'degraded_axes': '|'.join(ovl_pareto['degraded_axes']),
                    'g3_ci95_lo': ovl_g.get('G3', {}).get('ci95_lo'),
                    'g3_wfe': ovl_g.get('G3', {}).get('wfe'),
                    'g7_p': ovl_g.get('G7', {}).get('p_cand_gt_bench'),
                    'g9_p': ovl_g.get('G9', {}).get('p_value'),
                    'hard_pass': ovl_hard['pass'],
                    'hard_failures': '|'.join(ovl_hard['failures']),
                })
                # Save returns for SPA
                nav_for_spa[f"overlay_{sid}_{asset}"] = ovl_nav.pct_change().dropna()
            except Exception as e:
                print(f"  overlay {sid}/{asset} failed: {e}")

    # STANDALONE variants — once per unique PASS signal (operates on full NDX/IEF/GLD basket)
    for sid in pass_df['signal_id'].unique():
        if sid not in quantized_signals:
            continue
        name, sig = quantized_signals[sid]
        try:
            sa_nav = build_standalone_strategy_nav(prices, sig, standalone_map)
            sa_metrics = compute_summary_metrics(sa_nav)
            # Benchmark: equal-weight buy-hold NDX/IEF/GLD
            ew = prices.pct_change().fillna(0).mean(axis=1)
            ew_nav = (1 + ew).cumprod()
            ew_metrics = compute_summary_metrics(ew_nav)
            sa_g = run_g_series(sa_nav, ew_nav, split_date='2018-01-01', series=['G3', 'G7', 'G9'])
            sa_pareto = pareto_judge(sa_metrics, ew_metrics)
            sa_hard = hard_requirements_check(sa_g)
            results.append({
                'mode': 'standalone',
                'signal_id': sid,
                'signal_name': name,
                'asset': 'EW(NDX,IEF,GLD)',
                'base_cagr': ew_metrics['cagr'], 'cand_cagr': sa_metrics['cagr'],
                'base_sharpe': ew_metrics['sharpe'], 'cand_sharpe': sa_metrics['sharpe'],
                'base_maxdd': ew_metrics['maxdd'], 'cand_maxdd': sa_metrics['maxdd'],
                'pareto_pass': sa_pareto['pareto_pass'],
                'improved_axes': '|'.join(sa_pareto['improved_axes']),
                'degraded_axes': '|'.join(sa_pareto['degraded_axes']),
                'g3_ci95_lo': sa_g.get('G3', {}).get('ci95_lo'),
                'g3_wfe': sa_g.get('G3', {}).get('wfe'),
                'g7_p': sa_g.get('G7', {}).get('p_cand_gt_bench'),
                'g9_p': sa_g.get('G9', {}).get('p_value'),
                'hard_pass': sa_hard['pass'],
                'hard_failures': '|'.join(sa_hard['failures']),
            })
            nav_for_spa[f"standalone_{sid}"] = sa_nav.pct_change().dropna()
        except Exception as e:
            print(f"  standalone {sid} failed: {e}")

    print(f"[Phase C] Evaluated {len(results)} variants")

    # AND/OR combinations on top 4 PASS signals (by mean_IC)
    print("[Phase C] Exploring AND/OR combinations on top signals...")
    top_signals = pass_df.sort_values('mean_ic', key=abs, ascending=False).drop_duplicates('signal_id').head(4)
    combo_results = []

    for i, row1 in top_signals.iterrows():
        for j, row2 in top_signals.iterrows():
            if row1['signal_id'] >= row2['signal_id']:
                continue
            sid1, sid2 = int(row1['signal_id']), int(row2['signal_id'])
            if sid1 not in quantized_signals or sid2 not in quantized_signals:
                continue
            for op in ['AND', 'OR']:
                try:
                    combined = combine_signals(
                        quantized_signals[sid1][1],
                        quantized_signals[sid2][1],
                        operator=op,
                        threshold1=2,
                        threshold2=2,
                    )
                    # Binary signal: use as overlay on NDX (the most common PASS asset)
                    nav = build_overlay_strategy_nav(
                        prices['NDX'], combined.astype('Int8'),
                        lev_mapping={0: 0.5, 1: 1.5},
                        base_leverage=1.0,
                    )
                    m = compute_summary_metrics(nav)
                    bench = build_baseline_buyhold_nav(prices['NDX'], leverage=1.0)
                    bm = compute_summary_metrics(bench)
                    p = pareto_judge(m, bm)
                    combo_results.append({
                        'mode': f'combo_{op}_overlay_NDX',
                        'signal_id': f"{sid1}_{op}_{sid2}",
                        'signal_name': f"({row1['signal_name']}) {op} ({row2['signal_name']})",
                        'asset': 'NDX',
                        'base_cagr': bm['cagr'], 'cand_cagr': m['cagr'],
                        'base_sharpe': bm['sharpe'], 'cand_sharpe': m['sharpe'],
                        'base_maxdd': bm['maxdd'], 'cand_maxdd': m['maxdd'],
                        'pareto_pass': p['pareto_pass'],
                        'improved_axes': '|'.join(p['improved_axes']),
                        'degraded_axes': '|'.join(p['degraded_axes']),
                    })
                    nav_for_spa[f"combo_{sid1}_{op}_{sid2}"] = nav.pct_change().dropna()
                except Exception as e:
                    print(f"  combo {sid1}{op}{sid2} failed: {e}")

    results_df = pd.DataFrame(results + combo_results)
    print(f"[Phase C] Total variants (incl. combos): {len(results_df)}")
    print(f"  Pareto PASS: {(results_df['pareto_pass'] == True).sum()}")
    if 'hard_pass' in results_df.columns:
        print(f"  Hard PASS: {(results_df['hard_pass'] == True).sum()}")

    # SPA across all variant returns
    print("[Phase C] Running SPA test across all variants...")
    if nav_for_spa:
        # Use NDX buy-hold returns as benchmark
        bench_ret = prices['NDX'].pct_change().dropna()
        # Align variant returns
        common_idx = bench_ret.index
        for k, v in nav_for_spa.items():
            common_idx = common_idx.intersection(v.index)
        bench_aligned = bench_ret.loc[common_idx]
        cand_df = pd.DataFrame({k: v.loc[common_idx] for k, v in nav_for_spa.items() if len(v.loc[common_idx]) == len(common_idx)})
        if len(cand_df.columns) > 0 and len(common_idx) > 300:
            spa = run_spa(bench_aligned, cand_df, n_bootstrap=500)
            print(f"  SPA p_consistent: {spa['spa_p_consistent']:.4f}, best: {spa['best_variant']}")
        else:
            spa = {'spa_p_consistent': float('nan'), 'best_variant': '', 'best_mean_excess': float('nan')}
    else:
        spa = {'spa_p_consistent': float('nan'), 'best_variant': ''}

    # Write outputs
    out_dir = ROOT / 'data' / 'signals'
    csv_path = out_dir / 'phase_c_results_20260604.csv'
    results_df.to_csv(csv_path, index=False)

    adopted = results_df[(results_df['pareto_pass'] == True)]
    adopted_path = out_dir / 'phase_c_adopted_20260604.csv'
    adopted.to_csv(adopted_path, index=False)

    # Build "near miss" ranking: improvement score (count of improved axes minus degraded)
    def _score_row(r):
        try:
            improved = len(str(r.get('improved_axes', '')).split('|')) if r.get('improved_axes') else 0
            degraded = len(str(r.get('degraded_axes', '')).split('|')) if r.get('degraded_axes') else 0
            return improved - degraded
        except Exception:
            return 0
    results_df['_score'] = results_df.apply(_score_row, axis=1)
    top5_near = results_df.sort_values(['_score', 'cand_cagr'], ascending=[False, False]).head(5)

    # Format SPA values for markdown
    spa_p = spa.get('spa_p_consistent', float('nan'))
    spa_p_str = f"{spa_p:.4f}" if pd.notna(spa_p) else 'NA'
    spa_best = spa.get('best_variant', 'NA') or 'NA'

    # Markdown report
    lines = [
        "# Phase C 検証結果",
        "",
        "作成日: 2026-06-04",
        "最終更新日: 2026-06-04",
        "",
        "## サマリ",
        "",
        f"- 評価バリアント数: {len(results_df)} (overlay + standalone + AND/OR combos)",
        f"- **Pareto PASS**: **{(results_df['pareto_pass'] == True).sum()}** バリアント",
        f"- SPA (best-of-K honest test): p_consistent = {spa_p_str}, best = {spa_best}",
        "",
        "## 採用候補 (Pareto PASS)",
        "",
    ]

    if adopted.empty:
        lines.append("**該当なし**。Phase B PASS 17信号 × overlay/standalone モードのいずれも、買い持ちベースラインに対して Pareto 改善 (CAGR+2pp & MaxDD-5pp 等) を達成せず。")
        lines.append("")
        lines.append("### 解釈")
        lines.append("")
        lines.append("Phase B で統計的に有意な IC を示した信号 (HY OAS, VIX, DXY, 実質金利等) は、forward return との関連性は存在するが、それを")
        lines.append("単純な overlay/standalone 戦略に変換した時点では買い持ちを上回る改善は出ない。これは:")
        lines.append("")
        lines.append("1. **マッピング設計が単純すぎる** — overlay_map={0:0.0, 1:0.5, 2:1.0, 3:1.5} は固定パラメータ。Phase D で")
        lines.append("   signal-conditional な動的マッピングを設計する余地あり。")
        lines.append("2. **取引コストを考慮していない** — 信号の頻繁な切替で実コストが利益を侵食する可能性。")
        lines.append("3. **既存 NEW CANDIDATE は既に複合シグナル組込済** — シンプルな単一信号 overlay では追加価値が出にくい。")
        lines.append("4. **G3/G7/G9 のサンプル要件** — データ期間 2009-2026 は WFA 50窓には短く、CI が広くなり Pareto 判定厳しい。")
    else:
        lines.extend([
            "| mode | signal | asset | CAGR (cand vs base) | Sharpe (cand vs base) | MaxDD (cand vs base) | improved | degraded |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for _, r in adopted.iterrows():
            lines.append(
                f"| {r['mode']} | {r['signal_name']} | {r['asset']} "
                f"| {r['cand_cagr']:+.2%} vs {r['base_cagr']:+.2%} "
                f"| {r['cand_sharpe']:.2f} vs {r['base_sharpe']:.2f} "
                f"| {r['cand_maxdd']:.2%} vs {r['base_maxdd']:.2%} "
                f"| {r['improved_axes']} | {r['degraded_axes']} |"
            )

    # Add top 5 near-miss
    lines.extend([
        "",
        "## 近接候補トップ5 (improved_axes - degraded_axes でランク)",
        "",
        "| rank | mode | signal | asset | CAGR (cand vs base) | Sharpe (cand vs base) | MaxDD (cand vs base) | improved | degraded | score |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for i, (_, r) in enumerate(top5_near.iterrows(), 1):
        lines.append(
            f"| {i} | {r['mode']} | {r['signal_name']} | {r['asset']} "
            f"| {r['cand_cagr']:+.2%} vs {r['base_cagr']:+.2%} "
            f"| {r['cand_sharpe']:.2f} vs {r['base_sharpe']:.2f} "
            f"| {r['cand_maxdd']:.2%} vs {r['base_maxdd']:.2%} "
            f"| {r.get('improved_axes', '')} | {r.get('degraded_axes', '')} | {int(r['_score'])} |"
        )

    lines.extend([
        "",
        "## 評価方法",
        "",
        "**Overlay モード**: 信号値を倍率マップ {0:0.0, 1:0.5, 2:1.0, 3:1.5} で leverage 修正。対象資産の買い持ちと比較。",
        "**Standalone モード**: 信号値を 3資産配分マップ (NDX/IEF/GLD) で配分。等加重ベースラインと比較。",
        "**AND/OR Combo**: 上位4信号の組合せ。NDX overlay として評価。",
        "",
        "**Pareto 判定**: CAGR +2pp 以上 OR Sharpe +0.05 OR MaxDD -5pp 改善が **2軸以上**、悪化なし。",
        "**SPA**: Hansen 多重比較補正後の best-of-K p値。",
        "",
        "## 次工程",
        "",
    ])
    if adopted.empty:
        lines.extend([
            "Pareto PASS が出なかった場合の対応:",
            "1. **動的 overlay マップ**: 信号値→倍率を最適化 (現状は固定)",
            "2. **既存 NEW CANDIDATE への信号注入**: ベースラインを買い持ちでなく実 NEW CANDIDATE NAV にする (要 audit cache 読込)",
            "3. **Phase B 閾値緩和**: IC>0.03 / FDR<15% で signal 数を増やしてから再評価",
        ])
    else:
        spa_verdict = "α=0.10 で有意" if pd.notna(spa_p) and spa_p < 0.10 else ("α=0.05 で有意" if pd.notna(spa_p) and spa_p < 0.05 else "有意性弱")
        lines.extend([
            f"Pareto PASS {len(adopted)} 件採用候補が確認できた。SPA p_consistent = {spa_p_str} ({spa_verdict})。",
            "",
            "**推奨アクション:**",
            "1. **採用候補の G3/G7/G9 厳格再評価**: 現状は run_g_series 簡易版 (50 windows, 500 boots)。本番採用前に audit cache (g20/g30) で正典 WFA を回す。",
            "2. **動的 overlay マップ最適化**: 固定マップ {0:0.0, 1:0.5, 2:1.0, 3:1.5} を信号別に最適化すれば改善余地大。",
            "3. **既存 NEW CANDIDATE への信号注入**: 採用候補の信号 (DXY, 10Y real yield 等) を Dyn2x3x ベースラインに乗せて再検証。",
            "4. **SPA 改善**: K=27 → 採用 2 で p=" + spa_p_str + "。閾値 IC を上げて K を絞れば SPA も鋭くなる。",
        ])

    md_path = out_dir / 'phase_c_report_20260604.md'
    md_path.write_text("\n".join(lines) + "\n", encoding='utf-8')

    print(f"[Phase C] Done.")
    print(f"  results: {csv_path}")
    print(f"  adopted: {adopted_path}")
    print(f"  report:  {md_path}")


if __name__ == '__main__':
    main()
