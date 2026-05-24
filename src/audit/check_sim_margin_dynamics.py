# -*- coding: utf-8 -*-
"""
check_sim_margin_dynamics.py (PoC 2.17)
========================================
証拠金維持率シミュレーション

5つの入金倍率×レバレッジシナリオで証拠金維持率を計算し、
強制清算リスクを評価する。

出力:
  audit_results/MARGIN_DYNAMICS_<DATE>.md
  audit_results/margin_dynamics.yaml
"""
import sys, os, types

# multitasking stub (must come BEFORE sys.path manipulation)
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

# Path resolution: src/audit/ -> src/ -> BASE
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, 'src'))

import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location(
    '_audit_strategy',
    pathlib.Path(__file__).parent / '_audit_strategy.py'
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_best_strategy_assets = _mod.build_best_strategy_assets

from cfd_leverage_backtest import OOS_START, DELAY

import numpy as np
import pandas as pd
from datetime import date

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TODAY = date.today().strftime('%Y%m%d')
TODAY_STR = date.today().strftime('%Y-%m-%d')
AUDIT_DIR = os.path.join(BASE, 'audit_results')
os.makedirs(AUDIT_DIR, exist_ok=True)

MARGIN_RATIO_LIQUIDATION = 1.0   # 100% — forced liquidation threshold
MARGIN_RATIO_WARNING     = 1.3   # 130% — warning zone threshold

MARGIN_SCENARIOS = [
    {'name': '(1) 超低資本 1x入金・L_s2実値', 'deposit_mult': 1.0, 'leverage_mode': 'actual'},
    {'name': '(2) 適切資本 2x入金・L_s2実値', 'deposit_mult': 2.0, 'leverage_mode': 'actual'},
    {'name': '(3) 余裕資本 3x入金・L_s2実値', 'deposit_mult': 3.0, 'leverage_mode': 'actual'},
    {'name': '(4) 適切資本 2x入金・L=5固定',  'deposit_mult': 2.0, 'leverage_mode': 'fixed5'},
    {'name': '(5) 適切資本 2x入金・L=7固定',  'deposit_mult': 2.0, 'leverage_mode': 'fixed7'},
]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_margin_ratio(nav_values: np.ndarray, deposit_mult: float,
                         L_eff: np.ndarray) -> np.ndarray:
    """
    margin_ratio_t = (NAV_t / NAV_0) * deposit_mult / (L_eff_t * 0.10)

    NAV_0 is the first element of nav_values (= 1.0 at inception).
    L_eff is the effective leverage array (same length as nav_values).
    """
    nav_0 = nav_values[0]
    margin_ratio = (nav_values / nav_0) * deposit_mult / (L_eff * 0.10)
    return margin_ratio


def run_scenario(sc: dict, assets: dict) -> dict:
    """Run one margin scenario and return result dict."""
    nav: pd.Series = assets['nav_baseline']
    nav_vals = nav.values.astype(float)
    dates_idx = assets['dates']

    deposit_mult    = sc['deposit_mult']
    leverage_mode   = sc['leverage_mode']

    if leverage_mode == 'actual':
        L_eff = assets['L_s2'].values.astype(float)
    elif leverage_mode == 'fixed5':
        L_eff = np.full(len(nav_vals), 5.0)
    elif leverage_mode == 'fixed7':
        L_eff = np.full(len(nav_vals), 7.0)
    else:
        raise ValueError(f'Unknown leverage_mode: {leverage_mode}')

    # Clip L_eff at floor to avoid division by zero (should not happen, but safety)
    L_eff = np.maximum(L_eff, 0.1)

    margin_ratio = compute_margin_ratio(nav_vals, deposit_mult, L_eff)

    # --- Overall stats ---
    forced_liq_events = int((margin_ratio < MARGIN_RATIO_LIQUIDATION).sum())
    warning_days      = int((margin_ratio < MARGIN_RATIO_WARNING).sum())
    min_idx           = int(np.argmin(margin_ratio))
    min_margin_ratio  = float(margin_ratio[min_idx])

    # Resolve date of minimum margin
    dates_arr = pd.to_datetime(dates_idx.values)
    min_margin_date = str(dates_arr[min_idx].date())

    passed = forced_liq_events == 0

    # --- IS period stats ---
    oos_ts = pd.Timestamp(OOS_START)
    is_mask  = dates_arr < oos_ts
    oos_mask = dates_arr >= oos_ts

    def _period_stats(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {'min_margin_ratio': float('nan'), 'min_margin_date': 'N/A',
                    'forced_liquidation_events': 0, 'warning_days': 0}
        mr_sub   = margin_ratio[mask]
        dates_sub = dates_arr[mask]
        idx_min  = int(np.argmin(mr_sub))
        return {
            'min_margin_ratio':        float(mr_sub[idx_min]),
            'min_margin_date':         str(dates_sub[idx_min].date()),
            'forced_liquidation_events': int((mr_sub < MARGIN_RATIO_LIQUIDATION).sum()),
            'warning_days':            int((mr_sub < MARGIN_RATIO_WARNING).sum()),
        }

    is_stats  = _period_stats(is_mask)
    oos_stats = _period_stats(oos_mask)

    # --- Worst 3-month window by minimum margin ---
    # Rolling 63-day window (≈ 3 months of trading days) — find window with lowest min
    window = 63
    n = len(margin_ratio)
    worst_window_min   = float('inf')
    worst_window_start = None
    worst_window_end   = None
    for i in range(n - window + 1):
        w_min = float(np.min(margin_ratio[i:i + window]))
        if w_min < worst_window_min:
            worst_window_min   = w_min
            worst_window_start = str(dates_arr[i].date())
            worst_window_end   = str(dates_arr[i + window - 1].date())

    # --- Correlation: L_s2 stats at low-margin days ---
    low_margin_mask = margin_ratio < 1.5
    if low_margin_mask.sum() > 0 and leverage_mode == 'actual':
        l_at_low_margin_mean   = float(L_eff[low_margin_mask].mean())
        l_at_low_margin_max    = float(L_eff[low_margin_mask].max())
        l_overall_mean         = float(L_eff.mean())
    else:
        l_at_low_margin_mean   = float('nan')
        l_at_low_margin_max    = float('nan')
        l_overall_mean         = float(L_eff.mean())

    return {
        'name':                     sc['name'],
        'deposit_mult':             deposit_mult,
        'leverage_mode':            leverage_mode,
        'forced_liquidation_events': forced_liq_events,
        'warning_days':             warning_days,
        'min_margin_ratio':         min_margin_ratio,
        'min_margin_date':          min_margin_date,
        'pass':                     passed,
        'is_stats':                 is_stats,
        'oos_stats':                oos_stats,
        'worst_3m_window': {
            'start':      worst_window_start,
            'end':        worst_window_end,
            'min_margin': worst_window_min,
        },
        'l_correlation': {
            'low_margin_days':          int(low_margin_mask.sum()),
            'l_mean_at_low_margin':     l_at_low_margin_mean,
            'l_max_at_low_margin':      l_at_low_margin_max,
            'l_overall_mean':           l_overall_mean,
        },
    }


# ---------------------------------------------------------------------------
# Minimum deposit recommendation
# ---------------------------------------------------------------------------
def find_min_deposit_mult(assets: dict, leverage_mode: str = 'actual',
                          search_range=(0.5, 10.0), steps: int = 100) -> float:
    """Find minimum deposit_mult for 0 forced liquidation events."""
    nav_vals = assets['nav_baseline'].values.astype(float)

    if leverage_mode == 'actual':
        L_eff = np.maximum(assets['L_s2'].values.astype(float), 0.1)
    elif leverage_mode == 'fixed5':
        L_eff = np.full(len(nav_vals), 5.0)
    elif leverage_mode == 'fixed7':
        L_eff = np.full(len(nav_vals), 7.0)
    else:
        L_eff = np.maximum(assets['L_s2'].values.astype(float), 0.1)

    lo, hi = search_range
    # Binary search
    for _ in range(50):
        mid = (lo + hi) / 2.0
        mr  = compute_margin_ratio(nav_vals, mid, L_eff)
        if (mr < MARGIN_RATIO_LIQUIDATION).sum() == 0:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < 0.001:
            break
    return round(hi, 3)


# ---------------------------------------------------------------------------
# Summary & risk level
# ---------------------------------------------------------------------------
def determine_risk_level(results: list) -> str:
    """Assess overall risk level based on scenario outcomes."""
    baseline = next((r for r in results if '(2)' in r['name']), None)
    sc3      = next((r for r in results if '(3)' in r['name']), None)

    if baseline and not baseline['pass']:
        return 'HIGH'
    if sc3 and not sc3['pass']:
        return 'MEDIUM'
    return 'LOW'


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------
def save_yaml(results: list, summary: dict) -> str:
    yaml_path = os.path.join(AUDIT_DIR, 'margin_dynamics.yaml')
    data = {
        'scenarios': [],
        'summary':   summary,
    }
    for r in results:
        sc_entry = {
            'name':                     r['name'],
            'deposit_mult':             r['deposit_mult'],
            'leverage_mode':            r['leverage_mode'],
            'forced_liquidation_events': r['forced_liquidation_events'],
            'warning_days':             r['warning_days'],
            'min_margin_ratio':         round(r['min_margin_ratio'], 4),
            'min_margin_date':          r['min_margin_date'],
            'pass':                     r['pass'],
        }
        data['scenarios'].append(sc_entry)

    try:
        import yaml
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False,
                           default_flow_style=False)
    except ImportError:
        import json
        yaml_path = yaml_path.replace('.yaml', '.json')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'Saved: {yaml_path}')
    return yaml_path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def generate_md(results: list, summary: dict, min_dm_actual: float) -> str:
    lines = []
    lines.append('# 2.17 証拠金維持率シミュレーション')
    lines.append('')
    lines.append(f'生成日: {TODAY_STR}')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 概要')
    lines.append('')
    lines.append('証拠金維持率 = (NAV_t / NAV_0) × 入金倍率 / (有効レバレッジ × 0.10)')
    lines.append('')
    lines.append('- **強制清算ライン**: 維持率 < 100%')
    lines.append('- **警告ゾーン**: 維持率 < 130%')
    lines.append(f'- **基準シナリオ**: {summary["baseline_scenario"]}')
    lines.append('')

    # Risk level callout
    risk = summary['risk_level']
    risk_icon = {'HIGH': '**CRITICAL**', 'MEDIUM': '**MEDIUM**', 'LOW': '**LOW**'}.get(risk, risk)
    lines.append(f'**総合リスク評価: {risk_icon}**')
    lines.append('')
    lines.append('---')
    lines.append('')

    # --- Main results table ---
    lines.append('## 1. シナリオ別結果（全期間）')
    lines.append('')
    lines.append('| シナリオ | 入金倍率 | レバレッジ | 強制清算イベント数 | 警告日数(<130%) | 最低維持率 | 最低維持率日 | 判定 |')
    lines.append('|:---------|--------:|----------:|-----------------:|---------------:|-----------:|:------------|:-----|')

    for r in results:
        lev_label = {
            'actual': 'L_s2実値',
            'fixed5': 'L=5固定',
            'fixed7': 'L=7固定',
        }.get(r['leverage_mode'], r['leverage_mode'])
        verdict = 'PASS' if r['pass'] else 'FAIL'
        lines.append(
            f"| {r['name']} | {r['deposit_mult']:.1f}x | {lev_label} "
            f"| {r['forced_liquidation_events']:,} "
            f"| {r['warning_days']:,} "
            f"| {r['min_margin_ratio']:.2%} "
            f"| {r['min_margin_date']} "
            f"| {verdict} |"
        )
    lines.append('')

    # --- Baseline scenario highlight ---
    baseline = next((r for r in results if '(2)' in r['name']), None)
    if baseline:
        lines.append('### 基準シナリオ (2) の詳細')
        lines.append('')
        if baseline['pass']:
            lines.append('基準シナリオ (2) は **PASS** — 強制清算イベントなし。')
        else:
            lines.append(f'> **CRITICAL: 基準シナリオ (2) に強制清算イベントが '
                         f'{baseline["forced_liquidation_events"]:,} 件発生。**')
            lines.append('> 入金額 2 倍では証拠金不足リスクが現実的に存在する。入金倍率の引き上げを推奨。')
        lines.append('')
        lines.append(f'- 警告日数 (<130%): **{baseline["warning_days"]:,} 日**')
        lines.append(f'- 最低維持率: **{baseline["min_margin_ratio"]:.2%}**  (date: {baseline["min_margin_date"]})')
        lines.append('')

    # --- IS / OOS breakdown ---
    lines.append('---')
    lines.append('')
    lines.append('## 2. IS / OOS 期間別内訳')
    lines.append('')
    lines.append(f'- **IS 期間**: 全開始日 〜 {OOS_START} の前日')
    lines.append(f'- **OOS 期間**: {OOS_START} 〜 最終日')
    lines.append('')
    lines.append('### 2-a. IS 期間')
    lines.append('')
    lines.append('| シナリオ | 強制清算 | 警告日数 | 最低維持率 | 最低維持率日 |')
    lines.append('|:---------|--------:|---------:|-----------:|:------------|')
    for r in results:
        is_s = r['is_stats']
        lines.append(
            f"| {r['name']} "
            f"| {is_s['forced_liquidation_events']:,} "
            f"| {is_s['warning_days']:,} "
            f"| {is_s['min_margin_ratio']:.2%} "
            f"| {is_s['min_margin_date']} |"
        )
    lines.append('')
    lines.append('### 2-b. OOS 期間')
    lines.append('')
    lines.append('| シナリオ | 強制清算 | 警告日数 | 最低維持率 | 最低維持率日 |')
    lines.append('|:---------|--------:|---------:|-----------:|:------------|')
    for r in results:
        oos_s = r['oos_stats']
        lines.append(
            f"| {r['name']} "
            f"| {oos_s['forced_liquidation_events']:,} "
            f"| {oos_s['warning_days']:,} "
            f"| {oos_s['min_margin_ratio']:.2%} "
            f"| {oos_s['min_margin_date']} |"
        )
    lines.append('')

    # --- Worst 3-month window ---
    lines.append('---')
    lines.append('')
    lines.append('## 3. 最悪3ヶ月ウィンドウ（証拠金維持率の最低期間）')
    lines.append('')
    lines.append('| シナリオ | ウィンドウ開始 | ウィンドウ終了 | 期間中最低維持率 |')
    lines.append('|:---------|:------------:|:------------:|-----------------:|')
    for r in results:
        w = r['worst_3m_window']
        lines.append(
            f"| {r['name']} "
            f"| {w['start']} "
            f"| {w['end']} "
            f"| {w['min_margin']:.2%} |"
        )
    lines.append('')

    # --- Leverage correlation ---
    lines.append('---')
    lines.append('')
    lines.append('## 4. 低証拠金日のレバレッジ相関 (L_s2実値シナリオのみ)')
    lines.append('')
    lines.append('低証拠金日 = margin_ratio < 150% の日')
    lines.append('')
    lines.append('| シナリオ | 低証拠金日数 | L_s2平均(低証拠金日) | L_s2最大(低証拠金日) | L_s2全体平均 |')
    lines.append('|:---------|------------:|---------------------:|---------------------:|-------------:|')
    for r in results:
        lc = r['l_correlation']
        if r['leverage_mode'] == 'actual':
            lm_mean = f"{lc['l_mean_at_low_margin']:.2f}" if not (isinstance(lc['l_mean_at_low_margin'], float) and np.isnan(lc['l_mean_at_low_margin'])) else 'N/A'
            lm_max  = f"{lc['l_max_at_low_margin']:.2f}"  if not (isinstance(lc['l_max_at_low_margin'], float)  and np.isnan(lc['l_max_at_low_margin']))  else 'N/A'
            lines.append(
                f"| {r['name']} "
                f"| {lc['low_margin_days']:,} "
                f"| {lm_mean} "
                f"| {lm_max} "
                f"| {lc['l_overall_mean']:.2f} |"
            )
    lines.append('')

    # --- Minimum deposit recommendation ---
    lines.append('---')
    lines.append('')
    lines.append('## 5. 最低入金倍率の推奨')
    lines.append('')
    lines.append('強制清算イベント = 0 を達成するための最低入金倍率（L_s2実値ベース）:')
    lines.append('')
    lines.append(f'**最低必要入金倍率: {min_dm_actual:.3f}x** (L_s2実値)')
    lines.append('')
    lines.append('| 推奨 | 入金倍率 | 根拠 |')
    lines.append('|:----|--------:|:----|')
    lines.append(f'| 最低限 | {min_dm_actual:.1f}x | 強制清算ゼロ ギリギリライン |')
    lines.append(f'| 推奨   | {max(min_dm_actual * 1.5, 2.0):.1f}x | 最低限の 1.5 倍バッファ |')
    lines.append(f'| 安全圏 | {max(min_dm_actual * 2.0, 3.0):.1f}x | 最低限の 2.0 倍バッファ |')
    lines.append('')
    lines.append(f'推奨値: **{summary["recommendation"]}**')
    lines.append('')

    # --- Summary ---
    lines.append('---')
    lines.append('')
    lines.append('## 6. 総合判定')
    lines.append('')
    pass_count = sum(1 for r in results if r['pass'])
    lines.append(f'- PASS シナリオ数: **{pass_count} / {len(results)}**')
    lines.append(f'- 基準シナリオ (2) 判定: **{"PASS" if summary["baseline_pass"] else "FAIL"}**')
    lines.append(f'- 総合リスク評価: **{summary["risk_level"]}**')
    lines.append('')

    if summary['risk_level'] == 'HIGH':
        lines.append('> **CRITICAL**: 適切資本 2x 入金でも強制清算リスクあり。')
        lines.append('> 入金倍率の大幅引き上げ、またはレバレッジ上限引き下げを強く推奨。')
    elif summary['risk_level'] == 'MEDIUM':
        lines.append('> **CAUTION**: 2x 入金は安全だが、3x 入金でも清算イベントが発生する期間がある。')
        lines.append('> バッファを持った入金管理を推奨。')
    else:
        lines.append('> **OK**: 2x 入金で強制清算リスクなし。余裕資本 3x では更に安全。')

    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 出力ファイル')
    lines.append('')
    lines.append(f'- `audit_results/MARGIN_DYNAMICS_{TODAY}.md`: 本レポート')
    lines.append('- `audit_results/margin_dynamics.yaml`: 構造化結果（YAML）')
    lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 60)
    print('PoC 2.17: 証拠金維持率シミュレーション')
    print('=' * 60)

    # 1. Load assets
    print('\n[1/4] ベスト戦略アセットをロード中...')
    assets = build_best_strategy_assets(force_rebuild=False)
    dates_arr = pd.to_datetime(assets['dates'].values)
    print(f'  期間: {dates_arr[0].date()} ~ {dates_arr[-1].date()}  ({len(dates_arr):,} days)')
    print(f'  L_s2: [{assets["L_s2"].min():.1f}, {assets["L_s2"].max():.1f}]  '
          f'median={assets["L_s2"].median():.1f}')
    print(f'  nav_baseline 最終値: {float(assets["nav_baseline"].iloc[-1]):.4f}')

    # 2. Run scenarios
    print('\n[2/4] 5シナリオを計算中...')
    results = []
    for sc in MARGIN_SCENARIOS:
        r = run_scenario(sc, assets)
        results.append(r)
        verdict = 'PASS' if r['pass'] else 'FAIL'
        print(f'  {sc["name"]:40s}  強制清算={r["forced_liquidation_events"]:4d}日  '
              f'最低維持率={r["min_margin_ratio"]:.2%}  [{verdict}]')

    # 3. Find minimum deposit_mult
    print('\n[3/4] 最低入金倍率を計算中 (L_s2実値)...')
    min_dm_actual = find_min_deposit_mult(assets, leverage_mode='actual')
    print(f'  最低必要入金倍率 (L_s2実値): {min_dm_actual:.3f}x')

    # 4. Build summary
    baseline = next((r for r in results if '(2)' in r['name']), None)
    risk_level = determine_risk_level(results)

    rec_mult = max(min_dm_actual * 1.5, 2.0)
    recommendation = (
        f'L_s2実値ベースの強制清算ゼロ最低倍率 {min_dm_actual:.2f}x に対し、'
        f'1.5 倍バッファの {rec_mult:.1f}x 以上の入金を推奨。'
        f'固定L=7使用時はさらに高い倍率が必要。'
    )

    summary = {
        'generated_date':      TODAY_STR,
        'baseline_scenario':   baseline['name'] if baseline else '(2)',
        'baseline_pass':       baseline['pass'] if baseline else False,
        'risk_level':          risk_level,
        'min_deposit_mult_actual_leverage': float(min_dm_actual),
        'recommendation':      recommendation,
        'pass_count':          sum(1 for r in results if r['pass']),
        'total_scenarios':     len(results),
    }

    # 5. Console summary table
    print('\n--- 結果サマリー ---')
    print(f'{"シナリオ":<42} {"強制清算":>8} {"警告日数":>8} {"最低維持率":>10} {"判定":>6}')
    print('-' * 80)
    for r in results:
        verdict = 'PASS' if r['pass'] else 'FAIL'
        print(f'{r["name"]:<42} {r["forced_liquidation_events"]:>8,} '
              f'{r["warning_days"]:>8,} {r["min_margin_ratio"]:>10.2%} {verdict:>6}')
    print(f'\n総合リスク評価: {risk_level}')

    # 6. Save YAML
    print('\n[4/4] ファイル出力中...')
    save_yaml(results, summary)

    # 7. Save MD
    md_content = generate_md(results, summary, min_dm_actual)
    md_path = os.path.join(AUDIT_DIR, f'MARGIN_DYNAMICS_{TODAY}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f'Saved: {md_path}')

    print('\nDone.')
    return results, summary


if __name__ == '__main__':
    main()
