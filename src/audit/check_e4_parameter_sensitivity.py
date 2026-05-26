"""
check_e4_parameter_sensitivity.py
E4戦略 パラメータ感度分析（ワンファクター法）
================================================
中心値 (k_lo=0.1, k_hi=0.8, vz_thr=0.7, N_lt2=750, target_vol=0.8, k_vz=0.3)
から各パラメータを±20%振ったときの Sharpe_OOS / CAGR_OOS 変動幅を計測する。
「ベストパラメータが孤立した峰か、広い台地か」を確認。

出力:
  audit_results/E4_PARAM_SENSITIVITY_20260526.md
  audit_results/e4_param_sensitivity.yaml
  audit_results/e4_param_sensitivity.csv
"""

import sys, os, types

# ── multitasking stub（sys.path 操作より前に定義）─────────────────────────────
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

# ── Path resolution: src/audit/ → src/ → project root ────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, 'src'))

import importlib.util
import pathlib
import datetime
import csv
import numpy as np
import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ── Dynamic import: _audit_strategy (sibling file) ───────────────────────────
_spec = importlib.util.spec_from_file_location(
    '_audit_strategy',
    pathlib.Path(__file__).parent / '_audit_strategy.py',
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_e4_strategy_assets      = _mod.build_e4_strategy_assets
build_e4_assets_with_override = _mod.build_e4_assets_with_override

# ── Standard imports ─────────────────────────────────────────────────────────
from cfd_leverage_backtest import calc_7metrics, OOS_START
from compute_cfd_worst10y import nav_to_annual, rolling_nY_cagr

# ── パラメータグリッド定義 ───────────────────────────────────────────────────
CENTER_PARAMS = dict(
    k_lo=0.1, k_hi=0.8, vz_thr=0.7, N_lt2=750,
    target_vol=0.8, k_vz=0.3,
)

SENSITIVITY_GRIDS = {
    'k_lo':       [0.08, 0.09, 0.10, 0.11, 0.12],
    'k_hi':       [0.64, 0.72, 0.80, 0.88, 0.96],
    'vz_thr':     [0.56, 0.63, 0.70, 0.77, 0.84],
    'N_lt2':      [600, 675, 750, 825, 900],
    'target_vol': [0.64, 0.72, 0.80, 0.88, 0.96],
    'k_vz':       [0.24, 0.27, 0.30, 0.33, 0.36],
}

# ロバスト判定基準
ROBUST_CAGR_RANGE  = 0.02   # ±20% 範囲で CAGR_OOS 変動幅 ≤ 2pp
ROBUST_SHARPE_RANGE = 0.05  # ±20% 範囲で Sharpe_OOS 変動幅 ≤ 0.05
FRAGILE_DROP_THRESH = 0.05  # 片側で 5pp 以上低下 → FRAGILE 候補


def compute_oos_metrics(nav: pd.Series, dates: pd.Series,
                        n_tr: int, n_years: float) -> dict:
    """OOS期間のメトリクスを計算して返す。"""
    base = calc_7metrics(nav, dates)
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, n=10)
    worst10y_star = float(np.min(r10)) if len(r10) >= 1 else np.nan
    return {
        'CAGR_OOS':      base.get('CAGR_OOS', np.nan),
        'Sharpe_OOS':    base.get('Sharpe_OOS', np.nan),
        'MaxDD_FULL':    base.get('MaxDD_FULL', np.nan),
        'Worst10Y_star': worst10y_star,
        'Trades_yr':     n_tr / n_years,
    }


def _fmt_pct(v):
    """None / nan → 'N/A'、それ以外 +xx.xx%"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v * 100:+.2f}%'


def _fmt_sharpe(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:+.3f}'


def _fmt_delta(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v * 100:+.2f}pp'


def judge_sensitivity(param_name: str, rows: list, center_cagr: float) -> dict:
    """5点の CAGR_OOS / Sharpe_OOS から感度判定を行う。

    Returns:
        dict with keys: cagr_min, cagr_max, cagr_range, sharpe_min, sharpe_max,
                        sharpe_range, verdict ('ROBUST' | 'FRAGILE' | 'MIXED')
    """
    cagr_vals   = [r['metrics']['CAGR_OOS']   for r in rows]
    sharpe_vals = [r['metrics']['Sharpe_OOS'] for r in rows]

    valid_cagr   = [v for v in cagr_vals   if not np.isnan(v)]
    valid_sharpe = [v for v in sharpe_vals if not np.isnan(v)]

    cagr_min = min(valid_cagr)   if valid_cagr   else np.nan
    cagr_max = max(valid_cagr)   if valid_cagr   else np.nan
    sharpe_min = min(valid_sharpe) if valid_sharpe else np.nan
    sharpe_max = max(valid_sharpe) if valid_sharpe else np.nan

    cagr_range   = cagr_max   - cagr_min   if not (np.isnan(cagr_min)   or np.isnan(cagr_max))   else np.nan
    sharpe_range = sharpe_max - sharpe_min if not (np.isnan(sharpe_min) or np.isnan(sharpe_max)) else np.nan

    # 判定ロジック
    # ROBUST: cagr_range ≤ 2pp AND sharpe_range ≤ 0.05
    # FRAGILE: 中心値が局所最大 AND 片側で 5pp 以上低下
    # それ以外: MIXED
    if not np.isnan(cagr_range) and not np.isnan(sharpe_range):
        is_robust = (cagr_range <= ROBUST_CAGR_RANGE) and (sharpe_range <= ROBUST_SHARPE_RANGE)
        # 中心値が局所最大かつ片側5pp低下チェック
        center_idx = len(rows) // 2  # 5点中の中心 = index 2
        center_val = cagr_vals[center_idx] if not np.isnan(cagr_vals[center_idx]) else center_cagr
        edge_drop = max(center_val - cagr_min, 0.0)
        is_fragile = (center_val == cagr_max) and (edge_drop >= FRAGILE_DROP_THRESH)

        if is_robust:
            verdict = 'ROBUST'
        elif is_fragile:
            verdict = 'FRAGILE'
        else:
            verdict = 'MIXED'
    else:
        verdict = 'MIXED'

    return dict(
        cagr_min=cagr_min, cagr_max=cagr_max, cagr_range=cagr_range,
        sharpe_min=sharpe_min, sharpe_max=sharpe_max, sharpe_range=sharpe_range,
        verdict=verdict,
    )


def build_param_section_md(param_name: str, rows: list,
                            center_value, center_metrics: dict,
                            sensitivity: dict) -> list:
    """1パラメータ分の MD セクションを生成する。"""
    lines = []
    lines.append(f'## {param_name} Sensitivity (±20%)')
    lines.append('')
    lines.append('| 値 | CAGR_OOS | Sharpe_OOS | MaxDD | Worst10Y★ | Δ (center比) | 判定 |')
    lines.append('|---|---|---|---|---|---|---|')

    for r in rows:
        m = r['metrics']
        val = r['value']
        is_center = abs(float(val) - float(center_value)) < 1e-9

        cagr_str   = _fmt_pct(m['CAGR_OOS'])
        sharpe_str = _fmt_sharpe(m['Sharpe_OOS'])
        maxdd_str  = _fmt_pct(m['MaxDD_FULL'])
        w10_str    = _fmt_pct(m['Worst10Y_star'])

        if is_center:
            delta_str = 'center'
            judge_str = ''
        else:
            delta_v = m['CAGR_OOS'] - center_metrics['CAGR_OOS']
            delta_str = _fmt_delta(delta_v) if not np.isnan(delta_v) else 'N/A'
            # 個別判定: center比CAGR変動が2pp以内ならOK
            if not np.isnan(delta_v):
                judge_str = 'OK' if abs(delta_v) <= ROBUST_CAGR_RANGE else 'NG'
            else:
                judge_str = ''

        if is_center:
            row_str = (f'| **{val}（中心）** | **{cagr_str}** | **{sharpe_str}** | '
                       f'{maxdd_str} | {w10_str} | {delta_str} | {judge_str} |')
        else:
            row_str = (f'| {val} | {cagr_str} | {sharpe_str} | '
                       f'{maxdd_str} | {w10_str} | {delta_str} | {judge_str} |')
        lines.append(row_str)

    # 感度判定行
    cagr_range_str = (
        f'[{_fmt_pct(sensitivity["cagr_min"])}, {_fmt_pct(sensitivity["cagr_max"])}]'
        if not np.isnan(sensitivity['cagr_range']) else 'N/A'
    )
    lines.append(
        f'| **感度判定** | | | | | **範囲: {cagr_range_str}** | **{sensitivity["verdict"]}** |'
    )
    lines.append('')
    return lines


def build_markdown(all_results: dict, center_metrics: dict,
                   summary: list, date_str: str) -> str:
    lines = []
    lines.append('# E4 Parameter Sensitivity Analysis')
    lines.append('')
    lines.append(f'作成日: {date_str}')
    lines.append(
        '中心値: k_lo=0.1, k_hi=0.8, vz_thr=0.7, N_lt2=750, target_vol=0.8, k_vz=0.3'
    )
    lines.append(
        '判定基準: ROBUST = ±20%範囲でCAGR_OOS変動幅 ≤ 2pp ∧ Sharpe変動 ≤ 0.05'
    )
    lines.append('')

    # 各パラメータセクション
    for param_name, rows in all_results.items():
        center_value = CENTER_PARAMS[param_name]
        sens = next(s for s in summary if s['param'] == param_name)
        lines += build_param_section_md(
            param_name, rows, center_value, center_metrics, sens
        )

    # 総合判定サマリー
    lines.append('## 総合判定サマリー')
    lines.append('')
    lines.append('| パラメータ | 中心 CAGR_OOS | ±20% CAGR範囲 | ±20% Sharpe範囲 | 判定 |')
    lines.append('|---|---|---|---|---|')
    for s in summary:
        cagr_range_str = (
            f'[{_fmt_pct(s["cagr_min"])}, {_fmt_pct(s["cagr_max"])}]'
            if not np.isnan(s['cagr_range']) else 'N/A'
        )
        sharpe_range_str = (
            f'[{_fmt_sharpe(s["sharpe_min"])}, {_fmt_sharpe(s["sharpe_max"])}]'
            if not np.isnan(s['sharpe_range']) else 'N/A'
        )
        lines.append(
            f'| {s["param"]} | {_fmt_pct(center_metrics["CAGR_OOS"])} | '
            f'{cagr_range_str} | {sharpe_range_str} | {s["verdict"]} |'
        )
    lines.append('')

    robust_count  = sum(1 for s in summary if s['verdict'] == 'ROBUST')
    fragile_count = sum(1 for s in summary if s['verdict'] == 'FRAGILE')
    total = len(summary)

    lines.append(
        'ROBUST基準: CAGR_OOS変動幅 ≤ 2pp ∧ Sharpe変動幅 ≤ 0.05'
    )
    lines.append(
        'FRAGILE基準: 中心値が局所最大 ∧ 片側で 5pp 以上低下'
    )
    lines.append(f'ROBUST数: {robust_count}/{total}')
    lines.append(f'FRAGILE数: {fragile_count}/{total}')

    if robust_count == total:
        overall = 'ROBUST'
    elif fragile_count > 0:
        overall = 'FRAGILE'
    else:
        overall = 'MIXED'
    lines.append(f'**全体判定: {overall}**')
    lines.append('')

    return '\n'.join(lines)


def build_yaml_data(all_results: dict, center_metrics: dict,
                    summary: list, date_str: str) -> dict:
    data = {
        'generated':    date_str,
        'strategy':     'S2_VZGated + LT2-N750 + E4 Regime k_lt',
        'center_params': {k: float(v) for k, v in CENTER_PARAMS.items()},
        'center_metrics': {
            k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
            for k, v in center_metrics.items()
        },
        'robust_criterion': {
            'CAGR_OOS_range_pp': ROBUST_CAGR_RANGE * 100,
            'Sharpe_range':      ROBUST_SHARPE_RANGE,
        },
        'parameters': [],
    }
    for s in summary:
        pdata = {
            'param':        s['param'],
            'verdict':      s['verdict'],
            'cagr_min':     (None if np.isnan(s['cagr_min'])   else float(s['cagr_min'])),
            'cagr_max':     (None if np.isnan(s['cagr_max'])   else float(s['cagr_max'])),
            'cagr_range':   (None if np.isnan(s['cagr_range']) else float(s['cagr_range'])),
            'sharpe_min':   (None if np.isnan(s['sharpe_min'])   else float(s['sharpe_min'])),
            'sharpe_max':   (None if np.isnan(s['sharpe_max'])   else float(s['sharpe_max'])),
            'sharpe_range': (None if np.isnan(s['sharpe_range']) else float(s['sharpe_range'])),
            'values': [],
        }
        for r in all_results[s['param']]:
            m = r['metrics']
            pdata['values'].append({
                'value':       float(r['value']),
                'CAGR_OOS':    (None if np.isnan(m['CAGR_OOS'])   else float(m['CAGR_OOS'])),
                'Sharpe_OOS':  (None if np.isnan(m['Sharpe_OOS']) else float(m['Sharpe_OOS'])),
                'MaxDD_FULL':  (None if np.isnan(m['MaxDD_FULL'])  else float(m['MaxDD_FULL'])),
                'Worst10Y_star': (None if np.isnan(m['Worst10Y_star']) else float(m['Worst10Y_star'])),
                'Trades_yr':   (None if np.isnan(m['Trades_yr'])  else float(m['Trades_yr'])),
            })
        data['parameters'].append(pdata)

    robust_count  = sum(1 for s in summary if s['verdict'] == 'ROBUST')
    fragile_count = sum(1 for s in summary if s['verdict'] == 'FRAGILE')
    total = len(summary)
    if robust_count == total:
        overall = 'ROBUST'
    elif fragile_count > 0:
        overall = 'FRAGILE'
    else:
        overall = 'MIXED'
    data['overall_verdict'] = overall
    data['robust_count']    = robust_count
    data['fragile_count']   = fragile_count
    return data


def write_csv(all_results: dict, csv_path: str):
    """パラメータ名 / 値 / 全メトリクス を CSV に書き出す。"""
    fieldnames = [
        'param', 'value', 'is_center',
        'CAGR_OOS', 'Sharpe_OOS', 'MaxDD_FULL', 'Worst10Y_star', 'Trades_yr',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for param_name, rows in all_results.items():
            center_value = CENTER_PARAMS[param_name]
            for r in rows:
                m = r['metrics']
                is_center = abs(float(r['value']) - float(center_value)) < 1e-9
                writer.writerow({
                    'param':         param_name,
                    'value':         r['value'],
                    'is_center':     is_center,
                    'CAGR_OOS':      '' if np.isnan(m['CAGR_OOS'])    else f"{m['CAGR_OOS']:.6f}",
                    'Sharpe_OOS':    '' if np.isnan(m['Sharpe_OOS'])  else f"{m['Sharpe_OOS']:.6f}",
                    'MaxDD_FULL':    '' if np.isnan(m['MaxDD_FULL'])   else f"{m['MaxDD_FULL']:.6f}",
                    'Worst10Y_star': '' if np.isnan(m['Worst10Y_star']) else f"{m['Worst10Y_star']:.6f}",
                    'Trades_yr':     '' if np.isnan(m['Trades_yr'])   else f"{m['Trades_yr']:.2f}",
                })


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    date_str  = datetime.date.today().isoformat()
    audit_dir = os.path.join(BASE, 'audit_results')
    os.makedirs(audit_dir, exist_ok=True)

    print('=' * 70)
    print('E4 パラメータ感度分析（ワンファクター法）')
    print('=' * 70)
    print(f'  BASE     = {BASE}')
    print(f'  出力先   = {audit_dir}')
    print(f'  中心値   = {CENTER_PARAMS}')
    print()

    # ── STEP 1: E4 アセット構築 / キャッシュ読み込み ──────────────────────────
    print('[STEP 1] E4戦略アセット構築 / キャッシュ読み込み ...')
    base_assets = build_e4_strategy_assets(force_rebuild=False)
    dates   = base_assets['dates']
    n_tr    = base_assets['n_tr']
    n_years = base_assets['n_years']
    print(f'  dates    : {dates.iloc[0].date()} -> {dates.iloc[-1].date()}')
    print(f'  n_tr     : {n_tr}  ({n_tr/n_years:.1f}/yr)')
    print(f'  e4_params: {base_assets["e4_params"]}')
    print()

    # ── STEP 2: 中心値メトリクス計算（キャッシュから nav_e4 を使う）──────────
    print('[STEP 2] 中心値メトリクス計算 ...')
    center_nav = base_assets['nav_e4']
    center_metrics = compute_oos_metrics(center_nav, dates, n_tr, n_years)
    print(f'  CAGR_OOS    = {_fmt_pct(center_metrics["CAGR_OOS"])}')
    print(f'  Sharpe_OOS  = {_fmt_sharpe(center_metrics["Sharpe_OOS"])}')
    print(f'  MaxDD_FULL  = {_fmt_pct(center_metrics["MaxDD_FULL"])}')
    print(f'  Worst10Y★  = {_fmt_pct(center_metrics["Worst10Y_star"])}')
    print(f'  Trades/yr   = {center_metrics["Trades_yr"]:.1f}')
    print()

    # ── STEP 3: ワンファクター感度計算 ─────────────────────────────────────────
    print('[STEP 3] ワンファクター感度計算 (6パラメータ × 5点 = 30 configs) ...')
    all_results = {}  # {param_name: [{'value': v, 'metrics': {...}}, ...]}
    summary     = []

    for param_name, grid in SENSITIVITY_GRIDS.items():
        center_value = CENTER_PARAMS[param_name]
        rows = []
        print(f'\n  --- {param_name} (center={center_value}) ---')

        for value in grid:
            is_center = abs(float(value) - float(center_value)) < 1e-9

            if is_center:
                # 中心値はキャッシュ済みの値を使い回す
                metrics = dict(center_metrics)
                tag = '（中心・キャッシュ）'
            else:
                # オーバーライドして再計算
                try:
                    assets_var = build_e4_assets_with_override(
                        param_name=param_name,
                        value=value,
                        base_assets=base_assets,
                    )
                    nav_var = assets_var['nav_e4']
                    metrics = compute_oos_metrics(nav_var, dates, n_tr, n_years)
                    tag = ''
                except Exception as e:
                    print(f'    [ERROR] {param_name}={value}: {e}')
                    metrics = {
                        'CAGR_OOS': np.nan, 'Sharpe_OOS': np.nan,
                        'MaxDD_FULL': np.nan, 'Worst10Y_star': np.nan,
                        'Trades_yr': np.nan,
                    }
                    tag = '（ERROR）'

            delta_cagr = (metrics['CAGR_OOS'] - center_metrics['CAGR_OOS']
                          if not np.isnan(metrics['CAGR_OOS']) else np.nan)
            print(
                f'    {param_name}={value}{tag:20s}  '
                f'CAGR_OOS={_fmt_pct(metrics["CAGR_OOS"]):10s}  '
                f'Sharpe={_fmt_sharpe(metrics["Sharpe_OOS"]):8s}  '
                f'Δ={_fmt_delta(delta_cagr):10s}'
            )
            rows.append({'value': value, 'metrics': metrics})

        all_results[param_name] = rows

        # 感度判定
        sens = judge_sensitivity(param_name, rows, center_metrics['CAGR_OOS'])
        sens['param'] = param_name
        summary.append(sens)
        print(
            f'  → 感度判定: {sens["verdict"]}  '
            f'CAGR範囲={_fmt_delta(sens["cagr_range"]) if not np.isnan(sens["cagr_range"]) else "N/A"}  '
            f'Sharpe範囲={sens["sharpe_range"]:+.3f}' if not np.isnan(sens['sharpe_range']) else
            f'  → 感度判定: {sens["verdict"]}'
        )

    print()

    # ── STEP 4: 総合サマリー表示 ────────────────────────────────────────────
    robust_count  = sum(1 for s in summary if s['verdict'] == 'ROBUST')
    fragile_count = sum(1 for s in summary if s['verdict'] == 'FRAGILE')
    total = len(summary)

    if robust_count == total:
        overall = 'ROBUST'
    elif fragile_count > 0:
        overall = 'FRAGILE'
    else:
        overall = 'MIXED'

    print('=' * 70)
    print('総合判定サマリー')
    print('=' * 70)
    print(f'  {"パラメータ":<15} {"中心 CAGR_OOS":>14} {"CAGR範囲":>22} {"Sharpe範囲":>14} {"判定"}')
    print(f'  {"-"*15} {"-"*14} {"-"*22} {"-"*14} {"-"*10}')
    for s in summary:
        cagr_range_str = (
            f'[{_fmt_pct(s["cagr_min"])}, {_fmt_pct(s["cagr_max"])}]'
            if not np.isnan(s['cagr_range']) else 'N/A'
        )
        sharpe_range_str = (
            f'[{_fmt_sharpe(s["sharpe_min"])}, {_fmt_sharpe(s["sharpe_max"])}]'
            if not np.isnan(s['sharpe_range']) else 'N/A'
        )
        print(f'  {s["param"]:<15} {_fmt_pct(center_metrics["CAGR_OOS"]):>14} '
              f'{cagr_range_str:>22} {sharpe_range_str:>14} {s["verdict"]}')
    print(f'\n  ROBUST {robust_count}/{total}   FRAGILE {fragile_count}/{total}')
    print(f'  全体判定: {overall}')
    print('=' * 70)
    print()

    # ── STEP 5: ファイル出力 ────────────────────────────────────────────────
    # Markdown
    md_filename = f'E4_PARAM_SENSITIVITY_{date_str.replace("-", "")}.md'
    md_path = os.path.join(audit_dir, md_filename)
    md_text = build_markdown(all_results, center_metrics, summary, date_str)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f'[OUT] {md_path}')

    # YAML
    yaml_path = os.path.join(audit_dir, 'e4_param_sensitivity.yaml')
    yaml_data = build_yaml_data(all_results, center_metrics, summary, date_str)
    if HAS_YAML:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)
    else:
        # yaml 未インストール時: 簡易 JSON 代替
        import json
        with open(yaml_path.replace('.yaml', '.json'), 'w', encoding='utf-8') as f:
            json.dump(yaml_data, f, ensure_ascii=False, indent=2)
        yaml_path = yaml_path.replace('.yaml', '.json')
    print(f'[OUT] {yaml_path}')

    # CSV
    csv_path = os.path.join(audit_dir, 'e4_param_sensitivity.csv')
    write_csv(all_results, csv_path)
    print(f'[OUT] {csv_path}')

    print()
    print(f'完了。全体判定: {overall}  (ROBUST {robust_count}/{total}, FRAGILE {fragile_count}/{total})')


if __name__ == '__main__':
    main()
