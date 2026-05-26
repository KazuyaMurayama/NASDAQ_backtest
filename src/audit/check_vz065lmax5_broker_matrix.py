"""
check_vz065lmax5_broker_matrix.py
PoC 2.14-VZ065LMAX5: ブローカー別コストシナリオ × vz065+lmax5 フルバックテスト比較
================================================================
S2_VZGated(l_max=5.0) + LT2-N750 + Regime k_lt (k_lo=0.1, k_hi=0.8, k_mid=0.5, vz_thr=0.65)
を6ブローカーコストシナリオで比較し、Markdownレポート + YAML を生成する。

出力:
  audit_results/VZ065LMAX5_BROKER_MATRIX_<DATE>.md
  audit_results/vz065lmax5_broker_matrix.yaml
"""

import sys, os, types

# multitasking stub (must come before sys.path manipulation)
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

# Path resolution: src/audit/ → src/ → project root
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, 'src'))

import importlib.util
import pathlib
import datetime
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Dynamic import: _audit_strategy (sibling file)
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    '_audit_strategy',
    pathlib.Path(__file__).parent / '_audit_strategy.py',
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_vz065lmax5_strategy_assets           = _mod.build_vz065lmax5_strategy_assets
build_vz065lmax5_strategy_nav_for_scenario = _mod.build_vz065lmax5_strategy_nav_for_scenario

# ---------------------------------------------------------------------------
# Dynamic import: _sweep_format (src/)
# ---------------------------------------------------------------------------
_spec2 = importlib.util.spec_from_file_location(
    '_sweep_format',
    pathlib.Path(BASE) / 'src' / '_sweep_format.py',
)
_smod = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_smod)
MD_HEADER_STRAT    = _smod.MD_HEADER_STRAT
fmt_row_strat      = _smod.fmt_row_strat
MD_METRIC_GLOSSARY = _smod.MD_METRIC_GLOSSARY

# ---------------------------------------------------------------------------
# Standard imports from src/
# ---------------------------------------------------------------------------
from cfd_leverage_backtest import calc_7metrics, IS_END, OOS_START
from compute_cfd_worst10y import nav_to_annual, rolling_nY_cagr

# ---------------------------------------------------------------------------
# Broker scenarios (E4 broker matrix と同一)
# ---------------------------------------------------------------------------
BROKER_SCENARIOS = [
    {'name': '(1) 現行(CFD_SPREAD_LOW 0.20%)',  'kind': 'spread_only', 'cfd_spread': 0.0020, 'extra_funding': 0.0,   'full_funding': None},
    {'name': '(2) くりっく株365 (4.92%全体)',    'kind': 'kurikku365',  'cfd_spread': 0.0,   'extra_funding': 0.0,   'full_funding': 0.0492},
    {'name': '(3) SBI-CFD 低推定 (SOFR+1.0%)',  'kind': 'sofr_plus',   'cfd_spread': 0.0020, 'extra_funding': 0.0100, 'full_funding': None},
    {'name': '(4) SBI-CFD 高推定 (SOFR+3.0%)',  'kind': 'sofr_plus',   'cfd_spread': 0.0020, 'extra_funding': 0.0300, 'full_funding': None},
    {'name': '(5) 楽天/IG-CFD (SOFR+3.0%)',     'kind': 'sofr_plus',   'cfd_spread': 0.0020, 'extra_funding': 0.0300, 'full_funding': None},
    {'name': '(6) GMO/DMM ロール (4.5%全体)',    'kind': 'roll_cost',   'cfd_spread': 0.0,   'extra_funding': 0.0,   'full_funding': 0.045},
]

# ---------------------------------------------------------------------------
# Pass/fail thresholds
# ---------------------------------------------------------------------------
THRESH_SHARPE   = 0.70
THRESH_CAGR_OOS = 0.20
THRESH_WORST10Y = 0.10

# ---------------------------------------------------------------------------
# E4 戦略比較値（e4_broker_matrix.yaml から取得）
# ---------------------------------------------------------------------------

def _load_e4_ref() -> dict:
    """E4 戦略の CAGR_OOS / Sharpe_OOS を e4_broker_matrix.yaml から読み込む。"""
    yaml_path = os.path.join(BASE, 'audit_results', 'e4_broker_matrix.yaml')
    if not os.path.exists(yaml_path):
        print(f'[WARN] e4_broker_matrix.yaml not found, comparison skipped.')
        return {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    ref = {}
    for sc in data.get('scenarios', []):
        name = sc['name']
        m = sc.get('metrics', {})
        cagr = m.get('CAGR_OOS')
        sharpe = m.get('Sharpe_OOS')
        if cagr is not None and sharpe is not None:
            ref[name] = (float(cagr), float(sharpe))
    print(f'[INFO] Loaded E4 ref from {yaml_path} ({len(ref)} scenarios)')
    return ref


def compute_metrics(nav, dates, assets: dict) -> dict:
    """NAVから9指標標準メトリクスdictを計算して返す。"""
    base = calc_7metrics(nav, dates)

    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, n=10)
    worst10y_star = float(np.min(r10)) if len(r10) >= 1 else np.nan

    r5 = rolling_nY_cagr(ann, n=5)
    p10_5y = float(np.percentile(r5, 10)) if len(r5) >= 5 else np.nan

    is_oos_gap = base['CAGR_IS'] - base['CAGR_OOS']
    trades_yr = assets['n_tr'] / assets['n_years']

    return {
        'CAGR_OOS':    base['CAGR_OOS'],
        'CAGR_IS':     base['CAGR_IS'],
        'Sharpe_OOS':  base['Sharpe_OOS'],
        'MaxDD_FULL':  base['MaxDD_FULL'],
        'Worst10Y_star': worst10y_star,
        'P10_5Y':      p10_5y,
        'IS_OOS_gap':  is_oos_gap,
        'Trades_yr':   trades_yr,
        'WFA_CI95_lo': None,
        'WFA_WFE':     None,
    }


def verdict(r: dict) -> str:
    ok_sharpe   = r['Sharpe_OOS']   >= THRESH_SHARPE
    ok_cagr     = r['CAGR_OOS']     >= THRESH_CAGR_OOS
    ok_worst10y = r['Worst10Y_star'] >= THRESH_WORST10Y
    if ok_sharpe and ok_cagr and ok_worst10y:
        return 'PASS'
    fails = []
    if not ok_sharpe:
        fails.append(f'Sharpe_OOS={r["Sharpe_OOS"]:+.3f}<{THRESH_SHARPE}')
    if not ok_cagr:
        fails.append(f'CAGR_OOS={r["CAGR_OOS"]*100:+.1f}%<{THRESH_CAGR_OOS*100:.0f}%')
    if not ok_worst10y:
        fails.append(f'Worst10Y★={r["Worst10Y_star"]*100:+.1f}%<{THRESH_WORST10Y*100:.0f}%')
    return 'FAIL (' + ', '.join(fails) + ')'


def overall_risk(results: list) -> str:
    pass_count = sum(1 for r in results if r['verdict'].startswith('PASS'))
    n = len(results)
    if pass_count == n:
        return 'LOW'
    elif pass_count >= n // 2:
        return 'MEDIUM'
    else:
        return 'HIGH'


def build_markdown(results: list, date_str: str, e4_ref: dict) -> str:
    lines = []
    lines.append('# 2.14-VZ065LMAX5 ブローカー別コストシナリオ × vz065+lmax5 フルバックテスト')
    lines.append('')
    lines.append(f'生成日: {date_str}')
    lines.append('')
    lines.append('**戦略:** S2_VZGated(l_max=5.0) + LT2-N750 + Regime k_lt '
                 '(k_lo=0.1, k_hi=0.8, k_mid=0.5, vz_thr=0.65)')
    lines.append('')

    # MD table (9指標標準 MD_HEADER_STRAT)
    lines.append(MD_HEADER_STRAT[0])
    lines.append(MD_HEADER_STRAT[1])
    for r in results:
        lines.append(fmt_row_strat(r['name'], r['metrics']))
    lines.append('')
    lines.append(MD_METRIC_GLOSSARY)
    lines.append('')

    # Pass/Fail verdict section
    lines.append('## Pass/Fail 判定')
    lines.append('')
    lines.append(f'| シナリオ | Sharpe_OOS≥{THRESH_SHARPE} | CAGR_OOS≥{THRESH_CAGR_OOS*100:.0f}% | Worst10Y★≥{THRESH_WORST10Y*100:.0f}% | 総合 |')
    lines.append('|:---------|:---:|:---:|:---:|:------|')
    for r in results:
        m = r['metrics']
        ok_s  = 'OK' if m['Sharpe_OOS']   >= THRESH_SHARPE   else 'NG'
        ok_c  = 'OK' if m['CAGR_OOS']     >= THRESH_CAGR_OOS else 'NG'
        ok_w  = 'OK' if m['Worst10Y_star'] >= THRESH_WORST10Y else 'NG'
        v     = r['verdict']
        lines.append(f'| {r["name"]} | {ok_s} | {ok_c} | {ok_w} | {v} |')
    lines.append('')

    risk = overall_risk(results)
    lines.append(f'## 総合リスク判定: **{risk}**')
    lines.append('')
    pass_count = sum(1 for r in results if r['verdict'].startswith('PASS'))
    lines.append(f'PASS: {pass_count}/{len(results)} シナリオ')
    lines.append('')
    if risk == 'LOW':
        lines.append('> 全シナリオ PASS。どのブローカーでも採用可能。')
    elif risk == 'MEDIUM':
        lines.append('> 一部シナリオで FAIL。コスト条件によっては採用を要再検討。')
    else:
        lines.append('> 過半数のシナリオが FAIL。ブローカー選定を慎重に行うこと。')
    lines.append('')

    # vz065+lmax5 vs E4 比較セクション
    if e4_ref:
        lines.append('## vz065+lmax5 vs E4 戦略 比較')
        lines.append('')
        lines.append('| シナリオ | E4 CAGR_OOS | vz065lmax5 CAGR_OOS | 差 | E4 Sharpe | vz065lmax5 Sharpe | 差 |')
        lines.append('|:---------|------------:|--------------------:|---:|----------:|------------------:|---:|')
        for r in results:
            m = r['metrics']
            new_cagr   = m['CAGR_OOS']
            new_sharpe = m['Sharpe_OOS']
            if r['name'] in e4_ref:
                e4_cagr, e4_sharpe = e4_ref[r['name']]
                diff_cagr = (new_cagr - e4_cagr) * 100
                diff_sharpe = new_sharpe - e4_sharpe
                lines.append(
                    f'| {r["name"]} '
                    f'| {e4_cagr*100:+.1f}% '
                    f'| {new_cagr*100:+.1f}% '
                    f'| {diff_cagr:+.1f}pp '
                    f'| {e4_sharpe:+.3f} '
                    f'| {new_sharpe:+.3f} '
                    f'| {diff_sharpe:+.3f} |'
                )
            else:
                lines.append(
                    f'| {r["name"]} | — | {new_cagr*100:+.1f}% | — | — | {new_sharpe:+.3f} | — |'
                )
        lines.append('')
        lines.append('*差 = vz065+lmax5 − E4 (CAGR は pp、Sharpe は絶対値)。E4 = S2_VZGated(l_max=7.0) + LT2-N750 + E4 Regime k_lt (vz_thr=0.70).*')
        lines.append('')

    return '\n'.join(lines)


def build_yaml_data(results: list, date_str: str) -> dict:
    data = {
        'generated': date_str,
        'strategy': 'S2_VZGated(l_max=5.0) + LT2-N750 + Regime k_lt (k_lo=0.1, k_hi=0.8, k_mid=0.5, vz_thr=0.65)',
        'thresholds': {
            'Sharpe_OOS': THRESH_SHARPE,
            'CAGR_OOS': THRESH_CAGR_OOS,
            'Worst10Y_star': THRESH_WORST10Y,
        },
        'scenarios': [],
    }
    for r in results:
        m = r['metrics']
        entry = {
            'name': r['name'],
            'kind': r['kind'],
            'verdict': r['verdict'],
            'metrics': {
                k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
                for k, v in m.items()
                if k not in ('WFA_CI95_lo', 'WFA_WFE')
            },
        }
        entry['metrics']['WFA_CI95_lo'] = None
        entry['metrics']['WFA_WFE'] = None
        data['scenarios'].append(entry)
    data['overall_risk'] = overall_risk(results)
    return data


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    date_str = datetime.date.today().isoformat()
    audit_dir = os.path.join(BASE, 'audit_results')
    os.makedirs(audit_dir, exist_ok=True)

    print('[2.14-VZ065LMAX5] ブローカー別コストシナリオ × vz065+lmax5 フルバックテスト比較')
    print(f'          BASE={BASE}')
    print(f'          出力先={audit_dir}')
    print()

    # Load E4 reference values
    e4_ref = _load_e4_ref()

    # Build / load vz065+lmax5 assets (cached)
    print('[STEP 1] vz065+lmax5戦略アセット構築 / キャッシュ読み込み ...')
    assets = build_vz065lmax5_strategy_assets(force_rebuild=False)
    dates = assets['dates']
    print(f'  dates             : {dates.iloc[0].date()} -> {dates.iloc[-1].date()}')
    print(f'  n_tr              : {assets["n_tr"]}  ({assets["n_tr"]/assets["n_years"]:.1f}/yr)')
    print(f'  vz065lmax5_params : {assets["vz065lmax5_params"]}')
    print()

    # Run each scenario
    print('[STEP 2] 各ブローカーシナリオのNAV計算 ...')
    results = []
    for sc in BROKER_SCENARIOS:
        print(f'  → {sc["name"]} ...')
        nav = build_vz065lmax5_strategy_nav_for_scenario(
            assets,
            cfd_spread=sc['cfd_spread'],
            extra_funding_ann=sc['extra_funding'],
            full_funding_ann=sc['full_funding'],
        )
        m = compute_metrics(nav, dates, assets)
        v = verdict(m)
        results.append({
            'name':    sc['name'],
            'kind':    sc['kind'],
            'metrics': m,
            'verdict': v,
        })
        e4_str = ''
        if sc['name'] in e4_ref:
            e4_cagr, e4_sharpe = e4_ref[sc['name']]
            diff_pp = (m['CAGR_OOS'] - e4_cagr) * 100
            e4_str = f'  E4CAGR={e4_cagr*100:+.2f}%  Δ={diff_pp:+.2f}pp'
        print(f'     CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%  '
              f'Sharpe={m["Sharpe_OOS"]:+.3f}  '
              f'Worst10Y★={m["Worst10Y_star"]*100:+.2f}%  '
              f'→ {v}{e4_str}')
    print()

    # Write Markdown
    md_filename = f'VZ065LMAX5_BROKER_MATRIX_{date_str.replace("-", "")}.md'
    md_path = os.path.join(audit_dir, md_filename)
    md_text = build_markdown(results, date_str, e4_ref)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f'[OUT] {md_path}')

    # Write YAML
    yaml_path = os.path.join(audit_dir, 'vz065lmax5_broker_matrix.yaml')
    yaml_data = build_yaml_data(results, date_str)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)
    print(f'[OUT] {yaml_path}')

    # Summary
    print()
    print('=' * 70)
    print('vz065+lmax5 ブローカー別コストシナリオ 総合結果サマリー')
    print('=' * 70)
    print(f'  {"シナリオ":<42} {"CAGR_OOS":>12} {"Sharpe":>10} {"Worst10Y★":>10} {"判定"}')
    print(f'  {"-"*42} {"-"*12} {"-"*10} {"-"*10} {"-"*20}')
    for r in results:
        m = r['metrics']
        print(f'  {r["name"]:<42} '
              f'{m["CAGR_OOS"]*100:>+10.2f}%  '
              f'{m["Sharpe_OOS"]:>+9.3f}  '
              f'{m["Worst10Y_star"]*100:>+9.2f}%  '
              f'{r["verdict"]}')
    risk = overall_risk(results)
    print(f'\n  総合リスク判定: {risk}')
    pass_count = sum(1 for r in results if r['verdict'].startswith('PASS'))
    print(f'  PASS: {pass_count}/{len(results)} シナリオ')
    print('=' * 70)

    if e4_ref:
        print()
        print('--- vz065+lmax5 vs E4 CAGR_OOS 差分サマリー ---')
        for r in results:
            m = r['metrics']
            if r['name'] in e4_ref:
                e4_cagr, _ = e4_ref[r['name']]
                diff_pp = (m['CAGR_OOS'] - e4_cagr) * 100
                print(f'  {r["name"]:<42} {diff_pp:+.2f}pp  '
                      f'(E4={e4_cagr*100:+.2f}%  vz065lmax5={m["CAGR_OOS"]*100:+.2f}%)')


if __name__ == '__main__':
    main()
