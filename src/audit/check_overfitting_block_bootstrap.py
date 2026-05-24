"""
check_overfitting_block_bootstrap.py
3.2 Block Bootstrap 信頼区間検定
=================================
OOS期間（2021-05-08〜最終日）の日次リターンに対して Stationary Block Bootstrap を適用し、
Sharpe Ratio の経験的信頼区間と H0: SR=0 への p値を計算する。
NAV 再構築は不要（リターン系列だけで完結、高速）。

出力:
  audit_results/BOOTSTRAP_{TODAY}.md
  audit_results/bootstrap_results.yaml
"""

import sys, os, types

# multitasking stub (must come BEFORE sys.path manipulation)
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, 'src'))

import importlib.util, pathlib
import datetime
import numpy as np

# ---------------------------------------------------------------------------
# Dynamic import: _audit_strategy (sibling file)
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    '_audit_strategy',
    pathlib.Path(__file__).parent / '_audit_strategy.py'
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_best_strategy_assets = _mod.build_best_strategy_assets

# ---------------------------------------------------------------------------
# YAML with json fallback
# ---------------------------------------------------------------------------
try:
    import yaml
    def _dump_yaml(obj, path):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(obj, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    HAS_YAML = True
except ImportError:
    import json
    def _dump_yaml(obj, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    HAS_YAML = False

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
B_BOOTSTRAP   = 5000
BLOCK_LENGTHS = [20, 63, 126]   # L = 1ヶ月 / 3ヶ月 / 6ヶ月
IS_N_CHUNKS   = 10               # IS期間の安定性検証: 10分割
RANDOM_SEED   = 42

AUDIT_DIR = os.path.join(BASE, 'audit_results')
os.makedirs(AUDIT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sharpe_ann(r, trading_days=252):
    """年率Sharpe (Rf=0)"""
    std = r.std(ddof=1)
    if std == 0:
        return 0.0
    return r.mean() / std * np.sqrt(trading_days)


def stationary_block_bootstrap(r, L, B):
    """
    Stationary Block Bootstrap (Politis-Romano 1994)
    p_geom = 1/L で幾何分布からブロック長をサンプリング
    境界は循環インデックスで wrap-around
    """
    T = len(r)
    results = []
    p = 1.0 / L
    for _ in range(B):
        sample = []
        while len(sample) < T:
            start  = np.random.randint(0, T)
            blklen = np.random.geometric(p)   # 期待値 = L
            for i in range(blklen):
                sample.append(r[(start + i) % T])
        sample = np.array(sample[:T])
        results.append(sharpe_ann(sample))
    return np.array(results)


def row_verdict(ci_lo, p_val):
    if ci_lo > 0.0 and p_val < 0.05:
        return 'PASS'
    elif ci_lo > -0.2:
        return 'WARN'
    else:
        return 'FAIL'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from cfd_leverage_backtest import OOS_START, IS_END, TRADING_DAYS

    TODAY = datetime.date.today()
    TODAY_STR = TODAY.strftime('%Y-%m-%d')

    print('=' * 60)
    print('3.2 Block Bootstrap 信頼区間検定')
    print('=' * 60)

    # Step 1: データ取得
    print('\n[Step 1] Loading best strategy assets ...')
    assets = build_best_strategy_assets()
    nav    = assets['nav_baseline']
    dates  = assets['dates']

    # OOS
    oos_mask = dates >= OOS_START
    nav_oos  = nav[oos_mask]
    r_oos    = nav_oos.pct_change().dropna().values  # numpy配列

    oos_end  = str(dates[oos_mask].iloc[-1].date())
    T        = len(r_oos)

    # IS
    is_mask  = dates <= IS_END
    nav_is   = nav[is_mask]
    r_is     = nav_is.pct_change().dropna().values

    print(f'  OOS: {OOS_START} ~ {oos_end}  (T = {T} 日)')
    print(f'  IS : {dates[is_mask].iloc[0].date()} ~ {IS_END}  (T_IS = {len(r_is)} 日)')

    # Step 2: 観測 Sharpe
    print('\n[Step 2] Computing observed Sharpe ...')
    sr_observed = sharpe_ann(r_oos, TRADING_DAYS)
    print(f'  Sharpe_OOS (observed) = {sr_observed:.4f}')

    # Step 3: Stationary Block Bootstrap
    print('\n[Step 3] Running Stationary Block Bootstrap ...')
    np.random.seed(RANDOM_SEED)

    bootstrap_results = {}  # L -> np.array(B,)
    for L in BLOCK_LENGTHS:
        print(f'  Bootstrap L={L} (B={B_BOOTSTRAP})...')
        bootstrap_results[L] = stationary_block_bootstrap(r_oos, L, B_BOOTSTRAP)
    print('  Done.')

    # Step 4: 統計量集計
    print('\n[Step 4] Summarizing statistics ...')
    summary = {}
    for L in BLOCK_LENGTHS:
        dist = bootstrap_results[L]
        summary[L] = {
            'CI95_lo':  float(np.quantile(dist, 0.025)),
            'CI95_hi':  float(np.quantile(dist, 0.975)),
            'median':   float(np.median(dist)),
            'p_value':  float((dist <= 0).mean()),  # H0: SR=0
            'P5':       float(np.quantile(dist, 0.05)),
            'P25':      float(np.quantile(dist, 0.25)),
            'P75':      float(np.quantile(dist, 0.75)),
            'P95':      float(np.quantile(dist, 0.95)),
        }

    # Step 5: IS 期間 安定性検証
    print('\n[Step 5] IS stability (10-fold) ...')
    chunks = np.array_split(r_is, IS_N_CHUNKS)
    is_stability = []
    for i, ck in enumerate(chunks):
        if len(ck) < 50:
            continue
        sr_ck = sharpe_ann(ck, TRADING_DAYS)
        is_stability.append({
            'chunk': i + 1,
            'n_days': len(ck),
            'SR_ann': float(sr_ck),
        })
    sr_is_values = [s['SR_ann'] for s in is_stability]
    is_mean = float(np.mean(sr_is_values))
    is_std  = float(np.std(sr_is_values))
    is_min  = float(np.min(sr_is_values))
    is_max  = float(np.max(sr_is_values))

    if abs(is_mean) > 1e-9:
        cov_val = is_std / abs(is_mean)
        cov_str = f'{cov_val:.3f}'
    else:
        cov_str = 'N/A'

    # Step 6: 判定
    print('\n[Step 6] Verdict ...')
    ref_L  = 63
    ci_lo  = summary[ref_L]['CI95_lo']
    p_val  = summary[ref_L]['p_value']
    verdict = row_verdict(ci_lo, p_val)
    print(f'  L=63: CI95_lo={ci_lo:.4f}, p_value={p_val:.4f} → {verdict}')

    # 判定理由
    if verdict == 'PASS':
        reason = (
            f'L={ref_L} ブロックでの CI95_lo = {ci_lo:.4f} > 0.0 かつ '
            f'p値 = {p_val:.4f} < 0.05。OOS Sharpe は統計的に有意にゼロより大きい。'
        )
    elif verdict == 'WARN':
        reason = (
            f'L={ref_L} ブロックでの CI95_lo = {ci_lo:.4f}（-0.2 〜 0 の範囲）。'
            f'正の Sharpe は示唆されるが有意水準に達しない（p = {p_val:.4f}）。'
        )
    else:
        reason = (
            f'L={ref_L} ブロックでの CI95_lo = {ci_lo:.4f} < -0.2。'
            f'OOS Sharpe の統計的有意性は確認されない（p = {p_val:.4f}）。'
        )

    # =========================================================================
    # Output: Markdown
    # =========================================================================
    md_path = os.path.join(AUDIT_DIR, f'BOOTSTRAP_{TODAY_STR.replace("-", "")}.md')

    # Bootstrap テーブル行
    def _tbl_row(L):
        s = summary[L]
        v = row_verdict(s['CI95_lo'], s['p_value'])
        return (
            f"| {L:>4} "
            f"| {s['CI95_lo']:>7.4f} "
            f"| {s['median']:>7.4f} "
            f"| {s['CI95_hi']:>7.4f} "
            f"| {s['p_value']:>14.4f} "
            f"| {v:<14} |"
        )

    # IS stability テーブル行
    is_rows = []
    for s in is_stability:
        is_rows.append(f"| {s['chunk']:>6} | {s['n_days']:>6} | {s['SR_ann']:>13.4f} |")

    md_lines = [
        '# 3.2 Block Bootstrap 信頼区間検定',
        '',
        f'- 生成日: {TODAY_STR}',
        '- 戦略: S2_VZGated + LT2-N750-k0.5-modeB',
        f'- OOS 期間: {OOS_START} 〜 {oos_end}  (T = {T} 日)',
        f'- Bootstrap: B = {B_BOOTSTRAP}, ブロック長 L = {BLOCK_LENGTHS}',
        f'- 乱数シード: {RANDOM_SEED}',
        '',
        f'## 観測 Sharpe_OOS: {sr_observed:.4f}',
        '',
        '## Bootstrap 信頼区間（OOS日次リターン）',
        '',
        '| ブロック長 L | CI95_lo | 中央値 | CI95_hi | p値 (H0:SR=0) | 判定           |',
        '|---:|---:|---:|---:|---:|:---|',
    ]
    for L in BLOCK_LENGTHS:
        md_lines.append(_tbl_row(L))

    md_lines += [
        '',
        '*判定基準: CI95_lo > 0 かつ p < 0.05 → PASS / CI95_lo > -0.2 → WARN / それ以外 → FAIL*',
        '',
        '## IS 期間 10分割 安定性検証',
        '',
        '| チャンク | 日数 | Sharpe (年率) |',
        '|---:|---:|---:|',
    ]
    md_lines.extend(is_rows)
    md_lines += [
        '',
        f'IS 平均: {is_mean:.3f} / 標準偏差: {is_std:.3f} / CoV: {cov_str} / 最小: {is_min:.3f} / 最大: {is_max:.3f}',
        '',
        f'## 総合判定 (L={ref_L}基準): **{verdict}**',
        '',
        f'理由: {reason}',
        '',
    ]

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f'\n[Output] {md_path}')

    # =========================================================================
    # Output: YAML (with json fallback)
    # =========================================================================
    yaml_path = os.path.join(AUDIT_DIR, 'bootstrap_results.yaml')

    out_data = {
        'strategy': 'S2_VZGated+LT2-N750-k0.5-modeB',
        'generated': TODAY_STR,
        'oos_period': {
            'start': OOS_START,
            'end': oos_end,
            'T': T,
        },
        'sr_observed': round(float(sr_observed), 6),
        'bootstrap': {
            'B': B_BOOTSTRAP,
            'seed': RANDOM_SEED,
            **{
                f'L{L}': {
                    'CI95_lo': round(summary[L]['CI95_lo'], 6),
                    'CI95_hi': round(summary[L]['CI95_hi'], 6),
                    'median':  round(summary[L]['median'],  6),
                    'p_value': round(summary[L]['p_value'], 6),
                    'P5':      round(summary[L]['P5'],      6),
                    'P25':     round(summary[L]['P25'],     6),
                    'P75':     round(summary[L]['P75'],     6),
                    'P95':     round(summary[L]['P95'],     6),
                }
                for L in BLOCK_LENGTHS
            },
        },
        'is_stability': {
            'n_chunks': IS_N_CHUNKS,
            'mean_SR': round(is_mean, 6),
            'std_SR':  round(is_std,  6),
            'min_SR':  round(is_min,  6),
            'max_SR':  round(is_max,  6),
            'CoV':     cov_str,
            'chunks':  is_stability,
        },
        'verdict': verdict,
        'verdict_basis': f'L={ref_L}',
        'verdict_reason': reason,
    }

    _dump_yaml(out_data, yaml_path)
    print(f'[Output] {yaml_path}')

    # =========================================================================
    # Console summary
    # =========================================================================
    print('\n' + '=' * 60)
    print(f'Sharpe_OOS (observed) : {sr_observed:.4f}')
    print(f'{"L":>6} {"CI95_lo":>10} {"median":>10} {"CI95_hi":>10} {"p_value":>10} {"verdict":>8}')
    for L in BLOCK_LENGTHS:
        s = summary[L]
        v = row_verdict(s['CI95_lo'], s['p_value'])
        print(f'{L:>6} {s["CI95_lo"]:>10.4f} {s["median"]:>10.4f} {s["CI95_hi"]:>10.4f} {s["p_value"]:>10.4f} {v:>8}')
    print(f'\nIS stability: mean={is_mean:.3f} std={is_std:.3f} CoV={cov_str}')
    print(f'\n総合判定 (L={ref_L}基準): {verdict}')
    print(f'  {reason}')
    print('=' * 60)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
