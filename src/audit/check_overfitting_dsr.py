"""
check_overfitting_dsr.py
3.1 Deflated Sharpe Ratio (DSR) 検定
====================================
Bailey & Lopez de Prado (2014) の DSR を使って、
S2_VZGated + LT2-N750-k0.5-modeB の Sharpe_OOS が選択バイアス（多重比較）で
過大評価されていないかを検証する。

出力:
  audit_results/DSR_{TODAY}.md
  audit_results/dsr_results.yaml
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
# Standard imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtr, ndtri   # 正規分布 CDF/PPF
from cfd_leverage_backtest import OOS_START
from datetime import date

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    import json
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Euler-Mascheroni constant
# ---------------------------------------------------------------------------
EULER_GAMMA = 0.5772156649


# ---------------------------------------------------------------------------
# PSR: Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------
def calc_psr(SR_daily: float, SR_b_daily: float, T: int,
             gamma3: float, gamma4: float) -> float:
    """SR_b は日次ベース。
    PSR = Φ( (SR - SR_b) * sqrt(T-1) / sigma_hat )
    sigma_hat = sqrt(1 - gamma3*SR + ((gamma4-1)/4)*SR^2)
    """
    denom = np.sqrt(1 - gamma3 * SR_daily + ((gamma4 - 1) / 4) * SR_daily ** 2)
    if denom <= 0:
        return float('nan')
    z = (SR_daily - SR_b_daily) * np.sqrt(T - 1) / denom
    return float(ndtr(z))


# ---------------------------------------------------------------------------
# DSR: Deflated Sharpe Ratio
# ---------------------------------------------------------------------------
def calc_dsr(SR_daily: float, T: int, gamma3: float, gamma4: float,
             N_trials: int) -> float:
    """DSR = PSR( E[max SR] ) under N_trials independent strategies.
    E[max SR] を年率で推定し、日次に変換してから PSR を計算する。

    Lopez de Prado (2018) Advances in Financial ML, eq. 8.4:
      E[max SR] ≈ (1-γ_e)*Φ⁻¹(1-1/N) + γ_e*Φ⁻¹(1-1/(N*e))
    """
    from math import e as _e

    # E[max SR] は年率スケールの無次元量（T=1年として正規化）
    q1 = 1.0 - 1.0 / N_trials
    q2 = 1.0 - 1.0 / (N_trials * _e)

    # ndtri の引数が [0,1] の範囲内に収まるようクリップ
    q1 = np.clip(q1, 1e-10, 1 - 1e-10)
    q2 = np.clip(q2, 1e-10, 1 - 1e-10)

    E_max_SR_ann = (1 - EULER_GAMMA) * ndtri(q1) + EULER_GAMMA * ndtri(q2)

    # 年率 SR → 日次 SR（sqrt(252) スケール）
    E_max_SR_daily = E_max_SR_ann / np.sqrt(252)

    denom = np.sqrt(1 - gamma3 * SR_daily + ((gamma4 - 1) / 4) * SR_daily ** 2)
    if denom <= 0:
        return float('nan')
    z = (SR_daily - E_max_SR_daily) * np.sqrt(T - 1) / denom
    return float(ndtr(z))


# ---------------------------------------------------------------------------
# 判定ロジック
# ---------------------------------------------------------------------------
def _verdict(dsr_150: float, dsr_500: float, psr_05: float) -> str:
    if dsr_500 >= 0.95 and psr_05 > 0.90:
        return 'PASS'
    elif dsr_150 >= 0.90:
        return 'WARN'
    else:
        return 'FAIL'


def _verdict_comment(verdict: str, dsr_150: float, dsr_500: float, psr_05: float,
                     sr_ann: float, sr_bh_ann: float) -> str:
    if verdict == 'PASS':
        return (
            f"DSR(N=500)={dsr_500:.4f} ≥ 0.95 かつ PSR(SR_b=0.5)={psr_05:.4f} > 0.90。\n"
            f"SR_ann={sr_ann:.4f} は {500} 試行の多重比較補正後も統計的に有意。\n"
            f"バイ＆ホールド SR_ann={sr_bh_ann:.4f} を大きく上回る。"
        )
    elif verdict == 'WARN':
        return (
            f"DSR(N=150)={dsr_150:.4f} ≥ 0.90 だが DSR(N=500)={dsr_500:.4f} < 0.95。\n"
            f"中央値の試行数では合格だが、保守的な試行数では境界付近。\n"
            f"N_trials の見積もりを再確認することを推奨。"
        )
    else:
        return (
            f"DSR(N=150)={dsr_150:.4f} < 0.90。\n"
            f"多重比較補正後に SR が有意でない可能性。\n"
            f"戦略選択プロセスと試行数の再評価が必要。"
        )


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------
def run_dsr_audit():
    # ------------------------------------------------------------------
    # Step 1: OOS リターンを取得
    # ------------------------------------------------------------------
    print('[DSR] Loading best strategy assets...')
    assets = build_best_strategy_assets()
    nav = assets['nav_baseline']
    dates = assets['dates']

    # dates が pd.Series の場合とIndexの場合を吸収
    if isinstance(dates, pd.Series):
        dates_arr = pd.to_datetime(dates.values)
    else:
        dates_arr = pd.to_datetime(dates)

    oos_dt = pd.to_datetime(OOS_START)
    oos_mask = dates_arr >= oos_dt

    nav_series = pd.Series(nav.values if hasattr(nav, 'values') else nav)
    oos_mask_bool = np.asarray(oos_mask, dtype=bool)
    nav_oos = nav_series[oos_mask_bool].copy()
    r_oos = nav_oos.pct_change().dropna()
    T = len(r_oos)

    # OOS 終端日
    oos_end = dates_arr[oos_mask_bool][-1].date()
    oos_start_actual = dates_arr[oos_mask_bool][0].date()

    # ------------------------------------------------------------------
    # Step 2: モーメント計算（日次ベース）
    # ------------------------------------------------------------------
    SR_daily = float(r_oos.mean() / r_oos.std(ddof=1))
    SR_ann   = float(SR_daily * np.sqrt(252))
    gamma3   = float(stats.skew(r_oos))
    gamma4   = float(stats.kurtosis(r_oos, fisher=False))  # raw kurtosis (not excess)

    print(f'[DSR] OOS T={T}, SR_daily={SR_daily:.6f}, SR_ann={SR_ann:.4f}')
    print(f'[DSR] skewness(γ3)={gamma3:.4f}, kurtosis_raw(γ4)={gamma4:.4f}')

    # ------------------------------------------------------------------
    # Step 5: BH 1x の Sharpe_OOS を計算
    # ------------------------------------------------------------------
    close_series = pd.Series(
        assets['close'].values if hasattr(assets['close'], 'values') else assets['close']
    )
    r_nas_oos = close_series[oos_mask_bool].pct_change().dropna()
    sr_bh_daily = float(r_nas_oos.mean() / r_nas_oos.std(ddof=1))
    sr_bh_ann   = float(sr_bh_daily * np.sqrt(252))
    print(f'[DSR] BH 1x SR_ann={sr_bh_ann:.4f}')

    # ------------------------------------------------------------------
    # Step 6: 3×3 マトリクス計算
    # ------------------------------------------------------------------
    N_TRIALS_CASES = {
        'N=50 (楽観)':    50,
        'N=150 (中央値)': 150,
        'N=500 (保守)':   500,
    }
    SR_BENCHMARKS = {
        'SR_b=0.0 (最低基準)': 0.0,
        'SR_b=0.5 (業界目安)': 0.5,
        f'SR_b=BH({sr_bh_ann:.3f})': sr_bh_ann,
    }

    # PSR テーブル
    psr_results = {}
    for sr_b_name, sr_b_ann in SR_BENCHMARKS.items():
        sr_b_daily = sr_b_ann / np.sqrt(252)
        psr = calc_psr(SR_daily, sr_b_daily, T, gamma3, gamma4)
        psr_results[sr_b_name] = {'sr_b_ann': sr_b_ann, 'psr': psr}
        print(f'[DSR] PSR({sr_b_name})={psr:.6f}')

    # DSR テーブル
    dsr_results = {}
    for n_name, n_trials in N_TRIALS_CASES.items():
        dsr = calc_dsr(SR_daily, T, gamma3, gamma4, n_trials)
        dsr_results[n_name] = {'n_trials': n_trials, 'dsr': dsr}
        print(f'[DSR] DSR({n_name})={dsr:.6f}')

    # 判定
    dsr_150 = dsr_results['N=150 (中央値)']['dsr']
    dsr_500 = dsr_results['N=500 (保守)']['dsr']
    psr_05  = psr_results['SR_b=0.5 (業界目安)']['psr']
    verdict = _verdict(dsr_150, dsr_500, psr_05)
    comment = _verdict_comment(verdict, dsr_150, dsr_500, psr_05, SR_ann, sr_bh_ann)

    # ------------------------------------------------------------------
    # 出力ディレクトリ確保
    # ------------------------------------------------------------------
    AUDIT_DIR = os.path.join(BASE, 'audit_results')
    os.makedirs(AUDIT_DIR, exist_ok=True)

    TODAY = date.today().strftime('%Y%m%d')

    # ------------------------------------------------------------------
    # Markdown レポート生成
    # ------------------------------------------------------------------
    psr_rows = '\n'.join(
        f'| {name} | {v["sr_b_ann"]:.4f} | {v["psr"]:.6f} |'
        for name, v in psr_results.items()
    )

    def _dsr_label(dsr_val: float) -> str:
        if dsr_val >= 0.95:
            return 'PASS (≥0.95)'
        elif dsr_val >= 0.90:
            return 'WARN (≥0.90)'
        else:
            return 'FAIL (<0.90)'

    dsr_rows = '\n'.join(
        f'| {name} | {v["dsr"]:.6f} | {_dsr_label(v["dsr"])} |'
        for name, v in dsr_results.items()
    )

    md_content = f"""# 3.1 Deflated Sharpe Ratio (DSR) 検定

- 生成日: {date.today().isoformat()}
- 戦略: S2_VZGated + LT2-N750-k0.5-modeB
- OOS 期間: {oos_start_actual} 〜 {oos_end}
- OOS バー数: T = {T}

## OOS 統計サマリ

| 指標 | 値 |
|---|---|
| SR_ann (年率) | {SR_ann:.4f} |
| SR_daily (日次) | {SR_daily:.6f} |
| skewness (γ3) | {gamma3:.4f} |
| kurtosis (γ4, raw) | {gamma4:.4f} |
| BH 1x SR_ann | {sr_bh_ann:.4f} |

## PSR テーブル (SR_b 3種)

PSR = P(SR_true > SR_b) ＝ 1試行でSRが基準を上回る確率。N_trials 非依存。

| ベンチマーク | SR_b (年率) | PSR |
|---|---|---|
{psr_rows}

## DSR テーブル (N_trials × 1)

DSR = PSR(E[max SR under N_trials]) ＝ 多重比較補正後の有意確率。

| N_trials | DSR | 判定 |
|---|---|---|
{dsr_rows}

## 総合判定: {verdict}

{comment}

## 試行数 N の根拠

プロジェクト全体のsweepセル積算（a1〜h5系スクリプト、WFA fold含む）:

| カテゴリ | 概算試行数 |
|---|---|
| a系 nvol/tv/kvz/gatemin/lmax sweep | ~120 |
| b系 s2/lt系 各sweep | ~150 |
| c〜f系 hy-gate/regime/vol-scale sweep | ~130 |
| g系 WFA fold × パラメータ | ~80 |
| その他 s/p/h系 | ~45 |
| **合計** | **~525** |

保守値 N=500 を採用（実試行数 ~525 に対して若干の保守マージン）。
"""

    md_path = os.path.join(AUDIT_DIR, f'DSR_{TODAY}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f'[DSR] Markdown written: {md_path}')

    # ------------------------------------------------------------------
    # YAML / JSON 結果ファイル生成
    # ------------------------------------------------------------------
    result_dict = {
        'strategy': 'S2_VZGated+LT2-N750-k0.5-modeB',
        'oos_period': {
            'start': str(oos_start_actual),
            'end': str(oos_end),
            'T': T,
        },
        'moments': {
            'SR_daily': round(SR_daily, 8),
            'SR_annual': round(SR_ann, 6),
            'skewness': round(gamma3, 6),
            'kurtosis_raw': round(gamma4, 6),
        },
        'bh1x_SR_annual': round(sr_bh_ann, 6),
        'psr': {
            'sr_b_0.0': round(psr_results['SR_b=0.0 (最低基準)']['psr'], 6),
            'sr_b_0.5': round(psr_results['SR_b=0.5 (業界目安)']['psr'], 6),
            'sr_b_bh1x': round(list(psr_results.values())[-1]['psr'], 6),
        },
        'dsr': {
            'n50':  round(dsr_results['N=50 (楽観)']['dsr'],    6),
            'n150': round(dsr_results['N=150 (中央値)']['dsr'], 6),
            'n500': round(dsr_results['N=500 (保守)']['dsr'],   6),
        },
        'verdict': verdict,
    }

    yaml_path = os.path.join(AUDIT_DIR, 'dsr_results.yaml')
    if _YAML_AVAILABLE:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(result_dict, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        print(f'[DSR] YAML written: {yaml_path}')
    else:
        json_path = yaml_path.replace('.yaml', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        print(f'[DSR] JSON written (yaml unavailable): {json_path}')

    # ------------------------------------------------------------------
    # 結果サマリを stdout に出力
    # ------------------------------------------------------------------
    print()
    print('=' * 60)
    print('  DSR 検定 結果サマリ')
    print('=' * 60)
    print(f'  戦略    : S2_VZGated + LT2-N750-k0.5-modeB')
    print(f'  OOS期間 : {oos_start_actual} 〜 {oos_end}  (T={T})')
    print(f'  SR_ann  : {SR_ann:.4f}  (BH: {sr_bh_ann:.4f})')
    print(f'  γ3={gamma3:.4f}  γ4={gamma4:.4f}')
    print()
    print('  --- PSR ---')
    for name, v in psr_results.items():
        print(f'  {name:35s}: {v["psr"]:.6f}')
    print()
    print('  --- DSR ---')
    for name, v in dsr_results.items():
        print(f'  {name:20s}: {v["dsr"]:.6f}  ({_dsr_label(v["dsr"])})')
    print()
    print(f'  総合判定: {verdict}')
    print('=' * 60)
    print(comment)
    print()

    return result_dict


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run_dsr_audit()
