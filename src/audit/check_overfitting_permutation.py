"""
check_overfitting_permutation.py
3.3 Permutation Test（シグナル置換検定）
=========================================
L_s2 (動的レバレッジシグナル) と lev_mod (S2+LT2 合成) をブロックシャッフルして
戦略の予測力が本物か（NASDAQ右肩上がりに乗っているだけか）を検定する。

OOS 期間のみで日次リターンを直接計算することで高速化。
B=1000 で 3種の検定を実施。

出力:
  audit_results/PERMUTATION_{TODAY}.md
  audit_results/permutation_results.yaml
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

# ---------------------------------------------------------------------------
# Dynamic import: _audit_strategy (sibling file)
# ---------------------------------------------------------------------------
import importlib.util, pathlib
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

import numpy as np
import pandas as pd
import time
import datetime

from cfd_leverage_backtest import OOS_START, IS_END, TRADING_DAYS, DELAY, CFD_SPREAD_LOW

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
B_PERM      = 1000
BLOCK_LEN   = 63      # 約3ヶ月
RANDOM_SEED = 42

AUDIT_DIR = os.path.join(BASE, 'audit_results')
os.makedirs(AUDIT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sharpe_ann(r):
    """年率 Sharpe (Rf=0)"""
    std = r.std(ddof=1)
    if std == 0:
        return 0.0
    return r.mean() / std * np.sqrt(TRADING_DAYS)


def block_shuffle(arr, block_len):
    """1次元配列をブロック長 block_len でシャッフルして返す（等長）"""
    arr = np.array(arr, dtype=float)
    T = len(arr)

    # ブロック先頭インデックスをシャッフル
    block_starts = list(range(0, T, block_len))
    np.random.shuffle(block_starts)

    result = []
    for start in block_starts:
        end = min(start + block_len, T)
        result.extend(arr[start:end])

    return np.array(result[:T])


def compute_oos_returns_from_signals(L_arr, lev_arr, oos_mask, assets,
                                     cfd_spread=CFD_SPREAD_LOW):
    """
    シグナルと対応する日次リターン（OOS のみ）を高速計算する。

    NAV 再構築の代わりに OOS 期間のみの daily return を直接計算。
    DELAY=2 シフトは全期間配列に対して行い、OOS マスクを後適用。

    Parameters
    ----------
    L_arr    : array-like, shape (T,) — 動的 CFD レバレッジ（未シフト）
    lev_arr  : array-like, shape (T,) — S2+LT2 合成レバレッジ（未シフト）
    oos_mask : np.ndarray(bool), shape (T,) — OOS 期間マスク
    assets   : dict — build_best_strategy_assets() の返値
    cfd_spread : float — 年率スプレッド

    Returns
    -------
    np.ndarray — OOS 期間の日次リターン配列
    """
    idx = assets['dates'].index

    r_nas = assets['close'].pct_change().fillna(0).values
    r_g2  = pd.Series(assets['gold_2x']).pct_change().fillna(0).values
    r_b3  = pd.Series(assets['bond_3x']).pct_change().fillna(0).values
    # sofr は np.ndarray (load_sofr が .values で返す)
    sofr  = np.asarray(assets['sofr'], dtype=float)

    # DELAY シフト（全期間配列）
    L_s   = pd.Series(np.asarray(L_arr,   dtype=float), index=idx).shift(DELAY).fillna(1.0).values
    lev_s = pd.Series(np.asarray(lev_arr, dtype=float), index=idx).shift(DELAY).fillna(0.0).values
    wn_s  = pd.Series(np.asarray(assets['wn_A'], dtype=float), index=idx).shift(DELAY).fillna(0.0).values
    wg_s  = pd.Series(np.asarray(assets['wg_A'], dtype=float), index=idx).shift(DELAY).fillna(0.0).values
    wb_s  = pd.Series(np.asarray(assets['wb_A'], dtype=float), index=idx).shift(DELAY).fillna(0.0).values

    # CFD リターン: L 倍 NASDAQ - (L-1) 分の借入コスト
    nas_ret = (L_s * r_nas
               - (L_s - 1.0) * (sofr + cfd_spread / TRADING_DAYS))
    daily   = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # OOS のみ抽出（ndarray bool マスクで高速）
    return daily[oos_mask]


def dist_summary(arr):
    """分布統計量 dict を返す（P5/P25/P50/P75/P95/mean/std）"""
    return {
        'P5':   float(np.quantile(arr, 0.05)),
        'P25':  float(np.quantile(arr, 0.25)),
        'P50':  float(np.quantile(arr, 0.50)),
        'P75':  float(np.quantile(arr, 0.75)),
        'P95':  float(np.quantile(arr, 0.95)),
        'mean': float(arr.mean()),
        'std':  float(arr.std(ddof=1)),
    }


def perm_verdict(p):
    if p < 0.05:
        return 'PASS'
    elif p < 0.10:
        return 'WARN'
    else:
        return 'FAIL'


def run_perm_loop(label, B, fn_perm_signal, B_PERM, B_BLOCK_LEN, assets, oos_mask, L_s2_values, lev_mod_arr):
    """汎用置換ループ（進捗計測付き）"""
    results = []
    t_start_loop = time.time()
    timing_10 = []

    for b in range(B_PERM):
        t0 = time.time()
        L_perm, lev_perm = fn_perm_signal(b, L_s2_values, lev_mod_arr, B_BLOCK_LEN)
        r_perm = compute_oos_returns_from_signals(L_perm, lev_perm, oos_mask, assets)
        results.append(sharpe_ann(r_perm))
        elapsed = time.time() - t0

        if b < 10:
            timing_10.append(elapsed)
            print(f'    iter {b}: {elapsed*1000:.2f} ms')
        if b % 100 == 0:
            elapsed_total = time.time() - t_start_loop
            print(f'  {label}: {b}/{B_PERM}  ({elapsed_total:.1f}s elapsed)')

    print(f'  {label}: Done. Total={time.time()-t_start_loop:.1f}s  avg_first10={np.mean(timing_10)*1000:.2f}ms')
    return np.array(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TODAY     = datetime.date.today()
    TODAY_STR = TODAY.strftime('%Y-%m-%d')
    TODAY_TAG = TODAY.strftime('%Y%m%d')

    print('=' * 60)
    print('3.3 Permutation Test（シグナル置換検定）')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Step 1: アセット取得と OOS 切り出し
    # ------------------------------------------------------------------
    print('\n[Step 1] Loading best strategy assets ...')
    assets = build_best_strategy_assets()

    dates      = assets['dates']
    close      = assets['close']
    lev_mod    = assets['lev_mod']
    L_s2       = assets['L_s2']          # pd.Series

    oos_mask = (dates >= OOS_START).values   # ndarray(bool)
    oos_end  = str(dates[oos_mask].iloc[-1].date())

    L_s2_values  = L_s2.values if hasattr(L_s2, 'values') else np.array(L_s2)
    lev_mod_arr  = np.array(lev_mod, dtype=float)

    print(f'  OOS: {OOS_START} ~ {oos_end}  (T_oos = {oos_mask.sum()} 日)')
    _L_valid = L_s2_values[~np.isnan(L_s2_values)]
    print(f'  L_s2  : [{_L_valid.min():.2f}, {_L_valid.max():.2f}]  median={np.median(_L_valid):.2f}  (nan={np.isnan(L_s2_values).sum()})')
    print(f'  lev_mod: [{lev_mod_arr.min():.2f}, {lev_mod_arr.max():.2f}]  median={np.median(lev_mod_arr):.2f}')

    # ------------------------------------------------------------------
    # Step 2: 観測 OOS Sharpe
    # ------------------------------------------------------------------
    print('\n[Step 2] Computing observed Sharpe ...')
    np.random.seed(RANDOM_SEED)

    r_obs = compute_oos_returns_from_signals(
        L_s2_values, lev_mod_arr, oos_mask, assets
    )
    sr_obs = sharpe_ann(r_obs)

    r_bh_oos = close.pct_change().fillna(0).values[oos_mask]
    sr_bh    = sharpe_ann(r_bh_oos)

    print(f'  Sharpe_OOS (observed) = {sr_obs:.4f}')
    print(f'  BH 1x Sharpe_OOS      = {sr_bh:.4f}')

    # ------------------------------------------------------------------
    # Step 3: 検定(a) L_s2 ブロックシャッフル
    # ------------------------------------------------------------------
    print('\n[Step 3] (a) L_s2 block shuffle permutation ...')
    results_a = []
    t_a = time.time()
    timing_a = []

    for b in range(B_PERM):
        t0 = time.time()
        L_perm = block_shuffle(L_s2_values, BLOCK_LEN)
        r_perm = compute_oos_returns_from_signals(L_perm, lev_mod_arr, oos_mask, assets)
        results_a.append(sharpe_ann(r_perm))
        dt = time.time() - t0
        if b < 10:
            timing_a.append(dt)
            print(f'    (a) iter {b:>3}: {dt*1000:.2f} ms')
        if b % 100 == 0:
            print(f'  (a) L_s2 block perm: {b}/{B_PERM}  ({time.time()-t_a:.1f}s)')

    results_a = np.array(results_a)
    p_a = float((results_a >= sr_obs).mean())
    print(f'  (a) Done.  p_a = {p_a:.4f}  avg_first10 = {np.mean(timing_a)*1000:.2f} ms')

    # ------------------------------------------------------------------
    # Step 4: 検定(b) L_s2 White-noise permutation
    # ------------------------------------------------------------------
    print('\n[Step 4] (b) L_s2 white-noise permutation (i.i.d. bootstrap) ...')
    results_b = []
    t_b = time.time()
    timing_b = []

    for b in range(B_PERM):
        t0 = time.time()
        L_perm = np.random.choice(L_s2_values, size=len(L_s2_values), replace=True)
        r_perm = compute_oos_returns_from_signals(L_perm, lev_mod_arr, oos_mask, assets)
        results_b.append(sharpe_ann(r_perm))
        dt = time.time() - t0
        if b < 10:
            timing_b.append(dt)
            print(f'    (b) iter {b:>3}: {dt*1000:.2f} ms')
        if b % 100 == 0:
            print(f'  (b) L_s2 white-noise perm: {b}/{B_PERM}  ({time.time()-t_b:.1f}s)')

    results_b = np.array(results_b)
    p_b = float((results_b >= sr_obs).mean())
    print(f'  (b) Done.  p_b = {p_b:.4f}  avg_first10 = {np.mean(timing_b)*1000:.2f} ms')

    # ------------------------------------------------------------------
    # Step 5: 検定(c) lev_mod ブロックシャッフル
    # ------------------------------------------------------------------
    print('\n[Step 5] (c) lev_mod block shuffle permutation ...')
    results_c = []
    t_c = time.time()
    timing_c = []

    for b in range(B_PERM):
        t0 = time.time()
        lev_perm = block_shuffle(lev_mod_arr, BLOCK_LEN)
        r_perm = compute_oos_returns_from_signals(L_s2_values, lev_perm, oos_mask, assets)
        results_c.append(sharpe_ann(r_perm))
        dt = time.time() - t0
        if b < 10:
            timing_c.append(dt)
            print(f'    (c) iter {b:>3}: {dt*1000:.2f} ms')
        if b % 100 == 0:
            print(f'  (c) lev_mod block perm: {b}/{B_PERM}  ({time.time()-t_c:.1f}s)')

    results_c = np.array(results_c)
    p_c = float((results_c >= sr_obs).mean())
    print(f'  (c) Done.  p_c = {p_c:.4f}  avg_first10 = {np.mean(timing_c)*1000:.2f} ms')

    # ------------------------------------------------------------------
    # Step 6: 判定
    # ------------------------------------------------------------------
    print('\n[Step 6] Verdict ...')
    verdict_a = perm_verdict(p_a)
    verdict_b_str = '(参考)'
    verdict_c = perm_verdict(p_c)

    if verdict_a == 'PASS' and verdict_c == 'PASS':
        overall = 'PASS'
    elif 'FAIL' in [verdict_a, verdict_c]:
        overall = 'FAIL'
    else:
        overall = 'WARN'

    print(f'  verdict_a = {verdict_a}  (p_a = {p_a:.4f})')
    print(f'  verdict_c = {verdict_c}  (p_c = {p_c:.4f})')
    print(f'  overall   = {overall}')

    # ------------------------------------------------------------------
    # Step 7: 分布統計量
    # ------------------------------------------------------------------
    ds_a = dist_summary(results_a)
    ds_b = dist_summary(results_b)
    ds_c = dist_summary(results_c)

    # ------------------------------------------------------------------
    # Output: Markdown
    # ------------------------------------------------------------------
    md_path = os.path.join(AUDIT_DIR, f'PERMUTATION_{TODAY_TAG}.md')

    def _fmt_ds_row(label, ds):
        return (
            f"| {label} "
            f"| {ds['P5']:.4f} "
            f"| {ds['P25']:.4f} "
            f"| {ds['P50']:.4f} "
            f"| {ds['P75']:.4f} "
            f"| {ds['P95']:.4f} "
            f"| {ds['mean']:.4f} "
            f"| {ds['std']:.4f} |"
        )

    md_lines = [
        '# 3.3 Permutation Test（シグナル置換検定）',
        '',
        f'- 生成日: {TODAY_STR}',
        '- 戦略: S2_VZGated + LT2-N750-k0.5-modeB',
        f'- 置換回数: B = {B_PERM}',
        f'- ブロック長: {BLOCK_LEN} 日（約3ヶ月）',
        f'- 乱数シード: {RANDOM_SEED}',
        f'- OOS 期間: {OOS_START} 〜 {oos_end}',
        '',
        '## 観測値',
        '',
        '| 指標 | 値 |',
        '|---|---|',
        f'| Sharpe_OOS (観測) | {sr_obs:.4f} |',
        f'| BH 1x Sharpe_OOS (参照) | {sr_bh:.4f} |',
        '',
        '## Permutation 検定結果',
        '',
        '| 検定 | 対象 | p値 | 判定 |',
        '|---|---|---|---|',
        f'| (a) L_s2 ブロックシャッフル | CFDレバ動的調整 | {p_a:.4f} | {verdict_a} |',
        f'| (b) L_s2 White-noise復元抽出 | 〃 (参考・i.i.d.) | {p_b:.4f} | {verdict_b_str} |',
        f'| (c) lev_mod ブロックシャッフル | S2+LT2合成シグナル | {p_c:.4f} | {verdict_c} |',
        '',
        '*判定: p < 0.05 → PASS / 0.05 ≤ p < 0.10 → WARN / p ≥ 0.10 → FAIL*',
        '',
        '## 置換分布サマリ',
        '',
        '| 検定 | P5 | P25 | 中央値 | P75 | P95 | 平均 | std |',
        '|---|---|---|---|---|---|---|---|',
        _fmt_ds_row('(a) L_s2 block', ds_a),
        _fmt_ds_row('(b) L_s2 white', ds_b),
        _fmt_ds_row('(c) lev_mod block', ds_c),
        '',
        '## 解釈',
        '',
        f'*置換分布の平均 ≈ BH 1x Sharpe ({sr_bh:.4f}) の場合、置換が正しく機能している証拠。*',
        '',
        f'*実際: (a) 平均={ds_a["mean"]:.4f}、(b) 平均={ds_b["mean"]:.4f}、(c) 平均={ds_c["mean"]:.4f}*',
        '',
        f'## 総合判定: {overall}',
        '',
        '*判定根拠: (a) かつ (c) が両方 PASS → PASS / いずれかが FAIL → FAIL / それ以外 → WARN*',
        '',
    ]

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f'\n[Output] {md_path}')

    # ------------------------------------------------------------------
    # Output: YAML
    # ------------------------------------------------------------------
    yaml_path = os.path.join(AUDIT_DIR, 'permutation_results.yaml')

    out_data = {
        'strategy': 'S2_VZGated+LT2-N750-k0.5-modeB',
        'generated': TODAY_STR,
        'oos_period': {
            'start': OOS_START,
            'end': oos_end,
        },
        'settings': {
            'B': B_PERM,
            'block_len': BLOCK_LEN,
            'seed': RANDOM_SEED,
        },
        'sr_observed': round(float(sr_obs), 6),
        'sr_bh1x_oos': round(float(sr_bh), 6),
        'tests': {
            'a_L_s2_block': {
                'p_value':   round(p_a, 6),
                'verdict':   verdict_a,
                'dist_mean': round(ds_a['mean'], 6),
                'dist_std':  round(ds_a['std'],  6),
                'dist_P5':   round(ds_a['P5'],   6),
                'dist_P25':  round(ds_a['P25'],  6),
                'dist_P50':  round(ds_a['P50'],  6),
                'dist_P75':  round(ds_a['P75'],  6),
                'dist_P95':  round(ds_a['P95'],  6),
            },
            'b_L_s2_whitenoise': {
                'p_value':   round(p_b, 6),
                'verdict':   '参考',
                'dist_mean': round(ds_b['mean'], 6),
                'dist_std':  round(ds_b['std'],  6),
                'dist_P5':   round(ds_b['P5'],   6),
                'dist_P25':  round(ds_b['P25'],  6),
                'dist_P50':  round(ds_b['P50'],  6),
                'dist_P75':  round(ds_b['P75'],  6),
                'dist_P95':  round(ds_b['P95'],  6),
            },
            'c_lev_mod_block': {
                'p_value':   round(p_c, 6),
                'verdict':   verdict_c,
                'dist_mean': round(ds_c['mean'], 6),
                'dist_std':  round(ds_c['std'],  6),
                'dist_P5':   round(ds_c['P5'],   6),
                'dist_P25':  round(ds_c['P25'],  6),
                'dist_P50':  round(ds_c['P50'],  6),
                'dist_P75':  round(ds_c['P75'],  6),
                'dist_P95':  round(ds_c['P95'],  6),
            },
        },
        'verdict': overall,
    }

    _dump_yaml(out_data, yaml_path)
    print(f'[Output] {yaml_path}')

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print(f'Sharpe_OOS (observed) : {sr_obs:.4f}')
    print(f'BH 1x Sharpe_OOS      : {sr_bh:.4f}')
    print()
    print(f'{"検定":<30} {"p値":>8} {"判定":>8}')
    print(f'{"(a) L_s2 block shuffle":<30} {p_a:>8.4f} {verdict_a:>8}')
    print(f'{"(b) L_s2 white-noise":<30} {p_b:>8.4f} {"(参考)":>8}')
    print(f'{"(c) lev_mod block shuffle":<30} {p_c:>8.4f} {verdict_c:>8}')
    print()
    print(f'置換分布平均: (a)={ds_a["mean"]:.4f}  (b)={ds_b["mean"]:.4f}  (c)={ds_c["mean"]:.4f}')
    print(f'BH参照値: {sr_bh:.4f}')
    print()
    print(f'総合判定: {overall}')
    print('=' * 60)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
