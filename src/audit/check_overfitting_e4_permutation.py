"""
check_overfitting_e4_permutation.py
E4戦略 Permutation Test（シグナル置換検定）
=============================================
S2_VZGated + LT2-N750 + E4 Regime k_lt (k_lo=0.1, k_hi=0.8, k_mid=0.5, vz_thr=0.7)
の予測力が本物か（NASDAQ右肩上がりに乗っているだけか）を検定する。

検定:
  (a) L_s2 ブロックシャッフル       — CFD動的レバレッジのアルファ
  (c) lev_mod_e4 ブロックシャッフル  — E4市場参加タイミングのアルファ
  (d) L_s2 + lev_mod_e4 同時シャッフル — 真のアルファ測定（KEY TEST）

OOS 期間のみで日次リターンを直接計算することで高速化。
B=1000, block_len=63, seed=42

出力:
  audit_results/E4_PERMUTATION_20260526.md
  audit_results/e4_permutation_results.yaml
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
build_e4_strategy_assets = _mod.build_e4_strategy_assets

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

from cfd_leverage_backtest import OOS_START, TRADING_DAYS, DELAY

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
B_PERM      = 1000
BLOCK_LEN   = 63      # 約3ヶ月
RANDOM_SEED = 42

AUDIT_DIR = os.path.join(BASE, 'audit_results')
os.makedirs(AUDIT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: sharpe
# ---------------------------------------------------------------------------

def sharpe_ann(r: np.ndarray) -> float:
    """年率 Sharpe (Rf=0)"""
    std = r.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(r.mean() / std * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------------------
# Helper: block shuffle
# ---------------------------------------------------------------------------

def block_shuffle(arr: np.ndarray, block_len: int, rng: np.random.RandomState) -> np.ndarray:
    """1次元配列をブロック長 block_len でシャッフルして返す（元長保持）。
    固定先頭インデックスでブロック分割してからシャッフル（長さ変化なし）。
    """
    arr = np.asarray(arr, dtype=float)
    T = len(arr)
    block_starts = list(range(0, T, block_len))
    perm = rng.permutation(len(block_starts))
    result = []
    for i in perm:
        start = block_starts[i]
        end = min(start + block_len, T)
        result.extend(arr[start:end])
    return np.array(result[:T])


def block_shuffle_synced(arr_a: np.ndarray, arr_b: np.ndarray,
                         block_len: int, rng: np.random.RandomState):
    """arr_a と arr_b を同じブロック構造でシャッフル（同期シャッフル、元長保持）"""
    arr_a = np.asarray(arr_a, dtype=float)
    arr_b = np.asarray(arr_b, dtype=float)
    T = len(arr_a)
    block_starts = list(range(0, T, block_len))
    perm = rng.permutation(len(block_starts))
    res_a, res_b = [], []
    for i in perm:
        start = block_starts[i]
        end = min(start + block_len, T)
        res_a.extend(arr_a[start:end])
        res_b.extend(arr_b[start:end])
    return np.array(res_a[:T]), np.array(res_b[:T])


# ---------------------------------------------------------------------------
# OOS Sharpe fast computation
# ---------------------------------------------------------------------------

def compute_oos_sharpe(L_arr, lev_arr, wn_arr, wg_arr, wb_arr,
                       r_nas, r_g2, r_b3, idx, oos_mask_arr):
    """
    OOSのSharpeを高速計算（NAVを都度再構築しない）。

    DELAY shiftを適用してOOSのみで日次リターンを計算する。
    ポジションコストは簡略化（cfd_spreadはゼロ近似、Sharpe比較が目的のため）。
    """
    L_shifted = pd.Series(np.asarray(L_arr, dtype=float), index=idx).shift(DELAY).fillna(1.0).values
    lev_s     = pd.Series(np.asarray(lev_arr, dtype=float), index=idx).shift(DELAY).fillna(0.0).values
    wn_s      = pd.Series(np.asarray(wn_arr, dtype=float), index=idx).shift(DELAY).fillna(0.0).values
    wg_s      = pd.Series(np.asarray(wg_arr, dtype=float), index=idx).shift(DELAY).fillna(0.0).values
    wb_s      = pd.Series(np.asarray(wb_arr, dtype=float), index=idx).shift(DELAY).fillna(0.0).values

    # ポジションコストは簡略化（Sharpe比較が目的のため cfd_spread は省略）
    nas_ret = L_shifted * r_nas
    daily   = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # OOSマスク適用
    r_oos = daily[oos_mask_arr]

    if r_oos.std(ddof=1) < 1e-12:
        return 0.0
    return float(r_oos.mean() / r_oos.std(ddof=1) * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------------------
# Distribution summary
# ---------------------------------------------------------------------------

def dist_summary(arr: np.ndarray) -> dict:
    """分布統計量 dict を返す"""
    return {
        'dist_mean': float(arr.mean()),
        'dist_std':  float(arr.std(ddof=1)),
        'dist_P5':   float(np.quantile(arr, 0.05)),
        'dist_P25':  float(np.quantile(arr, 0.25)),
        'dist_P50':  float(np.quantile(arr, 0.50)),
        'dist_P75':  float(np.quantile(arr, 0.75)),
        'dist_P95':  float(np.quantile(arr, 0.95)),
    }


def perm_verdict(p: float) -> str:
    if p < 0.05:
        return 'PASS'
    elif p < 0.10:
        return 'WARN'
    else:
        return 'FAIL'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TODAY     = datetime.date.today()
    TODAY_STR = TODAY.strftime('%Y-%m-%d')
    TODAY_TAG = TODAY.strftime('%Y%m%d')

    print('=' * 60)
    print('E4戦略 Permutation Test（シグナル置換検定）')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Step 1: E4アセット取得と OOS 切り出し
    # ------------------------------------------------------------------
    print('\n[Step 1] Loading E4 strategy assets ...')
    assets = build_e4_strategy_assets(force_rebuild=False)

    dates         = assets['dates']
    close         = assets['close']
    L_s2          = assets['L_s2']          # pd.Series
    lev_mod_e4    = assets['lev_mod_e4']    # np.ndarray
    wn_A          = assets['wn_A']
    wg_A          = assets['wg_A']
    wb_A          = assets['wb_A']
    gold_2x       = assets['gold_2x']
    bond_3x       = assets['bond_3x']

    idx = dates.index

    # OOS マスク
    oos_mask_arr = (dates >= OOS_START).values   # ndarray(bool)
    oos_end_str  = str(dates[oos_mask_arr].iloc[-1].date())
    oos_idx      = np.where(dates >= OOS_START)[0]

    # 共通配列
    L_s2_arr       = L_s2.values if hasattr(L_s2, 'values') else np.array(L_s2)
    lev_mod_e4_arr = np.asarray(lev_mod_e4, dtype=float)
    wn_arr         = np.asarray(wn_A, dtype=float)
    wg_arr         = np.asarray(wg_A, dtype=float)
    wb_arr         = np.asarray(wb_A, dtype=float)

    # 日次リターン配列
    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3  = pd.Series(bond_3x).pct_change().fillna(0).values

    print(f'  OOS: {OOS_START} ~ {oos_end_str}  (T_oos = {oos_mask_arr.sum()} 日)')
    _L_valid = L_s2_arr[~np.isnan(L_s2_arr)]
    print(f'  L_s2      : [{_L_valid.min():.2f}, {_L_valid.max():.2f}]  median={np.median(_L_valid):.2f}  (nan={np.isnan(L_s2_arr).sum()})')
    print(f'  lev_mod_e4: [{lev_mod_e4_arr.min():.2f}, {lev_mod_e4_arr.max():.2f}]  median={np.median(lev_mod_e4_arr):.2f}')

    # ------------------------------------------------------------------
    # Step 2: 観測 OOS Sharpe
    # ------------------------------------------------------------------
    print('\n[Step 2] Computing observed Sharpe ...')

    sr_obs = compute_oos_sharpe(
        L_s2_arr, lev_mod_e4_arr, wn_arr, wg_arr, wb_arr,
        r_nas, r_g2, r_b3, idx, oos_mask_arr
    )

    r_bh_oos = close.pct_change().fillna(0).values[oos_mask_arr]
    sr_bh    = sharpe_ann(r_bh_oos)

    print(f'  Sharpe_OOS (observed E4) = {sr_obs:.4f}')
    print(f'  BH 1x Sharpe_OOS         = {sr_bh:.4f}')

    # ------------------------------------------------------------------
    # Step 3: 検定(a) L_s2 ブロックシャッフル（lev_mod_e4は固定）
    # ------------------------------------------------------------------
    print('\n[Step 3] (a) L_s2 block shuffle permutation (lev_mod_e4 fixed) ...')
    results_a = []
    t_a = time.time()
    timing_a = []

    rng_a = np.random.RandomState(RANDOM_SEED)
    for b in range(B_PERM):
        t0 = time.time()
        L_perm = block_shuffle(L_s2_arr, BLOCK_LEN, rng_a)
        sr_perm = compute_oos_sharpe(
            L_perm, lev_mod_e4_arr, wn_arr, wg_arr, wb_arr,
            r_nas, r_g2, r_b3, idx, oos_mask_arr
        )
        results_a.append(sr_perm)
        dt = time.time() - t0
        if b < 10:
            timing_a.append(dt)
            print(f'    (a) iter {b:>3}: {dt*1000:.2f} ms')
        if b % 100 == 0:
            print(f'  (a) L_s2 block perm: {b}/{B_PERM}  ({time.time()-t_a:.1f}s elapsed)')

    results_a = np.array(results_a)
    p_a = float((results_a >= sr_obs).mean())
    verdict_a = perm_verdict(p_a)
    ds_a = dist_summary(results_a)
    print(f'  (a) Done.  p_a={p_a:.4f} [{verdict_a}]  avg_first10={np.mean(timing_a)*1000:.2f}ms  total={time.time()-t_a:.1f}s')

    # ------------------------------------------------------------------
    # Step 4: 検定(c) lev_mod_e4 ブロックシャッフル（L_s2は固定）
    # ------------------------------------------------------------------
    print('\n[Step 4] (c) lev_mod_e4 block shuffle permutation (L_s2 fixed) ...')
    results_c = []
    t_c = time.time()
    timing_c = []

    rng_c = np.random.RandomState(RANDOM_SEED + 1)
    for b in range(B_PERM):
        t0 = time.time()
        lev_perm = block_shuffle(lev_mod_e4_arr, BLOCK_LEN, rng_c)
        sr_perm = compute_oos_sharpe(
            L_s2_arr, lev_perm, wn_arr, wg_arr, wb_arr,
            r_nas, r_g2, r_b3, idx, oos_mask_arr
        )
        results_c.append(sr_perm)
        dt = time.time() - t0
        if b < 10:
            timing_c.append(dt)
            print(f'    (c) iter {b:>3}: {dt*1000:.2f} ms')
        if b % 100 == 0:
            print(f'  (c) lev_mod_e4 block perm: {b}/{B_PERM}  ({time.time()-t_c:.1f}s elapsed)')

    results_c = np.array(results_c)
    p_c = float((results_c >= sr_obs).mean())
    verdict_c = perm_verdict(p_c)
    ds_c = dist_summary(results_c)
    print(f'  (c) Done.  p_c={p_c:.4f} [{verdict_c}]  avg_first10={np.mean(timing_c)*1000:.2f}ms  total={time.time()-t_c:.1f}s')

    # ------------------------------------------------------------------
    # Step 5: 検定(d) L_s2 + lev_mod_e4 同時ブロックシャッフル（KEY TEST）
    # ------------------------------------------------------------------
    print('\n[Step 5] (d) L_s2 + lev_mod_e4 SYNCHRONIZED block shuffle (KEY TEST) ...')
    results_d = []
    t_d = time.time()
    timing_d = []

    rng_d = np.random.RandomState(RANDOM_SEED + 2)
    for b in range(B_PERM):
        t0 = time.time()
        L_perm, lev_perm = block_shuffle_synced(
            L_s2_arr, lev_mod_e4_arr, BLOCK_LEN, rng_d
        )
        sr_perm = compute_oos_sharpe(
            L_perm, lev_perm, wn_arr, wg_arr, wb_arr,
            r_nas, r_g2, r_b3, idx, oos_mask_arr
        )
        results_d.append(sr_perm)
        dt = time.time() - t0
        if b < 10:
            timing_d.append(dt)
            print(f'    (d) iter {b:>3}: {dt*1000:.2f} ms')
        if b % 100 == 0:
            print(f'  (d) simultaneous block perm: {b}/{B_PERM}  ({time.time()-t_d:.1f}s elapsed)')

    results_d = np.array(results_d)
    p_d = float((results_d >= sr_obs).mean())
    verdict_d = perm_verdict(p_d)
    ds_d = dist_summary(results_d)
    print(f'  (d) Done.  p_d={p_d:.4f} [{verdict_d}]  avg_first10={np.mean(timing_d)*1000:.2f}ms  total={time.time()-t_d:.1f}s')

    # ------------------------------------------------------------------
    # Step 6: 判定
    # ------------------------------------------------------------------
    print('\n[Step 6] Verdict ...')
    # 主判定は (d)
    overall = verdict_d
    print(f'  verdict_a (L_s2 block)         = {verdict_a}  (p={p_a:.4f})')
    print(f'  verdict_c (lev_mod_e4 block)   = {verdict_c}  (p={p_c:.4f})')
    print(f'  verdict_d (simultaneous, KEY)  = {verdict_d}  (p={p_d:.4f})')
    print(f'  overall (主判定=d)             = {overall}')

    # ------------------------------------------------------------------
    # Output: YAML
    # ------------------------------------------------------------------
    yaml_path = os.path.join(AUDIT_DIR, 'e4_permutation_results.yaml')

    out_data = {
        'strategy': 'S2_VZGated+LT2-N750+E4_RegimeKlt',
        'generated': TODAY_STR,
        'oos_period': {
            'start': str(OOS_START),
            'end': oos_end_str,
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
                **{k: round(v, 6) for k, v in ds_a.items()},
            },
            'c_lev_mod_e4_block': {
                'p_value':   round(p_c, 6),
                'verdict':   verdict_c,
                **{k: round(v, 6) for k, v in ds_c.items()},
            },
            'd_simultaneous_block': {
                'p_value':   round(p_d, 6),
                'verdict':   verdict_d,
                **{k: round(v, 6) for k, v in ds_d.items()},
            },
        },
        'verdict': overall,
    }

    _dump_yaml(out_data, yaml_path)
    print(f'\n[Output] {yaml_path}')

    # ------------------------------------------------------------------
    # Output: Markdown
    # ------------------------------------------------------------------
    md_path = os.path.join(AUDIT_DIR, f'E4_PERMUTATION_{TODAY_TAG}.md')

    def _fmt_ds_row(label, p, v, ds):
        return (
            f"| {label} "
            f"| {p:.4f} "
            f"| **{v}** "
            f"| {ds['dist_P5']:.4f} "
            f"| {ds['dist_P25']:.4f} "
            f"| {ds['dist_P50']:.4f} "
            f"| {ds['dist_P75']:.4f} "
            f"| {ds['dist_P95']:.4f} "
            f"| {ds['dist_mean']:.4f} "
            f"| {ds['dist_std']:.4f} |"
        )

    ref_note = '旧戦略: (a)p=0.248 FAIL, (c)p=0.055 WARN, (d)未実施'

    md_lines = [
        '# E4戦略 Permutation 過学習検定レポート',
        '',
        f'**戦略名:** S2_VZGated + LT2-N750 + E4 Regime k_lt (k_lo=0.1, k_hi=0.8, k_mid=0.5, vz_thr=0.7)',
        f'**OOS 期間:** {OOS_START} 〜 {oos_end_str}（約4.9年）',
        f'**作成日:** {TODAY_STR}',
        f'**Permutation反復:** B={B_PERM}, block_len={BLOCK_LEN}',
        '',
        f'主眼: テスト(d) L_s2+lev_mod_e4 同時置換が「真のアルファ測定」。',
        f'{ref_note}',
        '',
        '## 観測値',
        '',
        '| 指標 | 値 |',
        '|---|---|',
        f'| Sharpe_OOS E4 (観測) | {sr_obs:.4f} |',
        f'| BH 1x Sharpe_OOS (参照) | {sr_bh:.4f} |',
        '',
        '## Permutation 検定結果',
        '',
        '| 検定 | 対象 | p値 | 判定 |',
        '|---|---|---|---|',
        f'| (a) L_s2 ブロックシャッフル | CFD動的レバレッジ（lev_mod_e4固定） | {p_a:.4f} | {verdict_a} |',
        f'| (c) lev_mod_e4 ブロックシャッフル | E4市場参加タイミング（L_s2固定） | {p_c:.4f} | {verdict_c} |',
        f'| **(d) L_s2+lev_mod_e4 同時シャッフル** | **真のアルファ測定（KEY TEST）** | **{p_d:.4f}** | **{verdict_d}** |',
        '',
        '*判定: p < 0.05 → PASS / 0.05 ≤ p < 0.10 → WARN / p ≥ 0.10 → FAIL*',
        '',
        '## 置換分布サマリ',
        '',
        '| 検定 | p値 | 判定 | P5 | P25 | 中央値 | P75 | P95 | 平均 | std |',
        '|---|---|---|---|---|---|---|---|---|---|',
        _fmt_ds_row('(a) L_s2 block', p_a, verdict_a, ds_a),
        _fmt_ds_row('(c) lev_mod_e4 block', p_c, verdict_c, ds_c),
        _fmt_ds_row('(d) simultaneous (KEY)', p_d, verdict_d, ds_d),
        '',
        '## 解釈',
        '',
        f'置換分布の平均がBH 1x Sharpe ({sr_bh:.4f}) に近い場合、置換が正しく機能している証拠。',
        '',
        f'実際: (a)平均={ds_a["dist_mean"]:.4f}  (c)平均={ds_c["dist_mean"]:.4f}  (d)平均={ds_d["dist_mean"]:.4f}',
        '',
        f'## 総合判定: {overall}',
        '',
        f'*(d) L_s2+lev_mod_e4 同時置換 p={p_d:.4f} が主判定。p < 0.05 → 真のアルファあり PASS*',
        '',
        '### 個別判定根拠',
        f'- **(a)** L_s2のみ置換: p={p_a:.4f} → {verdict_a}（CFD動的レバレッジの寄与）',
        f'- **(c)** lev_mod_e4のみ置換: p={p_c:.4f} → {verdict_c}（E4市場参加タイミングの寄与）',
        f'- **(d)** 同時置換（KEY）: p={p_d:.4f} → {verdict_d}（真のアルファ測定）',
        '',
    ]

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f'[Output] {md_path}')

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('E4戦略 Permutation Test 結果サマリ')
    print('=' * 60)
    print(f'Sharpe_OOS (E4 observed) : {sr_obs:.4f}')
    print(f'BH 1x Sharpe_OOS         : {sr_bh:.4f}')
    print()
    print(f'{"検定":<35} {"p値":>8} {"判定":>8}')
    print(f'{"(a) L_s2 block shuffle":<35} {p_a:>8.4f} {verdict_a:>8}')
    print(f'{"(c) lev_mod_e4 block shuffle":<35} {p_c:>8.4f} {verdict_c:>8}')
    print(f'{"(d) simultaneous block (KEY)":<35} {p_d:>8.4f} {verdict_d:>8}')
    print()
    print(f'置換分布平均: (a)={ds_a["dist_mean"]:.4f}  (c)={ds_c["dist_mean"]:.4f}  (d)={ds_d["dist_mean"]:.4f}')
    print(f'BH参照値: {sr_bh:.4f}')
    print()
    print(f'総合判定 (主判定=d): {overall}')
    print('=' * 60)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
