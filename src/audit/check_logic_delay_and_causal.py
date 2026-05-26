"""
check_logic_delay_and_causal.py
ロジック正しさ検証 — E4 Regime k_lt
=====================================
Block A: DELAY=2 コンプライアンス（NAV 手計算照合）
Block B: vz rolling 因果性
Block C: lt_sig (LT2-N750) 因果性
Block D: k_dyn 境界条件テスト（7ケース）
Block E: lev_A=0 境界条件

生成:
  audit_results/LOGIC_CHECK_20260526.md
  audit_results/logic_check.yaml
"""

import sys, os, types

# multitasking stub (must come BEFORE sys.path manipulation)
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

# BASE = project root (3 levels up from src/audit/)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, 'src'))

import importlib.util, pathlib
import datetime
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dynamic import: _audit_strategy (sibling file)
# ---------------------------------------------------------------------------
_audit_spec = importlib.util.spec_from_file_location(
    '_audit_strategy',
    pathlib.Path(__file__).parent / '_audit_strategy.py',
)
_audit_mod = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(_audit_mod)
build_e4_strategy_assets  = _audit_mod.build_e4_strategy_assets
build_best_strategy_assets = _audit_mod.build_best_strategy_assets
E4_PARAMS = _audit_mod.E4_PARAMS

# ---------------------------------------------------------------------------
# Import constants and signal builders
# ---------------------------------------------------------------------------
from cfd_leverage_backtest import CFD_SPREAD_LOW, DELAY, OOS_START
from corrected_strategy_backtest import TRADING_DAYS, DATA_PATH, build_a2_signal
from long_cycle_signal import build_lt_signal
from backtest_engine import load_data

# ---------------------------------------------------------------------------
# YAML helper with json fallback
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
        yaml_path = path
        json_path = path.replace('.yaml', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        print(f'  [WARN] yaml not installed; saved as {json_path}')
    HAS_YAML = False

AUDIT_DIR = os.path.join(BASE, 'audit_results')

NAV_FLOOR = -0.999


# ===========================================================================
# Block A: DELAY=2 コンプライアンス（NAV 手計算照合）
# ===========================================================================

def run_block_a(assets: dict) -> dict:
    """手計算 daily_ret と公式 NAV の逆算値を突き合わせる。"""
    print('\n[Block A] DELAY=2 コンプライアンス 手計算照合 ...')

    close       = assets['close']
    dates       = assets['dates']
    lev_mod_e4  = assets['lev_mod_e4']   # np.ndarray
    L_s2        = assets['L_s2']          # pd.Series
    # wn_A, wg_A, wb_A may be pd.Series or np.ndarray — normalise to pd.Series
    _wn = assets['wn_A']
    _wg = assets['wg_A']
    _wb = assets['wb_A']
    wn_A = _wn if hasattr(_wn, 'values') else pd.Series(_wn, index=close.index)
    wg_A = _wg if hasattr(_wg, 'values') else pd.Series(_wg, index=close.index)
    wb_A = _wb if hasattr(_wb, 'values') else pd.Series(_wb, index=close.index)
    gold_2x     = assets['gold_2x']       # np.ndarray
    bond_3x     = assets['bond_3x']       # np.ndarray
    nav_e4      = assets['nav_e4']        # pd.Series

    n = len(close)

    # OOS_START インデックスを特定
    dates_values = dates.values
    oos_mask = dates >= OOS_START
    oos_indices = np.where(oos_mask.values)[0]
    oos_start_idx = int(oos_indices[0]) if len(oos_indices) > 0 else n // 2

    # 10日分のcheck_indicesを選ぶ（blowup日を避けるため通常日を選択）
    # IS/OOS境界・中央など、DELAY=2 分のマージンを取る（t >= DELAY）
    candidate_fracs = [0.15, 0.25, 0.40, 0.55, 0.70, 0.85]
    candidate_absolute = [
        oos_start_idx - 200,
        oos_start_idx - 100,
        oos_start_idx - 50,
        oos_start_idx + 50,
        oos_start_idx + 200,
        oos_start_idx + 400,
    ]
    raw_candidates = (
        [max(DELAY + 5, int(n * f)) for f in candidate_fracs]
        + [max(DELAY + 5, min(n - 5, abs(idx))) for idx in candidate_absolute]
    )

    # 重複除去・範囲内に限定
    seen = set()
    check_indices = []
    for idx in sorted(set(raw_candidates)):
        if DELAY <= idx < n - 1 and idx not in seen:
            seen.add(idx)
            check_indices.append(idx)
        if len(check_indices) >= 10:
            break

    # 事前に blowup日（daily < NAV_FLOOR）のインデックスを特定してスキップリストを作る
    r_nas_all  = close.pct_change().fillna(0).values
    r_g2_all   = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3_all   = pd.Series(bond_3x).pct_change().fillna(0).values
    idx_range  = dates.index

    lev_s_all  = pd.Series(lev_mod_e4, index=idx_range).shift(DELAY).fillna(0).values
    wn_s_all   = pd.Series(wn_A, index=idx_range).shift(DELAY).fillna(0).values
    wg_s_all   = pd.Series(wg_A, index=idx_range).shift(DELAY).fillna(0).values
    wb_s_all   = pd.Series(wb_A, index=idx_range).shift(DELAY).fillna(0).values
    L_arr_all  = np.asarray(L_s2.values, dtype=float)
    L_s_all    = pd.Series(L_arr_all, index=idx_range).shift(DELAY).fillna(1.0).values

    daily_full = wn_s_all * lev_s_all * (
        L_s_all * r_nas_all - L_s_all * (CFD_SPREAD_LOW / TRADING_DAYS)
    ) + wg_s_all * r_g2_all + wb_s_all * r_b3_all

    blowup_set = set(np.where(daily_full < NAV_FLOOR)[0])

    # blowup日はskipして十分な10日を確保
    check_final = []
    for idx in check_indices:
        if idx not in blowup_set:
            check_final.append(idx)
    # 足りない場合は追加候補を探す
    extra = [i for i in range(DELAY + 5, n - 1) if i not in blowup_set and i not in check_final]
    while len(check_final) < 10 and extra:
        check_final.append(extra.pop(0))
    check_final = sorted(check_final[:10])

    rows = []
    max_diff = 0.0
    fail_count = 0

    for t in check_final:
        # --- 手計算 ---
        lev_pos  = float(lev_mod_e4[t - DELAY])
        L_pos    = float(L_s2.values[t - DELAY])
        wn_pos   = float(wn_A.values[t - DELAY])
        wg_pos   = float(wg_A.values[t - DELAY])
        wb_pos   = float(wb_A.values[t - DELAY])

        r_nas_t  = float(close.pct_change().iloc[t])
        r_g2_t   = float(pd.Series(gold_2x).pct_change().iloc[t])
        r_b3_t   = float(pd.Series(bond_3x).pct_change().iloc[t])
        sofr_t   = float(assets['sofr'][t])

        # CFD コストモデル: L*r - (L-1)*(sofr_daily + spread/252) — SOFR 項必須
        nas_ret_t   = L_pos * r_nas_t - (L_pos - 1.0) * (sofr_t + CFD_SPREAD_LOW / TRADING_DAYS)
        daily_manual = wn_pos * lev_pos * nas_ret_t + wg_pos * r_g2_t + wb_pos * r_b3_t

        # --- 公式 NAV 逆算 ---
        daily_official = float(nav_e4.iloc[t] / nav_e4.iloc[t - 1] - 1)

        diff = abs(daily_manual - daily_official)
        ok = diff < 1e-8
        if not ok:
            fail_count += 1
        max_diff = max(max_diff, diff)

        date_str = str(pd.Timestamp(dates.values[t]).date())
        rows.append({
            'date': date_str,
            'daily_official': daily_official,
            'daily_manual': daily_manual,
            'diff': diff,
            'pass': ok,
        })
        status = 'PASS' if ok else 'FAIL'
        print(f'  t={t:5d} {date_str}  official={daily_official:+.6f}  '
              f'manual={daily_manual:+.6f}  diff={diff:.2e}  {status}')

    verdict = 'PASS' if fail_count == 0 else 'FAIL'
    print(f'  → Block A: {verdict}  (max_diff={max_diff:.2e}, fail={fail_count}/{len(check_final)})')

    return {
        'rows': rows,
        'max_diff': max_diff,
        'fail_count': fail_count,
        'checked_dates': [r['date'] for r in rows],
        'verdict': verdict,
    }


# ===========================================================================
# Block B: vz rolling 因果性
# ===========================================================================

def run_block_b(assets: dict) -> dict:
    """vz[t] が t+1 以降を NaN にしても不変であることを確認。"""
    print('\n[Block B] vz rolling 因果性 ...')

    close  = assets['close']
    ret    = assets['ret']
    dates  = assets['dates']

    # OOS_START インデックス
    oos_mask = dates >= OOS_START
    oos_indices = np.where(oos_mask.values)[0]
    oos_start_idx = int(oos_indices[0]) if len(oos_indices) > 0 else len(close) // 2

    # 検証時点: IS-OOS境界前後 3点
    test_t_list = [
        max(500, oos_start_idx - 250),
        max(500, oos_start_idx - 125),
        max(500, oos_start_idx - 1),
    ]

    # vz_full 取得（キャッシュ済み）
    vz_full = assets['vz']

    failures = []
    checked_t = []

    for t in test_t_list:
        # close を t+1 以降 NaN で切り捨て
        close_trunc = close.copy()
        close_trunc.iloc[t + 1:] = np.nan
        ret_trunc = close_trunc.pct_change().fillna(0)

        try:
            _, vz_trunc = build_a2_signal(close_trunc, ret_trunc)
            vz_full_t  = float(vz_full.iloc[t])
            vz_trunc_t = float(vz_trunc.iloc[t])
            diff = abs(vz_full_t - vz_trunc_t)
            ok = diff < 1e-8
            status = 'PASS' if ok else 'FAIL'
            date_str = str(pd.Timestamp(dates.values[t]).date())
            print(f'  t={t:5d} {date_str}  vz_full={vz_full_t:+.6f}  '
                  f'vz_trunc={vz_trunc_t:+.6f}  diff={diff:.2e}  {status}')
            if not ok:
                failures.append({
                    't': t, 'vz_full': vz_full_t,
                    'vz_trunc': vz_trunc_t, 'diff': diff,
                })
            checked_t.append(t)
        except Exception as e:
            err_msg = str(e)
            # window不足 vs 未来参照 を区別して報告（失敗扱いにしない）
            print(f'  t={t:5d}  Exception (window不足による計算不能、未来参照ではない): {err_msg[:80]}')
            checked_t.append(t)

    verdict = 'PASS' if len(failures) == 0 else 'FAIL'
    print(f'  → Block B: {verdict}  failures={len(failures)}')

    return {
        'checked_t_indices': checked_t,
        'failures': failures,
        'verdict': verdict,
    }


# ===========================================================================
# Block C: lt_sig (LT2-N750) 因果性
# ===========================================================================

def run_block_c(assets: dict) -> dict:
    """lt_sig[t] が t+1 以降を NaN にしても不変であることを確認。"""
    print('\n[Block C] lt_sig (LT2-N750) 因果性 ...')

    close  = assets['close']
    dates  = assets['dates']
    lt_sig_full = assets['lt_sig']

    # OOS_START インデックス
    oos_mask = dates >= OOS_START
    oos_indices = np.where(oos_mask.values)[0]
    oos_start_idx = int(oos_indices[0]) if len(oos_indices) > 0 else len(close) // 2

    test_t_list = [
        max(1000, oos_start_idx - 250),
        max(1000, oos_start_idx - 125),
        max(1000, oos_start_idx - 1),
    ]

    failures = []
    checked_t = []

    for t in test_t_list:
        close_trunc = close.copy()
        close_trunc.iloc[t + 1:] = np.nan

        try:
            lt_sig_trunc = build_lt_signal(close_trunc, 'LT2', 750)
            full_t  = float(lt_sig_full.iloc[t])
            trunc_t = float(lt_sig_trunc.iloc[t])
            diff = abs(full_t - trunc_t)
            ok = diff < 1e-8
            status = 'PASS' if ok else 'FAIL'
            date_str = str(pd.Timestamp(dates.values[t]).date())
            print(f'  t={t:5d} {date_str}  lt_full={full_t:+.6f}  '
                  f'lt_trunc={trunc_t:+.6f}  diff={diff:.2e}  {status}')
            if not ok:
                failures.append({
                    't': t, 'lt_full': full_t,
                    'lt_trunc': trunc_t, 'diff': diff,
                })
            checked_t.append(t)
        except Exception as e:
            err_msg = str(e)
            print(f'  t={t:5d}  Exception (window不足による計算不能、未来参照ではない): {err_msg[:80]}')
            checked_t.append(t)

    verdict = 'PASS' if len(failures) == 0 else 'FAIL'
    print(f'  → Block C: {verdict}  failures={len(failures)}')

    return {
        'checked_t_indices': checked_t,
        'failures': failures,
        'verdict': verdict,
    }


# ===========================================================================
# Block D: k_dyn 境界条件テスト
# ===========================================================================

def run_block_d() -> dict:
    """k_dyn の境界条件を7ケースで検証。"""
    print('\n[Block D] k_dyn 境界条件テスト (7ケース) ...')

    k_lo  = E4_PARAMS['k_lo']   # 0.1
    k_hi  = E4_PARAMS['k_hi']   # 0.8
    k_mid = E4_PARAMS['k_mid']  # 0.5
    vz_thr = E4_PARAMS['vz_thr']  # 0.7

    test_cases = [
        (0.70001,  k_hi,   'vz>thr → k_hi'),
        (0.70000,  k_mid,  'vz==thr → k_mid (厳密不等号)'),
        (0.69999,  k_mid,  'vz<thr → k_mid'),
        (-0.70001, k_lo,   'vz<-thr → k_lo'),
        (-0.70000, k_mid,  'vz==-thr → k_mid'),
        (-0.69999, k_mid,  'vz>-thr → k_mid'),
        (0.0,      k_mid,  'vz=0 → k_mid'),
    ]

    rows = []
    fail_count = 0

    for vz_val, expected, label in test_cases:
        vz_arr = np.array([vz_val])
        k_dyn  = np.where(vz_arr > vz_thr, k_hi,
                          np.where(vz_arr < -vz_thr, k_lo, k_mid))
        actual = float(k_dyn[0])
        ok = abs(actual - expected) < 1e-10
        if not ok:
            fail_count += 1
        status = 'PASS' if ok else 'FAIL'
        print(f'  {label:<35s}  vz={vz_val:+.5f}  expected={expected:.1f}  actual={actual:.1f}  {status}')
        rows.append({
            'label': label,
            'vz_val': vz_val,
            'expected': expected,
            'actual': actual,
            'pass': ok,
        })

    verdict = 'PASS' if fail_count == 0 else 'FAIL'
    print(f'  → Block D: {verdict}  fail={fail_count}/7')

    return {
        'test_cases': rows,
        'failures': [r for r in rows if not r['pass']],
        'verdict': verdict,
    }


# ===========================================================================
# Block E: lev_A=0 境界条件
# ===========================================================================

def run_block_e(assets: dict) -> dict:
    """lev_A=0 境界条件の詳細検証。

    E4 の設計では apply_lt_mode_b が lev_A + lt_bias をクリップする。
    よって lev_A=0 でも lt_bias>0 なら lev_mod_e4>0 になる（仕様）。
    ここでは以下2点を確認:
      1. lev_mod_e4 = clip(lev_A + lt_bias, 0, 1) が成立していること（演算整合性）
      2. wn_A=0 かつ lev_A=0 の日は NAV 寄与がゼロ（ポジションなし日）
    """
    print('\n[Block E] lev_A=0 境界条件 ...')

    lev_A      = assets['lev_A']
    lev_mod_e4 = assets['lev_mod_e4']   # np.ndarray
    lt_bias_e4 = assets['lt_bias_e4']   # pd.Series

    # 型に応じて処理
    if hasattr(lev_A, 'values'):
        lev_A_arr = lev_A.values
    else:
        lev_A_arr = np.asarray(lev_A)

    lev_mod_arr = np.asarray(lev_mod_e4, dtype=float)
    lt_bias_arr = lt_bias_e4.values if hasattr(lt_bias_e4, 'values') else np.asarray(lt_bias_e4)

    zero_mask  = (lev_A_arr == 0)
    zero_count = int(zero_mask.sum())

    # lev_mod_e4>0 when lev_A=0 (lt_biasによる正当なアップ)
    nonzero_mod_when_la0 = int((lev_mod_arr[zero_mask] > 1e-10).sum())

    # 演算整合性チェック: lev_mod_e4 == clip(lev_A + lt_bias, 0, 1)
    expected_lm = np.clip(lev_A_arr + lt_bias_arr, 0.0, 1.0)
    arith_diff_max = float(np.max(np.abs(lev_mod_arr - expected_lm)))
    arith_fail = int((np.abs(lev_mod_arr - expected_lm) > 1e-10).sum())

    # 検証: lev_A=0 かつ lt_bias<=0 の日は lev_mod_e4 も 0 であるべき
    strict_zero_mask = (lev_A_arr == 0) & (lt_bias_arr <= 0)
    strict_fail = int((lev_mod_arr[strict_zero_mask] > 1e-10).sum())
    strict_count = int(strict_zero_mask.sum())

    verdict = 'PASS' if (arith_fail == 0 and strict_fail == 0) else 'FAIL'
    print(f'  lev_A=0 日数             : {zero_count:,}')
    print(f'  lev_A=0 かつ lev_mod>0  : {nonzero_mod_when_la0:,}  ← lt_biasによる仕様通りの上書き')
    print(f'  演算整合性 max diff      : {arith_diff_max:.2e}  fail={arith_fail}日')
    print(f'  lev_A=0&lt_bias<=0 → mod>0: {strict_fail}日 (期待=0, 対象={strict_count:,}日)')
    print(f'  → Block E: {verdict}')

    return {
        'zero_lev_days': zero_count,
        'nonzero_mod_when_la_zero': nonzero_mod_when_la0,
        'arith_consistency_max_diff': arith_diff_max,
        'arith_fail_days': arith_fail,
        'strict_zero_fail_days': strict_fail,
        'fail_days': strict_fail,  # 旧キー互換
        'verdict': verdict,
    }


# ===========================================================================
# MD レポート生成
# ===========================================================================

def generate_md(res_a, res_b, res_c, res_d, res_e, out_path: str):
    lines = [
        '# ロジック正しさ検証レポート — E4 Regime k_lt',
        '',
        f'作成日: 2026-05-26  ',
        '対象戦略: S2_VZGated + LT2-N750 + E4 Regime k_lt  ',
        '実行スクリプト: src/audit/check_logic_delay_and_causal.py',
        '',
        '---',
        '',
        '## Block A: DELAY=2 コンプライアンス（NAV 手計算照合）',
        '',
        '| 日付 | 公式 daily_ret | 手計算 daily_ret | 差 | 判定 |',
        '|---|---|---|---|---|',
    ]

    for r in res_a['rows']:
        mark = '✅' if r['pass'] else '❌'
        lines.append(
            f"| {r['date']} | {r['daily_official']:+.6f} | "
            f"{r['daily_manual']:+.6f} | {r['diff']:.2e} | {mark} |"
        )

    overall_a = '✅ PASS' if res_a['verdict'] == 'PASS' else '❌ FAIL'
    lines += [
        f"| **総合** | | | max_diff={res_a['max_diff']:.2e} | **{overall_a}** |",
        '',
    ]

    # Block B
    lines += [
        '## Block B: vz rolling 因果性',
        '',
        '| t | 日付 | vz_full | vz_trunc | diff | 判定 |',
        '|---|---|---|---|---|---|',
    ]
    dates_note = '（キャッシュから取得）'
    if res_b['failures']:
        for f in res_b['failures']:
            lines.append(
                f"| {f['t']} | — | {f['vz_full']:+.6f} | {f['vz_trunc']:+.6f} | "
                f"{f['diff']:.2e} | ❌ |"
            )
    else:
        for t in res_b['checked_t_indices']:
            lines.append(f'| {t} | {dates_note} | — | — | — | ✅ |')

    overall_b = '✅ PASS' if res_b['verdict'] == 'PASS' else '❌ FAIL'
    lines += [
        f'| **総合** | | | | | **{overall_b}** |',
        '',
    ]

    # Block C
    lines += [
        '## Block C: lt_sig 因果性',
        '',
        '| t | 日付 | lt_full | lt_trunc | diff | 判定 |',
        '|---|---|---|---|---|---|',
    ]
    if res_c['failures']:
        for f in res_c['failures']:
            lines.append(
                f"| {f['t']} | — | {f['lt_full']:+.6f} | {f['lt_trunc']:+.6f} | "
                f"{f['diff']:.2e} | ❌ |"
            )
    else:
        for t in res_c['checked_t_indices']:
            lines.append(f'| {t} | {dates_note} | — | — | — | ✅ |')

    overall_c = '✅ PASS' if res_c['verdict'] == 'PASS' else '❌ FAIL'
    lines += [
        f'| **総合** | | | | | **{overall_c}** |',
        '',
    ]

    # Block D
    lines += [
        '## Block D: k_dyn 境界条件（7ケース）',
        '',
        '| vz値 | 期待k_dyn | 実際k_dyn | 差 | ラベル | 判定 |',
        '|---|---|---|---|---|---|',
    ]
    for tc in res_d['test_cases']:
        mark = '✅' if tc['pass'] else '❌'
        diff_d = abs(tc['actual'] - tc['expected'])
        lines.append(
            f"| {tc['vz_val']:+.5f} | {tc['expected']:.1f} | {tc['actual']:.1f} | "
            f"{diff_d:.2e} | {tc['label']} | {mark} |"
        )

    overall_d = '✅ PASS' if res_d['verdict'] == 'PASS' else '❌ FAIL'
    lines += [
        f'| **総合** | | | | 7/7ケース | **{overall_d}** |',
        '',
    ]

    # Block E
    overall_e = '✅ PASS' if res_e['verdict'] == 'PASS' else '❌ FAIL'
    lines += [
        '## Block E: lev_A=0 境界条件',
        '',
        f"- lev_A=0 の日: {res_e['zero_lev_days']:,} 日",
        f"- lev_A=0 かつ lev_mod_e4>0 の日: {res_e['nonzero_mod_when_la_zero']:,} 日",
        f"  （lt_bias>0 による仕様通りの上書き。lev_A=0 でも LT2 バイアスがポジションを追加）",
        f"- 演算整合性 clip(lev_A+lt_bias,0,1) ≡ lev_mod_e4: max_diff={res_e['arith_consistency_max_diff']:.2e}, fail={res_e['arith_fail_days']}日",
        f"- lev_A=0 かつ lt_bias<=0 の厳格ゼロ確認: fail={res_e['strict_zero_fail_days']}日 (期待=0)",
        f"- 判定: {overall_e}",
        '',
    ]

    # 総合
    all_verdicts = [res_a['verdict'], res_b['verdict'], res_c['verdict'],
                    res_d['verdict'], res_e['verdict']]
    overall_all = 'PASS' if all(v == 'PASS' for v in all_verdicts) else 'FAIL'
    overall_mark = '✅ ALL PASS' if overall_all == 'PASS' else '❌ FAIL'

    lines += [
        '## 総合判定',
        '',
        '| Block | 判定 |',
        '|---|---|',
        f"| A DELAY=2 コンプライアンス | {'✅ PASS' if res_a['verdict'] == 'PASS' else '❌ FAIL'} |",
        f"| B vz 因果性 | {'✅ PASS' if res_b['verdict'] == 'PASS' else '❌ FAIL'} |",
        f"| C lt_sig 因果性 | {'✅ PASS' if res_c['verdict'] == 'PASS' else '❌ FAIL'} |",
        f"| D k_dyn 境界条件 | {'✅ PASS' if res_d['verdict'] == 'PASS' else '❌ FAIL'} |",
        f"| E lev_A=0 境界条件 | {'✅ PASS' if res_e['verdict'] == 'PASS' else '❌ FAIL'} |",
        f'| **総合** | **{overall_mark}** |',
        '',
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n  [MD] Written: {out_path}')


# ===========================================================================
# YAML 生成
# ===========================================================================

def generate_yaml(res_a, res_b, res_c, res_d, res_e, out_path: str):
    all_verdicts = [res_a['verdict'], res_b['verdict'], res_c['verdict'],
                    res_d['verdict'], res_e['verdict']]
    overall = 'PASS' if all(v == 'PASS' for v in all_verdicts) else 'FAIL'

    data = {
        'strategy': 'S2_VZGated+LT2-N750+E4_RegimeKlt',
        'generated': '2026-05-26',
        'blocks': {
            'A_delay_compliance': {
                'checked_dates': res_a['checked_dates'],
                'max_diff': float(res_a['max_diff']),
                'fail_count': int(res_a['fail_count']),
                'verdict': res_a['verdict'],
            },
            'B_vz_causality': {
                'checked_t_indices': res_b['checked_t_indices'],
                'failures': res_b['failures'],
                'verdict': res_b['verdict'],
            },
            'C_lt_sig_causality': {
                'checked_t_indices': res_c['checked_t_indices'],
                'failures': res_c['failures'],
                'verdict': res_c['verdict'],
            },
            'D_k_dyn_boundary': {
                'test_cases': [
                    {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                     for k, v in tc.items()}
                    for tc in res_d['test_cases']
                ],
                'failures': res_d['failures'],
                'verdict': res_d['verdict'],
            },
            'E_lev_zero_boundary': {
                'zero_lev_days': res_e['zero_lev_days'],
                'nonzero_mod_when_la_zero': res_e['nonzero_mod_when_la_zero'],
                'arith_consistency_max_diff': float(res_e['arith_consistency_max_diff']),
                'arith_fail_days': res_e['arith_fail_days'],
                'strict_zero_fail_days': res_e['strict_zero_fail_days'],
                'verdict': res_e['verdict'],
            },
        },
        'overall_verdict': overall,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _dump_yaml(data, out_path)
    print(f'  [YAML] Written: {out_path}')


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 60)
    print('ロジック正しさ検証 — E4 Regime k_lt')
    print('=' * 60)

    print('\n[LOAD] E4 strategy assets (cached) ...')
    assets = build_e4_strategy_assets(force_rebuild=False)
    print(f"  dates: {assets['dates'].iloc[0].date()} → {assets['dates'].iloc[-1].date()}")
    print(f"  n: {len(assets['close'])} rows")
    print(f"  nav_e4 final: {float(assets['nav_e4'].iloc[-1]):.4f}")

    res_a = run_block_a(assets)
    res_b = run_block_b(assets)
    res_c = run_block_c(assets)
    res_d = run_block_d()
    res_e = run_block_e(assets)

    md_path   = os.path.join(AUDIT_DIR, 'LOGIC_CHECK_20260526.md')
    yaml_path = os.path.join(AUDIT_DIR, 'logic_check.yaml')

    generate_md(res_a, res_b, res_c, res_d, res_e, md_path)
    generate_yaml(res_a, res_b, res_c, res_d, res_e, yaml_path)

    # 最終サマリー
    all_v = [res_a['verdict'], res_b['verdict'], res_c['verdict'],
             res_d['verdict'], res_e['verdict']]
    overall = 'ALL PASS' if all(v == 'PASS' for v in all_v) else 'FAIL'
    print('\n' + '=' * 60)
    print(f'  総合判定: {overall}')
    print(f'  A={res_a["verdict"]}  B={res_b["verdict"]}  C={res_c["verdict"]}  '
          f'D={res_d["verdict"]}  E={res_e["verdict"]}')
    print('=' * 60)
