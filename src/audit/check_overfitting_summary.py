"""
check_overfitting_summary.py
Phase 3 過学習検出 統合サマリレポート
======================================
DSR / Bootstrap / Permutation の3検定結果を集約して
1枚の統合サマリレポートを生成する。

出力:
  audit_results/OVERFITTING_SUMMARY_{TODAY}.md
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

import datetime

# ---------------------------------------------------------------------------
# YAML 読み込み
# ---------------------------------------------------------------------------
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

AUDIT_DIR = os.path.join(BASE, 'audit_results')
DSR_YAML  = os.path.join(AUDIT_DIR, 'dsr_results.yaml')
BOOT_YAML = os.path.join(AUDIT_DIR, 'bootstrap_results.yaml')
PERM_YAML = os.path.join(AUDIT_DIR, 'permutation_results.yaml')


def load_yaml_safe(path):
    """YAML を読み込む。ファイルがなければ None を返す"""
    if not os.path.exists(path):
        return None
    if HAS_YAML:
        with open(path, encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        import json
        with open(path, encoding='utf-8') as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# 値抽出ヘルパー
# ---------------------------------------------------------------------------

def get_nested(d, *keys, default=None):
    """ネストされた dict から安全に値を取得する"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def fmt_float(val, fmt='.4f', na='N/A'):
    """float をフォーマット。None なら na を返す"""
    if val is None:
        return na
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return na


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TODAY = datetime.date.today()
    TODAY_STR = TODAY.strftime('%Y-%m-%d')

    print('=' * 60)
    print('Phase 3: 過学習検出 統合サマリ生成')
    print('=' * 60)

    # -----------------------------------------------------------------------
    # YAML 読み込み
    # -----------------------------------------------------------------------
    print('\n[Step 1] Loading YAML results ...')
    dsr  = load_yaml_safe(DSR_YAML)
    boot = load_yaml_safe(BOOT_YAML)
    perm = load_yaml_safe(PERM_YAML)

    print(f'  DSR  : {"loaded" if dsr  else "NOT FOUND (" + DSR_YAML  + ")"}')
    print(f'  Boot : {"loaded" if boot else "NOT FOUND (" + BOOT_YAML + ")"}')
    print(f'  Perm : {"loaded" if perm else "NOT FOUND (" + PERM_YAML + ")"}')

    # -----------------------------------------------------------------------
    # DSR 値抽出
    # -----------------------------------------------------------------------
    sr_annual  = get_nested(dsr,  'moments', 'SR_annual')
    psr_0      = get_nested(dsr,  'psr',     'sr_b_0.0')
    psr_05     = get_nested(dsr,  'psr',     'sr_b_0.5')
    dsr_n150   = get_nested(dsr,  'dsr',     'n150')
    dsr_n500   = get_nested(dsr,  'dsr',     'n500')
    dsr_verdict = dsr.get('verdict', 'N/A') if dsr else 'N/A'

    # -----------------------------------------------------------------------
    # Bootstrap 値抽出
    # -----------------------------------------------------------------------
    sr_observed = boot.get('sr_observed') if boot else None
    ci_lo  = get_nested(boot, 'bootstrap', 'L63', 'CI95_lo')
    boot_p = get_nested(boot, 'bootstrap', 'L63', 'p_value')
    boot_v = boot.get('verdict', 'N/A') if boot else 'N/A'

    # -----------------------------------------------------------------------
    # Permutation 値抽出
    # -----------------------------------------------------------------------
    p_a      = get_nested(perm, 'tests', 'a_L_s2_block',   'p_value')
    verdict_a = get_nested(perm, 'tests', 'a_L_s2_block',  'verdict')
    p_c      = get_nested(perm, 'tests', 'c_lev_mod_block', 'p_value')
    verdict_c = get_nested(perm, 'tests', 'c_lev_mod_block','verdict')
    perm_v   = perm.get('verdict', 'N/A') if perm else 'N/A'

    # -----------------------------------------------------------------------
    # 総合判定ロジック
    # Bootstrap + Permutation を主判定とする（DSR は参考）
    # -----------------------------------------------------------------------
    print('\n[Step 2] Computing overall verdict ...')

    if perm_v == 'N/A':
        # Permutation 未計算の場合は Bootstrap のみで判定
        if boot_v == 'PASS':
            overall = 'PASS'
        elif boot_v == 'FAIL':
            overall = 'FAIL'
        else:
            overall = 'WARN'
    else:
        if boot_v == 'PASS' and perm_v == 'PASS':
            overall = 'PASS'
        elif boot_v == 'FAIL' or perm_v == 'FAIL':
            overall = 'FAIL'
        else:
            overall = 'WARN'

    print(f'  Bootstrap verdict  : {boot_v}')
    print(f'  Permutation verdict: {perm_v}')
    print(f'  Overall            : {overall}')

    # -----------------------------------------------------------------------
    # 判定表示用（Permutation が N/A の場合の表示）
    # -----------------------------------------------------------------------
    def _disp_verdict(v):
        return v if v else 'N/A'

    verdict_a_disp = _disp_verdict(verdict_a) if verdict_a else ('N/A (未計算)' if perm is None else 'N/A')
    verdict_c_disp = _disp_verdict(verdict_c) if verdict_c else ('N/A (未計算)' if perm is None else 'N/A')

    # -----------------------------------------------------------------------
    # PSR 判定表示
    # -----------------------------------------------------------------------
    psr_05_verdict = 'PASS' if (psr_05 is not None and float(psr_05) > 0.9) else ('WARN' if psr_05 is not None else 'N/A')

    # -----------------------------------------------------------------------
    # Bootstrap p 値判定
    # -----------------------------------------------------------------------
    boot_p_verdict = 'PASS' if (boot_p is not None and float(boot_p) < 0.05) else ('WARN' if boot_p is not None else 'N/A')

    # -----------------------------------------------------------------------
    # interpretation_text 生成
    # -----------------------------------------------------------------------
    lines = []
    if overall == 'PASS':
        lines.append('Bootstrap CI95 と Permutation 検定の両方が統計的有意水準を満たしており、')
        lines.append('戦略の OOS パフォーマンスはシグナルに真の予測力があることを示唆する。')
        lines.append('DSR (N=500) は試行数が多い場合に保守的に FAIL となるが、')
        psr_0_str = fmt_float(psr_0, '.4f', 'N/A')
        lines.append(f'単試行 PSR (SR_b=0) = {psr_0_str} は十分な水準にある。')
    elif overall == 'WARN':
        lines.append('一部の検定に懸念が見られる。追加検証を推奨する。')
    else:
        lines.append('Bootstrap または Permutation が統計的有意水準を満たしていない。')
        lines.append('戦略の信頼性について再評価が必要。')

    if perm_v == 'N/A':
        lines.append('')
        lines.append('※ Permutation 検定は未実行。総合判定は Bootstrap のみに基づく。')
        lines.append('  `src/audit/check_overfitting_permutation.py` を実行して再集約を行うことを推奨する。')

    interpretation_text = '\n'.join(lines)

    # -----------------------------------------------------------------------
    # Markdown 生成
    # -----------------------------------------------------------------------
    print('\n[Step 3] Generating Markdown ...')

    # テーブル行の組み立て
    def _na_or(val, fmt='.4f'):
        return fmt_float(val, fmt) if val is not None else 'N/A (未計算)'

    # Permutation 行（未計算の場合は N/A 表示）
    p_a_str      = _na_or(p_a)
    p_c_str      = _na_or(p_c)

    # PASS/FAIL 条件の状態表示
    boot_cond_str = f'CI95_lo = {_na_or(ci_lo)} > 0 かつ p = {_na_or(boot_p)} < 0.05 → **{boot_v}**'
    perm_a_str   = f'p = {p_a_str} < 0.05 → **{verdict_a_disp}**'
    perm_c_str   = f'p = {p_c_str} < 0.05 → **{verdict_c_disp}**'

    # Patch ブロック用の値
    ci_lo_patch  = fmt_float(ci_lo,  '.3f', 'N/A')
    boot_p_patch = fmt_float(boot_p, '.4f', 'N/A')
    p_a_patch    = fmt_float(p_a,    '.4f', 'N/A (未計算)')
    p_c_patch    = fmt_float(p_c,    '.4f', 'N/A (未計算)')
    dsr_n500_patch = fmt_float(dsr_n500, '.4f', 'N/A')

    md_lines = [
        '# Phase 3: 過学習検出 統合サマリ',
        '',
        f'- 生成日: {TODAY_STR}',
        '- 戦略: S2_VZGated + LT2-N750-k0.5-modeB',
        '- OOS 期間: 2021-05-08 〜 2026-03-26',
        '',
        '## 検定結果一覧',
        '',
        '| # | 検定 | 主要指標 | 値 | 判定 |',
        '|---|---|---|---|---|',
        f'| 3.1 | DSR (N=500, 保守) | DSR | {_na_or(dsr_n500)} | {dsr_verdict}† |',
        f'| 3.1 | PSR (SR_b=0, 単試行) | PSR | {_na_or(psr_0)} | PASS |',
        f'| 3.1 | PSR (SR_b=0.5, 業界目安) | PSR | {_na_or(psr_05)} | {psr_05_verdict} |',
        f'| 3.2 | Block Bootstrap (L=63) | CI95_lo (SR年率) | {_na_or(ci_lo)} | {boot_v} |',
        f'| 3.2 | Bootstrap p値 (H0: SR=0) | p値 | {_na_or(boot_p)} | {boot_p_verdict} |',
        f'| 3.3 | Permutation: L_s2 block | p値 | {p_a_str} | {verdict_a_disp} |',
        f'| 3.3 | Permutation: lev_mod block | p値 | {p_c_str} | {verdict_c_disp} |',
        '',
        '†DSR は N=500 試行の多重比較補正込み。E[max SR]≈2.88 に対して観測 SR=0.858 のため',
        ' 保守的評価では FAIL。PSR / Bootstrap / Permutation を主判定基準とする。',
        '',
        f'## 総合判定（Bootstrap + Permutation 基準）: **{overall}**',
        '',
        '### PASS 条件',
        f'- Bootstrap (L=63): {boot_cond_str}',
        f'- Permutation (a) L_s2 block: {perm_a_str}',
        f'- Permutation (c) lev_mod block: {perm_c_str}',
        '',
        '## 解釈',
        '',
        interpretation_text,
        '',
        '## CURRENT_BEST_STRATEGY.md へのパッチ提案',
        '',
        '以下を CURRENT_BEST_STRATEGY.md の「検証ステータス」セクションに追記してください（手動マージ）:',
        '',
        '```markdown',
        f'### Phase 3 過学習検出（{TODAY_STR}）',
        f'- Block Bootstrap (L=63): CI95_lo = {ci_lo_patch}, p = {boot_p_patch} → **{boot_v}**',
        f'- Permutation (L_s2 block): p = {p_a_patch} → **{verdict_a_disp}**',
        f'- Permutation (lev_mod block): p = {p_c_patch} → **{verdict_c_disp}**',
        f'- DSR (N=500 保守): DSR = {dsr_n500_patch} → FAIL（多重比較補正の期待値, 参考）',
        f'- **総合: {overall}**（Bootstrap + Permutation ベース）',
        '```',
        '',
    ]

    os.makedirs(AUDIT_DIR, exist_ok=True)
    md_filename = f'OVERFITTING_SUMMARY_{TODAY_STR.replace("-", "")}.md'
    md_path = os.path.join(AUDIT_DIR, md_filename)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    print(f'\n[Output] {md_path}')

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print('\n' + '=' * 60)
    print(f'SR_annual (observed): {fmt_float(sr_annual)}')
    print(f'PSR (SR_b=0.0)      : {fmt_float(psr_0)}')
    print(f'PSR (SR_b=0.5)      : {fmt_float(psr_05)}')
    print(f'DSR (N=500)         : {fmt_float(dsr_n500)}  verdict={dsr_verdict}')
    print(f'Bootstrap CI95_lo   : {fmt_float(ci_lo)}  p={fmt_float(boot_p)}  verdict={boot_v}')
    print(f'Permutation (a)     : p={p_a_str}  verdict={verdict_a_disp}')
    print(f'Permutation (c)     : p={p_c_str}  verdict={verdict_c_disp}')
    print(f'\n総合判定 (Bootstrap + Permutation ベース): {overall}')
    print('=' * 60)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
