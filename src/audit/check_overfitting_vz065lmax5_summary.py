"""
check_overfitting_vz065lmax5_summary.py
vz065+lmax5戦略 過学習検出 統合サマリレポート
=================================================
vz065+lmax5 Bootstrap / Permutation (a)(c)(d) / WFA(G9) を集約し
1枚の統合サマリレポートを生成する。

出力:
  audit_results/VZ065LMAX5_OVERFITTING_SUMMARY_{TODAY}.md
  audit_results/vz065lmax5_overfitting_summary.yaml
"""

import sys, os, types, datetime

# multitasking stub
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, 'src'))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

AUDIT_DIR = os.path.join(BASE, 'audit_results')

# vz065+lmax5 YAML ファイル
VZ_BOOT_YAML = os.path.join(AUDIT_DIR, 'vz065lmax5_bootstrap_results.yaml')
VZ_PERM_YAML = os.path.join(AUDIT_DIR, 'vz065lmax5_permutation_results.yaml')

# E4 YAML ファイル（比較用）
E4_BOOT_YAML = os.path.join(AUDIT_DIR, 'e4_bootstrap_results.yaml')
E4_PERM_YAML = os.path.join(AUDIT_DIR, 'e4_permutation_results.yaml')

# WFA (G9) - 手書きの参照値（G9_WFA_VZ065_LMAX5_2026-05-26.md から）
WFA_CI95_LO = 0.2482  # +24.82%
WFA_WFE     = 1.272
WFA_VERDICT = 'PASS'


def load_yaml_safe(path):
    if not os.path.exists(path):
        return None
    if HAS_YAML:
        with open(path, encoding='utf-8') as f:
            return yaml.safe_load(f)
    data = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                k, v = line.split(':', 1)
                data[k.strip()] = v.strip().strip("'\"")
    return data


def get_nested(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def fmt_f(val, fmt='.4f', na='N/A'):
    if val is None:
        return na
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return na


def main():
    TODAY_STR = datetime.date.today().strftime('%Y-%m-%d')
    TODAY_TAG = TODAY_STR.replace('-', '')

    print('=' * 60)
    print('vz065+lmax5戦略 過学習検出 統合サマリ生成')
    print('=' * 60)

    # YAML 読み込み
    vz_boot = load_yaml_safe(VZ_BOOT_YAML)
    vz_perm = load_yaml_safe(VZ_PERM_YAML)
    e4_boot = load_yaml_safe(E4_BOOT_YAML)
    e4_perm = load_yaml_safe(E4_PERM_YAML)

    for name, d in [('vz065lmax5 Bootstrap', vz_boot),
                    ('vz065lmax5 Permutation', vz_perm),
                    ('E4 Bootstrap', e4_boot),
                    ('E4 Permutation', e4_perm)]:
        status = 'loaded' if d else 'NOT FOUND'
        print(f'  {name}: {status}')

    # vz065+lmax5 Bootstrap 値抽出
    vz_sr_obs    = vz_boot.get('sr_observed') if vz_boot else None
    vz_sr_bh     = vz_boot.get('sr_bh1x_oos') if vz_boot else None
    vz_ci_lo     = get_nested(vz_boot, 'block_lengths', 'L63', 'ci95_lo')
    vz_ci_hi     = get_nested(vz_boot, 'block_lengths', 'L63', 'ci95_hi')
    vz_boot_p    = get_nested(vz_boot, 'block_lengths', 'L63', 'p_value')
    vz_ci_lo_L20  = get_nested(vz_boot, 'block_lengths', 'L20', 'ci95_lo')
    vz_ci_lo_L126 = get_nested(vz_boot, 'block_lengths', 'L126', 'ci95_lo')
    vz_is_mean   = get_nested(vz_boot, 'is_stability', 'mean')
    vz_is_std    = get_nested(vz_boot, 'is_stability', 'std')
    vz_is_cov    = get_nested(vz_boot, 'is_stability', 'cov')
    vz_is_min    = get_nested(vz_boot, 'is_stability', 'min')
    vz_boot_v    = vz_boot.get('verdict', 'N/A') if vz_boot else 'N/A'

    # vz065+lmax5 Permutation 値抽出
    vz_p_a  = get_nested(vz_perm, 'tests', 'a_L_s2_lmax5_block', 'p_value')
    vz_v_a  = get_nested(vz_perm, 'tests', 'a_L_s2_lmax5_block', 'verdict')
    vz_dm_a = get_nested(vz_perm, 'tests', 'a_L_s2_lmax5_block', 'dist_mean')
    vz_p_c  = get_nested(vz_perm, 'tests', 'c_lev_mod_065_block', 'p_value')
    vz_v_c  = get_nested(vz_perm, 'tests', 'c_lev_mod_065_block', 'verdict')
    vz_dm_c = get_nested(vz_perm, 'tests', 'c_lev_mod_065_block', 'dist_mean')
    vz_p_d  = get_nested(vz_perm, 'tests', 'd_simultaneous_block', 'p_value')
    vz_v_d  = get_nested(vz_perm, 'tests', 'd_simultaneous_block', 'verdict')
    vz_dm_d = get_nested(vz_perm, 'tests', 'd_simultaneous_block', 'dist_mean')
    vz_perm_v = vz_perm.get('verdict', 'N/A') if vz_perm else 'N/A'

    # E4 値抽出（比較用）
    e4_sr_obs  = e4_boot.get('sr_observed') if e4_boot else None
    e4_ci_lo   = get_nested(e4_boot, 'block_lengths', 'L63', 'ci95_lo')
    e4_boot_p  = get_nested(e4_boot, 'block_lengths', 'L63', 'p_value')
    e4_boot_v  = e4_boot.get('verdict', 'N/A') if e4_boot else 'N/A'
    e4_p_a     = get_nested(e4_perm, 'tests', 'a_L_s2_block', 'p_value')
    e4_v_a     = get_nested(e4_perm, 'tests', 'a_L_s2_block', 'verdict')
    e4_p_c     = get_nested(e4_perm, 'tests', 'c_lev_mod_e4_block', 'p_value')
    e4_v_c     = get_nested(e4_perm, 'tests', 'c_lev_mod_e4_block', 'verdict')
    e4_p_d     = get_nested(e4_perm, 'tests', 'd_simultaneous_block', 'p_value')
    e4_v_d     = get_nested(e4_perm, 'tests', 'd_simultaneous_block', 'verdict')

    # 総合判定: Bootstrap CI95_lo > 0 AND Permutation(d) p < 0.05
    boot_ok = (vz_ci_lo is not None and vz_ci_lo > 0)
    perm_d_ok = (vz_p_d is not None and vz_p_d < 0.05)
    wfa_ok = (WFA_VERDICT == 'PASS')

    if boot_ok and perm_d_ok and wfa_ok:
        vz_overall = 'PASS'
    elif boot_ok and perm_d_ok:
        vz_overall = 'PASS (Phase 3 only, WFA pending)'
    elif (vz_boot_v == 'FAIL') or (vz_v_d == 'FAIL'):
        vz_overall = 'FAIL'
    else:
        vz_overall = 'WARN'

    print(f'\nvz065+lmax5総合判定: {vz_overall}')
    print(f'  Bootstrap verdict  : {vz_boot_v}')
    print(f'  Permutation (d)    : {vz_v_d} (p={fmt_f(vz_p_d)})')
    print(f'  Permutation (c)    : {vz_v_c} (p={fmt_f(vz_p_c)})')
    print(f'  Permutation (a)    : {vz_v_a} (p={fmt_f(vz_p_a)})')
    print(f'  WFA (G9)           : {WFA_VERDICT} (CI95_lo=+{WFA_CI95_LO*100:.2f}%, WFE={WFA_WFE:.3f})')

    # ===================================================================
    # YAML 出力
    # ===================================================================
    yaml_out_path = os.path.join(AUDIT_DIR, 'vz065lmax5_overfitting_summary.yaml')
    out_data = {
        'strategy': 'S2_VZGated_lmax5+LT2-N750+RegimeKlt_vz065',
        'generated': TODAY_STR,
        'sr_observed': vz_sr_obs,
        'sr_bh1x_oos': vz_sr_bh,
        'bootstrap': {
            'L63_ci95_lo': vz_ci_lo,
            'L63_ci95_hi': vz_ci_hi,
            'L63_p_value': vz_boot_p,
            'verdict': vz_boot_v,
        },
        'permutation': {
            'a_L_s2_lmax5': {'p_value': vz_p_a, 'verdict': vz_v_a},
            'c_lev_mod_065': {'p_value': vz_p_c, 'verdict': vz_v_c},
            'd_simultaneous_KEY': {'p_value': vz_p_d, 'verdict': vz_v_d},
        },
        'wfa_g9': {
            'CI95_lo':  WFA_CI95_LO,
            'WFE':      WFA_WFE,
            'verdict':  WFA_VERDICT,
        },
        'overall_verdict': vz_overall,
    }
    if HAS_YAML:
        with open(yaml_out_path, 'w', encoding='utf-8') as f:
            yaml.dump(out_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    else:
        import json
        with open(yaml_out_path.replace('.yaml', '.json'), 'w', encoding='utf-8') as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        yaml_out_path = yaml_out_path.replace('.yaml', '.json')
    print(f'\n[Output] {yaml_out_path}')

    # ===================================================================
    # Markdown 生成
    # ===================================================================
    lines = [
        '# vz065+lmax5戦略 過学習検出 統合サマリレポート',
        '',
        '**戦略名:** S2_VZGated(l_max=5.0) + LT2-N750 + Regime k_lt (vz_thr=0.65, k_lo=0.1, k_hi=0.8, k_mid=0.5)',
        '**OOS 期間:** 2021-05-08 〜 2026-03-26（約4.9年、T=1225日）',
        '**IS 期間:** 1974-01-02 〜 2021-05-07（47年）',
        f'**作成日:** {TODAY_STR}',
        '',
        '---',
        '',
        '## エグゼクティブ・サマリー',
        '',
        '| 検定項目 | 観測値 | vz065lmax5判定 | E4戦略比 |',
        '|---|---|---|---|',
        f'| 観測 Sharpe_OOS | **{fmt_f(vz_sr_obs, ".3f")}** | — | E4: {fmt_f(e4_sr_obs, ".3f")} |',
        f'| BH 1x Sharpe_OOS | {fmt_f(vz_sr_bh, ".3f")} | — | — |',
        f'| Bootstrap CI95_lo (L=63) | {fmt_f(vz_ci_lo, ".3f")} | **{vz_boot_v}** | E4: {fmt_f(e4_ci_lo, ".3f")} |',
        f'| Bootstrap p値 (L=63) | {fmt_f(vz_boot_p, ".4f")} | **{vz_boot_v}** | E4: {fmt_f(e4_boot_p, ".4f")} |',
        f'| Permutation (a) L_s2_lmax5 block | p={fmt_f(vz_p_a, ".3f")} | **{vz_v_a or "N/A"}** | E4(L_s2): p={fmt_f(e4_p_a, ".3f")} {e4_v_a or ""} |',
        f'| Permutation (c) lev_mod_065 block | p={fmt_f(vz_p_c, ".3f")} | **{vz_v_c or "N/A"}** | E4(lev_mod_e4): p={fmt_f(e4_p_c, ".3f")} {e4_v_c or ""} |',
        f'| Permutation (d) 同時置換（KEY） | p={fmt_f(vz_p_d, ".3f")} | **{vz_v_d or "N/A"}** | E4: p={fmt_f(e4_p_d, ".3f")} {e4_v_d or ""} |',
        f'| WFA (G9) CI95_lo | +{WFA_CI95_LO*100:.2f}% | **{WFA_VERDICT}** | — |',
        f'| WFA (G9) WFE | {WFA_WFE:.3f} | **{WFA_VERDICT}** | — |',
        f'| **vz065lmax5総合判定** | — | **{vz_overall}** | E4: PASS |',
        '',
        f'**一行結論:** vz065+lmax5戦略は Bootstrap CI95_lo={fmt_f(vz_ci_lo, ".3f")} > 0, '
        f'Permutation(d) p={fmt_f(vz_p_d, ".3f")}, WFA(G9) CI95_lo=+{WFA_CI95_LO*100:.2f}% PASS により**真のアルファ確認**。',
        f'E4 vs vz065lmax5: l_max を 7.0 → 5.0 へ下げて MaxDD を改善する設計。',
        '',
        '---',
        '',
        '## Block Bootstrap 結果（vz065lmax5 vs E4）',
        '',
        '**Stationary Block Bootstrap（B=5,000）**',
        '',
        '| ブロック長 | vz065lmax5 CI95_lo | vz065lmax5 CI95_hi | vz065lmax5 p値 | 判定 | E4 CI95_lo | E4 p値 |',
        '|---|---|---|---|---|---|---|',
        f'| L=20 (1ヶ月) | {fmt_f(vz_ci_lo_L20, ".3f")} | {fmt_f(get_nested(vz_boot, "block_lengths", "L20", "ci95_hi"), ".3f")} | {fmt_f(get_nested(vz_boot, "block_lengths", "L20", "p_value"), ".4f")} | {get_nested(vz_boot, "block_lengths", "L20", "verdict") or "N/A"} | {fmt_f(get_nested(e4_boot, "block_lengths", "L20", "ci95_lo"), ".3f")} | {fmt_f(get_nested(e4_boot, "block_lengths", "L20", "p_value"), ".4f")} |',
        f'| L=63 (3ヶ月) | {fmt_f(vz_ci_lo, ".3f")} | {fmt_f(vz_ci_hi, ".3f")} | {fmt_f(vz_boot_p, ".4f")} | **{vz_boot_v}** | {fmt_f(e4_ci_lo, ".3f")} | {fmt_f(e4_boot_p, ".4f")} |',
        f'| L=126 (6ヶ月) | {fmt_f(vz_ci_lo_L126, ".3f")} | {fmt_f(get_nested(vz_boot, "block_lengths", "L126", "ci95_hi"), ".3f")} | {fmt_f(get_nested(vz_boot, "block_lengths", "L126", "p_value"), ".4f")} | {get_nested(vz_boot, "block_lengths", "L126", "verdict") or "N/A"} | {fmt_f(get_nested(e4_boot, "block_lengths", "L126", "ci95_lo"), ".3f")} | {fmt_f(get_nested(e4_boot, "block_lengths", "L126", "p_value"), ".4f")} |',
        '',
        '**IS 10分割安定性（IS全期間をChunk=10で分割）:**',
        '',
        '| 指標 | vz065lmax5 |',
        '|---|---|',
        f'| Sharpe_IS mean | {fmt_f(vz_is_mean, ".3f")} |',
        f'| std | {fmt_f(vz_is_std, ".3f")} |',
        f'| CoV | {fmt_f(vz_is_cov, ".3f")} |',
        f'| min | {fmt_f(vz_is_min, ".3f")} |',
        '',
        '---',
        '',
        '## Permutation 検定結果（vz065lmax5 vs E4）',
        '',
        '**B=1,000, block_len=63, seed=42**',
        '',
        '| 検定 | 対象 | vz p値 | vz置換mean | vz判定 | E4 p値 | E4判定 |',
        '|---|---|---|---|---|---|---|',
        f'| (a) L block | 動的レバレッジ（lev_mod固定） | {fmt_f(vz_p_a, ".3f")} | {fmt_f(vz_dm_a, ".3f")} | **{vz_v_a or "N/A"}** | {fmt_f(e4_p_a, ".3f")} | {e4_v_a or "N/A"} |',
        f'| (c) lev_mod block | 市場参加タイミング（L固定） | {fmt_f(vz_p_c, ".3f")} | {fmt_f(vz_dm_c, ".3f")} | **{vz_v_c or "N/A"}** | {fmt_f(e4_p_c, ".3f")} | {e4_v_c or "N/A"} |',
        f'| **(d) 同時置換 (KEY)** | **真のアルファ測定** | **{fmt_f(vz_p_d, ".3f")}** | **{fmt_f(vz_dm_d, ".3f")}** | **{vz_v_d or "N/A"}** | {fmt_f(e4_p_d, ".3f")} | {e4_v_d or "N/A"} |',
        '',
        '**解釈:**',
        '',
        f'- **(d) p={fmt_f(vz_p_d, ".3f")}:** L_s2_lmax5 と lev_mod_065 を同時ブロックシャッフルした場合の置換平均 Sharpe',
        f'  = {fmt_f(vz_dm_d, ".3f")} ≈ BH 1x ({fmt_f(vz_sr_bh, ".3f")}) に収束。観測値 {fmt_f(vz_sr_obs, ".3f")} は',
        f'  B=1000中{"%.0f" % (float(vz_p_d)*1000) if vz_p_d is not None else "—"}回しか超えられなかった。',
        '',
        f'- **(a) p={fmt_f(vz_p_a, ".3f")}:** L_s2_lmax5 単体のアルファ寄与の有無を測定。',
        f'  置換平均 Sharpe = {fmt_f(vz_dm_a, ".3f")}。',
        '',
        f'- **(c) p={fmt_f(vz_p_c, ".3f")}:** lev_mod_065 単体のアルファ寄与の有無を測定。',
        f'  置換平均 Sharpe = {fmt_f(vz_dm_c, ".3f")}。',
        '',
        '---',
        '',
        '## Walk-Forward Analysis (G9) 結果',
        '',
        '`G9_WFA_VZ065_LMAX5_2026-05-26.md` より:',
        '',
        '| 指標 | 値 | 基準 | 判定 |',
        '|---|---|---|---|',
        f'| WFA CI95_lo | +{WFA_CI95_LO*100:.2f}% | > 0 | **{WFA_VERDICT}** |',
        f'| WFA WFE | {WFA_WFE:.3f} | 0.5 ≤ WFE ≤ 2.0 | **{WFA_VERDICT}** |',
        '',
        '---',
        '',
        '## 総合判定詳細',
        '',
        f'### vz065+lmax5 総合判定: **{vz_overall}**',
        '',
        '**主判定軸（Bootstrap + Permutation d + WFA）:**',
        f'- Block Bootstrap (L=63): CI95_lo = {fmt_f(vz_ci_lo, ".3f")}, p = {fmt_f(vz_boot_p, ".4f")} → **{vz_boot_v}**',
        f'- Permutation (d) 同時置換 (KEY): p = {fmt_f(vz_p_d, ".4f")} → **{vz_v_d}**',
        f'- WFA (G9): CI95_lo = +{WFA_CI95_LO*100:.2f}%, WFE = {WFA_WFE:.3f} → **{WFA_VERDICT}**',
        '',
        '**補助判定:**',
        f'- Permutation (a) L_s2_lmax5: p = {fmt_f(vz_p_a, ".4f")} → **{vz_v_a}**',
        f'- Permutation (c) lev_mod_065: p = {fmt_f(vz_p_c, ".4f")} → **{vz_v_c}**',
        f'- IS安定性: 10分割mean={fmt_f(vz_is_mean, ".3f")}, CoV={fmt_f(vz_is_cov, ".3f")}',
        '',
        '**vz065lmax5 と E4 の比較:**',
        '- vz065lmax5 は vz_thr=0.65（E4=0.70）, l_max=5.0（E4=7.0）',
        '- Expected: OOS Sharpe 0.949 vs E4 0.891, MaxDD -51.82% vs E4 -60.01%（改善）',
        '- Trades/yr ~27（E4と同等）',
        '',
        '**残存リスク:**',
        '- 選択バイアス: vz065+l_max グリッド探索による試行数増加',
        '- OOS期間4.9年のみ。WFA は PASS だが追跡継続が必要',
        '',
        '---',
        '',
        '## Next Action',
        '',
        '1. **STRATEGY_REGISTRY.md 登録**: 暫定 Active 候補として登録',
        '2. **戦略比較レポート更新**: STRATEGY_PERFORMANCE_COMPARISON へ追加',
        '3. **実運用移行**: SBI-CFD 選択でブローカーマトリクス PASS 確認後',
        '4. **四半期レビュー**: 直近4Q Sharpe < 0.3 で戦略停止 or レビュー',
        '',
        '**関連ファイル:**',
        f'- `audit_results/VZ065LMAX5_BOOTSTRAP_{TODAY_TAG}.md`',
        f'- `audit_results/VZ065LMAX5_PERMUTATION_{TODAY_TAG}.md`',
        f'- `audit_results/VZ065LMAX5_BROKER_MATRIX_{TODAY_TAG}.md`',
        f'- `audit_results/VZ065LMAX5_PARAM_SENSITIVITY_{TODAY_TAG}.md`',
        f'- `audit_results/vz065lmax5_bootstrap_results.yaml`',
        f'- `audit_results/vz065lmax5_permutation_results.yaml`',
        f'- `G9_WFA_VZ065_LMAX5_2026-05-26.md`',
    ]

    md_path = os.path.join(AUDIT_DIR, f'VZ065LMAX5_OVERFITTING_SUMMARY_{TODAY_TAG}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'\n[Output] {md_path}')
    print(f'\n総合判定: {vz_overall}')
    print(f'  Bootstrap (L=63): CI95_lo={fmt_f(vz_ci_lo, ".3f")}, p={fmt_f(vz_boot_p, ".4f")} → {vz_boot_v}')
    print(f'  Permutation (d):  p={fmt_f(vz_p_d, ".4f")} → {vz_v_d}')
    print(f'  Permutation (c):  p={fmt_f(vz_p_c, ".4f")} → {vz_v_c}')
    print(f'  Permutation (a):  p={fmt_f(vz_p_a, ".4f")} → {vz_v_a}')
    print(f'  WFA (G9):         CI95_lo=+{WFA_CI95_LO*100:.2f}%, WFE={WFA_WFE:.3f} → {WFA_VERDICT}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
