# -*- coding: utf-8 -*-
"""
check_realworld_report_corpus.py (Phase 2 - 2.8)
=================================================
Deep Researchレポートから抽出した商品・コストデータを根拠台帳(YAML)として出力し、
product_costs.py との差分マトリックスを生成する。

出力:
  audit_results/reports_corpus.yaml
  audit_results/CHECK_CORPUS_<DATE>.md
"""
import sys, os, types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

_AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_AUDIT_DIR)
_BASE = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _AUDIT_DIR)

# ---------------------------------------------------------------------------
from datetime import date

TODAY = date.today().strftime('%Y%m%d')
AUDIT_DIR = os.path.join(_BASE, 'audit_results')
os.makedirs(AUDIT_DIR, exist_ok=True)

CORPUS = {
    'kurikku365_nasdaq': {
        'name': 'くりっく株365 NASDAQ100',
        'category': 'exchange_futures_cfd',
        'source_report': '2026-05-15_nasdaq-4x-leverage-products.md',
        'funding_rate_ann': 0.0492,
        'funding_confidence': 'disclosed',  # TFX公式に毎週公表
        'sofr_spread_est': 0.0133,          # 4.92% - 3.59% ≈ 1.33%
        'sofr_spread_confidence': 'estimated',
        'margin_per_lot_jpy': 8960,
        'margin_confidence': 'disclosed',   # TFX公式
        'commission_per_trade_jpy': 30,     # SBIの場合
        'commission_confidence': 'disclosed',
        'max_leverage_theoretical': 33,
        'max_leverage_confidence': 'estimated',  # 証拠金基準額から計算
        'settlement': 'T+2',
        'tax_group': 'B',
        'nisa_eligible': False,
        'note': 'くりっく株365のfunding_rateはポジション全体に対する年率コスト（先物ロール相当）',
        'available_at_sbi': True,
    },
    'sbi_cfd_nq100': {
        'name': 'SBI CFD 米国NQ100',
        'category': 'otc_cfd',
        'source_report': '2026-05-15_nasdaq-4x-leverage-products.md',
        'spread_pt': 1.58,
        'spread_confidence': 'disclosed',
        'max_leverage': 10,
        'max_leverage_confidence': 'disclosed',  # 法定規制
        'funding_ann': None,
        'funding_confidence': 'not_disclosed',   # SBI公式に非公開
        'funding_est_sofr_spread': None,         # 推計不能
        'settlement': 'T+0',
        'tax_group': 'B',
        'nisa_eligible': False,
        'reference_asset': 'E-mini NASDAQ100 CME先物',
        'available_at_sbi': True,
        'note': 'オーバーナイト金利は非公開。価格調整額（先物ロール）との二重計上疑惑あり',
    },
    'ig_cfd_nasdaq': {
        'name': 'IG証券 CFD NASDAQ100',
        'category': 'otc_cfd',
        'source_report': '2026-05-15_nasdaq-4x-leverage-products.md',
        'spread_pt': 2.0,
        'spread_confidence': 'disclosed',
        'funding_ann': 0.0659,          # SOFR+3.0% = 6.59% (SOFR=3.59%時)
        'funding_confidence': 'disclosed',  # IG公式に明示
        'sofr_spread': 0.0300,
        'sofr_spread_confidence': 'disclosed',
        'max_leverage': 10,
        'settlement': 'T+0',
        'tax_group': 'B',
        'nisa_eligible': False,
        'available_at_sbi': False,
    },
    'rakuten_cfd_nasdaq': {
        'name': '楽天証券 CFD 米国NAS100',
        'category': 'otc_cfd',
        'source_report': '2026-05-15_nasdaq-4x-leverage-products.md',
        'spread_pt': 1.7,
        'spread_confidence': 'disclosed',
        'funding_ann': 0.065,           # SOFR+約3% ≈ 6.5%（確認値）
        'funding_confidence': 'disclosed',
        'sofr_spread': 0.0300,
        'sofr_spread_confidence': 'estimated',
        'max_leverage': 10,
        'settlement': 'T+0',
        'tax_group': 'B',
        'nisa_eligible': False,
        'available_at_sbi': False,
    },
    'gmo_cfd_nasdaq': {
        'name': 'GMOクリック証券 CFD NASDAQ100',
        'category': 'otc_cfd',
        'source_report': '2026-05-15_nasdaq-4x-leverage-products.md',
        'spread_pt': 1.8,
        'spread_confidence': 'disclosed',
        'funding_ann': None,            # 日次ON金利なし
        'funding_confidence': 'disclosed',
        'roll_cost_est_ann': 0.045,     # 推定4.5%（先物コンタンゴ）
        'roll_cost_confidence': 'estimated',
        'method': 'roll_cost',          # 四半期ロール方式
        'max_leverage': 10,
        'settlement': 'T+0',
        'tax_group': 'B',
        'nisa_eligible': False,
        'available_at_sbi': False,
    },
    'dmm_cfd_nasdaq': {
        'name': 'DMM CFD NASDAQ100',
        'category': 'otc_cfd',
        'source_report': '2026-05-15_nasdaq-4x-leverage-products.md',
        'spread_pt': 2.0,
        'spread_confidence': 'disclosed',
        'funding_ann': None,
        'funding_confidence': 'disclosed',
        'roll_cost_est_ann': 0.045,
        'roll_cost_confidence': 'estimated',
        'method': 'roll_cost',
        'max_leverage': 10,
        'settlement': 'T+0',
        'tax_group': 'B',
        'nisa_eligible': False,
        'available_at_sbi': False,
    },
    'tqqq': {
        'name': 'ProShares UltraPro QQQ (TQQQ)',
        'category': 'leveraged_etf',
        'source_report': '2026-05-15_nasdaq-4x-leverage-products.md',
        'ter': 0.0086,
        'ter_confidence': 'disclosed',
        'sofr_multiplier': 2.0,         # 内部スワップ
        'sofr_multiplier_confidence': 'estimated',  # OLS beta_SOFR=-2.13 実証
        'dividend_yield': 0.003,
        'dividend_confidence': 'disclosed',
        'nisa_eligible': False,
        'tax_group': 'A',
        'vol_drag': True,
        'available_at_sbi': True,
        'internal_cost_est_ann': 0.0723,  # 0.86% + SOFR×2(7.18%) - div(0.81%) ≈ 7.23%
        'internal_cost_confidence': 'estimated',
    },
    'qld': {
        'name': 'ProShares Ultra QQQ (QLD)',
        'category': 'leveraged_etf',
        'source_report': '2026-05-15_nasdaq-4x-leverage-products.md',
        'ter': 0.0095,
        'ter_confidence': 'disclosed',
        'sofr_multiplier': 1.0,
        'sofr_multiplier_confidence': 'estimated',
        'vol_drag': True,
        'nisa_eligible': False,
        'tax_group': 'A',
        'available_at_sbi': True,
    },
    'ugl_gold2x': {
        'name': 'ProShares Ultra Gold (UGL) 2x',
        'category': 'leveraged_etf',
        'source_report': '2026-05-15_nasdaq-4x-leverage-products.md',
        'ter': 0.0095,                  # 0.95%/yr（実際にSBIで取引可能）
        'ter_confidence': 'disclosed',
        'sofr_multiplier': 1.0,
        'swap_spread': 0.0050,
        'nisa_eligible': False,
        'tax_group': 'A',
        'available_at_sbi': True,
        'bt_sim_product': 'WisdomTree 2036 LSE',
        'bt_sim_ter': 0.0049,           # シムで使用したTER
        'bt_sim_ter_confidence': 'disclosed',
        'ter_gap': 0.0046,              # 0.95% - 0.49% = 0.46%/yr
        'note': 'バックテストのシムproxy(WisdomTree 2036 LSE, 0.49%)はSBI証券では取引不可。実際はUGL(0.95%)を使用',
    },
    'wisdomtree_gold2x_lse': {
        'name': 'WisdomTree 2036 2x Gold ETP (LSE)',
        'category': 'leveraged_etp',
        'source_report': '2026-05-15_nasdaq-4x-leverage-products.md',
        'ter': 0.0049,
        'ter_confidence': 'disclosed',
        'available_at_sbi': False,
        'available_at_jp_retail': False,
        'note': 'バックテストのシムproxyとして使用。日本個人投資家には実質購入不可（LSE上場）',
    },
    'tmf': {
        'name': 'Direxion Daily 20+ Year Treasury Bull 3x (TMF)',
        'category': 'leveraged_etf',
        'source_report': '2026-05-15_nasdaq-4x-leverage-products.md',
        'ter_sim': 0.0091,              # バックテストで使用した旧TER
        'ter_sim_confidence': 'disclosed',
        'ter_current': 0.0106,          # 現行実際値 1.06%
        'ter_current_confidence': 'disclosed',
        'ter_gap': 0.0015,              # 1.06% - 0.91% = 0.15%/yr
        'sofr_multiplier': 2.0,
        'swap_spread': 0.0050,
        'dividend_yield': 0.035,
        'nisa_eligible': False,
        'tax_group': 'A',
        'available_at_sbi': True,       # SBIで取引可能
        'note': 'シム値0.91%（旧TER）vs現行1.06%。差分15bps/yr',
    },
    'sbi_cfd_regulation': {
        'name': 'SBI CFD 規制・証拠金仕様',
        'category': 'regulation',
        'source_report': '2026-05-20_sbi-cfd-nasdaq-8x-9x-leverage-guide.md',
        'max_leverage_personal': 10,
        'max_leverage_confidence': 'disclosed',   # 金融商品取引業等に関する内閣府令
        'margin_rate_min': 0.10,
        'margin_confidence': 'disclosed',
        'forced_liquidation_trigger_pct': 100.0,  # 維持率100%割れ
        'liquidation_confidence': 'disclosed',
        'margin_call': False,           # 追証なし（自動ロスカット）
        'lot_value_jpy_approx': 4200000, # 1ロット約420万円（29000pt×145円）
        'lot_value_confidence': 'estimated',
        'min_lot': 0.01,
    },
    'sbi_cfd_lc_stress': {
        'name': 'SBI CFD ロスカット耐久テスト（レポート記載値）',
        'category': 'stress_test',
        'source_report': '2026-05-20_sbi-cfd-nasdaq-8x-9x-leverage-guide.md',
        'scenarios': [
            {'deposit_jpy': 1000000, 'leverage': 9.0, 'lc_drop_pct': 1.2},
            {'deposit_jpy': 2000000, 'leverage': 4.5, 'lc_drop_pct': 13.6},
            {'deposit_jpy': 2700000, 'leverage': 3.3, 'lc_drop_pct': 22.2},
        ],
        'confidence': 'disclosed',
        'note': 'ポジション900万円固定での各入金額別ロスカット耐久率',
    },
    'bt_cost_model': {
        'name': 'バックテスト現行コストモデル(Scenario D)',
        'category': 'backtest_model',
        'source_report': 'src/product_costs.py',
        'cfd_spread_low': 0.0020,       # CFD_SPREAD_LOW (0.20%/yr)
        'cfd_spread_confidence': 'disclosed',
        'cfd_note': 'これはスプレッド相当値であり、オーバーナイト金利（funding）は別途SOFR×(L-1)として計上',
        'sofr_proxy': 'DTB3 (FRED 3M T-bill)',
        'delay_days': 2,
    },
}

# ---------------------------------------------------------------------------
# product_costs.py インポート
# ---------------------------------------------------------------------------
from product_costs import TQQQ as TQQQ_COSTS, TMF as TMF_COSTS, GOLD2X as GOLD2X_COSTS


# ---------------------------------------------------------------------------
# YAML保存
# ---------------------------------------------------------------------------
def save_yaml(corpus: dict) -> None:
    try:
        import yaml
        yaml_path = os.path.join(AUDIT_DIR, 'reports_corpus.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(corpus, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        print(f'Saved: audit_results/reports_corpus.yaml')
    except ImportError:
        import json
        json_path = os.path.join(AUDIT_DIR, 'reports_corpus.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        print(f'PyYAML未インストール。JSONで代替出力: audit_results/reports_corpus.json')


# ---------------------------------------------------------------------------
# 差分マトリックス計算
# ---------------------------------------------------------------------------
def compute_gap_matrix() -> list:
    """バックテストシム値 vs Corpus実商品の差分マトリックス"""
    rows = []

    # TQQQ
    tqqq_c = CORPUS['tqqq']
    rows.append({
        'product': 'TQQQ',
        'item': 'TER',
        'sim_value': TQQQ_COSTS.ter,
        'realworld_value': tqqq_c['ter'],
        'gap_bps': (tqqq_c['ter'] - TQQQ_COSTS.ter) * 10000,
        'confidence': tqqq_c['ter_confidence'],
        'verdict': 'PASS' if abs(tqqq_c['ter'] - TQQQ_COSTS.ter) * 10000 <= 5 else 'WARN',
    })
    rows.append({
        'product': 'TQQQ',
        'item': 'SOFR multiplier',
        'sim_value': TQQQ_COSTS.sofr_multiplier,
        'realworld_value': tqqq_c['sofr_multiplier'],
        'gap_bps': (tqqq_c['sofr_multiplier'] - TQQQ_COSTS.sofr_multiplier) * 10000,
        'confidence': tqqq_c['sofr_multiplier_confidence'],
        'verdict': 'PASS' if tqqq_c['sofr_multiplier'] == TQQQ_COSTS.sofr_multiplier else 'WARN',
    })

    # TMF - ter差分
    tmf_c = CORPUS['tmf']
    rows.append({
        'product': 'TMF',
        'item': 'TER (sim vs current)',
        'sim_value': TMF_COSTS.ter,
        'realworld_value': tmf_c['ter_current'],
        'gap_bps': (tmf_c['ter_current'] - TMF_COSTS.ter) * 10000,
        'confidence': tmf_c['ter_current_confidence'],
        'verdict': 'WARN' if (tmf_c['ter_current'] - TMF_COSTS.ter) * 10000 > 10 else 'PASS',
    })

    # GOLD2X - TER差分（シムproxy vs 実商品UGL）
    gold_c = CORPUS['ugl_gold2x']
    rows.append({
        'product': 'Gold 2x',
        'item': 'TER (sim WisdomTree vs UGL)',
        'sim_value': GOLD2X_COSTS.ter,         # 0.0095（UGL）だが simでは0.0049使用
        'realworld_value': gold_c['ter'],        # 0.0095 UGL
        'gap_bps': (gold_c['ter'] - gold_c['bt_sim_ter']) * 10000,  # 0.95%-0.49%=46bps
        'confidence': 'disclosed',
        'verdict': 'WARN',  # 46bps差は要注意
        'note': f'シムproxy({gold_c["bt_sim_ter"]*100:.2f}%) vs 実商品UGL({gold_c["ter"]*100:.2f}%)',
    })

    # CFD_SPREAD_LOW の実商品比較
    rows.append({
        'product': 'CFD(NASDAQ)',
        'item': 'Annual spread (bt: 0.20%) vs くりっく365 funding (4.92%)',
        'sim_value': 0.0020,
        'realworld_value': 0.0492,
        'gap_bps': (0.0492 - 0.0020) * 10000,  # 472bps
        'confidence': 'disclosed',
        'verdict': 'FAIL',  # 大きな乖離
        'note': 'btの0.20%はスプレッド相当。くりっく365の実際のfundingコスト4.92%とは概念が異なる',
    })
    rows.append({
        'product': 'CFD(NASDAQ)',
        'item': 'Annual spread (bt: 0.20%) vs SBI/楽天/IG funding (SOFR+3%≈6.6%)',
        'sim_value': 0.0020,
        'realworld_value': 0.066,
        'gap_bps': (0.066 - 0.0020) * 10000,   # 640bps
        'confidence': 'disclosed',
        'verdict': 'FAIL',
        'note': 'SBI/楽天/IGのオーバーナイト金利（SOFR+3%）との乖離は約640bps',
    })

    return rows


# ---------------------------------------------------------------------------
# 合格判定
# ---------------------------------------------------------------------------
def evaluate_corpus(corpus: dict) -> tuple:
    issues = []
    if len(corpus) < 14:
        issues.append(f'商品数 {len(corpus)} < 14')
    for k, v in corpus.items():
        # confidence フィールドの存在確認（'confidence'または*_confidenceがあればOK）
        has_confidence = any('confidence' in kk for kk in v.keys())
        if not has_confidence:
            issues.append(f'{k}: confidence フィールドなし')
    # TMF ter_current の明記
    tmf = corpus.get('tmf', {})
    if 'ter_current' not in tmf:
        issues.append('tmf: ter_current 未記載')
    # Gold2x の bt_sim 商品明記
    ugl = corpus.get('ugl_gold2x', {})
    if 'bt_sim_ter' not in ugl:
        issues.append('ugl_gold2x: bt_sim_ter 未記載')
    verdict = 'PASS' if not issues else 'FAIL'
    return verdict, issues


# ---------------------------------------------------------------------------
# MD生成
# ---------------------------------------------------------------------------
def _yn(val: bool) -> str:
    return '✅' if val else '❌'


def _confidence_main(entry: dict) -> str:
    """主要なconfidenceフィールドを1つ返す"""
    for k, v in entry.items():
        if k.endswith('_confidence') and isinstance(v, str):
            return f'{k}={v}'
    if 'confidence' in entry:
        return f'confidence={entry["confidence"]}'
    return '-'


def generate_md(corpus: dict, gap_rows: list, verdict: str, issues: list) -> str:
    today_str = date.today().strftime('%Y-%m-%d')
    lines = []

    lines.append('# Phase 2 実現性チェック (2.8): 商品コーパス根拠台帳')
    lines.append('')
    lines.append(f'**実行日**: {today_str}')
    lines.append('**対象レポート**: 3本のDeep Researchレポート（2026-05-15, 2026-05-17, 2026-05-20）')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append(f'## 1. 抽出商品一覧（{len(corpus)}商品）')
    lines.append('')
    lines.append('| # | 商品キー | 商品名 | カテゴリ | SBI取引可否 | Confidence主要項目 |')
    lines.append('|---|---------|--------|---------|------------|-----------------|')
    for i, (key, entry) in enumerate(corpus.items(), 1):
        name = entry.get('name', key)
        category = entry.get('category', '-')
        avail = entry.get('available_at_sbi')
        avail_str = '✅' if avail is True else ('❌' if avail is False else '-')
        conf_str = _confidence_main(entry)
        lines.append(f'| {i} | `{key}` | {name} | {category} | {avail_str} | {conf_str} |')

    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 2. バックテスト定数 vs 実商品 差分マトリックス')
    lines.append('')
    lines.append('| 商品 | 項目 | シム値 | 実商品値 | 差分(bps) | 信頼度 | 判定 |')
    lines.append('|-----|-----|-------|---------|---------|------|-----|')
    for row in gap_rows:
        product = row['product']
        item = row['item']
        sim_v = f"{row['sim_value']*100:.4f}%" if isinstance(row['sim_value'], float) else str(row['sim_value'])
        real_v = f"{row['realworld_value']*100:.4f}%" if isinstance(row['realworld_value'], float) else str(row['realworld_value'])
        gap_bps = f"{row['gap_bps']:.1f}"
        conf = row.get('confidence', '-')
        verdict_row = row.get('verdict', '-')
        verdict_icon = {'PASS': '✅ PASS', 'WARN': '⚠️ WARN', 'FAIL': '❌ FAIL'}.get(verdict_row, verdict_row)
        note = row.get('note', '')
        note_str = f'<br>*{note}*' if note else ''
        lines.append(f'| {product} | {item} | {sim_v} | {real_v} | {gap_bps} | {conf} | {verdict_icon}{note_str} |')

    lines.append('')
    lines.append('### ⚠️ 重要発見')
    lines.append('')
    lines.append('CFD NASDASQスリーブのコスト乖離:')
    lines.append('- バックテスト: `CFD_SPREAD_LOW = 0.20%/yr`（スプレッド相当）')
    lines.append('- くりっく株365実コスト: `4.92%/yr`（ポジション全体にかかるfunding）**→ 差分 472bps**')
    lines.append('- SBI/楽天/IG CFD実コスト: `SOFR+3% ≈ 6.6%/yr`**→ 差分 640bps**')
    lines.append('')
    lines.append('これはコストモデルの概念的な乖離であり、check_sim_broker_matrix_cagr.py (2.14)で定量評価する。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 3. 合格判定')
    lines.append('')
    lines.append('合格基準:')

    # 商品数
    count_ok = len(corpus) >= 14
    lines.append(f'- 抽出商品数 ≥ 14: {"✅" if count_ok else "❌"} ({len(corpus)}商品)')

    # confidence設定済み
    conf_issues = [iss for iss in issues if 'confidence' in iss]
    lines.append(f'- confidence設定済み: {"✅" if not conf_issues else "❌"}')

    # TMF ter_current
    tmf_ok = 'ter_current' in corpus.get('tmf', {})
    lines.append(f'- TMF ter_current明記: {"✅" if tmf_ok else "❌"}')

    # Gold2x bt_sim_ter
    gold_ok = 'bt_sim_ter' in corpus.get('ugl_gold2x', {})
    lines.append(f'- Gold2xシムproxyとの乖離明記: {"✅" if gold_ok else "❌"}')

    lines.append('')
    if issues:
        lines.append('課題:')
        for iss in issues:
            lines.append(f'- {iss}')
        lines.append('')

    lines.append(f'**総合判定: {verdict}**')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 4. 出力ファイル')
    lines.append('')
    lines.append('- `audit_results/reports_corpus.yaml`: 商品コーパス（後続チェックスクリプトの入力源）')
    lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 60)
    print('Phase 2 チェック 2.8: 商品コーパス根拠台帳生成')
    print('=' * 60)

    # 1. YAML出力
    save_yaml(CORPUS)

    # 2. 差分マトリックス計算
    gap_rows = compute_gap_matrix()

    # 3. 合格判定
    verdict, issues = evaluate_corpus(CORPUS)

    # 4. コンソール出力
    print(f'\n商品数: {len(CORPUS)}')
    print(f'判定: {verdict}')
    if issues:
        for iss in issues:
            print(f'  ISSUE: {iss}')

    # 5. MD出力
    md = generate_md(CORPUS, gap_rows, verdict, issues)
    md_path = os.path.join(AUDIT_DIR, f'CHECK_CORPUS_{TODAY}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved: {md_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
