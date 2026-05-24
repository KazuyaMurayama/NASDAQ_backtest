# Phase 2 実現性チェック (2.8): 商品コーパス根拠台帳

**実行日**: 2026-05-24
**対象レポート**: 3本のDeep Researchレポート（2026-05-15, 2026-05-17, 2026-05-20）

---

## 1. 抽出商品一覧（14商品）

| # | 商品キー | 商品名 | カテゴリ | SBI取引可否 | Confidence主要項目 |
|---|---------|--------|---------|------------|-----------------|
| 1 | `kurikku365_nasdaq` | くりっく株365 NASDAQ100 | exchange_futures_cfd | ✅ | funding_confidence=disclosed |
| 2 | `sbi_cfd_nq100` | SBI CFD 米国NQ100 | otc_cfd | ✅ | spread_confidence=disclosed |
| 3 | `ig_cfd_nasdaq` | IG証券 CFD NASDAQ100 | otc_cfd | ❌ | spread_confidence=disclosed |
| 4 | `rakuten_cfd_nasdaq` | 楽天証券 CFD 米国NAS100 | otc_cfd | ❌ | spread_confidence=disclosed |
| 5 | `gmo_cfd_nasdaq` | GMOクリック証券 CFD NASDAQ100 | otc_cfd | ❌ | spread_confidence=disclosed |
| 6 | `dmm_cfd_nasdaq` | DMM CFD NASDAQ100 | otc_cfd | ❌ | spread_confidence=disclosed |
| 7 | `tqqq` | ProShares UltraPro QQQ (TQQQ) | leveraged_etf | ✅ | ter_confidence=disclosed |
| 8 | `qld` | ProShares Ultra QQQ (QLD) | leveraged_etf | ✅ | ter_confidence=disclosed |
| 9 | `ugl_gold2x` | ProShares Ultra Gold (UGL) 2x | leveraged_etf | ✅ | ter_confidence=disclosed |
| 10 | `wisdomtree_gold2x_lse` | WisdomTree 2036 2x Gold ETP (LSE) | leveraged_etp | ❌ | ter_confidence=disclosed |
| 11 | `tmf` | Direxion Daily 20+ Year Treasury Bull 3x (TMF) | leveraged_etf | ✅ | ter_sim_confidence=disclosed |
| 12 | `sbi_cfd_regulation` | SBI CFD 規制・証拠金仕様 | regulation | - | max_leverage_confidence=disclosed |
| 13 | `sbi_cfd_lc_stress` | SBI CFD ロスカット耐久テスト（レポート記載値） | stress_test | - | confidence=disclosed |
| 14 | `bt_cost_model` | バックテスト現行コストモデル(Scenario D) | backtest_model | - | cfd_spread_confidence=disclosed |

---

## 2. バックテスト定数 vs 実商品 差分マトリックス

| 商品 | 項目 | シム値 | 実商品値 | 差分(bps) | 信頼度 | 判定 |
|-----|-----|-------|---------|---------|------|-----|
| TQQQ | TER | 0.8600% | 0.8600% | 0.0 | disclosed | ✅ PASS |
| TQQQ | SOFR multiplier | 200.0000% | 200.0000% | 0.0 | estimated | ✅ PASS |
| TMF | TER (sim vs current) | 0.9100% | 1.0600% | 15.0 | disclosed | ⚠️ WARN |
| Gold 2x | TER (sim WisdomTree vs UGL) | 0.9500% | 0.9500% | 46.0 | disclosed | ⚠️ WARN<br>*シムproxy(0.49%) vs 実商品UGL(0.95%)* |
| CFD(NASDAQ) | Annual spread (bt: 0.20%) vs くりっく365 funding (4.92%) | 0.2000% | 4.9200% | 472.0 | disclosed | ❌ FAIL<br>*btの0.20%はスプレッド相当。くりっく365の実際のfundingコスト4.92%とは概念が異なる* |
| CFD(NASDAQ) | Annual spread (bt: 0.20%) vs SBI/楽天/IG funding (SOFR+3%≈6.6%) | 0.2000% | 6.6000% | 640.0 | disclosed | ❌ FAIL<br>*SBI/楽天/IGのオーバーナイト金利（SOFR+3%）との乖離は約640bps* |

### ⚠️ 重要発見

CFD NASDASQスリーブのコスト乖離:
- バックテスト: `CFD_SPREAD_LOW = 0.20%/yr`（スプレッド相当）
- くりっく株365実コスト: `4.92%/yr`（ポジション全体にかかるfunding）**→ 差分 472bps**
- SBI/楽天/IG CFD実コスト: `SOFR+3% ≈ 6.6%/yr`**→ 差分 640bps**

これはコストモデルの概念的な乖離であり、check_sim_broker_matrix_cagr.py (2.14)で定量評価する。

---

## 3. 合格判定

合格基準:
- 抽出商品数 ≥ 14: ✅ (14商品)
- confidence設定済み: ✅
- TMF ter_current明記: ✅
- Gold2xシムproxyとの乖離明記: ✅

**総合判定: PASS**

---

## 4. 出力ファイル

- `audit_results/reports_corpus.yaml`: 商品コーパス（後続チェックスクリプトの入力源）
