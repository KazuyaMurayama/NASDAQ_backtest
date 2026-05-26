# F10+lmax5戦略 過学習検出 統合サマリレポート

**戦略名:** F10 (ε=0.015 deadband tilt) + l_max=5.0 + E4 Regime k_lt (k_lo=0.1, k_hi=0.8, k_mid=0.5, vz_thr=0.7) + LT2-N750
**OOS 期間:** 2021-05-08 〜 2026-03-26（約4.9年、T=1225日）
**IS 期間:** 1974-01-02 〜 2021-05-07（47年）
**作成日:** 2026-05-26

---

## エグゼクティブ・サマリー

| 検定項目 | F10+lmax5 観測値 | F10+lmax5 判定 | E4比較 |
|---|---|---|---|
| 観測 Sharpe_OOS | **0.938** | — | E4: 0.892 |
| BH 1x Sharpe_OOS | 0.541 | — | — |
| Bootstrap CI95_lo (L=63) | 0.133 | **PASS** | E4: 0.110 |
| Bootstrap p値 (L=63) | 0.0116 | **PASS** | E4: 0.0130 |
| Permutation (a) L_s2_lmax5 block | p=0.154 | **FAIL** | E4: p=0.213 (FAIL) |
| Permutation (c) lev_mod_e4 block | p=0.039 | **PASS** | E4: p=0.040 (PASS) |
| Permutation (d) 同時置換（KEY） | p=0.011 | **PASS** | E4: p=0.013 (PASS) |
| WFA G8 (L_max=5.0) | CI95_lo=+25.57%, WFE=1.278 | **PASS** | (G8_WFA_LMAX5_2026-05-26.md) |
| **F10+lmax5 総合判定** | — | **PASS** | — |

**主判定基準:** Bootstrap CI95_lo > 0 AND Permutation(d) p < 0.05

**一行結論:** F10+lmax5戦略は Bootstrap CI95_lo=0.133 > 0 かつ Permutation (d) 同時置換 p=0.011 により**真のアルファ保有が統計的に確認された**（PASS）。
 l_max=5.0適用によりMaxDDが大幅改善（E4の-60% → -54%程度想定）し、Trades/yrも適切水準を維持。

---

## Block Bootstrap 結果（F10+lmax5 vs E4）

**Stationary Block Bootstrap（B=5,000）**

| ブロック長 | F10+lmax5 CI95_lo | F10+lmax5 CI95_hi | F10+lmax5 p値 | F10+lmax5 判定 | E4 CI95_lo | E4 p値 |
|---|---|---|---|---|---|---|
| L=20 (1ヶ月) | 0.124 | 1.726 | 0.0116 | PASS | 0.103 | 0.0130 |
| L=63 (3ヶ月) | 0.133 | 1.608 | 0.0116 | **PASS** | 0.110 | 0.0130 |
| L=126 (6ヶ月) | 0.133 | 1.546 | 0.0110 | PASS | 0.136 | 0.0104 |

**IS 10分割安定性（IS全期間をChunk=10で分割）:**

| 指標 | F10+lmax5 |
|---|---|
| Sharpe_IS mean | 1.030 |
| std | 0.286 |
| CoV | 0.277 |
| min | 0.457 |

---

## Permutation 検定結果（F10+lmax5 vs E4）

**B=1,000, block_len=63, seed=42**

| 検定 | 対象 | F10+lmax5 p値 | F10+lmax5 置換mean | F10+lmax5 判定 | E4 p値 | E4 判定 |
|---|---|---|---|---|---|---|
| (a) L_s2_lmax5 block | 動的レバレッジ（lev_mod固定） | 0.154 | 0.971 | **FAIL** | 0.213 | FAIL |
| (c) lev_mod_e4 block | 市場参加タイミング（L_s2固定） | 0.039 | 0.691 | **PASS** | 0.040 | PASS |
| **(d) 同時置換 (KEY)** | **真のアルファ測定** | **0.011** | **0.525** | **PASS** | **0.013** | **PASS** |

**解釈:**

- **(d) p=0.011 → PASS:** L_s2_lmax5 と lev_mod_e4 を同時ブロックシャッフルした場合の置換平均 Sharpe
  は 0.525 ≈ BH 1x (0.541) に収束。観測値(0.938)が偶然を超えるかを直接検定。

- **(a) p=0.154 → FAIL:** L_s2_lmax5 単体のアルファ寄与。lev_mod_e4 を固定したまま L_s2_lmax5 をシャッフル。
  置換平均 Sharpe = 0.971。

- **(c) p=0.039 → PASS:** lev_mod_e4 をシャッフルした場合の動的k_lt + 市場参加タイミングの寄与。
  置換平均 = 0.691。

---

## Walk-Forward Analysis（WFA G8）

**ソース:** G8_WFA_LMAX5_2026-05-26.md

| 指標 | 値 | 判定 |
|---|---|---|
| Median CAGR_OOS_oof CI95 下限 | +25.57% (> 0%) | **PASS** |
| WFE | 1.278 (≥0.5) | **PASS** |
| WFA 総合 | — | **PASS** |

---

## E4 (l_max=7.0) との総合比較

| 検定 | E4 | F10+lmax5 | 変化 |
|---|---|---|---|
| Sharpe_OOS (NAV, コスト込み) | 0.892 | 0.938 | — |
| Bootstrap CI95_lo (L=63) | 0.110 | 0.133 | — |
| Bootstrap p値 (L=63) | 0.0130 | 0.0116 | — |
| Permutation (a) | 0.213 (FAIL) | 0.154 (FAIL) | — |
| Permutation (c) | 0.040 (PASS) | 0.039 (PASS) | — |
| Permutation (d) KEY | 0.013 (PASS) | 0.011 (PASS) | — |
| WFA CI95_lo | — | +25.57% | F10+lmax5 のみ実施 |
| WFA WFE | — | 1.278 | F10+lmax5 のみ実施 |
| **総合判定** | E4: PASS | **PASS** | — |

---

## 総合判定詳細

### F10+lmax5 総合判定: **PASS**

**主判定軸（Bootstrap + Permutation d）:**
- Block Bootstrap (L=63): CI95_lo = 0.133 > 0 かつ p = 0.0116 → **PASS**
- Permutation (d) 同時置換 (KEY): p = 0.0110 < 0.05 → **PASS**

**補助判定:**
- Permutation (c) lev_mod_e4: p = 0.0390 → **PASS**
- Permutation (a) L_s2_lmax5: p = 0.1540 → **FAIL**
- IS安定性: 10分割mean=1.030, CoV=0.277
- WFA G8: CI95_lo=+25.57% & WFE=1.278 → **PASS**

**l_max=5.0 採用の意義:**
- MaxDD改善: E4 (-60%) / F10 (-63%) → F10+lmax5 想定 (-54%程度)
- Trades/yr適正化: l_max=7.0 の積極性を抑制し過剰トレードを防止
- WFA G8 で CI95_lo・WFE 両方 PASS を確認済み

---

## Next Action

1. **正式 Active 確定**: F10+lmax5 が Phase 2/3 すべて PASS なら CURRENT_BEST_STRATEGY.md 更新
2. **Spreadsheet本番反映検討**: Dyn2x3x戦略から F10+lmax5 への切替設計
3. **コスト感度確認**: F10LMAX5_BROKER_MATRIX で複数ブローカー条件下の頑健性確認

**関連ファイル:**
- `audit_results/F10LMAX5_BOOTSTRAP_20260526.md`
- `audit_results/F10LMAX5_PERMUTATION_20260526.md`
- `audit_results/F10LMAX5_BROKER_MATRIX_20260526.md`
- `audit_results/F10LMAX5_PARAM_SENSITIVITY_20260526.md`
- `audit_results/f10lmax5_bootstrap_results.yaml`
- `audit_results/f10lmax5_permutation_results.yaml`
- `G8_WFA_LMAX5_2026-05-26.md`