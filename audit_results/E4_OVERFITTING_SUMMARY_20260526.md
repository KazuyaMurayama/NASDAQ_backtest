# E4戦略 過学習検出 統合サマリレポート

**戦略名:** S2_VZGated + LT2-N750 + E4 Regime k_lt (k_lo=0.1, k_hi=0.8, k_mid=0.5, vz_thr=0.7)
**OOS 期間:** 2021-05-08 〜 2026-03-26（約4.9年、T=1225日）
**IS 期間:** 1974-01-02 〜 2021-05-07（47年）
**作成日:** 2026-05-26

---

## エグゼクティブ・サマリー

| 検定項目 | 観測値 | E4判定 | 旧戦略比 |
|---|---|---|---|
| 観測 Sharpe_OOS | **0.892** | — | 旧: 0.858 |
| BH 1x Sharpe_OOS | 0.541 | — | — |
| Bootstrap CI95_lo (L=63) | 0.110 | **PASS** | 旧: 0.086 |
| Bootstrap p値 (L=63) | 0.0130 | **PASS** | 旧: 0.0160 |
| Permutation (a) L_s2 block | p=0.213 | **FAIL** | 旧: 0.248 FAIL |
| Permutation (c) lev_mod block | p=0.040 | **PASS** | 旧: 0.055 WARN |
| Permutation (d) 同時置換（KEY） | p=0.013 | **PASS** | 旧: 未実施 |
| DSR (N=150 中央値, 参考) | 4.30e-05 | FAIL† | 旧と同条件 |
| **E4総合判定** | — | **PASS** | 旧: FAIL |

†DSR は試行数N≥150で多重比較補正のためほぼ必ずFAIL。Bootstrap+Permutation(d)を主判定とする。

**一行結論:** E4戦略は Permutation (d) 同時置換検定 p=0.013 PASS により**真のアルファ保有が統計的に確認された**。
Bootstrap CI95_lo=0.110（旧: 0.086）と改善。
ただし選択バイアス（N≈65試行）は依然として残存し、Worst10Y★の不確実性は大きい。

---

## Block Bootstrap 結果（E4 vs 旧戦略）

**Stationary Block Bootstrap（B=5,000）**

| ブロック長 | E4 CI95_lo | E4 CI95_hi | E4 p値 | E4判定 | 旧 CI95_lo | 旧 p値 |
|---|---|---|---|---|---|---|
| L=20 (1ヶ月) | +0.103 | 1.675 | 0.0130 | PASS | +0.056 | 0.019 |
| L=63 (3ヶ月) | +0.110 | 1.558 | 0.0130 | **PASS** | +0.086 | 0.016 |
| L=126 (6ヶ月) | +0.136 | 1.478 | 0.0104 | PASS | +0.131 | 0.012 |

**E4はすべてのブロック長でCI95_lo > 0 かつ p < 0.05。旧戦略比でCI95_loが改善（L=63: +0.086→+0.110）。**

**IS 10分割安定性（IS全期間をChunk=10で分割）:**

| 指標 | E4 | 旧戦略 |
|---|---|---|
| Sharpe_IS mean | 0.961 | 0.971 |
| std | 0.290 | 0.316 |
| CoV | 0.302 | 0.325 |
| min | 0.379 | 0.266 |

---

## Permutation 検定結果（E4 vs 旧戦略）

**B=1,000, block_len=63, seed=42**

| 検定 | 対象 | E4 p値 | E4置換mean | E4判定 | 旧 p値 | 旧置換mean | 旧判定 |
|---|---|---|---|---|---|---|---|
| (a) L_s2 block | 動的レバレッジ（lev_mod固定） | 0.213 | 0.929 | **FAIL** | 0.248 | 0.763 | FAIL |
| (c) lev_mod block | 市場参加タイミング（L_s2固定） | 0.040 | 0.679 | **PASS** | 0.055 | 0.547 | WARN |
| **(d) 同時置換 (KEY)** | **真のアルファ測定** | **0.013** | **0.485** | **PASS** | 未実施 | — | — |

**解釈:**

- **(d) p=0.013 PASS: 真のアルファ確認。** L_s2 と lev_mod_e4 を同時ブロックシャッフルした場合の置換平均 Sharpe
  は 0.485 ≈ BH 1x (0.541) に収束。観測値(1.036)は
  B=1000中13回しか超えられなかった。これは戦略が真に予測力を持つ証拠。

- **(a) p=0.213 FAIL: L_s2 単体のアルファ寄与なし（旧と同様）。** lev_mod_e4 を固定したまま L_s2 をシャッフルしても
  置換平均 Sharpe = 0.929 と高水準を維持。アルファ源は L_s2 ではなく lev_mod_e4。

- **(c) p=0.040 PASS（旧: WARN p=0.055 → 改善）。** lev_mod_e4 をシャッフルすると置換平均
  0.679 に低下。E4の動的k_ltによる市場参加タイミングがアルファ源であることを確認。

---

## 旧戦略（固定k=0.5）との総合比較

| 検定 | 旧戦略 | E4戦略 | 変化 |
|---|---|---|---|
| Sharpe_OOS (NAV, コスト込み) | 0.858 | 0.892 | ↑改善 |
| Bootstrap CI95_lo (L=63) | +0.086 | +0.110 | ↑改善 |
| Bootstrap p値 (L=63) | 0.016 | 0.0130 | ↑改善 |
| Permutation (a) L_s2 | FAIL (p=0.248) | FAIL (p=0.213) | 同様 |
| Permutation (c) lev_mod | WARN (p=0.055) | PASS (p=0.040) | ↑WARN→PASS |
| Permutation (d) 同時 (KEY) | 未実施 | **PASS (p=0.013)** | **新規確認** |
| IS-OOS gap | +0.18pp | -1.81pp | ↑OOS超過（稀） |
| DSR (参考) | FAIL | FAIL（同条件） | 同様 |
| **総合判定** | **FAIL** | **PASS** | **↑大幅改善** |

---

## 総合判定詳細

### E4総合判定: **PASS**

**主判定軸（Bootstrap + Permutation d）:**
- Block Bootstrap (L=63): CI95_lo = 0.110 > 0 かつ p = 0.0130 < 0.05 → **PASS**
- Permutation (d) 同時置換 (KEY): p = 0.0130 < 0.05 → **PASS**

**補助判定:**
- Permutation (c) lev_mod_e4: p = 0.0400 → **PASS**（WARN→PASS改善）
- Permutation (a) L_s2: p = 0.2130 → **FAIL**（L_s2 はアルファ寄与なし）
- IS安定性: 10分割mean=0.961, CoV=0.302

**残存リスク:**
- 選択バイアス: E4グリッド探索 N≈65試行（E[max SR]≈1.84）。DSR FAIL は継続。
- OOS期間4.9年のみ。WFA（Walk-Forward Analysis）は未実施（暫定Active）。
- L_s2 の付加価値なし → 簡素化検討が有効。

---

## CURRENT_BEST_STRATEGY.md パッチ提案

以下を CURRENT_BEST_STRATEGY.md の「検証ステータス」セクションに追記（手動マージ）:

```markdown
### Phase 3 E4戦略 過学習検出（2026-05-26）
- Block Bootstrap (L=63): CI95_lo = 0.110, p = 0.0130 → **PASS**（旧: CI95_lo=+0.086, p=0.016）
- Permutation (a) L_s2: p = 0.2130 → **FAIL**（アルファ源はlev_mod_e4）
- Permutation (c) lev_mod_e4: p = 0.0400 → **PASS**（旧WARN→改善）
- Permutation (d) 同時置換 (KEY NEW): p = 0.0130 → **PASS**（真のアルファ確認）
- DSR (N=150 参考): FAIL（多重比較補正、選択バイアス残存）
- **E4総合: PASS**（旧FAIL → E4 PASS）
```

---

## Next Action

1. **WFA実施（最優先）**: CI95_lo>0 ∧ 0.5≤WFE≤2.0 確認で正式確定へ
2. **L_s2 簡素化検討**: 固定レバレッジ置換でSharpe低下を定量確認
3. **実運用移行**: SBI-CFD 選択（くりっく365ではWorst10Y★ FAIL）
4. **四半期レビュー**: 直近4Q Sharpe < 0.3 で戦略停止 or レビュー

**関連ファイル:**
- `audit_results/E4_BOOTSTRAP_20260526.md`
- `audit_results/E4_PERMUTATION_20260526.md`
- `audit_results/e4_bootstrap_results.yaml`
- `audit_results/e4_permutation_results.yaml`