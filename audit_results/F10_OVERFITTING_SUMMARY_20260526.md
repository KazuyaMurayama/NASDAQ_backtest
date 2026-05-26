# F10戦略 過学習検出 統合サマリレポート

**戦略名:** F10 = F8-R5 (tilt=10.0, CALM_BOOST) + ε=0.015 deadband
  + E4 lev_mod_e4 (k_lo=0.1, k_hi=0.8, k_mid=0.5, vz_thr=0.7)
  + S2 L_s2 (l_max=7.0)
**OOS 期間:** 2021-05-08 〜 2026-03-26（約4.9年、T=1225日）
**IS 期間:** 1974-01-02 〜 2021-05-07（47年）
**作成日:** 2026-05-26

---

## エグゼクティブ・サマリー

| 検定項目 | 観測値 | F10判定 | E4比較 |
|---|---|---|---|
| 観測 Sharpe_OOS | **0.935** | — | E4: 0.892 |
| BH 1x Sharpe_OOS | 0.541 | — | — |
| Bootstrap CI95_lo (L=63) | 0.162 | **PASS** | E4: 0.110 |
| Bootstrap p値 (L=63) | 0.0102 | **PASS** | E4: 0.0130 |
| Permutation (a) L_s2 block | p=0.184 | **FAIL** | E4: 0.213 FAIL |
| Permutation (c) lev_mod block | p=0.038 | **PASS** | E4: 0.040 PASS |
| Permutation (d) 同時置換（KEY） | p=0.012 | **PASS** | E4: 0.013 PASS |
| WFA CI95_lo (G7, 50窓) | +27.91% | **PASS** | E4: +26.51% |
| WFA WFE (G7, 50窓) | 1.208 | **PASS** | E4: 1.131 |
| **F10総合判定** | — | **PASS** | E4: PASS |

**判定基準:** PASS = (Bootstrap CI95_lo > 0) AND (Permutation (d) p < 0.05) AND (WFA CI95_lo > 0 ∧ 0.5 ≤ WFE ≤ 2.0)

**一行結論:** F10戦略は Bootstrap CI95_lo = 0.162 > 0, Permutation (d) p = 0.0120,
WFA CI95_lo = +27.91%, WFE = 1.208 の3軸検定で**真のアルファ保有が確認された** (PASS)。
ε=0.015 deadband 適用後も E4 (旧 Active, WFA CI95_lo=+26.51%) を上回るロバスト性を維持。

---

## Block Bootstrap 結果（F10 vs E4）

**Stationary Block Bootstrap（B=5,000）**

| ブロック長 | F10 CI95_lo | F10 CI95_hi | F10 p値 | F10判定 | E4 CI95_lo | E4 p値 |
|---|---|---|---|---|---|---|
| L=20 (1ヶ月) | 0.135 | 1.714 | 0.0098 | PASS | 0.103 | 0.0130 |
| L=63 (3ヶ月) | 0.162 | 1.588 | 0.0102 | **PASS** | 0.110 | 0.0130 |
| L=126 (6ヶ月) | 0.181 | 1.522 | 0.0080 | PASS | 0.136 | 0.0104 |

**IS 10分割安定性（nav_f10）:**

| 指標 | F10 |
|---|---|
| Sharpe_IS mean | 0.959 |
| std | 0.298 |
| CoV | 0.311 |
| min | 0.354 |

---

## Permutation 検定結果（F10 vs E4）

**B=1,000, block_len=63, seed=42**

| 検定 | 対象 | F10 p値 | F10置換mean | F10判定 | E4 p値 | E4判定 |
|---|---|---|---|---|---|---|
| (a) L_s2 block | 動的レバレッジ（lev_mod固定） | 0.184 | 0.958 | **FAIL** | 0.213 | FAIL |
| (c) lev_mod_e4 block | 市場参加タイミング（L_s2固定） | 0.038 | 0.710 | **PASS** | 0.040 | PASS |
| **(d) 同時置換 (KEY)** | **真のアルファ測定** | **0.012** | **0.516** | **PASS** | 0.013 | PASS |

**解釈:**

- **(d) p=0.012 → PASS: 真のアルファ判定** L_s2 と lev_mod_e4 を同時ブロックシャッフルした場合の置換平均 Sharpe
  は 0.516 ≈ BH 1x (0.541) に収束。観測値(0.935)は
  B=1000中の限定的な置換でしか超えられなかった。これは F10 戦略が真に予測力を持つ証拠。

- **(a) p=0.184 → FAIL: L_s2 単体のアルファ寄与判定** lev_mod_e4 を固定したまま L_s2 をシャッフルしたときの結果。

- **(c) p=0.038 → PASS: lev_mod_e4 単体のアルファ寄与判定** L_s2 を固定したまま lev_mod_e4 をシャッフルしたときの結果。

---

## Walk-Forward Analysis (G7) 結果

参照: `G7_WFA_F10_2026-05-26.md`

| 指標 | F10 | E4 (旧 Active) | 差 |
|---|---|---|---|
| WFA CI95_lo | +27.91% | +26.51% | +1.40pp |
| WFA WFE | 1.208 | 1.131 | +0.077 |
| WFA t_p | 0.0000 | < 0.0001 | — |
| 判定 | **PASS** | PASS | — |

**F10 WFA は E4 を CI95_lo +1.40pp、WFE +0.077 で上回る。** ε=0.015 deadband 適用後も 50窓 Out-of-Sample 安定性は維持された。

---

## F10 vs E4 総合比較

| 検定 | E4戦略 | F10戦略 | 変化 |
|---|---|---|---|
| Sharpe_OOS (NAV, コスト込み) | 0.892 | 0.935 | +0.043 |
| Bootstrap CI95_lo (L=63) | 0.110 | 0.162 | +0.052 |
| Bootstrap p値 (L=63) | 0.0130 | 0.0102 | — |
| Permutation (a) L_s2 | FAIL (p=0.213) | FAIL (p=0.184) | — |
| Permutation (c) lev_mod | PASS (p=0.040) | PASS (p=0.038) | — |
| Permutation (d) 同時 (KEY) | PASS (p=0.013) | **PASS (p=0.012)** | — |
| WFA CI95_lo | +26.51% | +27.91% | +1.40pp |
| WFA WFE | 1.131 | 1.208 | +0.077 |
| **総合判定** | **PASS** | **PASS** | — |

---

## 総合判定詳細

### F10総合判定: **PASS**

**主判定軸（Bootstrap + Permutation d + WFA）:**
- Block Bootstrap (L=63): CI95_lo = 0.162 > 0 かつ p = 0.0102 → **PASS**
- Permutation (d) 同時置換 (KEY): p = 0.0120 → **PASS**
- WFA (G7, 50窓): CI95_lo = +27.91%, WFE = 1.208 → **PASS**

**補助判定:**
- Permutation (c) lev_mod_e4: p = 0.0380 → **PASS**
- Permutation (a) L_s2: p = 0.1840 → **FAIL**
- IS安定性: 10分割mean=0.959, CoV=0.311

**残存リスク:**
- 選択バイアス: F10 (ε-deadband スイープ) で局所最適化された可能性。DSR FAIL は継続（多重比較補正）。
- ε=0.015 は離散最適候補。感度分析（F10_PARAM_SENSITIVITY）で台地性を確認。
- E4 と比較し IS-OOS gap が拡大（注: G7_WFA_F10 §記載のとおり -4.31pp gap、WFAは PASS で許容）。

---

## CURRENT_BEST_STRATEGY.md パッチ提案

以下を CURRENT_BEST_STRATEGY.md に追記候補:

```markdown
### Phase 3 F10戦略 過学習検出（2026-05-26）
- Block Bootstrap (L=63): CI95_lo = 0.162, p = 0.0102 → **PASS**
- Permutation (a) L_s2: p = 0.1840 → **FAIL**
- Permutation (c) lev_mod_e4: p = 0.0380 → **PASS**
- Permutation (d) 同時置換 (KEY): p = 0.0120 → **PASS**
- WFA (G7, 50窓): CI95_lo = +27.91%, WFE = 1.208 → **PASS**
- **F10総合: PASS**（E4 PASS と同等以上、WFA で E4 を上回る）
```

---

## Next Action

1. **F10 を Active 昇格判断**: WFA + Bootstrap + Permutation すべて PASS なら CURRENT_BEST_STRATEGY.md 更新
2. **感度分析確認**: `F10_PARAM_SENSITIVITY_*.md` で eps/k_lo/k_hi/vz_thr のロバスト性確認
3. **ブローカー選定**: `F10_BROKER_MATRIX_*.md` でくりっく365/GMOロールでの Worst10Y★ FAIL シナリオ把握
4. **実運用移行**: SBI-CFD 選択（コスト最低シナリオ）

**関連ファイル:**
- `audit_results/F10_BROKER_MATRIX_20260526.md`
- `audit_results/F10_BOOTSTRAP_20260526.md`
- `audit_results/F10_PERMUTATION_20260526.md`
- `audit_results/F10_PARAM_SENSITIVITY_20260526.md`
- `audit_results/f10_bootstrap_results.yaml`
- `audit_results/f10_permutation_results.yaml`
- `G7_WFA_F10_2026-05-26.md`