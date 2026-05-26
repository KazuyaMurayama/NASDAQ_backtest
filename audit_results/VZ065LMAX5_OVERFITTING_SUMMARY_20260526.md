# vz065+lmax5戦略 過学習検出 統合サマリレポート

**戦略名:** S2_VZGated(l_max=5.0) + LT2-N750 + Regime k_lt (vz_thr=0.65, k_lo=0.1, k_hi=0.8, k_mid=0.5)
**OOS 期間:** 2021-05-08 〜 2026-03-26（約4.9年、T=1225日）
**IS 期間:** 1974-01-02 〜 2021-05-07（47年）
**作成日:** 2026-05-26

---

## エグゼクティブ・サマリー

| 検定項目 | 観測値 | vz065lmax5判定 | E4戦略比 |
|---|---|---|---|
| 観測 Sharpe_OOS | **0.950** | — | E4: 0.892 |
| BH 1x Sharpe_OOS | 0.541 | — | — |
| Bootstrap CI95_lo (L=63) | 0.134 | **PASS** | E4: 0.110 |
| Bootstrap p値 (L=63) | 0.0118 | **PASS** | E4: 0.0130 |
| Permutation (a) L_s2_lmax5 block | p=0.147 | **FAIL** | E4(L_s2): p=0.213 FAIL |
| Permutation (c) lev_mod_065 block | p=0.017 | **PASS** | E4(lev_mod_e4): p=0.040 PASS |
| Permutation (d) 同時置換（KEY） | p=0.005 | **PASS** | E4: p=0.013 PASS |
| WFA (G9) CI95_lo | +24.82% | **PASS** | — |
| WFA (G9) WFE | 1.272 | **PASS** | — |
| **vz065lmax5総合判定** | — | **PASS** | E4: PASS |

**一行結論:** vz065+lmax5戦略は Bootstrap CI95_lo=0.134 > 0, Permutation(d) p=0.005, WFA(G9) CI95_lo=+24.82% PASS により**真のアルファ確認**。
E4 vs vz065lmax5: l_max を 7.0 → 5.0 へ下げて MaxDD を改善する設計。

---

## Block Bootstrap 結果（vz065lmax5 vs E4）

**Stationary Block Bootstrap（B=5,000）**

| ブロック長 | vz065lmax5 CI95_lo | vz065lmax5 CI95_hi | vz065lmax5 p値 | 判定 | E4 CI95_lo | E4 p値 |
|---|---|---|---|---|---|---|
| L=20 (1ヶ月) | 0.145 | 1.743 | 0.0112 | PASS | 0.103 | 0.0130 |
| L=63 (3ヶ月) | 0.134 | 1.646 | 0.0118 | **PASS** | 0.110 | 0.0130 |
| L=126 (6ヶ月) | 0.153 | 1.571 | 0.0108 | PASS | 0.136 | 0.0104 |

**IS 10分割安定性（IS全期間をChunk=10で分割）:**

| 指標 | vz065lmax5 |
|---|---|
| Sharpe_IS mean | 1.035 |
| std | 0.284 |
| CoV | 0.274 |
| min | 0.443 |

---

## Permutation 検定結果（vz065lmax5 vs E4）

**B=1,000, block_len=63, seed=42**

| 検定 | 対象 | vz p値 | vz置換mean | vz判定 | E4 p値 | E4判定 |
|---|---|---|---|---|---|---|
| (a) L block | 動的レバレッジ（lev_mod固定） | 0.147 | 0.982 | **FAIL** | 0.213 | FAIL |
| (c) lev_mod block | 市場参加タイミング（L固定） | 0.017 | 0.660 | **PASS** | 0.040 | PASS |
| **(d) 同時置換 (KEY)** | **真のアルファ測定** | **0.005** | **0.494** | **PASS** | 0.013 | PASS |

**解釈:**

- **(d) p=0.005:** L_s2_lmax5 と lev_mod_065 を同時ブロックシャッフルした場合の置換平均 Sharpe
  = 0.494 ≈ BH 1x (0.541) に収束。観測値 0.950 は
  B=1000中5回しか超えられなかった。

- **(a) p=0.147:** L_s2_lmax5 単体のアルファ寄与の有無を測定。
  置換平均 Sharpe = 0.982。

- **(c) p=0.017:** lev_mod_065 単体のアルファ寄与の有無を測定。
  置換平均 Sharpe = 0.660。

---

## Walk-Forward Analysis (G9) 結果

`G9_WFA_VZ065_LMAX5_2026-05-26.md` より:

| 指標 | 値 | 基準 | 判定 |
|---|---|---|---|
| WFA CI95_lo | +24.82% | > 0 | **PASS** |
| WFA WFE | 1.272 | 0.5 ≤ WFE ≤ 2.0 | **PASS** |

---

## 総合判定詳細

### vz065+lmax5 総合判定: **PASS**

**主判定軸（Bootstrap + Permutation d + WFA）:**
- Block Bootstrap (L=63): CI95_lo = 0.134, p = 0.0118 → **PASS**
- Permutation (d) 同時置換 (KEY): p = 0.0050 → **PASS**
- WFA (G9): CI95_lo = +24.82%, WFE = 1.272 → **PASS**

**補助判定:**
- Permutation (a) L_s2_lmax5: p = 0.1470 → **FAIL**
- Permutation (c) lev_mod_065: p = 0.0170 → **PASS**
- IS安定性: 10分割mean=1.035, CoV=0.274

**vz065lmax5 と E4 の比較:**
- vz065lmax5 は vz_thr=0.65（E4=0.70）, l_max=5.0（E4=7.0）
- Expected: OOS Sharpe 0.949 vs E4 0.891, MaxDD -51.82% vs E4 -60.01%（改善）
- Trades/yr ~27（E4と同等）

**残存リスク:**
- 選択バイアス: vz065+l_max グリッド探索による試行数増加
- OOS期間4.9年のみ。WFA は PASS だが追跡継続が必要

---

## Next Action

1. **STRATEGY_REGISTRY.md 登録**: 暫定 Active 候補として登録
2. **戦略比較レポート更新**: STRATEGY_PERFORMANCE_COMPARISON へ追加
3. **実運用移行**: SBI-CFD 選択でブローカーマトリクス PASS 確認後
4. **四半期レビュー**: 直近4Q Sharpe < 0.3 で戦略停止 or レビュー

**関連ファイル:**
- `audit_results/VZ065LMAX5_BOOTSTRAP_20260526.md`
- `audit_results/VZ065LMAX5_PERMUTATION_20260526.md`
- `audit_results/VZ065LMAX5_BROKER_MATRIX_20260526.md`
- `audit_results/VZ065LMAX5_PARAM_SENSITIVITY_20260526.md`
- `audit_results/vz065lmax5_bootstrap_results.yaml`
- `audit_results/vz065lmax5_permutation_results.yaml`
- `G9_WFA_VZ065_LMAX5_2026-05-26.md`