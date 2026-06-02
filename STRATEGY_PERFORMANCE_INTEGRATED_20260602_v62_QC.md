# v6.2 並列エージェント QC レポート (2026-06-02)

> v6.2 (STRATEGY_PERFORMANCE_INTEGRATED_20260602_v62.md) に対して **3 並列 QC エージェント** を起動し、独立検証を実施。本ファイルは検証結果と推奨修正をまとめる。

## 🎯 QC スコープ

| Agent | 役割 | 結果 |
|---|---|---|
| Agent 1 (Explore) | raw 値 vs CSV 整合性 (12 チェック) | **11/12 PASS、1 MARGINAL (誤検知)** |
| Agent 2 (general-purpose) | 計算ロジック (5 領域) | **5/5 PASS** |
| Agent 3 (general-purpose) | NEW SOTA 妥当性 + 内部整合性 | **NEW SOTA: 要追加検証 (即時昇格不可)** |

---

## ✅ §1 Agent 1: raw 値検証 (11/12 PASS)

| # | チェック | 状態 | 値 |
|---|---|---|---|
| 1a | g19a F10 ε=0.015 moderate CAGR_OOS_net | ✅ PASS | 0.194410 |
| 2a | g19b NEW SOTA CAGR_OOS | ✅ PASS | 0.214943 |
| 2b | g19b NEW SOTA IS_OOS_gap | ✅ PASS | -0.012665 (負) |
| 3a | g19c DH WFA moderate CI95_lo | ✅ PASS | 0.175390 |
| 3b | g19c DH WFA moderate WFE | ✅ PASS | 0.662122 |
| 4a | g19c signal audit raw_a2 mean diff | ⚠ MARGINAL→**訂正**: 実値 **0.000111** で v6.2 記載と一致 | (Agent 誤読、別行と混同) |
| 5a | g19d F10-E4 additional trades | ✅ PASS | 7.884/yr |
| 6a | g19e F8 R5 moderate CAGR_OOS | ✅ PASS | 0.194293 |
| 6b | g19e F7v3+E4 moderate CAGR_OOS | ✅ PASS | 0.189254 |
| 6c | g19e NDX B&H CAGR_OOS | ✅ PASS | 0.082685 |
| 6d | g19e NDX B&H MaxDD | ✅ PASS | -0.779324 |
| 6 | g19e row count (9 vs 想定 10) | ⚠→**訂正**: 実 9 行 (2戦略×4 spread + 1 BH = 9) で正しい | (Agent 想定誤) |

→ **Agent 1 の MARGINAL/MISMATCH は両方 Agent 側の確認誤りで、レポート値は正確**。

---

## ✅ §2 Agent 2: 計算ロジック (5/5 PASS)

| 項目 | 確認 |
|---|---|
| Tax model (§3-A `(pre - 0.66%) × 0.8273`) | g18 line 86-93 で正確 |
| Daily cost `\|Δposition\| × spread_ow` | g18 line 96-122 で正確 |
| F10 deadband ε ロジック | g14 line 132-149 で正確 |
| NEW SOTA construction (vz=0.65/lmax=7/eps=0.015) | g19b で正確に組み立てられている |
| WFA 50窓設計 | g14 line 164-195 で 1977-2026 年次非重複、`short_flag` 80%閾値 |

---

## ⚠️ §3 Agent 3: NEW SOTA 妥当性 — **要追加検証 (HIGH SUSPICION)**

### §3-1 重大な指摘 — 負 gap は generalization ではなく OOS regime fit の疑い

| 観察 | データ | 解釈 |
|---|---|---|
| **CAGR_IS は変わらない** | vz=0.65/lmax=7: 0.2023<br>vz=0.70/lmax=7 (F10): 0.2019 | +0.04pp の僅差 → vz=0.65 は IS 期で **structural advantage を持たない** |
| **CAGR_OOS のみ +2.05pp 改善** | vz=0.65: 0.2149<br>vz=0.70: 0.1944 | OOS 4.9 年のみで発生 |
| **vz=0.65 で gap 負方向はlmax=7のみ** | lmax=5.0: gap=+1.23pp<br>lmax=5.5: gap=+0.73pp<br>**lmax=7.0: gap=-1.27pp** | 非単調パターン → 構造的因果なし、**lmax=7 でのみ偶発的に発火** |
| **Worst10Y★ -0.44pp 悪化** | NEW SOTA: +9.98%<br>F10 ε=0.015: +10.42% | 「vz=0.65 が protect better」物語と矛盾 |
| **MaxDD ほぼ同等** | NEW SOTA: -65.95%<br>F10: -66.03% | DD 抑制効果も微小 |

### §3-2 Agent 3 の結論

> **NEW SOTA は「OOS regime spurious fit」の疑い濃厚**。負 gap = OOS > IS は「健全な generalization」ではなく、2022 NDX -34% drawdown の単一イベントで vz=0.65 の早期 k_lo=0.1 が偶然救済した結果の可能性が支配的。**vz_thr robustness sweep / WFA / 年次寄与分解 / Bootstrap を全て済ませてから昇格判定すべき**。

### §3-3 DH Dyn [A] 解釈について

- ✅ signal IS/OOS 分布同一は signal-overfit 否定の **必要条件** として成立
- ⚠ ただし「macro spurious のみ」結論は v6.2 で**未証明**
- 追加で必須: SOFR rolling 5y mean IS vs OOS 数値提示、TMF/LBUL.L cost drag 寄与分解、IS 期に OOS SOFR レベルを当てた counterfactual bootstrap

---

## 🛠 §4 v6.2 必須修正 (本 QC 反映)

### §4-1 NEW SOTA を「即時 v7 昇格対象」→ **「要追加検証 (v6.3 で WFA + sweep)」** に格下げ

| 項目 | v6.2 初版 | **v6.2 QC 反映版** |
|---|---|---|
| §0 NEW SOTA ラベル | 🔴 NEW SOTA | 🔍 NEW CANDIDATE (要 v6.3 検証) |
| §7-2 採用判断 | 即時 v7 候補昇格対象 | **WFA / robustness sweep / 年次寄与分解 完了後判定** |
| §3-3 解釈 | "negative gap = excellent generalization" | "negative gap は **OOS 偶発 lmax=7 spike**、要 v6.3 で追加検証" |

### §4-2 v6.3 で実施必須の検証

1. **vz_thr robustness sweep**: 0.625 / 0.65 / 0.675 / 0.70 / 0.725 を lmax=7+ε=0.015 で全比較 (5 configs)
2. **NEW SOTA WFA 50窓**: g14 同等で CI95_lo / WFE / t-stat 計算
3. **年次寄与分解**: NEW SOTA vs F10 の OOS +2.05pp が 2022 (1 年) 由来か全年均等か
4. **Bootstrap on OOS**: +2.05pp 改善の 95%CI が 0 を跨ぐか
5. **Permutation test**: vz_thr のラベルシャッフルで gap=-1.27pp が偶然か検定

### §4-3 DH Dyn [A] 追加分析 (v7 課題に追加)

1. **SOFR 数値根拠**: §4-3 で「OOS 3.59% 想定」と書くだけでは不十分、IS と OOS の rolling 5y mean を実数提示
2. **Cost attribution**: TMF / LBUL.L borrow cost 寄与度分解
3. **Counterfactual bootstrap**: IS 期に OOS の SOFR レベルを当てて IS CAGR を再計算

---

## 📊 §5 QC 反映後の戦略ランキング (修正版)

| Rank | Strategy | CAGR_OOS_net | 採用判断 |
|:--:|---|---:|---|
| 🔍 1? | NEW CANDIDATE (vz=0.65+lmax=7+F10ε) | +21.49% (要検証) | **WFA + sweep + bootstrap 完了まで保留** |
| 2 | F10 ε=0.015 ★ (現 v6.1 推奨) | +19.44% | **暫定首位を維持** |
| 3 | F8 R5_CALM_BOOST | +19.43% | F10 と同等、独自性弱 |
| 4 | F7v3+E4 A:tilt=2.0 | +18.93% | tilt 構造差別化候補 |
| 5 | D5 vz=0.65/lmax=5.5 | +17.86% | MaxDD 最浅 / Sharpe 最高クラス |
| 6 | E4 Regime k_lt ◆ (現 Active) | +17.75% | **Active 継続維持** |
| 7 | DH Dyn 2x3x [A] | +9.56% | 要 v7 macro 分析 |
| – | NDX 1x B&H | +8.27% | ベンチマーク |

---

## 🎯 §6 QC 総合判定

| カテゴリ | 状態 |
|---|---|
| raw 値 / 計算ロジック | **全て正確 (Agent 1 + Agent 2 = 17/17 検証)** |
| NEW SOTA 数値計算 | ✅ 妥当 |
| **NEW SOTA generalization 解釈** | **❌ 過解釈、OOS spurious fit の疑い濃厚** |
| F10 ε robustness | ✅ 良好 (ε=0.010-0.025 で頑健) |
| DH [A] signal overfit 否定 | ✅ 部分支持 (分布同一) |
| DH [A] macro spurious 結論 | ⚠ 未証明 |
| **NEW SOTA 即時昇格** | **❌ 不可、v6.3 検証必須** |

### 推奨アクション

1. **本 QC レポート (v62_QC.md) を v6.2 と同時に公開** ✅
2. **v6.2 メイン文書を本 QC 反映版に更新** (NEW SOTA → NEW CANDIDATE)
3. **v6.3 で 5 検証 (§4-2) を実施**
4. **その上で NEW CANDIDATE の昇格 / 棄却判定**

---

*管理: Claude (Opus 4.7) / 並列 QC エージェント (Explore + general-purpose × 2) / 2026-06-02*
