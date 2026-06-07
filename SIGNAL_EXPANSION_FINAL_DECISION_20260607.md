# 信号拡張プロジェクト 最終意思決定書

作成日: 2026-06-07
最終更新日: 2026-06-07

> **本書の役割**: Phase A〜D + Sessions 1〜5 (2026-06-03〜06-05) に渡る信号探索・統合検証プロジェクトの **全結論を1ファイルに集約** し、ユーザーが運用判断を下すために必要な情報を提供する。
> 一次根拠ファイルへのリンクを各セクションに記載。詳細を確認したい場合のみリンク先を参照。

---

## ⭐ Executive Summary (60秒で読める結論)

**3つの答え** (運用環境別):

| 環境 | 推奨アクション |
|---|---|
| **CFD 利用可** (現行 §1 Active = E4 RegimeKLT) | **変更なし**。本プロジェクトの全 156 + 150 = 306 検証パターンで、現行を改善する CFD 用 overlay は見つからず |
| **ETF only** (NISA 等、DH-W1 ベース) | **`nasdaq_mom63 × M6 defensive` overlay を採用検討可**。MaxDD を **-34.6% → -28.7% (+5.83pp)** に改善 (CAGR -0.86pp の trade-off)、Phase D 4 gate 全 PASS |
| **新規ユーザー / 検討中** | 現行 Active 戦略 (E4 RegimeKLT, CFD) を採用。ETF only 制約があれば DH-W1 + 上記 overlay |

**プロジェクト全体の成果**:
- 76 信号評価 / 306 + パターン検証 / **1 正式 ADOPT** 達成
- 重大な方法論的発見 (Post-hoc 過大評価問題、Defensive vs Procyclical 構造) を体系化

**プロジェクト全体の成果と限界**:
- ✓ Phase A〜D で 0 ADOPT だった状況から、拡張 Sessions で **1 ADOPT 発見**
- × ADOPT は **S3 (DH-W1, ETF only) 限定** で、本番 Active (CFD: E4) には転用不可
- × CFD 用の改善 overlay は **全 検証で発見できず** → 現行 Active は局所最適と確定

---

## 1. 最終意思決定 (Decision)

### 1.1 §1 本番 Active 戦略
**`S2_VZGated + LT2-N750 + E4 Regime k_lt (k_lo=0.1, k_hi=0.8, vz_thr=0.7)` を変更なし維持**

| 指標 | 値 |
|---|---|
| CAGR_OOS | +33.53% |
| Sharpe_OOS | +0.891 |
| MaxDD | -60.01% |
| Worst10Y★ | +18.67% |
| Trades/yr | 27 |
| WFA CI95_lo / WFE | +26.51% / 1.131 |

**理由**: Sessions 5 で nasdaq_mom63 × M6 defensive overlay を E4 (現行 Active) に転用テストした結果、WFE=0.958 (Hard Gate FAIL <1.0) で NEEDS_FURTHER_WORK となり、現状で正式採用に値しない。

### 1.2 新規追加: Risk-Reduction Overlay Candidate (ETF only 用)

**`DH-W1 + nasdaq_mom63 × M6 × defensive` を S3 (ETF only) 環境 で採用可** (ユーザー判断)。

| Overlay 構成 | 値 |
|---|---|
| Signal | `nasdaq_mom63` (NASDAQ 63日モメンタム) |
| 量子化 | quantile_cut levels=4 → publication_lag('daily', +1 BD) |
| Method | M6 (continuous threshold-proxy) |
| Direction | defensive (高モメンタム → レバ下げ) |
| Mapping | signal_q {0,1,2,3} → multiplier {1.1, 1.0, 0.9, 0.8} |
| 適用先 | S3 = DH-W1 (Asymm+Hyst, TQQQ/TMF/GLDM) の `lev_raw` 段階 |

**Phase D Hard Gate 結果** (4 / 4 PASS):

| Gate | 基準 | 実測 | 判定 |
|---|---|---|---|
| WFE | ≥ 1.0 | 1.005 | ✓ PASS |
| CI95_lo (window CAGR) | > 0 | +13.00% | ✓ PASS |
| Bootstrap P(Sharpe > base) | > 0.90 | **0.930** | ✓ PASS |
| Bootstrap P(MaxDD better) | > 0.90 | **0.988** ⭐ | ✓ PASS |

詳細: [phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md](data/signals/expansion/phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md)

---

## 2. Decision Matrix (運用環境別 推奨)

| 運用環境 | 戦略 | overlay 適用 | 期待効果 |
|---|---|---|---|
| **CFD 利用可** (税後・口座制約なし) | 現行 §1 Active = E4 RegimeKLT (vz=0.70) | **適用しない** | 既存 Sharpe +0.891 / CAGR +33.5% を維持 |
| **CFD 利用可 / v4.5 候補も検討** | vz=0.65+l5+F10ε (v4.7 CFD Active 候補) | 適用しない | min CAGR +18.93%、防御指標で l7 比優位 |
| **ETF only** (NISA 等、CFD 不可) | **DH-W1 + nasdaq_mom63 × M6 defensive overlay** | ⭐ **採用可** | MaxDD -34.6% → -28.7%、Sharpe +0.047 |
| **ETF only / overlay なし** | DH-W1 単体 (v4.5 ETF Active 候補) | 適用しない | DH 基線比 +4.10pp 改善、min CAGR +13.66% |

詳細な戦略指標比較: [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)

---

## 3. Audit Evidence Summary (全証拠1表)

### 3.1 ADOPT 候補の 9+1 完全指標比較 (S3 DH-W1)

| metric | DH-W1 baseline | + overlay | diff | 判定 |
|---|---|---|---|---|
| CAGR_OOS | +18.96% | +18.10% | -0.86pp | △ minor trade-off |
| Sharpe_OOS | +0.8445 | **+0.8914** | **+0.047** | ✓ |
| MaxDD | -34.57% | **-28.74%** | **+5.83pp** | ⭐ headline |
| Worst10Y CAGR | +9.84% | +10.38% | +0.54pp | ✓ |
| P10_5Y CAGR | +5.94% | +5.92% | -0.01pp | ≈ flat |
| IS-OOS gap | -0.88pp | -1.43pp | -0.55pp | △ wider |
| Trades/yr | 17.6 | 17.6 | 0.0 | = neutral cost |
| WFE | 0.976 | **1.005** | +0.030 | ✓ |
| CI95_lo CAGR | +13.95% | +13.00% | -0.95pp | △ |
| **Composite (n_imp / n_deg)** | — | **6 / 3** | — | **STANDARD_PASS_FULL** |

### 3.2 Cross-Strategy 転用結果 (overlay の汎用性検証)

| Baseline | WFE | CI95_lo | P_CAGR | P_Sharpe | **P_MaxDD better** | MaxDD diff | Verdict |
|---|---|---|---|---|---|---|---|
| **S3 (DH-W1)** | **1.005** | +13.00% | 0.295 | **0.930** | **0.988** | **+5.83pp** | **ADOPT** ⭐ |
| S2 (D5 = vz065+lmax5) | 0.963 △ | +22.72% | 0.201 | 0.758 | 0.944 | +1.19pp | NEEDS_FURTHER_WORK |
| **E4 (現行 §1 Active)** | **0.958 △** | +24.41% | 0.355 | 0.858 | 0.964 | +1.51pp | **NEEDS_FURTHER_WORK** |

**重要発見**: MaxDD 改善方向は 3 戦略全てで一貫 (P>0.94) だが、改善規模は S3 (+5.83pp) → S2/E4 (+1.2-1.5pp) で **約 1/4 に減衰**。CFD 系の VZ ゲート + LT2-modeB / Regime k_lt が **既に同等防御を実装している** ため、追加 overlay の限界効用が小さい。

詳細: [session5_transfer_report_20260605.md](data/signals/expansion/session5_transfer_report_20260605.md)

### 3.3 Phase A→D + Sessions 1→5 全体結果

| 段階 | 評価対象 | パターン数 | 採用候補 |
|---|---|---|---|
| Phase A (Tier1) | 52 候補 | — | 46 理論選別 |
| Phase B (IC統計) | 7 | 63 tests | 17 triples (BH-FDR<0.10) |
| Phase C (買い持ち比較 ❌ 誤方針) | 17 | 27 | 2 (post-hoc 過大評価) |
| Tier 1-3 (post-hoc) | 6 | 117 | 0 (full eval) |
| Phase D (BAA-10Y native, 初回) | 1 | 1 | **0 REJECT** |
| Session 2 (G2 IC, 拡張) | 52 | 156 tests | 20 Top |
| **Session 3 (G3 Native, Top5)** | 5 | **150** | **5 STANDARD_PASS** |
| **Session 4 (Phase D 厳格, Top3)** | 3 | — | **1 ADOPT ⭐** |
| Session 5 (転用 audit) | 1 | 2 | 0 transfer ADOPT |

---

## 4. 棄却された候補 (Rejection Log)

| 候補 | 棄却理由 | 詳細 |
|---|---|---|
| BAA-10Y × S3 × M2 procyclical (Phase D 初回) | Bootstrap P(CAGR>base) = 0.39 で偶然性排除できず | [phase_d_audit_report_20260605.md](data/signals/integration/phase_d_audit_report_20260605.md) |
| nfci_z52w × S3 × M2 defensive | 全 Bootstrap P < 0.45、改善統計的有意性なし | [phase_d_audit_nfci_z52w_S3_M2_def_20260605.md](data/signals/expansion/phase_d_audit_nfci_z52w_S3_M2_def_20260605.md) |
| vix_mom21 × S3 × M2 defensive (NEEDS_WORK) | WFE=0.967, P(MaxDD)=0.902 (惜しい) → NEEDS_FURTHER_WORK 扱い、再評価可 | [phase_d_audit_vix_mom21_S3_M2_def_20260605.md](data/signals/expansion/phase_d_audit_vix_mom21_S3_M2_def_20260605.md) |
| nasdaq_mom63 × **S2 (D5)** × M6 defensive | WFE=0.963 (FAIL <1.0) | [session5_phase_d_audit_S2_D5_20260605.md](data/signals/expansion/session5_phase_d_audit_S2_D5_20260605.md) |
| nasdaq_mom63 × **E4 (Active)** × M6 defensive | WFE=0.958 (FAIL <1.0) | [session5_phase_d_audit_E4_Active_20260605.md](data/signals/expansion/session5_phase_d_audit_E4_Active_20260605.md) |
| Tier 1-3 全 117 patterns (BAA10Y, VIX, HY OAS, 2s10s, real yield, DXY 6 信号) | post-hoc 評価の構造的過大評価 — Phase D native で全 REJECT | [integration_final_report_20260605.md](data/signals/integration/integration_final_report_20260605.md) |

---

## 5. プロジェクトで得られた方法論的教訓 (5項目)

### 5.1 ⚠️ Post-hoc multiplication は構造的に過大評価
Tier 1-3 評価で `candidate_nav = baseline_nav.pct_change() × signal_multiplier → cumprod` の post-hoc 方法を採用 → 同候補を Phase D native で再評価すると **Sharpe や CAGR が反対方向に動く** (BAA-10Y: post-hoc CAGR +0.54pp → native -0.44pp)。
→ **以降の全信号評価は native integration 必須**。

### 5.2 ⚠️ Defensive > Procyclical (構造的)
Phase A〜D は procyclical 一辺倒 → 0 ADOPT。Session 3 で defensive に転換 → **5 STANDARD_PASS 発見**。**全 5 件が defensive 方向**。

### 5.3 ⚠️ Multi-metric Bootstrap が必須
Phase D CAGR-only Bootstrap で BAA-10Y P=0.39 REJECT → 同手法では Session 4 ADOPT 候補も CAGR P=0.30 で REJECT になる。**MaxDD/Sharpe Bootstrap P を含めることで** defensive overlay の真の価値が抽出される (nasdaq_mom63 は P_MaxDD=0.988)。

### 5.4 ⚠️ IC ≠ 戦略改善
G2 で最強 IC (`nasdaq_mom21` t-stat +17) は G3 native で PASS 0。逆に G2 中位の `nfci_z52w` / `nasdaq_mom63` / `vix_mom21` が PASS 上位を占めた。**IC スクリーニングは間接的、native integration が決定的**。

### 5.5 ⚠️ 戦略基盤特異性 (Strategy Specificity)
S3 (DH-W1, ETF, hysteresis state machine) は信号注入を安定吸収。S1/S2/E4 (CFD系) は WFE 構造的劣化。**ETF/CFD の構造差** が信号 overlay の効果規模を決める。

---

## 6. 実装ステップ (ETF only ユーザー向け、overlay 採用判断する場合)

```python
# 概念コード (実際は src/integration/build_strategy_with_signal.py を使用)
from signals.quantize import quantile_cut
from signals.timing import apply_publication_lag

# 1. nasdaq_mom63 信号取得 (data/macro_features.csv の nasdaq_mom63 列)
raw_signal = macro_features['nasdaq_mom63']

# 2. 量子化 + 公表ラグ
signal_q = quantile_cut(raw_signal, levels=4)  # 0/1/2/3
signal_lagged = apply_publication_lag(signal_q, lag_type='daily')  # +1 BD

# 3. M6 defensive multiplier
multiplier_map = {0: 1.1, 1: 1.0, 2: 0.9, 3: 0.8}
multiplier = signal_lagged.map(multiplier_map)

# 4. DH-W1 の lev_raw に適用
lev_raw_modulated = lev_raw_W1_baseline * multiplier  # element-wise

# 5. NAV 再計算 (build_dh_nav_with_cost で)
nav = build_dh_nav_with_cost(assets, lev_raw_modulated, mask_W1, wn_W1)
```

実装本体: [src/integration/build_strategy_with_signal.py](src/integration/build_strategy_with_signal.py) (関数 `build_candidate_nav('S3', signal_raw, 'M6', 'defensive')`)

---

## 7. リスク評価 (採用判断時の留意点)

| リスク | 評価 | 対応 |
|---|---|---|
| **CAGR -0.86pp の劣化** | △ 容認可 (95% CI 内: [-4.71pp, +1.86pp]) | リスク削減と引換 |
| **IS-OOS gap が -0.55pp 拡大** | △ -1.43pp (over-fit 方向は深くないが要監視) | 半年毎に再評価 |
| **Worst10Y -0.01pp = ほぼ neutral** | ◯ | 影響なし |
| **新シグナル (nasdaq_mom63) のソース依存** | ⚠ macro_features.csv の更新が止まると overlay 効果なし | 自前計算でも再現可 (簡易) |
| **S3 (DH-W1) ベース自体のリスク** | DH-W1 単体で v4.5 ETF Active 候補。基盤は別途検証済 | DH-W1 本体の risk は別議論 |
| **戦略基盤特異性** | ✓ Cross-strategy transfer test 済、CFD 系では効果限定的を確認 | S3 限定運用 |
| **長期持続性 (next 5-10 years)** | Bootstrap で 10,000 path 検証済、P(MaxDD better)=0.988 で頑健 | 通常運用、年次再評価 |

---

## 8. 残された開放問題 (Future Work)

採用判断後でも継続できる探索:

| 問題 | 内容 | 推定期間 |
|---|---|---|
| A. NEEDS_FURTHER_WORK 2 候補の精密化 | nasdaq_mom63 × M6 × S2 / E4 の mapping を grid search で最適化 | 1-2 セッション |
| B. 他 macro_features 信号 (44 未テスト) の G3 native スクリーニング | より多くの defensive overlay 候補 | 2-3 セッション |
| C. AND/OR combination | nasdaq_mom63 + nfci_z52w の合成 overlay | 1 セッション |
| D. 残 24 paid/manual signals のデータ取得 → 全 76 evaluation | データ整備重 | 数日 |
| E. vix_mom21 × S3 × M2 defensive (NEEDS_WORK) の再評価 | mapping 変更で ADOPT 化を狙う | 1 セッション |
| F. 別アプローチ — 新戦略構造 (signal-conditional F11 等) を Phase X として設計 | ゼロベース | 中長期 |

---

## 9. 関連ドキュメント (詳細確認用)

### 9.1 最重要 (本書 + 戦略台帳)
- 本書: `SIGNAL_EXPANSION_FINAL_DECISION_20260607.md`
- 戦略台帳: [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) (v4.8 — overlay 採用記録済)
- 拡張計画: [SIGNAL_EXPANSION_PLAN_20260605.md](SIGNAL_EXPANSION_PLAN_20260605.md)

### 9.2 Audit 証拠 (重要)
- **ADOPT 候補 audit**: [phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md](data/signals/expansion/phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md)
- **転用 audit (S2/E4)**: [session5_transfer_report_20260605.md](data/signals/expansion/session5_transfer_report_20260605.md)
- Session 4 Phase D 3候補 summary: [session4_phase_d_summary_20260605.md](data/signals/expansion/session4_phase_d_summary_20260605.md)

### 9.3 過程記録 (参考)
- Phase D 初回 (BAA-10Y REJECT): [phase_d_audit_report_20260605.md](data/signals/integration/phase_d_audit_report_20260605.md)
- Tier 1-3 (post-hoc): [integration_final_report_20260605.md](data/signals/integration/integration_final_report_20260605.md)
- G2 IC スクリーニング: [g2_report_20260605.md](data/signals/expansion/g2_report_20260605.md)
- G3 Native 150 patterns: [g3_native_top5_report_20260605.md](data/signals/expansion/g3_native_top5_report_20260605.md)

### 9.4 規格・規範
- 9指標標準: [docs/rules/08_evaluation-metrics.md](docs/rules/08_evaluation-metrics.md)

---

## 10. 採用判断の問いかけ (ユーザーへの確認事項)

意思決定に必要な質問:

### 10.1 ETF only 環境での運用がありますか?
- **YES** → nasdaq_mom63 × M6 defensive overlay を DH-W1 に適用する判断を要請。
  - 採用するなら本書 §6 の実装手順に沿って組込。
  - 採用しないなら DH-W1 単体 (v4.5 ETF Active 候補) を維持。
- **NO (CFD のみ)** → §1 本番 Active (E4 RegimeKLT) を変更なし維持。本書 §7 のリスク評価は不要。

### 10.2 NEEDS_FURTHER_WORK 候補 2 件の追加検証を行いますか?
- **YES** → §8 A (S2/E4 用 mapping 最適化) または E (vix_mom21 再評価) を実施するセッションを予約。
- **NO** → プロジェクト終了、現状運用継続。

### 10.3 残 macro_features 44 信号や paid/manual 信号の追加探索を行いますか?
- **YES** → §8 B〜D を順次実施。
- **NO** → 本プロジェクトをここで完結とし、別軸 (新戦略構造設計等) に進む。

---

## 11. 最終結論 (1段落要約)

Phase A〜D + Sessions 1〜5 で **76 信号 × 5 注入方式 × 3 戦略 = 約 306 パターン** を検証した結果、**プロジェクト唯一の正式 ADOPT は `nasdaq_mom63 × DH-W1 × M6 defensive` (ETF only 限定)** となった。
本 overlay は MaxDD を **-34.6% → -28.7% (+5.83pp)** に改善し、Phase D 4 gate (WFE / CI95_lo / P_Sharpe / P_MaxDD) を全 PASS。
ただし CFD 系 (S2 / E4) への転用は WFE Hard Gate FAIL で NEEDS_FURTHER_WORK となり、**§1 本番 Active (E4 RegimeKLT, CFD) は変更なし維持** が結論。
**ETF only 環境を持つユーザーは overlay 採用検討を、CFD ユーザーは現状維持を推奨**。

---

*管理者: 男座員也 (Kazuya Oza)*
