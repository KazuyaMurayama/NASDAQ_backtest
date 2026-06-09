# セッション学習保存: 信号拡張プロジェクト (Phase A→D + Sessions 1-5)

作成日: 2026-06-07
最終更新日: 2026-06-07 (v1.1: V7 pure_boost を Shortlisted 追加 + 税後 CAGR 規約反映) / **2026-06-07 末 (v1.2: §6 環境別ベスト戦略表に v2 aftertax 列追加 / §3 V7 セクション v2 反映。v1 は 2026 YTD partial year を完全1年扱いで OOS aftertax を 1.9〜3.8pp 過小評価していた)**
管理者: 男座員也 (Kazuya Oza)

> **本書の役割**: 2026-06-03〜06-07 に渡る信号拡張プロジェクト (Phase A→D + Sessions 1-5) で得られた学習を、新セッション開始時の引き継ぎ用に集約する。詳細な意思決定根拠は [`SIGNAL_EXPANSION_FINAL_DECISION_20260607.md`](SIGNAL_EXPANSION_FINAL_DECISION_20260607.md) (v2) を参照。

---

## 1. Quick Reference (新セッション開始時に最初に読む)

| 項目 | 内容 |
|---|---|
| **検証済みプロジェクト** | 信号拡張 (Phase A→D + Sessions 1-5, 2026-06-03 〜 06-07) |
| **規模** | 76 信号 × 5 注入方式 × 3 戦略基盤 ≒ 306+ patterns 検証 |
| **最終結論** | §1 Active (E4 RegimeKLT) **変更なし維持**、ETF only 環境で `nasdaq_mom63 × DH-W1 × M6 defensive` overlay 採用可 |
| **一次根拠** | [`SIGNAL_EXPANSION_FINAL_DECISION_20260607.md`](SIGNAL_EXPANSION_FINAL_DECISION_20260607.md) (v2, canonical 2021-05-08 split, 9+1 標準指標) |
| **戦略台帳** | [`STRATEGY_REGISTRY.md`](STRATEGY_REGISTRY.md) (本セッションで DH-W1 系 / overlay を §2 Shortlisted 登録) |
| **採否** | ADOPT 1 / REJECT 5+ / NEEDS_FURTHER_WORK 3 / **Shortlisted 追加 1 (V7 pure_boost, 2026-06-07 後半)** |

---

## 2. 方法論的教訓 (5 項目, 次セッションへのアドバイス)

> 詳細は FINAL_DECISION §5 を参照。本節は「次にやるべき / やるべきでない」形式で再整理。

### A. ❌ Post-hoc multiplication 評価は使わない

`candidate_nav = baseline.pct_change() × multiplier → cumprod` 方式は **構造的に過大評価** する。
Tier 1-3 で 117 patterns を評価した結果、Phase D native で再評価すると Sharpe/CAGR が反対方向に動いた (例: BAA-10Y post-hoc +0.54pp → native -0.44pp)。

→ **新シグナル評価では必ず [`src/integration/build_strategy_with_signal.py`](src/integration/build_strategy_with_signal.py) の native integration を使う**。post-hoc は screening 用途すら避ける。

### B. ✓ Defensive 方向を最優先で試す

Phase A〜D は procyclical 一辺倒で 0 ADOPT だった。Session 3 で defensive (高シグナル → レバ下げ) に転換した瞬間に **5 STANDARD_PASS** が見つかった。Phase D 全 5 PASS 候補は全 defensive 方向。

→ **新セッションでは defensive を先に試す**。procyclical は trim 効果が限定的という構造的傾向あり。

### C. ✓ Multi-metric Bootstrap を使う (CAGR-only は禁)

CAGR-only Bootstrap だと defensive overlay の本質的価値 (MaxDD 改善) を捉えられない。
nasdaq_mom63 × S3 M6 def は **P_CAGR=0.30 (FAIL)** だが **P_MaxDD=0.988 / P_Sharpe=0.93 (PASS)** であり、後者の 2 軸が改善を有意化した。

→ **MaxDD / Sharpe / CAGR の 3 metrics で Bootstrap P を取る**。defensive overlay 評価では特に MaxDD を重視する。

### D. ⚠ IC は決定的ではない (native integration が本評価)

G2 で IC trend だった `nasdaq_mom21` (t=+17, 全シグナル中 1位) は G3 native で PASS 0。
中位 IC の `nasdaq_mom63` / `nfci_z52w` / `vix_mom21` が PASS 上位を占めた。

→ **IC を必須スクリーニングと位置付けず、native integration を本評価とする**。IC は粗いふるい程度。

### E. ⚠ 戦略基盤特異性 (ETF vs CFD)

DH-W1 (ETF, hysteresis state machine) は信号注入を**安定吸収**するが、CFD 系 (F10/D5/E4) は WFE が **構造的に劣化** する。Cross-strategy transfer (Session 5) で確認済。

→ **Cross-strategy 転用前に基盤の構造制約を確認**。S3 で ADOPT したら他基盤に流用できると期待しない。基盤ごとに独立評価する。

---

## 3. 検証済みデータ (再実行不要 — 同じ実験を避ける)

> 以下は "やった" ものリスト。同じ実験を新セッションで繰り返さない。

| 段階 | 評価対象 | 結果ファイル |
|---|---|---|
| Phase A (Tier1 理論選別) | 52 候補 → 46 残存 | (履歴) |
| Phase B (IC スクリーニング) | 7 信号 (#6, #21, #23, #26, #28, #41 + nasdaq_mom21派生) | 17 triples (BH-FDR<0.10) |
| Phase C (買い持ち比較 ❌ 誤方針) | 17 候補 / 27 評価 | 2 (post-hoc 過大評価で REJECT) |
| Tier 1-3 (post-hoc) | 6 信号 / 117 patterns | 0 (full eval) → **全 post-hoc で構造的過大評価** |
| Phase D (BAA-10Y native, 初回) | 1 | 0 REJECT (Bootstrap P=0.39) |
| G2 IC スクリーニング (拡張) | 52 信号 (9 REPO 派生 + 43 macro_features) / 156 tests | [`data/signals/expansion/g2_ic_screening_20260605.csv`](data/signals/expansion/g2_ic_screening_20260605.csv) |
| G3 Native (Top5 評価) | 5 × 3 戦略 × 5 方式 × 2 方向 = **150 patterns** | [`data/signals/expansion/g3_native_top5_results_20260605.csv`](data/signals/expansion/g3_native_top5_results_20260605.csv) |
| Phase D 厳格 audit (Session 4) | 3 候補 (nasdaq_mom63 / nfci_z52w / vix_mom21, S3 base) | [`data/signals/expansion/phase_d_audit_*_20260605.md`](data/signals/expansion/) |
| Cross-strategy transfer (Session 5) | 1 候補 (nasdaq_mom63 × M6 def) × 2 戦略 (S2_D5 / E4_Active) | [`data/signals/expansion/session5_*`](data/signals/expansion/) |

---

## 4. 採用 / 棄却 / 保留候補一覧

### ADOPT (1)
- **`nasdaq_mom63 × S3 (DH-W1) × M6 defensive` (V0 mapping)** — Phase D 4 gate 全 PASS
  - Mapping: signal_q {0,1,2,3} → multiplier {1.1, 1.0, 0.9, 0.8} ("V0 defensive")
  - MaxDD -34.57% → **-28.74% (+5.83pp)** ⭐
  - Sharpe_OOS +0.8448 → **+0.8918 (+0.047)**
  - Bootstrap P(MaxDD better) = **0.988** / P(Sharpe better) = 0.930
  - Audit: [`phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md`](data/signals/expansion/phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md)
  - **制約**: S3 (ETF only) 限定。CFD 系には転用不可。

### Shortlisted 追加 (1, 2026-06-07 後半)
- **`nasdaq_mom63 × S3 (DH-W1) × M6 defensive` (V7 pure_boost mapping)** — S3 overlay tuning sweep より
  - Mapping: signal_q {0,1,2,3} → multiplier {1.20, 1.10, 1.00, 1.00} ("V7 pure_boost")
  - **min(CAGR_IS, CAGR_OOS) > 18% target を達成する唯一の "MaxDD 据置" mapping**
  - CAGR_IS +18.61% / CAGR_OOS +19.18% (canonical daily split) / IS-OOS gap **-0.57pp (overlay候補中最良)**
  - MaxDD = baseline と同値 -34.57% (V0 のように改善せず、CAGR boost に振った)
  - Sharpe_OOS 0.841 (baseline 0.845 とほぼ同等)
  - WFA_CI95_lo +14.06% (annual CAGR) / WFE(50w Sharpe) 1.029
  - **Phase D Bootstrap audit 未実施** (P_MaxDD/P_Sharpe/P_CAGR Pending) → 正式 ADOPT 化保留
  - **位置付け**: V0 (MaxDD改善) と相補的な "CAGR死守" 候補。ユーザーリスク選好で V0/V7 を選択。
  - Source: [`S3_OVERLAY_TUNING_REPORT_20260607.md`](S3_OVERLAY_TUNING_REPORT_20260607.md) §6.2 / [`s3_overlay_tuning_20260607.csv`](data/signals/expansion/s3_overlay_tuning_20260607.csv)
  - **税後 CAGR (v2 corrected)**: NISA 内非課税で V0 と同様 pretax 値そのまま使用可 (V7 NISA OOS = +17.69% calendar / +19.18% canonical daily)。課税口座は ×0.8273 を実年数 5.232 年で正規化 → V7 OOS aftertax = **+14.77%** (v1 +12.76% は 2026 YTD partial year バグで -2.01pp 過小評価していた)。詳細 [`aftertax_cagr_v2_20260607.csv`](data/signals/expansion/aftertax_cagr_v2_20260607.csv) / 計算 [`scripts/compute_aftertax_cagr_v2_20260607.py`](scripts/compute_aftertax_cagr_v2_20260607.py)

### REJECT (5+)
| 候補 | 棄却理由 |
|---|---|
| `BAA-10Y × S3 × M2 procyclical` (Phase D 初回) | Bootstrap P=0.39, 偶然性排除できず |
| `nfci_z52w × S3 × M2 defensive` | 全 Bootstrap P < 0.45, 統計的有意性なし |
| Tier 1-3 全 **117 patterns** (BAA10Y/VIX/HY OAS/2s10s/real yield/DXY) | post-hoc 評価の構造的過大評価 — Phase D native で全 REJECT |
| `nasdaq_mom21` 全派生 (PASS 0) | G2 IC 最強 (t=+17) なれど G3 native で全 FAIL |
| Phase A〜D 全 procyclical 候補 | 0 ADOPT (方向性自体が誤り) |

### NEEDS_FURTHER_WORK (3)
| 候補 | 状態 | 次の検証 |
|---|---|---|
| `vix_mom21 × S3 × M2 defensive` | P_MaxDD=0.902 (惜) | mapping 再評価で ADOPT 化を狙う |
| `nasdaq_mom63 × S2 (D5) × M6 def` | WFE=0.963 (FAIL <1.0) | grid search で mapping 最適化 |
| `nasdaq_mom63 × E4 (Active) × M6 def` | WFE=0.958 (FAIL <1.0) | E4 専用 `lev_mod_e4` への直接適用ルート整備 |

---

## 5. 未テスト領域 (新セッション着手対象)

優先度順:

| 優先度 | テーマ | 内容 |
|---|---|---|
| **MID** | 残 44 macro_features 信号の G3 native スクリーニング | より多くの defensive overlay 候補発掘。`data/signals/expansion/untested_signal_inventory_20260605.csv` 参照 |
| **MID** | `vix_mom21 × S3 × M2 defensive` の mapping 再評価 | NEEDS_WORK → ADOPT 化狙い (P_MaxDD=0.902 を改善) |
| **MID** | `nasdaq_mom63 + nfci_z52w` の AND/OR 合成 overlay | 単独 PASS 信号の組合せで強化 |
| **LOW** | 残 24 paid/manual signals のデータ取得 → 全 76 evaluation | データ整備重 |
| **LONG** | Phase X: 別アプローチ — signal-conditional F11 等の新戦略構造設計 | ゼロベース、中長期 |

---

## 6. 環境別ベスト戦略一覧 (採用判断材料)

> [`STRATEGY_REGISTRY.md`](STRATEGY_REGISTRY.md) §2 と同期。**pretax は canonical split (2021-05-08) ベース**、**aftertax は v2 corrected (calendar split 2020-12-31, 実年数 5.232 yr 正規化, ×0.8273 規約)** ベース ([`aftertax_cagr_v2_20260607.csv`](data/signals/expansion/aftertax_cagr_v2_20260607.csv))。
> 🔧 **2026-06-07 末**: 税後 CAGR を v2 に修正。v1 は `years=len()` で 2026 YTD partial year を完全1年として扱うバグを含み、OOS aftertax を 1.9〜3.8pp 過小評価していた。

| 環境 | 戦略 | CAGR_IS (pretax) | CAGR_OOS (pretax) | OOS aftertax v2 (CFD/ETF課税) | OOS aftertax v2 (NISA非課税) | Sharpe_OOS | MaxDD | Worst10Y★ | P10_5Y▷ | Trades/yr | 推奨 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **CFD §1 Active** | E4 RegimeKLT (vz_thr=0.70) | +31.76% | +33.44% | **+24.42%** (v1 +20.99 → +3.43pp) | N/A (NISA 非適用) | 0.892 | -60.01% | +18.67% | +9.78% | 27 | **維持** |
| CFD 副候補 | S1 F10 (vz=0.65, lmax=7, F10ε=0.015) | +32.58% | +36.75% | **+26.85%** (v1 +23.04 → +3.80pp) | N/A | 0.935 | -63.09% | +18.58% | +10.27% | 119 (proxy) | Registry §2 |
| CFD 副候補 | S2 D5 (vz=0.65, lmax=5) | +29.61% | +33.40% | **+23.82%** (v1 +20.48 → +3.34pp) | N/A | 0.950 | -51.82% | +18.33% | +11.87% | 119 (proxy) | Registry §2 |
| **ETF only** | S3 DH-W1 (Asymm+Hyst) | +18.10% | +18.91% | **+14.54%** (v1 +12.57 → +1.97pp) | **+17.43%** (=pretax) | 0.845 | -34.57% | +10.37% | +4.82% | 69 (proxy) | Registry §2 |
| **ETF only ⭐ V0 def** | S3 + `nasdaq_mom63 × M6 def` overlay {1.1,1.0,0.9,0.8} | +16.69% | +18.06% | **+13.84%** (v1 +11.97 → +1.87pp) | **+16.60%** (=pretax) | **0.892** | **-28.74%** | +10.75% | +5.21% | 59 (proxy) | Registry §2 ⭐ (MaxDD改善優先) |
| **ETF only ⭐ V7 pure_boost** (2026-06-07 追加) | S3 + `nasdaq_mom63 × M6 def` overlay {1.20,1.10,1.00,1.00} | **+18.61%** | **+19.18%** | **+14.77%** (v1 +12.76 → +2.01pp) | **+17.69%** (=pretax, calendar) | 0.841 | -34.57% | +11.02% | +5.22% | 26.5 (proxy) | Registry §2 ⭐ (CAGR死守、canonical daily で min>18% 達成、calendar では 17.69% で僅か未達) |

**18% target との関係 (V7)**:
- canonical daily split (§4 / §3.1 系): V7 min(IS+18.61%, OOS+19.18%) = **+18.61% > 18% target 達成**
- calendar split v2 (NISA): V7 min(IS+18.77%, OOS+17.69%) = **+17.69%** — 18% target 僅か未達 (-0.31pp)
- この 1.5pp の差は **OOS 起点定義の 4 ヶ月差 (2021-01-01 vs 2021-05-08)** によるもので、戦略性能の差ではない。NISA 内運用なら canonical daily 値 (+19.18% pretax) の方が実態に近い指標

---

## 7. 重要な実装ファイル (再利用可能なモジュール)

> 新セッションで信号評価を再開する際は、これらを再利用すること。新規スクリプト作成より優先。

### 統合・評価
| ファイル | 役割 |
|---|---|
| [`src/integration/build_strategy_with_signal.py`](src/integration/build_strategy_with_signal.py) | Generic native injector (S1/S2/S3 × M1-M6 × 2 directions) |
| [`src/integration/nine_metric_eval.py`](src/integration/nine_metric_eval.py) | 9+1 metric evaluator with full / relaxed judgment |
| [`src/integration/build_w1_baa.py`](src/integration/build_w1_baa.py) | DH-W1 native overlay template |

### Phase D audit suite
| ファイル | 役割 |
|---|---|
| [`src/integration/phase_d_wfa.py`](src/integration/phase_d_wfa.py) | WFA 50w window 計算 |
| [`src/integration/phase_d_bootstrap.py`](src/integration/phase_d_bootstrap.py) | Multi-metric Bootstrap (CAGR / Sharpe / MaxDD) |
| [`src/integration/phase_d_metrics.py`](src/integration/phase_d_metrics.py) | Phase D 指標計算 |

### Runner scripts
| ファイル | 役割 |
|---|---|
| [`scripts/run_g2_ic_screening.py`](scripts/run_g2_ic_screening.py) | G2 IC screening (52 信号一括) |
| [`scripts/run_g3_native_top5.py`](scripts/run_g3_native_top5.py) | G3 native evaluation (Top5 × 3 戦略 × 5 方式 × 2 方向) |
| [`scripts/run_session4_phase_d.py`](scripts/run_session4_phase_d.py) | Phase D audit runner |
| [`scripts/recompute_9metrics_for_decision.py`](scripts/recompute_9metrics_for_decision.py) | 9 metric canonical recompute |

---

## 8. 新セッション開始時の推奨読書順序

1. **本書** ([`LESSONS_LEARNED_20260607.md`](LESSONS_LEARNED_20260607.md)) — 全体俯瞰、教訓と方針
2. [`SIGNAL_EXPANSION_FINAL_DECISION_20260607.md`](SIGNAL_EXPANSION_FINAL_DECISION_20260607.md) v2 — 詳細な意思決定根拠
3. [`CURRENT_BEST_STRATEGY.md`](CURRENT_BEST_STRATEGY.md) — 戦略台帳の現状 (Active)
4. [`STRATEGY_REGISTRY.md`](STRATEGY_REGISTRY.md) — 全戦略一覧 (Active / Shortlisted / Rejected / Deferred)
5. [`data/signals/expansion/g3_native_top5_results_20260605.csv`](data/signals/expansion/g3_native_top5_results_20260605.csv) — G3 raw 結果 (新候補探す時)
6. [`data/signals/expansion/untested_signal_inventory_20260605.csv`](data/signals/expansion/untested_signal_inventory_20260605.csv) — 未テスト信号一覧 (次の探索対象)

---

## 9. 次セッションで避けるべき罠 (Anti-Patterns)

| ❌ アンチパターン | 理由 / 代替 |
|---|---|
| **Post-hoc multiplication で評価** | 構造的過大評価 (Tier 1-3 で 117 patterns 無駄にした実績)。代替: native integration |
| **Procyclical のみ試す** | Phase A〜D で 0 ADOPT を生んだ。代替: **defensive を先に試す** |
| **CAGR-only Bootstrap** | defensive overlay の真の価値 (MaxDD 改善) を見逃す。代替: 3 metrics (CAGR/Sharpe/MaxDD) |
| **G2 IC 結果のみで native スキップ** | IC ≠ 戦略改善 (nasdaq_mom21 t=+17 が PASS 0 だった事例)。代替: native integration を本評価 |
| **CFD strategy へ overlay を強行** | WFE 構造的劣化を放置 (S2/E4 で WFE<1.0 確認済)。代替: S3 限定で運用、CFD 系は別アプローチ |
| **2018-01-01 split を使う** | `nine_metric_eval.py` の default、非標準。代替: **canonical 2021-05-08 を明示指定** |

---

## 10. 関連ドキュメント

### 最重要
- 本書: [`LESSONS_LEARNED_20260607.md`](LESSONS_LEARNED_20260607.md)
- 最終意思決定書: [`SIGNAL_EXPANSION_FINAL_DECISION_20260607.md`](SIGNAL_EXPANSION_FINAL_DECISION_20260607.md) (v2)
- 戦略台帳: [`STRATEGY_REGISTRY.md`](STRATEGY_REGISTRY.md)
- 現行ベスト: [`CURRENT_BEST_STRATEGY.md`](CURRENT_BEST_STRATEGY.md)

### Audit 証拠
- ADOPT 候補 audit: [`data/signals/expansion/phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md`](data/signals/expansion/phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md)
- 転用 audit (S2/E4): [`data/signals/expansion/session5_transfer_report_20260605.md`](data/signals/expansion/session5_transfer_report_20260605.md)
- Session 4 Phase D 3候補 summary: [`data/signals/expansion/session4_phase_d_summary_20260605.md`](data/signals/expansion/session4_phase_d_summary_20260605.md)

### 規格・規範
- 9指標標準: [`docs/rules/08_evaluation-metrics.md`](docs/rules/08_evaluation-metrics.md)

---

*管理者: 男座員也 (Kazuya Oza)*
