# Phase B 通過信号 → 既存ベスト戦略 統合検証 計画 v1

作成日: 2026-06-04
最終更新日: 2026-06-04

> 本計画は Phase B (`screening_report_20260604.md`) で統計的に有効性を確認した 6 信号を、既存ベスト戦略 3 系統に **どの方法で組み込むと改善するか** を網羅検証する。Phase C (`phase_c_report_20260604.md`) の「買い持ちベースライン比較」は方針誤りであり、本計画で **既存戦略ベースライン比較** に修正する。

---

## 0. 背景・修正理由

`phase_c_report_20260604.md` では、信号 overlay を **買い持ち (NDX/IEF/GLD)** と比較したが、これは既存運用上意味のない比較。実運用は `NEW CANDIDATE`/`D5`/`DH-W1` のいずれかであり、**これらをベースラインとして信号で改善できるかを問うべき**。

本計画では:
- ベースライン = 既存 3 戦略 (S1/S2/S3)
- 評価対象 = 各戦略への信号注入バリアント
- 改善基準 = `docs/rules/08_evaluation-metrics.md` の 9+1 指標
- 採用判定 = Pareto 改善 (≥2 指標改善 + 重大悪化なし) + 統計頑健性

---

## 1. ベースライン戦略 (S1 / S2 / S3)

| 略称 | フル名 | プラットフォーム | CAGR_OOS | Sharpe_OOS | MaxDD | Trades/yr | NAV 取得元 |
|---|---|---|---|---|---|---|---|
| **S1** | `vz=0.65 + lmax=7 + F10ε=0.015` (NEW CANDIDATE) | CFD (max lev 7x) | +21.49% | +0.83 | -66.0% | 52 | `audit_results/_cache/f10_nav_cache.pkl` (要確認) |
| **S2** | `D5: vz=0.65 / lmax=5.5` | CFD | +17.87% | +0.79 | -55.88% | ~50 | `audit_results/_cache/vz065lmax5_nav_cache.pkl` |
| **S3** | `DH-W1 (Asymm+Hyst, DH base)` | ETF (TQQQ/TMF/GLDM) | +12.24% | +0.65 | -38.9% | 37 | 新規生成要 (`src/dh_w1_*.py` から) |

### 1.1 ベースライン NAV 取得手順

| 戦略 | 取得方法 | 確認/生成タスク |
|---|---|---|
| S1 | 既存キャッシュ pickle ロード (`pd.read_pickle`) | **CHECK-1**: `audit_results/_cache/` 内のキャッシュ確認、命名特定 |
| S2 | 同上 | **CHECK-2**: `vz065lmax5_nav_cache.pkl` の生成元コード特定 |
| S3 | 既存スクリプト (`src/c4_dh_w1_*.py` 系) で NAV 再生成 | **GEN-1**: 該当スクリプト実行 + 結果キャッシュ |

これら 3 NAV を `data/signals/integration/baseline_navs_20260604.parquet` に統合保存し、以降のすべての検証で参照する。

---

## 2. Phase B 通過信号 (6 unique / 17 triples)

| ID | 信号 | Phase B 有効資産 (mean_IC, 20d / 60d) | 量子化スキーマ | データ source |
|---|---|---|---|---|
| **#6** | VIX level | NDX (+0.22 / **+0.34**) | quantile_cut 4lv | `data/vixcls_daily.csv` |
| **#21** | ICE BofA HY OAS | NDX (+0.27 / **+0.53★**), IEF (-0.07 / -0.11) | quantile_cut 4lv | `data/hy_spread_daily.csv` |
| **#23** | BAA-10Y credit spread (HY-IG proxy) | NDX (+0.16 / +0.20) | quantile_cut 4lv | computed: `data/baa10y_daily.csv` |
| **#26** | 2s10s spread | NDX (-0.07 / -0.06) | quantile_cut 4lv | computed: DGS10 - DGS2 |
| **#28** | 10Y real yield | GLD (+0.07 / +0.05), IEF (+0.10 / +0.07) | quantile_cut 4lv | computed: DGS10 - CPI YoY |
| **#41** | DXY | NDX (+0.09 / +0.20), GLD (+0.10) | quantile_cut 4lv | `data/dxy_daily.csv` |

### 2.1 信号の方向 (Phase B 実測ベース)

| Signal | NDX | IEF | GLD |
|---|---|---|---|
| #6 VIX | +直 (高VIX→順方向) | — | — |
| #21 HY OAS | +直 | -逆 | — |
| #23 BAA10Y | +直 | — | — |
| #26 2s10s | -逆 | — | — |
| #28 10Y real yield | — | +直 | +直 |
| #41 DXY | +直 | — | +直 |

これは Phase B で統計的に確認された **forward return との関係**。各 method で **procyclical 注入** は本表の符号に従う。

---

## 3. 注入方法 (Injection Methods M1-M5)

各 (signal × strategy) ペアに対し、以下のいずれかの方式で組込:

### M1: Leverage Mask (Binary Gate)

```
sig_binary = (signal_q >= mask_threshold).astype(int)   # 0 or 1
new_lev = base_lev * sig_binary  (defensive) or new_lev = base_lev * (1 - sig_binary*0.5) (procyclical inversion)
```

**変数**:
- `mask_threshold`: signal_q ∈ {0,1,2,3} から閾値 ∈ {2, 3} (2variants)
- `direction`: defensive (高信号→cash化) / procyclical (Phase B方向にbias)

### M2: Continuous Leverage Tilt

```
mult_map = {0: m0, 1: m1, 2: m2, 3: m3}   # signal_q level → leverage multiplier
new_lev = base_lev * mult_map[signal_q]
```

**variants**:
- `M2-Defensive`: {0: 1.2, 1: 1.0, 2: 0.7, 3: 0.3} (高信号で減レバ)
- `M2-Procyclical`: {0: 0.7, 1: 0.9, 2: 1.1, 3: 1.3} (高信号で増レバ)

### M3: Asset Tilt (Rotation, S3 のみ)

DH-W1 は 3資産 (TQQQ/TMF/GLDM) を持つので、信号で配分比率を動的調整:

```
alloc_map = {0: [0.7, 0.2, 0.1], 1: [0.5, 0.3, 0.2], 2: [0.3, 0.4, 0.3], 3: [0.1, 0.5, 0.4]}
weights = alloc_map[signal_q]
```

**variants**:
- `M3-RiskOff`: 上記 (高信号 risk-off → defensive配分)
- `M3-Reverse`: 反転 (高信号で TQQQ heavy)

### M4: Vol Target Modifier

```
vol_target_adj = base_vol_target * vol_mult_map[signal_q]
new_lev = vol_target_adj / realized_vol_lookback
```

**variants**:
- `M4-VolAdj`: vol_mult = {0: 1.5, 1: 1.0, 2: 0.7, 3: 0.5} (高信号でvol_target↓)

### M5: Entry/Exit Filter (Hard Gate)

```
if signal_q >= exit_threshold: exit_all()  # forced exit
if signal_q < entry_threshold and not in_position: skip_entry()
```

**variants**:
- `M5-StopOnly`: exit_threshold=3, entry_threshold=0 (extreme時のみexit)
- `M5-FilterEntry`: exit_threshold=3, entry_threshold=1 (低信号時はentry禁止)

---

## 4. パターン総数マトリクス (156 patterns)

### Tier 1 (高優先, 必須): Single signal × Base strategy × M1/M2

| 計算 | 数 |
|---|---|
| 6 信号 × 3 戦略 × M1 × 2方向 | 36 |
| 6 信号 × 3 戦略 × M2 × 2方向 | 36 |
| **Tier 1 小計** | **72** |

### Tier 2 (中優先): M3 / M4 / M5

| 計算 | 数 | 備考 |
|---|---|---|
| M3 (asset tilt): 6 × **S3のみ** × 2方向 | 12 | DH base 構造必須 |
| M4 (vol target): 6 × 3 × 1方向 | 18 | |
| M5 (entry/exit): 6 × 3 × 2 variants | 36 | StopOnly + FilterEntry |
| **Tier 2 小計** | **66** | |

### Tier 3 (探索): AND/OR Combinations

| 計算 | 数 |
|---|---|
| Top4信号 (#21, #6, #41, #23) → 6 pairs × 2 ops | 12 合成信号 |
| 12 合成 × 3 戦略 × M1 (mask) | **36** |

### Tier 4 (補助): PCA Composite from Phase B

| 計算 | 数 |
|---|---|
| Phase B PCA blocks (credit_stress, sentiment) × 3 戦略 × M2 | 6 (実装時可能なら) |

### 総数: **180 patterns** (Tier 1+2+3 = 174 + Tier 4 補助 6)

> 縮小オプション: Tier 2 の M5 を 1 variant に絞れば -18、Tier 3 を Top3信号 (#21,#6,#41) のみで 3 pairs × 2 × 3 = 18 → 計 156。

実行は Tier 1 → Tier 3 → Tier 2 → Tier 4 の優先順で、**Tier 1 結果次第で Tier 2/3 の縮小** を判断する **gating execution** とする。

---

## 5. 評価指標 (9+1 metric standard, `docs/rules/08`準拠)

各バリアントについて以下 10 指標を計算し、対応する **ベースライン戦略 (Si)** との差分を出す。

| 略号 | metric | 改善判定基準 | 重大悪化 上限 |
|---|---|---|---|
| ⑨ | **CAGR_OOS** | +0.5pp 以上 | -1.0pp まで |
| ⑨ | 累積 CAGR OOS/IS 比 | OOS/IS が baseline比で改善 | -10% まで |
| 直接 | **IS-OOS gap CAGR** | \|gap\| が baseline比 0.5pp 縮小 | +1.5pp 拡大まで |
| © | **Sharpe_OOS** | +0.03 以上 | -0.03 まで |
| © | **MaxDD** | -2pp 以上改善 (絶対値減少) | +5pp 悪化まで |
| ⑨ | **Worst 10Y★ CAGR** | +0.5pp 以上 | -1pp まで |
| ⑨ | **P10 5Y▷ CAGR** | +0.5pp 以上 | -1pp まで |
| ◎ | **Trade 回/年** | (制約なし) | 200/yr 超過は不採用 |
| ◎ | **Overfit (WFE)** | ≥1.0 を維持 | <0.95 は不採用 |
| r | **CI95_lo (rolling WFA)** | +0.1 以上 | -0.05 まで |

---

## 6. 採用判定 (3段階)

各 (signal, strategy, method, direction) 組合せについて:

### 6.1 Strong PASS (即採用候補)

- 上記 10 指標のうち **≥4 指標が改善判定基準を満たす**
- **どの指標も重大悪化なし**
- G3 WFA: CI95_lo > 0, WFE > 1.0
- Bootstrap P(候補 > base) > 0.80 (Phase C 0.90 から緩和、実データ短いため)

→ `STRATEGY_REGISTRY.md` に Active 候補として登録

### 6.2 Standard PASS (Phase D 評価対象)

- ≥2 指標改善 + 1 指標も悪化軽微 (重大悪化なし)
- G3 PASS

→ Phase D で精密 audit (`g20`/`g30` 系) に投入

### 6.3 Marginal / FAIL

- 1 指標のみ改善 or 重大悪化あり

→ 棄却 (理由記録)

---

## 7. 多重比較補正

180 patterns × 10 metrics = 大量検定。"best-of-K" honest 評価のため:

- **SPA test (Hansen)** を Tier ごとに実施
- Tier 1 終了時: 72 patterns 内で SPA p_consistent 検査
- 全 Tier 終了時: 180 patterns 全体で SPA + Romano-Wolf

SPA p_consistent < 0.10 を必須要件として課す。

---

## 8. 実行スケジュール (11セッション目安)

| Session | 内容 | パターン数 | 出力 |
|---|---|---|---|
| **S1** (本セッション) | 計画 spec 確定 + 検証準備 | — | `SIGNAL_INTEGRATION_PLAN_20260604.md`, `scripts/prepare_baseline_navs.py` |
| **S2** | Baseline NAV 3戦略確定 + Tier 1 (S1=NEW CANDIDATE) | 24 | NAV parquet + Tier1-S1 results CSV |
| **S3** | Tier 1 (S2=D5) | 24 | Tier1-S2 results CSV |
| **S4** | Tier 1 (S3=DH-W1) | 24 | Tier1-S3 results CSV |
| **S5** | Tier 1 集計 + gating判断 + Tier3 (combinations) | 36 | Tier1 summary + Tier3 results |
| **S6** | Tier 2 M3 (asset tilt, S3only) | 12 | Tier2-M3 results |
| **S7** | Tier 2 M4 (vol target, 3戦略) | 18 | Tier2-M4 results |
| **S8** | Tier 2 M5 (entry/exit, 3戦略×2variants) | 36 | Tier2-M5 results |
| **S9** | Tier 4 PCA Composite | 6 | Tier4 results |
| **S10** | 全 Tier 統合集計 + SPA 多重比較 | — | `integration_results_20YYMMDD.csv` |
| **S11** | 採用判定 + STRATEGY_REGISTRY 更新 + `INTEGRATION_DEBATE_<date>.md` | — | 採用候補確定, CURRENT_BEST_STRATEGY 更新可能性 |

各 Session 完了時に commit + push。Tier 1 が全て FAIL なら Tier 2/3 縮小判断。

---

## 9. 実装 (本セッション開始準備分)

### 9.1 必須準備 modules

```
src/integration/
├── __init__.py
├── baseline_loader.py        # S1/S2/S3 ベースライン NAV 取得
├── injection_methods.py      # M1-M5 の実装 (overlay の拡張)
├── pattern_enumerator.py     # 180 patterns の (signal, strategy, method, direction) 列挙
├── nine_metric_eval.py       # 10指標計算 + ベースライン差分
└── adoption_judge.py         # Strong/Standard/Marginal 判定

scripts/
├── prepare_baseline_navs.py  # S1/S2/S3 NAV を baseline_navs_20260604.parquet に保存
└── run_integration_tier.py   # Tier ごとの実行ランナー (--tier 1/2/3/4 で切替)
```

### 9.2 出力データ構造

```
data/signals/integration/
├── baseline_navs_20260604.parquet     # S1/S2/S3 daily NAV
├── tier1_results_20YYMMDD.csv         # Tier1 (72 patterns) の 10指標
├── tier2_results_20YYMMDD.csv         # Tier2
├── tier3_results_20YYMMDD.csv         # Tier3
├── tier4_results_20YYMMDD.csv         # Tier4
├── integration_full_20YYMMDD.csv      # 全 patterns 統合
└── integration_report_20YYMMDD.md     # 最終採用判定レポート

docs/signals/
└── integration_debate_20YYMMDD.md     # 採用判定議事録
```

### 9.3 Pattern Schema (CSV header)

```
pattern_id, tier, signal_id, signal_name, base_strategy, method, direction,
cagr_oos, cagr_oos_diff, oos_is_ratio, oos_is_diff,
is_oos_gap, is_oos_gap_diff, sharpe_oos, sharpe_diff, maxdd, maxdd_diff,
worst_10y, worst_10y_diff, p10_5y, p10_5y_diff,
trades_yr, wfe, ci95_lo, ci95_lo_diff,
n_improved, n_degraded, judgment
```

---

## 10. 成功基準

### 10.1 定量

- Tier 1 (72 patterns) 完了時、**Strong PASS が 1件以上** 存在 → Tier 2/3 縮小不要
- 全 Tier 完了時、**Standard PASS 5-10件、Strong PASS 1-3件** が現実的期待値
- SPA p_consistent < 0.10 維持

### 10.2 定性

- 採用候補は **経済的に説明可能** な組合せ (Phase B の信号方向に整合)
- ベースライン戦略を **CAGR下げず Sharpe/MaxDD/Worst10Y のいずれかで明確改善**

---

## 11. リスク・代替

| リスク | 対処 |
|---|---|
| ベースライン NAV キャッシュ無し | DH-W1 系スクリプト再走行で 1日かけて生成 |
| 180 patterns 計算時間が膨大 | Tier 1 で gating + 並列化 (multiprocessing) |
| Phase B PASS 17 triple ≠ 6 signals × 全資産PASS (一部資産のみ) | 信号は全資産対応で注入、戦略のbase asset対応で評価 |
| SPA で全 FAIL | 緩和: SPA p < 0.15 まで許容、Phase D 厳格評価で再検証 |
| Tier 1 で 0採用 | 注入マップ最適化 (M2 倍率) を別 Phase で探索 |

---

## 12. 関連ドキュメント

- 設計上位仕様: `SIGNAL_DISCOVERY_PLAN_20260603.md`
- Phase A-C 実装計画: `IMPLEMENTATION_PLAN_SIGNAL_DISCOVERY_20260603.md`
- Phase B 結果: `data/signals/screening_report_20260604.md`
- Phase C 結果 (買い持ち比較・本計画で置換): `data/signals/phase_c_report_20260604.md`
- 9指標規格: `docs/rules/08_evaluation-metrics.md`
- 戦略台帳: `STRATEGY_REGISTRY.md`
- 現行ベスト: `CURRENT_BEST_STRATEGY.md`

---

## 13. 次工程 (本セッション内で着手)

1. ✅ 本 spec を `SIGNAL_INTEGRATION_PLAN_20260604.md` として commit/push
2. **CHECK-1/CHECK-2**: 既存 NAV キャッシュ確認 → S1/S2 ベースライン取得可能性判定
3. **GEN-1**: DH-W1 NAV 再生成スクリプトの所在特定
4. `src/integration/` モジュール骨格作成 + `scripts/prepare_baseline_navs.py` stub
5. Tier 1 実行開始 (S2 セッション)
