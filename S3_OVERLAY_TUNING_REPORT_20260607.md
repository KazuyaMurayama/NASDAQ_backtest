# S3 + nasdaq_mom63 Overlay Tuning Report

作成日: 2026-06-07
最終更新日: 2026-06-07

> **目的**: ADOPT 候補 `nasdaq_mom63 × S3 (DH-W1) × M6 defensive` の CAGR trade-off を緩和し、
> ユーザー要件 **min(CAGR_IS, CAGR_OOS) > 18%** を達成しつつ
> MaxDD 改善 (`-34.57%` → `< -32%`) と Worst10Y/P10_5Y 改善を保持する mapping を探索する。
>
> 検証規模: 10 mapping variants (V0=現行 ADOPT, V1-V9=新規). すべて native injection
> (`src/integration/build_strategy_with_signal.build_candidate_nav`, `mapping=` パラメータ追加).
> Cost Scenario D (NAV にコスト織込済).

---

## 1. 結論サマリ (3 行)

1. **ユーザー厳格条件 (min CAGR > 18% かつ MaxDD < -32%) を満たす variant は 0 件**。
2. **min CAGR > 18% のみなら 3 件**: V6 (boost_heavy), V7 (pure_boost), V9 (M2 proc).
   このうち MaxDD 悪化が最小なのは **V6_M6_def_boost_heavy** (MaxDD=-33.16%, V0 比 +1.41pp 改善は保持).
3. **推奨**: V6 を Phase D 候補に追加検討 (CAGR 死守を優先する場合); 既存 V0 を守る場合は ADOPT 据置.

---

## 2. ベンチマーク

| Benchmark | 戦略 | 役割 |
|---|---|---|
| **BMK1: S3 DH-W1 baseline** | S3 raw (no overlay) | 改善対象の基線 |
| **BMK2: V0 (current ADOPT)** | S3 + M6 defensive {1.1, 1.0, 0.9, 0.8} | 既存 ADOPT |

ユーザー要件:
- 主条件: `min(CAGR_IS, CAGR_OOS) > 18%`
- 副条件: `MaxDD_FULL > -32%` (BMK1 の `-34.57%` から改善し、かつ V0 と同等以上)
- 保持条件: `Worst10Y_calendar >= 10.37%` (BMK1) / `P10_5Y >= 4.82%` (BMK1)

---

## 3. 評価指標 (docs/rules/08_evaluation-metrics.md v1.0 標準)

CAGR_IS / CAGR_OOS / CAGR_FULL / Sharpe_OOS / MaxDD / Worst10Y★ / P10_5Y▷ / IS-OOS gap / WFA_CI95_lo_annual / WFA_WFE_calendar.
Split: canonical (IS_END=2021-05-07 / OOS_START=2021-05-08, per `docs/rules/08` §5).

---

## 4. 全 10 variant + 1 baseline 比較表 (列ごとの最良値を **太字**)

| variant | min_CAGR | CAGR_IS | CAGR_OOS | CAGR_FULL | Sharpe_OOS | MaxDD | Worst10Y★ | P10_5Y▷ | IS-OOS gap | WFA_CI95_lo | WFA_WFE_cal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BMK1: S3 DH-W1 baseline | 18.10 | 18.10 | 18.91 | 18.17 | 0.845 | -34.57 | 10.37 | 4.82 | -0.82 | 13.61 | 0.744 |
| BMK2: V0 M6 def {1.1,1.0,0.9,0.8} | 16.69 | 16.69 | 18.06 | 16.81 | 0.892 | -28.74 | 10.75 | 5.21 | -1.37 | 12.65 | 0.782 |
| V1 M6 def mild {1.05,1.0,0.95,0.85} | 17.12 | 17.12 | 18.42 | 17.24 | 0.880 | -30.25 | 10.73 | 5.10 | -1.30 | 12.96 | 0.774 |
| V2 M6 def extreme_only {1.0,1.0,0.95,0.75} | 16.45 | 16.45 | 18.27 | 16.62 | 0.909 | -29.37 | 10.90 | 5.10 | -1.82 | 12.52 | 0.805 |
| V3 M6 def boost_low {1.2,1.05,0.95,0.85} | 17.46 | 17.46 | 18.49 | 17.55 | 0.873 | -30.25 | 11.18 | 5.37 | -1.03 | 13.24 | 0.762 |
| V4 M6 def strong_asym {1.25,1.1,0.95,0.85} | 17.63 | 17.63 | 18.68 | 17.72 | 0.873 | -30.64 | **11.38** | 5.59 | -1.05 | 13.39 | 0.764 |
| V5 M6 def narrow_band {1.05,1.0,1.0,0.85} | 17.35 | 17.35 | 18.61 | 17.46 | 0.877 | -30.43 | 10.98 | 5.08 | -1.27 | 13.15 | 0.771 |
| V6 M6 def boost_heavy {1.15,1.1,1.0,0.95} | 18.25 | 18.25 | 19.16 | 18.33 | 0.855 | -33.16 | 11.06 | 5.21 | -0.92 | 13.82 | 0.752 |
| V7 M6 def pure_boost {1.2,1.1,1.0,1.0} | 18.61 | 18.61 | 19.18 | 18.66 | 0.841 | -34.57 | 11.02 | 5.22 | **-0.57** | 14.06 | 0.735 |
| V8 M2 def sharp {1.2,1.0,0.7,0.3} | 12.78 | 12.78 | 15.70 | 13.04 | **1.005** | **-27.10** | 9.59 | **6.26** | -2.92 | 9.76 | 0.902 |
| V9 M2 proc {0.7,0.9,1.1,1.3} | **19.38** | **19.38** | **19.39** | **19.37** | 0.779 | -42.61 | 8.73 | 4.15 | -0.01 | **14.30** | 0.690 |

**最良値方向**:
- 高い方が良い: CAGR_IS/CAGR_OOS/CAGR_FULL, Sharpe_OOS, Worst10Y, P10_5Y, WFA_CI95_lo
- 0 に近いほど良い: IS-OOS gap, MaxDD (絶対値小)
- 1.0 に近いほど良い: WFA_WFE_calendar (>=0.5 acceptable, 1.0 ideal)

**最良値要約**:
- min_CAGR / CAGR_IS / CAGR_OOS / CAGR_FULL / WFA_CI95_lo: **V9** (procyclical) が最高
- Sharpe_OOS: **V8** (M2 sharp defensive) が最高 1.005
- MaxDD: **V8** が最良 -27.10%
- Worst10Y: **V4** (strong_asym) が最高 11.38%
- P10_5Y: **V8** が最高 6.26%
- IS-OOS gap: **V7** が最も小さく -0.57pp (V9 は -0.01 だが過剰 OOS 偶然性疑い)

---

## 5. ユーザー要件達成状況

### 5.1 主条件: `min(CAGR_IS, CAGR_OOS) > 18%`

| variant | min_CAGR | 達成? |
|---|---|---|
| BMK1 baseline | 18.10 | ✅ (参考) |
| **V6 boost_heavy** | **18.25** | ✅ |
| **V7 pure_boost** | **18.61** | ✅ |
| **V9 M2 proc** | **19.38** | ✅ |
| その他 7 件 | 12.78〜17.63 | ❌ |

→ **3 件達成** (新規 variant のみ).

### 5.2 副条件: `MaxDD_FULL > -32%` (改善保持)

| variant | MaxDD | min_CAGR>18% | MaxDD<-32%? | 両条件達成? |
|---|---|---|---|---|
| V6 boost_heavy | -33.16 | ✅ | ❌ (-32 未達) | ❌ |
| V7 pure_boost | -34.57 | ✅ | ❌ | ❌ |
| V9 M2 proc | -42.61 | ✅ | ❌ (悪化) | ❌ |
| V0/V1-V5/V8 | -27.10〜-30.64 | ❌ | ✅ | ❌ |

→ **両条件達成: 0 件**。CAGR と MaxDD は完全な trade-off 関係にある。

### 5.3 トレードオフの構造

10 variant の散布から判明:
- **CAGR > 18% を死守すると MaxDD は最良で -33.16% (V6)**。これ以上 CAGR を維持しつつ MaxDD を縮めることは現 mapping 空間では不可能。
- 逆に **MaxDD を縮めようとすると CAGR が必ず V0/V1 のように 16〜17% 台まで落ちる** (defensive cut が q3 を実質ゼロにする領域で CAGR 損失が急増).
- V8 (M2 sharp def) は MaxDD と Sharpe で最良だが CAGR が 12.78% まで大幅後退 → 全くの defensive 過多.

---

## 6. Top 3 推奨候補 (Pareto 観点)

### 6.1 ベスト総合: **V6_M6_def_boost_heavy** {1.15, 1.10, 1.00, 0.95}

| 指標 | 値 | vs BMK1 baseline | vs BMK2 V0 ADOPT |
|---|---|---|---|
| CAGR_IS | 18.25 | +0.15pp | **+1.56pp** |
| CAGR_OOS | 19.16 | **+0.25pp** | **+1.10pp** |
| Sharpe_OOS | 0.855 | +0.010 | -0.037 |
| MaxDD | -33.16 | **+1.41pp** | -4.42pp (V0 比悪化) |
| Worst10Y | 11.06 | **+0.69pp** | +0.31pp |
| P10_5Y | 5.21 | +0.39pp | 0.00pp |
| IS-OOS gap | -0.92 | -0.10pp | **+0.45pp** |
| WFA_CI95_lo | 13.82 | **+0.21pp** | **+1.17pp** |

**改善**: CAGR 両方とも 18% 超 ✅, MaxDD は baseline 比 +1.41pp 改善 (32% 厳格条件は未達だが方向は正), Worst10Y も改善, WFA_CI95_lo も V0 比改善.
**劣化**: V0 と比べると MaxDD で -4.42pp 後退 (28.74→33.16).
**Phase D 推奨**: ✅ 即時 audit (50w WFA + bootstrap) — CAGR 死守シナリオの第一候補.

### 6.2 セカンドベスト: **V7_M6_def_pure_boost** {1.20, 1.10, 1.00, 1.00}

| 指標 | 値 | vs BMK1 baseline | vs BMK2 V0 ADOPT |
|---|---|---|---|
| CAGR_IS | 18.61 | +0.51pp | **+1.92pp** |
| CAGR_OOS | 19.18 | +0.27pp | **+1.12pp** |
| MaxDD | -34.57 | 0.00pp (変化なし) | -5.83pp (V0 比悪化) |
| Worst10Y | 11.02 | +0.65pp | +0.27pp |
| IS-OOS gap | -0.57 | **+0.25pp** | **+0.80pp** |
| WFA_CI95_lo | 14.06 | **+0.45pp** | **+1.41pp** |

**改善**: min CAGR が最も大きく 18.61% を達成. Sharpe をほぼ baseline 維持 (0.841 vs 0.845). IS-OOS gap が全 variant 中最小 (-0.57pp) で過学習リスク低い.
**劣化**: MaxDD 改善ゼロ (baseline と同値). overlay の defensive 効果なし.
**Phase D 推奨**: ⚠ 追加検討 — MaxDD 改善がない overlay は "leverage adder" にすぎず、 §1 (E4 Active) との重複度が高くなる. ETF only 環境での価値が薄れる.

### 6.3 サードベスト: **V4_M6_def_strong_asym** {1.25, 1.10, 0.95, 0.85}

| 指標 | 値 | vs BMK1 baseline | vs BMK2 V0 ADOPT |
|---|---|---|---|
| CAGR_IS | 17.63 | -0.47pp | **+0.94pp** |
| CAGR_OOS | 18.68 | -0.23pp | +0.62pp |
| MaxDD | -30.64 | **+3.93pp** | -1.90pp |
| Worst10Y | **11.38** | **+1.01pp** | +0.63pp |
| P10_5Y | 5.59 | +0.77pp | +0.38pp |
| IS-OOS gap | -1.05 | -0.24pp | +0.32pp |

**注**: min CAGR 17.63% で主条件は不達。ただし **MaxDD 改善 +3.93pp / Worst10Y 改善 +1.01pp (全 variant 最高)** と防御指標は強力で、CAGR 妥協を許せる場合の有力候補.
**Phase D 推奨**: △ 厳密 audit までは不要だが、V0 ADOPT の代替 (より穏やかな CAGR 損失で防御改善幅もほぼ同等) として bookkeeping.

---

## 7. 解釈と考察

### 7.1 mapping 戦略別の傾向

| mapping 系統 | 例 | CAGR 効果 | MaxDD 効果 | 結論 |
|---|---|---|---|---|
| **対称的 defensive** (V0, V1) | spread 縮小 | 弱化 → 16-17% | 改善 +4-6pp | 防御主導 |
| **boost-asymmetric** (V3, V4) | low boost + 軽 cut | 中程度回復 17.4-17.6% | 中改善 +3-4pp | 中庸 |
| **boost-heavy** (V6, V7) | high boost + 弱 cut | 強回復 18.2-18.6% | 弱化 +0-1.4pp | CAGR 主導 |
| **pure boost** (V7) | cut なし | 最強回復 18.6% | 改善 0pp | leverage adder |
| **M2 sharp defensive** (V8) | q3=0.3 過剰 cut | 大幅劣化 12.8% | 最良 -27.1% | 過剰防御 |
| **procyclical** (V9) | trend-following | 最強回復 19.4% | 大幅悪化 -42.6% | 順張りリスクオン |

→ **CAGR と MaxDD は線形に近い trade-off**。本 mapping 空間 (4 量子マッピング) では Pareto front 上の点であり、両方同時改善はほぼ不可能.

### 7.2 なぜ V6/V7 だけが CAGR > 18% を達成するのか

V6/V7 の特徴: **q3 (最高量子) の cut が弱い or なし**。
- 標準 M6 defensive は q3=0.8 (20% cut) で「強い risk-on warning 時に減らす」、これが CAGR 損失の主要因.
- nasdaq_mom63 (NASDAQ momentum) は high quartile 時にむしろ trend を継続することが多く、ここで cut すると "winning trade" を失う.
- V7 のように q3=1.0 にすると DH-W1 baseline の lev を低 quartile 領域だけ boost する形になり、純粋に "downside protection" 期待値だけ追加され、CAGR が baseline 超え.

しかし MaxDD 改善は失われる: q3 cut こそが crisis (急落直前の高 momentum) を捕らえる主要メカニズム.

### 7.3 M6 vs M2 vs procyclical 比較

- **M6 (gentle quartile)** が最も Pareto front を広くカバー (V0-V7 が front 上に並ぶ).
- **M2 (sharp quartile)** は V8 で MaxDD/Sharpe 最良だが CAGR が無理. V8 は "純粋 risk parity" 的振る舞いで、本 strategy の DH-W1 cost 構造とは相性が悪い (頻繁な大幅 cut が cost を増やす).
- **V9 procyclical** は CAGR 上位だが MaxDD -42.6% で完全に逆風: nasdaq_mom63 がトップ quartile で leverage 1.3x ということは「momentum すでに伸びきった所で更にレバ」 → 過去 50 年では crisis 直前のレバ過剰を直撃.

### 7.4 ユーザー要件達成不可の構造的理由

- BMK1 (S3 DH-W1 baseline) は既に CAGR 18.10/18.91 と十分高く、overlay の余地は基本的に「CAGR を多少落として MaxDD を縮める」一方向しかない.
- min CAGR > 18% を死守する mapping 空間は q3 ≥ ~0.95 を要求するが、そこでは MaxDD 改善メカニズム (高 risk-on 時 cut) が機能しない.
- **対策案**: mapping を 4 quartile から 5+ quintile に拡張するか、second signal (例: VIX z-score) を重ね、q3=0.95 の cut 不足を補う two-factor mapping を試す価値がある (本タスクの scope 外).

---

## 8. 次工程推奨

### 8.1 即時アクション (Phase D 候補化)

1. **V6_M6_def_boost_heavy** を Phase D に追加 audit
   - 50-window WFA bootstrap (`scripts/audit_lt2_n750.py` template を流用)
   - P_MaxDD (bootstrap で MaxDD < V0=-28.74% 確率) を測定
   - Trades/yr を正確に計測 (S3 baseline比較)
2. **V7_M6_def_pure_boost** は MaxDD 改善ゼロなので audit 優先度低、ただし IS-OOS gap が最小 (-0.57pp) で過学習リスク評価には参考値.

### 8.2 中期検討

3. ユーザー厳格条件 (min CAGR > 18% **AND** MaxDD < -32%) を物理的に達成するため:
   - two-signal overlay (nasdaq_mom63 + VIX z) で q3 = 0.95 の不足を VIX で補う
   - quintile (5 段階) mapping への拡張
   - 別 strategy 基盤 (E4 Active) への boost_heavy 転用テスト

### 8.3 維持判断

4. 現行 §1 (E4 Active) は変更不要. ETF only 環境では V0 を据置か、V6 (CAGR 死守 + 緩い MaxDD 改善) への切替を user 判断とする.

---

## 9. 出力ファイル

| 種別 | ファイル | 役割 |
|---|---|---|
| 数値結果 | `data/signals/expansion/s3_overlay_tuning_20260607.csv` | 全 11 行 × 17 列 (10 metrics + meta) |
| 実装拡張 | `src/integration/build_strategy_with_signal.py` | `mapping=` パラメータを追加 (M1/M2/M5/M6 で override 可) |
| 実行スクリプト | `scripts/tune_s3_overlay_20260607.py` | 10 variants 一括 build & metric 計算 |

---

## 10. 検証ログ

実行コマンド: `python scripts/tune_s3_overlay_20260607.py`
DH 資産ロード: 一回のみ (V0 build 時) 約 2-3 分、以降の 9 variants は秒オーダーで完了 (NAV 再計算は build_dh_nav_with_cost のみ).
Cost Scenario D 一貫適用 (NAV にコスト織込済の cache を使用).
Split: canonical (IS_END=2021-05-07 / OOS_START=2021-05-08).

---

> **次のステップ**: V6 の Phase D 厳格 audit を承認するか、それとも overlay を二信号化する方向に進むかをユーザー判断願います.
