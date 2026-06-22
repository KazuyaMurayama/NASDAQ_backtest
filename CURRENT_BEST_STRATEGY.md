# CURRENT BEST STRATEGY — 単一の真実 (Single Source of Truth)

> **このファイルは「現行のベスト戦略」を一意に特定するための正典です。**
> **「ベスト戦略は？」と問われた時、Claude / 人間ともにまずこのファイルだけを見れば良いように設計されています。**

作成日: 2026-05-11
最終更新日: **2026-06-15 (v8: ベスト戦略を B3a_k365、ベスト戦略候補を P09_C1 に確定。冒頭 §🏆 に追記。一次根拠 LEVERUP_SWEEP_RESULTS_20260612.md・QC LEVERUP_QC_SIGNOFF_20260613.md[独立再実装で§6.1値80/79再現]。本番稼働は E4 継続・P09 GAS並走・切替判断7月)** / 2026-06-07 (v4.9.2: 税後 CAGR を canonical split (IS_END=2021-05-07) で統一。`scripts/compute_aftertax_cagr_v3_20260607.py` で全戦略を pretax と同じ canonical split に同期、calendar/canonical split duality を廃止) / 2026-06-08 (標準10指標・IS/OOS min 表記・ⓒ/⓽コスト前提明確化 全セクション適用) / 2026-06-10 (v4.5推奨表: 全値⓽に統一・比較注意書き削除、V0/V7 overlay・P7投信の3行追加) / 2026-06-10 v2 (v4.5推奨表: E4 ⓽行追加で§1 Active vs vz065_l5 を同一基準で直接比較可能化) / **2026-06-10 v3 (コスト誤謬修正: E4 CAGR⓽ を CFD_SPREAD_LOW=0.20%/yr の誤値 +27.41% から SBI CFD 3.0% 正値 +20.0% に修正)** / **2026-06-10 v4 (構成・コスト注意事項削除、一次根拠を SBI CFD g14 ベースに更新、Shortlisted から CFD_SPREAD_LOW 誤値を除去)** / **2026-06-11 v5 (v4.5表を realistic full L×(SBI CFD建玉金利=想定元本全額, 一次確認済) + 正典窓(49窓)WFA で更新。E4 ⓽OOS: g14 (L-1)×借入基準 +22.4% → full L×正基準 +18.06% に訂正。R4: DH-W1 Trades 68.7→17.6 訂正(NAV符号反転の疑似指標を実リバランス値に修正)。R9: vz065_l7 N/A→CI95+16.45%/WFE1.328 補填。⚠マーク除去・vz065 WFE CAUTION注記に置換)** / **2026-06-11 v6 (ETF指定戦略 DH-W1/V0/V7/P7 の NASDAQ脚コストを CFD→TQQQ に是正。設計商品TQQQの正値で点指標を更新: DH-W1 min +14.73→+15.85%, V0 +13.43→+14.27%, V7 +15.07→+16.27%, P7 +15.84→+16.92%。CFD環境(E4/vz065)は不変。WFA列は†CFD基準据置・TQQQ再計算pending)** / **2026-06-11 v7 (ベスト候補再構成: CFD戦略 E4/vz065_l7 と ETF DH-W1/V0 を §アーカイブへ退避[削除せず指標保存]、vz065_l5 はCFD候補として残置[ユーザー判断]、V7-TQQQ/P09_TQQQ/LU1 を主候補・LU2 を別枠に追加。v6 の † WFA を TQQQ基準・正典窓49窓で再計算し確定[CFD分枝が旧†値を dWFE≤0.0002 で再現]、† 解除。CI95_lo: V7 +14.09→+15.48 / P7 +16.57→+17.85。標準10指標列を compute_10metrics §3.12 と全件整合確認。E4 は本番Active継続中=切替はユーザー判断)** / **2026-06-11 v7.2 (P09_TQQQ Active候補昇格を反映、最終確定評価レポート HORIZON_AND_SCORECARD_20260611.md への参照を追加[6次元スコアカード P09 8.12首位級・5y/10y分布・年次リターン表]、一次根拠ファイル表に同レポートと EVALUATION_UPGRADE_RESULTS_20260611.md を追加)**

---

## 🏆 現行ベスト戦略（v8・2026-06-15 確定）

> **ベスト戦略 = B3a_k365 ／ ベスト戦略候補 = P09_C1**（ユーザー決定 2026-06-15）。
> 一次根拠: [LEVERUP_SWEEP_RESULTS_20260612.md](LEVERUP_SWEEP_RESULTS_20260612.md)（Phase B/C・6次元採点・multi-metric bootstrap §8.1）。
> 品質保証: [LEVERUP_QC_SIGNOFF_20260613.md](LEVERUP_QC_SIGNOFF_20260613.md)（3エージェント独立レビュー＋**独立再実装で§6.1値を80セル中79再現**、コスト/レバ/充填/C1/k365/税/WFA/CPCV/bootstrap の実装健全性を確認）。
> 全値⓽税後（×0.8273・特定口座20.315%課税。TQQQ/くりっく株365 はNISA対象外）。min(IS,OOS)=保守的採用基準。

### ベスト戦略: B3a_k365（CAGR重視）

**構成**: DH-W1（Asymm Hysteresis, Enter≥0.7/Exit≤0.3）＋ mom63 V7ブーストマップ **{Q0:1.40, Q1:1.40, Q2:1.05, Q3:1.00}** × uniform leverage **×1.15** ＋ P09 OUT充填（Gold常時＋Bond@`bond_mom252>0`・逆ボラW63・T+5ラグ）＋ **C1**（OUT∧bondOFF日に現金利回りSOFR計上）。**コスト**: ≤3×=TQQQ（swap0.5%+TER0.86%）、>3×超過分=くりっく株365（取引所CFD・SOFR+0.75pp ≒ 超過0.25%/yr）。

| 指標（⓽税後 / ⓒ税前） | 値 | 備考 |
|---|---:|---|
| CAGR_IS⓽ / CAGR_OOS⓽ | +23.10% / **+20.98%** | min(IS,OOS)=OOS |
| **min⓽（点推定／選択バイアス割引後）** | **+20.98% / ≈20.1%** | best-of-44のため割引。bootstrap下限18.2% |
| IS-OOS gap⓽ | +2.57pp | 過学習兆候は軽微 |
| Sharpeⓒ / MaxDDⓒ | 0.904 / **−38.2%** | DDはP09比 −3.2pp（レバの対価） |
| Worst10Y★⓽ / P10_5Y⓽ / Worst5Y⓽ | +14.53% / +8.08% / +0.10% | 10年テール最強。Worst5Yは実質ゼロ（感度で反転） |
| WFA CI95_loⓡ / WFEⓞ | +22.52% / 0.987 | α/β PASS |
| CPCV p10 / Regime_min | +16.01% / −2.88% | Regime_minは弱気相場で悪化（レバの帰結） |
| boot P(min vs V7) | 0.893 | 強い傾向・95%未達（年次blockで0.94接近） |
| Trades/yrⓞ / >3x日比率 | 33.3 / 37.7% | >3x日はくりっく株365運用 |
| worst暦年 | −22.4%（2015） | P09より深い（5年未満取崩し資金には不適） |

### ベスト戦略候補: P09_C1（バランス・risk-adjusted重視）

**構成**: V7-TQQQ（デフォルトmom63マップ {Q0:1.20, Q1:1.10, Q2:1.00, Q3:1.00}・scale 1.0）＋ P09 OUT充填 ＋ C1。**>3×超過課金なし**（ただし約6.1%の日は実効L>3＝完全CFD不要ではない）。

| 指標（⓽税後 / ⓒ税前） | 値 |
|---|---:|
| CAGR_IS⓽ / CAGR_OOS⓽(=min) | +19.88% / **+17.77%** |
| Sharpeⓒ / MaxDDⓒ | 0.912 / −34.99% |
| Worst10Y★⓽ / P10_5Y⓽ / Worst5Y⓽ | +11.49% / +7.02% / −0.58% |
| WFA CI95_loⓡ / WFEⓞ / CPCV p10 / Regime_min | +18.96% / 0.989 / +14.15% / **−0.08%（最良）** |
| boot P / Trades/yrⓞ / >3x日比率 | 0.820 / 29.2 / ~6.1% |
| 2008防御 / worst暦年 | **+19.86%（最強）** / −18.7%（2015） |

**位置づけ**: 6次元バランス採点で首位（8.50）・CFD依存最小・2008防御最強・Regime_min最良。min⓽は20%未達だが、**Sharpe（risk-adjusted）はB3aと同等**で、純レバアップでないぶん頑健。**B3aのSharpeはP09比で中立**＝レバはリターンもリスクも比例増（multi-metric bootstrap §8.1）。

### 採用判定・全ゲートPASS

B3a/P09_C1 ともハードベト無し（MaxDD<−50%/WFE>1.5/W10Y★<0/Regime_min<−10%）・WFA正典49窓・CPCV(45fold)・レジーム層別・**multi-metric bootstrap（MaxDD/Worst10Y★/Sharpe 対V7・対P09）完了**。B3aの採用是非は「CAGR+10年テール取得 vs MaxDD−5〜7.5pp悪化・Sharpe中立」のリスク選好判断（CAGR優先・DD許容と整合）。

### B3a レバ水準ダイヤル（B3c ↔ B3a ↔ レバ拡張フロンティア）

B3c/B3a/拡張は**マップ同一・uniform leverage のみ差＝連続したリスク選好ダイヤル**（[LEVERUP_EXTENSION_RESULTS_20260616.md](LEVERUP_EXTENSION_RESULTS_20260616.md)）。

| 点 | scale | min⓽ | MaxDD | Sharpe | 性格 |
|---|---:|---:|---:|---:|---|
| B3c | 1.10 | +20.41% | −37.1% | 0.911 | DD最浅・WFE最良 |
| **B3a（現ベスト）** | 1.15 | +20.98% | −38.2% | 0.904 | バランス点 |
| 拡張 | 1.25 | +22.07% | −40.4% | 0.892 | やや攻め（取引不変） |
| 拡張 | 1.35（強map） | +23.83% | −45.0% | 0.882 | 攻め・CI95_lo最高+26.6% |
| 上端(combo) | — | +23.97% | −48.1% | 0.943 | ベト間近・非推奨 |

**CAGRは scale を上げれば +24%近くまでベト内で伸びる（純レバ＝Sharpe非改善・MaxDD/worst年(−26%)/bear(−5.40%)を比例超で食う・min改善は95%非有意）**。フリーランチは無く**レバ点はリスク選好で選ぶ**。QC v2（独立4エージェント・[サインオフ](LEVERUP_EXTENSION_QC_SIGNOFF_20260616.md)）: **数値正・scale1.25/1.35のCAGR優位はOOS境界(2019-2023)に頑健**（単一強気OOS依存を否定）。ただし**Worst5Yは scale1.25で−0.04%/combo−0.28%にマイナス転落**（B3a +0.10%・scale1.35強mapのみ+0.33%）。**combo/enter0.60はOOS境界脆弱＋MaxDD実運用ベト超過リスクで非推奨**。B3aはバランス点維持、CAGR上端の現実解は scale1.35強map。
>
> **実務限界検証（2026-06-17・[MARGIN_CAPACITY_STRESS_RESULTS_20260617.md](MARGIN_CAPACITY_STRESS_RESULTS_20260617.md)）**: scale1.35強map（最大レバ6.48倍・>3x日43.6%）の追証/強制ロスカット・容量を全期間検証。清算トリガーは「NASDAQ単日下落≥k365証拠金率」で**レバ水準に非依存**。取引所最小4.24%だと単日−4.24%で清算（全期間81日該当・1987/2008/COVID全清算）→CAGR−1.18pp・最悪1イベント−34%AUM（底値締め出し）。**だがk365証拠金を実効8%以上（excess建玉の15-20%の余剰現金）にすれば全期間52年で清算0回・損害0pp**。**容量**: くりっく株365が薄く（推定OI 500-10,000枚）、AUM¥30Mで最大329枚／¥100Mで1,096枚＝**容量天井は概ね¥30-100M帯**（実OI確認前提）。TQQQ(≤3x)無制約。**採用条件＝8%証拠金＋容量内AUM**。保守化はレバ点を下げれば清算/容量リスク同時軽減。

### 🟠 P09 レバ拡張フロンティア（前向きに採用検討中・2026-06-21 ユーザー方針確定）

P09_C1（scale 1.0・min⓽+17.77%）の**高CAGR側レバ拡張ダイヤル**。strong-map ×scale で CAGR_OOS を +24〜29% へ延長。**ユーザー方針（2026-06-21）: P09×scale を前向きに採用検討**（[LEVERUP_SCALE_FRONTIER_20260619.md](LEVERUP_SCALE_FRONTIER_20260619.md)）。全数値 税後 ×0.8273。

| scale | CAGR_OOS⓽ | MaxDD | Worst10Y★ | regime_min(bear) | WFE | 性格 / 注記 |
|---|---:|---:|---:|---:|---:|---|
| **1.4** | **+24.34%** | −46.48% | +17.87% | −5.86% | 0.936 | **−50%以内・過学習なしの実質上限＝最も信頼できる拡張点** |
| 1.6 | +26.21% | −51.95% | +19.36% | −7.71% | 0.898 | MaxDD ベト−50%超 |
| 1.8 | +27.81% | −57.00% | +20.22% | −9.56% | 0.858 | gap>5pp 過学習フラグ・bearベト接近 |
| 2.0 | +29.11% | −61.63% | +20.84% | **−11.41%** | 0.818 | gap>5pp・regime_min<−10% ハードベト |

**純レバアップ**＝Sharpe非改善（1.07→1.03漸減）・MaxDDは scale比例で悪化。**DD許容ありでも −50%以内に収めるなら実質 scale≈1.4が上限**（1.6で既に−51.95%）。高scale(1.8/2.0)は過学習＋bearハードベトで非推奨。本番採用は B3a/P09_C1 等との最終比較とユーザー承認待ち。

> **❌ B1×Scale（不採用・2026-06-21）**: B1(P09_C1+IN脚下方偏差ブレーキ)×scale で「高CAGR×DD抑制両立」を検証したが、**同MaxDD帯で P09×低スケールが両軸(CAGR・MaxDD)で同等以上＝被支配**。**B1×2.0(+24.07%/−47.08%) ≈ P09×1.4(+24.34%/−46.48%)**で、ブレーキは実効的に「低スケール化装置」。**レバ拡張は B1 でなく P09 で実施すること**。詳細＝[B1_SCALE_FRONTIER_20260621.md](B1_SCALE_FRONTIER_20260621.md)、STRATEGY_REGISTRY §3.4。関連: A7 IN脚ボラブレーキ／A0配分A1-A6 も全て「時機でなくデレバ・ノイズ」で不採用＝**DD削減はレバ水準ダイヤル（B3c/P09低スケール）が本質**。

### ⚙ 運用ステータス（重要・「ベスト戦略」≠「本番稼働中」）

- **本番Active（スプレッドシート稼働中）= E4 RegimeKLT**（下記 §アーカイブ）。**P09 は GAS（NASDAQ-strategy-gas リポ）で並走運用中**。
- B3a_k365 を「ベスト戦略」と確定したのは**分析上の結論**。本番切替は別判断で、**E4→P09 切替判断は7月上旬**予定。B3a/P09_C1/B3c への切替は GAS の BOOST_MAP/STRATEGY CONFIG 変更で追従可能。
- 改善余地（検証済 2026-06-15／QC是正 06-16）: 成功要素グラフト探索（[MULTISTRATEGY_COMBINE_RESULTS_20260615.md](MULTISTRATEGY_COMBINE_RESULTS_20260615.md)）で5要素を B3a に重畳評価。**結論: B3aに追加で効く成功要素は無し**。当初 G5_vix_hard（vix defensiveオーバーレイ）を改善候補としたが、**3エージェントQC（[QCサインオフ](MULTISTRATEGY_COMBINE_QC_SIGNOFF_20260616.md)）でMaxDD改善の66%は単なる一律デレバ・vix時機は非有意(p=0.40)と判明し撤回**。**リスク低減（MaxDD縮小）は exotic オーバーレイでなく B3c_k365（uniform 1.10・BAL総合8.361で全候補最高・低取引）で行うのが最良**。vol-target/レジーム連動/bondOFF→Gold/LT2移植も B3a 上で効果なし。**risk-returnはレバ水準（B3a↔B3c）で制御するのが本質**。

---

## 🆕 v4.5 (2026-06-05) ─ 環境別 Active 候補 + 保守的採用基準導入 / v4.9 (2026-06-08) ─ ルール簡素化

### 保守的採用基準 min(IS, OOS) CAGR (現行 v4.9 確定ルール)
v4.5 (2026-06-05) で **min(IS, OOS) CAGR** を保守的期待リターン指標として導入、**v4.9 (2026-06-08) で標準化確定**。
- ✅ **min(IS, OOS) CAGR の標準化**: IS と OOS の低い方を保守的期待リターンとして使用、OOS 単独評価より優先
- ✅ WFE 補助判定: > 1.5 で regime luck 警告
- ❌ **削除 (v4.9)**: 当初の「3 軸 (min + Worst10Y + P10_5Y) すべて baseline 以上」必須条件は過度に restrictive と判断され撤回。Worst10Y / P10_5Y は §3.12 9 指標として参照するが強制条件ではない。総合判断はユーザー裁量
- 詳細: [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md §7-2](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md), [EVALUATION_STANDARD.md §3.13](EVALUATION_STANDARD.md)

### v7 推奨ベスト候補（2026-06-11）— 標準10指標

> **v7 のベスト候補は ETF（TQQQ）/投信 中心。** CFDからは E4・vz065_l7 を §アーカイブへ退避し、**vz065_l5 のみ CFD候補として残置**（ユーザー判断 2026-06-11）。退避理由は **realistic CFDコスト是正（full-L×SBI CFD金利）で CFDのCAGR優位が ~20%→16%台に縮小し、対して MaxDD −59% でリスク・リワードが不均衡**だから（「CFD回避」ではなく「優位消失」）。**現行本番 Active の E4 は実運用継続中**＝切替はユーザー判断。
>
> **⚠ 税制（重要・2026-06-11訂正）**: **TQQQ・UGL(金2x)・TMF(債3x) はレバレッジ型のため NISA成長投資枠の対象外**（高レバ＝信用類似で除外）。CFDも対象外。→ **CFDもETF(TQQQ系)も等しく特定口座で20.315%課税**。tax はCFD vs ETF の差別化要因ではない（差はコスト構造 TQQQ~9%/yr vs CFD~20%/yr financing と MaxDD のみ）。全行 CAGR⓽ は ×0.8273 後＝課税後で正しい。NISA可能性があるのは P7 のOUT期1倍投信スリーブのみ。
>
> **全値⓽税後（手取り）**。CAGR は IS/OOS 両記載、**min(IS,OOS)** = 保守的採用基準値。
>
> **コスト前提 (v7)**: NASDAQ脚 = TQQQ ETF（2×SOFR+swap0.5%+TER0.86%≈9.1%/yr@3x）。点指標（CAGR/Sharpe/MaxDD/W10Y★/P10）も WFA（WFEⓞ/CI95ⓡ_lo）も **TQQQ基準・正典窓(49窓)** で統一再計算済み（v6 の † 仮置きを解除）。CAGR⓽ = pretax×0.8273。Sharpeⓒ/MaxDDⓒ は税前。**LU1/LU2 は実効>3×期を CFD金利(SOFR+3.0%)で正しく加重控除済**。**全戦略レバ型でNISA対象外＝特定口座20.315%課税前提**（CAGR⓽=課税後で正しい。「NISA非課税で税前=手取り」は本戦略群には不適用）。

| 環境 | 戦略 | **CAGR⓽ IS / OOS（min）** | IS-OOS gap⓽ | Sharpeⓒ | MaxDDⓒ | Worst10Y★⓽ | P10⓽ 5Y | Tradeⓞ/yr | WFEⓞ | CI95ⓡ_lo | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **ETF基準（CAGR死守）** | **V7-TQQQ**<br>DH-W1 + mom63 V7 boost<br>{1.2,1.1,1.0,1.0}, TQQQ | IS +16.27% / OOS +16.80%（**min +16.27%**） | −0.53pp | 0.877 | −34.47% | +10.08% | +5.15% | 25.2 | ✅ 0.976 | **+15.48%** | 🟡 ETF基準候補（TQQQ基準WFA再計算済・†解除） |
| **ETF Active候補🟢（2026-06-11昇格）** | **P09_TQQQ**<br>V7-TQQQ + OUT期 Gold常時+Bond(mom252>0)逆ボラ充填 | IS +18.84% / OOS +17.51%（**min +17.51%**） | +1.32pp | **0.901** | −35.18% | +11.45% | +6.56% | 29.2 | ✅ 1.017 | **+17.94%** | 🟢 **Active候補（拡張評価6次元スコアカード総合8.12/10で首位級・ユーザー承認）**。WFA α/β+permutation+CPCV+Regime+5y/10yテール優位。対base平均bootstrapは非有意だが頑健性で採用。最悪暦年2015 −18.7% |
| **ETF攻め②（強boost）** | **LU1**<br>P09_TQQQ + 強boost {1.4,1.2,1.05,1.0} | IS +19.39% / OOS +18.05%（**min +18.05%**） | +1.34pp | 0.900 | **−34.95%** | +12.27% | +6.82% | 35.2 | ✅ 1.012 | **+18.56%** | 🟠 攻めレバ変種・**CFD格下げ方針と緊張関係**（保有日**29%が実効>3×→CFD/信用要**・超過分CFD金利SOFR+3.0%計上済。増分はskillでなくレバ。主表残置は >3×=29%<LU2の66% ∧ WFE1.012>1.0 のため） |
| **投信環境 (NISA等)・中庸推奨⭐** | **DH-W1 P7** GOLD75/BOND25スリーブ | IS +18.89% / OOS +16.92%（**min +16.92%**） | +1.97pp | 0.861 | −48.10% | +11.41% | +6.38% | 17.6 | ✅ 1.042 | **+17.85%** | 🟢 投信環境 中庸推奨⭐（TQQQ基準WFA再計算済・†解除） |
| **CFD候補（課税環境・唯一）** | **vz=0.65+l5 (vz065_l5)** | IS +16.84% / OOS +20.85%（**min +16.84%**） | −4.01pp | 0.769 | −59.08% | +6.55% | +2.42% | 84.9 | 1.348 | +16.27% | 🔵 CFD候補（ユーザー判断2026-06-11で残置）。⚠ **拡張評価で脆弱性露出**: CPCV p10 +6.67%(最下位)・**Regime_min −14.89%(bear最悪)**・全ストレス窓マイナス・WFE1.35(regime luck)。min単一splitは見栄え良いが頑健性は最低（[EVALUATION_UPGRADE_RESULTS_20260611.md](EVALUATION_UPGRADE_RESULTS_20260611.md)）。MaxDD−59%・Trades85/年。値はCFD基準・正典窓49窓 |

**↳ 攻め③（別枠・要検証）— レバ駆動・CFD必須・WFE<1.0 のため主候補表から分離**

| 戦略 | CAGR⓽ IS / OOS（min） | gap⓽ | Sharpeⓒ | MaxDDⓒ | W10Y★⓽ | P10⓽5Y | Trd/yr | WFEⓞ | CI95ⓡ_lo | 留保事項 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **LU2** uniform IN ×1.15 | IS +20.63% / OOS +18.83%（**min +18.83%**） | +1.80pp | 0.872 | −38.68% | +12.37% | +6.70% | 29.2 | **0.988** | +19.74% | ⚠ **保有日66%が実効>3×＝CFD/信用必須**（超過分SOFR+3.0%計上済）。**WFE 0.988<1.0**（postIS汎化やや弱）。CAGR優位は形でなく一律レバ増。CFD格下げ方針と整合させるなら本枠は採用慎重 |

> **v7.2 注記 (2026-06-11) ─ 最終確定評価レポート**
> P09_TQQQ Active候補昇格の評価一式（標準10指標・ローリング5y/10y分布・6次元スコアカード P09 8.12 ≈ LU1 8.09 > V7 6.93 ≫ vz065_l5 3.57ベト失格・年次リターン表 LU1/P09/NASDAQ1X B&H 1974–2026）は **[HORIZON_AND_SCORECARD_20260611.md](HORIZON_AND_SCORECARD_20260611.md) を最終確定レポート**とする。拡張評価（CPCV/レジーム層別/ストレス窓/bootstrap）の一次根拠は [EVALUATION_UPGRADE_RESULTS_20260611.md](EVALUATION_UPGRADE_RESULTS_20260611.md)。
>
> **v7 注記 (2026-06-11) ─ ベスト候補の再構成（CFD退避＋TQQQ基準WFA確定＋攻め候補追加）**
> - **CFD退避**: E4 RegimeKLT・vz065_l7 を §アーカイブへ移動。**vz065_l5 は CFD候補として残置**（ユーザー判断 2026-06-11、CFDを選択肢に残すため）。退避根拠は「TQQQ是正でCFDが劣後」**ではない**（是正はETF脚のみでCFD数値は不変）。正しい理由は **NISA非課税かつMaxDD/Sharpe重視という選好の下では、20.3%課税＋MaxDD−65%(E4)/−68%(l7)＋Sharpe0.678(E4) のCFDは非選好**。E4は本番Active継続中＝実運用切替はユーザー判断。
> - **TQQQ基準WFA確定（† 解除）**: v6 で仮置きだった WFEⓞ/CI95ⓡ_lo 列を TQQQ基準・正典窓(49窓)で再計算。CFD分枝が旧†値を **dWFE≤0.0002 / dCI95≤0.005pp で再現**（harness健全）。TQQQ基準では CI95_lo が +1.0〜1.3pp 上昇（DH-W1 +13.69→+14.99 / V0 +12.57→+13.55 / V7 +14.09→**+15.48** / P7 +16.57→**+17.85**）、WFE は比率のため不変。全行 β(WFE∈[0.5,2.0])・α(CI95>0&t_p<0.05) PASS。一次根拠 `audit_results/tqqq_wfa_recompute_20260611.csv`。
> - **攻め候補追加**: P09_TQQQ（min **+17.51%**, +1.24pp/DD許容）・LU1（min **+18.05%**, 強boost・29%>3×）を主表に、LU2 を別枠に。P09_TQQQ は WFA α/β＋permutation PASS だが **対baseline bootstrap非有意（P=0.80）** で Active昇格保留（最悪暦年2015 −18.7%、gold+bond同時安）。
> - **指標検証**: 標準10指標の列（CAGR_IS/OOS⓽・gap⓽・Sharpeⓒ・MaxDDⓒ・W10Y★⓽§3.5・P10⓽5Y▷§3.6・Tradesⓞ・WFEⓞ・CI95ⓡ_lo）を `compute_10metrics`（§3.12正典）と突き合わせ全件整合を確認済。LU1/LU2 は >3×超過分を CFD金利で加重控除した CFD-recost 値を採用（一次根拠 `audit_results/lu_cfd_recost_20260611.csv`）。
> - **税制（2026-06-11訂正）**: TQQQ/UGL/TMF はレバ型でNISA対象外、CFDも対象外 → CFD・ETF共に特定口座20.315%課税。**tax はCFD vs ETF の差別化要因ではない**（差はコスト構造とMaxDD）。よって採否は **CAGR⓽（課税後）と MaxDD/Sharpe の素のバランス**で見る。P7のOUT期1倍投信のみNISA可能性あり。
>
> **旧 v6/v5 注記（履歴）**: ETF脚 CFD→TQQQ コスト是正（点指標 +0.84〜1.20pp、ランキング不変 P7>V7>DH-W1>V0、CFD版0.00pp再現）と、realistic full L×・正典窓49窓基準の確立。詳細は git log（v5/v6 コミット）参照。

**P09_TQQQ** は STRATEGY_REGISTRY §2 に **Active候補（2026-06-11昇格・ユーザー承認）** 登録済み（評価一式は [HORIZON_AND_SCORECARD_20260611.md](HORIZON_AND_SCORECARD_20260611.md)）。**V7-TQQQ / LU1 / LU2 の §2 登録は次工程**（本表が一次根拠）。**本番運用切替（E4→P09、GAS通知システム）は実装計画進行中**。退避した CFD戦略は下記 §アーカイブに全指標を保存。

### 📦 アーカイブ（2026-06-11 v7 で主候補から退避・指標は保存）

> 以下は**削除ではなく退避**。判断材料・履歴・将来の再評価のために全指標を保持する。値は退避時点（CFD戦略=CFD基準・正典窓49窓 realistic full L×、ETF戦略=TQQQ基準）。

| 退避戦略 | 環境 | CAGR⓽ IS / OOS（min） | gap⓽ | Sharpeⓒ | MaxDDⓒ | W10Y★⓽ | P10⓽5Y | Trd/yr | WFEⓞ | CI95ⓡ_lo | 退避理由 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **E4 RegimeKLT**（**本番Active継続中**） | CFD | IS +16.93% / OOS +18.06%（min +16.93%） | −1.13pp | 0.678 | −65.05% | +5.82% | −0.68% | 27.1 | 1.094 | +15.63% | realistic CFDコスト是正でCAGR優位消失（min+16.9%≈ETF）なのに **MaxDD−65%・Sharpe0.678** ＝リスク・リワード不均衡。**実運用は継続中**＝切替はユーザー判断 |
| vz=0.65+l7 (vz065_l7) | CFD | IS +16.81% / OOS +22.47%（min +16.81%） | −5.66pp | 0.764 | −68.44% | +6.27% | −1.64% | 104.7 | 1.328 | +16.45% | 同上＋MaxDD−68%・Trades105/年でリスク最大 |
| DH-W1 (Asymm Hyst) | ETF | IS +15.85% / OOS +16.60%（min +15.85%） | −0.75pp | 0.883 | −34.44% | +9.54% | +4.79% | 17.6 | 0.996 | +14.99% | overlay版 V7-TQQQ が min/CI95 で上回り基準として劣後（素のDH-W1は冗長） |
| DH-W1 + mom63 V0 def {1.1,1.0,0.9,0.8} | ETF | IS +14.27% / OOS +15.41%（min +14.27%） | −1.14pp | 0.912 | −28.71% | +9.49% | +4.81% | 31.2 | 1.036 | +13.55% | MaxDD最小(−28.7%)・Sharpe最高(0.912)だがユーザー選好（CAGR優先・DD許容）に不一致 |

- **E4 注**: ⚠ E4 は依然として **§1 本番 Active（実運用スプレッドシート稼働中）**。本アーカイブは「v7 のベスト候補比較表から外す」意味であり、**実運用を止めた訳ではない**。NISA前提のベスト候補へ切替えるかはユーザー判断。
- **V0 注**: MaxDD −28.71%・Sharpe 0.912 は全候補中最良。**守り最優先**ならアーカイブから復帰可（CAGR は最低水準 min +14.27%）。

### 棄却された v4.x 改善案 (min ルール下で REF を下回る)
- **vz=0.65+l7+F10ε-AH/AT/HL** (v4.4 採用→v4.5 棄却) — OOS 単独では魅力的だが min(IS, OOS) で REF 劣後、WFE>1.5 で regime luck 疑い (STRATEGY_REGISTRY §3 Rejected)
- **DH-T4** (v3 で push→v4 で破棄) — ETF レバ操作違反 (lev_mod 連続 scaling)
- **DH-Z2** (v4 採用→v4.3 で W1 に置換) — Trades 152→18 で W1 が優位

### v4.5 整理対象ファイル
- [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) v4.5 (§7-2 min ルール明文化、§0' 5 戦略 + min 表記、§6-4 AH 棄却根拠)
- [STRATEGY_DH_REFINEMENT_20260603.md](STRATEGY_DH_REFINEMENT_20260603.md) v1.1 (DH-W1 詳細検証)
- [STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md) (§2 に v4.5 推奨 2 件、§3 に AH/AT/HL 棄却 3 件)

---

## 投信環境 Active 候補 (2026-06-07 追加) ─ DH-W1 キャッシュ・スリーブ 1倍投信置換

> **位置付け**: ETF 環境 Active 候補 **DH-W1** は全営業日の **46.9%(6,171日)をキャッシュ 0% で待機**する。この待機資金を 1 倍投信(ゴールド/米国債20年超)で運用置換した派生系。1 倍投信は **SOFR/スワップなし・信託報酬<0.2%・5営業日ラグ**で実装可能（レバ脚は DELAY=2 / SOFR/スワップ込み、税 20.315%×0.8273）。§1 本番 Active (CFD: E4 Regime k_lt) の置換ではなく、**ETF/投信のみ環境(NISA 等)で DH-W1 を更に押し上げる候補**。

### OUT(キャッシュ)期 6,171日の各1倍資産の素の挙動
| 資産 | 年率リターン | 年率Vol | R/R | 含意 |
|---|---:|---:|---:|---|
| NASDAQ 1x | **−7.63%** ⚠ | 26.3% | −0.29 | キャッシュ期＝risk-off、NASDAQ は大損 → スリーブ不適 |
| Gold 1x | +4.07% | 21.5% | +0.19 | 堅実 |
| Bond 1x (米国債22yr) | **+6.51%** ✅ | **11.9%** | **+0.55** | 低ボラ＋プラスで最良 |

### 4戦略 標準指標（全コスト後・税20.315%後・WFA 50窓）

> ⚠ **DH-W1 baseline 表記注意**: 本表の DH-W1 値 (+13.66%) は **v4.5 §7-2 min(IS,OOS) CAGR = IS 値**（旧 OOS split 基準）。
> 上記 v4.5 推奨表の canonical OOS +18.91% とは **split 日付と計算基準が異なる**。各 Cash Sleeve 戦略の差分（+2.78pp 等）はこの +13.66% ベースの**相対比較値**として正しい。

| 戦略 | CAGR_OOS | Sharpe_OOS | MaxDD | Worst10Y★ | P10_5Y | IS-OOS gap | Trades/yr | WFE | CI95_lo | 性格 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DH-W1 (baseline) | +13.66%† | +0.844 | **−34.57%** | +8.40% | +5.11% | +1.62 | 17.8 | 0.997 | +13.95% | 現状(キャッシュ)† |
| 🟢 P2 GOLD100 | **+16.44%** | **+0.875** | −58.53% | +9.43% | +8.06% | **+0.97** | 18.8 | 1.229 | +16.04% | 攻め (OOS 最高) |
| 🟢 **P7 GOLD75/BOND25** ⭐ | +14.90% | +0.827 | −48.23% | +9.92% | +8.05% | +3.28 | 18.8 | 1.043 | +16.74% | **中庸推奨** |
| 🟢 P5 GOLD50/BOND50 | +13.28% | +0.758 | −35.97% | +10.08% | +8.09% | +5.50 | 18.8 | 0.875 | **+17.23%** | 守り (DD 最良) |

- **P2 GOLD100**: ベース比 +2.78pp の OOS 最高・最汎化(gap +0.97)。代償は MaxDD −58.5%(Gold 単独の宿命)。攻め優先向け。
- **P7 GOLD75/BOND25 ⭐(新規・推奨)**: P2 と P5 の中間最適。OOS +14.90% を確保しつつ Bond 混で MaxDD を −48.23% に緩和。総合バランス最良。
- **P5 GOLD50/BOND50**: MaxDD をベース同等(−36.0%)維持・CI95_lo/P10/Worst10Y 最高。守り最優先向け。gap +5.50pp(汎化やや弱)。

> **昇格保留の理由**: WFA 50窓は CI95_lo>0 (α) ∩ 0.5≤WFE≤2.0 (β) を全 PASS だが、**t_p(permutation 検定)・block bootstrap は未実施**。正式 Active 昇格にはこれらの統計検証が必要（[tasks.md](tasks.md) Pending）。全コスト表・執行ラグ検証は [CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md) §4-§6 参照。STRATEGY_REGISTRY §2 に 3 件登録済。

---

## Shortlisted（次善候補 / WFA 完了）

> CAGR 数値は SBI CFD 3.0%/yr ベースで未計算。詳細指標は [STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md) §2 参照。

| 戦略 | Tradeⓞ/yr | WFEⓞ | 棄却理由 |
|---|---:|---:|---|
| F8 R5_CALM_BOOST | 182 | ✅ 1.208 | Trades/yr E4 比 7 倍。OOS 偶然性疑い |
| F7v3+E4 A:tilt=2.0 | 183 | ✅ 1.203 | 同上 |
| LT2-N750 固定k=0.5 | 27 | ✅ 1.145 | E4 に Worst10Y★/IS-OOS gap/CAGR で劣後。fallback 候補 |

---

## 一次根拠ファイル（GitHub 上の正典）

| ファイル | 役割 |
|---|---|
| [HORIZON_AND_SCORECARD_20260611.md](HORIZON_AND_SCORECARD_20260611.md) | **P09_TQQQ Active候補昇格の最終確定レポート**（標準10指標・5y/10y分布・6次元スコアカード・年次リターン表） |
| [EVALUATION_UPGRADE_RESULTS_20260611.md](EVALUATION_UPGRADE_RESULTS_20260611.md) | 拡張評価一次根拠（CPCV45折・レジーム層別・ストレス窓・bootstrap） |
| [7STRATEGY_PERFORMANCE_REPORT_20260529.md](7STRATEGY_PERFORMANCE_REPORT_20260529.md) | **E4 指標の一次根拠**（SBI CFD 3.0%/yr ベース、CAGR⓽ OOS+22.4%/min+20.0%） |
| [g14_wfa_sbi_cfd_summary.csv](g14_wfa_sbi_cfd_summary.csv) | g14 WFA SBI CFD サマリ（CI95_lo⓽=+16.3%、WFE=1.15） |
| [src/g14_wfa_sbi_cfd.py](src/g14_wfa_sbi_cfd.py) | g14 WFA 実行スクリプト（SBI_CFD_SPREAD=3.0%/yr） |
| [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md) | E4 パラメータ sweep（構造的採用根拠・64 config PASS 12） |
| [src/e4_regime_klt.py](src/e4_regime_klt.py) | E4 実装 |
| [src/product_costs.py](src/product_costs.py) | コスト定数の単一の真実 |
| [analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md) | 投信環境コスト・執行ラグ・P2/P7/P5 検証 |

---

## 「ベスト戦略は？」と問われたときの参照プロトコル (Claude 必読)

### 手順

1. **本ファイル (`CURRENT_BEST_STRATEGY.md`) の冒頭ブロックを引用する** — 最優先・最新の真実
2. `tasks.md` の最新 ✅ Completed エントリと突き合わせて整合確認
3. 矛盾があれば必ずユーザーに報告 → 本ファイル更新の提案

### 絶対にやってはいけないこと

- ❌ CSV を Sharpe 降順で並べて「トップ」を答える
- ❌ F7v3/F8 系 (高 Trades/yr) を「Sharpe が高いから」とベストとして提示する
- ❌ MEMORY.md 内の固定記述を一次根拠にする

---

## ⛔ 廃止された旧推奨 (ブラックリスト)

| 旧推奨 | 廃止日 | 廃止理由 |
|---|---|---|
| `F8 R5_CALM_BOOST` Sharpe +0.934, Trades/yr 182 | 2026-05-24 | Trades/yr 182回 (E4比7倍) のコスト負担と OOS 偶然性疑いによりShortlisted降格 |
| `F7v3+E4 A:tilt=2.0` Sharpe +0.926, Trades/yr 183 | 2026-05-24 | 同上。tilt系は全期間でE4との改善幅が軽微、IS-OOS gap拡大 |
| `S2_VZGated + LT2-N750-k0.5-modeB (固定k)` Sharpe 0.858 | 2026-05-24 | E4 が CAGR/Sharpe/Worst10Y★/IS-OOS gap で優位 |
| `S2_VZGated` CAGR_OOS +27.57%, Sharpe_OOS 0.769 | 2026-05-21 | S2+LT2 が全指標で上回ることを確認 |
| `DH Dyn 2x3x [A]` CAGR 22.50%, Sharpe 0.993 | 2026-05-21 | S2_VZGated が上回ることを確認 |
| `Ens2(Asym+Slope)` CAGR 28.58%, Sharpe 1.031 | 2026-04-21 | `DH Dyn 2x3x [A]` に置換 |

---

## 命名規則 (今後の再発防止)

1. **`FINAL_` プレフィックスは使用禁止**
2. **`<TOPIC>_YYYY-MM-DD.md` 形式を使用**
3. **新レポートが旧レポートを置き換える時は、必ず旧レポート冒頭に SUPERSEDED ヘッダを追加**
4. **本ファイル (`CURRENT_BEST_STRATEGY.md`) を必ず同時更新**

---

## メタ情報

変更履歴は git log で追跡可能 (`git log --follow CURRENT_BEST_STRATEGY.md`)

### 変更履歴
- **2026-06-21 (P09レバ拡張＝前向き採用検討 / B1×scale＝不採用)**: ユーザー方針確定により §🟠「P09 レバ拡張フロンティア」を新設（B3aレバ水準ダイヤル直後）。**P09_C1 strong-map×scale{1.4,1.6,1.8,2.0}＝高CAGR側ダイヤル（CAGR_OOS +24.34%→+29.11%、税後）を前向きに採用検討**。−50%以内・過学習なしの実質上限は **scale≈1.4（MaxDD−46.48%・regime−5.86%・WFE0.936）**。高scale(1.8/2.0)は IS-OOSギャップ>5pp過学習＋regime_min<−10%(2.0)ハードベトで非推奨。一方 **B1×Scale（B1下方偏差ブレーキ×レバ拡張）は不採用**＝同MaxDD帯で P09×低スケールが両軸で同等以上（B1×2.0≈P09×1.4）・ブレーキは実効「低スケール化装置」。関連の A7 IN脚ボラブレーキ／A0配分A1-A6 も全て「時機でなくデレバ・ノイズ」で不採用＝**DD削減はレバ水準ダイヤル（B3c/P09低スケール）が本質**を再確認。STRATEGY_REGISTRY §2（P09レバ拡張＝Shortlisted/採用検討中）・§3.4（B1×scale＝Rejected）に登録。一次根拠: [LEVERUP_SCALE_FRONTIER_20260619.md](LEVERUP_SCALE_FRONTIER_20260619.md)、[B1_SCALE_FRONTIER_20260621.md](B1_SCALE_FRONTIER_20260621.md)、[A7_DD_REDUCTION_VARIATIONS_20260621.md](A7_DD_REDUCTION_VARIATIONS_20260621.md)。§1 本番Active(E4)・§🏆ベスト戦略(B3a_k365)は変更なし。
- **2026-06-15 (v8: ベスト戦略確定)**: ユーザー決定により **ベスト戦略=B3a_k365（CAGR重視）／ベスト戦略候補=P09_C1（バランス重視）** を冒頭 §🏆 に確定記載。B3a_k365 = DH-W1+V7マップ{Q0:1.40,Q1:1.40,Q2:1.05,Q3:1.00}×uniform1.15＋P09 OUT充填＋C1（bondOFF日SOFR）、コスト ≤3x=TQQQ/>3x=くりっく株365。min⓽ +20.98%（選択バイアス割引後≈20.1%）・MaxDD−38.2%・WFA CI95_lo+22.52%・全ゲートPASS。P09_C1 = V7-TQQQ＋P09充填＋C1（min⓽ +17.77%・6次元バランス採点首位8.50・2008防御最強・CFD依存最小）。一次根拠 [LEVERUP_SWEEP_RESULTS_20260612.md](LEVERUP_SWEEP_RESULTS_20260612.md)、QC [LEVERUP_QC_SIGNOFF_20260613.md](LEVERUP_QC_SIGNOFF_20260613.md)（3エージェント＋独立再実装で§6.1値80セル中79再現）。**本番稼働は E4 継続・P09 GAS並走・E4→P09切替判断は7月上旬**（B3a/P09への本番切替は別判断）。
- **2026-06-11 (v7: ベスト候補再構成)**: ユーザー判断により CFD戦略 **E4 RegimeKLT・vz065_l7** と ETF の素 DH-W1・V0 を **§アーカイブへ退避（削除せず全指標保存）**。**vz065_l5（レバ5倍・CFD中で最バランス）は唯一のCFD候補として残置**（評価指標変更でCFDが再評価される可能性に備える）。主候補を **V7-TQQQ（基準）/ P09_TQQQ（攻め・DD許容）/ LU1（攻め・強boost）/ P7（投信中庸⭐）/ vz065_l5（CFD）** に再編、**LU2** は CFD必須・WFE<1.0 のため別枠。v6 で † 仮置きだった WFA（WFEⓞ/CI95ⓡ_lo）を **TQQQ基準・正典窓49窓で再計算し確定**（CFD分枝が旧†値を dWFE≤0.0002/dCI95≤0.005pp で再現、harness `src/audit/tqqq_wfa_recompute_20260611.py`）。CI95_lo: V7 +14.09→+15.48% / P7 +16.57→+17.85% 等に上方更新、† 解除。標準10指標列を `compute_10metrics`（§3.12正典）と全件整合確認。LU1/LU2 は実効>3×期を CFD金利(SOFR+3.0%)で加重控除（`lu_cfd_recost_20260611.csv`）。**E4 は本番Active継続中**＝実運用切替はユーザー判断。退避理由の正典化: 「TQQQ是正でCFD劣後」ではなく「NISA非課税・DD/Sharpe重視の選好下でCFDが非選好」。
- **2026-06-07 (DH-W1 Cash-Sleeve 4戦略)**: ETF 環境 Active 候補 DH-W1 の OUT(キャッシュ 46.9%)期を 1 倍投信で運用置換する 4 戦略を **「投信環境 Active 候補」セクション** として §Shortlisted 直前に新設。**P2 GOLD100**(攻め, OOS +16.44% 最高)/**P7 GOLD75/BOND25 ⭐**(中庸推奨, MaxDD −48.23%)/**P5 GOLD50/BOND50**(守り, MaxDD −35.97% 最良)。全 WFA 50窓 α∩β PASS。検証済み: OUT 期資産挙動、全商品コスト表(SOFR/TER/スワップ/売買/税 20.315%)、執行ラグ(レバ脚 DELAY=2 / 投信スリーブ 5BD)。t_p/bootstrap 未実施で正式昇格保留。§1 本番 Active(E4 Regime k_lt)は変更なし。STRATEGY_REGISTRY §2 に 3 件登録。一次根拠: [CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md)。
- **2026-06-05 (v4.8: Session 4 + 5)**: `nasdaq_mom63 × S3 × M6 defensive` overlay を **Risk-Reduction Overlay Candidate** として §Shortlisted 直前に新設。Session 4 で S3 (DH-W1, ETF only) に対し Phase D 4 gate 全 PASS → ADOPT。Session 5 で S2 (D5) / E4 (現行 Active) への転用を audit-grade で検証 → 両方 NEEDS_FURTHER_WORK (WFE<1.0、ハードゲート不通過。P(MaxDD better)>0.94 で方向性は一貫)。結論: overlay は **S3 限定 (strategy-specific)**。§1 本番 Active (CFD: E4 Regime k_lt) は変更なし。
- **2026-06-05 (v4.7: CFD Active 候補を l7 → l5 に置換)**: ユーザー判断により vz=0.65+l5+F10ε を CFD 環境 Active 候補に確定。l7 (旧 REF) は副候補に降格。理由: l5 は min CAGR -1.30pp の trade-off と引換に **Worst10Y +12.67% (vs l7 +9.96%、+2.71pp)、P10_5Y +8.75% (vs l7 +4.05%、+4.70pp 大幅改善)、MaxDD -56.72% (vs l7 -65.95%、9.23pp 浅化)、Sharpe +0.841 (vs l7 +0.829)、Trades 86 (vs l7 105、18% 低コスト)** で防御指標が圧倒的優位。
- **2026-06-05 (v4.6: lmax sweep)**: vz=0.65+F10ε の lmax を l5/l5.5/l7 で比較、l5.5/l5 を Shortlisted 追加。
- **2026-06-05 (v4.5: 保守的採用基準 + 環境別 Active 候補導入)**: 本ファイル冒頭に **「v4.5 環境別 Active 候補」セクション** + **「命名規則」セクション** を新設。要点:
  - **min(IS, OOS) CAGR + Worst10Y + P10_5Y の 3 軸保守的尺度** を Active 昇格判断の必須条件として導入 (詳細: STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md §7-2)
  - **CFD 環境 Active 候補**: vz=0.65+l7+F10ε (min CAGR=+20.23%、5 戦略中 1 位)
  - **ETF 環境 Active 候補**: DH-W1 (DH Asymm+Hysteresis、ETF 制約下で DH 基線を +4.10pp 上回る唯一の改善)
  - 棄却: vz=0.65+l7+F10ε-AH/AT/HL (OOS のみ評価では魅力的だが min ルールで全敗、WFE>1.5 で regime luck 疑い)
  - 「vz=0.65+l7+F10ε」を「NEW/NEW CANDIDATE」と呼ぶ表記を廃止
  - §1 正式 Active は **未変更** (E4 RegimeKLT を維持)。CFD Active への昇格判断はユーザー承認後
- 2026-05-24 (tilt系棄却・E4 復帰): F7v3+E4 および F8-R5 を Trades/yr 過多（182〜183回/年、E4比7倍）・OOS偶然性疑い・IS-OOS gap拡大（−4.26〜4.28pp）を理由に Shortlisted 降格。E4 Regime k_lt (27回/年) を正式 Active に復帰。棄却判断: コスト加味の実質 CAGR は E4 と同等以下と判断。
- 2026-05-24 (G5 WFA + F8-R5 昇格→即降格): F8-R5 WFA PASS (CI95_lo=+27.92%) を確認するも上記理由で採用見送り。
- 2026-05-24 (G4 WFA + F7v3+E4 昇格→即降格): F7v3+E4 WFA PASS (CI95_lo=+27.15%) を確認するも上記理由で採用見送り。
- 2026-05-24 (G3 WFA): E4 Regime k_lt WFA PASS (CI95_lo=+26.51%, WFE=+1.131)。正式 Active 確定。
- 2026-05-24: E4 Regime k_lt 暫定昇格 (k_lo=0.1, k_hi=0.8, vz_thr=0.7)
- 2026-05-22: B6 N=1500 暫定昇格・即差し戻し
- 2026-05-21: B1 S2+LT2 採用
- 2026-05-12: Scenario D 補正適用
- 2026-05-11: 初版作成

---

*管理者: 男座員也（Kazuya Oza）*
