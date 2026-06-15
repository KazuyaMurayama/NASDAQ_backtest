# 成功要素グラフト × 複数戦略検証 結果（B3a土台・両面探索）

作成日: 2026-06-15
最終更新日: 2026-06-15

> **計画**: [MULTISTRATEGY_COMBINE_PLAN_20260615.md](MULTISTRATEGY_COMBINE_PLAN_20260615.md)。ベスト戦略 [B3a_k365](CURRENT_BEST_STRATEGY.md) を固定土台に、他戦略で成功実績のある5要素を native 統合し、防御（MaxDD/2008/Sharpe是正）と攻め（CAGR）の両面で改善を探索。
> 評価規律: ハードベト＋WFA正典49窓＋CPCV＋レジーム層別＋multi-metric bootstrap＋6次元採点。[LESSONS_LEARNED_20260607.md](LESSONS_LEARNED_20260607.md) 5教訓厳守。実行=Sonnet、計画・採点・統合=Opus。
> 全要素にサニティアンカー（要素無効化時にB3a素地 min⓽+20.98%・MaxDD−38.20% を±0.05pp再現）を課し全PASS。

---

## 1. 結論（要点）

1. **探索した5要素のうち、改善レバーは G5（vix_mom21 防御オーバーレイ）1つのみ**。残り4要素（G1 vol-target / G2 レジーム連動 / G3 bondOFF→Gold / G4 LT2移植）と組合せ（G5+G4）は B3a 上で改善を出さず。**B3aが既に強く、過剰な追加要素は効かない**ことが確認された（正直な負の結果も成果）。
2. **G5_vix_hard = B3aの最大弱点を統計的有意に是正するリスク低減レバー**: B3aに vix_mom21 の defensive マップ`{Q0:1.00,Q1:1.00,Q2:0.92,Q3:0.85}`を native 重畳 → **MaxDD −38.20%→−35.93%（+2.27pp・bootstrap P=0.96(block21)/0.88(block252)＝有意）**、**Sharpe 0.904→0.928（+0.024）**、CAGR劣化は **min⓽ −0.32pp のみ**。>3x日比率も 37.7%→32.8% に低下（CFD依存も軽減）。
3. **ただし6次元採点では G5 は B3a をわずかに下回る**（BAL 8.13 vs 8.27 / CAGR重視 7.00 vs 7.23）。③リスク+0.55・②頑健性+0.21・⑥有意性+0.23 の改善を、**⑤コスト −2.35（取引 33→53/yr の減点）が相殺**したため。**この取引コストは min⓽ に既に課金済み**（コストエンジンがCAGRから控除）なので採点器の取引ペナルティは一部二重計上気味＝採点はG5に対し保守的。
4. → **G5_vix_hard は「B3aを安全側に振るオプションのdefensiveオーバーレイ」**。CAGRを実質維持しつつ MaxDD を有意に縮め Sharpe を上げる。採否はユーザーのリスク選好（MaxDD縮小 vs 取引頻度2倍弱）次第。**B3aを置換する strict dominator ではない**。

---

## 2. Phase 1（Stage 0 スクリーン・標準10指標＋ベト）

| 要素 | 内容 | 方向 | min⓽ | MaxDD | Sharpe | 判定 |
|---|---|---|---:|---:|---:|---|
| **G5_vix_hard** | vix_mom21 defensiveマップ{Q2:0.92,Q3:0.85} | 防御 | +20.66% | **−35.93%** | **0.928** | ✅ 生存 |
| G4 LT2 k020 | LT2-N750 Mode-A k=0.20 | 攻め | +21.40% | −39.89% | 0.865 | △ 通過（要注意） |
| G1 vol-target | 実効レバ vol-target governor (tv25/30/35%) | 防御 | +19.2〜20.6% | ±0 | ↓ | ✕ クローズ（発動率低） |
| G2 レジーム連動 | bear/highvol でレバ縮小 | 防御 | +19.1〜20.6% | ほぼ±0〜−1.8 | — | ✕ クローズ（min⓽劣化過大） |
| G3 bondOFF→Gold | OUT∧bondOFF日をGold一部充填 (g25/50/100%) | 両面 | +20.8〜21.0% | −38.7〜−55.6% | ↓ | ✕ クローズ（CAGR/DD両悪化） |

- **G1**: B3aは実現volが目標vol近傍で推移しガバナーがほぼ発動せず、発動時もCAGRを削るのみ。
- **G2**: bear限定は効果薄、bear∨highvol は2008窓+1.2〜1.8pp/Regime_min改善を出すが min⓽ −1.2〜1.9pp 劣化（基準超）。highvol日が53%を占めブル収益も削る。
- **G3**: bondOFF日の多くが株・Gold同時下落局面のためGold充填でDD悪化。**C1のSOFR計上が正しかったことを裏付け**。

## 3. Phase 2（フルゲート: WFA49＋CPCV＋レジーム＋stress＋multi-metric bootstrap）

サニティ全PASS（B3a/V7 既知値一致）。

| 構成 | min⓽ | WFA CI95_lo | WFE | CPCV p10 | Regime_min | MaxDD | Sharpe | 2008窓 | Trades/yr | ベト |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B3a_k365（基線） | +21.05% | +22.52% | 0.987 | +16.01% | −2.88% | −38.20% | 0.904 | +16.8% | 33.3 | PASS |
| **G5_vix_hard** | +20.73% | +21.54% | 0.990 | +15.64% | −2.10% | **−35.93%** | **0.928** | +18.7% | 52.9 | PASS |
| G4 LT2 k020 | +21.47% | +21.09% | 1.108 | +15.79% | −4.05% | −39.89% | 0.865 | +18.8% | 122.5 | PASS |

**multi-metric bootstrap（対B3a素地・block21／block252）**:
- **G5: MaxDD P=0.959/0.879（有意・≥0.80）**、Sharpe P=0.764、min⓽ P=0.31（非有意=防御trimの当然）、Worst10Y★ P=0.37/0.22（−0.8pp劣化は非有意＝点推定の揺らぎ）。→ **MaxDD改善は本物**。
- G4: 対B3a min⓽ P=0.286（**非有意**）、Sharpe P=0.307、MaxDD悪化。CAGR優位は統計的に確認できず。WFE 1.108（IS-OOS差拡大傾向）・取引122/yr。

→ **G5 = Phase 2 PASS（防御改善が有意）**。G4 = 条件付き（CAGR非有意・⑤⑥③減点・実運用コスト懸念）。

## 4. Phase 3（組合せ B3a+G5+G4）

サニティ両方向PASS。結果: **組合せは上乗せ無し**。B3a+G5+G4 は G5単体比で MaxDD −3.72pp 悪化（−39.65%）・Sharpe −0.041・取引124.5/yr・対G5 bootstrap全軸 P<0.33。**G4のLT2逆張りが、G5が抑えたい高vol局面で逆張り増レバして防御を部分破壊**。→ **G5単体が最良**。

## 5. Phase 4（6次元採点・再現可能採点器）

| 候補 | ①Ret | ②Rob | ③Risk | ④Tail | ⑤Cost | ⑥Sig | 総合BAL | 総合CAGR重視 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V7_TQQQ | 6.40 | 6.64 | 7.38 | 7.15 | 9.00 | 6.00 | 7.00 | 6.00 |
| P09_C1 | 8.79 | 9.40 | 7.86 | 8.82 | 8.42 | 6.75 | **8.50** | 6.87 |
| **B3a_k365** | 10.0 | 8.69 | 7.41 | 9.51 | 4.07 | 7.51 | **8.27** | **7.23** |
| **B3a+G5_vix_hard** | 10.0 | 8.90 | **7.97** | 9.49 | **1.72** | **7.74** | 8.13 | 6.99 |

- B3a→G5 の次元変化: ③リスク **+0.55**↑・②頑健性 +0.22↑・⑥有意性 +0.23↑（いずれも期待方向）／⑤コスト **−2.35**↓（取引33→53）／①④はほぼ不変。
- **G5 総合 < B3a**（BAL −0.14 / CAGR −0.24）。主因は⑤コストの取引ペナルティ。
- **採点の保守性に関する注記**: G5の取引増コストは min⓽ に既に課金済み（コストエンジン）。採点器⑤の Trades ペナルティはその一部を二重に減点。一方 excess_ratio（CFD依存=>3x日比率）は G5 でむしろ改善（37.6%→32.8%）。採点器は取引頻度を保守的に重く見るため G5 に不利に出ている。

## 6. 推奨

- **B3aは best として維持**（6次元採点で G5/組合せいずれも B3a を上回らず）。
- **G5_vix_hard は「リスク低減オプション」として STRATEGY_REGISTRY §2 に登録**: B3aに vix_mom21 defensiveオーバーレイを重ね、**MaxDD有意縮小（+2.27pp）・Sharpe+0.024 を CAGR −0.32pp で得る**。取引が33→53/yrに増える。**MaxDDをP09水準に近づけたい局面の選択肢**（B3aの「対P09 MaxDD −7.5pp悪化」という最大弱点を約2.3pp埋め戻す）。
- **採否はユーザーのリスク選好で決定**: 「名目CAGR最大・取引最小」なら素のB3a／「MaxDDを有意に縮めSharpeを上げ、取引2倍弱を許容」なら B3a+G5。
- 棄却した G1/G2/G3/G4・組合せは再検証不要（本結果が一次根拠）。

## 7. 正直な留保

1. G5の総合スコアがB3aを下回るのは採点器のコスト次元設計（取引ペナルティ）に依存。実損益（CAGR）は既に取引コスト込みで min⓽ −0.32pp のみ。採点を「コスト重視設計」と理解した上で判断すべき。
2. G5の Worst10Y★ −0.8pp 劣化は bootstrap 非有意（点推定の揺らぎ域）。
3. 本探索は B3a 単体土台（計画の選択）。P09_C1 土台への G5 適用は未検証（別途必要なら実施可）。
4. 全要素 native 統合・lookahead 是正済み（G2はvol中央値をIS-onlyで推定）。post-hoc評価なし（Lesson A遵守）。

---

## 成果物一覧

| 種別 | ファイル |
|---|---|
| 計画 | [MULTISTRATEGY_COMBINE_PLAN_20260615.md](MULTISTRATEGY_COMBINE_PLAN_20260615.md) |
| Phase1 G1 | `src/audit/combine_g1_voltarget_20260615.py` / `audit_results/combine_g1_voltarget_20260615.csv` |
| Phase1 G2 | `src/audit/combine_g2_regimelev_20260615.py` / `...csv` |
| Phase1 G3 | `src/audit/combine_g3_bondoffgold_20260615.py` / `...csv` |
| Phase1 G4 | `src/audit/combine_g4_lt2_20260615.py` / `...csv` |
| Phase1 G5 | `src/audit/combine_g5_defoverlay_20260615.py` / `...csv` |
| Phase2 | `src/audit/combine_phase2_fullgate_20260615.py` / `...csv` |
| Phase3 | `src/audit/combine_phase3_g5g4_20260615.py` / `...csv` |
| Phase4採点 | `src/audit/scorecard_g5_20260615.py` / `...csv` |

---

*管理者: 男座員也（Kazuya Oza）*
