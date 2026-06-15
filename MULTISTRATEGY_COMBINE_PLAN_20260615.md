# 成功要素グラフト × 複数戦略検証 計画（B3a土台・両面探索）

作成日: 2026-06-15
最終更新日: 2026-06-15

> **目的**: ベスト戦略 [B3a_k365](CURRENT_BEST_STRATEGY.md) を固定土台に、リポジトリ内の**他戦略で成功実績のある要素**を native 統合で移植し、(防御=MaxDD/2008/Sharpe是正) と (攻め=CAGR押上げ) の**両面**で改善候補を探す。
> 評価規律は [LEVERUP_SWEEP_RESULTS_20260612.md](LEVERUP_SWEEP_RESULTS_20260612.md) と同じ全ゲート（ハードベト＋WFA正典49窓＋CPCV＋レジーム層別＋multi-metric bootstrap＋選択バイアス規律）。[LESSONS_LEARNED_20260607.md](LESSONS_LEARNED_20260607.md) の5教訓を厳守。
> 計画=Opus、実行=Sonnetサブエージェント、採点・統合=Opus。

---

## 0. 前提と規律（着手前チェック）

### 0.1 土台（固定）と比較基線
- **土台**: B3a_k365 = DH-W1（Asymm Hyst, Enter≥0.7/Exit≤0.3）+ V7マップ`{Q0:1.40,Q1:1.40,Q2:1.05,Q3:1.00}`×uniform1.15 + P09 OUT充填（Gold常時＋Bond@`bond_mom252>0`・逆ボラW63・T+5）+ C1（OUT∧bondOFF日にSOFR）+ コスト（≤3x=TQQQ / >3x=くりっく株365 0.25%/yr）。
- **比較基線**: ①B3a素地（本検証の改善判定の主基準）②V7-TQQQ ③P09_C1。
- **B3a素地の指標（QC独立再現済・税後⓽）**: min⓽ +20.98%（割引後≈20.1%）/ Sharpe 0.904 / MaxDD −38.20% / Worst10Y★ +14.53% / Regime_min −2.88% / 2008窓 +16.78% / boot P(min vs V7) 0.893。

### 0.2 守るべき教訓（LESSONS_LEARNED）
- **A. post-hoc乗算評価は禁止** → 各要素は `src/integration/build_strategy_with_signal.py` 系の native 統合で組み込む（baseline.pct_change()×mult→cumprod 方式は使わない）。
- **B. defensive方向を優先的に試す** → 防御要素（G1/G2/G5）を先に回す。
- **C. multi-metric bootstrap 必須** → min⓽/MaxDD/Worst10Y★/Sharpe の4軸で対B3a素地・対V7。
- **D. IC は決定的でない** → IC スクリーニングは粗い篩。native フルゲートが本評価。
- **E. ETF/CFD 構造特異** → 他土台（CFD系/S3）由来の要素は **B3a土台で独立再検証必須**（転用実績を前提にしない）。

### 0.3 再検証しない（棄却済・RESEARCH_CONTEXT §3.2）
アンサンブル統合 / A2 ConvictionScore 直接レバ変換 / 単純動的レバ（P系 SOFR・Momentum・Composite・Kelly）/ リバランス頻度・部分リバ。これらは扱わない。

---

## 1. 検証する成功要素（5つ・両面）

各要素は B3a に**単独で**native統合し、まず安価スクリーン、生存者をフルゲートへ。再利用元コードは「仕様理解＋データ入力」に使い、統合層は B3a 構築コード（`leverup_b1c1_20260612.py` / `k365_recost_20260612.py`）の流儀に合わせて実装。

| # | 要素 | 設計（B3aへの統合方法） | 再利用元 | 方向 | 仮説 |
|---|---|---|---|---|---|
| **G1** | vol-target / DD-brakeガバナー | IN期の実効レバ `L` を `min(L, L×target_vol/realized_vol_63d)` でスケール（上限cap=現行L）。target_vol ∈ {25%,30%,35%}（年率）でスイープ。実現volはNASDAQ 63日（publication lag遵守） | vz065の `lmax` cap思想（`src/dynamic_leverage_strategies.py`） | 防御 | 高vol局面でレバを自動縮小→MaxDD−38%・2008弱体を是正。単一パラメータで過学習小 |
| **G2** | レジーム連動ブースト（E4移植） | V7ブースト倍率と/またはuniform scaleを**レジーム別**に縮小: 高vol or trend:bear で `scale×{0.85,0.90}`・boost mapを減衰。レジームラベルは `regime_labeler_20260611` 由来（trend/vol） | `src/e4_regime_klt.py`（k_lt のvol-regime条件） | 防御 | Regime_min −2.88%・2008防御−3pp を直接是正。弱気でレバを引く |
| **G3** | C2: bondOFF→Gold一部充填 | `_build_p09_nav_c1` のOUT∧bondOFF日を「現金(SOFR)」から「Gold比率 g + 現金(1−g)」へ。g ∈ {25%,50%,100%}。Gold脚は既存 gold_1x | `run_p02_p09_backtest`（P09充填）, `prepare_gold_local` | 両面 | OUT_bondOFF（全日23%・年+1.3%のみ）をGoldで底上げ（OUT_bondON+10.6%との差縮小）。C1(+0.25pp)の先 |
| **G4** | LT2-N750 長期逆張り移植 | 長期サイクル逆張りシグナルをB3aのIN期レバに加算/乗算統合（N=750）。**Lesson E によりCFD実績は前提にせず再評価** | `src/long_cycle_signal.py`, `src/b1_s2_lt2.py` | 攻め | 長期逆張りで押し目を厚く張る。Sharpe/gap改善の実績（CFD土台）がTQQQ土台で出るか |
| **G5** | 第2シグナル防御オーバーレイ | mom63はB3aがブースト使用中のため**別シグナル**でdefensive trim: vix_mom21 もしくは nfci_z52w を `{q0:1.00,q1:1.00,q2:0.92,q3:0.85}` 等のdefensiveマップで native 統合（高シグナル=リスクオン過熱→レバ減） | `src/integration/build_strategy_with_signal.py`, 信号拡張の G2/G3 CSV | 防御 | M6/V0 defensive（既ADOPT・MaxDD−5.83pp）の発想を**B3a土台で別シグナル**により再現。Lesson E で要再検証 |

> **G2 と G5 の使い分け**: G2 は「レジームでレバ縮小」（既存レジームラベル）、G5 は「外部シグナルでレバ縮小」（native信号統合）。両方防御だが機構が異なるため別要素として独立評価。

---

## 2. 評価ゲート（全要素・全組合せ共通）

| 段階 | 内容 | 通過条件 |
|---|---|---|
| Stage 0 安価スクリーン | 標準10指標（CAGR_IS/OOS・min⓽・gap・Sharpe・MaxDD・Worst10Y★・P10_5Y・Worst5Y・Trades/yr・>3x比率）＋4ハードベト | ベト無 ∧ (防御要素: MaxDD or Worst10Y or Sharpe が B3a素地より改善 ∧ min⓽劣化≤1.0pp) / (攻め要素: min⓽ ≥ B3a素地 −0.3pp) |
| Stage 1 フルゲート | WFA正典49窓(α/β)＋CPCV p10(N=10,k=2,embargo=21)＋レジーム層別(Regime_min)＋stress窓(2000/2008/2020/2022/2015) | ベト無 ∧ WFA α∩β PASS |
| Stage 2 統計 | **multi-metric bootstrap**（min⓽/MaxDD/Worst10Y★/Sharpe, N=10000, block21＋252感度）対B3a素地・対V7 | 改善方向のP(better)を開示（教訓C・有意/非有意を明記） |
| Stage 3 採点 | 再現可能6次元採点器（[scorecard_recompute](src/audit/scorecard_recompute_20260612.py)）でB3a/P09_C1と並記 | バランス採点・CAGR重視採点の両モード |

**ハードベト**: MaxDD<−50% / WFE>1.5 / Worst10Y★⓽<0 / Regime_min<−10%。
**選択バイアス規律**: 要素は本計画で事前登録。パラメータスイープ（target_vol・g・scale等）の最良値選択には deflated 値（選択バイアス補正）を併記。組合せは個別生存者のみから構成（総当たり禁止）。

---

## 3. 段取り（Phase 1→4）

| Phase | 内容 | 実行 |
|---|---|---|
| **Phase 1**（防御先行・Lesson B） | G1/G2/G5（防御）を各 native 統合＋Stage 0。並行して G3/G4（両面・攻め）も Stage 0 | Sonnet 並列（要素ごと独立サブエージェント） |
| **Phase 2** | Stage 0 生存者に Stage 1（フルゲート）＋Stage 2（bootstrap） | Sonnet |
| **Phase 3** | 生存要素の上位2-3を組合せ（例: 防御スタック G1+G2、攻めスタック G3+G4、混合 G1+G3）→ フルゲート＋bootstrap | Sonnet |
| **Phase 4** | Stage 3 採点＋B3a/P09_C1並記→改善候補の確定・正直な留保。CURRENT_BEST/REGISTRY 反映要否を提案 | Opus |

各 Phase 後にメインが結果を要約し、次 Phase の構成（どの組合せを回すか）を決める。

---

## 4. 採用判定基準

- **防御要素**: B3a素地比で MaxDD改善 ≥+2pp（or Worst10Y★ ≥+0.5pp or Sharpe ≥+0.02）∧ min⓽劣化 ≤1.0pp ∧ ベト無 ∧ multi-metric bootstrap で改善方向 P≥0.80。
- **攻め要素**: min⓽ 改善（割引後でも B3a素地超）∧ ベト無 ∧ MaxDD悪化 ≤+2pp。
- **組合せ**: 相互作用がプラス（単独より総合改善）∧ ベト無。
- **最終**: 6次元採点（両モード）で B3a を上回る or 「B3aの弱点（MaxDD/2008/Sharpe）を実質改善しCAGRをほぼ維持」する構成のみ改善候補化。20%は無理に届かせない（ベト緩和禁止）。

---

## 5. 成果物

| 種別 | ファイル |
|---|---|
| 計画（本書） | `MULTISTRATEGY_COMBINE_PLAN_20260615.md` |
| 各要素スクリプト | `src/audit/combine_g1_voltarget_*.py` 〜 `combine_g5_defoverlay_*.py`（ASCII print・cp932・docstring に native統合方法と再利用境界を明記） |
| CSV | `audit_results/combine_*_<date>.csv` |
| 結果レポート | `MULTISTRATEGY_COMBINE_RESULTS_<date>.md`（要素別Stage結果・組合せ・6次元採点・正直な留保） |

全成果物は毎回GitHubにpush後、検証済みURLを3列表で報告。

---

## 6. リスクと中止条件

- G4（LT2移植）はCFD土台実績のTQQQ転用＝Lesson E で劣化しうる。Stage 0 で min⓽劣化>1.0pp なら早期クローズ。
- G5（第2シグナル）は信号拡張で多数棄却済み。native フルゲートPASS かつ defensive 改善が出た場合のみ進める（IC高だけでは進めない・Lesson D）。
- 全要素が改善を出さない可能性は十分ある（B3aは既に強い）。その場合「B3aで打ち止め・改善なし」も正式な結論として報告する（無理な採用はしない）。

---

*管理者: 男座員也（Kazuya Oza）*
