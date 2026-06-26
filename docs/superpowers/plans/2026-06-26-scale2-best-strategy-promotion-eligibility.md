# Scale2.0（P09_STR scale2.0）ベスト戦略 昇格適格性 再検証 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ユーザー指示「Scale2.0 をベスト戦略にしていく」に対し、即昇格せず、独立QCで昇格適格性（確定値の再現性・MaxDDの経路頑健性・IS-OOS gapの過学習妥当性・ベスト判定基準との突合）を検証し、合格なら昇格手順を、不合格なら是正案を提示する。

**Architecture:** 既存の検証済みビルダー（`p09_strongmap_scale_dial_20260623.py` が呼ぶ `_build_full_c1` strong-map）を再利用し、新規シミュレーションは最小化。Stage 1 で確定値を独立再現、Stage 2 で前提（経路依存MaxDD・過学習gap）を第一原理で攻める。レポートはVETO/合否ラベルを使わず生の計測値のみ（プロジェクト永続ルール）。

**Tech Stack:** Python（numpy/pandas）、既存 `src/audit/` モジュール群、`compute_10metrics`、`crisis_window_timing` 系の経路頑健検定。

---

## 背景・確定済みの事実（再litigateしない前提）

「Scale2.0」= **P09_STR（strong boost map `{Q0:1.60,Q1:1.50,Q2:1.10,Q3:1.00}`）× lev_scale=2.0**。正典 `P09_STR_SCALE_DIAL_20260623.md`（2026-06-23確定・FRONTIER 4桁再現サニティPASS）の確定値：

| 指標 | Scale2.0 確定値 | 現行ベスト B3a_k365 | ベスト判定基準 |
|---|---|---|---|
| CAGR OOS⓽（min） | **+29.11%** | +20.98% | 大きいほど良 |
| CAGR IS⓽ | +35.38% | +23.10% | — |
| Sharpe_Full ⓒ | +1.028 | 0.904 | 大きいほど良 |
| **MaxDD ⓒ** | **−61.63%** | −38.20% | **<−50%=ハードベト抵触** |
| Worst10Y★⓽ | +20.84% | +14.53% | >0 |
| **IS-OOS gap** | **+7.57pp** | +2.57pp | **≥+5pp=過学習警戒** |
| **Regime_min** | **−11.41%** | −2.88% | **<−10%=ベト抵触** |
| WFE | 0.818 | 0.987 | — |
| 最大実効レバ | **9.6x**（全IN日L≥3） | ≤4.83x | くりっく株365証拠金/容量制約 |

**生成コード `p09_strongmap_scale_dial_20260623.py` のVETOロジック（L76-213）が Scale2.0 を3項目で自動VETO判定する**（MaxDD<−50%・Regime_min<−10%）。

**判定の核心**: Scale2.0 を「ベスト」に載せることは、CURRENT_BEST §1 が採用している「全ハードベトPASS」基準を緩めることを意味する。本検証はその数値が頑健かを確認し、最終的な基準緩和の可否をユーザーに提示する。

**再litigateしない前提（EVALUATION_STANDARD §1.5 v1.9）**: >3xレバ証拠金は担保（自己資金）でCAGRドラッグを生まない。唯一の継続コストは金利相当額（計上済）。実コスト=テールリスク。

---

## File Structure

- `src/audit/scale2_promotion_qc_20260626.py` （**Create**）— Stage1+Stage2 を1スクリプトに。確定値の独立再現＋経路頑健MaxDD検定＋gap反実仮想。検証済みビルダーのみ再利用、再実装なし。ASCII print、temp file なし。
- `audit_results/scale2_promotion_qc_20260626.csv`（生成）— scale 1.4/1.6/1.8/2.0 横並びの再現値＋経路頑健指標。
- `SCALE2_PROMOTION_ELIGIBILITY_20260626.md`（**Create**）— 最終レポート。生の計測値のみ、VETO/合否ラベル無し。昇格可否はユーザー判断材料として提示。

---

## Task 1: Stage 1 — Scale2.0 確定値の独立再現

**Files:**
- Create: `src/audit/scale2_promotion_qc_20260626.py`
- Reuse: `src/audit/p09_strongmap_scale_dial_20260623.py`（同一ビルダー `_build_full_c1` + STRONG_MAP + EXCESS_EXTRA）

- [ ] **Step 1: スクリプト骨子を書く（multitasking stub・path・import）**

`p09_strongmap_scale_dial_20260623.py` L33-73 と同一の import 構成を使う（`strategy_runners`, `unified_metrics.compute_10metrics`, `k365_recost_20260612._build_full_c1` + `EXCESS_EXTRA_K365_CENTRE`, `run_p01/p02` ヘルパ, `regime_labeler`, `extended_eval._eval_one`）。STRONG_MAP={0:1.60,1:1.50,2:1.10,3:1.00}、SCALES=[1.4,1.6,1.8,2.0]。

- [ ] **Step 2: build_strong(sc) を再現し、scale2.0 の確定値を独立算出**

`_build_full_c1(shared, dates_dt, n_years, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr, v7_map=STRONG_MAP, lev_scale=sc, excess_extra=EXCESS_EXTRA)` を呼び、`compute_10metrics` → `_apply_aftertax` で CAGR_IS/OOS/MaxDD/Sharpe/Worst10Y/gap、`_eval_one` で WFE/CI95/Regime_min を算出。

- [ ] **Step 3: 正典アンカーと4桁照合（独立再現の合否）**

期待値（`p09_strongmap_scale_dial_20260623.py` STRONG_ANCHORS）: sc2.0 → CAGR_IS +35.3755% / OOS +29.1102% / MaxDD −61.6342%。許容 ±0.15pp。一致＝Stage1 PASS（確定値は今も再現する）。不一致＝コードドリフトを報告し原因究明（昇格判断を止める）。

Run: `python src/audit/scale2_promotion_qc_20260626.py`
Expected: `STAGE1 REPRODUCE: sc2.0 IS+35.38% OOS+29.11% MaxDD-61.63% -> MATCH (tol 0.15pp)`

---

## Task 2: Stage 2-A — MaxDD −61.63% の経路頑健性検証

**Files:**
- Modify: `src/audit/scale2_promotion_qc_20260626.py`（Stage 2-A 関数追加）
- Reference: EVALUATION_STANDARD §4.4（R-STAT-1/2/3）, memory `project_nasdaq_methodology_lessons.md`

**前提（memory教訓）**: MaxDD/Worst10Y は経路依存極値。block=21 ブートストラップは無効（R-STAT-1）。実NAVから直接算出が正しい。本タスクは「Scale2.0 の MaxDD −61.63% が過小評価されていないか（実は更に深い局面が隠れていないか）」を経路頑健に確認する。

- [ ] **Step 1: 実NAVから MaxDD と最深DD局面の期間・年を直接算出**

`(nav/nav.cummax()-1).min()` の値と、その drawdown が発生した開始/底/回復の日付・継続日数を抽出。scale1.0/1.4/2.0 を横並びにし、scale比例で深くなる経路（同一暴落シーケンスがレバ倍率で増幅）であることを確認。

- [ ] **Step 2: 経路集約指標で裏取り（block≥252 が許容される指標のみ）**

R-STAT-2の補助に従い、time-under-water（水面下日数）・avg-drawdown・CVaR-of-DD を scale1.0/2.0 で算出。これらは経路集約指標なのでscale間の相対比較に使える。「MaxDDだけでなく回復時間・平均的な痛みも scale2.0 で悪化」かを定量化。**block=21 のMaxDDブートストラップは使わない（R-STAT-1遵守）。**

- [ ] **Step 3: 最悪暴落窓での実損失（無傷の経路）**

2000-2002 ドットコム・2008 GFC・2022 利上げの各窓で、scale2.0 の実NAV累積損失を算出（窓を壊さない）。「−61.63% は単一の最悪局面か、複数局面で同等の深さか」を提示。

Run: `python src/audit/scale2_promotion_qc_20260626.py`
Expected: `STAGE2A: MaxDD=-61.63% path=[start..bottom..recover], TUW=Ndays, worst windows: dotcom -X% gfc -Y% 2022 -Z%`

---

## Task 3: Stage 2-B — IS-OOS gap +7.57pp の過学習妥当性（反実仮想）

**Files:**
- Modify: `src/audit/scale2_promotion_qc_20260626.py`（Stage 2-B 関数追加）

**前提（反実仮想で攻める）**: gap +7.57pp は「IS期に都合よく効いた」過学習の証拠か、それとも「レバが強気年だけ増幅する構造」の必然か。正典の観測：OUT年（2008/2022/2001/2002）はscale不変、強気年だけ振幅拡大。これが正しければ gap拡大は過学習でなく構造由来。

- [ ] **Step 1: gap を scale 1.0→2.4 で分解（IS-CAGR と OOS-CAGR を別々に追跡）**

scale上げで IS-CAGR と OOS-CAGR がそれぞれどれだけ伸びるかを算出。gap拡大が「ISだけ伸びてOOSが頭打ち（過学習）」か「両方伸びるが IS の方が急（構造）」かを切り分け。

- [ ] **Step 2: OUT年scale不変性の確認（構造仮説の検証）**

2008/2022/2001/2002 の年次リターンが全scaleで同一（+20.3%/+0.7%/+2.8%/+15.0%）であることを実NAVで確認。OUT期はレバ非適用＝C1充填がscale非依存、という構造を裏取り。これが成立すれば「下方は守られ上方の振幅だけ拡大」＝gapは純レバの幾何効果。

- [ ] **Step 3: 一律デレバ双子との比較（時機 vs レバ水準・R-STAT-3）**

scale2.0 と同一平均エクスポージャになる「一律デレバ scale」を求め、両者の gap・MaxDD・CAGR を比較。Scale2.0 の gap が「賢いタイミング由来」でなく「単にレバ水準が高いから」であることを示す（memory教訓: DD削減もgap拡大もレバ水準ダイヤルが本質）。

Run: `python src/audit/scale2_promotion_qc_20260626.py`
Expected: `STAGE2B: gap decomposition IS+X OOS+Y; OUT-years scale-invariant=True; gap is leverage-geometric not timing`

---

## Task 4: Stage 2-C — ベスト判定基準との突合・昇格すると何を緩めるか

**Files:**
- Modify: `src/audit/scale2_promotion_qc_20260626.py`（突合サマリ出力）

- [ ] **Step 1: 現行ベスト判定基準を明示的にリスト化**

CURRENT_BEST §1 / STRATEGY_REGISTRY の判定基準（MaxDD<−50%・gap≥+5pp・Regime_min<−10% の3ハードベト、Sharpe risk-adjusted、最大実効レバ実務制約）を列挙。Scale2.0 が各基準でどう位置するかを生の数値で対照（合否ラベルは付けない＝プロジェクト永続ルール）。

- [ ] **Step 2: 「昇格すると緩めることになる基準」を特定**

Scale2.0 をベストにすると、(a) MaxDD許容を −38%→−62% に拡大、(b) 過学習gap許容を +2.57pp→+7.57pp に拡大、(c) Regime_min許容を −2.88%→−11.41% に拡大、(d) 最大実効レバを ≤4.83x→9.6x に拡大、することを定量提示。CAGR +20.98%→+29.11% の対価として何を引き受けるかを明示。

- [ ] **Step 3: 中間scale（1.4）との比較を併示**

正典が「MaxDD−50%以内の実質上限」とする scale1.4（CAGR_OOS +24.34%・MaxDD −46.48%・gap +3.80pp・Regime −5.86%）を対照列に置き、「CAGRを少し諦めれば基準内に収まる」選択肢があることを数値で示す（ユーザーがscale水準を再選定する材料）。

---

## Task 5: 最終レポート作成・QC・push

**Files:**
- Create: `SCALE2_PROMOTION_ELIGIBILITY_20260626.md`

- [ ] **Step 1: レポート執筆（生の計測値のみ・VETO/合否ラベル禁止）**

H1直下に作成日/最終更新日（2026-06-26）、著者 男座員也。構成: §0サマリ（Scale2.0の確定値再現＋昇格すると緩める基準＋scale1.4代替の3点）、§1 メカニズム（strong-map純レバ）、§2 Stage1再現、§3 Stage2-A 経路頑健MaxDD、§4 Stage2-B gap反実仮想、§5 Stage2-C 基準突合、§6 QC、§7 結論（分析であり推奨ではない・最終判断はユーザー）、§8 脚注。**MaxDD/Regime/gap は生の計測値と事実記述のみ。「ベト」「不合格」と書かない。**

- [ ] **Step 2: analysis-qa-checklist 準拠の自己QC**

税基準統一（CAGR=税後×0.8273、Sharpe/MaxDD=税前ⓒ）、経路依存指標にblock=21未使用（R-STAT-1）、確定値とアンカーの4桁照合、Scale2.0の数値がCURRENT_BEST正典と整合、を確認しレポート§6に記載。

- [ ] **Step 3: git add/commit（main直接・末尾Co-Authored-By）**

```bash
git add src/audit/scale2_promotion_qc_20260626.py audit_results/scale2_promotion_qc_20260626.csv SCALE2_PROMOTION_ELIGIBILITY_20260626.md docs/superpowers/plans/2026-06-26-scale2-best-strategy-promotion-eligibility.md
git commit -m "..."
git push origin main
```

- [ ] **Step 4: push後にブランチ実値取得・URL存在確認・3列表で報告**

`git rev-parse --abbrev-ref HEAD` → `gh api repos/KazuyaMurayama/NASDAQ_backtest/contents/SCALE2_PROMOTION_ELIGIBILITY_20260626.md?ref=<branch>` で200確認 → `/blob/<branch>/<path>` でMarkdownリンク化。

- [ ] **Step 5: 合否に応じた次アクション提示**

確定値が再現し基準緩和をユーザーが受け入れる前提なら → CURRENT_BEST §🏆 / STRATEGY_REGISTRY §1 への昇格手順を提示（別タスク・ユーザー最終承認後）。基準内を優先するなら → scale1.4 への再選定を推奨。**昇格作業自体は本検証では実行せず、ユーザーの最終判断を仰ぐ。**

---

## Self-Review

- **Spec coverage**: 「再検証して合格なら昇格・不合格なら是正案」→ Task1(再現)/Task2-4(QC)/Task5(判定と次アクション)で網羅。
- **永続ルール遵守**: レポートにVETO/合否ラベルを書かない（Task5 Step1で明示）。生の計測値のみ。
- **方法論教訓遵守**: block=21 をMaxDDに当てない（Task2 Step2）、時機vsデレバ切り分け（Task3 Step3）、税基準統一（Task5 Step2）。
- **再実装回避**: 検証済みビルダー `_build_full_c1` を再利用、独立再現はアンカー照合で担保。

*管理者: 男座員也（Kazuya Oza）*
