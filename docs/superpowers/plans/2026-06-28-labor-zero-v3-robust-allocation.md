# 労働補填ゼロ v3（全開始年頑健化）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** v2（sc2.6・運用1,400万/予備2,600万債券・1,400万割れ全額）が 2012-2014 開始で破綻した過剰適合を解消し、1975-2020 の全46開始年×各データ最大ホライズンで労働補填0年（厳格）を満たす設計 v3 を、複合レバー探索で見つける（無ければ破綻最小の最良設計を出す）。

**Architecture:** 既存 `labor_zero_harness_v2_20260627.py` の `simulate_v2` を拡張した v3 シミュレータ（分割投入・予備混合・拡張開始年集合）を1ファイルで作る。探索はマルチエージェントWorkflowで複合レバーを並列スイープ→メインが全46開始年で独立再検証。評価は必ず 1975-2020 を含む。

**Tech Stack:** Python（既存ハーネス・pandas・numpy）、Workflow（並列エージェント）。税後×0.8273・同一ビルダー。

---

## 背景（このplanが解く問題）

- **v2 の過剰適合（実証済 LABOR_ZERO_V2_HOLDOUT_20260628.md）**: v2 は予備を「1回全額投入」。IS（1975-2005）では最悪暴落が運用初期に来たため1発で足りたが、2012-2014 開始は暴落（2015 sc2.6 −38.2%）が**弾を撃ち尽くした後**に来て破綻（運用+予備0・7〜9労働年）。
- **2015 は株も債券も金もマイナス**（sc2.6 −38.2% / bond −2.3% / gold −9.5%）＝「暴落時に低相関資産が助ける」が効かない年。**よって予備の資産変更だけでは救えない可能性が高い**。本命の対策は「弾の温存（分割投入）」と「初期DDを浅くする（レバ低下）」。
- **合格条件（ユーザー確定）**: 全46開始年（1975-2020）で労働0（厳格）。無ければ破綻開始年を最小化する最良設計を提示し「達成不能」と正直に報告。

---

## v3 で導入する未試行レバー（v2 にないもの）

| レバー | 内容 | v2 との差 | 仮説 |
|---|---|---|---|
| **G1 分割投入** | 予備を「1回全額」でなく N トランシェに分割（例: 1,400万割れで予備の1/2、さらに割れたら残り） | v2=1発 | 暴落が後から来ても二の矢を残す＝2012-2014救済の本命 |
| **G2 投入閾値の階層** | 複数閾値（例: 1,400万で1段・1,000万で2段・600万で3段） | v2=単一閾値 | 急落時ほど厚く撃つ |
| **G3 予備の混合** | 予備を bond/gold/cash/sofr の加重混合 | v2=bond単独 | 2015型（株債同時安）で gold/cash が下支え（ただし2015 goldも−9.5%＝限定的、要検証） |
| **G4 レバ低下** | 運用を sc2.6→sc2.4/2.2/2.0/X4eq に下げ、その分予備を厚く/分割 | v2=sc2.6固定 | 初期DDを浅くし破綻の引き金を弱める |
| **G5 運用/予備比** | 運用1,000〜2,000万 × 予備3,000〜2,000万 | v2=14/26固定 | 比率の再最適化 |

> 評価軸（全レバー共通・厳格条件）: **1975-2020 全46開始年×各データ最大ホライズン**で `labor_years==0` かつ floor を最大化。終端資産中央値は tie-break。

---

## File Structure

- **Create `src/audit/labor_zero_v3_harness_20260628.py`**: v3 シミュレータ。`simulate_v3()`（分割投入・階層閾値・予備混合・拡張開始年）、`run_all_starts_v3()`（1975-2020・可変ホライズン）、`load_mixed_reserve()`（bond/gold/cash/sofr 加重）、`_self_test()`（v2 を v3 の特殊形＝1トランシェ・bond単独 として再現し、IS labor0/floor18.12・holdout 3破綻 を再現）。**まずこれを単独で検証**。
- **Create `src/audit/labor_zero_v3_sweep_20260628.py`**: v3 複合レバー全探索（メイン側の確定スイープ。Workflow と独立に同じ格子を回し突き合わせる土台）。
- **Workflow（インライン・別ファイル化しない）**: 4〜6エージェントが G1〜G5 の異なる断面を並列探索→構造化出力→メインが上位候補を `run_all_starts_v3` で全46開始年・独立再検証。
- **Create `LABOR_ZERO_V3_ROBUST_20260628.md`**: v3 結果レポート（達成可否・最良設計・全46開始年表・破綻メカニズム・QC）。
- **Update `LABOR_ZERO_V2_HOLDOUT_20260628.md`**: §4 から v3 結果へのリンクを追記。
- **Update memory** `project_nasdaq_labor_zero_allocation.md`。

---

## Task 1: v3 ハーネスのセルフテスト（v2 を特殊形として再現）

**Files:**
- Create: `src/audit/labor_zero_v3_harness_20260628.py`
- 検証: 既存 `labor_zero_harness_v2_20260627.py`（v2基準）/ `labor_zero_v2_holdout_20260628.py`（holdout基準）

- [ ] **Step 1: v3 シミュレータを書く（分割投入・階層閾値・予備混合・拡張開始年）**

```python
# simulate_v3(rets, reserve_series, start, horizon, *, strat, run0, reserve0,
#             tranche_thresholds, tranche_fracs, spend, data_end):
#   tranche_thresholds: list of run-balance thresholds (降順) e.g. [14e6, 10e6, 6e6]
#   tranche_fracs:      list of 投入する「その時点の残予備」割合 e.g. [0.5, 0.5, 1.0]
#   各年: 残高がthr[i]を割り、かつそのトランシェ未投入なら、残予備×frac[i]を運用へ。
#   v2互換 = tranche_thresholds=[14e6], tranche_fracs=[1.0]（=全額1発）。
# reserve_series は load_mixed_reserve(weights) で作った加重後の年次税後Series。
# horizon = min(horizon, data_end-start+1) でデータ末尾を厳守。
# 返り値: dict(labor_years, floor_M, floor_yr, terminal_M, n_years, tranches_fired)
```

- [ ] **Step 2: load_mixed_reserve を書く（予備の加重混合）**

```python
# load_mixed_reserve(weights) -> Series(year->frac after-tax)
#   weights: dict like {"bond":0.7,"gold":0.2,"cash":0.1}; sum==1.
#   既存 load_sleeve_returns() の各modeを年次で加重平均。cash=0系列。
```

- [ ] **Step 3: セルフテスト＝v2 を v3 の特殊形で再現**

```python
# _self_test():
#   v3(strat=sc2.6, run0=14M, reserve0=26M, reserve=mixed{bond:1.0},
#      tranche_thresholds=[14M], tranche_fracs=[1.0]) を
#   (a) IS 1975-2005 x20y -> labor==0 and floor≈18.12M
#   (b) holdout 2006-2020 data-max -> 3 starts fail (2012/2013/2014), labor_total 一致
#   の両方で既存値を再現したら PASS。再現しなければ HALT（土台が誤り）。
```

Run: `PYTHONIOENCODING=utf-8 python src/audit/labor_zero_v3_harness_20260628.py`
Expected: `SELF-TEST PASS: v2 reproduced (IS labor=0 floor=18.12; holdout fails=[2012,2013,2014])`

- [ ] **Step 4: コミット（セルフテストPASS後のみ）**

```bash
git add src/audit/labor_zero_v3_harness_20260628.py
git commit -m "feat(labor-zero): v3 harness (split-tranche refill + mixed reserve), v2 reproduced as special case"
```

---

## Task 2: メイン確定スイープ（Workflow と独立に突き合わせる土台）

**Files:**
- Create: `src/audit/labor_zero_v3_sweep_20260628.py`
- Output: `audit_results/labor_zero_v3_sweep_20260628.csv`

- [ ] **Step 1: 全46開始年・可変ホライズンの評価関数**

```python
# eval_config(rets, cfg) -> dict:
#   全46開始年(1975-2020)で simulate_v3 を data-max horizon で回し、
#   labor_total, starts_fail(list), min_floor_M, term_median_M を返す。
#   合格 = (labor_total==0)。
```

- [ ] **Step 2: 複合レバー格子を定義してスイープ**

```python
# 格子（厳格条件下で現実的な広さ）:
#   strat ∈ {sc2.6, sc2.4, sc2.2, sc2.0, X4eq_sc2.2}
#   run0  ∈ {10,12,14,16,18}M ; reserve0 = 40M-run0
#   tranche ∈ {
#     [(14M,1.0)],                      # v2互換
#     [(14M,0.5),(8M,1.0)],             # 2段
#     [(16M,0.4),(10M,0.5),(6M,1.0)],   # 3段
#     [(18M,0.33),(12M,0.5),(6M,1.0)],
#   }
#   reserve_mix ∈ {bond, bond70/gold20/cash10, bond50/cash50, gold50/bond50, cash}
#   注: thresholdsはrun0以下にclip。約 5*5*4*5 = 500 構成。
# 各構成 eval_config し CSV 出力。labor_total==0 を抽出して floor 降順表示。
```

- [ ] **Step 3: 実行して合格設計の有無を確認**

Run: `PYTHONIOENCODING=utf-8 python src/audit/labor_zero_v3_sweep_20260628.py`
Expected: 標準出力に `PASS configs (labor_total==0): N 件` と上位の floor。0件なら最小破綻設計を表示。

- [ ] **Step 4: コミット**

```bash
git add src/audit/labor_zero_v3_sweep_20260628.py audit_results/labor_zero_v3_sweep_20260628.csv
git commit -m "feat(labor-zero): v3 main combined-lever sweep over 46 starts (strict labor-zero)"
```

---

## Task 3: マルチエージェントWorkflow探索＋メイン独立検証

**Files:**
- Workflow をインライン実行（Workflow ツール）。schema で構造化出力。
- メインが上位候補を Task 1 のハーネスで全46開始年・独立再検証。

- [ ] **Step 1: Workflow を実行（4〜6エージェント並列）**

設計（各エージェントが異なる断面を深掘り、メインが統合・検証）:
- A: 分割投入の段数・割合（G1/G2）を sc2.6 固定で最適化
- B: 予備混合（G3）を分割投入と組み合わせて 2012-2014 を狙い撃ち
- C: レバ低下（G4・sc2.4/2.2/X4eq）×厚い予備
- D: 運用/予備比（G5）の再最適化
- E（懐疑・必須）: 候補の「2012-2014 救済が他開始年を壊していないか」を全46開始年で反証
- 統合: 各候補を構造化（strat/run0/tranche/mix/labor_total/min_floor/fails）で返す

- [ ] **Step 2: メインが上位候補を独立再検証（厳格条件）**

```python
# Workflow が返した上位 ~10 候補を、Task1 ハーネスの run_all_starts_v3 で
# 全46開始年・data-max horizon で再計算。labor_total と fails が Workflow 報告と
# 一致するか突き合わせ（不一致は予算/実装差を疑い是正）。合格 = labor_total==0。
```

- [ ] **Step 3: 合否で分岐**

- 合格設計あり → floor 最大・終端中央値で v3 確定。
- 合格設計なし → 破綻開始年を最小化する最良設計を v3 候補とし、「全開始年達成は不能」と正直に結論（最小破綻数・どの開始年が原理的に救えないか）。

---

## Task 4: v3 レポート作成＋QC＋成果物報告

**Files:**
- Create: `LABOR_ZERO_V3_ROBUST_20260628.md`
- Update: `LABOR_ZERO_V2_HOLDOUT_20260628.md`（§4にv3リンク）
- Update: memory `project_nasdaq_labor_zero_allocation.md`

- [ ] **Step 1: レポート執筆**（達成可否・最良設計・全46開始年表・破綻メカニズム・v2との対比）。合否ラベル（VETO等）は付さず生計測値（labor年数・floor・終端）で提示。

- [ ] **Step 2: QC（analysis-qa-checklist）**: ①v3ハーネスのv2再現サニティ ②全46開始年の独立再検証一致 ③厳密40M予算 ④分割投入の経済的整合（2012-2014が救われるパス・トレース）⑤Workflow候補とメイン検証の数値一致 ⑥データ末尾厳守 ⑦税基準統一。**第一原理で前提を攻める**（「2012-2014救済が他年を壊していないか」「短ホライズン開始年は未顕在なだけでないか」）。

- [ ] **Step 3: 数値再検証**（レポートの主要数値を CSV と突き合わせ・一致確認）。

- [ ] **Step 4: コミット・push・URL 200検証・3列表で報告**

```bash
git add LABOR_ZERO_V3_ROBUST_20260628.md LABOR_ZERO_V2_HOLDOUT_20260628.md audit_results/labor_zero_v3_*.csv
git commit -m "analysis(labor-zero): v3 robust-allocation result over all 46 starts (1975-2020)"
git push origin main
```

---

## Self-Review（plan点検）

- **スコープ**: v3 ハーネス→確定スイープ→Workflow探索→レポートの4タスク。各タスクは単独でtestable（セルフテスト/スイープ出力/検証一致/レポート）。
- **過剰適合の再発防止**: 全タスクが**1975-2020全46開始年**を評価軸に固定（v2 の失敗＝IS 31開始年のみ評価、を構造的に防ぐ）。Workflow に懐疑エージェントE（他開始年を壊さないか反証）を必須化。
- **正直な失敗報告**: 厳格条件を満たせない場合に「達成不能＋最小破綻」を出す分岐を Task3 Step3 に明記（プラセボ達成を避ける）。
- **型整合**: `simulate_v3` の tranche 表現 `[(threshold, frac), ...]` を Task1/2/3 で一貫使用。`load_mixed_reserve(weights dict)` も一貫。
- **プレースホルダ無し**: 各 Step に実行コマンド・期待出力・確定する格子を明記済。

---

## Execution Handoff

本plan保存後、**Inline Execution（executing-plans）**で進める（Task1→4を順に・各タスクでチェックポイント）。Task3 のみ Workflow ツールを内部で使う。
