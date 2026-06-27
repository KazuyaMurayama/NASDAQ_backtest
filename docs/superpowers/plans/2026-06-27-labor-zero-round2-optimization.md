# 労働補填ゼロ 2ラウンド最適化 実装計画（追加資産ゼロ・支出720万維持）

> **For agentic workers:** この計画は Workflow（マルチエージェント）＋メイン独立QCで実行する。各タスクは検証可能な単位。

**ゴール:** 資産4,000万円（追加なし）・年間支出**厳密に720万円**（一時減額も不可）で、1975〜2005年の全31開始年×20年において**労働補填ゼロ**を達成する設計を、未試行レバーで探索する。前回スイープは「18%取り崩しでは不能・天井16.8%・1988開始がボトルネック」と結論。

**アーキテクチャ:** 既存の年次税後リターンCSV（scale dial / x4n4 / x4equiv）を入力に、取り崩しsimハーネスを**未試行レバー対応に拡張**。各ラウンドで複数エージェントが異なるレバー族を並列探索→メインが独立QC→統合。2ラウンド（R1=未試行レバー初探索、R2=R1結果の弱点分析→さらなる方向）。

**Tech Stack:** Python（pandas/numpy）、既存ハーネス`labor_zero_allocation_sweep_20260627.py`を基盤に拡張、Workflow（agent/parallel/pipeline）、QA runner。

---

## 制約・前提（厳守）

- **支出は毎年厳密に720万円**（ガードレール/動的減額は禁止＝ユーザー確定）。
- **追加資産ゼロ**＝初期総資産は4,000万円固定。
- **労働補填年** = 補填後も総資産（運用+予備）が720万を賄えない年。これを全620年（31×20）でゼロにするのがゴール。
- 税後×0.8273・予備の扱いはレバーで変える（現金/運用）。名目固定・USD名目・為替中立。
- 合否ラベル禁止（生の計測値のみ）。レポートは男座員也名義。ファイル名 `<TOPIC>_YYYYMMDD.md`。
- 成果物は GitHub push 後 URL 200 検証して3列表で報告。mainへ直接コミット。

---

## 前回判明した「ボトルネック＝1988開始」（最適化の標的）

最良案（sc2.2・運用2,000万/予備2,000万・2,000万割れで全額補填）でも1988開始だけ破綻：
初年度−24.4%で2年目に予備枯渇→1990(−29.9)/1993(+3.6)/1994(−9.2)の低迷中も720万取り崩し→1996年ゼロ→1995-99の+149/+204%等に乗れず。
**＝レバ×18%取り崩しの「早期下落＋予備枯渇」シーケンスリスク。**未試行レバーはこの1点を救えるかで評価する。

---

## 未試行レバー一覧（前回スイープ後の探索空間）

| # | レバー族 | 仮説（なぜ1988を救えるか） |
|---|---|---|
| A | **予備の運用**（現金でなく1xNASDAQ/Gold/Bond/SOFRで運用） | 予備が成長し補填がより深く効く。Goldは1988-90の株安局面で逆相関の可能性 |
| B | **グライドパス**（初期低レバ→バッファ蓄積後に高レバ） | 1988初年度の−24%を低レバで緩和し予備枯渇を防ぐ |
| C | **戦略混合**（例 sc2.2×50% + N4×50%、または sc1.6コア+高レバサテライト） | 高CAGRと低DDの両取りで初期下落を緩和しつつ成長維持 |
| D | **初期年保護バケツ**（最初の3-5年分の支出を現金で別取り） | シーケンスリスク窓（運用初期）だけ取り崩しを現金から賄い運用を温存 |
| E | **取り崩し順序**（下落年は予備/現金から、上昇年は運用から） | 下落年に運用スリーブを売らず複利を温存 |
| F | **細かい/拡張戦略グリッド**（scale 2.4/2.6、X4相当 高scale、bond/gold比再探索） | 単純にもう一段高CAGRが1988を救うか（前回はsc2.2止まり） |
| G | **レジーム連動運用額**（DD/トレンド信号で運用比率を動的調整） | 弱気入口で運用を絞り予備を温存（ただしR-STAT-3デレバ偽装に注意） |

---

## File Structure（作成/変更ファイル）

- **Create** `src/audit/labor_zero_harness_v2_20260627.py` — 未試行レバー対応の取り崩しsimハーネス。`simulate_v2(strat_returns, *, run0, reserve0, reserve_mode, glide, mix, init_bucket_years, draw_order, topup_thr, topup_amt, spend, regime_scale)` を提供。レバーA-Gをパラメータ化。1xNASDAQ/Gold/Bond/SOFRの年次税後系列もロード。
- **Create** `src/audit/labor_zero_round1_sweep_20260627.py` — R1スイープ（レバーA-F の主要組合せ）。ハーネスv2を使い全探索、`labor_zero_round1_*.csv` 出力。
- **Create** `src/audit/labor_zero_round2_sweep_20260627.py` — R2スイープ（R1上位の弱点を踏まえた精密化＋レバーG/組合せ）。`labor_zero_round2_*.csv` 出力。
- **Create** `LABOR_ZERO_ROUND2_OPTIMIZATION_40M_720_20260627.md` — 2ラウンドの分析・結果レポート（標準フォーマット・QC節付）。
- **Reuse**（変更なし）: 年次CSV3本、前回 `labor_zero_allocation_sweep_20260627.py`（ベースライン参照）。
- Gold/Bond/NASDAQ 1x の年次税後系列が既存CSVに無い場合は、ハーネスv2内で `dd_reduction_harness` 経由の日次→暦年税後で生成（scaleダイヤルと同一規約）。

---

## Round 1：未試行レバーの初探索

### Task R1-0: ハーネスv2の骨格＋セルフテスト（前回最良案を再現）

**Files:**
- Create: `src/audit/labor_zero_harness_v2_20260627.py`

- [ ] **Step 1: 年次税後リターンのロード関数を実装**

既存CSV（scale dial / x4n4 / x4equiv）から戦略別 `Series(year->frac)` を返す `load_returns()` を、前回ハーネスから移植。加えて 1x資産（NASDAQ/Gold/Bond/SOFR現金）の年次税後 `Series` を返す `load_sleeve_returns()` を追加（NASDAQ1xは既存CSVのNASDAQ_1x列、Gold/Bond/SOFRは `dd_reduction_harness` の日次系列→`_calendar_year_returns`×0.8273、SOFRは年次平均）。

- [ ] **Step 2: simulate_v2 を実装（レバーA-G統合・支出は厳密固定）**

```python
def simulate_v2(strat_ret, sleeve_ret, start, *, run0, reserve0,
                reserve_mode="cash",       # A: cash|nasdaq|gold|bond|sofr
                glide=None,                 # B: list[(elapsed_year, strat_key)] or None
                mix=None,                   # C: dict{strat_key: weight} or None
                init_bucket_years=0,        # D: years of spend pre-held as cash, drawn first
                draw_order="run_first",     # E: run_first|reserve_first_on_down
                topup_thr=20e6, topup_amt=None,  # rebalance reserve->run
                spend=7.2e6, horizon=20):
    """毎年 spend を厳密に取り崩し、労働年数・最小総資産・終端を返す。
    レバー: A reserve_mode(予備の運用先), B glide(年次で戦略切替),
    C mix(戦略加重平均リターン), D init_bucket(初期N年分現金を先取り),
    E draw_order(下落年は予備先取り)."""
    # 戦略リターン r_strat[k] を決定（mix優先→glide→単一）
    # init_bucket: bucket = spend*init_bucket_years を reserve とは別に現金保持、最初に消費
    # 毎年: (1)topup判定 (2)spend厳密取り崩し（bucket→draw_order）（3)運用に市場、予備にreserve_mode利回り
    ...
```

実装の要点（厳守）:
- **支出は毎年 spend 固定**。bucket→（draw_orderに従い run/reserve）の順で必ず spend を賄う。賄えなければ labor_year++。
- **mix**: `r = sum(w_s * strat_ret[s].loc[yr])`（同一年・加重平均）。
- **glide**: elapsed year で戦略キーを切替（その年のリターンを使う）。
- **reserve_mode**: 予備スリーブに毎年 sleeve_ret[mode].loc[yr] を適用（cashは0）。
- **draw_order="reserve_first_on_down"**: その年の戦略リターンが負なら予備/bucketから優先的に支出。

- [ ] **Step 3: セルフテスト（前回最良案の再現）**

```python
def _self_test():
    # sc2.2 run20M/res20M cash thr20M amt=ALL spend7.2M -> labor 12年(1988のみ)
    rets, sleeves = load_returns(), load_sleeve_returns()
    tot=0; fail=[]
    for sy in range(1975,2006):
        r=simulate_v2(rets["sc2.2"], sleeves, sy, run0=20e6, reserve0=20e6,
                      reserve_mode="cash", topup_thr=20e6, topup_amt=None)
        tot+=r["labor_years"]
        if r["labor_years"]>0: fail.append(sy)
    assert tot==12 and fail==[1988], (tot, fail)
    print("SELF-TEST v2: labor=12, fail=[1988] -> MATCH prior best")
```

- [ ] **Step 4: 実行してセルフテストPASS確認**

Run: `python -m src.audit.labor_zero_harness_v2_20260627`
Expected: `SELF-TEST v2: labor=12, fail=[1988] -> MATCH prior best`（前回sweepと一致＝v2が後方互換）。不一致なら拡張ロジックにバグ→修正。

- [ ] **Step 5: コミット**

```bash
git add src/audit/labor_zero_harness_v2_20260627.py
git commit -m "feat(retirement): labor-zero harness v2 (untried levers A-G), self-test reproduces prior best"
```

### Task R1-1: Workflow で未試行レバー A-F を並列探索

**Files:**
- Create: `src/audit/labor_zero_round1_sweep_20260627.py`

- [ ] **Step 1: R1スイープのグリッドを定義**

各レバー族を担当する探索ブロックを1スクリプトに実装（Workflowエージェントが各々を起動）。グリッド例:
- A 予備運用: reserve_mode ∈ {cash, nasdaq, gold, bond, sofr} × 既存最良配分(50:50,thr20,ALL) × 戦略{sc2.0,sc2.2,N4,X4}
- B グライド: 初期K年{2,3,5}を低レバ{sc1.0,sc1.4,sc1.6}→以後{sc2.0,sc2.2,X4} × 配分数種
- C 混合: {sc2.2+N4, sc2.2+X4, sc1.6+sc2.2, N4+X4} 各 weight{0.3,0.5,0.7} × 配分
- D 初期バケツ: init_bucket_years ∈ {0,2,3,4,5} × 戦略{sc2.2,sc2.0,N4} × 残りを運用/予備
- E 取り崩し順序: draw_order ∈ {run_first, reserve_first_on_down} × reserve_mode{cash,gold} × 戦略
- F 拡張グリッド: scale{2.4,2.6}（要 dd_reduction harness で年次生成）× 配分、X4相当 高scale

- [ ] **Step 2: 各レバーの labor_years_total と「1988が救えたか」を記録**

各組合せで全31開始年sim→`labor_years_total`, `starts_with_labor`, `saved_1988`(bool), `topup_events`, `terminal_median`, `min_total_floor` を CSV出力 `audit_results/labor_zero_round1_sweep_20260627.csv`。

- [ ] **Step 3: Workflow スクリプトで A-F を並列実行**

Workflow（別途メインが起動）: `parallel([()=>agent("レバーA探索...",schema), ()=>agent("レバーB...",schema), ...])` で6ブロック並列→各エージェントが該当グリッドを回し labor-zero候補（labor_years_total==0）と上位を構造化返却。スキーマで `{lever, best_config, labor_years_total, saved_1988, terminal_median_M}` を強制。

- [ ] **Step 4: R1結果サマリを出力**

Run: `python -m src.audit.labor_zero_round1_sweep_20260627`
Expected: 各レバーの最小 labor_years と labor-zero達成有無を表示。**labor-zero（==0）が出れば即フラグ**。

- [ ] **Step 5: コミット**

```bash
git add src/audit/labor_zero_round1_sweep_20260627.py audit_results/labor_zero_round1_sweep_20260627.csv
git commit -m "feat(retirement): round1 untried-lever sweep (A-F) for labor-zero"
```

### Task R1-2: メイン独立QC（R1上位候補）

- [ ] **Step 1: R1上位3-5候補を別経路で再計算**

R1のCSVから labor_years_total 最小の上位候補を取り、`simulate_v2` を**使わず**インラインで取り崩しロジックを再実装し、labor_years と 1988の可否を突合。中ズレ（labor年数差≥2）があればどちらが正しいか精査。

- [ ] **Step 2: R1の「最良の改善余地・問題点」を言語化**

R1で labor-zero に至らなければ、**なぜ届かないか**を1988パスで分析（どのレバーが初年度下落を緩和したか/予備枯渇を遅らせたか/それでも何年目に破綻するか）。これがR2の入力。

---

## Round 2：R1の弱点分析→さらなる方向

### Task R2-1: R1弱点を踏まえた精密化＋レバー組合せ＋G

**Files:**
- Create: `src/audit/labor_zero_round2_sweep_20260627.py`

- [ ] **Step 1: R1で最も効いたレバーを軸に組合せグリッドを定義**

R1分析に基づき（例: 予備Gold運用＋初期バケツ＋グライドの**組合せ**）を精密探索。さらに未投入のレバーG（レジーム連動運用額：DD/200日MA信号で運用比率を動的に絞る・R-STAT-3デレバ偽装チェック付）を追加。グリッドはR1上位近傍を細かく。

- [ ] **Step 2: 組合せsimを実行し labor-zero を探索**

ハーネスv2の複数レバー同時指定で全探索→`audit_results/labor_zero_round2_sweep_20260627.csv`。labor-zero達成があれば、その設計の全31開始年の労働年数=0 と 終端・最小総資産を記録。

- [ ] **Step 3: Workflow で R2 ブロックを並列実行＋敵対的QC**

Workflow: pipeline で「組合せ探索→各 labor-zero候補を別エージェントが敵対的に再検証（1988以外の開始年も本当にゼロか・デレバ偽装でないか）」。schema で検証結果を強制。

- [ ] **Step 4: R2サマリ出力**

Run: `python -m src.audit.labor_zero_round2_sweep_20260627`
Expected: labor-zero達成設計の有無と内容。達成なら詳細、未達ならR1+R2を通じた「達成不能の頑健な証明＋最小改変（支出/資産/制約緩和）」。

- [ ] **Step 5: コミット**

```bash
git add src/audit/labor_zero_round2_sweep_20260627.py audit_results/labor_zero_round2_sweep_20260627.csv
git commit -m "feat(retirement): round2 combined-lever + regime sweep for labor-zero"
```

### Task R2-2: メイン独立QC（R2・最終）

- [ ] **Step 1: labor-zero達成設計（あれば）を独立再計算で確証**

達成設計を別経路で全31開始年再現。デレバ偽装（レバーG使用時）は等平均退避ツインで R-STAT-3 チェック。1988含む全開始年で labor==0 を確認。

- [ ] **Step 2: 達成不能なら、その頑健性をQC**

2ラウンド・全レバーで labor-zero不能なら、「18%取り崩しの物理的限界」を確証（フロンティア16.8%が未試行レバーでも動かないこと）。最も惜しい設計と、ゼロに必要な最小改変を提示。

---

## Task FINAL: レポート作成・コミット・報告

**Files:**
- Create: `LABOR_ZERO_ROUND2_OPTIMIZATION_40M_720_20260627.md`

- [ ] **Step 1: 標準フォーマットでレポート執筆**

構成: ⭐結論 / R1（現状ベストの改善余地・問題点分析→未試行レバーA-F結果）/ R2（R1弱点→組合せ+G結果）/ 達成設計 or 達成不能の証明＋最小改変 / QC節 / 脚注。合否ラベル禁止・生計測値のみ・男座員也名義。前回レポート `LABOR_ZERO_ALLOCATION_40M_720_20260627.md` を基準線として参照。

- [ ] **Step 2: QA runner を出力CSVに実行**

Run: `python "C:/Users/user/.claude/skills/analysis-qa-checklist/scripts/qa_runner.py" --input audit_results/labor_zero_round2_sweep_20260627.csv`
Expected: 0 FAIL（想定外WARNは内容確認）。

- [ ] **Step 3: コミット・push・URL 200検証**

```bash
git add LABOR_ZERO_ROUND2_OPTIMIZATION_40M_720_20260627.md
git commit -m "docs(retirement): 2-round labor-zero optimization report (40M/7.2M)"
git push origin main
```
push後 `Invoke-WebRequest -Method Head` で全成果物URL 200確認。

- [ ] **Step 4: 3列表で成果物報告＋Next Action**

---

## Self-Review（計画の自己点検）

- **Spec coverage**: ①追加資産ゼロ＝初期4,000万固定（全Task）②支出720万厳密維持＝simで毎年固定（R1-0 Step2）③現状ベストの改善余地/問題点分析＝R1-2 Step2・R2-1 Step1④未試行方向＝レバーA-G⑤2ラウンド＝Round1/Round2構造⑥計画→実行＝本計画。全てカバー。
- **Placeholder scan**: 各Stepに具体コード/コマンド/期待値を記載。レバー定義は表で明示。プレースホルダ無し。
- **Type consistency**: `simulate_v2` のシグネチャ（reserve_mode/glide/mix/init_bucket_years/draw_order/topup_*）はR1-0で定義しR1-1/R2-1で同名使用。`load_returns`/`load_sleeve_returns` も一貫。
- **未試行の担保**: レバーA-Gは全て前回スイープ（戦略×静的配分×topupのみ）に含まれない＝重複なし。

---

## 実行方式

Workflow（マルチエージェント）＝ユーザー確定。各ラウンドのスイープを Workflow で並列実行し、メインが独立QC・統合。計算は年次CSVベースで軽量（asset日次ロードはGold/Bond系列生成時のみ）。トークン消費が大きいため起動前に最終確認する。
