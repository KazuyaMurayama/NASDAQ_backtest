# labor-zero v6 批判的検証（C1–C5 反証キャンペーン）実行計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

作成日: 2026-07-02
最終更新日: 2026-07-02

**Goal:** LABOR_ZERO_V6_RETURN_MC_20260629.md の主張 C1–C5 を、反証を第一目的として複数の独立角度（方法論感度・ルックアヘッド是正・白紙再実装・細密グリッド・将来悪化ストレス）から攻撃し、各主張を「支持/覆す/条件付き」の判定と証拠で確定させる。

**Architecture:** 既存 v6 ハーネス（`sim_one`/`mc_prob`）をアンカーとして固定し、①白紙独立再実装で数値土台を検証 → ②ブートストラップ方式・投入判定規約・確率定義を動かして結論の保存性を測る → ③グリッドを細密化しペア経路検定で差の有意性を付す → ④ストレスで実運用頑健性を暴く。全比較は共通シード＝共通経路のペア比較（McNemar 型）で行い、N=2000 の分解能問題を回避する。

**Tech Stack:** Python 3 (numpy/pandas)・既存 `src/audit/labor_zero_v6_*.py`・`labor_zero_v3_harness_20260628.py`（リターン系列ローダ）。

---

## §A 現状確認（2026-07-02 実施済み）

| 項目 | 結果 |
|---|---|
| リポ状態 | `main`・origin と同期済（HEAD=f626aff）。ハーネス5本＋QC2本すべて実在。 |
| 正典レポート | LABOR_ZERO_V6_RETURN_MC_20260629.md（239行）本文確認済。§3.2b/§3.4b/§3.5/§7 の数値・主張を抽出済。 |
| 申し送り | `docs/prompts/NEXT_SESSION_VERIFY_LABOR_ZERO_V6_20260701.md` の T1–T5 と本計画は整合。 |
| 関連commit | 8e3cc4b / 6356b70 / 379e6cb / ed5817f / 7ddde3e すべて `git log` で実在確認。 |
| 方法論正典 | EVALUATION_STANDARD.md §4.4（R-STAT-1/2/3）本文確認済。 |

**再現すべきアンカー数値（既存ハーネスの主張値・すべて block=5, N=2000, seed=20260629）:**

| # | 構成 | 主張値 |
|---|---|---|
| A1 | V6-C: run20/res20・thr20M・hold=True | P(labor0)=0.8745 |
| A2 | V6-C: run20/res20・thr26M・hold=True | 0.8965 |
| A3 | V6-C: run10/res30・thr14M・hold=True | 0.9170 |
| A4 | V6-A: floor3.6M・top_wr0.16・run20/res20・thr20M | 0.9995（5シード 0.9985–1.0000） |
| A5 | V6-A′: 同上 thr26M | 0.9945・実質終端中央220.7億 |
| A6 | 開始資産49M・run:res=1:1・thr=run0 | 0.957（0.950–0.957） |
| A7 | セルフテスト: 1975開始 labor=0 / 2012開始 labor=8（v3一致） |

### 計画立案時のコード読解で発見した新規疑義（本計画で必ず攻める）

**疑義D1（最重要・ルックアヘイド）**: `labor_zero_v6_return_mc_20260629.py` の `sim_one`（行113–122）は、**年初の投入判定に「その年の年次リターン r_run」を使い、投入した資金がその年のリターンを満額享受する**。現実には年次リターンは年末まで観測できないため、これは1年分の未来情報を使う実装。特に「早め投入（thr↑）が効く」（C1）・hold-if-crash（C2の前提）は、「非マイナスと判明している年の年初に投入できる」という反実仮想の恩恵を受けており、**C1 の +4.2pp・C3 の非対称が look-ahead 起因のアーティファクトである可能性**がある。→ Task 3 で3規約比較。

**疑義D2（統計分解能）**: N=2000 の P 分解能は 1/2000=0.0005。C4 の「thr26M で P−0.5pp」は**わずか10経路の差**であり、シード間ばらつき（±1pp）と同オーダー。→ 全タスク共通のペア経路検定（共通シード＝同一経路集合上で不一致経路を直接数える）と Wilson CI で決着させる。

**疑義D3（標本の構造）**: ①ラップアラウンド連結（2025→1975 の接合はレジーム断絶）②51標本という小標本からの復元 ③**sc2.6 自体が同じ史実51年で選抜された戦略**であり、その実現年次リターンを bootstrap する構造は選択バイアスを継承する（P は「バックテスト通りに戦略が機能する」条件付き確率）。→ Task 2（部分史実・定常bootstrap）と Task 6（リターン下方シフト）で暴露し、覆せなくても留保として定量を正典に載せる。

---

## §B 統計枠組み（全タスク共通・新設ヘルパー）

新規ファイル `src/audit/labor_zero_v6_stats_helpers_20260702.py` に以下を置き、各ハーネスから import する。

```python
"""Wilson CI + paired-path comparison helpers for labor-zero v6 critical verification."""
import numpy as np
from math import sqrt

def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for a binomial proportion k/n. Returns (lo, hi)."""
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)

def paired_diff(labors_a, labors_b):
    """Same-seed same-paths comparison of two configs.
    labors_a/b: int arrays of labor_years per path (identical path sets).
    Returns dict(n, a_only, b_only, mcnemar_p) where a_only = #paths where
    A achieves labor==0 but B does not. Exact binomial (two-sided) on discordant pairs."""
    a0 = (np.asarray(labors_a) == 0)
    b0 = (np.asarray(labors_b) == 0)
    a_only = int(np.sum(a0 & ~b0))
    b_only = int(np.sum(~a0 & b0))
    m = a_only + b_only
    if m == 0:
        return dict(n=len(a0), a_only=0, b_only=0, mcnemar_p=1.0)
    from scipy.stats import binomtest
    p = binomtest(min(a_only, b_only), m, 0.5).pvalue
    return dict(n=len(a0), a_only=a_only, b_only=b_only, mcnemar_p=float(p))
```

**規律**: ペア比較を成立させるため、**構成間比較は必ず同一 seed・同一経路生成順**で行う（`mc_prob` は経路生成が構成に依存しないのでこれが成立する——ただし白紙再実装側でも同じ構造を保つこと）。scipy が無い環境なら binomtest を正規近似に落としてよい（m≥10 で十分）。

**報告規律**: 数値4桁・シード複数（基準5本、重要比較20本）・レンジと CI 併記・合否ラベルなし生計測値。

---

## Task 0: アンカー再現（土台の固定）

**Files:**
- 実行のみ（新規ファイルなし）

- [ ] **Step 0-1:** `python src/audit/labor_zero_v6_return_mc_20260629.py` を実行し、セルフテスト PASS（A7）と R1 の 20:20 P=0.875 を確認
- [ ] **Step 0-2:** `python src/audit/labor_zero_v6_thr_dial_20260701.py` と `labor_zero_v6a_grid_20260701.py` を実行し A1–A5 を再現（4桁一致）
- [ ] **Step 0-3:** `python src/audit/labor_zero_v6_startasset_sweep_20260629.py` で A6 を再現
- [ ] **Step 0-4:** 不一致があれば**ここで停止して原因特定**（環境差/データ差/コード差の切り分け）。一致すればアンカー固定完了。

## Task 1: T4 白紙独立再実装（QCのQC）

**Files:**
- Create: `src/audit/labor_zero_v6_indep_20260702.py`

前任 QC（`labor_zero_v6_thr_qc_20260701.py` 等）は同一人物が書いたもの。今回は **v6/v3/既存QCのいずれのシミュレーション関数も import せず**、`sim`/`make_path` を白紙から書く（データローダ `v3.load_rets`/`load_mixed_reserve` のみ共有可。系列値そのものは検証対象外のため）。

- [ ] **Step 1-1:** 正典レポート §0.2–0.3 の**日本語仕様だけ**を仕様書として実装する（既存コードを見ずに書く。書き終えてから差分を突き合わせる）。年内の処理順序も自分の解釈で一度書き、既存実装（投入→支出→成長）と食い違ったら**その食い違い自体を記録**する（仕様の曖昧さの証拠になる）
- [ ] **Step 1-2:** アンカー A1(0.8745)/A3(0.9170)/A4(0.9995)/A6(0.957) を同一シードで再現。ズレたら原因（処理順・端数・ブロック生成）を特定し、どちらが仕様として正しいか判定
- [ ] **Step 1-3:** P(labor0)==P(ruin0)（名目固定）の定義上の帰結を**1経路の手計算トレース**（払えない年→全額支出→資産0→以後毎年labor）で確認し、フロア版で分離することも1経路トレースで確認。トレースはハーネスの `--trace` 出力として恒久化
- [ ] **Step 1-4:** commit: `git add src/audit/labor_zero_v6_indep_20260702.py; git commit -m "v6 critverify T4: clean-room reimplementation reproduces/refutes anchors"`

## Task 2: T1 方法論感度（block・定常bootstrap・部分史実・定義）

**Files:**
- Create: `src/audit/labor_zero_v6_method_sens_20260702.py`

代表4構成（A1/A3/A4/A5）＋開始資産49M（A6）を全感度軸で回し、**C1–C5 の「順位・符号」が保存されるか**を見る（絶対値の変動は想定内。順位反転が反証）。

- [ ] **Step 2-1:** block∈{3,4,5,6,8,10} × 代表構成。C1 の thr/配分順位（10:30>20:20、thr26>thr20）と C3 の非対称（V6-A は thr20 が thr26 以上）が全 block で保存されるか
- [ ] **Step 2-2:** **定常ブートストラップ**（Politis–Romano・平均ブロック長5年）を実装して同じ比較を実行:

```python
def make_stationary_path(rng, run_h, res_h, n=20, mean_block=5.0):
    """Stationary bootstrap: geometric block lengths (mean mean_block), wrap-around."""
    H = len(run_h)
    p_new = 1.0 / mean_block
    rp = np.empty(n); sp = np.empty(n)
    idx = int(rng.integers(0, H))
    for k in range(n):
        rp[k] = run_h[idx]; sp[k] = res_h[idx]
        if rng.random() < p_new:
            idx = int(rng.integers(0, H))
        else:
            idx = (idx + 1) % H
    return rp, sp
```

- [ ] **Step 2-3:** **部分史実**: 抽出元を 1975–2000（26年）/ 2000–2025（26年）に制限して同比較（レジーム依存の暴露）。ラップアラウンド接合の影響は「wrap禁止版 make_block_path（ブロックが端を跨がない）」も1本走らせて分離
- [ ] **Step 2-4:** **定義感度**: horizon∈{15,20,25,30}・「総資産<フロアで即labor」定義・（フロア版の）cut_years を主指標にした場合、で C3/C4 の推奨（V6-A 現状最良）が反転しないか
- [ ] **Step 2-5:** R-STAT-1/2/3 照合メモを書く: P(labor=0) は経路依存**サバイバル**指標であり、block=5「年」は暴落クラスタ長（2000–02=3年）以上で R-STAT-1 の「block≪危機長」違反には当たらない、という前セッションの暗黙の主張を明文化して検証（block=3 で暴落クラスタが割れて P がどう動くかを機構として説明できるか）
- [ ] **Step 2-6:** commit

## Task 3: 疑義D1 ルックアヘッド是正（本計画の独自攻撃・最優先級）

**Files:**
- Create: `src/audit/labor_zero_v6_lookahead_20260702.py`

投入判定の情報規約を3通り実装し、C1（thr↑+補填厚めの+4.2pp）と C3（フロア有無の逆転）が**非ルックアヘッド規約でも生き残るか**を測る。

- [ ] **Step 3-1:** 3規約を実装:
  - **(a) 現行**（look-ahead）: 年初に当年 r_run を見て投入、投入金は当年リターン享受（既存 `sim_one` と一致することをアンカーで確認）
  - **(b) ラグ信号**: hold_if_crash 判定に**前年** r_run[k-1] を使う（k=0 は投入可）。投入金は当年リターン享受
  - **(c) 年末投入**: 支出・成長の後に当年実現 r_run で判定して資金移動（投入金の運用リターン享受は翌年から。当年は債券リターンを享受済）
- [ ] **Step 3-2:** 代表構成（A1/A2/A3/A4/A5）× 3規約 × シード5本 ×（Task 2 の block5 基準）。ペア経路検定で (a) vs (b)/(c) の P 差と、**各規約内での thr/配分順位**を報告
- [ ] **Step 3-3:** 判定: 順位が (b)(c) で保存 → C1/C3 支持を強化。消失/反転 → **C1/C3 は look-ahead アーティファクト**として正典訂正（§3.2b/§3.4b/§0.3 と 30秒サマリ・§5 推奨表に波及）
- [ ] **Step 3-4:** commit

## Task 4: T2 タイミング系の再探索（「全滅」は本当か）

**Files:**
- Create: `src/audit/labor_zero_v6_timing2_20260702.py`

前回未探索の「より賢いタイミング」を、**規約(b)/(c)（非ルックアヘッド）でも**走らせる（look-ahead 下では早期投入が過大評価されるため、公平な土俵で再評価する意味がある）。

- [ ] **Step 4-1:** 追加ルール実装（各 run20/res20・10:30 の2配分 × thr{20,26}M）:
  - 回復確認投入: 直近2年連続 非マイナスで投入
  - 回復初動投入: マイナス年の翌 非マイナス年に投入（現行と同じだが規約(b)で）
  - runスリーブの実現DDトリガー: run残高がピーク比 −X%（X∈{20,30,40}）を割った後、非マイナス年に投入
  - 部分+早期の組合せ: thr26M で50%投入 → 残りを回復確認で投入
  - 予備側条件: 債券リターンが正の年のみ投入（株債同時安の回避）
- [ ] **Step 4-2:** 全ルールをペア経路検定で基準（全額一括+hold）と比較。**mcnemar_p と不一致経路数**で報告
- [ ] **Step 4-3:** 判定: いずれも基準を CI 有意に超えない → C2 支持（探索軸を明記して「この範囲で全滅」と限定表現）。超えるものがあれば C2 を覆し正典訂正
- [ ] **Step 4-4:** commit

## Task 5: T3 V6-A 細密グリッド＋top_wr 掃引（支出ルール側の穴）

**Files:**
- Create: `src/audit/labor_zero_v6a_fine_20260702.py`

- [ ] **Step 5-1:** thr∈{12,14,16,18,20,22,24,26,28,30}M × 配分{10:30,14:26,20:20,24:16,30:10} × floor=3.6M・top_wr=0.16。**シード20本**（seed=20260629+i）で P の平均±レンジ＋Wilson CI（pooled n=40,000）。0.9995 ピークの単峰性/ノイズ判定
- [ ] **Step 5-2:** **top_wr∈{0.12,0.14,0.16,0.18,0.20}** × floor∈{3.0,3.6,4.43}M × thr{20,26}M（配分20:20固定）。top_wr が P・保証フロア・終端を動かすか——**支出ルール側の未掃引ダイヤル**の決着
- [ ] **Step 5-3:** V6-A′ フロンティア検証: 「P−0.5pp で終端倍増」が thr∈{24,26,28,30} × top_wr × floor の近傍で滑らかに成立するか（thr26M 固有の偶然か）。終端中央値・p5 も4桁で
- [ ] **Step 5-4:** 判定: C3/C4 を支持/覆す/条件付き。「保証フロア360万は thr/配分/top_wr で上がらない」も P≥0.999 制約の最大フロア探索で再確認
- [ ] **Step 5-5:** commit

## Task 6: T5 実運用ストレス（史実パス依存の暴露）

**Files:**
- Create: `src/audit/labor_zero_v6_stress_20260702.py`

- [ ] **Step 6-1:** **一律リターンシフト**: run 年次リターンに −2pp / −5pp（債券は不変と、債券も −1pp の2種）を掛けた歴史から bootstrap。V6-A（floor3.6M）の P と「P≥0.999 を保てる最大フロア」の劣化曲線を出す（フロア360万が悪化世界で何万まで下がるか）
- [ ] **Step 6-2:** **最悪10年ブロック強制先頭**: 史実の最悪10年連続窓（run 累積リターン最小の窓を機械的に特定）を経路先頭に固定し、残り10年を bootstrap。V6-A/V6-C の P を報告（シーケンスリスクの下限提示）
- [ ] **Step 6-3:** C5 のストレス版: 開始資産 49M/60M の名目固定が −2pp 世界でどこまで落ちるか（「フロア化が唯一の道」の増強 or 弱体化）
- [ ] **Step 6-4:** commit

## Task 7: 判定・正典訂正・報告・申し送り

**Files:**
- Create: `LABOR_ZERO_V6_CRITVERIFY_20260702.md`（検証レポート正典）
- Modify: `LABOR_ZERO_V6_RETURN_MC_20260629.md`（覆した主張がある場合のみ訂正＋最終更新日）
- Modify: memory `project_nasdaq_labor_zero_refill_dial.md`（結果を追記）

- [ ] **Step 7-1:** C1–C5 それぞれに**支持/覆す/条件付き**の判定表を作る。各判定に「試みた反証角度の一覧と、崩れなかった/崩れた証拠（数値・CI・ペア検定）」を紐づける
- [ ] **Step 7-2:** 覆した/条件付きの主張は正典レポート本文（該当§と30秒サマリ・§5推奨表・§6留保）を訂正
- [ ] **Step 7-3:** 「まだ検証できていない残り穴」を列挙（最低限: sc2.6 の選択バイアス（疑義D3③）・リターンとインフレの同時確率化（v5×v6結合）・年次粒度（年内の暴落・回復は不可視）・税制/制度変更・bootstrap の iid ブロック仮定）
- [ ] **Step 7-4:** 一時ファイル削除確認 → `git add` → commit → `git push origin main` → `gh api repos/KazuyaMurayama/NASDAQ_backtest/contents/<path>?ref=main` で URL 検証 → 3列表で成果物報告
- [ ] **Step 7-5:** memory 更新（判定結果と教訓）

---

## 進め方・規律（申し送りプロンプト準拠）

- Task 2–6 は相互独立なので**サブエージェント並列可**（model=sonnet 可）。ただし**親が各ヘッドライン数値を独立に再現してから採用**（前回教訓: Workflow提案をメイン反証した事例）。
- 判定を分ける比較は必ず**ペア経路検定**（§B）を添える。「P が 0.3pp 動いた」だけでは判定しない。
- 実行順序: Task 0 → 1 →（2・3 並列）→（4・5・6 並列）→ 7。Task 1 でアンカーが崩れたら以降は原因確定を最優先に組み替える。
- 想定計算量: `sim_one` は 20年×N2000 で構成あたり <1秒。最重量は Task 5 Step 5-1（10×5×20シード=1000 run ≈ 数分）。全体で1セッション内に収まる。
- Desktop 直下生成禁止・一時ファイルは削除・ASCII print・CSV は `audit_results/`。

## 判定の事前基準（確証バイアス防止のため先に固定）

| 主張 | 支持の条件 | 覆すの条件 |
|---|---|---|
| C1 | thr/配分順位が 全block・定常bs・規約(b)(c)・部分史実の過半で保存、かつペア検定 p<0.05 | 非ルックアヘッド規約 or 定常bsで順位消失/反転 |
| C2 | Task 4 の全追加ルールが基準をペア検定で有意に超えない | 1つでも有意に超える（＋独立再現） |
| C3 | V6-A の「現状≥thr26M」が シード20本 CI とペア検定で維持 | V6-A で thr↑/配分変更が有意に P を上げる |
| C4 | フロンティア（P−0.5pp↔終端倍増）が thr近傍・top_wr・floor で滑らかに成立 | thr26M のみの孤立点（ノイズ）と判明 |
| C5 | 部分史実・ストレス下でも「資産増額<フロア化」の効率順位が保存 | いずれかで順位反転 |

※「絶対値が動く」は覆す条件ではない（bootstrap 方式で絶対値が動くのは想定内）。**順位・符号・推奨の反転**のみを反証とする。
