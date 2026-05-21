# 06. 新検証アイデア着手前プロトコル（必須・スキップ禁止）

新しい戦略・改善アイデア・パラメータ探索を始める前に**以下4ステップを順番に実施**する。
スキップした場合は重複研究とみなし、検証結果は採用候補から除外する。

## Step 1: 重複チェック（30秒）
1. `STRATEGY_REGISTRY.md` を読み込む
2. アイデアのキーワード（例: "VolSpike", "VIX", "SOXL", "Regime", "LongCycleSignal", "TimingGate"）で `Theme` 列と §5 逆引きインデックスを検索
3. ヒット時の対応:
   - `Status = Rejected` → `Decision Reason` を読み、再挑戦の妥当性を**1段落で明文化**してから着手
   - `Status = Deferred` → §4 の解放条件が満たされているか確認
   - `Status = Active / Shortlisted` → 差分が明確でなければ**中止**

## Step 2: 評価基準の確認（30秒）
1. `EVALUATION_STANDARD.md` の `§0 標準前提サマリ` を読む
2. 検証スクリプトの冒頭コメントに必ず明記:
   - `# Evaluation Standard: v1.0`
   - `# Cost Scenario: D`（または逸脱理由）
   - `# IS: 1974-01-02〜2021-05-07 / OOS: 2021-05-08〜現在`

## Step 3: 現行ベストとの差分仮説を言語化（必須）
着手前にユーザーへ1〜3行で報告:
- 現行ベスト（`CURRENT_BEST_STRATEGY.md` 参照）の CAGR_OOS / Sharpe_OOS に対し
- **何を変えると、どの指標が、どれくらい改善する仮説か**
- 失敗時に STRATEGY_REGISTRY に何と記録するか（棄却理由のテンプレ）

## Step 4: 検証完了後の登録（必須・完了報告前に実施）
採用・棄却・保留いずれの場合も `STRATEGY_REGISTRY.md` に1行追記してから報告完了とする。
**追記なしの完了報告は不可。**

### 例外
- 既存戦略のバグ修正・コスト前提アップデート等の「再計算」は Step 1〜3 省略可
- ただし Step 4（`STRATEGY_REGISTRY.md` の該当行更新）は必須
