## 🔍 戦略監査 — Claude Code 自動実行プロトコル（毎回必須・ユーザー確認不要）

本セクションは Claude Code 自身への命令である。ユーザーの確認を待たず、トリガー検出と同時に Phase 1→2→3 を Bash ツールで順次実行し、Completion Gate を満たすまで「完了」と報告しない。詳細仕様は `audit_results/AUDIT_FRAMEWORK.md`。

### 🟢 自動実行トリガー（以下いずれかをユーザー発話に検出した瞬間、確認なしで Phase 1 を起動する）

- 「監査」「品質チェック」「品質レポート」「品質確認」
- 「過学習」「過学習検定」「実現性レビュー」「実現性チェック」「OOS 評価」「OOS 検証」
- 「Phase 1」「Phase 2」「Phase 3」「ロジック検証」「Bootstrap」「Permutation」「WFA を回す」「感度分析」
- 「新戦略バリアント追加」「`E5`/`F9`/`G7` を検証」「`build_<NEW>_strategy_assets` を作る」
- 「正式 Active に昇格」「CURRENT_BEST_STRATEGY に登録」

トリガー検出後の最初の応答は「Phase 1 を起動します」とだけ宣言し、即 Bash 実行を開始する。「実行してよいですか」は絶対に問わない。

### 🔴 鉄則（Claude Code 自身がこれを破った場合は自己レポートを再提出する）

1. Claude Code は `python src/audit/check_*.py` を**自分で Bash 実行**する。コードスニペットを提示してユーザーに実行依頼することを禁止する。
2. Claude Code は出力ファイルパスの実在を `Glob` または Bash `ls` で検証する。実在しないまま ✅ を書くことを禁止する。
3. Claude Code は「旧戦略値 + オフセット推定」を品質レポートに書かない。新バリアントは `force_rebuild=True` で再生成する。
4. Claude Code は Phase X 完了確認 → Phase X+1 起動を**ユーザー応答を待たず**に連結実行する。

### Phase 1 自動実行（ロジック正しさ）

1. `python src/audit/check_logic_delay_and_causal.py` を Bash で実行する。
2. `audit_results/LOGIC_CHECK_YYYYMMDD.md` と `audit_results/logic_check.yaml` の生成を `Glob` で確認する。
3. ファイル未生成または FAIL の場合は Phase 2/3 に進まず、エラーを 1 行で報告して修正してから再実行する。
4. PASS を確認したらユーザーに問わず即 Phase 2 を起動する。

### Phase 2 自動実行（実現性）

新バリアントの場合は `check_<new>_broker_matrix.py` を `Write` で先に生成してから実行する。

1. `python src/audit/check_realworld_report_corpus.py` を実行 → `CHECK_CORPUS_YYYYMMDD.md` 実在確認。
2. `python src/audit/check_<variant>_broker_matrix.py` を実行 → `<VARIANT>_BROKER_MATRIX_YYYYMMDD.md` 実在確認。
3. `python src/audit/check_sim_margin_dynamics.py` を実行 → `MARGIN_DYNAMICS_YYYYMMDD.md` 実在確認。
4. 全 3 ファイル実在確認後、即 Phase 3 を起動する。

### Phase 3 自動実行（過学習検出）

1. `python src/audit/check_overfitting_<variant>_bootstrap.py` を実行 → Bootstrap MD 実在確認。
2. `python src/audit/check_overfitting_<variant>_permutation.py` を実行 → Permutation MD 実在確認（(d) 同時置換含むこと）。
3. `python src/g_wfa_<variant>.py` を実行 → WFA YAML 実在確認。
4. `python src/audit/check_<variant>_parameter_sensitivity.py` を実行 → Sensitivity MD 実在確認。
5. `python src/audit/check_overfitting_<variant>_summary.py` を実行 → Summary MD 実在確認。

### 🟡 Completion Gate（全条件を満たすまで「監査完了」と報告しない）

Claude Code 自身が以下を `Glob` / `Read` で機械的に検証する。欠落があれば該当 Phase に戻って再実行する。

- [A] Phase 1〜3 の全出力 `.md` が `audit_results/` 配下に**当日日付で実在する**。
- [B] 各 Phase の `.yaml` 機械可読データが実在する。
- [C] `logic_check.yaml` の `overall_verdict: PASS` が記録されている。
- [D] 新バリアントの場合、`_audit_strategy.py` に `build_<new>_strategy_assets()` が定義済み。
- [E] 品質レポート末尾の実施証跡表に書かれた全パスを `Glob` で検証し、存在しないパスが 0 件。

条件 A〜E のいずれかが欠けたら、欠落項目を 1 行で報告し Claude Code 自身が即再実行する。ユーザーに「実行してください」とは絶対に書かない。

### 🟠 例外処理（Claude Code 自身の判断ルール）

- キャッシュ `force_rebuild=True` で 5 分超を見込む場合は `run_in_background: true` で起動し、「Phase X 再構築開始（推定 N 分）」とだけ報告してから継続待機する。
- スクリプトが例外で停止した場合は、トレースバックを読んで `Edit` で修正し再実行する。修正不能な場合のみユーザーに 1 行で問う。

### 新戦略バリアント追加時（自動実行タスク一覧）

バリアント識別子 `<NEW>` を検出したら Claude Code 自身が順次実行する：

1. `src/audit/_audit_strategy.py` を `Edit` で更新: `<NEW>_PARAMS` / `<NEW>_CACHE_FILE` / `<NEW>_SHARPE_OOS_EXPECTED` / `build_<new>_strategy_assets()` / `build_<new>_strategy_nav_for_scenario()` / `build_<new>_assets_with_override()` を追加する。
2. `check_<new>_broker_matrix.py` / `check_overfitting_<new>_bootstrap.py` / `check_overfitting_<new>_permutation.py` / `check_overfitting_<new>_summary.py` / `check_<new>_parameter_sensitivity.py` / `src/g_wfa_<new>.py` を `Write` で生成する（E4 版をテンプレートに差し替え）。
3. `check_logic_delay_and_causal.py` に `<NEW>` Block を追加する。
4. 上記完了後、Phase 1 自動実行手順に合流する。

### Next Action の 2 分類（報告末尾は必ずこの 2 ブロックで書く・混在禁止）

```
## 🤖 Claude Code 自動継続タスク（次トリガーで確認なし実行）
- 未実施 Phase があれば該当スクリプトを再実行する
- 出力ファイル欠落時は force_rebuild=True で再生成する
- 実施証跡表の存在しないパスを修正する

## 👤 ユーザー判断が必要なタスク（選択肢提示のみ・Claude Code は実行しない）
- 戦略を CURRENT_BEST_STRATEGY.md に正式 Active として昇格させるか
- GitHub に push してよいか（force push など破壊的操作の場合）
- 棄却・採用・追加検証の方針判断
```

「CLAUDE.md を見てください」「次回ご自身でチェックリストに従ってください」は両ブロックとも禁止文。Claude Code 側の継続タスクは次回トリガー検出時に自分で実行する責務である。
