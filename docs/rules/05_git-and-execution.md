# 05. Git操作 + 実行計画

## 🌿 Git操作

### ❌ 禁止
- **ブランチ作成禁止**（Claude Code のセッション切替時に読み損ねるため）
- 禁止コマンド: `git checkout -b` / `git switch -c` / `git branch <name>`
- ユーザー明示的指示なき限り、現在ブランチの変更禁止

### ✅ 許可
- 現在ブランチ上の `git add` / `git commit` / `git push`
- 読み取り系: `git status` / `git log` / `git diff`
- `git pull` による最新化

## 🚀 実行計画・タイムアウト対策
- タスクを細かいサブタスクに分割、各ステップで保存・コミット
- 長時間処理が見込まれる場合、**チェックポイントを明示的に計画**
- タイムアウト原因を予測し対策、妥当性もチェック

### サブエージェント利用時の失敗学習（2026-04-19）
- **1 subagent = 1 push_files call**（tool uses を最小化）
- subagent には **fetch させない**（content を事前定義して手渡す）
- get_file_contents の多用はタイムアウトの主原因
