# Git 操作ルール

## ❌ 禁止事項
- **ブランチ作成は一切禁止**（セッションをまたぐと読み取りにくいため）
  - 禁止コマンド: `git checkout -b` / `git switch -c` / `git branch <name>`
- ユーザーから明示的な指示がない限り、現在のブランチを変更しない。
- force push・reset --hard 等の破壊的操作は明示的指示がない限り禁止。

## ✅ 許可操作
- 現在のブランチ上での `git add`, `git commit`, `git push`
- 読み取り操作: `git status`, `git log`, `git diff`, `git branch`
- `git pull` による最新化

## プッシュ手順
```bash
git push -u origin <current-branch>
```
- ネットワークエラー時は指数バックオフで最大4回リトライ（2s, 4s, 8s, 16s）

## コミットメッセージ
- 変更の「なぜ」を1〜2文で記述
- フッターにセッションURLを含める
