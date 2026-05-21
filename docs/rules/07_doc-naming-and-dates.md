# 07. ドキュメント命名・日付ルール

## 命名規則（再発防止）

新規レポート作成時:

1. **`FINAL_` プレフィックスは禁止** — 「FINAL」と名乗ったファイルが後で覆されると参照地獄になる
2. **`<TOPIC>_YYYY-MM-DD.md` または `REPORT_YYYY-MM-DD.md` 形式**を使用
3. **旧レポートを置き換える時は、必ず旧レポート冒頭に SUPERSEDED ヘッダを追加**:
   ```markdown
   > ⛔ **このドキュメントは SUPERSEDED (置換済み) です**
   > - 廃止日: YYYY-MM-DD
   > - 後継ファイル: [新レポート名](新レポート.md)
   > - 現行ベスト戦略: [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)
   > - 廃止理由: <一行で理由>
   ```
4. **CURRENT_BEST_STRATEGY.md と tasks.md を同時更新**

## 日付ルール

レポート・分析・調査系 .md ファイルを新規作成する際は、H1直下に必ず記載:

```
作成日: YYYY-MM-DD
最終更新日: YYYY-MM-DD
```

- 更新時は **最終更新日のみ** を当日付に書き換える（作成日は固定）
- 除外: README / CLAUDE.md / FILE_INDEX / tasks.md / CHANGELOG / LICENSE
