# 07. ドキュメント命名・日付ルール（v2.0 / 2026-06-03 改訂）

## 命名規則（再発防止）

新規レポート作成時:

1. **`FINAL_` プレフィックスは禁止** — 「FINAL」と名乗ったファイルが後で覆されると参照地獄になる
2. **`<TOPIC>_YYYYMMDD.md` 形式**を使用（**サフィックス・ハイフンなし**）
   - 例: `STRATEGY_COMPARISON_20260603.md`、`WFA_REPORT_20260603.md`
3. **旧レポートを置き換える時は、必ず旧レポート冒頭に SUPERSEDED ヘッダを追加**:
   ```markdown
   > ⛔ **このドキュメントは SUPERSEDED (置換済み) です**
   > - 廃止日: YYYY-MM-DD
   > - 後継ファイル: [新レポート名](新レポート_YYYYMMDD.md)
   > - 現行ベスト戦略: [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)
   > - 廃止理由: <一行で理由>
   ```
4. **`CURRENT_BEST_STRATEGY.md` と `tasks.md` を同時更新**

## 同日中の追加更新（バージョン管理）

同日中に**実質的に別版**として残す場合（例: 大幅な誤り修正・新発見を反映した再生成）:

- 1回目: `<TOPIC>_20260603.md`
- 2回目: `<TOPIC>_20260603-v2.md`
- 3回目: `<TOPIC>_20260603-v3.md`

**日付が変わったら v サフィックスはリセット**:
- 翌日1回目: `<TOPIC>_20260604.md`（v なし）
- 翌日2回目: `<TOPIC>_20260604-v2.md`

> ⚠️ **同日中の軽微な修正は上書き＋H1直下「最終更新日」更新で対応**。`-v2` を作るのは、後から両版を比較・参照できる必要がある場合のみ。

## 表記の区別（ファイル名 vs 本文）

- **ファイル名**: ハイフン **なし** `YYYYMMDD`（例: `20260603`）
- **本文中の日付表記**: ハイフン **あり** `YYYY-MM-DD`（例: `2026-06-03`）

## 旧形式（廃止・新規禁止）

- ❌ `2026-06-03_STRATEGY_COMPARISON.md`（プレフィックス・ハイフン）
- ❌ `STRATEGY_COMPARISON_2026-06-03.md`（サフィックス・ハイフン）— **旧 v1.0 形式**
- ❌ `FINAL_STRATEGY_COMPARISON.md`（FINAL プレフィックス）
- ✅ `STRATEGY_COMPARISON_20260603.md`（**現行 v2.0 ルール**）

> 既存の v1.0 形式ファイル（`<TOPIC>_2026-06-03.md` 等）は遡及リネーム不要。新規作成からのみ v2.0 形式に従う。

## 対象外（日付サフィックスを入れない）

- README.md / CLAUDE.md / FILE_INDEX.md / tasks.md / CHANGELOG.md / LICENSE.md
- `EVALUATION_STANDARD.md`、`SPEC.md`、`HYPOTHESES.md` 等の仕様書系
- `CURRENT_BEST_STRATEGY.md`（常に最新で参照される単一ファイル）
- パイプライン自動生成ファイル（例: `R*_results.csv` / `audit_results/*` / `outputs/*`）

## 日付メタデータ（H1直下）

レポート・分析・調査系 .md ファイルを新規作成する際は、H1直下に必ず記載:

```
作成日: YYYY-MM-DD
最終更新日: YYYY-MM-DD
```

- 更新時は **最終更新日のみ** を当日付に書き換える（作成日は固定）
- 除外: README / CLAUDE.md / FILE_INDEX / tasks.md / CHANGELOG / LICENSE
