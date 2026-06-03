# 09. 統合比較レポート標準（採用判断・戦略比較）

- バージョン: v1.0
- 発効日: 2026-06-03
- 準拠: `STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md` §0'/§1'/§5/§6

> 採用判断を目的とする統合比較レポートは **§0' / §1' / §5 / §6** の4セクション必須。
> 1つでも欠けると再提出対象（`CLAUDE.md §3 rule 7` 参照）。

---

## §1 適用トリガー

以下のいずれかに該当するとき、本ルールが**自動適用**される:

| トリガー | 例 |
|---|---|
| 戦略採用判断レポートを作成・更新するとき | `STRATEGY_PERFORMANCE_INTEGRATED_*` 系ファイル |
| 複数戦略の Active/候補/棄却 判定をするとき | `CURRENT_BEST_STRATEGY.md` 更新根拠レポート |
| OOS期間延長後の全戦略再評価レポートを作成するとき | OOS延長 re-evaluation |

**適用外（sweep/gridスキャン）**: `STRATEGY_COMPARISON_*` 系・sweep MD は §0'（10列）のみ必須（既存 `CLAUDE.md §3` rules 1-6）。

---

## §2 必須4セクション（毎回・例外なし）

| セクション | 名称 | フォーマッタ/テンプレート |
|---|---|---|
| **§0'** | 候補戦略 統合比較表（11列・累積CAGR⓽列含む） | `MD_HEADER_INTEGRATED` / `fmt_row_integrated()` |
| **§1'** | コスト・税金調整前提（14ステップ） | 静的テンプレート（§4 参照） |
| **§5** | 年次リターン表（1977-2026 / moderate / 税後） | `fmt_annual_table()` |
| **§6** | 統計サマリ（1974-2026 / moderate / 7行） | `fmt_stats_table()` |

---

## §3 §0' 列定義（v6.1 / v4.2 累積CAGR列追加 / 11列）

列順（左→右）:
`Strategy → CAGR⓽_OOS → 累積CAGR⓽ OOS/IS → IS-OOS gap → Sharpe ⓒ → MaxDD ⓒ → Worst10Y★⓽ → P10⓽ 5Y▷ → Trade ⓞ → Overfit ⓞ → CI95 ⓡ`

| 列 | 計算基準 | 記号 |
|---|---|---|
| CAGR⓽_OOS | 年率複利 / 税後・moderate コスト後 | ⓽ |
| 累積CAGR⓽ OOS/IS | `Π(1+yr_aft)→CAGR` / IS(1977-2020)/OOS(2021-2026)を並列表示 / 税後 | ⓽ |
| IS-OOS gap | `CAGR_IS_aft − CAGR_OOS_aft`（pp）/ 税後・moderate | ⓒ |
| Sharpe ⓒ_OOS | Rf=0 / コスト後（税前据置） | ⓒ |
| MaxDD ⓒ | コスト後（税前） | ⓒ |
| Worst10Y★⓽ | 最悪10年ローリング CAGR / 税後 | ⓽ |
| P10⓽ 5Y▷ | 5年CAGR 第10パーセンタイル / 税後 | ⓽ |
| Trade ⓞ (回/年) | 年間リバランス回数（原値） | ⓞ |
| Overfit ⓞ (WFE) | WFE判定 / ✅LOW=0.5-2.0 / ⚠MED>2.0 / ❌HIGH<0.5 | ⓞ |
| CI95ⓡ_lo | WFA CI95下限 + §3-A 税調整後 | ⓡ |

**記号凡例**:
- ⓽ = 税後（手取り）
- ⓒ = コスト後（税引前据置）
- ⓞ = 原値（コスト・税で不変）
- ⓡ = 再計算値（WFA実測 + §3-A税調整）

**moderate ケース定義**:
- 日次取引コスト: CFD spread 0.05%（往復スプレッド）
- 税: `CAGR_net = (CAGR − 0.66%) × 0.8273` 逐年複利（§3-A モデル）

---

## §4 §1' コスト・税金調整前提テンプレート（静的）

§1' は以下のファイルから**そのままコピー**する（値変更はユーザー承認後のみ）:

**参照元**: [`STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md`](../../STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) §1' セクション

コピー対象（3サブセクション）:
- §1'-1 メイン調整テーブル（全14ステップ）— Step 1〜Step 5 + MaxDD/Sharpe/IS-OOS gap/Trades/Overfit/CI95/約定遅延/税適用タイミングの定義
- §1'-2 ETF戦略コスト前提（DH Dyn [A] のみ — TQQQ/TMF/2036 TER・借入金利・コミッション）
- §1'-3 取引コストケース設定（CFD 4ケース: measured 0.020% / opt 0.030% / **moderate 0.050%** / cons 0.100%）

> §1' の値を変更する場合は必ずユーザーに変更内容を確認してから更新する。

---

## §5 §5 年次リターン表の要件

- **期間**: 1977-2026（1974-1976は LT2-N750 ウォームアップ期間のため除外）
- **コスト・税**: moderate ケース（§3 定義と同一）
- **2026年**: 部分年（〜3月末）— 年率換算なし、実績パーセンテージのみ記載
- **データ出典**: `g20f_unified_yearly_returns.csv` 等の日次 NAV から算出した年次リターン
- **必須列**: 比較対象の全戦略 + ベンチマーク（NDX 1x B&H）を最終列に

### 出力フォーマット

```python
from _sweep_format import fmt_annual_table

# yearly_returns[strategy_name][year] = return (比率, 税後・moderate コスト後)
md_table = fmt_annual_table(
    strategies=['NEW 🟢', 'D5 v0.65/lmax=5.5', 'DH [A]', 'NDX 1x B&H'],
    yearly_returns=yearly_returns,
    start_year=1977,
)
```

---

## §6 §6 統計サマリの要件

- **期間**: 1974-2026（§5 と異なり IS 期間全体を含む）
- **7行**: 平均(mean) / 中央値(median) / 標準偏差(std) / 最大(max) / 最小(min) / プラス年数 / マイナス年数
- **コスト・税**: moderate ケース（§5 と同一）
- **必須列**: 比較対象の全戦略 + ベンチマーク（NDX 1x B&H）

### 出力フォーマット

```python
from _sweep_format import fmt_stats_table

md_table = fmt_stats_table(
    strategies=['NEW 🟢', 'D5 v0.65/lmax=5.5', 'DH [A]', 'NDX 1x B&H'],
    yearly_returns=yearly_returns,
    start_year=1974,
)
```

---

## §7 フォーマッタ import 一覧

統合比較レポートで使用するフォーマッタはすべて `src/_sweep_format.py` から import する（手書きヘッダ禁止）:

```python
from _sweep_format import (
    MD_HEADER_INTEGRATED,   # §0' 11列ヘッダ
    fmt_row_integrated,     # §0' 1戦略1行
    fmt_annual_table,       # §5 年次リターン表
    fmt_stats_table,        # §6 統計サマリ
)
```

---

## §8 セルフチェックリスト（報告前に全項目確認）

- [ ] §0' が `MD_HEADER_INTEGRATED` / `fmt_row_integrated` を使用しているか？
- [ ] §0' に累積CAGR⓽列（OOS/IS 両値を1セルで並列表示）が含まれているか？
- [ ] §1' の14ステップテーブルが含まれているか（§1'-1/§1'-2/§1'-3の3サブセクション）？
- [ ] §5 年次リターン表（1977-2026 / moderate / 税後）が含まれているか？
- [ ] §6 統計サマリ（1974-2026 / moderate / 7行）が含まれているか？
- [ ] §5/§6 にベンチマーク（NDX 1x B&H）列が含まれているか？

---

## §9 改訂履歴

| 版 | 日付 | 変更内容 |
|---|---|---|
| v1.0 | 2026-06-03 | 初版。4セクション（§0'/§1'/§5/§6）必須化。`MD_HEADER_INTEGRATED`（11列・累積CAGR⓽）/ `fmt_annual_table` / `fmt_stats_table` 定義 |
