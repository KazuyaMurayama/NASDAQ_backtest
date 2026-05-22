# 08. 評価指標ルール（厳守・ユーザー指示なし変更禁止）

> **このリポジトリにおける戦略評価・比較・WFA・パラメータスイープ・ベスト戦略更新の全場面で、以下の「統一9指標」のみを使用すること。**
> **ユーザーからの明示的な指示がない限り、これ以外の指標で評価・ランキング・優劣判定を行ってはならない。**

- バージョン: v1.0（本ルールファイル）
- 準拠する正典: [`EVALUATION_STANDARD.md`](../../EVALUATION_STANDARD.md) **v1.1 §3**
- 発効日: 2026-05-22
- 関連実装: [`src/g1_wfa.py`](../../src/g1_wfa.py) (`compute_summary_stats`)

---

## §1 統一9指標（これ以外による評価は禁止）

### §1.1 標準7指標（全戦略で必須）

| # | 指標 | 計算式参照 | 用途 |
|---|---|---|---|
| 1 | **CAGR_IS / CAGR_OOS / CAGR_FULL** | [EVALUATION_STANDARD §3.1](../../EVALUATION_STANDARD.md#§3.1) | 区間別年率複利成長率（OOS が最重要） |
| 2 | **Sharpe_OOS**（Rf=0） | [EVALUATION_STANDARD §3.2](../../EVALUATION_STANDARD.md#§3.2) | リスク調整後リターン（高金利期の過大評価注意必須） |
| 3 | **MaxDD**（FULL） | [EVALUATION_STANDARD §3.3](../../EVALUATION_STANDARD.md#§3.3) | 最大ドローダウン（FULL 期間で算出） |
| 4 | **Worst10Y★**（カレンダー年） | [EVALUATION_STANDARD §3.5](../../EVALUATION_STANDARD.md#§3.5) | カレンダー年ベース10年ローリング最悪 CAGR |
| 5 | **P10_5Y▷** | [EVALUATION_STANDARD §3.6](../../EVALUATION_STANDARD.md#§3.6) | 5年CAGR 分布の第10パーセンタイル |
| 6 | **IS-OOS gap** | [EVALUATION_STANDARD §3.8](../../EVALUATION_STANDARD.md#§3.8) | 過剰適合検出（CAGR_IS − CAGR_OOS, pp） |
| 7 | **Trades/yr** | [EVALUATION_STANDARD §3.7](../../EVALUATION_STANDARD.md#§3.7) | 年間リバランス回数（税ドラッグ算定の根拠） |

### §1.2 WFA補助2指標（Walk-Forward Analysis 実施時に必須）

| # | 指標 | 計算式参照 | 判定基準 |
|---|---|---|---|
| 8 | **WFA_CI95_lo** | [EVALUATION_STANDARD §3.9](../../EVALUATION_STANDARD.md#§3.9) | 非重複1年窓 CAGR の t 分布95%CI下限。**正値 ⇒ α基準合格** |
| 9 | **WFA_WFE** | [EVALUATION_STANDARD §3.10](../../EVALUATION_STANDARD.md#§3.10) | Walk-Forward Efficiency。**0.5 ≤ WFE ≤ 2.0 ⇒ β基準合格** |

---

## §2 廃止・禁止指標（ユーザー明示指示なしに使用禁止）

以下の指標は **v1.1 で廃止**された。これらを用いた評価・ランキング・優劣判定はすべて「非標準」とみなす：

| 廃止指標 | 廃止理由 |
|---|---|
| **Stable_Sharpe** | 統一指標セットから除外（v1.1） |
| **WinRate_yr** | 統一指標セットから除外（v1.1） |
| **WorstK5_mean_CAGR** | Worst10Y★ / P10_5Y▷ に統合済み（v1.1） |
| **IR_vs_BH** | ベンチマーク依存で頑健性不足（v1.1） |

- 上記指標を含むレポート・スクリプト・比較表は **`EVALUATION_STANDARD.md` §6 の参考値判定フロー** が適用される
- ユーザーから明示的に「Stable_Sharpe で比較せよ」等の指示があった場合のみ使用可。その場合も**「非標準・参考値」とラベル付け必須**

---

## §3 適用タイミング（全場面で適用）

以下のすべての場面で本ルールが**自動適用**される。Claude は本ルールを毎回チェックすること：

1. **戦略比較**（複数戦略の優劣判定・ランキング表作成）
2. **WFA 実施**（Walk-Forward Analysis の summary stats 出力）
3. **パラメータスイープ**（l_max, n_vol, k_vz 等の感度分析）
4. **ベスト戦略更新**（`CURRENT_BEST_STRATEGY.md` の差し替え判定）
5. **新規戦略の採用判定**（`STRATEGY_REGISTRY.md` への登録時の Status 決定）
6. **再計算・OOS 延長後の再評価**

---

## §4 違反時の扱い

本ルールに違反した評価（=9指標以外による比較・廃止指標使用・指標欠落）は以下のとおり扱う：

1. **当該レポート / 比較結果は「非標準（参考値）」扱い**
   → `EVALUATION_STANDARD.md` §6 の参考値判定フローを適用
2. **`CURRENT_BEST_STRATEGY.md` の更新根拠にできない**
3. **Claude は違反を検出した場合、即座にユーザーへ報告し、9指標で再計算する案を提示**すること
4. **ユーザーが明示的に「9指標以外で評価せよ」と指示した場合のみ例外**。その場合も「非標準」ラベルを付ける

---

## §5 検証スクリプトでの明示義務

新規・既存問わず、戦略検証スクリプトの冒頭コメントには以下を必ず記載：

```python
# Evaluation Standard: v1.1
# Cost Scenario: D （src/product_costs.py 2026-05-12 基準）
# IS: 1974-01-02〜2021-05-07 / OOS: 2021-05-08〜現在
# Metrics: 標準7 + WFA補助2 (docs/rules/08_evaluation-metrics.md)
```

レポート（.md）の必須ヘッダにも `EVALUATION_STANDARD バージョン: v1.1` を明記する（[`EVALUATION_STANDARD.md` §5.2](../../EVALUATION_STANDARD.md#§5.2) 参照）。

---

## §6 セルフチェックリスト（評価実施前に必ず確認）

- [ ] CAGR_IS / CAGR_OOS / CAGR_FULL を全て出力するか？
- [ ] Sharpe_OOS（Rf=0）を出力するか？
- [ ] MaxDD（FULL）を出力するか？
- [ ] Worst10Y★（カレンダー年）を出力するか？
- [ ] P10_5Y▷ を出力するか？
- [ ] IS-OOS gap を出力するか？
- [ ] Trades/yr を記録するか？
- [ ] WFA 実施時: WFA_CI95_lo と WFA_WFE を追加するか？
- [ ] **禁止指標（Stable_Sharpe / WinRate_yr / WorstK5_mean_CAGR / IR_vs_BH）を使っていないか？**
- [ ] レポート / スクリプトに `Evaluation Standard: v1.1` を明記したか？

---

## §7 改訂履歴

| 版 | 日付 | 変更内容 |
|---|---|---|
| v1.0 | 2026-05-22 | 初版発行。`EVALUATION_STANDARD.md` v1.1 §3 と同期。統一9指標を厳守ルール化、廃止4指標の使用禁止を明文化。 |

---

*本ルールは `EVALUATION_STANDARD.md` v1.1 を一次根拠とする。両者の整合性が常に保たれていることを Claude / 人間ともに確認すること。*
