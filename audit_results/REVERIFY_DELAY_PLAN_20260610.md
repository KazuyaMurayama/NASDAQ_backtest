# 焦点再検証計画 — DELAY修正(レビュー) vs 初版、どちらが正しいか

作成日: 2026-06-10
最終更新日: 2026-06-10

> **目的**: レビューで結論が覆った核心（realistic経路の `_DELAY=1`→`2` 修正）について、**修正が本当に正しいのか／初版(DELAY=1)が正しかったのか**を、思い込みなく数値で決着させる。報告のみ・正典未変更。

## 何が争点か
- scenarioD は `build_nav_strategy`（`cfd_leverage_backtest.py`）内部で `DELAY=2`（`corrected_strategy_backtest.DELAY`）を使う。
- realistic 経路（audit独自実装）は初版 `_DELAY=1`、修正版 `_DELAY=2`。
- 初版「ETF/投信>CFD 逆転」はこの差で生じた。**どちらの DELAY が scenarioD と同じ時間規約か**が全て。

## 検証タスク（実行する）

### V1. DELAY 意味論の特定（コード）
`build_nav_strategy`（`cfd_leverage_backtest.py`）と `corrected_strategy_backtest.py` の `DELAY` 適用箇所を読み、**シグナル→建玉のラグが具体的に何営業日 shift か**を確定（`shift(DELAY)` か `shift(DELAY-1)` か等、実装の正確なオフセット）。EVALUATION_STANDARD §2.1（DELAY=2）と整合するか。

### V2. コスト分離 round-trip テスト（決定的）
realistic ビルダー（`_build_nav_realistic` / `_build_nav_vz065_realistic`）のコストを **scenarioD と完全同一**に設定する:
- CFDスプレッド = `CFD_SPREAD_LOW = 0.0020`
- 財務 = `(L-1)×(sofr_daily + 0.0020/252)`（borrowed、scenarioD式）
- TER/配分/その他 = scenarioD と同一
- DELAY = 2

この条件で realistic ビルダーが生成する**日次NAVを scenarioD の NAV と1営業日ごとに diff**。
- **合格条件**: 日次NAV最大絶対差 < 1e-6（または10指標が ±0.01pp/±0.001 以内一致）。
  → realistic経路は構造的にscenarioDと等価＝**DELAY=2修正は正しい**。realistic化で変わるのは「コストだけ」と確定。
- 同テストを **DELAY=1** でも実施し、**不一致（1日ズレ）になること**を示す → 初版DELAY=1がバグだった直接証拠。

### V3. spread_cost 修正の正しさ
修正後の `pos_change = |Δ(wn_s×lev_s×L_shifted)|` が、正典 `g18_daily_trade_cost_wfa.py` の `pos = wn_a*lm*L_s2; dpos=|diff(pos)|` と**式・変数対応が一致**するか確認。二重計上・取りこぼしがないか。

### V4. 訂正値のクロスチェック
- E4 realistic(full L×, DELAY=2) = +21.83% が、リポジトリの SBI CFD 3.0% 結果（`g14_wfa_sbi_cfd_summary.csv` / `7STRATEGY_PERFORMANCE_REPORT_20260529.md`、+22.4%）と整合する理由・残差を説明。
- 初版(+15.35%)との差 −6.5pp が DELAY のみで説明できるか（V2のDELAY=1再現と一致するか）。

## 判定基準
| 結果 | 結論 |
|---|---|
| V2: DELAY=2でscenarioD完全再現 ∧ DELAY=1で不一致 | **レビュー(修正DELAY=2)が正しい**。初版DELAY=1はバグ確定 |
| V2: DELAY=1でscenarioD完全再現 ∧ DELAY=2で不一致 | 初版が正しくレビューが誤り（要再訂正） |
| V2: どちらでも不一致 | realistic経路に別の構造差。原因特定し再設計 |

## 成果物
- 検証スクリプト/診断: `src/audit/`（新規, push）
- 結果: 本ファイルに追記 or `audit_results/REVERIFY_DELAY_RESULT_20260610.md`（push）

## モデル使い分け
- 実行・判定: 敵対的レビュー（Opus サブエージェント）が自分の前回主張も含め批判的に再検証。
- 統合・最終判定: Fable（メイン）。

---

*管理者: 男座員也（Kazuya Oza）*
