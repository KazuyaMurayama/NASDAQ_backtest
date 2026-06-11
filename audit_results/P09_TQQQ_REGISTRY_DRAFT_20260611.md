# P09_TQQQ — STRATEGY_REGISTRY §2 Shortlist エントリ案（DRAFT・登録未実施）

作成日: 2026-06-11
最終更新日: 2026-06-11

> **本ファイルは STRATEGY_REGISTRY.md §2 への追加候補の「下書き」である。**
> **registry 本体は編集していない。Active 昇格はユーザー判断待ち。**
> 一次根拠スクリプト: [`src/audit/p09_tqqq_ggate_20260611.py`](../src/audit/p09_tqqq_ggate_20260611.py)
> 一次根拠データ: [`audit_results/p09_tqqq_ggate_20260611.csv`](p09_tqqq_ggate_20260611.csv)

---

## エントリ案

| 項目 | 内容 |
|---|---|
| **名称** | P09_TQQQ |
| **構成** | DH-W1 (Asymm+Hyst) + mom63 V7 boost `{q0:1.20, q1:1.10, q2:1.00, q3:1.00}` + Bond-timing 条件付き OUT-fill（Gold 常時 + Bond は `bond_mom252>0` の時のみ、inverse-vol W63）。NASDAQ IN レッグは **TQQQ ETF コスト**（`L·r_nas − max(L−1,0)·(SOFR+swap/252) − TER_TQQQ/252`）で計上 |
| **環境** | **ETF only / NISA**（NASDAQ レッグを TQQQ で運用する前提。CFD 環境ではない） |
| **OUT-fill ラグ** | T+5（fund lag, LAG_DAYS=5）|
| **コスト** | Scenario D（`src/product_costs.py`）。TQQQ TER 0.86% + SOFR×2.0 + swap 0.50% |
| **status** | **Shortlist（Active 昇格はユーザー判断待ち）** |

---

## 標準10指標（税後 ×0.8273：CAGR / Worst10Y / P10。Sharpe / MaxDD は税引き前）

| # | 指標 | P09_TQQQ | baseline TQQQ-V7 | 差分 |
|---|---|---:|---:|---:|
| 1 | CAGR_OOS (税後) | +17.51% | +16.80% | +0.71pp |
| – | CAGR_IS (税後) | +18.84% | +16.27% | +2.57pp |
| 2 | **min(IS,OOS) CAGR (税後)** | **+17.51%** | +16.27% | **+1.24pp** |
| 3 | IS-OOS gap CAGR | +1.33pp | −0.53pp | — |
| 4 | Sharpe_OOS (税前) | 0.901 | 0.877 | +0.024 |
| 5 | MaxDD FULL (税前) | −35.18% | −34.47% | −0.71pp（悪化）|
| 6 | Worst10Y★ CAGR (税後) | （ggate 未出力・validate CSV 参照）| — | — |
| 7 | P10_5Y▷ CAGR (税後) | （同上）| — | — |
| 8 | Trades/yr | 29.2 | 25.2 | +4.0 |
| 9 | Overfit(WFE) | ✅ LOW (1.02) | — | — |
| 10 | WFA_CI95_lo (税後) | +17.94% | — | — |

> min(IS,OOS) +17.51% は ETF only 環境の現行 Active 候補 DH-W1 (+13.66%, EVAL_STD §3.13) を上回る。

---

## G-gate / Phase-D ロバストネス結果（10,000 ブートストラップ・block 21d / canonical WFA / permutation）

| ゲート | 閾値 | 実測値 | 判定 |
|---|---|---:|:---:|
| Bootstrap P(min(IS,OOS) CAGR better) | > 0.90 | 0.797 | ❌ FAIL |
| Bootstrap P(Sharpe_OOS better) | > 0.90 | 0.586 | ❌ FAIL |
| Bootstrap P(MaxDD better) | > 0.90 | 0.604 | ❌ FAIL |
| WFA α: CI95_lo > 0 AND t_p < 0.05 | CI95_lo>0, t_p<0.05 | +17.94%, t_p≈0 | ✅ PASS |
| WFA β: WFE ∈ [0.5, 2.0] | 0.5–2.0 | 1.017 | ✅ PASS |
| Permutation（OUT-fill mask block-shuffle）| p < 0.05 | p≈0.000 | ✅ PASS |
| **総合** | 全ゲート PASS | **3/6 PASS** | ❌ **FAIL** |

**Bootstrap CI95_lo（ペア差・参考）**: min CAGR −3.66pp / Sharpe −0.222 / MaxDD −7.52pp（いずれも負 → ベースライン超過が統計的に確定しない）。

---

## 解釈・注記

1. **P09_TQQQ は「単独戦略」としては頑健**：WFA α/β PASS（CI95_lo +17.94%・t_p≈0・WFE 1.017）で過学習兆候なし、permutation p≈0 で OUT-fill タイミングが非ランダム（真のシグナル）であることを確認。
2. **ただし「ベースライン TQQQ-V7 に対する増分優位」は頑健でない**：3 つの paired block bootstrap すべてで P(better) が 0.90 を大きく下回り（0.797/0.586/0.604）、CI95_lo が 3 指標とも負。OUT-fill 上乗せの平均効果は正（min +1.61pp）だが、ばらつきが大きく統計的に確定しない。MaxDD はむしろ −0.71pp 悪化。
3. **結論**: Active 昇格は **時期尚早**。現状は **Shortlist 据え置き**が妥当。昇格には ① ベースライン比 bootstrap の改善（fill レシピ最適化 or OUT 日の選別強化）か、② 「ベースライン超過」ではなく「絶対性能 + WFA + permutation」で昇格判断する基準への合意、のいずれかが必要。
4. **LU1（aggressive-leverage variant）について**: LU1 は V7 boost を `{1.40,1.20,1.05,1.00}` に強化した攻め型派生（min(IS,OOS) +18.12%, MaxDD −34.95%）。**lead ではない**。リスク許容度が高いユーザー向けの参考variant にとどめ、本 Shortlist エントリの主候補は P09_TQQQ とする。

---

## 改訂履歴

| 日付 | 変更 |
|---|---|
| 2026-06-11 | 初版。G-gate 3/6 PASS（WFA α/β + permutation PASS、bootstrap 3 指標 FAIL）。Shortlist 据え置き提案。|
