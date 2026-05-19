# P5 ブロックブートストラップ・ストレステスト

作成日: 2026-05-18
最終更新日: 2026-05-18

## 概要

P4でGRAY判定を受けたDyn系3コンボ (P01/P02/P03) に対して、
Stationary Block Bootstrap により OOS期間のサンプル感度を評価。
ADOPT / GRAY / REJECT を確定する。

- OOS期間: 2021-05-08 〜 2026-03-26 (≈1250日)
- P4 REJECT コンボ (Baseline/P05/P06) はREJECT維持を確認

## 手法

| 項目 | 設定 |
|---|---|
| 手法 | Stationary Block Bootstrap (Politis-Romano 1994) |
| ブロック平均長 | L = 21日 (月次) |
| 反復数 | B = 2000 |
| 対象期間 | OOS (2021-05-08〜2026-03-26) |
| シード | 42 |
| Paired Bootstrap | 同一インデックス系列でBaseline比較 |

## 判定基準

| 基準 | 閾値 |
|---|---|
| C1: CAGR 5%ile | ≥ 10% |
| C2: CAGR median | ≥ 15% |
| C3: Sharpe 5%ile | ≥ 0.3 |
| C4: Sharpe median | ≥ 0.5 |
| C5: pct_above (CAGR ≥ 15%) | ≥ 50% |
| C6: pct_above (Sharpe ≥ 0.5) | ≥ 60% |
| C7: MaxDD 95%ile | ≥ -60% |
| C8: ΔSharpe vs Baseline 5%ile | > 0 |
| C9: p_value (Δ ≤ 0) | < 0.10 |

## 結果テーブル — 絶対基準

| コンボ | CAGR_med | CAGR_p05 | SR_med | SR_p05 | MaxDD_p95 | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | 16.2% | -3.9% | 0.686 | -0.011 | -23.8% | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| P01_Dyn×HY | 20.2% | 1.1% | 0.837 | 0.176 | -20.4% | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| P02_Dyn×CPI | 19.6% | 1.3% | 0.841 | 0.176 | -19.8% | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| P03_Dyn×MA | 19.2% | -0.0% | 0.807 | 0.132 | -20.7% | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| P05_HY×CPI | 14.9% | -5.1% | 0.634 | -0.051 | -24.6% | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |
| P06_HY×MA | 13.9% | -7.0% | 0.595 | -0.106 | -25.9% | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |

## 結果テーブル — 相対基準 (Baseline比)

| コンボ | ΔSR_p05 | ΔSR_med | p_val(Δ≤0) | C8 | C9 |
|---|---:|---:|---:|:---:|:---:|
| Baseline | N/A | N/A | N/A | ✗ | ✗ |
| P01_Dyn×HY | -0.034 | 0.150 | 0.091 | ✗ | ✓ |
| P02_Dyn×CPI | -0.031 | 0.157 | 0.088 | ✗ | ✓ |
| P03_Dyn×MA | -0.050 | 0.122 | 0.133 | ✗ | ✗ |
| P05_HY×CPI | -0.116 | -0.035 | 0.773 | ✗ | ✗ |
| P06_HY×MA | -0.174 | -0.080 | 0.950 | ✗ | ✗ |

## 最終判定

| コンボ | P4_status | P5_verdict | 最終判定 |
|---|---|---|---|
| Baseline | REJECT | MARGINAL | **REJECT** |
| P01_Dyn×HY | GRAY | MARGINAL | **GRAY** |
| P02_Dyn×CPI | GRAY | MARGINAL | **GRAY** |
| P03_Dyn×MA | GRAY | MARGINAL | **GRAY** |
| P05_HY×CPI | REJECT | MARGINAL | **REJECT** |
| P06_HY×MA | REJECT | FRAGILE | **REJECT** |

## 投資判断サマリー

**GRAY維持候補**: P01_Dyn×HY, P02_Dyn×CPI, P03_Dyn×MA

P5でMARGINAL判定。絶対水準は一定の安定性を示すが、Baseline比優位性が統計的に十分でなく、実運用採用は慎重に判断すること。
**REJECT確定候補**: Baseline, P05_HY×CPI, P06_HY×MA

P4 REJECT維持またはP5でFRAGILE。ブートストラップ分布が採用基準を満たず、現行DH Dyn [A] Baselineを維持する。
