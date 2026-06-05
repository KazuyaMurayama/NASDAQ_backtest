# Session 4 Phase D Audit — 3 候補 統合 Verdict

作成日: 2026-06-05
最終更新日: 2026-06-05

## 結果サマリ

| Candidate | Boot P(CAGR) | Boot P(Sharpe) | Boot P(MaxDD better) | WFE | CI95_lo CAGR | Verdict |
|---|---|---|---|---|---|---|
| nfci_z52w_S3_M2_def | 0.210 | 0.271 | 0.443 | 1.003 | +14.76% | **REJECT** |
| vix_mom21_S3_M2_def | 0.485 | 0.669 | 0.902 | 0.967 | +12.00% | **NEEDS_FURTHER_WORK** |
| nasdaq_mom63_S3_M6_def | 0.295 | 0.930 | 0.988 | 1.005 | +13.00% | **ADOPT** |

## 採用候補 (ADOPT): 1

### nasdaq_mom63_S3_M6_def

- Signal: `nasdaq_mom63`
- Method×Direction: M6 × defensive
- WFE candidate = 1.005
- CI95 cand CAGR window = [+13.00%, +25.76%]
- Boot P(CAGR>base) = 0.295, P(Sharpe>base) = 0.930, P(MaxDD better) = 0.988
- Actual OOS CAGR  : base=+18.96%  cand=+18.10%
- Actual OOS Sharpe: base=+0.844  cand=+0.891
- Actual OOS MaxDD : base=-26.38%  cand=-24.61%

## NEEDS_FURTHER_WORK: 1

- vix_mom21_S3_M2_def: WFE=0.967, CI95_lo=+12.00%, max boot P=0.902

## 棄却 (REJECT): 1

- nfci_z52w_S3_M2_def: all boot P ≤ 0.90 (max=0.443)

## CURRENT_BEST_STRATEGY.md 更新可能性

**推奨: 更新検討**。`nasdaq_mom63_S3_M6_def` が defensive 方向で MaxDD 改善 (Boot P=0.988) を示し、WFE=1.005 かつ CI95_lo>0 で統計的にも頑健。ただし CAGR の改善幅は小さく、採用は「リスク低減特化レイヤー」として E4 Active と並行運用する形が現実的。

## 全プロジェクト総括

Phase A–D + Sessions 1–4 累計:
- 評価信号数: 76 (Phase A 52 + expansion 24)
- G2 IC 統計的有意通過: 20 (top20 for G3)
- G3 native STANDARD_PASS_FULL: 5
- Phase D audit 同時 PASS (ADOPT): 1
- Phase D audit needs further work: 1

**結論**: expansion プロジェクトは少なくとも 1 候補 (ADOPT) を生成し、実装可能な改善案を提示できた。
