# Phase 3: 過学習検出 統合サマリ

- 生成日: 2026-05-24
- 戦略: S2_VZGated + LT2-N750-k0.5-modeB
- OOS 期間: 2021-05-08 〜 2026-03-26

## 検定結果一覧

| # | 検定 | 主要指標 | 値 | 判定 |
|---|---|---|---|---|
| 3.1 | DSR (N=500, 保守) | DSR | 0.0000 | FAIL† |
| 3.1 | PSR (SR_b=0, 単試行) | PSR | 0.9686 | PASS |
| 3.1 | PSR (SR_b=0.5, 業界目安) | PSR | 0.7812 | WARN |
| 3.2 | Block Bootstrap (L=63) | CI95_lo (SR年率) | 0.0859 | PASS |
| 3.2 | Bootstrap p値 (H0: SR=0) | p値 | 0.0160 | PASS |
| 3.3 | Permutation: L_s2 block | p値 | 0.2480 | FAIL |
| 3.3 | Permutation: lev_mod block | p値 | 0.0550 | WARN |

†DSR は N=500 試行の多重比較補正込み。E[max SR]≈2.88 に対して観測 SR=0.858 のため
 保守的評価では FAIL。PSR / Bootstrap / Permutation を主判定基準とする。

## 総合判定（Bootstrap + Permutation 基準）: **FAIL**

### PASS 条件
- Bootstrap (L=63): CI95_lo = 0.0859 > 0 かつ p = 0.0160 < 0.05 → **PASS**
- Permutation (a) L_s2 block: p = 0.2480 < 0.05 → **FAIL**
- Permutation (c) lev_mod block: p = 0.0550 < 0.05 → **WARN**

## 解釈

Bootstrap または Permutation が統計的有意水準を満たしていない。
戦略の信頼性について再評価が必要。

## CURRENT_BEST_STRATEGY.md へのパッチ提案

以下を CURRENT_BEST_STRATEGY.md の「検証ステータス」セクションに追記してください（手動マージ）:

```markdown
### Phase 3 過学習検出（2026-05-24）
- Block Bootstrap (L=63): CI95_lo = 0.086, p = 0.0160 → **PASS**
- Permutation (L_s2 block): p = 0.2480 → **FAIL**
- Permutation (lev_mod block): p = 0.0550 → **WARN**
- DSR (N=500 保守): DSR = 0.0000 → FAIL（多重比較補正の期待値, 参考）
- **総合: FAIL**（Bootstrap + Permutation ベース）
```
