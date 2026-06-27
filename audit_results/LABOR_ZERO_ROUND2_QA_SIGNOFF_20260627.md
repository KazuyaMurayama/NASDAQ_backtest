# Analysis QA Sign-off — 労働補填ゼロ Round-2 最適化

**Analysis title:** 引退後・労働補填ゼロ 2ラウンド最適化（資産4,000万・支出720万）
**Reviewer:** 男座員也（Kazuya Oza）／独立QC（第一原理で前提を攻撃）
**Author:** 男座員也（Kazuya Oza）
**Date reviewed:** 2026-06-27
**Intended audience:** 本人（運用方針の意思決定）
**Delivery format:** report（`LABOR_ZERO_ROUND2_OPTIMIZATION_40M_720_20260627.md`）

---

## Automated QA Results

| Check | Status | Notes |
|---|---|---|
| qa_runner.py（winners CSV） | PASS | 4 checks / 0 FAIL / 0 WARN |
| qa_runner.py（sweep CSV） | PASS | 0 FAIL（列名WARNは命名規約のみ・データ問題なし） |

---

## Manual Checklist Summary

| Section | Status | Issues found |
|---|---|---|
| 1. Question framing | PASS | ゴール（追加資産ゼロ・支出維持・労働補填ゼロ）に正対。ただし「支出維持」が**名目固定**と暗黙定義されていた点を明示化（Issue 1）。 |
| 2. Data sourcing | PASS | 年次税後リターンは既存検証済CSV。債券=1x投信時変デュレーション22y・×0.8273。2022債券−22.3%も標本内。 |
| 3. Transformations & calculations | PASS | 独立2経路（harness simulate_v2 vs インラインsim）で全候補一致。厳密40M予算遵守（超過バケツ設計は是正済）。 |
| 4. Statistical validity | **FAIL→修正** | **単一パス過学習の体系的検証が不足**していた（Issue 2）。市場一様縮小・債券ヘアカット・インフレ・真の最小余裕を追加し定量化。 |
| 5. Findings & conclusions | **FAIL→修正** | **インフレ前提の未明記**（Issue 1・最重要）。headline留保の floor 帰属に**誤記**（Issue 3）。両方修正。 |
| 6. Presentation | PASS | リンク・数値整合。修正後に再確認。 |

---

## Issues Found

| # | Severity | Description | Resolution | Status |
|---|---|---|---|---|
| 1 | **MUST FIX** | **支出が名目固定720万（インフレ非連動）であることが未明記**。インフレ連動（実質購買力維持）にすると労働補填ゼロは即破綻（+2%/年で24労働年・+3.5%で38）。1975-2005の高インフレ下では実質維持シナリオで未達＝結論の適用範囲を大きく限定する前提。 | §4.2にインフレ感度を新設・headline留保とNext Action・脚注に明記。「名目固定が必要条件」と結論を限定。 | Fixed |
| 2 | **MUST FIX** | **頑健性検証が「単一年半減」のみ**で、徐々に弱い市場・債券前提・前提感度が未定量。過学習の度合いが不明だった。 | `labor_zero_round2_robustness_20260627.py` を作成。S1市場一様縮小（f=0.82まで=約18%弱い市場まで持つ）・S2債券×0.4まで頑健・S3インフレ・S4真の最小余裕+1,092万 を定量化し§4.1に表で追加。 | Fixed |
| 3 | SHOULD FIX | headline留保の誤記：「floor 1,810万は最も穏当な開始年の値で binding 開始年ではもっと薄い」と記載していたが、QCで **floor 1,812万は最も制約的な1989開始（1995年）の値**＝binding 開始年の floor であることを確認（誤記）。 | headline留保を訂正（benign でなく binding の floor）。§4.1 S4で根拠提示。 | Fixed |
| 4 | MINOR | 債券のディスインフレ・バイアス（1975-2005の金利15%→1%で債券有利）が未言及。 | 脚注に追記。ただしS2で債券×0.4まで頑健なため致命的でないことも明記。 | Fixed |
| 5 | （確認） | 計算の正しさ（労働補填ゼロ・floor・終端）はQCで**正しいと確認**（独立2経路一致・QA 0 FAIL）。主張の数値自体に誤りなし。 | — | Confirmed correct |

---

## Delivery Decision

- [x] **Approved with caveats** — must-fix項目（Issue 1/2/3）を修正済。結論「名目固定支出・史実パス上で労働補填ゼロ達成可能」は数値的に正しいが、**インフレ連動では崩れる**という重大な適用限界を明示した上で提示。

**Caveat statement for delivery:**
> 労働補填ゼロ（sc2.6・運用1,400万/予備2,600万債券・1,400万割れ全額）は**支出を名目720万に20年固定する前提で・かつ1975-2025の史実パス上**で成立する。**支出をインフレ連動（実質購買力維持）にすると達成できない**（+2%/年で24労働年）。また市場が史実比18%超弱い場合・特定の大回復年（1987/89/91/92）が史実より弱い場合も崩れる。「理論上達成可能な設計が存在する」ことの証明であり、将来や実質ベースでの保証ではない。

**Reviewer signature:** 男座員也（Kazuya Oza） / 独立QC  **Date:** 2026-06-27
