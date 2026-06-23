# annual_returns_sc180_200_20260620.csv — 税基準の注意（2026-06-22 追記）

⚠ **このCSVの年次リターン列（sc1.40_strong_pct / sc1.60_strong_pct / sc1.80_strong_pct / sc2.00_strong_pct / NASDAQ_1x_BH_pct）は PRE-TAX（譲渡益税前）です。**

- 確認の目印: `NASDAQ_1x_BH_pct` の 2008 = **−40.54%（pre-tax）**。after-tax なら ×0.8273 = −33.54%。
- **税後（after-tax）の戦略系列（例 B1×scale, B3a, P09_C1 の年次）と同じ表に並べる場合は、この5列に ×0.8273 を適用してから比較すること。**
- 混合（pre-tax 列 と after-tax 列を無変換で同表）は誤り。2026-06-21 の `B1_SCALE_FRONTIER_20260621.md` レポート生成時に独立QCで検出・是正済（生成器 `src/audit/gen_b1_scale_report_20260621.py` は ×0.8273 を適用して co-list）。
- 税後係数 ×0.8273 の根拠は `EVALUATION_STANDARD.md` §1（日本居住者 譲渡益税 20.315% の実効係数）。
- 一般則: 年次リターンを co-list する際は**全系列の税基準を揃える**（`EVALUATION_STANDARD.md` §0 / `CURRENT_BEST_STRATEGY.md` 参照プロトコル「税基準の混在に注意」）。

関連: [LEVERUP_SCALE_FRONTIER_20260619.md](../LEVERUP_SCALE_FRONTIER_20260619.md)（このCSVの初出）、[B1_SCALE_FRONTIER_20260621.md](../B1_SCALE_FRONTIER_20260621.md)（税後化して co-list）。
