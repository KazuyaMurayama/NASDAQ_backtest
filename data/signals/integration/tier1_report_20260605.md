# Tier 1 統合検証 中間レポート (Session S2)

作成日: 2026-06-05
最終更新日: 2026-06-05

> Session S2 (Phase 3 deliverable) per `SIGNAL_INTEGRATION_PLAN_20260604.md`.
> Tier 1 = M1 (binary lev mask) + M2 (continuous lev tilt) × 6 Phase B PASS signals × 3 baseline strategies × 2 directions = **72 patterns**.

---

## サマリ

評価パターン数: **72** (6 signals × 3 strategies × M1/M2 × {defensive, procyclical})

評価窓: 1974-01-02 → 2026-03-26 (13,169 obs); IS/OOS split = 2018-01-01。

ベースライン NAV（IS+OOS 通期 CAGR）:
- S1 (NEW CANDIDATE / F10): +32.89%、end NAV 2.81e+6
- S2 (D5 / vz065lmax5)    : +29.90%、end NAV 8.57e+5
- S3 (DH-W1, hysteresis)  : +18.17%、end NAV 6.13e+3

### Judgment counts

| Strategy | STRONG_PASS | STANDARD_PASS | MARGINAL | FAIL | Total |
|---|---|---|---|---|---|
| S1 | 0 | 0 | 20 | 4 | 24 |
| S2 | 0 | 0 | 20 | 4 | 24 |
| S3 | 0 | 0 | 18 | 6 | 24 |
| **計** | **0** | **0** | **58** | **14** | **72** |

**重要な所見**: STRONG / STANDARD パスはゼロ。Tier 1（return-level overlay の単純な M1/M2）では Phase D 進出基準（n_improved ≥ 2 かつ重大劣化なし）を満たすパターンは見つからなかった。原因は §「主要な棄却要因」を参照。

---

## STRONG_PASS 候補 (採用候補)

該当なし。

---

## STANDARD_PASS 候補 (Phase D 評価対象)

該当なし。

---

## ベスト 5 パターン（n_improved 降順 → sharpe_diff 降順）

全件 MARGINAL（n_improved ≥ 1 で重大劣化あり）。10 metric 全展開:

| Rank | Pattern | Strategy | Method/Dir | Signal | CAGR_OOS (cand/base/diff) | Sharpe_OOS (cand/base/diff) | MaxDD (cand/base/diff) | Worst10Y (cand/base/diff) | P10_5Y (cand/base/diff) | IS-OOS gap diff | n_imp/n_deg | Judg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | T1_M1_23_S1_procyclical | S1 | M1 procyclical | BAA-10Y credit spread | 28.93% / 36.90% / **-7.96pp** | 1.026 / 0.944 / **+0.082** | -54.4% / -63.1% / **+8.7pp** | +13.5% / +16.0% / -2.5pp | +10.8% / +10.3% / +0.5pp | +0.010 | 3 / 2 | MARGINAL |
| 2 | T1_M2_6_S3_procyclical | S3 | M2 procyclical | VIX level | 23.02% / 21.11% / **+1.90pp** | 0.952 / 0.903 / **+0.049** | -41.7% / -34.6% / -7.1pp | +7.7% / +9.0% / -1.3pp | +5.3% / +4.8% / +0.5pp | +0.034 | 3 / 3 | MARGINAL |
| 3 | T1_M1_41_S2_procyclical | S2 | M1 procyclical | DXY | 35.45% / 34.23% / **+1.22pp** | 1.024 / 0.990 / **+0.034** | -39.7% / -51.8% / **+12.1pp** | +8.7% / +15.9% / -7.3pp | +8.1% / +11.9% / -3.7pp | +0.149 | 3 / 3 | MARGINAL |
| 4 | T1_M1_41_S1_procyclical | S1 | M1 procyclical | DXY | 38.36% / 36.90% / **+1.47pp** | 0.978 / 0.944 / **+0.033** | -45.4% / -63.1% / **+17.7pp** | +8.6% / +16.0% / -7.4pp | +7.9% / +10.3% / -2.3pp | +0.160 | 3 / 3 | MARGINAL |
| 5 | T1_M1_26_S1_procyclical | S1 | M1 procyclical | 2s10s spread | 22.05% / 36.90% / **-14.85pp** | 0.951 / 0.944 / +0.007 | -55.0% / -63.1% / **+8.1pp** | +12.7% / +16.0% / -3.3pp | +11.4% / +10.3% / +1.2pp | -0.015 | 3 / 2 | MARGINAL |

注: 太字 = 改善判定軸（IMP_THR を超過）。MARGINAL は n_improved ≥ 1 だが重大劣化軸（DEG_THR を下回る）が ≥ 1 ある場合。

---

## 主要な棄却要因（FAIL 14 件の分析）

### Defensive direction の構造的欠陥（FAIL の 13 / 14 を占める）

| Method × Dir | FAIL 件数 | FAIL 率 |
|---|---|---|
| **M1 defensive**   | 10 / 18 | 55.6% |
| **M2 defensive**   |  3 / 18 | 16.7% |
| M1 procyclical |  1 / 18 |  5.6% |
| M2 procyclical |  0 / 18 |  0.0% |

→ **defensive（リスク回避方向にレバ縮小）は OOS パフォーマンスを大幅劣化させる**ことが構造的に判明。

### DXY と 2s10s defensive が壊滅的

特に `DXY defensive` (M1/M2 両方) と `2s10s defensive` (全戦略) は全 6 軸 / 5 軸が DEG_THR を割る重度劣化。これらの信号は「リスク高→レバ縮小」のシンプル mapping では使えない。

### Procyclical は方向は概ね正しいが PASS 基準には不足

Procyclical は MARGINAL に集中（17/18 M1, 18/18 M2）= 何らかの改善はあるが、improvement と degradation がトレードオフで PASS には届かず。これは「信号は意味があるが、Tier 1 の単純な lev_mod multiplier では captured しきれない」シグナル。

---

## Per-strategy insights — どの method × signal が有効か

### S1 (NEW CANDIDATE = F10)
- Best signal: **BAA-10Y credit spread (mean n_improved=1.50)** と **VIX level (1.50)**
- Best pattern: `T1_M1_23_S1_procyclical` — Sharpe +0.082, MaxDD +8.7pp 改善（CAGR -7.96pp トレードオフ）
- DXY procyclical も MaxDD で +17.7pp の劇的改善（CAGR +1.47pp 維持）

### S2 (D5 = vz065lmax5)
- Best signal: **BAA-10Y credit spread (1.50)**
- Best pattern: `T1_M1_41_S2_procyclical` (DXY) — CAGR +1.22pp / Sharpe +0.034 / MaxDD +12.1pp（worst10y/p10_5y で trade-off）
- S2 は元々 IS-OOS gap が大きいため、is_oos_gap_diff で +0.149 など gap 拡大に偏る傾向あり

### S3 (DH-W1)
- Best signal: **BAA-10Y credit spread (1.75)** ← 全戦略で最高スコア
- Best pattern: `T1_M2_6_S3_procyclical` (VIX) — CAGR +1.90pp / Sharpe +0.049 / P10_5Y +0.51pp（MaxDD -7.1pp）
- S3 は元のベースラインが保守的（HOLD mask mean = 0.531）なため、defensive overlay が壊滅的に効く（M1 defensive 4/6 FAIL）

---

## 全方向×シグナルの平均 n_improved（適性マトリクス）

| Signal | S1 | S2 | S3 | 総合適性 |
|---|---|---|---|---|
| #23 BAA-10Y credit spread | 1.50 | 1.50 | **1.75** | **最高** |
| #6  VIX level             | 1.50 | 1.25 | 1.25 | 高（S1 で best tied） |
| #21 ICE BofA HY OAS       | 1.25 | 1.25 | 1.50 | 中〜高 |
| #41 DXY                   | 1.25 | 1.25 | 0.75 | 中（S1/S2 限定） |
| #26 2s10s spread          | 1.25 | 1.25 | 1.00 | 中 |
| #28 10Y real yield        | 1.00 | 1.00 | 1.00 | 低 |

---

## Tier 2/3/4 着手方針

Tier 1 結果に基づく削減・集中方針:

### 1. Direction の絞り込み（即時適用）
- **defensive 方向は M3/M4/M5 でも要警戒**。`risk_off` / `stop_only` 系は Tier 2 で同様の壊滅的劣化が起きる可能性高。
- procyclical / vol_adj / filter_entry など「リスク低 → 縮小」ではない方向を優先評価。

### 2. Signal の絞り込み（Tier 3 への引き継ぎ）
- Tier 3 AND/OR の TOP4_IC は plan で `[21, 6, 41, 23]` 指定だが、Tier 1 結果からは **`[23, 6, 21, 26]` または `[23, 6, 41, 21]`** の方が n_improved 平均で優位。
- ただし TOP4 は IC ベースで Phase B 段階で確定。Tier 3 enumerator はそのまま維持し、追加で `[23, 6]` ペアの精査を行う。

### 3. Method の縮小
- **M1 defensive** は FAIL 55.6% なので Tier 2 以降では M1 defensive 系（M3 risk_off, M5 stop_only）を縮小、procyclical / asymmetric (vol_adj) に集中。
- **M2** はトレードオフ型（n_deg=3 が多い）なので、metric weight 調整 or asymmetric threshold が必要。

### 4. 信号粒度の見直し
- Tier 1 の判定が「n_improved=3, n_degraded=2-3」に集中 = 「改善も劣化も両方ある」中庸領域。これは 4-level quantile が粗すぎる可能性を示唆。
- Tier 2/3 で 8-level quantile やローリング quantile に切り替えてからの判定が必要。

---

## 次セッション

**Session S3**: Tier 2 (M3/M4/M5 = 66 patterns) を実行。Tier 1 で見えた `defensive=避ける` 方針を組み入れ、特に M4 (vol_target_mod / vol_adj) と M5 (filter_entry) に注目。

その後 Tier 3 (AND/OR 36 patterns) → Tier 4 (PCA 6 patterns) を順次実行し、180 パターン総合で Phase D 候補を選定。

---

## 付随アーティファクト

| ファイル | 説明 |
|---|---|
| `data/signals/integration/baseline_navs_20260605.parquet` | S1/S2/S3 三戦略 NAV 統合（13,169 obs × 3 cols） |
| `audit_results/_cache/dh_w1_nav_cache.pkl` | S3 (DH-W1) NAV 再生成キャッシュ |
| `data/signals/integration/tier1_results_20260605.csv` | Tier 1 72 パターンの 32 列詳細メトリクス |
| `scripts/regen_dh_w1_nav.py` | S3 NAV 再生成スクリプト |
| `scripts/run_integration_tier.py` | Tier-N 統合ランナー |
