# Phase D 厳格 Audit Report — S3 × BAA-10Y × M2 procyclical

作成日: 2026-06-05
最終更新日: 2026-06-05

## 1. 候補

| 項目 | 内容 |
|---|---|
| 戦略ベース | S3 (DH-W1, Asymm+Hyst, TQQQ/TMF/GLDM) |
| 注入信号 | #23 BAA-10Y credit spread (Phase B PASS) |
| 注入方式 | M2 procyclical mapping `{0:0.7, 1:0.9, 2:1.1, 3:1.3}` |
| 注入箇所 | DH-W1 の `lev_raw` 段階 (**Native integration**) |
| 注入実装 | `src/integration/build_w1_baa.py::build_W1_baa10y` |

### 1.1 Phase A-C / Tier1-3 までの根拠
- Phase B IC (BAA-10Y → NDX 60d): +0.20, BH-FDR p < 0.10 ✓
- Tier 1 relaxed PASS: n_imp=2 (CAGR_OOS +0.54pp, Sharpe +0.067), n_deg=0 (relaxed thresholds)
- これは **post-hoc NAV multiplication** で得た結果。本 Phase D は native lev_raw 注入で再評価する。

## 2. Integration Methodology

### 2.1 アプローチ: NATIVE (Fork build_W1)

`build_W1` を fork し、信号由来の乗数を NAV 計算前に `lev_raw` に乗じる:

```python
mask = hold_mask_W1(a)
lev_raw_base = np.asarray(a['lev_raw']) * mask        # 既存 W1
mult = build_baa_multiplier(a['dates'])                # 信号→乗数
lev_raw_mod = lev_raw_base * mult                      # ← 注入
nav, cost = build_dh_nav_with_cost(close, lev_raw_mod, wn, wg, wb, ...)
```

これにより:
- 信号は NAV 算出前に lev に作用 → ターンオーバー / 取引コストも信号駆動分を反映
- post-hoc NAV multiplication と比べて経済的に正しい（コストの整合性、複利順序）

### 2.2 Sanity
- `peak_lev_eff`: baseline=2.40x → candidate=3.12x (max_mult=1.3 由来)
- 標準キャップ 3.0 は厳密超過するが、procyclical 注入の宿命であり、相対比較目的のため許容
- BAA-10Y 信号は `quantile_cut(levels=4)` で四分位化 → `publication_lag('daily')` で T+1 ずらし
- 戦略期間内のサンプル数: バケット 0/1/2/3 ≈ 19% / 19% / 19% / 19% (残り 24% は信号開始前/欠損で mult=1.0)

## 3. Audit Result Summary

### 3.1 9+1 メトリクス (audit grade, native NAV)

| metric | baseline | candidate | diff | 判定 |
|---|---|---|---|---|
| CAGR_OOS_pct | +18.961 | +18.517 | **-0.443** | ❌ 悪化 |
| IS-OOS_gap_pp | -0.882 | +0.089 | **+0.971** | ❌ ギャップ拡大 |
| Sharpe_OOS | +0.844 | +0.963 | **+0.118** | ✓ 改善 |
| MaxDD_full_pct | -34.573 | -39.300 | **-4.728** | ❌ 悪化 (-2pp 厳しい閾値超え) |
| Worst10Y_pct | +9.839 | +11.638 | **+1.800** | ✓ 改善 |
| P10_5Y_pct | +5.936 | +6.193 | +0.257 | △ marginal (>0.5pp 未達) |
| Trades_per_yr | 17.6 | 21.4 | +3.7 | ✓ cap 200/yr 内 |
| WFE_full | 0.976 | 0.973 | -0.003 | ◯ 同等 |
| CI95_lo_window_CAGR | +13.95% | +14.07% | +0.12pp | ◯ ほぼ同等 |

**Pareto 集計** (relaxed 9+1 閾値):
- n_improved = **3** (Sharpe, Worst10Y, marginally is_oos_gap fails since diff is positive=worse)
  - 厳密には: Sharpe (+0.118 ≥ 0.03 ✓), Worst10Y (+1.80pp ≥ 0.5pp ✓), MaxDD は 悪化、CAGR_OOS は 悪化
  - **実カウント: 2** (Sharpe, Worst10Y のみ)
- n_degraded = **1** (MaxDD diff = -4.73pp、厳格閾値 -2pp 超え。relaxed 閾値 -10pp 内)
  - relaxed 評価では MaxDD は閾値内
- IS-OOS gap diff = **+0.97pp**: 改善 (-0.5pp) ではない、severe deg (+3.0pp) でもない → 中立

→ **STANDARD_PASS_RELAXED ギリギリ不達**: n_imp=2 (Sharpe, Worst10Y), n_deg=0 (relaxed) で formally `STANDARD_PASS_RELAXED` だが、CAGR_OOS -0.44pp/IS-OOS gap +0.97pp の二重悪化と MaxDD -4.7pp は **経済的に重大な悪化**。

### 3.2 WFA 50 窓 (audit grade)

| metric | baseline | candidate |
|---|---|---|
| 全期間 Sharpe | +0.938 | +0.947 |
| 窓 mean Sharpe (50 窓) | +0.916 | +0.921 |
| **WFE = mean/full** | **0.976** | **0.973** |
| 窓 mean Sharpe (OOS 窓のみ) | +1.046 | +1.127 |
| 窓 mean CAGR (全) | +21.38% | +22.02% |
| **CI95_lo of cand window CAGR** | +13.95% | **+14.07%** |
| CI95 of (cand - base) window CAGR | — | [-1.23%, +2.49%] |

**重要所見**:
- WFE 0.973 < 1.0 → strict gate "WFE ≥ 1.0" **FAIL**
- candidate CI95_lo > 0 ✓ (絶対判定は PASS)
- **diff CI95 は [−1.23%, +2.49%] で 0 を跨ぐ** → 信号注入による改善は窓内 CAGR で **統計的有意性なし**

### 3.3 Block Bootstrap (10,000 resamples, block_size=60, OOS only)

| metric | value |
|---|---|
| 実 OOS CAGR baseline | +18.96% |
| 実 OOS CAGR candidate | +18.52% |
| **実 diff** | **-0.44pp** (candidate が悪い) |
| bootstrap median diff | -0.75pp |
| bootstrap CI95 of diff | [-6.97pp, +4.16pp] |
| bootstrap median candidate CAGR | +19.25%, [P5=+3.95%, P95=+38.04%] |
| bootstrap median baseline CAGR | +19.99%, [P5=+1.80%, P95=+42.15%] |
| **P(cand > base)** | **39.1%** |

**重要所見**:
- **OOS 期間 (2021-05-08 〜 2026-03-26) では actual diff = -0.44pp と既に baseline 劣後**
- 全期間で見える「improvement」は **IS 期間に集中** (IS-OOS gap diff +0.97pp が裏付け)
- P(cand > base) = 39.1% は PASS gate 90% を大きく下回り **DECISIVE FAIL**

## 4. 最終判定 (Hard requirement matrix)

| Hard requirement | 結果 | 判定 |
|---|---|---|
| WFA WFE ≥ 1.0 | 0.973 | **✗** |
| WFA CI95_lo > 0 | +14.07% | ✓ |
| WFA diff CI95 lo > 0 | -1.23% | **✗** |
| Bootstrap P(cand>base) > 0.90 | 0.391 | **✗** |
| Native build feasibility | feasible (build_W1 fork) | ✓ |
| 9+1 Pareto (≥3 imp, 重大悪化なし) | n_imp=2, MaxDD -4.7pp 悪化 | **✗** |
| Peak lev cap (3.0x relaxed→3.9x) | 3.12x | ✓ |
| OOS actual CAGR improvement | -0.44pp | **✗** |

**4 / 8 gates failed (うち決定的 fail: Bootstrap P=39%, OOS actual CAGR -0.44pp, WFA diff CI95 跨 0)**

## 5. 結論

# **REJECT**

### 5.1 Rejection Rationale

1. **OOS 実績で既に劣後** (実 diff -0.44pp): Phase A-C で観測された improvement は post-hoc NAV multiplication に依存しており、native 注入では消失。
2. **Bootstrap で改善が偶然と判定** (P=39%, CI95 of diff straddles 0): 仮に improve に見える期間があっても、ブロック再標本化で過半が baseline 以下に転落 → robust ではない。
3. **MaxDD が 4.7pp 悪化** (-34.6% → -39.3%): procyclical 注入は bull で leverage を高めるため bear で下落が深まる構造的副作用。
4. **IS-OOS gap が +0.97pp 拡大** (-0.88 → +0.09): IS overfit の典型サイン。Phase B IC (+0.20, p<0.10) は OOS 構造には汎化していない。
5. **WFA WFE 改善ゼロ** (-0.003): 信号注入による walk-forward 安定化効果なし。

### 5.2 Phase B/Phase A 結果との整合

Phase B (BAA-10Y → NDX 60d IC=+0.20) は **forward-return 予測力** を測定。NAV 統合に持ち込む際:
- Forward IC の正値 ≠ leverage modulation の正値
- M2 procyclical (bull で leverage UP) は IC の本質 (credit spread 拡大 = リスク高 → underperformance 予測) を **誤った時間配置で活用**
- M2 defensive (bull で leverage DOWN) でも Phase A-C で REJECT 済み (relaxed PASS せず)

→ **Phase B の信号品質が高くても、注入方式が NAV カーブと整合しない場合は ADOPT に至らない**。

## 6. 採用判断 (action items)

### 6.1 CURRENT_BEST_STRATEGY.md 更新
**No** — 現行ベスト戦略 (S2_VZGated + LT2-N750 + E4 Regime k_lt) はそのまま維持。

### 6.2 STRATEGY_REGISTRY.md 追記
**Reject 候補として記録**:
- ID: `DH-W1+BAA10Y+M2proc` (Rejected)
- 理由: Phase D audit FAIL (OOS actual diff -0.44pp, Bootstrap P=39%, MaxDD -4.7pp)
- 教訓: Forward IC の存在は NAV 統合の十分条件ではない。M2 procyclical のような構造的 leverage 拡大は MaxDD/Worst-bear で penalty を受ける。

### 6.3 Phase B 信号洗練の方向性 (次サイクル)
1. **Direction 検証**: BAA-10Y が defensive (credit spread → risk-off) の方が経済的整合的。M2 defensive で再評価する価値あり (Phase A で済んだ可能性は確認要)。
2. **Time-lag 探索**: 60d forward IC が +0.20 でも 1-5d / 20d では？最適 horizon を見直す。
3. **Window-stratified 検証**: IC を IS/OOS 別々に算出 → IS のみで効くなら統合価値なし。
4. **Composite signal**: BAA-10Y + 他 Phase B PASS signal (#11 unemployment, #19 retail sales 等) で diversify する余地。

### 6.4 Methodology Improvement
- Phase A-C の post-hoc NAV multiplication は **screening としては有効だが、Phase D 直前段階で必ず native build に置き換える** ことを SOP 化。
- Tier 1 relaxed 判定の n_imp=2 は **STANDARD_PASS_RELAXED の閾値が緩すぎる** 可能性。CAGR_OOS と MaxDD のいずれかが悪化していたら relaxed でも FAIL とする追加条件を提案。

## 7. Deliverables

| 成果物 | 説明 | パス |
|---|---|---|
| build_w1_baa.py | Native build of DH-W1 × BAA-10Y × M2 procyclical | `src/integration/build_w1_baa.py` |
| phase_d_wfa.py | 50 窓 WFA driver | `src/integration/phase_d_wfa.py` |
| phase_d_bootstrap.py | 10,000 paired block bootstrap driver | `src/integration/phase_d_bootstrap.py` |
| phase_d_metrics.py | 9+1 metric comparator | `src/integration/phase_d_metrics.py` |
| phase_d_wfa_50w_20260605.csv | per-window WFA results | `data/signals/integration/phase_d_wfa_50w_20260605.csv` |
| phase_d_wfa_50w_20260605_SUMMARY.csv | WFA summary stats | `data/signals/integration/phase_d_wfa_50w_20260605_SUMMARY.csv` |
| phase_d_bootstrap_20260605.csv | bootstrap summary | `data/signals/integration/phase_d_bootstrap_20260605.csv` |
| phase_d_metrics_20260605.csv | 9+1 metric comparison | `data/signals/integration/phase_d_metrics_20260605.csv` |
| phase_d_audit_report_20260605.md | 本レポート | `data/signals/integration/phase_d_audit_report_20260605.md` |

---

**Phase D は厳格に動作した**: post-hoc Tier 1 relaxed PASS が、native NAV 統合 + WFA + Bootstrap の 3 ステップ厳格検証で REJECT に転落。**signal-discovery → integration pipeline は今回の信号 (BAA-10Y × M2 procyclical) に対しては合格点を出していない**。

これは pipeline の失敗ではなく **pipeline の正しい動作**: relaxed screening (Tier 1) で広く拾い、audit (Phase D) で厳格に絞る、という設計通りの結果。
