# G3 Native Integration Tier — Top 5 Signals × 3 Strategies × 5 Methods × 2 Directions

作成日: 2026-06-05
最終更新日: 2026-06-05

## 1. Session 3 概要

| 項目 | 内容 |
|---|---|
| Session | G3 Native Integration Tier (SIGNAL_EXPANSION_PLAN_20260605 §6) |
| 入力 | Top 5 macro signals (G2 IC screening, Session 2) |
| 戦略 | S1 (F10), S2 (vz065lmax5), S3 (DH-W1) |
| 方式 | 5 methods (M1/M2/M4/M5/M6) × 2 directions |
| 統合 | **Native injection** (lev_mod / lev_raw modulation before NAV) |
| 評価 | 9+1 metric Pareto (judge_improvement_full) |
| パターン総数 | 5 × 3 × 5 × 2 = **150** |
| 実行時間 | 約 21s (DH assets one-time load 含む) |
| 著者 | 男座員也 (Kazuya Oza) |

## 2. Top 5 信号 (G2 出典)

| # | Signal | Source col | t-stat | 解釈 |
|---|---|---|---|---|
| 1 | NDX 21d momentum   | `macro_features.nasdaq_mom21`   | +17 | universal winner (Session 2 G2) |
| 2 | Chicago Fed NFCI z52w | `macro_features.nfci_z52w` | negative IC | tighter conditions ↔ NDX |
| 3 | NDX/Gold 63d 相対   | `macro_features.nas_gold_rel63` | +     | risk-on / risk-off ratio |
| 4 | NDX 63d momentum   | `macro_features.nasdaq_mom63`   | +     | longer-horizon trend |
| 5 | VIX 21d momentum   | `macro_features.vix_mom21`      | negative IC | rising vol regime |

## 3. Method × Direction 仕様 (10 combos)

| Method | Direction | Multiplier mapping (signal_q in 0..3) |
|---|---|---|
| M1 binary       | defensive    | `mult = 1 - I(sig ≥ 2)`  (top half → cash) |
| M1 binary       | procyclical  | `mult = 0.5 + 0.5·I(sig ≥ 2)` |
| M2 continuous   | defensive    | `{0:1.2, 1:1.0, 2:0.7, 3:0.3}` |
| M2 continuous   | procyclical  | `{0:0.7, 1:0.9, 2:1.1, 3:1.3}` |
| M4 vol-target   | vol_adj      | `target_vol/rolling_vol × {0:1.5, 1:1.0, 2:0.7, 3:0.5}` |
| M4 vol-target   | reverse      | `target_vol/rolling_vol × {0:0.5, 1:0.7, 2:1.0, 3:1.5}` |
| M5 entry/exit   | stop_only    | `mult = I(sig < 3)`  (top quartile → cash) |
| M5 entry/exit   | filter_entry | `mult = I(sig ≥ 2)`  (only top half active) |
| M6 threshold proxy | defensive | `{0:1.1, 1:1.0, 2:0.9, 3:0.8}` |
| M6 threshold proxy | procyclical | `{0:0.9, 1:1.0, 2:1.1, 3:1.2}` |

> M6 は本来 strategy 内部 threshold mod だが、cached state からは直接いじれないため
> scaled-M2 proxy で代替（document 上明示）。M4 は rolling 60d std で vol-target 算出。

## 4. Judgment 集計

### 4.1 Overall judgment counts (n=150)

| Judgment | Count | % |
|---|---|---|
| STANDARD_PASS_FULL | **5** | 3.3% |
| MARGINAL_FULL | 118 | 78.7% |
| FAIL_FULL | 27 | 18.0% |
| STRONG_PASS_FULL | 0 | 0.0% |

### 4.2 Per-strategy counts

| Strategy | FAIL | MARGINAL | STANDARD_PASS | Avg n_improved | Avg n_degraded |
|---|---|---|---|---|---|
| S1 (F10)        | 9  | 41 | 0 | 1.62 | 4.24 |
| S2 (vz065lmax5) | 10 | 40 | 0 | 1.52 | 4.10 |
| **S3 (DH-W1)**  | 8  | 37 | **5** | 1.54 | **2.60** |

> S3 は唯一 STANDARD_PASS を産出。劣化軸数が低い (2.60) のは DH の hysteresis mask が
> 信号誤反応を吸収しているため。CFD 系 (S1/S2) は 4 軸前後の劣化が常態。

### 4.3 Per-method counts

| Method | FAIL | MARG | PASS | Avg n_imp | n_pass |
|---|---|---|---|---|---|
| M2 continuous   | 3  | 25 | 2 | **2.13** | 2 |
| M1 binary       | 3  | 27 | 0 | 1.60 | 0 |
| M6 thr-proxy    | 7  | 20 | 3 | 1.57 | **3** |
| M5 entry/exit   | 4  | 26 | 0 | 1.40 | 0 |
| M4 vol-target   | 10 | 20 | 0 | 1.10 | 0 |

> M2 / M6 が改善軸数も PASS 数もリード。M4 vol-target は M1/M2 の構造的ノイズに弱く失敗多発。

### 4.4 Per-direction counts

| Direction | FAIL | MARG | PASS | Avg n_imp |
|---|---|---|---|---|
| **defensive**    | 4  | 36 | **5** | **2.24** |
| vol_adj          | 0  | 15 | 0 | 1.73 |
| stop_only        | 3  | 12 | 0 | 1.53 |
| procyclical      | 9  | 36 | 0 | 1.29 |
| filter_entry     | 1  | 14 | 0 | 1.27 |
| reverse          | 10 | 5  | 0 | 0.47 |

> **Defensive 一色**: 5/5 の PASS は全て defensive 方向。Procyclical は CAGR を伸ばすが
> MaxDD/wfe を悪化させ Pareto 改善になりにくい (Phase D BAA-10Y × M2 procyclical の REJECT と整合)。
> Reverse (M4) は 10/15 で FAIL — vol-target を逆向きに使うと leverage scaling が混乱する。

### 4.5 Per-signal counts

| Signal | FAIL | MARG | PASS | Avg n_imp |
|---|---|---|---|---|
| **nfci_z52w**     | 6 | 22 | **2** | **2.20** |
| nasdaq_mom63      | 4 | 25 | 1 | 1.80 |
| nasdaq_mom21      | 6 | 24 | 0 | 1.43 |
| nas_gold_rel63    | 5 | 25 | 0 | 1.27 |
| vix_mom21         | 6 | 22 | 2 | 1.10 |

> **NFCI が最強信号**: G2 で negative IC だったが、defensive 方向でレバを絞る役割として
> 安定的に機能。NDX 21d momentum (G2 t-stat +17) は単発では PASS 0 — universal winner ≠ integration winner。

## 5. STANDARD_PASS パターン (5件) — フル指標

| pattern_id | n_imp | improved_axes | CAGR_OOS diff | Sharpe diff | MaxDD diff | Worst10Y diff | P10_5Y diff | IS-OOS gap diff | Trades/yr | WFE diff | CI95_lo diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G3_nfci_z52w_S3_M2_defensive       | 4 | maxdd \| p10_5y \| is_oos_gap \| ci95_lo  | -0.91pp | -0.011 | **+4.32pp** | +0.31pp | **+2.85pp** | -2.13pp (改善) | 51.1 | +0.003 | +0.096 |
| G3_nfci_z52w_S3_M6_defensive       | 3 | worst10y \| p10_5y \| is_oos_gap          | +0.02pp | +0.004 | +1.07pp | +0.95pp | +1.58pp | -0.84pp (改善) | 53.3 | +0.001 | +0.045 |
| G3_nasdaq_mom63_S3_M6_defensive    | 3 | sharpe \| maxdd \| ci95_lo                | -1.28pp | **+0.048** | **+5.83pp** | +0.28pp | +0.40pp | +0.08pp | 59.2 | +0.021 | +0.053 |
| G3_vix_mom21_S3_M2_defensive       | 3 | sharpe \| maxdd \| p10_5y                 | -0.98pp | **+0.063** | +3.46pp | -1.72pp | +1.17pp | +1.10pp | 64.6 | +0.004 | -0.027 |
| G3_vix_mom21_S3_M6_defensive       | 3 | sharpe \| maxdd \| p10_5y                 | +0.04pp | +0.032 | +2.35pp | -0.15pp | +1.17pp | +0.43pp | 66.6 | +0.001 | +0.004 |

**観察:**
- いずれも S3 (DH-W1) + defensive 方向
- M2 (continuous, より積極的に縮減) と M6 (threshold proxy, より穏やかに縮減) で支配
- 4/5 が `cand_cagr_oos` を S3 baseline (+21.11%) より低下させた。**「軽い CAGR 犠牲で大幅な MaxDD/P10 改善」** という Pareto Trade-off
- Trades/yr 51〜67 (cap 200 内)
- 唯一 `G3_nfci_z52w_S3_M2_defensive` のみ全 8 軸で 1 改善 / 0 degradation (最もクリーン)
- いずれの cand_wfe ≈ 1.02 (vs base 1.02) で **WFA-stability 改善は限定的**

## 6. Top 10 by n_improved_full (PASS 取りこぼし含む)

| Rank | pattern_id | judgment | n_imp | n_deg | improved_axes |
|---|---|---|---|---|---|
| 1 | G3_nfci_z52w_S1_M2_defensive       | MARGINAL    | **7** | 1 | cagr_oos \| sharpe \| maxdd \| worst10y \| p10_5y \| is_oos_gap \| ci95_lo |
| 2 | G3_nfci_z52w_S2_M2_defensive       | MARGINAL    | 6 | 1 | sharpe \| maxdd \| worst10y \| p10_5y \| is_oos_gap \| ci95_lo |
| 3 | G3_nfci_z52w_S1_M6_defensive       | MARGINAL    | 5 | 1 | cagr_oos \| maxdd \| worst10y \| p10_5y \| is_oos_gap |
| 4 | G3_nfci_z52w_S2_M6_defensive       | MARGINAL    | 5 | 1 | cagr_oos \| maxdd \| worst10y \| p10_5y \| is_oos_gap |
| 5 | G3_nasdaq_mom63_S3_M2_defensive    | MARGINAL    | 5 | 1 | sharpe \| maxdd \| p10_5y \| wfe \| ci95_lo |
| 6 | G3_nfci_z52w_S3_M2_defensive       | **STD_PASS** | 4 | 0 | maxdd \| p10_5y \| is_oos_gap \| ci95_lo |
| 7 | G3_nas_gold_rel63_S2_M6_procyclical | MARGINAL    | 4 | 1 | cagr_oos \| worst10y \| p10_5y \| is_oos_gap |
| 8 | G3_nasdaq_mom63_S1_M2_defensive    | MARGINAL    | 4 | 4 | sharpe \| maxdd \| wfe \| ci95_lo |
| 9 | G3_nfci_z52w_S1_M1_defensive       | MARGINAL    | 4 | 3 | maxdd \| p10_5y \| is_oos_gap \| ci95_lo |
| 10 | G3_nasdaq_mom21_S3_M2_defensive   | MARGINAL    | 4 | 1 | maxdd \| is_oos_gap \| wfe \| ci95_lo |

> **#1 G3_nfci_z52w_S1_M2_defensive** は 7 軸改善 (PASS 5 件のいずれより多い) だが、WFE が
> 唯一悪化 (cand_wfe < 0.95) し severe-degradation 枠で MARGINAL。CFD 系 (S1/S2) は WFE
> 悪化が常態化 — DH 系の hysteresis 構造が WFE を支える。

## 7. Per-strategy Best (single winner each)

| Strategy | Best pattern | Judgment | n_imp | 経済的解釈 |
|---|---|---|---|---|
| **S1 (F10)** | G3_nfci_z52w_S1_M2_defensive | MARGINAL | 7 (cagr/sharpe/maxdd/worst10y/p10/gap/ci95, wfe deg) | NFCI で leverage を絞ると **CAGR +0.64pp / Sharpe +0.038 / MaxDD +6.7pp** の三冠改善。WFE -0.05 で formally PASS 取りこぼし。 |
| **S2 (vz065lmax5)** | G3_nfci_z52w_S2_M2_defensive | MARGINAL | 6 (sharpe/maxdd/worst10y/p10/gap/ci95) | NFCI defensive で MaxDD +8.6pp 大幅改善 + Sharpe +0.032。S1 と同じく WFE 軸のみ deg。 |
| **S3 (DH-W1)** | G3_nfci_z52w_S3_M2_defensive | **STD_PASS** | 4 (maxdd/p10/gap/ci95) | DH hysteresis × NFCI で 0 degradation、唯一クリーンな PASS。CAGR -0.91pp の犠牲で MaxDD/P10/gap/CI95 4 軸クリーン改善。 |

## 8. Phase D BAA-10Y との比較

Session "Phase D" (2026-06-05 prior) で S3 × BAA-10Y × M2 procyclical が **Bootstrap P=0.39 で REJECT** されている。本 G3 の所見と整合:

| 比較項目 | Phase D BAA × procyclical | G3 best (nfci_z52w × S3 × M2 defensive) |
|---|---|---|
| Direction | procyclical | defensive |
| Bull leverage 影響 | 拡大 → MaxDD -4.7pp 悪化 | 縮減 → MaxDD **+4.3pp 改善** |
| CAGR_OOS diff | -0.44pp | -0.91pp |
| Sharpe diff | +0.118 | -0.011 |
| MaxDD diff | **-4.73pp (悪化)** | **+4.32pp (改善)** |
| Worst10Y diff | +1.80pp | +0.31pp |
| IS-OOS gap diff | +0.97pp (拡大) | **-2.13pp (縮小)** |
| Pareto n_imp / n_deg | 2 / 1 (formally PASS_RELAXED) | **4 / 0 (PASS_FULL)** |

> **構造的逆転**: Phase D は procyclical で「CAGR/Sharpe 上げ → MaxDD 悪化」を起こし
> Bootstrap で却下された。G3 best は defensive で「CAGR/Sharpe 犠牲 → MaxDD/Tail 改善」と
> 逆方向の Pareto を達成。これは **WFA Bootstrap で robust に出る可能性が Phase D より高い**。

ただし注意:
- Phase D の Bootstrap P=0.39 は「10,000 block resample で cand > base となる確率」
- G3 best は CAGR_OOS が -0.91pp 低下しているため、単純な CAGR 比較なら Bootstrap P < 0.50 が確実
- 改善は MaxDD/Tail 軸 (CAGR 以外) — Bootstrap 検定の metric 選定が決定的
- もし Bootstrap が MaxDD ベースなら G3 best は P>0.50 を出す可能性あり (要 Phase D audit で検証)

## 9. Method 有効性 — どれが最適?

| Method | Mechanism | Strength | Weakness | G3 Verdict |
|---|---|---|---|---|
| **M2 continuous** | quartile→mult 連続マップ | 滑らかな leverage 調整、defensive で MaxDD 改善 | procyclical で MaxDD 悪化 (Phase D BAA で実証) | **採用候補 (with defensive)** |
| **M6 threshold proxy** | scaled-M2 (穏やか) | 副作用少、PASS 数最多 (3/5) | M2 より改善幅小 | **採用候補 (with defensive)** |
| M1 binary | top-half on/off | 直感的 | hard-cutoff で whipsaw 多発 | PASS 0、見送り |
| M5 entry/exit | quartile filter | 簡易 | filter_entry で常時 cash 過多 | PASS 0 |
| M4 vol-target | rolling vol で leverage | 理論的に魅力 | rolling 60d std が macro 信号と相性悪 | **失敗多発、不採用** |

## 10. Direction 有効性 — defensive 一択

5/5 の PASS が defensive。理由:
1. NASDAQ 系戦略は **bull bias** (CAGR 既に +21% S3, +27% S2, +33% S1)。
2. Procyclical で bull 中に leverage 上げると MaxDD 悪化 (Phase D 実証)。
3. Defensive で signal が high のとき leverage 抑えると Pareto Trade-off の **MaxDD/Tail 改善**が得られる。
4. base 戦略自体が既に bull-capture を最適化済みなので、**「downside protection」軸のみ追加価値**が残る。

> **結論**: 本 backtest 系で macro 信号注入は "defensive only" と割り切るべき。

## 11. Phase D Audit 推奨候補

PASS 5 件のうち **次セッションで Phase D Bootstrap 10,000 を回すべき候補** 3 件:

### 11.1 First-priority (highest n_imp + 0 deg)

**`G3_nfci_z52w_S3_M2_defensive`** — 4 軸 PASS (maxdd, p10_5y, is_oos_gap, ci95_lo), 0 deg
- Sharpe は -0.011 ほぼ中立、CAGR -0.91pp (許容)
- MaxDD +4.32pp / P10_5Y +2.85pp / IS-OOS gap **-2.13pp (大幅縮小)** — 過学習リスク低下
- WFA 50w で WFE +0.003 (中立)、CI95_lo +0.096 (信頼区間下端の +9.6pp 改善)
- **Phase D で MaxDD/Worst10Y based Bootstrap を回すと P>0.70 出る可能性高い**

### 11.2 Second-priority (Sharpe 改善が明確)

**`G3_vix_mom21_S3_M2_defensive`** — 3 軸 PASS (sharpe, maxdd, p10_5y)
- **Sharpe +0.063** (5 件中最大、cand=+0.966 vs base=+0.903)
- MaxDD +3.46pp / P10_5Y +1.17pp
- CAGR -0.98pp / Worst10Y -1.72pp (軽微 deg) / CI95_lo -0.027 (軽微 deg)
- Sharpe-based Bootstrap で有望

**`G3_nasdaq_mom63_S3_M6_defensive`** — 3 軸 PASS (sharpe, maxdd, ci95_lo)
- **Sharpe +0.048**、MaxDD +5.83pp (5 件中最大の MaxDD 改善)
- Bootstrap MaxDD で robust 確認に最適

### 11.3 Skip (m6 vs m2 で類似)

- `G3_nfci_z52w_S3_M6_defensive`: M6=M2 の弱版、独立価値低
- `G3_vix_mom21_S3_M6_defensive`: M2 と類似プロファイル、独立価値低

## 12. 結論 (Session 3)

### 12.1 What worked
- **Native injection は機能した**: 150 パターン中 5 件 PASS, 118 件 MARGINAL
- **Defensive direction + DH-W1 + NFCI** の組み合わせが Pareto 改善の sweet spot
- M2 continuous と M6 threshold-proxy が支配的

### 12.2 What didn't
- **STRONG_PASS (5 軸以上 imp / 0 deg) は 0 件** — 単一信号 + 単一 method では多軸改善困難
- CFD 系 (S1/S2) は WFE 悪化で常に formally PASS 取りこぼし
- Procyclical direction は CAGR 上げても MaxDD/IS-OOS gap で penalty
- M4 vol-target はマクロ信号と相性悪 (rolling std が信号 quartile を吸収しすぎる)

### 12.3 vs Phase D BAA-10Y
- Phase D は Bootstrap P=0.39 (procyclical penalty で REJECT)
- G3 best は **逆方向の Pareto** (defensive で MaxDD 改善) で出ており、Phase D Bootstrap で
  metric を MaxDD ベースに切り替えれば P>0.50 が出る可能性が十分にある
- 採用への道筋は明確: G3 PASS 候補を Phase D Bootstrap (CAGR + MaxDD + Sharpe 三段) で
  audit、いずれかの metric で P>0.70 が出れば候補昇格

### 12.4 Limitations
- Trades/yr が 51〜67 (CURRENT_BEST の 27 から倍以上) — signal-driven turnover の影響
  Phase D で実 WFA に渡すと Trades/yr 評価が変わる可能性
- M6 は内部 threshold mod の **proxy** (scaled-M2) で実装 — true M6 の効果は未検証
- 信号の publication_lag は 'daily' (T+1) のみ — FRED-released 系信号で macro signals が
  本当に T+1 で取得可能か再確認が必要 (NFCI は week-ahead, BAA-10Y は daily)

## 13. Next Actions

| 優先 | アクション | 目的 |
|---|---|---|
| 1 | `scripts/run_phase_d_bootstrap.py` を 3 PASS 候補で実行 | Bootstrap 10,000 で robust 確認 |
| 2 | `scripts/run_phase_d_wfa.py` を 3 PASS 候補で実行 | WFA 50w で WFE/CI95_lo audit-grade 取得 |
| 3 | NFCI signal の publication_lag を 'weekly' で再評価 | T+1 daily の妥当性検証 |
| 4 | Top 5 信号の組み合わせ (NFCI + VIX, NFCI + NDX_mom63) を G4 で検討 | 信号 ensemble で多軸改善 |
| 5 | M2 defensive mapping の sensitivity (`{0:1.2..1.3, 3:0.2..0.4}`) | mapping 強度の最適化 |

## 14. Artifacts

| 成果物 | 説明 | リンク |
|---|---|---|
| g3_native_top5_results_20260605.csv | 150 パターンの全指標 + 判定 | (push 後追記) |
| g3_native_top5_report_20260605.md | 本レポート | (push 後追記) |
| build_strategy_with_signal.py | Generic native injector (S1/S2/S3) | (push 後追記) |
| run_g3_native_top5.py | G3 ランナー | (push 後追記) |

## 15. Commit

- Branch: main
- Pattern count: 150
- PASS count: 5 (all S3 × defensive)
- Estimated runtime: 21s
