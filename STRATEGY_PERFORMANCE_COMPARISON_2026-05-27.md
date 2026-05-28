# 戦略パフォーマンス比較表 v1.8 — 2026-05-27 統合21戦略 全 WFA 完了版

作成日: 2026-05-27
最終更新日: 2026-05-27
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**
生成スクリプト: `src/gen_strategy_comparison_20260527.py`

> ### ◆ 現行ベスト戦略
> **E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7)**
> CAGR_OOS **+33.53%** | Sharpe **+0.891** | MaxDD **−60.0%** | Trades/yr **27** | G3 WFA PASS ✓

---

## 📋 §1 比較前提

| 項目 | 定義 |
|------|------|
| **IS** | 1974-01-02 〜 2021-05-07（47.3年） |
| **OOS** | 2021-05-08 〜 2026-03-26（4.9年） |
| **FULL** | 1974-01-02 〜 2026-03-26（52.26年） |
| **コスト** | Scenario D（`src/product_costs.py` 2026-05-12 基準） |
| **DELAY** | 2営業日（look-ahead bias 対策） |
| **Sharpe Rf** | 0 |
| **CURRENT_BEST** | E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7)（◆, G3 WFA PASS 確定） |
| **WFA** | G10: 49窓（252日 calendar-year-anchored non-overlapping）|

| 凡例 | 意味 |
|------|------|
| ◆ | 現行ベスト戦略 |
| ✅ | Shortlisted（WFA PASS・ベスト昇格候補） |
| ⚠ | IS-OOS gap 警戒（gap ≤ −4.0pp） |
| ‡ | 参照行（採用候補外、比較基準として掲載） |
| [B]/[A]/[C]/[D5] | 実験系統ラベル |
| WARN | 改善効果なし・実験目的のみ |
| FAIL | 指標悪化・不採用確定 |

---

## 📊 §2 全戦略 統合比較表（21戦略 × 9指標 / 10列）

> **単位**: CAGR_OOS / Worst10Y★ / P10▷ / MaxDD = %、IS-OOS gap = pp、Tr = 回/年
> ★ = Sharpe_OOS > +0.885 / ◎ = > +0.770（S2ベースライン）
> **Overfit(WFE)**: ✅ LOW (WFE∈[0.5,2.0]) / ⚠ MED (WFE>2.0) / ❌ HIGH (WFE<0.5)
> WFA 実施完了: **21/21 行（G3/G7/G8/G10 + G11 一括補完）**

| Strategy | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | Overfit<br>(WFE) | CI95<br>_lo |
|:---------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|:----------------:|-----------:|
| **E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7) ◆** | +33.5% | +0.89 ★ | -60.0% | +18.7% |  +9.8% | -1.81pp |  27 | ✅ LOW<br>(1.1) | +0.265 |
| **[Active候補] F10 ε=0.015 ✅⚠** | +36.8% | +0.93 ★ | -63.1% | +18.6% | +10.3% | -4.31pp |  52 | ✅ LOW<br>(1.2) | +0.279 |
| **[Active候補] F10+lmax5 ✅⚠** | +33.6% | +0.94 ★ | -54.2% | +16.9% | +12.8% | -3.37pp |  52 | ✅ LOW<br>(1.3) | +0.256 |
| **[Active候補] D5 vz=0.65/lmax=5.0 ✅⚠** | +33.5% | +0.95 ★ | -51.8% | +18.3% | +11.9% | -3.91pp |  27 | ✅ LOW<br>(1.3) | +0.248 |
| [B] B4 k_lo=0/k_hi=0.7/vz=0.7 | +33.3% | +0.89 ★ | -60.4% | +18.6% |  +9.9% | -1.62pp |  27 | ✅ LOW<br>(1.1) | +0.265 |
| [A] A1 α=2 (soft regime) | +33.5% | +0.89 ★ | -61.6% | +18.6% |  +7.8% | -1.33pp |  27 | ✅ LOW<br>(1.1) | +0.271 |
| [A] A1 α=3 (soft regime) | +33.9% | +0.89 ★ | -62.9% | +18.4% |  +7.4% | -1.59pp |  27 | ✅ LOW<br>(1.1) | +0.275 |
| [A] A1 α=5 (soft regime) | +34.6% | +0.90 ★ | -65.0% | +18.3% |  +6.9% | -2.06pp |  27 | ✅ LOW<br>(1.1) | +0.279 |
| [A] A1 α=8 (soft regime) | +35.1% | +0.90 ★ | -67.1% | +18.2% |  +6.4% | -2.48pp |  27 | ✅ LOW<br>(1.1) | +0.281 |
| [A] A2 lmax_base=6/vol_sens=2 | +32.6% | +0.90 ★ | -58.7% | +18.5% | +10.7% | -1.19pp |  27 | ✅ LOW<br>(1.1) | +0.261 |
| [A] A2B rolling VOL_REF | +31.7% | +0.89 ★ | -56.9% | +18.6% | +11.1% | -0.87pp |  27 | ✅ LOW<br>(1.1) | +0.255 |
| [A] A3 VoV dual gate (vov=1.3/α=0.2) | +33.5% | +0.89 ★ | -60.0% | +18.7% |  +9.8% | -1.81pp |  27 | ✅ LOW<br>(1.1) | +0.265 |
| [C] C2 adaptive deadband (ε₀=0.020) | +36.8% | +0.93 ★ | -63.0% | +18.6% | +10.3% | -4.28pp |  51 | ✅ LOW<br>(1.2) | +0.279 |
| [C] C3 Yang-Zhang (yz_n=10/vz=0.7) | +33.3% | +0.89 ★ | -58.4% | +19.4% | +10.6% | -2.05pp |  27 | ✅ LOW<br>(1.1) | +0.261 |
| [D5] vz=0.60/lmax=4.5 | +28.9% | +0.88 ◎ | -49.2% | +18.1% | +11.8% | -0.15pp |  27 | ✅ LOW<br>(1.2) | +0.237 |
| [D5] vz=0.60/lmax=5.0 | +31.3% | +0.90 ★ | -51.1% | +18.8% | +11.9% | -1.28pp |  27 | ✅ LOW<br>(1.2) | +0.250 |
| [D5] vz=0.65/lmax=4.5 | +30.9% | +0.93 ★ | -49.9% | +17.7% | +11.8% | -2.46pp |  27 | ✅ LOW<br>(1.2) | +0.236 |
| [D5] vz=0.65/lmax=5.5 ✅ | +34.6% | +0.94 ★ | -53.4% | +18.8% | +11.5% | -4.14pp |  27 | ✅ LOW<br>(1.3) | +0.257 |
| [D5] vz=0.65/lmax=6.0 ✅ | +35.3% | +0.94 ★ | -54.9% | +19.5% | +10.9% | -4.20pp |  27 | ✅ LOW<br>(1.2) | +0.263 |
| [D5] vz=0.65/lmax=7.0 ✅⚠ | +37.0% | +0.95 ★ | -58.7% | +18.7% |  +9.2% | -5.22pp |  27 | ✅ LOW<br>(1.2) | +0.269 |
| [D5] vz=0.70/lmax=5.0 | +30.6% | +0.90 ★ | -54.1% | +16.9% | +12.1% | -1.00pp |  27 | ✅ LOW<br>(1.2) | +0.245 |

*◎ Sharpe_OOS > +0.770（S2ベースライン超過）/ ★ > +0.885（現行ベスト超過）。進格: CI95_lo > 0 かつ 0.5 ≤ WFE ≤ 2.0。Overfit(WFE): ✅LOW=WFE∈[0.5,2.0] / ⚠MED=WFE>2.0 / ❌HIGH=WFE<0.5。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカの ◎/★ とは別意味）。*

**WFA 完了行**: **21/21**（G3 ◆ + G7/G8 + G10 vz=0.65 系3行 + **G11 一括補完 14行**）
**WFA 未実施行**: 0/21（全行完了 — 2026-05-27 G11 完了）

---

## 🔬 §3 実験別判定サマリー

| 実験ID | 仮説 | 判定 | 結論 |
|--------|------|:----:|------|
| **B4** k_lo=0 | k_lo=0 でも E4 に近い性能が出るか | FAIL | Sharpe=+0.887 < E4 +0.891。k_lo=0.1 が最適。変更不要。 |
| **A1** α=2 | soft sigmoid (緩慢遷移) で MaxDD 改善 | WARN | MaxDD=-61.7% 改善不十分。Sharpe=+0.886 < E4。 |
| **A1** α=3 | soft sigmoid 中間値 | WARN | Sharpe=+0.890 ≈ E4。MaxDD=-62.9% 微改善。差異微小。 |
| **A1** α=5 | soft sigmoid 中間値 | WARN | Sharpe=+0.898 向上だが gap=-2.06pp 拡大。 |
| **A1** α=8 | soft sigmoid (急遷移) で Sharpe 向上 | WARN | Sharpe=+0.904 向上だが MaxDD=-67.1%⚠ 悪化。 |
| **A2** base=6/sens=2 | vol 感応 l_max で MaxDD 改善 | WARN | MaxDD=-58.7% 改善。CAGR=+32.6% 低下。E4 非優越。 |
| **A2B** rolling VOL_REF | 10年ローリング VOL_REF で中立化 | FAIL | vol_ref_t=0.161（mean < 常時閾値）→ l_max 常時削減バイアス。CAGR=+31.7% 大幅低下。 |
| **A3** vov=1.3/α=0.2 | VoV z-score で Volcker 期を回避 | FAIL | MaxDD=-60.01% 変化なし。1980-82 Volcker 期は vol 低下期→ VoV 0 日発火。構造的限界。 |
| **C2** ε₀=0.020/adaptive | 適応 deadband で IS-OOS gap 縮小 | WARN | F10 と同性能。gap=-4.28pp⚠ 改善なし。Trades=51 高コスト。 |
| **C3** yz_n=10/vz=0.7 | Yang-Zhang 推定量で MaxDD 改善 | WARN | MaxDD=-58.4% 改善・Worst10Y=+19.37% 向上。ただし Sharpe=+0.888 < E4。WFA 未実施。 |
| **D5** vz=0.65/lmax=5.5 | vz_thr=0.65 × l_max チューニング | **PASS** | G10 PASS (CI95_lo=+25.7%, WFE=+1.257)。CAGR=+34.6%, Sharpe=+0.945。gap=-4.14pp⚠ |
| **D5** vz=0.65/lmax=6.0 | vz_thr=0.65 × l_max チューニング | **PASS** | G10 PASS (CI95_lo=+26.3%, WFE=+1.244)。CAGR=+35.3%, Sharpe=+0.939。gap=-4.20pp⚠ |
| **D5** vz=0.65/lmax=7.0 | vz_thr=0.65 × l_max チューニング | **PASS⚠** | G10 PASS (CI95_lo=+26.9%, WFE=+1.230)。CAGR=+37.0%（最高）。gap=-5.22pp⚠⚠ 過学習警戒。 |
| **D5** vz=0.60/lmax=4.5 | MaxDD 最小化フロンティア探索 | TBD | MaxDD=-49.2%（最良）。Sharpe=+0.885 < E4。gap=-0.15pp 最良。WFA 未実施。 |
| **D5** vz=0.60/lmax=5.0 | vz=0.60 系 l_max スイープ | TBD | MaxDD=-51.1%。Sharpe=+0.900。WFA 未実施。 |
| **D5** vz=0.65/lmax=4.5 | vz=0.65 系 l_max 下限探索 | TBD | MaxDD=-49.9%（vz=0.65 系最良）。Sharpe=+0.931。WFA 未実施。 |
| **D5** vz=0.70/lmax=5.0 | vz_thr=0.70 (E4) × lmax=5.0 | TBD | MaxDD=-54.1%。Sharpe=+0.896。WFA 未実施。 |

---

## 📈 §4 G10 WFA サマリー（vz=0.65 ロー 全5戦略）

> 実施: 2026-05-27 | ソース: `g10_wfa_vz065_lmax_row_summary.csv`
> 49窓 WFA（252日 calendar-year-anchored, non-overlapping）
> 進格条件 α: CI95_lo > 0 / 条件 β: 0.5 ≤ WFE ≤ 2.0

| Strategy | 判定 | CI95_lo | WFE | 備考 |
|----------|:----:|--------:|----:|------|
| REF-E4 | ✅ PASS | +26.51% | +1.131 | 参照行（E4 現行ベスト） |
| vz065-lmax5 | ✅ PASS | +24.82% | +1.272 | Active 候補。Sharpe 最高（WFA内）。 |
| vz065-lmax5p5 | ✅ PASS | +25.71% | +1.257 | CAGR=+34.6%, Sharpe=+0.945。 |
| vz065-lmax6 | ✅ PASS | +26.31% | +1.244 | CAGR=+35.3%, Sharpe=+0.939。CI95_lo=+26.3%（E4 比較: +26.5%）。 |
| vz065-lmax7 | ✅ PASS | +26.95% | +1.230 | CAGR=+37.0%（最高）。gap=-5.22pp⚠⚠ 過学習リスク。 |

**全5戦略 WFA PASS**: vz=0.65 レジームは統計的に頑健。
**WFE 傾向**: l_max が低いほど WFE 高（vz065-lmax5: 1.272 → 最高の OOS 効率性）。
**CI95_lo 傾向**: l_max が高いほど CI95_lo 高だが gap も拡大 → 過学習トレードオフ。

---

## 🏆 §5 採用判断サマリー

### 今回実験の総括（最高値レコード）

| 指標 | 優勝戦略 | 値 |
|------|---------|---|
| CAGR_OOS 最高 | D5 vz=0.65/lmax=7.0⚠ | +37.0% |
| Sharpe_OOS 最高 | D5 vz=0.65/lmax=5.0 (Active候補) | +0.949 |
| MaxDD 最良 | D5 vz=0.60/lmax=4.5 | −49.2% |
| Worst10Y★ 最高 | C3 yz_n=10 | +19.37% |
| P10_5Y 最高 | F10+lmax5 (Active候補) | +12.8% |
| IS-OOS gap 最小 | D5 vz=0.60/lmax=4.5 | −0.15pp |
| WFA CI95_lo 最高 | F10 ε=0.015 (Active候補) | +27.9% |
| Trades/yr 最少 | 全 E4/D5 系 | 27 |

### 実験別採否

| 実験 | 採否 | 理由 |
|------|:----:|------|
| **B4** k_lo=0 | ❌ 棄却 | k_lo=0.1 の E4 が優越。変更不要。 |
| **A1** soft regime (α=2/3/5/8) | ❌ 棄却 | MaxDD 悪化 or Sharpe 改善微小。過学習リスク高。 |
| **A2** dyn lmax | ❌ 棄却 | CAGR 低下。vol 感応 l_max は CAGR/Sharpe を下げる。 |
| **A2B** rolling VOL_REF | ❌ 棄却 | ローリング中央値 < 固定 VOL_REF → 常時 l_max 削減バイアス。根本欠陥。 |
| **A3** VoV dual gate | ❌ 棄却 | 1980-82 Volcker 期（最大 MaxDD）に無効。構造的限界確認。 |
| **C2** adaptive deadband | ❌ 不採用 | F10 と同性能。gap 改善なし。F10 ε=0.015 固定の方がシンプル。 |
| **C3** Yang-Zhang | 🔶 保留 | MaxDD/Worst10Y 改善。Sharpe やや低下。WFA 未実施→次フェーズ候補。 |
| **D5** vz=0.65 × lmax | ✅ 確認 | vz_thr=0.65 が Pareto 最適（全 l_max でベスト Sharpe）。D5 vz=0.65/lmax=5.0 が Active 候補。 |

### ◆ 現行ベスト維持 + 採用候補

> **◆ 現行ベスト: E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7)** — Active 変更なし
>
> **Active 候補（ユーザー判断待ち, WFA PASS済み）**:
> 1. **D5 vz=0.65/lmax=5.0**: Sharpe=+0.949（最高）, MaxDD=-51.8%（最良）, Trades=27（低コスト）
>    G10 PASS (CI95_lo=+24.8%, WFE=+1.272)。gap=-3.91pp⚠ が唯一の懸念。
> 2. **F10 ε=0.015**: CAGR=+36.8%（高リターン）。G7 PASS。Trades=52 高コスト・gap=-4.31pp⚠。
> 3. **F10+lmax5**: P10_5Y=+12.8%（最高）, MaxDD改善。G8 PASS。Trades=52 高コスト。
>
> **次善候補（WFA PASS済み）**:
> - **D5 vz=0.65/lmax=5.5**: CAGR=+34.6%, Sharpe=+0.945。G10 PASS。
> - **D5 vz=0.65/lmax=6.0**: CAGR=+35.3%, Sharpe=+0.939。G10 PASS。
>
> **WFA 候補（要 WFA 実施）**:
> - **C3 Yang-Zhang**: Worst10Y=+19.37%（最高）。要 WFA。
> - **D5 vz=0.65/lmax=4.5**: MaxDD=-49.9%（vz=0.65 系最良）。要 WFA。

---

## 📁 §6 一次根拠ファイル

| ファイル | 実験 | 役割 |
|----------|------|------|
| `b4_klo_zero_results.csv` | B4 / E4 | k_lo=0 実験 9指標（E4 REF含む） |
| `a1_soft_regime_klt_results.csv` | A1 | soft sigmoid 実験 9指標（α=2/3/5/8/100） |
| `a2_dyn_lmax_results.csv` | A2 | vol 感応 l_max 実験 9指標 |
| `a2b_dyn_lmax_rolling_vref_results.csv` | A2B | rolling VOL_REF 実験 9指標 |
| `a3_regime_asset_tilt_results.csv` | A3 | VoV+vz dual gate 実験 9指標 |
| `c2_adaptive_deadband_results.csv` | C2 | adaptive deadband 実験 9指標 |
| `c3_yang_zhang_results.csv` | C3 | Yang-Zhang 実験 9指標 |
| `d5_vz_lmax_grid_results.csv` | D5 | vz_thr×l_max グリッド 9指標（20 config） |
| `g10_wfa_vz065_lmax_row_summary.csv` | G10 | vz=0.65 ロー WFA サマリー（5戦略） |
| `g10_wfa_vz065_lmax_row_per_window.csv` | G10 | vz=0.65 ロー WFA 窓別詳細 |
| `f10lmax5_fullmetrics.csv` | F10+lmax5 | F10+lmax5 フル 9指標 |
| `g8_wfa_lmax5_summary.csv` | G8 | F10+lmax5 / E4-lmax5 WFA |
| `f10_epsilon_deadband_results.csv` | F10 | F10 ε sweep 9指標 |
| `g7_wfa_f10_summary.csv` | G7 | F10 ε=0.015 WFA |
| `g3_wfa_e4_summary.csv` | G3 | E4 ◆ WFA |
| `g11_wfa_all_remaining_summary.csv` | G11 | 残り 14戦略一括 WFA（B4, A1×4, A2/A2B/A3, C2/C3, D5 4行） |
| `g11_wfa_all_remaining_per_window.csv` | G11 | 残り 14戦略 WFA 窓別詳細 |
| `EVALUATION_STANDARD.md` | — | 評価基準 v1.2（10指標標準） |
| `CURRENT_BEST_STRATEGY.md` | — | ベスト戦略単一の真実 |

---

## 📝 §7 改訂履歴

| 版 | 日付 | 変更内容 |
|----|------|---------|
| **v1.0** | 2026-05-27 | 初版。Group A × 4 + Group B × 13 計 17 戦略の §3.12 準拠 9指標比較表（2 テーブル分割）。 |
| **v1.7** | 2026-05-27 | Group A/B 分割を廃止。**1 テーブル × 21 戦略** に統合。A1 α=3/5、D5 vz=0.60/lmax=5.0、vz=0.65/lmax=4.5、vz=0.70/lmax=5.0 の 4 行を追加。実験別判定サマリーを 1 テーブルに統合。 |
| **v1.8** | 2026-05-27 | **G11 一括 WFA 完了**（14戦略 / 全 PASS）→ **全 21 行に CI95_lo / WFE 数値が埋まる**。`EVALUATION_STANDARD.md` を v1.2 に更新し `OvFit` 列を §3.12 標準セットに追加。統合表ヘッダを **11 列**（10指標 + Strategy）に拡張。 |
| **v1.9** | 2026-05-28 | **OvFit + WFE 列を統合** → `Overfit(WFE)` 1列に。判定基準をIS-OOS gapベースからWFEベースに変更（✅LOW=WFE∈[0.5,2.0]）。§3.12 v1.3 / **10列**（9指標 + Strategy）。 |

---

*管理者: Kazuya Murayama*
*準拠: `EVALUATION_STANDARD.md v1.3` / `src/_sweep_format.py MD_HEADER_STRAT (9指標 / 10列)`*
