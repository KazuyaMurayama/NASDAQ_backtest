# Session Summary — 2026-05-18 (P1〜P5 タイミング戦略 / Phase T1-T2 タートル研究)

作成日: 2026-05-18
最終更新日: 2026-05-18

> 📌 本セッションで実施した全バックテスト・結果・次フェーズ計画の集約。
> 次セッション開始時はこのファイル + [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) + [tasks.md](tasks.md) を参照。

---

## 0. 結論サマリー (1行)

**現行ベスト `DH Dyn 2x3x [A]` は本セッションで検証した全ての候補戦略（P1-P5 タイミングゲート × 38 + 34コンボ、Phase T1-T2 純タートル × 2）を統計的に超えるものを生み出さなかった。**ただし Phase T3 (タートル要素を DH Dyn に部分注入) の可能性は残存。

---

## 1. 本セッションで完了したフェーズ

| Phase | 内容 | 主成果物 | 結論 |
|---|---|---|---|
| **P1** | タイミング戦略データ取得 (7 FRED系列) | [P1_DATA_FETCH_RESULTS_2026-05-18.md](P1_DATA_FETCH_RESULTS_2026-05-18.md) | data/timing_signals_raw.csv 生成 |
| **P2** | Top5シグナル単独バックテスト (38コンボ) | [P2_SINGLE_SIGNAL_RESULTS_2026-05-18.md](P2_SINGLE_SIGNAL_RESULTS_2026-05-18.md) | Dyn_Correlation のみが2022を実質削減。OOS≥20% 未達 |
| **P3** | シグナル組合せバックテスト (34コンボ) | [P3_COMBINATION_RESULTS_2026-05-18.md](P3_COMBINATION_RESULTS_2026-05-18.md) | 全コンボ CAGR_OOS<20%。Best secondary: HY×CPI |
| **P4** | 過学習確認 (DSR + 5-Fold WF-CV) | [P4_OVERFITTING_CHECK_2026-05-18.md](P4_OVERFITTING_CHECK_2026-05-18.md) | ADOPT 0。Dyn系3コンボ GRAY (PSR 0.92-0.93) |
| **P5** | ブロックブートストラップ (B=2000, L=21) | [P5_BOOTSTRAP_STRESS_2026-05-18.md](P5_BOOTSTRAP_STRESS_2026-05-18.md) | ΔSharpe 5%ile ≤ 0 全コンボ。GRAY 維持 / REJECT 確定 |
| **タートル研究計画** | Opus設計、論点5/G決定 | [TURTLE_RESEARCH_2026-05-18.md](TURTLE_RESEARCH_2026-05-18.md) / [TURTLE_RESEARCH_PLAN_2026-05-18.md](TURTLE_RESEARCH_PLAN_2026-05-18.md) | 論点5: B (max 4 Unit), 論点G: B (0.30%/side) |
| **Phase T1** | タートルコアモジュール実装 | src/turtle_core.py / turtle_state.py / turtle_costs.py + 42 ユニットテスト | 全件グリーン |
| **Phase T2** | T1/T2 純タートルバックテスト | [T1_T2_RESULTS_2026-05-18.md](T1_T2_RESULTS_2026-05-18.md) | **T1 REJECT** (CAGR 29.66%, MaxDD -60.85%) / **T2 BLOW-UP** (1981口座破綻) |

---

## 2. 本セッションで検証した主要戦略 (6戦略の指標一覧)

### 期間定義
- **FULL**: 1974-01-02 〜 2026-03-26 (52.23年, 13169 bars)
- **IS**: 1974-01-02 〜 2021-05-07
- **OOS**: 2021-05-08 〜 2026-03-26

### 指標表

| # | 戦略 | CAGR_IS | CAGR_OOS | CAGR_FULL | Sharpe_FULL | MaxDD_FULL | Worst5Y | Worst10Y | トレード/年 | 出典 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **DH Dyn 2x3x [A]** (Scenario D 補正・現行ベスト) | n/a | n/a | **+22.50%** | **0.993** | -45.08% | +0.87% | n/a | ~27 | [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) |
| 2 | DH Dyn 2x3x [A] (補正前正典) | n/a | n/a | +30.81% | 1.298 | -31.36% | +4.77% | ~+12%(類似戦略) | ~27 | CLAUDE.md / [THRESHOLD_SWEEP_A_REPORT_2026-04-21.md](THRESHOLD_SWEEP_A_REPORT_2026-04-21.md) |
| 3 | P02_Dyn×CPI [mult] (Best 2022防御) | +22.18% | **+19.43%** | n/a | 0.833 (OOS) | -46.37% | **+0.49%** | n/a | ~27 (連続リバランス) | [P3_COMBINATION_RESULTS_2026-05-18.md](P3_COMBINATION_RESULTS_2026-05-18.md) |
| 4 | P05_HY×CPI [mult] (Best Worst5Y secondary) | +25.93% | +15.65% | n/a | 0.667 (OOS) | -44.98% | **+6.04%** | n/a | ~27 | 同上 |
| 5 | P01_Dyn×HY [mult] (Best DSR候補) | +22.24% | +19.92% | n/a | 0.829 (OOS) | -42.85% | -0.25% | n/a | ~27 | 同上 |
| 6 | **T1 Pure Turtle Long-Only** (本セッション) | **+33.27%** | **-0.66%** | +29.66% | 0.986 | **-60.85%** | **-14.86%** | **-7.95%** | **4.48** | [T1_T2_RESULTS_2026-05-18.md](T1_T2_RESULTS_2026-05-18.md) |
| 7 | **T2 Pure Turtle Long/Short** (本セッション) | -6.13% | 0.00% (frozen) | -5.57% | 0.584 | **-105.46%** ⚠ | -98.20% | -86.57% | 2.43 | 同上 |

### 戦略の位置付け (定性評価)

| 戦略 | 強み | 弱み | 判定 |
|---|---|---|---|
| 1. DH Dyn [A] (Scenario D) | リアルなコスト後の現行ベスト | CAGR 22.5%、MaxDD -45% は妥協 | **採用中** |
| 2. DH Dyn [A] (補正前) | 文献値の参照点 | コスト補正前なので実運用と乖離 | 参考のみ |
| 3. P02_Dyn×CPI | 2022 防御 (-14.33% vs -30.55%)、Worst5Y > 0 | OOS 19.4% で primary 未達。DSR GRAY | **GRAY** (Phase T6で再評価検討) |
| 4. P05_HY×CPI | Worst5Y +6.04% (secondary best) | DSR REJECT、2022 防御弱い | REJECT (P4) |
| 5. P01_Dyn×HY | Sharpe_OOS 0.829 (best) | Worst5Y -0.25%、DSR GRAY | **GRAY** |
| 6. T1 Pure Turtle | **2022 -0.99%** (vs -30.55%、+29.6pp 優位)、1981/1988 大幅優位 | レンジ相場の whipsaw 致命的、MaxDD -60.85% | **REJECT** (が要素抽出余地あり) |
| 7. T2 Pure Turtle L/S | (なし) | 1981 で口座破綻 | **REJECT (BLOW-UP)** |

⚠ MaxDD -105% は実運用では不可能 (口座が負残高にならない)。本セッション T2 ではブローカー強制決済を模した 95% drawdown floor で停止しているが、MTM計算上の最大ドローダウンとして -105% を記録。

---

## 3. 年次リターン参照先 (MD/CSV ファイル一覧)

チャットに載せきれない年次データは以下を参照:

| 戦略 | 年次リターンファイル |
|---|---|
| DH Dyn 2x3x [A] | [YEARLY_RETURNS_REPORT_2026-05-12_v4.md](YEARLY_RETURNS_REPORT_2026-05-12_v4.md) / [YEARLY_RETURNS_REPORT_2026-04-20_v3.md](YEARLY_RETURNS_REPORT_2026-04-20_v3.md) (補正前) |
| P01〜P10 タイミング組合せ | [P3_COMBINATION_RESULTS_2026-05-18.md](P3_COMBINATION_RESULTS_2026-05-18.md) §"年次リターン" |
| P2 単独シグナル38コンボ | [P2_SINGLE_SIGNAL_RESULTS_2026-05-18.md](P2_SINGLE_SIGNAL_RESULTS_2026-05-18.md) |
| T1 Pure Turtle Long-Only | [t1_yearly_returns.csv](t1_yearly_returns.csv) + [T1_T2_RESULTS_2026-05-18.md](T1_T2_RESULTS_2026-05-18.md) §4 |
| T2 Pure Turtle Long/Short | [t2_yearly_returns.csv](t2_yearly_returns.csv) + [T1_T2_RESULTS_2026-05-18.md](T1_T2_RESULTS_2026-05-18.md) §4 |
| Buy & Hold 3x / A2 Optimized / Dyn-Hybrid 各種 (旧研究) | [ADDITIONAL_ANALYSIS_REPORT_2026-03-30.md](ADDITIONAL_ANALYSIS_REPORT_2026-03-30.md) (Worst10Y 計算あり) |

---

## 4. 次セッション継続ポイント (Phase T3〜T6)

### Phase T3 (T3-T6 統合バックテスト) — 優先順位

| 変種 | 構成 | 事前確率 | 期待効果 |
|---|---|---|---|
| **T6 Hybrid Stop** | DH Dyn [A] + 2N stop 注入のみ (サイジングは連続のまま) | **最有望** | T1の「2022 -0.99%」防御効果を借用 |
| T4 Sized | DH Dyn シグナル × turtle Unit sizing | 中 | テール抑制 |
| T3 Gate | DH Dyn [A] × turtle_state ∈ {long, flat} | 低 | P1-P5で乗算ゲート系は全滅 |
| T5 Sleeves | TQQQ + Gold2x + Bond3x 独立 turtle (12 Unit max) | 低-中 | Gold/Bond は商品先物ロジック親和性低 |

### 実装ファイル (次セッション作業対象)
- `src/turtle_t3_gate.py`
- `src/turtle_t4_sized.py`
- `src/turtle_t5_sleeves.py`
- `src/turtle_t6_hybrid_stop.py`
- `T3_T6_RESULTS_YYYY-MM-DD.md`

### Phase T4 (DSR + WF-CV) / T5 (Bootstrap) / T6 (採用判断)
- Phase T3 で通過した変種のみ進める (P4/P5 スクリプト再利用)

---

## 5. 確定済み設計判断 (再質問不要)

| 論点 | 決定 | 確定日 |
|---|---|---|
| 論点5 (タートル複数市場ルール) | **B: T1/T2 max 4 Unit、T5 各スリーブ独立 4×3=12 Unit** | 2026-05-18 |
| 論点G (スリッページ) | **B: 0.30%/side (全トレード適用)** | 2026-05-18 |
| 論点1 (ショート扱い) | A (Long-only) を主軸 + T2 で B (SQQQ) 並行検証 → T2は1981で破綻、ロング限定で確定 | 2026-05-18 |
| 論点2 (3xレバ Unit 計算) | A (TQQQ合成系列のATR を使用、自動的に Unit 1/3) | 2026-05-18 |
| 論点3 (DPP) | TQQQ 合成系列 ATR を直接使用 | 2026-05-18 |
| 論点4 (離散vs連続混合) | A (完全並列。T1/T2純離散、T3/T4で混合は乗算ゲートのみ) | 2026-05-18 |
| 論点6 (ピラ上限) | T1/T2 は原典通り 4、T7 で {2,3,4,5} 感度分析 | 2026-05-18 |
| 論点7 (取引コスト) | 上記スリッページ + TQQQ TER 0.86% + 2×SOFR financing + 0.50% swap | 2026-05-18 |
| 論点8 (vol target 重複) | T4 では DH Dyn vol target を無効化 | 2026-05-18 |

---

## 6. P1-P5 全72コンボの最終判定

| Combo | P4_DSR | P5_Bootstrap | 最終判定 |
|---|---|---|---|
| Baseline | REJECT | MARGINAL | **REJECT** |
| P01_Dyn×HY | GRAY (PSR 0.931) | MARGINAL (p_val 0.091) | **GRAY** |
| P02_Dyn×CPI | GRAY (PSR 0.932) | MARGINAL (p_val 0.088) | **GRAY** |
| P03_Dyn×MA | GRAY (PSR 0.922) | MARGINAL (p_val 0.133) | **GRAY** |
| P05_HY×CPI | REJECT (PSR 0.870) | MARGINAL | **REJECT** |
| P06_HY×MA | REJECT (PSR 0.844) | FRAGILE | **REJECT** |
| その他 66 コンボ | REJECT | n/a | **REJECT** |

→ ADOPT 確定ゼロ。Dyn系3コンボは GRAY 維持 (運用判断保留)、それ以外は REJECT。

---

## 7. 実装スクリプト (本セッション追加分)

| ファイル | 役割 |
|---|---|
| [src/p1_fetch_timing_data.py](src/p1_fetch_timing_data.py) | P1 FREDデータ取得 |
| [src/p2_single_signal_backtest.py](src/p2_single_signal_backtest.py) | P2 単独シグナル38コンボ |
| [src/p4_overfitting_check.py](src/p4_overfitting_check.py) | P4 DSR + 5-Fold WF-CV |
| [src/p5_bootstrap_stress.py](src/p5_bootstrap_stress.py) | P5 Stationary Block Bootstrap |
| [src/turtle_core.py](src/turtle_core.py) | wilder_atr / Donchian / unit_size |
| [src/turtle_state.py](src/turtle_state.py) | TurtleState 状態機械 |
| [src/turtle_costs.py](src/turtle_costs.py) | slippage / holding cost |
| [src/turtle_data.py](src/turtle_data.py) | TQQQ/SQQQ 合成OHLC ビルダー |
| [src/turtle_sim.py](src/turtle_sim.py) | T1/T2 シミュレーター |
| [src/turtle_t1_pure_long.py](src/turtle_t1_pure_long.py) | T1 ドライバ |
| [src/turtle_t2_long_short.py](src/turtle_t2_long_short.py) | T2 ドライバ |
| [tests/test_turtle_core.py](tests/test_turtle_core.py) | 42 ユニットテスト |

---

## 8. 次セッション開始プロンプト (引き継ぎ用)

```
本セッションでは Phase T3 (タートル要素の DH Dyn [A] への部分注入) を実装したい。
SESSION_SUMMARY_2026-05-18.md を参照して、優先順位通り T6 (Hybrid Stop) から実装してほしい。

具体作業:
1. src/turtle_t6_hybrid_stop.py 作成
   - DH Dyn [A] のサイジング・シグナルはそのまま
   - 2N stop だけ追加: equity drawdown from entry > 2*entry_N → force flat
   - 再エントリ条件: 価格が 10日高値を超えたら復帰
2. src/corrected_strategy_backtest.py のロジックを基盤として流用
3. バックテスト実行: 1974-2026, IS=〜2021-05-07, OOS=2021-05-08〜
4. T6_RESULTS_YYYY-MM-DD.md 作成
   - 比較: T6 vs DH Dyn [A] (P2 baseline)
   - 採用基準: CAGR_FULL ≥ 22.50% かつ Worst5Y ≥ +0.87% かつ MaxDD改善 (>-45.08%)
5. 通過したら Phase T4 (DSR) へ
```
