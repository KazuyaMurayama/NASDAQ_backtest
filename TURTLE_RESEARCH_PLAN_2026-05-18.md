# タートル流投資手法 NASDAQ 3xレバレッジ適用 バックテスト計画

作成日: 2026-05-18
最終更新日: 2026-05-18

> 📌 **本計画は P1-P5 タイミングシグナル研究（2026-05-18 完了, 全コンボ ADOPT 不達）の **次フェーズ** として位置付けられる**。
> 仕様根拠: [TURTLE_RESEARCH_2026-05-18.md](TURTLE_RESEARCH_2026-05-18.md) / 比較対象: [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)（DH Dyn 2x3x [A]）

---

## 0. エグゼクティブサマリー

P1-P5 で「外部シグナル乗算ゲート（マクロ・テクニカル・相関・CPPI）」は **DSR / WF-CV / Bootstrap いずれも DH Dyn [A] を統計的に超えなかった**。本計画では発想を変え、**サイジング思想そのもの**（=タートルの離散ブレイクアウト + Unit-based + 2N stop + Pyramiding）を導入し、3 つの作業仮説を検証する: H1 単独運用、H2 ゲート併用、H3 サイジング差し替え。最重要論点は「**離散シグナル × 3xレバレッジで vol-drag に耐えられるか**」「**1979-82 / 2015-16 のレンジ相場でチョップ損失が許容範囲か**」の 2 点。期待最終成果は **DH Dyn [A] の派生として [B]/[C]/[D] のいずれかを正典に追加するか、または「適用不可」の結論を 1 本のレポートに集約**する。

---

## 1. 研究目的と仮説

### 1.1 なぜタートル流を今やるのか

| 観点 | DH Dyn [A] の現状 | タートルが補える可能性 |
|---|---|---|
| サイジング | vol target × asym EWMA × slope × mom の **連続関数**。常時市場にエクスポーズ。 | 「Out」状態を持つ **離散シグナル**。大暴落入口でゼロ化できる。 |
| ストップ | リバランス閾値 0.15 のみ。明示的損切りなし。 | 2N stop = ATR ベースの **明示損切り**。テールリスクの上限を決められる。 |
| 順張り | 200MA / Slope / Mom の 3 系統だが価格水準ではなく **変化率/EWMA** を見る。 | 20/55 日 Donchian は **絶対水準ブレイク**。トレンド最強の局面でフルレバを正当化しやすい。 |
| ピラミッディング | なし（重みは連続変化のみ）。 | 0.5N 進捗で +1 Unit、最大 4 Unit。**強トレンドでの過小エクスポージャー**を防ぐ。 |

### 1.2 3 つの作業仮説

#### H1: タートル単独運用は DH Dyn [A] を超えるか

- **検証物**: TQQQ ロングのみ（または TQQQ/SQQQ 切替）にタートル System 1+2 を素朴に適用
- **採用基準**: CAGR_FULL ≥ 30.81% **かつ** Sharpe ≥ 1.298 **かつ** Worst5Y ≥ +4.77%
- **棄却基準 (falsification)**: 上記いずれか 1 つでも下回る、または MaxDD < -45%、または 2000/2008/2022 のどれかが -50% 超

#### H2: タートル離散シグナルを DH Dyn [A] にゲートとして追加すると改善するか

- **検証物**: DH Dyn [A] の wn (NASDAQ 重み) に対し、`turtle_state ∈ {long, flat, short}` を乗算
- **採用基準**: P5 まで通過し、ΔCAGR ≥ +1%, ΔSharpe ≥ +0.05, ΔWorst5Y ≥ 0%（劣化なし）
- **棄却基準**: P4 (DSR) で DSR_q05 < 0、または P5 (Bootstrap) で ΔSharpe 5%ile ≤ 0
- ⚠️ **P1-P5 で「外部ゲート」は全滅している**ため、タートルゲートだけが通る可能性は事前確率として低い

#### H3: タートルのリスク管理（Unit-based sizing + 2N stop）で DH Dyn の連続レバを置換すると改善するか

- **検証物**: DH Dyn [A] のシグナル群はそのまま **方向決定**に使い、**サイズ決定**はタートルの Unit 式に置換。エントリー後 2N で離散カット。
- **採用基準**: Worst5Y を +3% 以上改善（=テール改善が主目的）。CAGR は -2% まで許容。
- **棄却基準**: Worst5Y が悪化、または取引回数が現状 27 回/年 → 100 回/年 超（実運用コスト破綻）

---

## 2. NASDAQ 単一銘柄環境への適合論点と設計判断

タートル原典は **複数商品先物・両方向・追加証拠金前提**。NASDAQ 単独 / 3xレバレッジ ETF / ロングバイアスの環境では **8 つの論点** に明示判断が必要。

### 論点 1: ショートの扱い

| 選択肢 | 内容 |
|---|---|
| A | ロングオンリー（フラット撤退のみ） |
| B | SQQQ への切替（タートル原典の両方向に近い） |
| C | NASDAQ 現物ショート（CFD 想定、SOFR 借株コスト） |

**採用案: A（ロングオンリー）+ T2 で B も並行検証**
**理由**:
- 3x 逆 ETF (SQQQ) は volatility drag が極端で、過去 20 年の単純保有 CAGR は -40%/年 級
- タートル原典でも「ショートは商品先物の正味プレミアム構造に依存」しており、株式インデックスは **長期上昇バイアス**で同等視できない
- ただし H1 検証として T2 で SQQQ 切替版も走らせ「ロングオンリーで十分」を実証

### 論点 2: ピラミッディングと 3xレバレッジの関係

タートル原典: 1 Unit = 口座 1% リスク、最大 4 Unit = 4% リスク
**3x ETF 適用時**: 4 Unit × 3x = 名目 12% 口座変動 / 1N 動き = **実質 12% リスク**

| 選択肢 | 内容 |
|---|---|
| A | Unit 計算で N を **TQQQ の N（NASDAQ の約 3 倍）** で使用 → 自動的に Unit 数が 1/3 になり実質リスク維持 |
| B | NASDAQ の N で計算した Unit を 3x ETF にそのまま当てる → 実質 12% リスク（積極） |
| C | 最大 Unit を 4 → 2 に削減（保守） |

**採用案: A（TQQQ の N で計算）+ T7 感度分析で B/C も検証**
**理由**:
- タートル原典 5.1 の「1 Unit リスク 1%」は **商品の N で測ったリスク**。レバレッジ ETF は N が 3 倍になるので Unit 数が自動調整される
- これにより「タートルのリスク管理思想」を最も忠実に再現できる

### 論点 3: Unit 計算の 3x 調整 (`Dollar_Per_Point`)

タートル原典: `Unit = 口座 × 0.01 / (N × DPP)`、ETF では `DPP = $1/株`

**採用案**:
```python
# TQQQ をシミュレートする場合
N_TQQQ_t = ATR_Wilder(TQQQ_synthetic, span=20)  # TQQQ 合成系列のATR
Unit_shares = (equity * 0.01) / (N_TQQQ_t * 1.0)
```
**理由**: TQQQ 合成系列（NASDAQ × 3x - financing - TER）から直接 ATR を計算する方が、NASDAQ × 3 の近似より誤差が小さい。`corrected_strategy_backtest.py` で既に TQQQ 合成系列を生成済み。

### 論点 4: 離散 vs 連続シグナルの整合

- **タートル**: `{long_1u, long_2u, ..., long_4u, flat, short_1u, ..., short_4u}` = **9 状態の離散**
- **DH Dyn [A]**: wn ∈ [0.30, 0.90] の **連続**

| 選択肢 | 内容 |
|---|---|
| A | 完全並列（T1/T2 で純タートル、T3/T4 で混合） |
| B | タートル状態を `turtle_intensity = unit_count / 4` の連続値に変換 |
| C | DH Dyn を量子化して 5 段階離散にしてからタートルと AND |

**採用案: A（T1/T2 完全離散、T3/T4 で混合は乗算ゲートのみ）**
**理由**: B/C は両者の思想を混ぜすぎて「何が効いたか」が分離できなくなる。**仮説検証性**を最優先する。

### 論点 5: 複数市場ルール (4/6/10/12 Unit)

タートル原典 7.1: 単一市場 4 / 密接相関 6 / 緩相関 10 / 単方向 12

**[ユーザー判断要]** — 3 つの選択肢:

| 選択肢 | 内容 |
|---|---|
| A | TQQQ/Gold2x/Bond3x を **緩相関 3 市場として扱い、合計 10 Unit 上限** |
| B | T1/T2 は単一市場 4 Unit。T5 でのみ 3 スリーブ独立 4 Unit ずつ（=合計 12）。 |
| C | T5 で密接相関扱いの 6 Unit を上限 |

**推奨: B**（理由: TQQQ/Gold/Bond は P1-P5 期間で年次相関 ρ ∈ [-0.4, +0.6] と振れるので「緩相関」は楽観的。**保守**を選んで T5 で 3×4=12 Unit、ただし各スリーブは独立に 4 Unit 上限）

### 論点 6: ピラミッディング上限

タートル原典: 4 Unit。**株式単一インデックスでは適切か？**

**採用案: T1/T2 は原典通り 4。T7 で {2, 3, 4, 5} を感度分析。**
**理由**: NASDAQ は商品先物よりトレンドが長く（1990s、2010s）、4 Unit 上限が過小の可能性がある一方、レンジ相場（1979-82、2015-16）では 4 Unit でも whipsaw コストが嵩む。原典で固定 → 感度で揺すぶる、の 2 段で結論を出す。

### 論点 7: 取引コスト・スリッページ

既存研究の整合: `CFD_SPREAD_LOW = 0.20%`, `TQQQ TER = 0.86%`, SOFR financing 2x (TQQQ)

**採用案**:
- エントリー / ピラミッド追加 / 2N stop / 出口 = いずれも **片道 0.20% スリッページ**を控除
- 日次保有コスト: `corrected_strategy_backtest.py` の Scenario D と同一（TER + SOFR financing）
- タートルは年 27 回 → 推定 60-120 回/年 にトレード増の見込み。**スリッページ累計を必ず可視化**

**理由**: P1-P5 で「コスト未補正での見せかけの優位性」を排除した経験を踏襲。

### 論点 8: vol target との重複

タートルの Unit sizing = `equity × 1% / (N × DPP)` は、本質的に **N で正規化したリスク等量配分** = **vol target sizing** と同等。DH Dyn [A] にも vol target 成分があり、**論点 4 で並列化** すれば重複は問題にならないが、**T4 で「タートル sizing + DH Dyn signal」を作る時のみ衝突**。

**採用案**: T4 では DH Dyn の vol target 項を **無効化**し、タートル Unit のみでサイズ決定。
**理由**: 二重正規化は防御過剰になり、CAGR を不必要に削る恐れ。

---

## 3. バックテスト変種定義 (T1〜T7)

### 変種一覧表

| ID | 名称 | 方向 | サイジング | エントリー | 出口 | 想定 CAGR | 想定 Sharpe | 想定 MaxDD |
|---|---|---|---|---|---|---|---|---|
| **T1** | Pure Turtle Long-Only | Long のみ | Unit×4 | S1: 20日高値, S2: 55日高値 | 10/20日安値 or 2N stop | 18-25% | 0.7-1.0 | -40 to -55% |
| **T2** | Pure Turtle Long/Short | TQQQ/SQQQ | Unit×4 | S1+S2 両方向 | 同上 | 12-22% | 0.6-0.95 | -45 to -60% |
| **T3** | Turtle Gate × DH Dyn [A] | Long のみ | 連続 (DH Dyn) | turtle_state=Out → wn=0 | turtle_state 反転 | 26-32% | 1.20-1.40 | -28 to -35% |
| **T4** | Turtle-sized DH Dyn | Long のみ | Unit (タートル) | DH Dyn シグナル方向 + Unit 進捗 | 2N stop + DH Dyn 反転 | 24-32% | 1.15-1.35 | -25 to -33% |
| **T5** | Turtle × 3スリーブ独立 | Long のみ | Unit×4 × 3 (上限 12) | 各スリーブ独立 S1+S2 | 同左 | 20-28% | 0.9-1.2 | -30 to -42% |
| **T6** | Hybrid Stop System | Long のみ | 連続 (DH Dyn) | DH Dyn と同じ | **2N stop だけ追加**、再エントリ条件: 10日高値 | 28-32% | 1.25-1.40 | -25 to -32% |
| **T7** | 感度分析 | (各バリアントで) | — | 期間 {10,20,40}, ストップ {1.5N,2N,3N}, Unit 上限 {2,3,4,5} | — | — | — | — |

### T1: Pure Turtle Long-Only — 擬似コード

```python
# Daily loop
for t in range(20, len(df)):
    H20 = df['High'].iloc[t-20:t].max()
    L10 = df['Low'].iloc[t-10:t].min()
    H55 = df['High'].iloc[t-55:t].max()
    N = wilder_atr(df, t, span=20)

    # Entry
    if position == 0:
        if df['Close'].iloc[t] > H20 and not skip_s1:
            enter_long(unit_count=1, entry_price=H20+tick, N=N)
        elif df['Close'].iloc[t] > H55:  # S2 or failsafe
            enter_long(unit_count=1, entry_price=H55+tick, N=N)
    # Pyramid
    elif position > 0 and position < 4:
        if df['Close'].iloc[t] > last_entry + 0.5 * last_N:
            add_unit(); raise_stop_all_to(last_entry - 2*last_N)
    # Stop / Exit
    if position > 0:
        if df['Low'].iloc[t] <= stop_level:
            exit_all(reason='2N_stop')
            skip_s1 = (gain > 0)   # 勝ち撤退ならS1スキップフラグ
        elif df['Close'].iloc[t] < L10:
            exit_all(reason='10d_exit')
            skip_s1 = (gain > 0)
```

**想定される失敗パターン**:
- 1979-82: NASDAQ レンジで 20日高値ブレイクを繰り返し whipsaw → 年 10 回前後の 2N stop
- 2022: 1月の最初の 20日安値ブレイク (=Long なら撤退) でフラットになりドローダウンは抑制されるが、**3月リバウンドで偽ブレイク** → 再損切り
- 2000-02: 似たパターン。S1 スキップが有効に働くか要検証

### T2: Pure Turtle Long/Short — 擬似コード

```python
# T1 のロジックに加えて
if df['Close'].iloc[t] < L20 and not skip_s1:
    enter_short_via_sqqq(...)  # SQQQ 合成系列にエントリ
```

**想定される失敗**: SQQQ の vol drag で、短期ベア相場（2018Q4 など）以外では負ける。ロングバイアスの株式インデックスでは **構造的に不利**。

### T3: Turtle as Gate

```python
# DH Dyn [A] の最終 wn 計算後
turtle_state = compute_turtle_state(df, t)  # {'long', 'flat'}
if turtle_state == 'flat':
    wn_final = 0.0
    wg_final = wb_final = 0.5  # ヘッジに退避
else:
    wn_final = wn_dh  # DH Dyn そのまま
```

**想定される改善**: 2008/2022 で early exit、2000-02 でレンジ回避。ただし P1-P5 で「乗算ゲート系全滅」の事実があるため、**事前確率は低い**。

### T4: Turtle-sized DH Dyn

```python
# DH Dyn のシグナル方向はそのまま使う
direction = sign(dh_signal[t])  # {+1, 0}
# サイズは Unit-based
unit_count = pyramiding_logic(...)
tqqq_weight = direction * (unit_count / 4.0) * 0.9  # 最大 90% TQQQ
# 2N stop で離散カット
if equity_dd_from_peak > 2 * N:
    tqqq_weight = 0
```

**想定される改善**: テール抑制（Worst5Y）。CAGR 維持の自信は低い。

### T5: Turtle × 3 スリーブ独立

```python
for sleeve in ['nasdaq', 'gold', 'bond']:
    state[sleeve] = compute_turtle_state(sleeve_price_series, t)
    weight[sleeve] = unit_count[sleeve] / 4.0 * sleeve_max_weight
# 合計 12 Unit 上限 (論点5の選択肢B)
```

**想定される失敗**: Gold/Bond で 20/55日ブレイクは TQQQ ほど整合しない（マクロ要因駆動）。

### T6: Hybrid Stop System

```python
# DH Dyn [A] そのまま運用、ただし
entry_N = wilder_atr(...)  # エントリー時の N を記録
if equity_drawdown_from_entry > 2 * entry_N:
    force_flat(); cooldown_days = 10
# 再エントリ: 価格が 10日高値を超えたら復帰
```

**想定される改善**: T3 がダメでも T6 はサイジングを変えないので **DH Dyn [A] の劣化バージョン** に過ぎない可能性。要 P5 まで検証。

### T7: 感度分析グリッド

| 軸 | 値 |
|---|---|
| エントリー期間 (S1) | {10, 20, 40} |
| エントリー期間 (S2) | {40, 55, 80} |
| 出口期間 (S1) | {5, 10, 20} |
| ストップ倍率 | {1.5N, 2N, 3N} |
| ピラ上限 | {2, 3, 4, 5} |
| ピラ間隔 | {0.5N, 1.0N, 1.5N} |

T1 を base に 3×3×3×3×3×3=729 通り → **正則化のためグリッド粗化**して 81 通り以内。

---

## 4. 実行ロードマップ

### Phase T1: コア実装（1週間）

**目的**: タートル必須要素を再利用可能なモジュールに切り出す

| 成果物 | 内容 |
|---|---|
| `src/turtle_core.py` | `wilder_atr()`, `compute_donchian()`, `unit_size()`, `pyramid_logic()`, `stop_logic()` |
| `src/turtle_state.py` | 状態機械 `{long_Nu, flat, short_Nu}` の遷移ロジック + S1 スキップフラグ |
| `tests/test_turtle_core.py` | 既知例（タートル原典の数値例 N 計算等）でユニットテスト |
| `src/turtle_costs.py` | スリッページ・TER・SOFR の統合（`product_costs.py` を流用） |

### Phase T2: 単独バックテスト（2週間）

| 成果物 | 内容 |
|---|---|
| `src/turtle_t1_pure_long.py` | T1 を 1974-2026 で実行、年次/月次出力 |
| `src/turtle_t2_long_short.py` | T2 同上 |
| `T1_T2_RESULTS_YYYY-MM-DD.md` | CAGR/Sharpe/MaxDD/Worst5Y/取引回数/2N stop ヒット回数の表 |
| `t1_t2_yearly_returns.csv` | 年次リターン |
| `t1_t2_trade_log.csv` | 全トレード（エントリー/出口/N/Unit/PnL） |

**判定**: H1 仮説の 1 次採用基準を満たすか。満たさなければ Phase T3 へ進むも、H1 単独棄却を明記。

### Phase T3: 統合バックテスト（2週間）

| 成果物 | 内容 |
|---|---|
| `src/turtle_t3_gate.py` | T3 = タートルゲート × DH Dyn [A] |
| `src/turtle_t4_sized.py` | T4 = タートルサイジング × DH Dyn シグナル |
| `src/turtle_t5_sleeves.py` | T5 = 3 スリーブ独立 |
| `src/turtle_t6_hybrid_stop.py` | T6 = DH Dyn [A] + 2N stop だけ追加 |
| `T3_T6_RESULTS_YYYY-MM-DD.md` | 4 変種の比較表 |

### Phase T4: 過学習確認（1週間）

P4 と同じ枠組み:

| 成果物 | 内容 |
|---|---|
| `src/turtle_p4_overfitting.py` | 5-Fold WF-CV + DSR 計算 |
| `T_P4_OVERFITTING_YYYY-MM-DD.md` | T1〜T6 × {Pure, Gate, Sized, Sleeves, Hybrid} の DSR q05 一覧 |

**判定基準**: DSR_q05 > 0 でないものは Phase T5 に進めず即脱落。

### Phase T5: ストレステスト（1週間）

| 成果物 | 内容 |
|---|---|
| `src/turtle_p5_bootstrap.py` | Block Bootstrap B=2000, block_len=63 |
| `T_P5_BOOTSTRAP_YYYY-MM-DD.md` | ΔSharpe 5%ile vs DH Dyn [A] の分布 |

**判定基準**: P5 と同じ「ΔSharpe 5%ile > 0 かつ ΔMaxDD 5%ile > 0」を要求。

### Phase T6: 採用判断（1週間）

| 成果物 | 内容 |
|---|---|
| `TURTLE_FINAL_REPORT_YYYY-MM-DD.md` | 全結果サマリー、採用/棄却理由 |
| `CURRENT_BEST_STRATEGY.md` | (採用時) DH Dyn 2x3x [B/C/D] として追記 |
| `tasks.md` | 完了/棄却ログ |

---

## 5. 評価指標と採用基準

### 5.1 主要指標（FULL: 1974-01-02 〜 2026-03-26）

- CAGR_FULL
- Sharpe_FULL（無リスク = SOFR 連動）
- MaxDD_FULL
- Worst5Y CAGR（過去最悪 5 年連続 CAGR）
- WinRate（年次勝率）
- 年間取引回数
- スリッページ累計コスト

### 5.2 採用基準

| 階級 | CAGR | Sharpe | MaxDD | Worst5Y |
|---|---|---|---|---|
| **1次採用** | ≥ 30% | ≥ 1.20 | ≥ -35% | ≥ +3% |
| **2次採用** | ≥ 22% | ≥ 1.05 | ≥ -45% | ≥ -3% |
| **不採用** | 上記未達 | — | — | — |

### 5.3 失格条件（ペナルティ）

- 大暴落年 2000 / 2008 / 2022 のいずれか **単年で -50% 超** = 即失格
- 取引回数 200 回/年 超 = 失格（実運用コスト破綻）
- DSR q05 ≤ 0 = Phase T5 進行不可
- Bootstrap ΔSharpe 5%ile ≤ 0 = 不採用

### 5.4 採用クラス（DH Dyn [A] との関係）

- **クラス [B]**: 単独でも DH Dyn [A] と同等以上 → [A] を置換候補に
- **クラス [C]**: [A] を補完するゲート/サイジング → [A]+[C] のハイブリッドを正典化
- **クラス [D]**: 特定局面（例: 2022 のような Triple Bear）限定で優位 → タクティカル用途

---

## 6. 既知の懸念とリスク

### 論点 A: NASDAQ 3xレバレッジ ETF を trend-following で長期保有できるか

**懸念**: TQQQ は日次リバランスなので、レンジ相場での vol drag が大きい（年率 -5〜-10%）。タートルは **長期トレンドに乗る**戦略で、レンジでの保有はむしろ vol drag を最大化する。
**検証方法**:
- T7 感度分析で「短期ブレイク (期間 10)」が「長期ブレイク (期間 55)」より NASDAQ で有利かを確認
- レジーム別 CAGR を出す（VIX バケット / NASDAQ_50dMA 上下 / etc.）

### 論点 B: System 1 のスキップロジック発火頻度

**懸念**: スキップは「直前 S1 ブレイクが勝ちトレードなら次をスキップ」。NASDAQ で 1974-2026 にこのロジックが**実用的な頻度**（年 1-3 回程度）で発火するか不明。発火しなければスキップ機構は **形骸化**。
**検証方法**: Phase T2 で「skip フラグの ON 日数 / 年」を必ず計測。

### 論点 C: レンジ相場での劣化

**1979-82, 1994, 2004-06, 2015-16** は NASDAQ がトレンドレス。タートルの想定される失敗:
- 20日高値ブレイク → 2N stop の往復 = whipsaw コスト
- 4 Unit 積み増し前にカット = 損失が「1 Unit × 2N」で固定 = リスク管理は機能するが**収益はマイナス**
**検証方法**:
- レンジ年だけ抽出した CAGR を別個に出す
- 「年間 2N stop ヒット数 ≤ 5」を健全性指標として追跡

### 論点 D: 2008 / 2022 ベアでブレイクアウト前のカット保証

**懸念**: 2022 は 1月から 12月まで段階的下落。タートルの 55日高値ブレイクは年中ほぼ発生せず、20日高値ブレイクは数回（3月、8月）あったが偽ブレイク → 2N stop。「2N stop が連続する」状態でも口座は守られるか？
**検証方法**:
- 2008 と 2022 だけのトレードログを Phase T2 で詳細出力
- 「最大連敗回数」「連敗中のドローダウン」を別途集計
- タートル原典 7.2 の「口座 10% 下落で計算用資産 20% 削減」が発動するかを確認

### 論点 E: P1-P5 研究との重複

P1-P5 の Dyn_Corr ガード / HY スプレッドガード が「乗算ゲート系」として全滅した事実 → **T3 (Turtle Gate) も同様に全滅**する確率は高い。
**判断**: T3 単独で時間を浪費しない。Phase T3 では T3 と T6 を**同時並行**で走らせ、T6（サイジング借用）に重みを置く。

### 論点 F: タートルの「複数市場分散」前提との乖離

タートルが原典通り機能した最大の理由は **20+ 商品市場の分散**。NASDAQ 単独 = 1 市場 = タートル原典の最小単位。Phase T3 で T5（3 スリーブ独立）を必須化することでこの懸念に部分対応するが、3 はまだ少ない。
**緩和策**: T5 では Gold/Bond の独立タートルを TQQQ と完全独立に運用し、相関の動的変化を吸収する。

### 論点 G: コスト累積（年 60-120 回トレード前提）

**[ユーザー判断要]** — スリッページ控除レート:

| 選択肢 | 内容 |
|---|---|
| A | 既存研究と同じ片道 0.20%（楽観） |
| B | 片道 0.30%（中立、CFD 標準スプレッド） |
| C | 片道 0.50%（保守、Worst case） |

**推奨: B**（理由: タートル想定の年 60-120 回は DH Dyn の 27 回より 3-4 倍多く、スリッページ感度が上がる。中立シナリオで判定し、A/C を感度分析）

---

## 7. 既存コードの活用

| ファイル | 役割 | タートル研究での活用 |
|---|---|---|
| `src/corrected_strategy_backtest.py` | DH Dyn [A] Scenario D 本体 | T3/T4/T6 のベースラインとして直接 import |
| `src/product_costs.py` | コスト定数の単一の真実 | `turtle_costs.py` でそのまま使用 |
| `src/dyn_lev_backtest.py` | 動的レバの基本フレーム | T6 の 2N stop 注入箇所として参考 |
| `src/p2_single_signal_backtest.py` | P1-P5 の単独シグナルテンプレ | T1/T2 のメトリクス出力を流用 |
| `src/p4_overfitting_check.py` | DSR + 5-Fold WF-CV | Phase T4 で再利用 |
| `src/p5_bootstrap_stress.py` | Block Bootstrap | Phase T5 で再利用 |
| `src/step_yearly_returns.py` | 年次リターン生成 | Phase T2/T3 のレポーティング |
| `NASDAQ_extended_to_2026.csv` | 価格データ（OHLC） | High/Low が ATR に必要 → カラム確認 |
| `src/sleeves_extended.py` | Gold/Bond スリーブ生成 | T5 で各スリーブの OHLC が必要 → 拡張要 |
| `TURTLE_RESEARCH_2026-05-18.md` | タートル仕様書 | Phase T1 実装の根拠 |

⚠️ **NASDAQ Composite 公式は OHLC 完全ではない期間あり**（1970s）。Phase T1 で「High/Low が無い日のフォールバック」を必ず設計する（候補: Close ± 1day_volatility による合成）。

---

## 8. 想定される最終成果

### 採用シナリオ（確率: 中）

| ID | 名称 | 採用根拠 |
|---|---|---|
| **DH Dyn 2x3x [A]** | 現行ベスト（連続レバ・閾値 0.15） | THRESHOLD_SWEEP_A_REPORT_2026-04-21.md |
| **DH Turtle [B]** | T1 が H1 を満たした場合（単独運用）| TURTLE_FINAL_REPORT (Phase T6) |
| **DH Dyn+TG [C]** | T3 or T6 が H2 を満たした場合（ゲート/Stop 借用）| 同上 |
| **DH Turtle-Sized [D]** | T4 が H3 を満たした場合（サイジング借用）| 同上 |

### 棄却シナリオ（確率: 高）

P1-P5 の経験から、**全変種が DH Dyn [A] を統計的に超えない**可能性は実質的に高い。その場合の成果物は:

- `TURTLE_NEGATIVE_RESULT_YYYY-MM-DD.md`: 「タートル流は NASDAQ 単独 3xレバ環境で DH Dyn [A] を超えなかった」という ↓ ネガティブ結果を **正典として残す**
- 副産物: T6（2N stop 追加）が局所改善を見せた場合、テールリスク管理オプションとして **CFD 運用ガイドに追記**

### 期待される副産物（採否によらず）

| 副産物 | 内容 |
|---|---|
| `src/turtle_core.py` | 再利用可能な ATR/Donchian/Unit ライブラリ |
| 1974-2026 トレードログ | タートルがどの局面でどう動くかの完全な記録 |
| レンジ相場での挙動知見 | 今後のシグナル研究で「whipsaw 耐性」評価軸として活用 |
| 2N stop ヒット頻度の経験則 | 他戦略のストップ設計に転用可能 |

---

## 付録: タスク管理ハンドオフ

新セッション継続時:
1. `tasks.md` の Phase T* エントリを確認
2. `CURRENT_BEST_STRATEGY.md` に「DH Turtle [B/C/D]」が追加されていないか確認
3. 本ファイル §4 のチェックリストを Phase 単位で完了/棄却マーク

---
*計画立案: Opus (2026-05-18)*
*関連: [TURTLE_RESEARCH_2026-05-18.md](TURTLE_RESEARCH_2026-05-18.md) / [TIMING_STRATEGY_RESEARCH_PLAN_2026-05-18.md](TIMING_STRATEGY_RESEARCH_PLAN_2026-05-18.md) / [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)*
