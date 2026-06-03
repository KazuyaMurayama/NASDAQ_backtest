# 新規シグナル探索 設計仕様 (Phase A〜C)

作成日: 2026-06-03
最終更新日: 2026-06-03

> 本ドキュメントは brainstorming セッション (`superpowers:brainstorming` skill) で確定した設計仕様。
> 後続の writing-plans で実行計画 (タスク粒度の TODO 化) に展開する。

---

## 0. 目的・背景

現行 Active 戦略 (`NEW CANDIDATE` / `E4 Regime k_lt` / `DH-Z2`) は WFA をパスしているが、以下の弱点が残存:

- MaxDD -65〜66% (`NEW CANDIDATE` / `F10`)
- 2021/2025 bull rally で -10〜16pp のミスストック
- Bootstrap P=90.7% / Permutation p=7.0% の marginal (`g20d/g20e`)

既存の入力は **価格由来シグナル** (LT2-N750 モメンタム、`vz_thr` レジーム、`F10ε` tilt、`lev_mod_065` mask) のみ。本プロジェクトは、トレーダー・アクチュアリー・ヘッジファンドマネージャーの 3 視点から **マクロ・センチメント・ファンダメンタル・ボラ構造・ポジショニング** など非価格系シグナルを系統的に洗い出し、定量・定性問わず 0/1 または 0-3 段階に量子化して、NASDAQ / Gold / Bond の先行指標としての有効性を検証する。

最終的に、検証通過シグナルを `GAS (Google Apps Script) で日次取得 → 既存戦略へ注入` する運用パイプラインへ繋ぐ。

---

## 1. スコープ

本仕様の対象は **Phase A〜C** のみ。

| Phase | 内容 | 本仕様 |
|---|---|---|
| A | 信号候補の理論的洗い出し + Tier1 採択 (52→約30) | ◎ 対象 |
| B | 経験的スクリーニング (IC / 予測力 / 安定性) で約30→約12 | ◎ 対象 |
| C | WFA 組込検証 (G1-G11) で約12→3-6 採択 | ◎ 対象 |
| D | GAS 日次取得パイプライン実装 | 対象外 (別仕様) |
| E | 本番運用切替・既存戦略統合 | 対象外 (別仕様) |

---

## 2. 確定制約

| # | 項目 | 値 | 理由 |
|---|---|---|---|
| 1 | 主目的 | CAGR 押し上げ + IS-OOS 頑健性 | 2021/2025 ミスストック改善 + marginal 解消 |
| 2 | 副目的 | MaxDD 削減 / Gold/Bond 配分 動的化 | Pareto 改善があれば歓迎 (主目的優先) |
| 3 | データ予算 | 〜$50/月 | Polygon Starter / Finnhub / Alpha Vantage Premium 等 |
| 4 | 統合方針 | Phase B 結果後に決定 (overlay と standalone の両モードを Phase C で並列検証) | overlay/standalone どちらが向くかは信号性質依存 |
| 5 | 資産優先度 | NASDAQ 主軸 / Gold・Bond は cross-asset 信号 | 既存 DH-Z2 構造を尊重 |
| 6 | 量子化規格 | 全信号 0/1 binary または 0-3 の 4 段階に正規化 | 運用一貫性 + 既存 `lev_mod_065` mask 風の組込容易性 |
| 7 | Look-ahead 防止 | t-1 公表ベース固定 (publication_lag を信号メタデータで保持) | バックテスト一貫性 |
| 8 | 量子化方針 | 経済的根拠で閾値を理論先 (event-driven は binary、連続量は z-score quantile cut) | Tier1 で確定済み |

---

## 3. アプローチ: Hybrid 3-Tier Funnel

```
Tier 1 (Phase A): 52 候補 → 30 採択
  └ 理論的妥当性レビュー (3視点 ×10カテゴリ ×3資産 マトリクス)
  └ ◎/○/△ ランキング → ◎+○ を採択

Tier 2 (Phase B): 30 → 12 通過
  └ IC / Hit rate / 安定性 / BH-FDR 多重補正
  └ Composite signal 探索 (主成分合成)

Tier 3 (Phase C): 12 → 3-6 採択
  └ Overlay モード × Standalone モード 両並列検証
  └ G1/G3/G7/G8/G9/G10/G11 SPA test
  └ Pareto判定 + Hard 統計 requirement
```

各 Tier の機能分離: Tier1 = 経済的根拠 prior、Tier2 = 経験的選別、Tier3 = 頑健性検証。

---

## 4. Phase A: 信号タクソノミー

### 4.1 マトリクス構造

| 軸 | 値 |
|---|---|
| 視点 | Trader / Actuary / Hedge Fund Manager (3視点) |
| カテゴリ | A:Breadth / B:Vol / C:Sentiment / D:Credit / E:YieldCurve / F:MacroNowcast / G:Earnings / H:CrossAsset / I:Calendar / J:NLP/Search (10カテゴリ) |
| ターゲット資産 | NASDAQ(N) / Gold(G) / Bond(B) |
| 量子化 (Q) | 0/1 binary または 0/1/2/3 4段階 |
| 優先度 (P) | ◎=Tier1最優先 / ○=採択 / △=Tier1棄却 |

### 4.2 候補リスト (52信号、Tier1 採択 ◎+○ = 約46、△ 棄却 = 6)

#### A. Breadth & Microstructure (Trader)

| # | 信号 | Tgt | データソース | Q | P |
|---|---|---|---|---|---|
| 1 | NDX 200DMA超え銘柄% | N | Yahoo Finance / Finnhub | 0/1/2/3 (>80/60/40/<40%) | ◎ |
| 2 | McClellan Oscillator (NDX) | N | StockCharts相当 (計算) | 0/1 (>+50/<-50) | ◎ |
| 3 | NDX New Highs - New Lows (52W) | N | Polygon Starter | 0/1 (拡張/縮小) | ◎ |
| 4 | A/D Line vs 価格ダイバージェンス | N | 計算 | 0/1 | ○ |
| 5 | NYSE TICK 終値分布 | N | Yahoo Finance | 0/1/2 | △ (高頻度ノイズ) |

#### B. Volatility Regime (Trader/Actuary)

| # | 信号 | Tgt | データソース | Q | P |
|---|---|---|---|---|---|
| 6 | VIX 絶対水準 | N | Yahoo Finance | 0/1/2/3 (<15/<22/<30/>30) | ◎ |
| 7 | VIX9D / VIX 比率 | N | Yahoo Finance | 0/1 (>1.0=stress) | ◎ |
| 8 | VIX Term Structure (1/2/3) | N | CBOE / Polygon | 0/1 (contango/backwardation) | ◎ |
| 9 | VVIX (vol of vol) | N | Yahoo Finance | 0/1/2 | ◎ |
| 10 | MOVE Index (bond vol) | B,N | Yahoo Finance / FRED | 0/1/2 | ◎ |
| 11 | GVZ (gold vol) | G | CBOE / Yahoo | 0/1 | ○ |

#### C. Sentiment & Positioning (Trader/HF)

| # | 信号 | Tgt | データソース | Q | P |
|---|---|---|---|---|---|
| 12 | CBOE PutCall Ratio (Equity) | N | CBOE公開 | 0/1/2 (極値) | ◎ |
| 13 | AAII Bull-Bear Spread (週次) | N | AAII公開 | 0/1/2 | ◎ |
| 14 | NAAIM Exposure Index (週次) | N | NAAIM公開 | 0/1/2 | ○ |
| 15 | CFTC CoT NQ Non-Commercial Net (週次) | N | CFTC公開 | 0/1/2 (z-score) | ◎ |
| 16 | CFTC CoT GC Net | G | CFTC公開 | 0/1/2 | ◎ |
| 17 | CFTC CoT ZB/ZN Net | B | CFTC公開 | 0/1/2 | ◎ |
| 18 | QQQ Daily Net Creation/Redemption | N | ICI / Finnhub | 0/1 | ○ |
| 19 | GLD/TLT Net Flows | G,B | 同上 | 0/1 | ○ |
| 20 | Margin Debt YoY (FINRA月次) | N | FINRA公開 | 0/1/2 | △ (月次粗い) |

#### D. Credit & Funding Stress (Actuary)

| # | 信号 | Tgt | データソース | Q | P |
|---|---|---|---|---|---|
| 21 | ICE BofA HY OAS | N | FRED (BAMLH0A0HYM2) | 0/1/2 | ◎ |
| 22 | ICE BofA IG OAS | N | FRED (BAMLC0A0CM) | 0/1 | ○ |
| 23 | HY-IG スプレッド差 | N | 計算 (21-22) | 0/1 | ◎ |
| 24 | SOFR-IORB スプレッド | B | NY Fed | 0/1 | ○ |
| 25 | 3M Treasury - SOFR | B,N | FRED | 0/1 | △ (LIBOR後継、データ短い) |

#### E. Yield Curve & Rates (Actuary/HF)

| # | 信号 | Tgt | データソース | Q | P |
|---|---|---|---|---|---|
| 26 | 2s10s スプレッド | B,N | FRED (T10Y2Y) | 0/1/2 | ◎ |
| 27 | 3M10Y スプレッド | B,N | FRED (T10Y3M) | 0/1/2 | ◎ |
| 28 | 10Y TIPS 実質金利 | G,N | FRED (DFII10) | 0/1/2/3 | ◎ |
| 29 | 5Y5Y 先物インフレ期待 | G,B | FRED (T5YIFR) | 0/1/2 | ◎ |
| 30 | CME FedWatch 25bp利下げ確率 (3M先) | B,N | CME公開 | 0/1/2/3 | ◎ |
| 31 | 10Y - 2Y Real Yield スプレッド | G | 計算 | 0/1 | ○ |

#### F. Macro Nowcasting (HF)

| # | 信号 | Tgt | データソース | Q | P |
|---|---|---|---|---|---|
| 32 | Atlanta Fed GDPNow | N,B | Atlanta Fed公開 | 0/1/2/3 | ◎ |
| 33 | NY Fed Nowcast | N,B | NY Fed公開 | 0/1/2 | ○ |
| 34 | Citi Economic Surprise Index (USMI) | N,B,G | scrape (有料tier必要可能性) | 0/1/2 | ◎ |
| 35 | Cleveland Fed Inflation Nowcast | G,B | Cleveland Fed | 0/1/2 | ○ |
| 36 | Chicago Fed NFCI | N | FRED (NFCI) | 0/1/2 | ◎ |

#### G. Earnings & Valuation (HF)

| # | 信号 | Tgt | データソース | Q | P |
|---|---|---|---|---|---|
| 37 | NDX Forward EPS Revision Breadth (4wk) | N | EODHD / Finnhub | 0/1/2 | ◎ |
| 38 | Equity Risk Premium (fwd yld - 10Y real) | N | 計算 | 0/1/2/3 | ◎ |
| 39 | NDX Forward P/E z-score | N | 計算 | 0/1/2 | ○ |
| 40 | Mag-7 EPS Revision Composite | N | Finnhub aggregated | 0/1/2 | ◎ |

#### H. Cross-Asset & FX (HF/Trader)

| # | 信号 | Tgt | データソース | Q | P |
|---|---|---|---|---|---|
| 41 | DXY 週次変化 | G,N | Yahoo / FRED | 0/1 | ◎ |
| 42 | Copper/Gold 比率 | N,B | 計算 | 0/1/2 | ◎ |
| 43 | Silver/Gold 比率 | G | 計算 | 0/1 | ○ |
| 44 | Oil (WTI) 5日変化率 | N,B | Yahoo | 0/1 | ○ |
| 45 | BTC/QQQ 相関 | N | 計算 | 0/1 | △ (証拠薄) |

#### I. Calendar / Seasonality (Trader)

| # | 信号 | Tgt | データソース | Q | P |
|---|---|---|---|---|---|
| 46 | FOMC ブラックアウト期間 (±7日) | N,B | Fed公開 | 0/1 | ◎ |
| 47 | NDX 決算シーズン (Mag-7) | N | Earnings Calendar | 0/1 | ○ |
| 48 | Triple Witching Friday (±3日) | N | 計算 | 0/1 | △ (既往否定) |

#### J. Behavioral / NLP / Search (HF)

| # | 信号 | Tgt | データソース | Q | P |
|---|---|---|---|---|---|
| 49 | Google Trends "recession" 90日Z | N,B,G | Google Trends公開 | 0/1/2 | ◎ |
| 50 | Fed議事要旨 hawkish-dovish (NLP) | B,N | FOMC文書 + LLM API | 0/1/2/3 | ◎ |
| 51 | Headline News Risk-off Composite (NLP) | N,B,G | News API + LLM | 0/1/2 | ○ |
| 52 | Google Trends "TQQQ"/"QQQ" 検索熱 | N | Google Trends | 0/1 | ○ |

### 4.3 Tier1 結果

| 区分 | 件数 | 内訳 |
|---|---|---|
| 総候補 | **52** | 10カテゴリ |
| ◎ Tier1 最優先 | **31** | #1,2,3,6,7,8,9,10,12,13,15,16,17,21,23,26,27,28,29,30,32,34,36,37,38,40,41,42,46,49,50 |
| ○ Tier1 採択 (補助) | **16** | #4,11,14,18,19,22,24,31,33,35,39,43,44,47,51,52 |
| △ Tier1 棄却 | **5** | #5 (TICK 高頻度ノイズ), #20 (Margin Debt 月次粗い), #25 (3M-SOFR データ短い), #45 (BTC/QQQ 証拠薄), #48 (Triple Witching 既往否定) |

**Phase B 入力方針**: ◎ 全 31 を core 入力、○ 16 は予備プール (◎ の同一カテゴリ内で取得失敗・データ不足が発生した場合の代替)。実効的に Phase B では 31 信号 × 3資産 × 3 horizon = **279 検定** を BH-FDR 補正下で評価する。

> ◎ 31 の段階で当初目標の "約 30" を満たすため、Phase B 入力は **◎ 全数の 31 信号** で確定。○ は安全策として保持。

### 4.4 Phase A 成果物

1. `signals/taxonomy_20260603.md` — 上記タクソノミー (本仕様から自動転記可)
2. `signals/data_lineage.md` — 各信号の source / publication_lag / earliest_date / cost_tier
3. `signals/tier1_selection_<date>.csv` — Phase B 入力候補リスト (Q/P/メタデータ付)

---

## 5. Phase B: 経験的スクリーニング

### 5.1 評価フレーム

| 評価軸 | 採用指標 | 推奨閾値 |
|---|---|---|
| 予測力 | Spearman 順位 IC (rolling 252d 平均) | \|IC\| > 0.05 |
| 統計的有意性 | t統計 → BH-FDR | FDR < 10% |
| 安全マージン | 条件付きヒット率 Wilson 下限 | base_rate + 3pp 以上 |
| 時系列安定性 | 期間半分割 IC 同符号 + Decade IC | 同符号 + 各decade \|IC\|>0.02 |

緩和オプション: 上記で 0 通過の場合のみ IC>0.03 / FDR<15% で再実行 (フォールバック)。

### 5.2 Forward Horizon

| 期間 | 用途 | ヘッドライン? |
|---|---|---|
| 5d | 短期ノイズ寄り | 補助 |
| **20d** | 既存戦略の regime shift スケール | ヘッドライン |
| 60d | ファンダ信号の効き | 補助 |
| 252d | 長期アンカー | 採用判定外 (検証用) |

ターゲット変数: **TQQQ / TLT / GLD の対数リターン** (取引商品基準)。NDX / TNX / GC1 現物は sanity check のみ。

### 5.3 Look-ahead 防止 (publication_lag)

| 信号タイプ | 適用日 |
|---|---|
| 日次 (VIX, FRED 日次) | t-1 close → t open |
| 週次 (CoT, AAII, NAAIM) | Tue公表 → 次 Tue close |
| 月次 (Margin Debt) | 公表日 +1 営業日 |
| イベント (FOMC NLP) | 当日 21:00 ET → 翌日 open |

実装: `src/signals/timing.py` に publication_lag メタ管理。

### 5.4 多重検定補正

総検定 = **31信号 × 3資産 × 3 horizon = 279**。BH-FDR 10% (主)、Bonferroni (参考)。
Composite 探索 (B-6) 追加時は最大 +12 検定。

### 5.5 採用判定ルール (Tier2 → Tier3)

```
PASS if (
    headline (20d) で BH-FDR<10% かつ |IC|>0.05
    AND ヒット率 Wilson下限 > base_rate + 3pp
    AND IS-OOS 半分割 IC 同符号
)
OR
PASS if (
    20d/60d 連続で |IC|>0.04 (安定)
    AND 経済的根拠で説明される方向性一致
    AND Tier1 ◎ flag
)
```

### 5.6 Composite Signal 探索

相関ブロック内の合成 (PCA 第1主成分) も同規格で評価:

| ブロック | 構成 |
|---|---|
| Sentiment Composite | PutCall + AAII + NAAIM + CoT NQ |
| Credit Stress Composite | HY OAS + HY-IG diff + NFCI |
| Macro Nowcast Composite | GDPNow + Citi Surprise + NFCI |
| Yield Curve Composite | 2s10s + 3M10Y + FedWatch |

最大 **+3〜4 合成信号** を Phase C に追加可能。

### 5.7 データ長別の取扱

- **15年+ カバー**: 標準フル評価
- **10-15年**: Decade test 緩和 (2010s/20s)
- **<10年**: Tier2 で除外 (Phase C で再考可)

### 5.8 Phase B 成果物

1. `signals/scorecard_<asset>_<horizon>.csv` — 全 270 組合せの IC/p/hit/stability
2. `signals/screening_report_<date>.md` — 採用 12 + 棄却 18 の理由付き
3. `signals/correlation_heatmap.png` — 採用信号間相関 (Phase C 重複排除)
4. `signals/data_lineage.md` (更新) — 各信号の確定 source / publication_lag / earliest_date

### 5.9 想定通過数 (カテゴリ別)

| カテゴリ | Tier1 ◎ | Phase B 通過予測 |
|---|---|---|
| Breadth | 3 | 1-2 |
| Vol | 5 | 2-3 |
| Sentiment | 5 | 2-3 |
| Credit | 2 | 1-2 |
| Yield/Rate | 5 | 2-3 |
| Macro | 3 | 1-2 |
| Earnings | 3 | 1 |
| Cross-Asset | 2 | 1 |
| Calendar | 1 | 0-1 |
| NLP/Search | 2 | 1 |
| **小計 (単独)** | **31** | **13-19** |
| Composite (探索) | (+4 候補) | (+1-2) |
| **合計 (Phase C 入力)** | | **14-21 → 12-15 に絞込** |

絞込ロジック: Tier2 通過 14-21 のうち、相関ブロック (5.6) で代表 1 を残し他を統合 (Composite に吸収)、加えて Hard要件 (publication_lag 安定、データ長 10年+) を最後の絞り。結果 **12-15 信号 + 1-2 合成** が Phase C 入力。

### 5.10 実装スタック

| 部品 | 採用 |
|---|---|
| 信号取得 (バックテスト) | 既存 `src/data_loaders/` + 新規 `src/data_loaders/signals/` |
| 量子化 | 新規 `src/signals/quantize.py` |
| IC計算 | `scipy.stats.spearmanr` + pandas rolling |
| FDR | `statsmodels.stats.multitest.multipletests(method='fdr_bh')` |
| t統計 | `statsmodels` Newey-West |
| Reporting | Jupyter notebook or markdown 生成スクリプト |

---

## 6. Phase C: WFA 組込検証

### 6.1 2モード並列検証

各 Phase B 通過信号 (12-15) を **両モード** で評価:

| モード | 構造 | 信号→出力 |
|---|---|---|
| **Overlay** | 既存 NEW CANDIDATE / E4 の mask/tilt を信号で動的修正 | signal ∈ {0,1,2,3} → lev_mod 倍率 |
| **Standalone** | F11+ 系として独立戦略化 | signal → TQQQ/TMF/Gold 配分を直接決定 |

選定アクション (Phase C 末):
- Overlay 勝者 → `F10_OL_<signalID>` として既存ベスト派生登録
- Standalone 勝者 → 新クラス F11/F12+ として独立登録
- 両モード PASS → 上位 1 系を採択、もう一方は併記

### 6.2 WFA プロトコル

最大 **102 戦略バリアント** (90 = 15信号×2モード×3資産 + 12 組合せ) を以下で評価:

| プロトコル | 目的 | 既存対応 | 判定 |
|---|---|---|---|
| **G1** 静的 IS/OOS | 基本パフォーマンス | ◎ | OOS Sharpe > baseline |
| **G3** rolling 50窓 WFA | 時系列安定性 | ◎ | CI95_lo > 0, WFE > 1.0 |
| **G7** Bootstrap OOS (10,000) | OOS 信頼区間 | ◎ | P(CAGR>baseline) > 90% |
| **G8** 年次寄与分解 | 単年集中度 | ◎ | 最大寄与年 < 35% |
| **G9** Permutation test | 偶然性除外 | ◎ | p < 5% (G11 で補正) |
| **G10** parameter robustness sweep (近傍5点) | 脆弱性 | ◎ g20a | 5点中4点で同符号 PASS |
| **G11** Hansen SPA / Romano-Wolf step-down ★新規 | 多重比較下の真の優位性 | **新規** | best 候補が偶然でないこと保証 |

**G11 新規追加**: 90+12 バリアントから "best" を選ぶ操作の p-hacking 補正。`arch` パッケージ `SPA` クラスで実装可能、追加工数 +0.5日。

### 6.3 採用基準 (主目的: CAGR + IS-OOS頑健性)

Pareto判定: 既存 NEW CANDIDATE をベースライン、以下のうち **2 軸以上で改善 & 1 軸も閾値超過悪化なし**:

| 軸 | 改善判定 | 悪化許容上限 |
|---|---|---|
| CAGR_OOS | +2pp 以上 | -1pp まで |
| Sharpe_OOS | +0.05 以上 | -0.05 まで |
| IS-OOS gap | \|gap\| 1pp 縮小 | +2pp 拡大まで |
| MaxDD | -5pp 改善 | +8pp 悪化まで |
| Trades/yr | (制約なし) | 200/yr 超過は不採用 |

**Hard requirement (全項目必須):**

- G3 PASS (CI95_lo > 0, WFE > 1.0)
- G7 Bootstrap P > 90%
- G9 Permutation p < 10%
- G11 SPA p < 10%

### 6.4 信号組合せ探索

Tier3 単独 PASS 上位 3-4 から **2 信号 AND/OR ロジック**:

| 組合せ | 期待効果 |
|---|---|
| Sentiment AND Credit | Risk-off 重畳検出 → MaxDD 削減 |
| Macro Nowcast AND Earnings Revision | Bull rally 確認 → CAGR 押し上げ |
| Vol regime AND Yield Curve | Tail risk 早期警戒 |

最大 12 追加バリアント、G11 SPA に同時投入。

### 6.5 クロス資産検証

NDX 信号 → TMF/GLD でも IC>0 か (spillover 確認)。効くものは DH-Z2 timing component 強化に活用。

### 6.6 計算量管理 (4 Level Funnel)

| Level | 戦略数 | 内容 | 期間 |
|---|---|---|---|
| 1: G1+G7 fast screen | 102 → ~40 | 並列 | 1日 |
| 2: G3 WFA 50窓 | 40 → ~15 | 並列 | 2-3日 |
| 3: G8/G9/G10 厳格 | 15 → ~8 | 順次 | 1-2日 |
| 4: G11 SPA + 組合せ最終 | 8 → 3-6 | 順次 | 半日 |

**Phase C 全体 約 1 週間 (フル並列)**

### 6.7 Phase C 成果物

1. `STRATEGY_PERFORMANCE_COMPARISON_<date>.md` — 既存 + 新採択戦略の9指標比較
2. `STRATEGY_REGISTRY.md` 更新 — Active/Shortlist 登録
3. `signals/wfa_results/<signal>_<mode>.md` — 各バリアント G1-G11 結果
4. `INTEGRATION_DEBATE_<date>.md` — 採択判断議事録
5. `CURRENT_BEST_STRATEGY.md` 更新可能性

### 6.8 失敗時フォールバック

| Phase C 結果 | アクション |
|---|---|
| 0 採用 | Phase B 閾値を IC>0.03, FDR<15% に緩和し再実行 |
| 0 採用 (緩めても) | 「現行 NEW CANDIDATE が局所最適に近い」と結論、Phase D は既存戦略のみ実装 |
| 6+ 採用 | 上位 3 を Active、残り Shortlist、相関再評価で冗長削減 |

---

## 7. 成功基準 (本プロジェクト全体)

### 7.1 定量

- Phase A: 30 信号の Tier1 採択完了 + データソース URL/cost 全数確定
- Phase B: scorecard 完了、12-15 信号が Tier3 へ通過、合成信号 1-2 追加
- Phase C: **最低 1 戦略以上が Pareto 改善 + Hard requirement 全 PASS で採択**

### 7.2 定性

- 全信号が Look-ahead bias なしで実装され、`publication_lag` がドキュメント化
- 採択戦略が「経済的に説明可能なメカニズム」を持つこと (purely data-mined の信号は採択せず)

---

## 8. 対象外 (Phase D/E)

以下は本仕様の対象外、別途仕様化:

- Phase D: GAS 日次取得パイプライン実装 (信号別取得スクリプト・耐障害性・キャッシュ・コスト管理)
- Phase E: 既存 Active 戦略への信号注入実装・本番切替・モニタリング

ただし Phase A の `data_lineage.md` で各信号の GAS 取得可能性 (auth要否・rate limit・cost tier) を記録しておくこと → Phase D 仕様化が半自動化される。

---

## 9. 保留事項 (Phase B/C 結果次第で確定)

| # | 項目 | 確定タイミング |
|---|---|---|
| 1 | overlay vs standalone どちらを主にするか | Phase C 後 |
| 2 | 採択戦略が現行 NEW CANDIDATE を replace するか追加するか | Phase C 後 |
| 3 | Composite signal を Phase C で組合せ探索の input に含めるか | Phase B 後 |
| 4 | G11 SPA 実装の優先度 (本仕様では新規追加・推奨) | Phase C 開始時に最終確認 |
| 5 | データソースで scraping が必要な信号 (Citi Surprise 等) の安定取得方法 | Phase B 着手時 |

---

## 10. 参照

- 現行戦略統合 MD: `STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md`
- 戦略台帳: `STRATEGY_REGISTRY.md`
- 9指標標準: `docs/rules/08_evaluation-metrics.md`
- 統合レポート規格: `docs/rules/09_integrated-report-standard.md`
- 戦略検証規格: `docs/rules/06_strategy-verification.md`
- ファイル命名・日付規格: `docs/rules/07_doc-naming-and-dates.md`
- WFA プロトコル既存: g20 シリーズ (v6.3 系)、g22 シリーズ (v4 DH-Z2)
- 関連グローバル: `~/.claude/CLAUDE.md` §13 (NASDAQ ルール所在)

---

**次工程**: 本仕様をユーザーレビュー → writing-plans skill で Phase A/B/C 各々のタスク粒度実行計画に展開。
