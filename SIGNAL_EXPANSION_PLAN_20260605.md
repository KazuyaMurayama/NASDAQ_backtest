# 信号拡張・既存戦略統合 検証計画 v2

作成日: 2026-06-05
最終更新日: 2026-06-05

> Phase A-D で評価した **6信号** (#6/#21/#23/#26/#28/#41) は既存ベスト戦略 (S1/S2/S3) を改善できなかった (Phase D 厳格 audit で Bootstrap P=0.39 で REJECT)。本計画では **未テスト 71+ 信号** を、Phase D で確立した **native integration プロトコル** で再評価する。

---

## 0. 教訓 (Phase A-D から)

### 0.1 Post-hoc 評価過大評価問題

Tier 1-3 (Sessions S2/S3) の評価方法は post-hoc multiplication:
```
candidate_nav = baseline_nav.pct_change() × signal_multiplier → cumprod
```

Phase D で同じ候補 (S3×BAA-10Y×M2 procyclical) を **native integration** で再評価:
```
build_W1_baa10y(assets, baa_signal_q):
  lev_raw = original_dh_w1_logic(assets)
  lev_raw_modulated = lev_raw × signal_multiplier  ← BEFORE NAV computation
  nav = build_dh_nav_with_cost(lev_raw_modulated, ...)
```

結果:
- **Post-hoc**: Sharpe diff +0.067, CAGR diff +0.54pp (改善あり)
- **Native**: Sharpe diff +0.118, CAGR diff **-0.44pp** (OOS でベース負け)
- **Bootstrap P(cand>base) = 0.39** (10,000 resamples)

→ **本計画では post-hoc 評価を一切採用しない**。最初から native integration。

### 0.2 IC が高くても戦略改善とは限らない

Phase B で BH-FDR<0.10 を通過した信号 (例: BAA-10Y NDX 60d IC=+0.20) でも、native integration で戦略改善には繋がらない場合あり。
→ **本計画では IC スクリーニングの位置付けを下げ、戦略との直接的 daily-return 相関 (Strategy IC) を主指標とする**

### 0.3 既存ベスト戦略の堅牢性

S1 (NEW CANDIDATE = vz=0.65+l7+F10ε), S2 (D5 = vz=0.65/lmax=5.5), S3 (DH-W1 Asymm+Hyst) は既に多種信号情報を内包。改善余地は極小。
→ 本計画は **限界効用低くても全候補を試す** ユーザー方針

---

## 1. スコープ

### 対象戦略 (Phase D と同じ 3 系統)
| 略称 | 戦略 | NAV cache |
|---|---|---|
| S1 | vz=0.65+lmax=7+F10ε=0.015 (NEW CANDIDATE) | `f10_nav_cache.pkl` |
| S2 | D5: vz=0.65/lmax=5.5 | `vz065lmax5_nav_cache.pkl` |
| S3 | DH-W1 (Asymm+Hyst, DH base) | `dh_w1_nav_cache.pkl` (Session S2 で生成済) |

### 対象信号 (未テスト 71+ candidates)
詳細は §2 で枠別列挙。

### 対象 method (M1-M5 既存 + M6-M8 新規)
詳細は §3。

---

## 2. 未テスト信号インベントリ (枠別)

### 2.1 Tier1 ◎ 未テスト 25 signals (Phase B でデータ未整備により評価不可)

| ID | 信号 | カテゴリ | データソース | データ取得方法 |
|---|---|---|---|---|
| 1 | NDX 200DMA超え銘柄% | Breadth | Yahoo / Polygon | A6 yahoo loader |
| 2 | McClellan Oscillator NDX | Breadth | StockCharts相当 (計算) | 新規 |
| 3 | NDX New Hi-Lo 52W | Breadth | Polygon Starter | 新規 |
| 7 | VIX9D over VIX 比率 | Vol | Yahoo (^VIX9D, ^VIX) | A7 yahoo loader |
| 8 | VIX Term Structure (VIX1/2/3) | Vol | CBOE / Polygon | A9 cboe loader |
| 9 | VVIX | Vol | Yahoo (^VVIX) | A7 yahoo loader |
| 10 | MOVE Index | Vol | Yahoo (^MOVE) | A7 yahoo loader |
| 12 | CBOE PutCall (Equity) | Sentiment | CBOE 公開 | A9 cboe loader |
| 13 | AAII Bull-Bear spread | Sentiment | AAII 公開 (週次手動) | A10 manual loader |
| 15 | CFTC CoT NQ Non-Comm Net | Sentiment | CFTC | A8 cftc loader |
| 16 | CFTC CoT GC Net | Sentiment | CFTC | A8 cftc loader |
| 17 | CFTC CoT ZB/ZN Net | Sentiment | CFTC | A8 cftc loader |
| 27 | 3M10Y spread | YieldCurve | FRED (T10Y3M) | A6 fred loader |
| 29 | 5Y5Y BEI | YieldCurve | FRED (T5YIFR) | A6 fred loader |
| 30 | CME FedWatch 25bp cut prob 3M | YieldCurve | CME 公開 | 新規 |
| 32 | Atlanta Fed GDPNow | Macro | Atlanta Fed | A10 manual loader |
| 34 | Citi Economic Surprise USMI | Macro | Citi scrape | A10 manual loader |
| 36 | Chicago Fed NFCI | Macro | FRED (NFCI) | A6 fred loader |
| 37 | NDX Forward EPS Rev 4wk | Earnings | EODHD / Finnhub | A10 manual loader |
| 38 | Equity Risk Premium | Earnings | computed (Fwd EPS Yld - 10Y real) | A10 manual loader |
| 40 | Mag-7 EPS Revision composite | Earnings | Finnhub | A10 manual loader |
| 42 | Copper/Gold ratio | Cross-Asset | Yahoo (HG=F / GC=F) | A7 yahoo loader |
| 46 | FOMC blackout window | Calendar | Fed schedule (manual) | A10 manual loader |
| 49 | Google Trends "recession" 90d Z | NLP/Search | pytrends | 新規 |
| 50 | Fed minutes hawkish-dovish NLP | NLP | FOMC 文書 + LLM API | A10 manual loader |

### 2.2 Tier1 ○ 未テスト 16 secondary signals

| ID | 信号 | データソース |
|---|---|---|
| 4 | A/D Line price divergence | Yahoo (計算) |
| 11 | GVZ (Gold Vol) | Yahoo (^GVZ) |
| 14 | NAAIM Exposure Index | NAAIM 公開 (週次) |
| 18 | QQQ creation/redemption | ICI / Finnhub (low_paid) |
| 19 | GLD/TLT flows | 同上 |
| 22 | ICE BofA IG OAS | FRED (BAMLC0A0CM) |
| 24 | SOFR-IORB spread | NY Fed |
| 31 | 10Y-2Y real yield diff | computed (DFII10-DFII2) |
| 33 | NY Fed Nowcast | NY Fed 公開 (週次) |
| 35 | Cleveland Fed Inflation Nowcast | Cleveland Fed |
| 39 | NDX Forward PE z-score | EODHD / Finnhub |
| 43 | Silver/Gold ratio | Yahoo (SI=F/GC=F) |
| 44 | Oil (WTI) 5d change | Yahoo (CL=F) |
| 47 | Mag-7 earnings season | Earnings Calendar |
| 51 | Headline News risk-off NLP | News API + LLM |
| 52 | Google Trends "TQQQ"/"QQQ" | pytrends |

### 2.3 新規信号 (オリジナル 52 候補外, 25 signals)

| カテゴリ | 信号 | データソース | 想定有効性 |
|---|---|---|---|
| **Options-Implied** | NEW-1: SPX 25-delta put skew | CBOE / Polygon | tail risk 早期警戒 |
| Options | NEW-2: GEX (S&P 500 Gamma Exposure) | SqueezeMetrics scrape | デリバ流動性 |
| Options | NEW-3: DIX (Dark Index) | SqueezeMetrics scrape | ダークプール買い |
| Options | NEW-4: 0DTE option flow ratio | CBOE 集計 | 短期投機度 |
| Options | NEW-5: VVIX/VIX 比 | Yahoo 派生 | vol risk premium |
| Options | NEW-6: VIX9D/VIX | Yahoo 派生 | 短期 stress |
| **Cross-Asset** | NEW-7: TED-equivalent (DTB3-SOFR) | FRED 派生 | 銀行 stress |
| Cross-Asset | NEW-8: USD/JPY 3M XCB (cross-currency basis) | Bloomberg/Polygon | ドル funding |
| Cross-Asset | NEW-9: HYG/SHY 比 (HY ETF vs short Treasury) | Yahoo | risk-on/off |
| Cross-Asset | NEW-10: Bond-Stock correlation rolling 60d | computed | レジーム検出 |
| Cross-Asset | NEW-11: AAA-BAA spread (credit quality) | FRED 派生 | クレジット |
| Cross-Asset | NEW-12: Single-A OAS (IG-HY中間) | FRED (BAMLC1A0C13Y) | クレジット詳細 |
| **政策・流動性** | NEW-13: Fed Balance Sheet 週次Δ | FRED (WALCL) | 流動性 |
| 政策 | NEW-14: Treasury General Account balance | FRED (WTREGEN) | 流動性 |
| 政策 | NEW-15: Reverse Repo balance | FRED (RRPONTSYD) | 流動性過剰 |
| 政策 | NEW-16: Net Treasury issuance schedule | Treasury 公開 | 流動性 |
| 政策 | NEW-17: Fed Funds Rate change Δ | FRED (DFF) 派生 | 金利政策 |
| **Yield Curve 詳細** | NEW-18: 30Y-10Y term premium | repo 既存 (DGS30-DGS10) | 長期 |
| YC | NEW-19: TIPS 5Y-10Y slope | FRED 派生 | 実質金利構造 |
| YC | NEW-20: Yield curve curvature (5Y - (2Y+10Y)/2) | FRED 派生 | 中期凹凸 |
| **Macro Surprise** | NEW-21: Bloomberg ECO US Surprise | scrape | マクロ予測誤差 |
| Macro | NEW-22: ISM Manufacturing PMI Δ | FRED (NAPM) | 景気 |
| Macro | NEW-23: Initial Jobless Claims weekly Δ | FRED (ICSA) | 労働市場 |
| **Behavioral** | NEW-24: Wikipedia "bear market" page views | Wikimedia API | 群衆心理 |
| **Volatility** | NEW-25: RVOL vs IV gap (60d realized vol - VIX) | computed | vol risk premium |

### 2.4 Repo 既存 CSV から即座に派生可能 (10 signals)

| ID | 派生信号 | 元 CSV |
|---|---|---|
| REPO-1 | AAA-BAA spread (credit quality) | aaa_monthly.csv + baa_monthly.csv |
| REPO-2 | 30Y-10Y term premium | dgs30_daily.csv + dgs10_daily.csv |
| REPO-3 | Fed Funds Δ (1mo) | dff_daily.csv |
| REPO-4 | Fed Funds vs 10Y spread | dff_daily.csv + dgs10_daily.csv |
| REPO-5 | DGP (??) 解析 | dgp_daily.csv (要確認) |
| REPO-6 | DRN (??) 解析 | drn_daily.csv (要確認) |
| REPO-7 | CPI surprise vs trailing avg | cpiaucsl_monthly.csv |
| REPO-8 | ML predictions 符号 | ml_oos_predictions.csv |
| REPO-9 | ML features 主成分 | ml_features.csv (PCA 後の PC1/PC2) |
| REPO-10 | Macro features (中身次第) | macro_features.csv |

### 2.5 信号総数

**71 signals = 25 (Tier1 ◎未テスト) + 16 (Tier1 ○) + 25 (新規) + 10 (Repo派生) − 5 (重複 e.g. #21 と REPO-1 関連)**

実効的に **約 65 unique signals** を新規評価対象とする。

---

## 3. 注入方式 (M1-M8)

Phase A-D の M1-M5 (既存) + M6-M8 (新規):

| 方式 | 仕組み | 既存/新規 |
|---|---|---|
| M1 | Leverage Mask (binary) | 既存 |
| M2 | Continuous Leverage Tilt | 既存 (Phase D で実証) |
| M3 | Asset Rotation (S3 only) | 既存 |
| M4 | Vol Target Modifier | 既存 |
| M5 | Entry/Exit Filter | 既存 |
| **M6** | **Internal Threshold Mod** (vz_thr/mask_thr/lt_thr を動的修正) | **新規** |
| **M7** | **Hybrid State Override** (Asymm+Hyst state を強制遷移) | **新規** (S3 主) |
| **M8** | **Cost Multiplier** (vol high → 取引コスト乗数 ↑) | **新規** |

### M6 数式
```
base_strategy(vz_thr=0.65) → strategy_vz_thr = 0.65 + signal_q × 0.05  (signal=0→0.65, signal=3→0.80)
```
高信号で閾値厳格化 → 取引頻度減・確信度高エントリーのみ。

### M7 数式 (S3 DH-W1 専用)
```
asymm_hyst_state: HOLD if signal_q ≤ 1 (low risk) else FORCE_OUT
```
extreme signal で全 cash 化、通常 hysteresis ロジック override。

### M8 数式
```
trade_cost_multiplier = 1 + signal_q × 0.5  (signal=0→1.0, signal=3→2.5)
```
高 vol/stress 時は slippage 拡大想定 → defensive な現実評価。

---

## 4. Gating Pipeline (G1-G5)

Phase A-D の Tier 概念を G1-G5 に置換 (gating 厳格化)。

### G1: Data Acquisition

各候補信号について:
- データソース確認 (free / low_paid / mid_paid)
- 取得スクリプト整備 (既存 loader 拡張)
- 過去 10 年以上のデータカバレッジ確認
- `data/signals/expansion/raw/<signal_id>.parquet` に保存

**通過基準**: 過去 10 年以上のデータが取れる
**期待通過数**: 50-60 / 65

### G2: Direct Strategy IC Screening

各 (signal × strategy) ペアについて:
- Strategy daily returns vs signal Spearman IC (rolling 252d)
- |IC| ≥ 0.03 で通過 (Phase B は forward return IC を使ったが、これは戦略 daily return との直接相関)

**通過基準**: |IC| ≥ 0.03 で BH-FDR p<0.15 (G2 は 50×3=150 検定、緩めの FDR で網広く)
**期待通過数**: 20-25 signals × 平均 1.5 戦略

### G3: Native Tier 1 (M2 procyclical only)

G2 通過信号を **native integration の M2 procyclical** で各戦略に注入。Phase D で確立した `build_<strategy>_<signal>.py` パターンを汎用化。

9+1 metric 評価で **n_imp ≥ 3, n_deg ≤ 1** 通過

**期待通過数**: 5-10 patterns

### G4: Native Tier 2 (M1/M4/M5/M6/M8 × 2 directions)

G3 通過信号を残りの 5 方式で評価 (M3 は S3 限定、M7 は S3 主)。

**期待通過数**: 10-20 patterns

### G5: Combos + Phase D Audit

G3+G4 通過上位 5 signals の AND/OR 合成 + 全通過候補に対し Phase D 厳格 audit (WFA 50窓 + Block Bootstrap 10,000)。

**最終採用基準**:
- Bootstrap P(cand>base) > 0.90
- WFA WFE ≥ 1.0
- WFA CI95_lo > 0
- 9+1 Pareto: n_imp ≥ 3, n_deg ≤ 1

**期待採用**: 0-3 final candidates

---

## 5. Native Integration Protocol

### 5.1 各戦略の改造ポイント

| 戦略 | 改造ファイル | 信号注入ポイント |
|---|---|---|
| S1 (F10) | 新規 `src/integration/build_f10_with_signal.py` | F10 の lev_raw 計算後、signal multiplier 適用 |
| S2 (D5) | 新規 `src/integration/build_d5_with_signal.py` | D5 の lev_raw 計算後、同様 |
| S3 (DH-W1) | 既存 `src/integration/build_w1_baa.py` を汎用化 → `build_w1_with_signal.py` | lev_raw 段階で multiplier |

### 5.2 汎用化 API

```python
def build_strategy_with_signal(
    strategy: str,                # 'S1' / 'S2' / 'S3'
    signal_q: pd.Series,          # quantized 0-3
    method: str = 'M2',           # 'M1'-'M8'
    direction: str = 'procyclical', # method-specific
    mapping: dict = None,         # override default
) -> pd.Series:                   # NAV
    ...
```

### 5.3 Phase D 確認済テンプレート

`src/integration/build_w1_baa.py` を base に S1/S2 用を作る。コア構造:
```
1. Load shared_assets (price data)
2. Quantize signal + apply publication_lag
3. Compute base strategy's internal state (lev_raw, mask, etc.)
4. Apply method-specific modulation:
   - M1: lev_raw × binary(signal)
   - M2: lev_raw × continuous_map(signal)
   - M6: threshold = base_thr + signal × delta, recompute lev_raw
   - M7: state override per signal
5. Compute NAV with daily costs
6. Return NAV series
```

---

## 6. 9+1 評価基準 (Phase D 確立済)

(Session S3 で Phase D で実装済 — そのまま使用)

| metric | 改善判定 | 重大悪化 (Strong/Standard PASS 阻害) |
|---|---|---|
| CAGR_OOS | +0.5pp | -2.0pp |
| IS-OOS gap | -0.5pp 縮小 | +3.0pp 拡大 |
| Sharpe_OOS | +0.03 | -0.05 |
| MaxDD | -2pp 改善 | +10pp 悪化 |
| Worst10Y | +0.5pp | -2pp |
| P10_5Y | +0.5pp | -2pp |
| Trades/yr | (制約なし) | 200/yr 超 |
| WFE | +0.05 | <0.95 |
| CI95_lo | +0.05 | -0.05 |

**Strong PASS**: ≥5 改善 + 重大悪化なし
**Standard PASS**: ≥3 改善 + 重大悪化なし
**Marginal**: ≥1 改善
**FAIL**: 改善なし

---

## 7. Phase D Hard Requirements

Strong/Standard PASS 候補は **必ず** Phase D 厳格 audit を通る:

| Gate | 基準 |
|---|---|
| WFA 50窓 WFE | ≥1.0 |
| WFA CI95_lo | >0 |
| Block Bootstrap 10,000 P(cand>base) | >0.90 |
| 9+1 Pareto | Strong/Standard PASS 維持 |

---

## 8. Session Schedule (10-15 sessions 目安)

| Session | 内容 | 目安パターン数 |
|---|---|---|
| **1** (本セッション) | spec 確定 + データ取得計画 + REPO 派生信号即時化 | — |
| **2** | G1 データ取得: Yahoo loaders 拡張 (#7,#8,#9,#10,#11,#42,#43,#44 + NEW-5/6/9) | ~12 signals |
| **3** | G1 データ取得: FRED loader 拡張 (#22,#24,#27,#29,#36,#31 + NEW-7/11/12/13-15/17/18-20/22/23) | ~16 signals |
| **4** | G1 データ取得: CFTC/CBOE/manual (#12,#13,#14,#15,#16,#17,#32,#33,#34,#35,#37-40,#46,#47,#49-52) | ~18 signals |
| **5** | G2 Direct IC Screening: 全 signals × 3 戦略 daily return | 150 tests |
| **6** | G3 Native Tier 1: M2 procyclical native | ~25 patterns |
| **7** | G4 Native Tier 2: M1/M4/M5/M6/M8 × 2 方向 | ~60 patterns |
| **8** | G4 続行 (M3 S3 only, M7 S3 主) | ~20 patterns |
| **9** | G5 Combos: AND/OR 上位5 signals | ~30 patterns |
| **10-12** | Phase D 厳格 audit: 通過候補それぞれに WFA + Bootstrap | 5-10 candidates × 1日 |
| **13** | 最終採用判定 + STRATEGY_REGISTRY 更新 + CURRENT_BEST_STRATEGY 更新可能性 | — |

---

## 9. 成功基準

### 定量
- G1: 50+ signals がデータ取得可能
- G2: 20+ signals が IC ≥ 0.03 通過
- G3+G4: ≥5 patterns が Standard PASS 達成
- **Phase D**: ≥1 candidate が Bootstrap P > 0.90 (Phase A-D で 0 だった)

### 定性
- 採用 0 でも、**「どの信号も既存戦略を改善しない」** ことが網羅証明される
- データソース・loader が今後の signal exploration に再利用可能
- 教訓 (post-hoc vs native, IC vs strategy improvement) が定着

---

## 10. リスク

| リスク | 対処 |
|---|---|
| データ取得失敗 (rate limit, scraping block) | キャッシュ + リトライ + 代替ソース |
| 71 signals 全 G1 通過すると G2-G5 計算量爆発 | Gating 厳格適用 |
| 全 candidate が Phase D で REJECT | "信号統合では限界" の証明として記録、別アプローチ (新戦略構造) を Phase X として別途 |
| Native integration の S1/S2 改造に時間がかかる | S1/S2 は M2 procyclical 1 方式のみ深掘り、S3 主体で多 method |
| 11+ session の長期計画でセッション間文脈ロス | 各 session 末で intermediate summary + commit |

---

## 11. 関連ドキュメント

- 上位設計: `SIGNAL_DISCOVERY_PLAN_20260603.md`, `IMPLEMENTATION_PLAN_SIGNAL_DISCOVERY_20260603.md`
- 統合計画 v1 (本計画で置換): `SIGNAL_INTEGRATION_PLAN_20260604.md`
- Phase B 結果: `data/signals/screening_report_20260604.md`
- Tier 1-3 最終: `data/signals/integration/integration_final_report_20260605.md`
- Phase D 結果 (REJECT): `data/signals/integration/phase_d_audit_report_20260605.md`
- 9指標規格: `docs/rules/08_evaluation-metrics.md`

---

## 12. 本セッション (Session 1) 着手内容

1. ✅ 本 spec を `SIGNAL_EXPANSION_PLAN_20260605.md` として commit/push
2. **DGP/DRN CSV の中身確認** — Repo 派生信号 (REPO-5/6) の特定
3. **macro_features.csv / ml_features.csv の中身確認** — REPO-9/10 の potential 評価
4. **REPO 派生信号 (REPO-1〜4, REPO-7) を即座にデータ化** — Session 2 で即評価可能に
5. **untested_signal_inventory_20260605.csv** 作成 — 71 signals 全てを 1 行ずつ列挙、status (data_available/needs_loader/needs_manual) 付き

これにより Session 2 以降の G1 データ取得が体系化される。
