# EVALUATION_STANDARD.md — NASDAQ Backtest 評価基準の単一の真実

> **このファイルは「戦略を評価するときの前提・指標・期間・コスト・レポート様式」を一意に固定するための正典です。**
> **新しい戦略検証・改良・比較を行う前に、必ず §0 を読み、各セクションの定義に従ってください。**
> **本書と矛盾する実装・レポートは「非標準」として §1.3 / §6 の参考値ルールが適用されます。**

- バージョン: **v2.0**
- 発行日: 2026-05-22（最終改訂 2026-06-19: **§3.12 統一指標を v2.0（10指標）へ刷新** — CAGR_IS追加・IS-OOS gap削除・Sharpeフル期間化・最悪単日(テール)追加・Worst5Y追加・旧Overfit+CI95を「頑強性/過学習」1列に統合・◎/★をフルSharpe実測で再較正(◎+0.934/★+1.100)。前回 v1.9: §1.5 >3xレバ証拠金は担保で継続的CAGRドラッグを生まない）
- 管理者: 男座員也（Kazuya Oza）
- 一次関連ファイル: [`CURRENT_BEST_STRATEGY.md`](CURRENT_BEST_STRATEGY.md), [`src/product_costs.py`](src/product_costs.py), [`tasks.md`](tasks.md), [`FILE_INDEX.md`](FILE_INDEX.md)

---

## §0 標準前提サマリ（ワンブロック・新検証着手前に必読）

> **このブロックを読まずに評価を始めないこと。** 本書の各セクション（§1〜§5）はこのサマリの根拠詳細です。

| 項目 | 標準値（v1.1） | 詳細 |
|---|---|---|
| コストシナリオ | **Scenario D**（TER + sofr_multiplier×SOFR + swap_spread） | §1 |
| SOFR proxy | DTB3（FRED 3M T-bill）、52年平均 4.37%/yr | §1 |
| データソース | `data/NASDAQ_extended_to_2026.csv`（13,169 bars） | §2 |
| 期間 IS | 1974-01-02 〜 2021-05-07（47.3年・約11,916 bars） | §2 |
| 期間 OOS | 2021-05-08 〜 2026-03-26（4.9年・約1,253 bars） | §2 |
| 期間 FULL | 1974-01-02 〜 2026-03-26（52.26年・13,169 bars） | §2 |
| DELAY | **2営業日**（シグナル発生→建玉、look-ahead bias 対策） | §2 |
| 年換算 | **252営業日**（暦日365換算は不可） | §3 |
| Sharpe Rf | **0**（高金利環境で過大評価される旨を注記必須） | §3 |
| Worst10Y | **カレンダー年ベース**（★印）。日次ローリングは旧定義・廃止 | §3 |
| Worst5Y | 日次ローリング 252×5 窓 CAGR の最小値 | §3 |
| P10_5Y | 日次ローリング 5年CAGR 分布の第10パーセンタイル（▷ 印） | §3 |
| MaxDD | FULL 期間の `(nav/nav.cummax() - 1).min()` | §3 |
| Trades/yr | `simulate_rebalance_A` の閾値越えカウント | §3 |
| リバランス税ドラッグ | -2.8%〜-5.2% CAGR（27 trades/yr, 20.315%） | §1 |
| **>3xレバ証拠金前提** | **証拠金は担保（自己資金）で損益は建玉全額に発生→継続的CAGRドラッグを生まない（after-tax CAGRは到達可能）。唯一の継続コストは金利相当額（計上）。実コスト=テールリスク。−0.9pp/−3.42pp は却下** | **§1.5（v1.9）** |
| レポート必須ヘッダ | バージョン / コスト Scenario / 期間 / DELAY / コード参照 | §5 |
| 参考値判定 | §6 のフローチャートに従う | §6 |

**チェック項目（新検証 PR 前に必須）**:
- [ ] Scenario D を使用したか（コスト無しは「参考値」扱い）
- [ ] FULL/IS/OOS の3区間で指標を出したか
- [ ] Worst10Y★ をカレンダー年方式で出したか
- [ ] DELAY=2 を使用したか
- [ ] §5 のレポートテンプレに沿ったか
- [ ] CURRENT_BEST_STRATEGY.md と整合確認したか
- [ ] **10指標チェック（§3.12 v2.0）**: CAGR_IS・CAGR_OOS・Sharpe(Full)・MaxDD・最悪単日・Worst10Y★・Worst5Y・P10_5Y▷・Trades/yr・頑強性(過学習) を全て出したか（IS-OOS gap列は削除＝2CAGR列から導出）
- [ ] **禁止指標不使用**: Stable_Sharpe/WinRate_yr/WorstK5_mean_CAGR/IR_vs_BH を使っていないか
- [ ] **WFA実施時**: WFA_CI95_lo (§3.9) と WFA_WFE (§3.10) を追加したか
- [ ] **実効レバ>3x の戦略**: §1.5（v1.9）のコスト前提を適用したか — (a)証拠金は担保でCAGRドラッグを生まないと認識、(b)金利相当額（≈SOFR+スプレッド）をコストモデルに計上、(c)CAGRから「証拠金取り置きドラッグ」を別途引いていない（二重控除禁止）、(d)高レバのリスクをテールリスク（強制ロスカット）として提示、(e)リスク低減はレバ水準（de-lever）で評価
- [ ] **統計検定の妥当性（§4.4）**: MaxDD等の**経路依存極値を block=21 でブートストラップしていないか**（R-STAT-1＝無効・P≈0.5固定）。時機効果は**無傷の暴落窓＋サイン検定**で検定したか（R-STAT-2）。DD削減が時機かデレバかを**等平均退避の一律デレバ双子**と比較したか（R-STAT-3）。**DD削減はレバ水準ダイヤルが本質**。シミュ内の意思決定ルールが**当期リターンを参照するルックアヘッド**になっていないか（R-STAT-4・lagged規約で再計測）

---

## §1 コスト前提（Scenario D）

### §1.1 Scenario D の定義（現行標準）

**Scenario D** は `src/product_costs.py` を単一の真実として、以下の日次コスト式を全レバレッジ商品に適用する：

```
daily_cost = (TER / 252) + (sofr_multiplier × SOFR_daily + swap_spread) / 252
```

ここで `SOFR_daily = DTB3_annual / 252`（FRED `DTB3` を 3M T-bill proxy として使用）。

**製品別パラメータ**（`src/product_costs.py` Last updated: 2026-05-12 と同期）:

| 商品 | レバ | TER/yr | SOFR mult | Swap spread/yr | Div yield | NISA |
|---|---|---|---|---|---|---|
| **TQQQ** (ProShares UltraPro QQQ 3x) | 3.0 | 0.86% | 2.0 | 0.50% | 0.30% | × |
| **TMF** (Direxion 20+ Year Treasury Bull 3x) | 3.0 | 0.91% | 2.0 | 0.50% | 3.50% | × |
| **GOLD2X** (UGL ProShares Ultra Gold 2x) | 2.0 | 0.95% (実) / **0.49% (sim proxy)** | 1.0 | 0.50% | 0.00% | × |

- **TQQQ の sofr_multiplier=2.0** は OLS β_SOFR = -2.1306 の実証ベース。
- **GOLD2X のシム TER は WisdomTree 2036 (LSE, 0.49%) を proxy** として使用しており、**実投資先 UGL (0.95%) との差 -10.5 bps/yr** はレポートに明示すること（§5.4）。
  - ※ `CURRENT_BEST_STRATEGY.md` 旧版に "Gold 2x: TER 0.50%" の記載があるが、正値は 0.49%（`src/product_costs.py` 準拠）。次回 CURRENT_BEST_STRATEGY.md 更新時に 0.49% へ統一する。
- **日本居住者税**: `JP_CAPITAL_GAINS_TAX = 20.315%`（`src/product_costs.py`）。
- **リバランス税ドラッグ**: 27 trades/yr 前提で **CAGR -2.8% (best) 〜 -5.2% (worst)** を別途減算可能と明記。シミュレーションには織り込まないが、§5 のレポート末尾「税考慮後の現実推計」セクションで提示すること。

### §1.2 シナリオの差分表（参考扱いを含む）

| Scenario | TER | SOFR financing | Swap spread | 状態 | 用途 |
|---|---|---|---|---|---|
| **A** | × | × | × | **廃止** | 旧版・絶対に新規採用しない |
| **B** | ○ | × | × | **廃止** | 中間検証用のみ・新規結論禁止 |
| **C** | ○ | ○（1×SOFR 一律） | × | **廃止** | 中間検証用のみ |
| **D** | ○ | ○（**製品別 mult**） | ○ | **現行標準** | 全新規検証で必須 |

- B/C で出した結果は **「参考値」** とし、最終判定の根拠にしない（§6）。
- **既存レポートに Scenario が明記されていない場合は Scenario A 相当（コスト無し）と推定** し、自動で「参考値」扱いに分類する。

### §1.3 「参考値」扱いとなるコスト前提

以下のいずれかに該当する数値は**「参考値（非標準）」**として扱い、現行ベストの判定根拠にしてはならない：

1. Scenario A/B/C で計算された CAGR・Sharpe
2. SOFR proxy として DTB3 以外を使用した結果（明記必須）
3. `timing_signals_raw.csv` を使用した P-series (P01/P02/P05) — DH Dyn [A] シグナルに外部マクロ指標（HY スプレッド / CPI 前年比）のゲートを乗算した派生戦略群。コストモデルが標準 Scenario D と異なる（HY/CPI データ取得コストが別途発生）ため、統合比較表に掲載されていても CAGR_OOS の直接比較は参考値扱いとする
4. `product_costs.py` の値と異なる TER/swap_spread をハードコードした実装

### §1.4 確定日

- Scenario D の現行値確定日: **2026-05-12**
- 本セクションで参照する `src/product_costs.py` のバージョンは 2026-05-12 commit を基準とする

### §1.5 レバレッジ >3x（取引所CFD・くりっく株365）の証拠金コスト前提（v1.9 是正・2026-06-18）

> **適用対象**: 実効レバレッジが **3倍を超える**戦略（>3x超過分を取引所CFD=くりっく株365で建てる）。**今後の>3x検証で必須**。コスト前提の本体・URL根拠・却下案の詳細は [PRODUCT_COST_COMPARISON_2026-06-10.md §10](PRODUCT_COST_COMPARISON_2026-06-10.md)。本節は単独参照でも誤らないよう同義を要約。

**最有力の前提（採用）— 証拠金はCAGRドラッグを生まない**
証拠金は「ポジションを裏付ける自己資金（担保）」であって別建てで遊ばせる現金ではない。レバレッジ取引の損益は**証拠金額でなく建玉全額（想定元本）に発生**し自己資金に適用される（eToro: "profit/loss … applied to the total exposure not the initial deposit"）。∴ 運用資金を全額そのまま運用し証拠金はエクイティで満たす（過少投資しない）のが現実形態。「証拠金を運用から取り置く＝CAGRを引く」という直感は現物積立のもので、証拠金取引には当てはまらない。実際 scale1.35強mapで必要証拠金は AUM比 平均3.56%・最大22.3%＝自己資金で容易にカバー。**バックテストの税後CAGR（金利相当額控除済・min⓽+23.83%）は強制ロスカットを織り込んでも維持**: M6（m=8%最小証拠金）で**強制清算は1975-77に3回（各13-19%AUM喪失）発生したが50年CAGR影響+0.08pp・MaxDD不変・min⓽+23.83%不変、5大危機はOUTで清算0**。∴ 現実的手取り≈+23.83%（−ロール〜0.1-0.2pp）。残るは将来の高レバIN中暴落というテール（構造保証でない）。
- **>3xの唯一の継続コスト = 金利相当額（≈USD SOFR+スプレッド。買い建てが支払う資金調達コスト・岡三）。コストモデルに計上済**（borrow=SOFR+0.5% ＋ k365 EXCESS_EXTRA 0.25%）。円建てなのでFXヘッジコスト無し（公式）。残るは四半期ロール〜0.1-0.2pp（推定・DATA GAP）。
- **高レバの本当のコストはCAGRでなくテールリスク（強制ロスカット）**: 損益が建玉全額に発生するため急落でエクイティが維持証拠金を割れば強制決済（損失は拠出額を超えうる）。これはリスク次元で、**低減はレバ水準を下げる**ことで行う（現金を遊ばせるより資本効率的）。

**却下した前提（3案・なぜ誤りか）**:
| # | 却下前提 | 算出値 | 却下理由 |
|---|---|---:|---|
| 1 | 証拠金は遊休現金、機会損失 = 証拠金×SOFR | −0.9pp | 証拠金は建玉を裏付け損益を稼ぐ自己資金（遊休でない）。反実仮想SOFRも誤り。二重に誤り。 |
| 2 | 運用額のf%を常時現金取り置き、戦略リターンを取り損ねる | −1.68〜−3.42pp | 過少投資のモデルで証拠金の仕組み上は不要。任意の保守姿勢で強制コストでない。リスク低減なら **de-lever（scale↓）が同CAGR・同MaxDDで優位**。 |
| 3 | 資金制約でL頭打ち→戦略崩壊 | −7〜−10pp | k365は3〜8%証拠金でレバを建てられ自己資金が容易に支える＝L頭打ちにならない（M5 v1の資本会計バグで撤回）。 |

> 派生誤り: 取り置き現金を「JPY 0%」とする案も却下＝バックテストはUSD建て（SOFRはUSD金利）で、USD戦略リターンとJPY 0%金利の引き算は通貨ミスマッチ。

**出典（2026-06-18・詳細と取得状況は [PRODUCT_COST_COMPARISON §10.3](PRODUCT_COST_COMPARISON_2026-06-10.md)）**: 【本文確認済】くりっく株365公式 https://www.clickkabu365.jp/about_cfd/ ／ 金利相当額・岡三 https://www.okasan-online.co.jp/kabu365/guide/interest.html ／ レバレッジ=損益は建玉全額・証拠金は担保（eToro・証拠金取引一般原理） https://www.etoro.com/trading/leverage-margin/ ／ 【URL実在・本文HTTP403で未取得・定義は公知】SOFR=USD翌日物 https://fred.stlouisfed.org/series/SOFR

**>3x戦略 評価チェック（必須）**:
- [ ] 金利相当額（≈SOFR+スプレッド）をコストモデルに計上したか（>3xの継続コストはこれ。配当相当額・円建てFX無も確認）
- [ ] CAGRから「証拠金取り置きドラッグ」を別途引いていないか（＝二重控除・誤り。証拠金は担保で継続コストでない）
- [ ] 高レバのリスクを**テールリスク（強制ロスカット・清算距離・最悪損害）**として提示したか（CAGRドラッグでなく）
- [ ] リスク低減を「現金取り置き」でなく**レバ水準（de-lever）**で評価したか
- [ ] 建玉容量を実OI（or 感度）で対比したか

---

## §2 期間定義

### §2.1 標準3区間

| 区間 | 開始 | 終了 | 長さ | bars 数 | 用途 |
|---|---|---|---|---|---|
| **IS**（In-Sample） | 1974-01-02 | 2021-05-07 | 47.36 年 | 約 11,916 | パラメータ調整・モデル学習 |
| **OOS**（Out-of-Sample） | 2021-05-08 | 2026-03-26 | 4.9 年 | 約 1,253 | 汎化性能評価（最重要） |
| **FULL** | 1974-01-02 | 2026-03-26 | 52.26 年 | 13,169 | MaxDD / Worst10Y / Worst5Y の母集団 |

- データソース: `data/NASDAQ_extended_to_2026.csv`（13,169 bars）
- 営業日定義: NYSE 営業日。`DELAY = 2 営業日`（シグナル→建玉のラグ・look-ahead bias 対策）
- IS と OOS の境界は **2021-05-07 / 2021-05-08** 固定。動かさない

### §2.2 OOS 延長プロトコル

OOS 終端は実データの最新日にあわせて延長されるが、以下のルールを守る：

1. **IS の終端は動かさない**（過剰適合の検出が機能しなくなる）
2. OOS 延長時は **`CURRENT_BEST_STRATEGY.md` の「期間」行と本書 §0・§2.1 を同時更新**
3. 延長後の OOS 期間で Sharpe_OOS / CAGR_OOS / IS-OOS gap を再計算し、ベスト戦略の地位を再評価する
4. 延長前後の値を併記してレポートする（旧 OOS と新 OOS の比較）

### §2.3 WF（Walk-Forward）/ CV（Cross-Validation）の扱い

- WF/CV は **補助的なロバストネス確認** として扱い、IS/OOS の固定区間評価を置き換えない
- WF を実施する場合：
  - 学習窓幅 / テスト窓幅 / step 幅 を必ずレポートに明記
  - 各 fold の CAGR・Sharpe の分布（min / median / max / std）を提示
- CV（時系列CV）を実施する場合：
  - fold 間に embargo を入れたか明記
  - パラメータの安定性を fold 間 std で評価
- **WF/CV の最良 fold を「OOS の代わり」として使うことは禁止**（§6 で参考値判定）

---

## §3 指標定義（計算式）

### §3.1 CAGR（年率複利成長率）

```
CAGR = (NAV_end / NAV_start) ** (1 / years) - 1
years = n_bars / 252
```

- 年換算は **252 営業日**ベース。暦日 365 換算は採用しない（標準誤差約 ±0.05% は許容範囲）
- 区間別に **CAGR_IS / CAGR_OOS / CAGR_FULL** を出すこと（OOS が最重要）
- コード参照: `src/corrected_strategy_backtest.py`, `src/b1_s2_lt2.py`

### §3.2 Sharpe Ratio

```
daily_ret = nav.pct_change()
Sharpe = (daily_ret.mean() / daily_ret.std()) * sqrt(252)
```

- **リスクフリーレート Rf = 0**（簡便化）
- **注意事項（レポートに明記必須）**: 1970〜80年代の高金利期や直近の SOFR > 4% 環境では Sharpe を過大評価する可能性がある。Rf > 0 換算した参考 Sharpe を併記することを推奨
- 区間別に **Sharpe_IS / Sharpe_OOS / Sharpe_FULL** を出すこと

### §3.3 MaxDD（最大ドローダウン）

```
dd = nav / nav.cummax() - 1
MaxDD = dd.min()
```

- **FULL 期間で計算**するのが標準（IS/OOS 個別の MaxDD は補助情報）
- レバレッジ商品として -50% 〜 -70% の値域は想定内。-80% 以下は要警戒

### §3.4 Worst5Y CAGR

```
roll_5y_cagr = (nav / nav.shift(1260)) ** (1/5) - 1   # 252 × 5 = 1260
Worst5Y = roll_5y_cagr.min()
```

- **日次ローリング 5年窓**ベース。最悪窓の CAGR
- FULL 期間で計算

### §3.5 Worst10Y★ CAGR（カレンダー年方式）

> **★ 印は「カレンダー年ベース」を意味する。日次ローリングではない。**

```
# カレンダー年で 10年ローリング窓を走査（例: 1974-1983, 1975-1984, ..., 2016-2025 が最終完全窓）
for start_year in range(first_year, last_year - 8):   # last_year - 8 で最終完全10年窓を含む
    window_nav = nav.loc[f"{start_year}-01-01":f"{start_year+9}-12-31"]
    cagr_window = (window_nav.iloc[-1] / window_nav.iloc[0]) ** (1.0 / 10) - 1
Worst10Y_star = min(cagr_window for all start_year)
```

- **必ず ★ 印を付けて表記**し、旧定義（日次ローリング 252×10）と区別すること
- **2026-05-20 以前のレポートに登場する「Worst10Y」（★なし）は日次ローリング窓**で計算されており ★ 付きの値とは別物。**旧値は廃止扱い**
- カレンダー年が10年に満たない不完全窓は計算対象外
- FULL 期間で計算

### §3.6 P10_5Y▷ CAGR

```
roll_5y_cagr = (nav / nav.shift(1260)) ** (1/5) - 1
P10_5Y = roll_5y_cagr.quantile(0.10)
```

- 日次ローリング 5年 CAGR 分布の **第10パーセンタイル**
- 「下位 10% の局面でもこの水準は確保できる」というロバスト性指標
- **▷ 印**を付けて表記

### §3.7 Trades / Year

- `simulate_rebalance_A` におけるリバランス閾値越えカウント / 年
- 現行ベスト戦略の基底 DH Dyn シグナルで **約 27 回/年（月約 2.3 回）**
- 税ドラッグ（§1.1）算定の根拠値

### §3.8 IS-OOS gap

```
IS-OOS gap = CAGR_IS - CAGR_OOS    （単位: percentage point, pp）
```

- **過剰適合検出の主要指標**
- 目安: < +2 pp（優秀）、+2〜+5 pp（許容）、> +5 pp（過剰適合疑い）、> +10 pp（過剰適合確定）
- 現行ベストの IS-OOS gap は `CURRENT_BEST_STRATEGY.md` を参照（OOS 延長で変化するため本書に固定値を持たない）

### §3.9 WFA_CI95_lo（WFA補助: 統計的有意性下限）

```
# 非重複1年窓 N 本の CAGR 配列 {c_1, ..., c_N} に対して (短窓除外後)
mean_c = mean(cagrs)
se     = std(cagrs, ddof=1) / sqrt(N)
t_crit = t.ppf(0.975, df=N-1)
WFA_CI95_lo = mean_c - t_crit * se
```

- **t 分布 95% 信頼区間の下限**。正値 ⇒ 「真の年率期待リターン > 0」が統計的に支持される（α基準）
- 実装ファイル: `src/g1_wfa.py` (`compute_summary_stats`)
- **注意**: 窓数 N ≈ 49 では自由度 48 の t 分布を使用。t_p < 0.05 も同時確認すること

### §3.10 WFA_WFE（WFA補助: IS→OOS汎化効率）

```
WFA_WFE = mean(CAGR of windows where start_date >= OOS_START)
        / mean(CAGR of windows where start_date <  OOS_START)
```

- **Walk-Forward Efficiency**。IS窓の平均 CAGR に対する postIS窓の平均 CAGR の比率
- 許容範囲: **0.5 ≤ WFE ≤ 2.0**（β基準）
  - < 0.5: IS 過学習の強い兆候（OOS で IS の半分以下）
  - > 2.0: 異常な OOS 好調（レジーム変化・幸運の可能性）
  - postIS 窓数 < 3 の場合は N/A 扱い（判定スキップ）
- 実装ファイル: `src/g1_wfa.py` (`compute_summary_stats`)
- IS 境界: 2021-05-07 / OOS 開始: 2021-05-08（§2.1 と整合）

### §3.11 コード参照（指標の正典）

| 指標 | 実装ファイル |
|---|---|
| CAGR / Sharpe / MaxDD | `src/corrected_strategy_backtest.py` |
| Worst5Y / P10_5Y | `src/cfd_leverage_backtest.py` |
| Worst10Y★ | `src/compute_cfd_worst10y.py`（カレンダー年実装の正典）/ `src/b1_s2_lt2.py`（呼出側） |
| Trades/yr | `src/dynamic_leverage_strategies.py` (`simulate_rebalance_A`) |
| WFA_CI95_lo / WFA_WFE | `src/g1_wfa.py` (`compute_summary_stats`) |

新しい指標を追加する場合は、本書 §3 に式・コード参照・注意事項を追記してから採用する。

### §3.12 統一指標セット（10指標）とレポート標準（v2.0 確定 2026-06-19）

すべての sweep / grid / 戦略比較レポートは以下の **10指標**を標準セットとして使用する。期間基準（IS=インサンプル / OOS=アウトオブサンプル / Full=全期間）を列に明示する。

| # | 指標 | 期間 | 種別 | MD表示（3-4行折返） | 列順 |
|---|---|---|---|---|---|
| 1 | In-sample CAGR | IS | ⓽税後 | `CAGR<br>IS<br>⓽` | 1 |
| 2 | Out-of-sample CAGR | OOS | ⓽ | `CAGR<br>OOS<br>⓽` | 2 |
| 3 | Sharpe（フル期間） | Full | ⓒ税引前 | `Sharpe<br>Full<br>ⓒ` | 3 |
| 4 | MaxDD | Full | ⓒ | `Max<br>DD<br>ⓒ` | 4 |
| 5 | **最悪単日（テール）** | Full | ⓒ | `最悪<br>単日<br>ⓒ` | **5（新規・MaxDD直後）** |
| 6 | Worst10Y★ CAGR | Full | ⓽ | `Worst<br>10Y★<br>⓽` | 6 |
| 7 | **Worst5Y CAGR** | Full | ⓽ | `Worst<br>5Y<br>⓽` | **7（新規）** |
| 8 | P10_5Y▷ CAGR | Full | ⓽ | `P10<br>5Y▷<br>⓽` | 8 |
| 9 | Trades/yr | — | ⓞ | `Trade<br>/年<br>ⓞ` | 9 |
| 10 | **頑強性・過学習（統合）** | — | 判定+証拠 | `頑強性<br>過学習` | **10（旧 Overfit+CI95 を統合）** |

**v2.0 変更点（v1.4→v2.0）**:
1. **CAGR_IS を第1列に追加**（CAGR_OOS と並べ、min(IS,OOS) と gap が一目で読める）。これに伴い v1.1 の「CAGR は OOS 1列のみ」ルールを**廃止**。
2. **IS-OOS gap 列を削除**（CAGR_IS − CAGR_OOS で導出可・情報ロスなし。gap は #10 頑強性判定の内部材料として存続）。
3. **Sharpe を OOS → フル期間に変更**（52年で統計的に頑健・他リスク指標が全てFull基準で整合）。旧 OOS Sharpe 値は「OOS(旧)」と明記して残す。
4. **最悪単日（テール）を MaxDD 直後に追加**（高レバの強制ロスカット引き金＝単日急落を捕捉。セルに発生日を小書き）。
5. **Worst5Y を追加**（5年保有最悪。Worst10Y/P10_5Y と並ぶ）。
6. **旧 Overfit(WFE)+CI95_lo を「#10 頑強性・過学習」1列に統合**（判定＋証拠2-3行。WFE/CI95_lo/CPCV/t_p/Regime を集約）。
7. **◎/★ Sharpe マーカをフル期間Sharpeで再較正**: ◎=**+0.934**（E4現Active 実測Sharpe_FULL）/ ★=**+1.100**（B3aベスト 実測）。旧 OOS基準 0.770/0.885 は廃止。
8. **取引コスト評価は日次レベル必須**（年率近似は高頻度・低Δ戦略を過大評価）。

**取引コスト評価ルール（v1.4 必須）**:
- 全戦略の `CAGR_IS` / `CAGR_OOS` は **日次レベルで取引コストを反映**してから算出すること
- 実装パターン:
  - CFD: `daily_cost(t) = |Δ(wn × lev_mod × L_s2)| × spread_one_way`
  - ETF: `daily_cost(t) = (|Δw_TQQQ| + |Δw_TMF| + ...) × per_unit_cost`
- 参考実装: [src/g18_daily_trade_cost_wfa.py](src/g18_daily_trade_cost_wfa.py)
- 表内に yr_cost (年率取引コスト概算) を表示することは**禁止**（CAGR に反映済みで二重表示になる）

**状態凡例マーカ**:
- **⓽** = 税後（手取り）: CAGR_IS, CAGR_OOS, Worst10Y★, Worst5Y, P10_5Y▷
- **ⓒ** = コスト後（税引き前）: Sharpe(Full), MaxDD, 最悪単日
- **ⓞ** = 原値（コスト・税で不変）: Trades/yr
- **期間基準**: CAGR は IS/OOS、Sharpe・MaxDD・最悪単日・Worst10Y/5Y・P10 は **Full（全期間）**、#10 頑強性は WFA/CPCV 由来。

**◎/★ Sharpe マーカ（フル期間・v2.0 再較正・2026-06-19）**: ◎ = Sharpe_FULL > **+0.934**（E4 現Active 水準）/ ★ = > **+1.100**（B3a ベスト水準）。実測 Sharpe_FULL から確定（取得元: E4 +0.934 / B3a +1.102 / P09_C1 +1.128 / scale1.35強map +1.079、`src/audit/unified_metrics.py::compute_10metrics`）。**Worst10Y★ の ★ は別意味**（最悪10年CAGRの記号）。

**列ヘッダの3-4行折り返し**: MD テーブルでは `<br>` で各列名を3-4行に折り返し列幅を最小化する。全指標がスクロールなしに収まることを確認（§5.6 参照）。

**#10 頑強性・過学習（統合列・v2.0）**: 戦略の妥当性（過学習リスクの低さ・頑強性/汎化）を1セルで示す。評価者が知りたい「過学習リスクはどの程度か／頑強性はどの程度か」に直接答える列。**判定ラベル＋証拠2-3行**。
- **判定ルール（再現可能・実装 `src/_sweep_format.py::_robustness_cell()`）**:
  - `❌過学習疑い`: `WFA_WFE>2.0 or <0.5` または `WFA_CI95_lo<0` または `|IS_OOS_gap|>5pp` のいずれか該当。
  - `✅頑強`: `0.5≤WFE≤2.0` ∧ `CI95_lo>0` ∧ `|gap|≤3pp` ∧（`CPCV_p10>0` 算出時）∧（`t_p<0.05` 算出時）∧（`Regime_min>−10%` 算出時）。
  - `⚠条件付`: 上記の中間（gap 3-5pp 等）。**未算出ゲートがあれば末尾に `(部分)`**。
- **証拠行**（算出済のみ・NaNは省略）: `WFE{x} CI95lo{y}` / `CPCV{z} t_p{w}` / `Reg{v}`。例: `✅頑強<br>WFE0.99 CI95lo+23%<br>CPCV+16% t_p≈0<br>Reg−2.9%`。
- 判定材料: `WFA_WFE`, `WFA_CI95_lo`, `IS_OOS_gap_pp`（CAGR_IS−CAGR_OOSから）, 任意 `CPCV_p10`/`t_p`/`Regime_min`。WFA未計算なら `—`。

**CSV 列順序**（標準・v2.0）: パラメータ列の後に:
`CAGR_IS, CAGR_OOS, CAGR_FULL, Sharpe_FULL, Sharpe_OOS, MaxDD_FULL, Worst1D, Worst1D_date, Worst10Y_star, Worst5Y, P10_5Y, Trades_yr, WFA_CI95_lo, WFA_WFE, IS_OOS_gap_pp`（任意: `CPCV_p10, t_p, Regime_min`）。
（#10 頑強性ラベルは CSV に保存せず、MD表示時に `_robustness_cell()` で算出。`Sharpe_OOS` は旧基準併記用に CSV 保持。）

**WFA 計算ポリシー（sweep スクリプト）**:
1. 各セルの一次指標（#1〜7）はインラインで必ず計算する。
2. `WFA_CI95_lo` / `WFA_WFE` は CSV に `NaN`、MD に `—` として出力（計算未実施マーカ）。
3. 促進ゲート `Sharpe_FULL > +0.80 AND |IS_OOS_gap| < 5pp AND CAGR_OOS > 0` を満たすセルは `wfa_queue.csv` に追記する（v2.0: Sharpe基準を Full に変更）。
4. `src/g2_wfa_shortlist.py` が `wfa_queue.csv` を読み WFA を計算し対象 CSV を更新する。

**MD テーブル列構成（厳格ルール・v2.0）**:

1. **MD ヘッダは Strategy/Param + 10指標 = 11列**（v2.0: 旧 Overfit+CI95 を #10 頑強性に統合・CAGR_IS追加・gap削除で正味10指標）。
2. **CAGR は IS / OOS の2列を併記**（v2.0 で v1.1「OOS1列のみ」を廃止）。**IS-OOS gap 列は出力しない**（2列から導出可・#10頑強性の内部材料）。
3. **Sharpe は フル期間**（`Sharpe_FULL`）。旧 OOS Sharpe で算出した過去表は「OOS(旧)」と明記して残す。
4. **MD ヘッダは必ず `src/_sweep_format.py` の `MD_HEADER_1P/2P/STRAT/INTEGRATED` を import して使用**。手書きヘッダ禁止。
5. **§5.3 必須指標テーブル（IS/OOS/FULL 縦長3列）は単体戦略レポート専用**。横並び比較表は本 §3.12 の10指標標準に従う。
6. 戦略横並び比較は `MD_HEADER_STRAT` / `fmt_row_strat`。§1.3 参考値戦略は `sharpe_ref_mark='‡'` / `maxdd_ref_mark='‡'`。

**実装チェックリスト（PR前必須・v2.0）**:
- [ ] `row` dict に `CAGR_IS, CAGR_OOS, Sharpe_FULL, MaxDD_FULL, Worst1D(+Worst1D_date), Worst10Y_star, Worst5Y, P10_5Y, Trades_yr, WFA_CI95_lo, WFA_WFE` が含まれる（#10頑強性は自動算出・任意 `CPCV_p10/t_p/Regime_min`）
- [ ] MD テーブルが `src/_sweep_format.py` の `MD_HEADER_*` を使用（11列・手書き禁止）
- [ ] **CAGR は IS/OOS の2列**・**IS-OOS gap 列は出力しない**・**Sharpe は Full**
- [ ] 最悪単日セルに発生日を小書き／#10 頑強性は判定＋証拠（未算出ゲートは末尾 `(部分)`）
- [ ] ◎/★ は フル期間Sharpe閾値（◎ +0.934 / ★ +1.100）で付与
- [ ] §1.3 参考値戦略は Sharpe/MaxDD に `‡`
- [ ] WFA 未計算戦略は #10 頑強性を `—`（or 部分表示）／促進ゲート通過セルを `wfa_queue.csv` に追記

### §3.13 標準化された保守的 CAGR 指標 — min(IS, OOS) CAGR (v4.5 確定 2026-06-05 / v4.9 改訂 2026-06-08)

戦略評価において **min(IS, OOS) CAGR を保守的期待リターン指標として標準化**する。**OOS 単独評価は将来期待値の推定に使用しないこと** ─ regime fit / selection bias / sample size 非対称によるバイアスを防ぐため。

#### 定義

**min(IS, OOS) CAGR** = `min(cum_CAGR_IS, cum_CAGR_OOS)`
- `cum_CAGR_IS` = §0' 「累積 CAGR ⓽ OOS/IS」列の IS 値 (1977-2020 暦年複利)
- `cum_CAGR_OOS` = 同列の OOS 値 (2021-2026 暦年複利)

#### 論拠
- **サンプルサイズ非対称**: IS=44 年 vs OOS=6 年。統計的信頼性は IS が圧倒的に高い
- **戦略選択バイアス補正**: 多変種を比較して「OOS 最良」を選ぶと selection bias 発生。min ルールは penalize
- **regime drift リスク**: OOS 期間 (例: 2021-2026 の COVID/AI rally + 2022 bear) の固有 macro が将来も続く保証なし

#### 用途
- §0' 累積 CAGR 列の表示: **OOS / IS / min を 3 段表示**
- 戦略間比較の主要指標として使用 (OOS 単独評価より優先)
- 標準 10 指標 (§3.12 v2.0) と併記して総合評価

#### WFE 補助判定 (regime luck 警告)

| WFE 値 | 判定 |
|---|---|
| ≤ 1.2 | ✅ OK (構造的優位または fair generalization) |
| 1.2 < WFE ≤ 1.5 | ⚠ 注意 (OOS やや有利、追加検証推奨) |
| **> 1.5** | ❌ **regime luck 強疑い** (採用判断に min ルール厳格適用推奨) |

#### v4.9 改訂事項 (2026-06-08)

v4.5 当初は **「3 軸 (min + Worst10Y + P10_5Y) すべて baseline 以上」を Active 候補昇格の必須条件**としていたが、過度に restrictive と判断され**この強制条件は削除**:
- ✅ 残存: **min(IS, OOS) CAGR の標準化** (保守的期待リターン指標として使用)
- ✅ 残存: WFE 補助判定 (regime luck 警告)
- ❌ 削除: 「3 軸すべて baseline 以上」必須条件 (Worst10Y / P10_5Y は §3.12 の 10 指標として参照するが強制ではない)

**過去の判定 (AH/AT/HL 棄却等) は当時のルールに基づくため変更しない** が、今後の評価は **min(IS, OOS) + 10 指標の総合判断** で進める。

#### 環境別 Active 候補 (v4.5)

| 環境 | 戦略 | min CAGR | Worst10Y | P10_5Y | WFE |
|---|---|---:|---:|---:|---:|
| **CFD 利用可** (v4.7 確定) | vz=0.65+l5+F10ε | +18.93% | **+12.67%** | **+8.75%** | 1.389 |
| ↳ 副候補 (攻め型) | vz=0.65+l7+F10ε (旧 REF) | **+20.23%** | +9.96% | +4.05% | 1.369 |
| **ETF only** | DH-W1 (Asymm+Hyst) | +13.66% | +9.84% | +5.94% | 0.997 |
| **投信環境** (2026-06-07) | DH_W1_CashSleeve_P7_GOLD75BOND25 ⭐ | OOS +14.90% (中庸推奨) | +9.92% | +8.05% | 1.043 |
| ↳ 投信攻め型 | DH_W1_CashSleeve_P2_GOLD100 | OOS +16.44% | +9.43% | +8.06% | 1.229 |
| ↳ 投信守り型 | DH_W1_CashSleeve_P5_GOLD50BOND50 | OOS +13.28% | +10.08% | +8.09% | 0.875 |

> **投信環境の位置付け** (2026-06-07): DH-W1 (ETF only) の OUT 期 (キャッシュ 46.9%/6,171日) を 1 倍投信 (Gold/Bond) で運用置換する派生系。NISA 等 ETF only 環境で更に最適化を狙う。全 4 戦略 WFA 50窓 α∩β PASS、ただし t_p/bootstrap 未実施で正式 §1 Active 昇格は保留。一次根拠: [analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md)

---

## §4 頑健性チェック

### §4.1 必須項目（新戦略 PR には必ず含める）

1. **Scenario D で IS / OOS / FULL の全指標を出す**（§3 全項目）
2. **IS-OOS gap の評価**（§3.8 の目安に照らして判定）
3. **MaxDD と Worst10Y★ の同時提示**（リスクと最悪局面の両面）
4. **A6 / A1 サニティチェック相当**（直近ベスト戦略構成と CAGR_OOS 差 ≤ 0.5 pp を確認）
   - A6 = `l_max` パラメータスイープ（詳細: [`A6_LMAX_SWEEP_2026-05-21.md`](A6_LMAX_SWEEP_2026-05-21.md)）
   - A1 = `n_vol` パラメータスイープ（詳細: [`A1_NVOL_SWEEP_2026-05-21.md`](A1_NVOL_SWEEP_2026-05-21.md)）
5. **`CURRENT_BEST_STRATEGY.md` との指標突き合わせ**（現行ベストを上回らない場合はその旨を明記）

### §4.2 推奨項目（強い結論を出す場合に追加）

1. **WF（Walk-Forward）** — 学習窓幅・テスト窓幅を明記し、fold 別 CAGR/Sharpe 分布を提示
2. **パラメータ感度分析** — 主要パラメータを ±20% 振ったときの CAGR_OOS 変動幅
3. **コスト感度分析** — Scenario D に対し TER / swap_spread / SOFR を ±10% 振った結果
4. **税後 CAGR の併記** — §1.1 のドラッグレンジを引いた値を提示
5. **Sharpe with Rf > 0** — Rf=4% などで再計算した参考値

### §4.3 失格ライン

以下に該当する戦略は本書上「ベスト候補としては失格」：

- IS-OOS gap > +10 pp
- MaxDD < -80%
- Worst10Y★ < 0%
- Sharpe_FULL < 0.5（v2.0: フル期間基準。旧基準は Sharpe_OOS < 0.5）

### §4.4 統計検定の妥当性（経路依存指標のブートストラップ禁止・2026-06-21 独立QCで確立）

> 独立QCで判明した方法論的誤りの正典。**MaxDD等の経路依存極値に block≪危機長 のブロックブートストラップを当ててはならない。**

#### 規則 R-STAT-1: 経路依存極値（MaxDD / Worst10Y 等）は block=21 でブートストラップしない
- **MaxDD は数ヶ月〜数年の連続下落で決まる経路極値**。block=21（1ヶ月）でブロックをランダム再配置すると、2000-2002/2008 等の多年暴落シーケンスが破壊され、再サンプル経路のMaxDDは「月をシャッフルした浅い代理量」になる。
- 結果、`timing_P_maxdd` 等は**真の効果と無関係に 0.5 付近に固定**される（退避量fbarと単調相関＝退避量しか見ていない指紋）。**「有意差なし」は「効果なし」でなく「検定が無力」**。
- リポ自身の [`src/audit/multimetric_bootstrap_20260615.py`](src/audit/multimetric_bootstrap_20260615.py) docstring が Worst10Y について同型の無効性を明記済（block=252推奨）。**同じ経路依存の MaxDD に block=21 + 0.90 ゲートを当てるのは誤り。**

#### 規則 R-STAT-2: 経路依存指標の時機効果は「無傷の経路」で検定する
> ⚠️ **適用の前提条件（2026-07-02 検出力分析で確定・[P09_SCALE_CRITVERIFY_20260702.md](P09_SCALE_CRITVERIFY_20260702.md) §2.3）**: 本検定は**検定対象の戦略が危機窓内に十分な IN 日を持つ場合にのみ**検定力を持つ。DH-W1 系（危機の94-100%がOUT）では5窓中3窓がIN日ゼロ・実質 tri_2015 の n=1 となり、**完璧なオラクル時機シグナルを注入しても TIMING_WEAK 判定になる（第二種過誤100%）**。この条件下の TIMING_WEAK は「時機なしの証拠」ではなく「検定不能」と読むこと。窓内IN比率を必ず併記する。
正しい代替（実装済・[`src/audit/crisis_window_timing_20260621.py`](src/audit/crisis_window_timing_20260621.py)）:
1. **無傷の暴落窓**（dotcom_2000/gfc_2008/covid_2020/bear_2022/tri_2015）でブレーキ vs 対照のMaxDDを比較（リサンプリングしない＝経路を壊さない）。`_maxdd_from_returns(r[mask])`。
2. **窓横断サイン検定**（二項）で「対照より浅い回数」を集計。
3. 補助: block≥252 / avg-drawdown / time-under-water / CVaR-of-DD など**経路集約指標**なら block ブートストラップ可。
- 小サンプル注意: 暴落窓は5本程度＝「効果なしの証明」でなく「効果を支持する証拠の有無」。5/5でも片側 binom p=0.031。

#### 規則 R-STAT-3: 時機 vs デレバの切り分け（同一平均エクスポージャ対照）
ブレーキ/オーバーレイのDD削減が「時機スキル」か「単なる平均デレバ」かは、**同じ平均退避率を全期間一律適用した双子**と比較して切り分ける（`build_uniform_delever`）。双子を上回らなければ「デレバのみ」。**DD削減はレバ水準ダイヤル（B3c/低スケール）で行うのが本質**（exoticブレーキ/配分/時機の小細工は一律デレバを統計的に上回らない＝G5 vix・A7・A0・B1で繰り返し確認）。

#### 規則 R-STAT-5: 税係数 ×0.8273 の適用規約（2026-07-02 監査で明文化）
- **由来**: 0.8273 = 1 − 0.20315 × 0.85（譲渡益税20.315%・85%課税=15%繰延仮定）。
- **規約の使い分け**: 10指標表は metric-level（CAGR/W10Y/P10 のみ・MaxDD/Sharpe/最悪単日は税前）＝現実比やや保守で妥当。**年次リターン系列への「負年含む全年 ×0.8273」は損失の同率即時還付を仮定する楽観**で、現実（繰越3年）比で**年次経路 MaxDD を 4.7〜8.1pp・Worst5Y を最大1.3pp 過小評価**する（scale が高いほど拡大）。サバイバル系（labor-zero）や経路指標を年次系列から出す場合は繰越3年シミュ（`src/audit/tax_model_audit_20260702.py` の variant b）を併記すること。
- 一次根拠: [P09_SCALE_CRITVERIFY_20260702.md](P09_SCALE_CRITVERIFY_20260702.md) §1.1。

#### 規則 R-STAT-4: シミュレーション内の意思決定ルールのルックアヘッド監査（2026-07-02 確立）
- **粗い時間粒度（年次・月次）のシミュレーションで、意思決定ルールが「当期のリターン」を参照してから当期の行動を決めていないかを必ず監査する**。当期リターンは期末まで観測できないため、`if r[k] >= 0: act_at_start_of_k` 型のルールは1期分の未来情報を使う（labor-zero v6 の補填投入 hold_if_crash で発生・投入ダイヤル系の結論が最大±10pp 歪んだ）。
- 検出したら **lagged（前期リターン参照）／期末実行の実装可能規約**で再計測し、同一シード・同一経路のペア比較（McNemar）でアーティファクト幅を定量する。**主報告は実装可能側の値**とし、当期参照値は「楽観上限」と明記。
- 兆候: 「効くダイヤル」を発見したとき（早期投入・退避タイミング等）こそ情報規約を疑う。**「待つルールが有効」は未来の符号を知っている場合にだけ成立しがち**。
- 補助規則: N=2,000 程度の MC で 1pp 未満の差を語るときは独立シード比較でなく**同一経路ペア検定**を使う（分解能 1/N の壁）。
- 一次根拠: [LABOR_ZERO_V6_CRITVERIFY_20260702.md](LABOR_ZERO_V6_CRITVERIFY_20260702.md)（白紙再実装・3規約・8ハーネス）。

一次根拠: [A7_DD_REDUCTION_VARIATIONS_20260621.md](A7_DD_REDUCTION_VARIATIONS_20260621.md) §8、[B1_SCALE_FRONTIER_20260621.md](B1_SCALE_FRONTIER_20260621.md) §4、[MULTISTRATEGY_COMBINE_QC_SIGNOFF_20260616.md](MULTISTRATEGY_COMBINE_QC_SIGNOFF_20260616.md)（G5 vix 撤回）、R-STAT-4 は [LABOR_ZERO_V6_CRITVERIFY_20260702.md](LABOR_ZERO_V6_CRITVERIFY_20260702.md)。

---

## §5 レポーティング必須項目

### §5.1 ファイル命名規則

- **`FINAL_` プレフィックス禁止**
- `<TOPIC>_YYYY-MM-DD.md` または `REPORT_YYYY-MM-DD.md` 形式
- 旧レポート置換時は旧ファイル冒頭に `SUPERSEDED` ヘッダ追加（テンプレは [`CURRENT_BEST_STRATEGY.md`](CURRENT_BEST_STRATEGY.md) 参照）
- 新レポート発行と同時に `CURRENT_BEST_STRATEGY.md` と `tasks.md` を更新

### §5.2 必須ヘッダ（レポート冒頭）

```markdown
# <Strategy Name> 検証レポート

- レポート日: YYYY-MM-DD
- EVALUATION_STANDARD バージョン: v2.0（執筆時の最新版を記載）
- コスト Scenario: **D**（src/product_costs.py 2026-05-12 基準）
- 期間: IS 1974-01-02〜2021-05-07 / OOS 2021-05-08〜YYYY-MM-DD / FULL 1974-01-02〜YYYY-MM-DD
- DELAY: 2 営業日
- データソース: data/NASDAQ_extended_to_2026.csv
- 主要コード: src/xxx.py, src/yyy.py
- CURRENT_BEST_STRATEGY 整合: [YES / NO（理由）]
```

### §5.3 必須指標テーブル

```markdown
| 指標 | IS | OOS | FULL | 備考 |
|---|---|---|---|---|
| CAGR | x.xx% | x.xx% | x.xx% | |
| Sharpe (Rf=0) | x.xxx | x.xxx | x.xxx | 高金利期で過大評価注意 |
| MaxDD | — | — | -x.xx% | FULL 期間 |
| Worst5Y CAGR | — | — | x.xx% | 日次ローリング |
| Worst10Y★ CAGR | — | — | x.xx% | カレンダー年ベース |
| P10_5Y▷ CAGR | — | — | x.xx% | 5年CAGR 分布 P10 |
| IS-OOS gap | — | — | +x.xx pp | |
| Trades/yr | — | — | 約 xx 回 | |
```

### §5.4 必須セクション（順序固定）

1. **サマリ**（採用判定 / 旧ベストとの差分・3〜5行）
2. **必須指標テーブル**（§5.3）
3. **戦略構成**（シグナル / レバレッジ / 配分 / 実装ファイル）
4. **コスト前提注記**（Scenario D の確認 + Gold proxy ギャップ -10.5 bps/yr 等）
5. **頑健性チェック結果**（§4.1 全項目 + §4.2 採用分）
6. **税考慮後の現実推計**（CAGR_OOS - 2.8〜5.2% の幅で提示）
7. **CURRENT_BEST_STRATEGY との比較**（指標横並び表）
8. **判定**（PASS / FAIL + 理由 + 後継 / 廃止の有無）
9. **改訂履歴**

### §5.5 レポート PR マージ前チェックリスト

- [ ] §5.2 必須ヘッダ全項目あり
- [ ] §5.3 必須指標テーブル全行埋まっている
- [ ] Worst10Y は ★ 付きでカレンダー年方式と明記
- [ ] Scenario D を使用、または非標準なら §6 に従い参考値表記
- [ ] §4.1 必須頑健性チェック全項目あり
- [ ] CURRENT_BEST_STRATEGY.md との比較表あり
- [ ] 採用 / 不採用の判定が明示
- [ ] CURRENT_BEST_STRATEGY.md と tasks.md の更新提案あり（採用時）
- [ ] **sweep スクリプトの場合**: §3.12 実装チェックリスト全項目（Trades_yr, WFA placeholders, <br>ヘッダ）

---

## §6 「参考値」判定フロー

新規 / 既存レポートの数値を見たとき、以下のフローで「正値」「参考値（非標準）」「廃止値」を判定する：

```
START
  │
  ├─ コスト Scenario は D か？
  │     ├─ NO（A/B/C/不明） ───► 【参考値】§1.3 該当
  │     └─ YES
  │           │
  ├─ src/product_costs.py 2026-05-12 と一致したパラメータか？
  │     ├─ NO（独自 TER/swap など）─► 【参考値】§1.3 該当
  │     └─ YES
  │           │
  ├─ 期間定義は §2.1 と一致するか？
  │     ├─ NO（独自 IS/OOS 境界、独自データ）─► 【参考値】期間非標準
  │     └─ YES
  │           │
  ├─ DELAY = 2 営業日か？
  │     ├─ NO ─► 【参考値】look-ahead bias 疑い
  │     └─ YES
  │           │
  ├─ Worst10Y はカレンダー年方式（★）か？
  │     ├─ NO（日次ローリング） ─► 当該指標のみ【旧値・廃止】
  │     │                          他指標は条件を満たせば正値
  │     └─ YES
  │           │
  ├─ WF / CV の最良 fold を OOS の代わりに使っているか？
  │     ├─ YES ─► 【参考値】§2.3 違反
  │     └─ NO
  │           │
  ├─ timing_signals_raw.csv 由来の P-series か？
  │     ├─ YES ─► 【参考値】§1.3-3 該当
  │     └─ NO
  │           │
  └─► 【正値】(EVALUATION_STANDARD v2.0 準拠)
```

**運用ルール**:
- 「参考値」と判定された数値は `CURRENT_BEST_STRATEGY.md` の判定根拠にできない
- レポート内に「正値」と「参考値」が混在する場合は、表内で **(参考値)** または **※非標準** とラベルを付ける
- 「廃止値」は引用しない。歴史的経緯として言及する場合は「廃止」と明示する

---

## §7 改訂履歴

| 版 | 日付 | 主な変更 |
|---|---|---|
| v1.0 | 2026-05-21 | 初版発行。Scenario D を現行標準として確定。Worst10Y★ をカレンダー年方式に統一。Sharpe Rf=0 の過大評価注意を明文化。参考値判定フロー（§6）を導入。 |
| v1.1 | 2026-05-22 | §3.9 WFA_CI95_lo・§3.10 WFA_WFE を WFA補助指標として追加（旧§3.9 コード参照を §3.11 に繰り下げ）。非標準WFA指標（Stable_Sharpe・WinRate_yr・WorstK5_mean_CAGR・IR_vs_BH）を廃止。統一指標セットを7+2=9指標に確定。§3.12 sweep スクリプト標準（9指標・WFA ポリシー・<br>列折り返し）追加。§5.5 sweep チェックリスト追加。 |
| v1.1.1 | 2026-05-22 | §3.12 に「CAGR は CAGR_OOS の1列のみ・手書きヘッダ禁止・戦略比較用 MD_HEADER_STRAT」を明文化。戦略比較スクリプト用チェックリストを新設。`src/_sweep_format.py` に `MD_HEADER_STRAT` / `fmt_row_strat` を追加しスコープを戦略比較まで拡張。 |
| v1.2 | 2026-05-27 | §3.12 統一指標セットに #8 `OvFit`（過学習リスクスコア）を追加し 9指標→10指標 / 11列ヘッダに拡張。`MD_HEADER_1P/2P/STRAT` / `fmt_row_1p/2p/strat` を全て更新し、`_ovfit()` ヘルパーを新規追加。OvFit は `\|IS_OOS_gap\|` から自動算出（✅ LOW: ≤2pp / ⚠ MED: ≤5pp / ❌ HIGH: >5pp）のため CSV 列追加は不要。 |
| v1.3 | 2026-05-28 | §3.12 Overfit(WFE) を WFA_WFE ベース判定に変更し OvFit列とWFE列を1列に統合（10列 = 9指標標準）。 |
| **v1.4** | **2026-06-02** | **§3.12 列順変更: IS-OOS gap CAGR を CAGR_OOS の右隣（第2列）へ移動・列名に CAGR 単位を明示。列ヘッダを4行折り返しに変更（状態凡例マーカ ⓽/ⓒ/ⓞ/ⓡ を独立行へ）。取引コスト評価を日次レベル必須化（年率近似は禁止）。yr_cost 列の表内表示禁止。`MD_HEADER_1P/2P/STRAT` 更新。** |
| **v1.5** | **2026-06-05** | **§3.13 新設: Active 候補昇格の保守的採用基準** ─ min(IS, OOS) CAGR + Worst10Y★ + P10_5Y▷ の 3 軸保守的尺度を Active 昇格の必須条件として確定。**OOS 単独評価 (CAGR_OOS のみ) は Active 昇格判断に使用禁止** (regime fit / selection bias / sample size 非対称の補正)。WFE 補助判定 (>1.5 で regime luck 警告) も明文化。適用例 (vz=0.65+l7+F10ε-AH v4.4 採用→v4.5 棄却) と環境別 Active 候補 (CFD: vz=0.65+l7+F10ε, ETF: DH-W1) 表を §3.13 末尾に記載。STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md §7-2 と整合。 |
| **v1.6** | **2026-06-08** | **§3.13 改訂: 「3 軸必須」条件を削除、min(IS, OOS) CAGR の標準化のみ残存**。Worst10Y / P10_5Y は §3.12 の 9 指標として参照するが強制条件ではない。過度に restrictive な「3 軸すべて baseline 以上」必須条件は実際の戦略判断には適用困難 → ユーザー裁量を残す形に簡素化。過去判定 (AH/AT/HL 棄却等) は当時のルールに基づくため変更せず。 |
| **v1.7** | **2026-06-17** | **§1.5 新設: レバレッジ>3x（取引所CFD・証拠金取引）の証拠金取り置きコスト前提を標準化**。>3x戦略は「証拠金率8%（最小4.24%の2倍）を無利息で取り置き＋資金制約＋四半期ロールコスト」で評価必須、証拠金ゼロ前提は非標準（参考値）。`src/product_costs.py` に K365_* 定数追加。§0サマリ・PRチェックリストに反映。一次根拠 MARGIN_CAPACITY_STRESS_RESULTS_20260617.md（※当時の現実ドラッグ scale1.35 ≈−0.9pp は v1.8 で撤回）。 |
| v1.8 | 2026-06-17 | **（v1.9で撤回）** §1.5 で機会損失を「現金取り置き×(戦略リターン−SOFR)」とし、リスク較正バッファ f_cal で scale1.35強map −3.42pp(→+20.41%) 等と計算。→ これは「過少投資（運用額の一部を遊休現金にする）」のモデルで、証拠金の仕組み上は不要な任意の保守姿勢＝強制コストでないため v1.9 で却下。 |
| **v1.9** | **2026-06-18** | **§1.5 最終是正（最重要）: 証拠金は「ポジションを裏付ける自己資金＝担保」で、損益は建玉全額に発生する（eToro）→ 証拠金は継続的なCAGRドラッグを生まない**。∴ バックテストの税後CAGR（金利相当額控除済）は到達可能（例 scale1.35強map +23.83%。M6で強制清算は1975-77に3回・各13-19%AUMだがCAGR影響+0.08pp・min⓽不変、5大危機はOUTで清算0）。**v2(×SOFR −0.9pp)・v3(現金取り置き −3.42pp)とも却下**。唯一の継続コストは金利相当額（≈SOFR+スプレッド・コストモデル計上済）、円建てでFX無し。高レバの実コストはCAGRドラッグでなく**テールリスク（強制ロスカット）**で、低減はレバ水準で。コスト前提の本体・URL根拠は [PRODUCT_COST_COMPARISON §10](PRODUCT_COST_COMPARISON_2026-06-10.md)。独立QC（前提攻撃型・URL本文確認）APPROVE_WITH_CAVEATS。 |
| **v2.0** | **2026-06-19** | **§3.12 統一指標を 9指標 → 10指標へ刷新（メジャー）**。変更: ①CAGR_IS を第1列追加（v1.1「OOS1列のみ」廃止・min/gapが一目）②IS-OOS gap 列削除（2CAGR列から導出・頑強性の内部材料化）③Sharpe を OOS→**フル期間**（統計頑健・他リスク指標とFull整合・旧OOS値は「OOS(旧)」明記）④**最悪単日（テール）を MaxDD直後に追加**（強制ロスカット引き金・発生日小書き）⑤**Worst5Y 追加** ⑥旧 Overfit(WFE)+CI95_lo を**「頑強性・過学習」1列に統合**（判定+証拠2-3行: WFE/CI95/CPCV/t_p/Regime）⑦◎/★を**フルSharpe実測で再較正**（◎+0.934=E4現Active / ★+1.100=B3aベスト・旧0.770/0.885廃止）。実装: `src/audit/unified_metrics.py`（Sharpe_FULL/Worst1D追加）・`src/_sweep_format.py` v2.0（ヘッダ4本/fmt_row4本/`_worst1d`/`_robustness_cell`、テスト 4+24 PASS）。回帰サニティ: 既存min⓽/MaxDD 完全再現。Sharpe_FULL/最悪単日は実NAVで実算出（推定不採用）。 |
| **v2.1** | **2026-06-22** | **§4.4 統計検定の妥当性 新設（独立QC確立）**: block=21 全系列MaxDDブートストラップが**経路依存極値に無効**と判明（多年暴落の並びを破壊し timing_P を真の効果と無関係に0.5付近へ固定・リポ自身の `multimetric_bootstrap` docstring が Worst10Y で同型を明記済なのに MaxDD に未適用だった）。R-STAT-1（経路依存極値に block ブートストラップ禁止）/ R-STAT-2（無傷の暴落窓＋窓横断サイン検定で時機を検定・`crisis_window_timing_20260621.py`）/ R-STAT-3（時機 vs デレバは等平均退避の一律デレバ双子で切り分け・DD削減はレバ水準ダイヤルが本質）を確立。§0サマリ・PRチェックリストにサーフェス。一次根拠 [A7_DD_REDUCTION_VARIATIONS_20260621.md](A7_DD_REDUCTION_VARIATIONS_20260621.md) §8 / [B1_SCALE_FRONTIER_20260621.md](B1_SCALE_FRONTIER_20260621.md) §4。 |

### 今後の改訂方針

- **OOS が延長された場合**（§2.2）: §0 と §2.1 の終端日を更新。版番号は v1.x（マイナー）
- **コストモデルが変わった場合**（例: SOFR proxy 変更、新製品追加）: §1 を更新。版番号は v2.0（メジャー）
- **指標が追加 / 廃止された場合**: §3 を更新。版番号は v1.x または v2.0（影響度による）
- **必ず `CURRENT_BEST_STRATEGY.md`・`tasks.md`・`FILE_INDEX.md` と同時更新**すること

---

*管理者: 男座員也（Kazuya Oza） / 本書は `CURRENT_BEST_STRATEGY.md` と対の正典です。両者の整合性が常に保たれていることを Claude / 人間ともに確認してください。*
