# EVALUATION_STANDARD.md — NASDAQ Backtest 評価基準の単一の真実

> **このファイルは「戦略を評価するときの前提・指標・期間・コスト・レポート様式」を一意に固定するための正典です。**
> **新しい戦略検証・改良・比較を行う前に、必ず §0 を読み、各セクションの定義に従ってください。**
> **本書と矛盾する実装・レポートは「非標準」として §1.3 / §6 の参考値ルールが適用されます。**

- バージョン: **v1.0**
- 発行日: 2026-05-21
- 管理者: Kazuya Murayama
- 一次関連ファイル: [`CURRENT_BEST_STRATEGY.md`](CURRENT_BEST_STRATEGY.md), [`src/product_costs.py`](src/product_costs.py), [`tasks.md`](tasks.md), [`FILE_INDEX.md`](FILE_INDEX.md)

---

## §0 標準前提サマリ（ワンブロック・新検証着手前に必読）

> **このブロックを読まずに評価を始めないこと。** 本書の各セクション（§1〜§5）はこのサマリの根拠詳細です。

| 項目 | 標準値（v1.0） | 詳細 |
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
| レポート必須ヘッダ | バージョン / コスト Scenario / 期間 / DELAY / コード参照 | §5 |
| 参考値判定 | §6 のフローチャートに従う | §6 |

**チェック項目（新検証 PR 前に必須）**:
- [ ] Scenario D を使用したか（コスト無しは「参考値」扱い）
- [ ] FULL/IS/OOS の3区間で指標を出したか
- [ ] Worst10Y★ をカレンダー年方式で出したか
- [ ] DELAY=2 を使用したか
- [ ] §5 のレポートテンプレに沿ったか
- [ ] CURRENT_BEST_STRATEGY.md と整合確認したか

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

### §3.9 コード参照（指標の正典）

| 指標 | 実装ファイル |
|---|---|
| CAGR / Sharpe / MaxDD | `src/corrected_strategy_backtest.py` |
| Worst5Y / P10_5Y | `src/cfd_leverage_backtest.py` |
| Worst10Y★ | `src/compute_cfd_worst10y.py`（カレンダー年実装の正典）/ `src/b1_s2_lt2.py`（呼出側） |
| Trades/yr | `src/dynamic_leverage_strategies.py` (`simulate_rebalance_A`) |

新しい指標を追加する場合は、本書 §3 に式・コード参照・注意事項を追記してから採用する。

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
- Sharpe_OOS < 0.5

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
- EVALUATION_STANDARD バージョン: v1.0
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
  └─► 【正値】(EVALUATION_STANDARD v1.0 準拠)
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

### 今後の改訂方針

- **OOS が延長された場合**（§2.2）: §0 と §2.1 の終端日を更新。版番号は v1.x（マイナー）
- **コストモデルが変わった場合**（例: SOFR proxy 変更、新製品追加）: §1 を更新。版番号は v2.0（メジャー）
- **指標が追加 / 廃止された場合**: §3 を更新。版番号は v1.x または v2.0（影響度による）
- **必ず `CURRENT_BEST_STRATEGY.md`・`tasks.md`・`FILE_INDEX.md` と同時更新**すること

---

*管理者: Kazuya Murayama / 本書は `CURRENT_BEST_STRATEGY.md` と対の正典です。両者の整合性が常に保たれていることを Claude / 人間ともに確認してください。*
