# 正典表修正提案（CANONICAL_TABLE_REVISION_PROPOSAL）

- 作成日: 2026-06-10
- 最終更新日: 2026-06-10
- 目的: `CURRENT_BEST_STRATEGY.md` v4.5 推奨表および `EVALUATION_STANDARD.md` に対して、2026-06-10 検証で確定した各事実を「現行→提案」形式で提示し、ユーザー承認後の正典反映を可能にすること
- 一次根拠:
  - `audit_results/VERIFICATION_REPORT_20260610.md`（v3・確定値、以下 VR）
  - `audit_results/SBI_CFD_FINANCING_BASIS_20260610.md`（以下 BASIS）
  - `audit_results/wfa_realistic_summary_20260610.csv`（以下 WFA_CSV）
- 対象ファイル（提案のみ・変更禁止）:
  - `CURRENT_BEST_STRATEGY.md`（v4.5 推奨表）
  - `EVALUATION_STANDARD.md`（§1.1, §3.12, §3.13）
  - `tasks.md`（残課題 Pending エントリ）
- ステータス: **提案のみ。本提案では正典を一切変更していない。**

---

## R4 — DH-W1 Trades/yr 誤り訂正

### 修正内容

| 項目 | 現行値（v4.5 推奨表） | 提案値 |
|---|---:|---:|
| DH-W1 `Tradeⓞ/yr` 列 | **68.7** | **17.6** |

### 根拠

VR §1「70セル差分マトリクス」ETF群表：

```
DHW1 現 Trades/yr = 68.7✗   DHW1 SD = 17.6   DHW1 RE = 17.6
```

VR §0 Q3 にて確定：
> ❌ **DH-W1 Trades/yr「68.7」は誤り**（NAV符号反転回数の疑似指標 `recompute_9metrics_for_decision.py::trades_per_yr_from_nav`。実リバランス = 17.6 回）— R4

`simulate_rebalance_A` による閾値越えカウント（`EVALUATION_STANDARD §3.7` 正典定義）は 17.6 回。68.7 は同関数とは無関係な NAV 符号反転カウントの誤実装値。

### 反映先

`CURRENT_BEST_STRATEGY.md` v4.5 推奨表 DH-W1 行 `Tradeⓞ/yr` 列を `68.7` → `17.6` に置換。  
また `EVALUATION_STANDARD §3.7` の「現行ベスト戦略の基底 DH Dyn シグナルで約 27 回/年（月約 2.3 回）」の記述との整合も確認すること（E4 の 27 回と DH-W1 の 17.6 回は別物のため混同注意）。

---

## R2 / R10 — CAGR 列の⓽税後明示・ⓒ税前との基準分離

### 修正内容

v4.5 推奨表の列ヘッダおよび各行ラベルについて、CAGR 値が税後（⓽）であることを現行以上に明示し、Sharpe/MaxDD 等のⓒ（税引き前）指標との基準差を注記する。

#### 現行（v4.5 推奨表冒頭注記の抜粋）

```
> **全値⓽税後（手取り）**。CAGR は IS/OOS 両記載、**min(IS,OOS)** = 保守的採用基準値。
```

Sharpe/MaxDD/Worst10Y/P10 の各列ヘッダは `Sharpeⓒ`、`MaxDDⓒ`、`Worst10Y★⓽`、`P10⓽ 5Y` と明記されているが、**「全値⓽」という冒頭宣言が CAGR 以外の列にも誤って適用されるリスク**がある。

#### 提案

1. 冒頭注記を以下に変更する：

```
> **CAGR（IS/OOS）は⓽税後（手取り）。Sharpe / MaxDD はⓒ税前（コスト後）。Worst10Y★ / P10 は⓽税後。**  
> **min(IS,OOS)** = 保守的採用基準値（§3.13）。
```

2. 列ヘッダに付与済みの `ⓒ`/`⓽` マーカは現行どおり維持する（変更不要）。  
3. CFD 行（E4 / vz065）の CAGR⓽ 値は SBI CFD 3.0% 基準のものであることを行内 `Status` 列注記に明示する（後述 R7 と連動）。

### 根拠

VR §0 Q1：
> CFD / ETF / 投信の **3別コスト環境**。同一表内で **CAGR=⓽税後 と Sharpe/MaxDD=ⓒ税前 が混在**（R2）。

VR §3 R2 判定: 確定。

`EVALUATION_STANDARD §3.12` 凡例マーカ定義：
- **⓽** = 税後（手取り）: CAGR_OOS, Worst10Y★ CAGR, P10_5Y▷ CAGR
- **ⓒ** = コスト後（税引き前）: Sharpe_OOS, MaxDD

本提案はこの正典定義を冒頭注記の文言に整合させるものである。

### 反映先

`CURRENT_BEST_STRATEGY.md` v4.5 推奨表の冒頭注記ブロック（`> **全値⓽税後**...` の行）。

---

## R7 — CFD realistic（SBI CFD / full L×）基準値の併記

### 修正内容

v4.5 推奨表の CFD 行（E4 / vz065_l5 / vz065_l7）に、realistic（SBI CFD 3.0%, full L×）基準の CAGR_OOSⓒ 税前値と WFA 実値を併記する。

#### 現行値（v4.5 推奨表）

| 戦略 | CAGR⓽ IS/OOS(min) | Sharpeⓒ | CI95ⓡ_lo | WFEⓞ |
|---|---|---:|---:|---:|
| E4 RegimeKLT (§1 Active) | IS +20.0% / OOS +22.4%（min +20.0%） | 0.79 | +16.3% | ✅ 1.15 |
| vz=0.65+l5 | IS +20.16% / OOS +18.93%（min +18.93%）⚠ | 0.841 ⚠ | N/A | ✅ 1.389 ⚠ |
| vz=0.65+l7 | IS +20.23% / OOS +21.49%（min +20.23%）⚠ | 0.829 ⚠ | N/A | N/A |

#### 提案値（realistic full L× 確定値）

| 戦略 | realistic CAGR_OOSⓒ（full L×） | (L-1)×借入のみ（下限感度） | WFA CI95_lo（full L×） | WFA WFE | 判定 |
|---|---:|---:|---:|---:|---|
| E4 RegimeKLT | **+21.83%** | +25.43% | **+16.64%** | **1.211** | CAUTION |
| vz=0.65+l5 | **+25.20%** | +29.14% | **+16.92%** | **1.477** | CAUTION |
| vz=0.65+l7 | **+27.16%** | +31.16% | **+17.37%** | **1.461** | CAUTION |

##### 税後換算（CAGR_OOSⓒ × 0.8273）

| 戦略 | CAGR_OOS⓽（realistic） |
|---|---:|
| E4 | **+18.06%** |
| vz065_l5 | **+20.85%** |
| vz065_l7 | **+22.47%** |

### 根拠

一次根拠: VR §0「現実コスト下のランキング（訂正後・realistic full L×）」、VR §2「2b. WFA（修正後 realistic NAV・full L×基準）」、WFA_CSV 全行。

CFD 財務 full L× が正式 Realistic 基準であることの根拠: BASIS §4 判定：
> **判定: full L×（想定元本全額）**。くりっく株365公式計算式・Saxo Bank 公式「約定代金の全額に対して課される」・IG証券「ポジション全体の金額が対象、証拠金を除いた借入相当分ではない」から確定。

リポジトリの SBI CFD 3.0% 修正値（E4 +22.4%）との整合確認（VR §0 Q2）：
> **E4 CAGR_OOS: realistic(full L×) +21.83% がリポジトリ修正後 +22.4% とほぼ一致 → 相互検証成立。リポジトリ修正の妥当性を独立確認。**

### 反映先

`CURRENT_BEST_STRATEGY.md` v4.5 推奨表の CFD 行ごとに「realistic(full L×)」欄を追加、または `Status` 列 / 注記に実値を埋め込む。  
既存の `⚠ 値の出典・コスト前提未確認` フラグは解消してよい（確認済みのため）。

---

## R9 — vz065_l7 の CI95 / WFE「N/A」を実値に補填

### 修正内容

| 項目 | 現行（v4.5 推奨表） | 提案値（realistic full L×） |
|---|---|---|
| vz065_l7 `CI95ⓡ_lo` | N/A | **+17.37%**（realistic full L×） |
| vz065_l7 `WFEⓞ` | N/A | **1.461**（CAUTION） |

参考値（scenarioD）:

| 項目 | scenarioD 値 |
|---|---|
| vz065_l7 CI95_lo（scenarioD） | +20.78% |
| vz065_l7 WFE（scenarioD） | 1.494 |

### 根拠

WFA_CSV（`wfa_realistic_summary_20260610.csv`）3行目：
```
vz065_l7,realistic,full,52,0.173667,1.461392,0.000008,0.300662,CAUTION,PASS
```

VR §0 Q3：
> ✅ 副候補 l7 の CI95/WFE「N/A」を確定（R9）。

### 反映先

`CURRENT_BEST_STRATEGY.md` v4.5 推奨表 vz=0.65+l7 行の `CI95ⓡ_lo` と `WFEⓞ` 列。

---

## R3 — DH-W1 CAGR の二重値整理

### 修正内容

v4.5 推奨表では DH-W1 の CAGR として canonical split 値（IS +15.31% / OOS +15.74%）が掲載されているが、投信環境セクションの Cash Sleeve baseline 表では「+13.66%（旧 split 基準）」が DH-W1 baseline として表示されており、**同一戦略に 2 つの CAGR 値が存在する状態**が続いている。

#### 現行

- v4.5 推奨表（ETF only 行）: `IS +15.31% / OOS +15.74%（min +15.31%）`（canonical split, pretax +18.08/18.96%の⓽税後）
- 投信環境 Cash Sleeve 表 baseline 行（†付き）: `+13.66%`（旧 split 基準、†注記あり）

#### 提案

1. Cash Sleeve baseline 表の「†」注記を以下に改める（実値の確定的整理）：

```
†: +13.66% は旧 OOS split（2020-12-31）基準の税後 CAGR。
   canonical split（2021-05-07 / IS_END=2021-05-07）の税後 CAGR は OOS +15.74%（pretax +18.96%）。
   各 Cash Sleeve 戦略の CAGR 差分（例: P7 +14.90% vs baseline +13.66% = +1.24pp）は
   旧 split 基準どうしの相対比較として正しい。
   ただし canonical 基準での DH-W1 baseline は +15.74% であり、この差分に +2.08pp を加えた
   canonical ベースでの比較値（例: P7 canonical ≈ +16.98pp 相当）も参考として示せる。
```

2. v4.5 推奨表の DH-W1 行から「旧 split +13.66%」への参照を削除し、canonical 値のみに統一する。

### 根拠

VR §3 R3 判定: 確定・解明：
> +15.31/15.74%はpretax+18.08/18.96%の税後値。+13.66%は別split/別コスト

VR §1「ETF群 + 投信」表：
```
DHW1 現 CAGR_OOS = +15.74%⓽   DHW1 SD = +18.96%   DHW1 RE = +18.68%
```

### 反映先

`CURRENT_BEST_STRATEGY.md` 「投信環境 Active 候補」セクションの Cash Sleeve 表 DH-W1 baseline 行 `†` 注記。  
「†: +13.66%(旧split)は ～ 」の注記を上記文言に差し替える。

---

## 二面性の明記（新規注記提案）

### 提案

v4.5 推奨表の末尾、または「v4.5 整理対象ファイル」セクションの前に以下の注記ブロックを追加する：

```markdown
### ⚠ 環境別優位性の二面性（2026-06-10 検証確定）

realistic（full L×）コスト条件下での比較ポイント：

- **税引前 CAGR（生値）**: CFD 群（E4 / vz065）が上位。ただし vz065 は IS-OOS gap が大きく
  （l5: −4.84pp、l7: −6.84pp）、OOS 期の regime luck 疑いがある。
- **min(IS, OOS) 保守基準**: CFD 群と ETF 群の差は +1〜2pp 程度に縮小。
- **after-tax（税口座考慮）**: CFD（課税 × 0.8273）vs ETF（NISA 非課税）。
  E4 税後 CAGR_OOS +18.06% は ETF 群（DH-W1 +18.68%・V7 +18.85%）に**逆転される**。
- **Sharpe / MaxDD**: ETF 群（特に V0 / DH-W1）が優位（V0 MaxDD −28.86% 最良）。

→ **「CFD 一択」でも「ETF 一択」でもない。税口座（NISA 可否）・リスク選好・レバ管理コスト
  の3軸で選択することが妥当。**

（根拠: `audit_results/VERIFICATION_REPORT_20260610.md` §0 エグゼクティブサマリ「読み方（重要）」）
```

### 反映先

`CURRENT_BEST_STRATEGY.md` v4.5 推奨表セクション末尾（「棄却された v4.x 改善案」の前）。

---

## EVALUATION_STANDARD §1.1 への追記提案

### 提案 A — 税乗数 0.8273 の出典明記

`EVALUATION_STANDARD §1.1` の「日本居住者税」行の直下または `src/product_costs.py` 参照部分に以下を追記する：

```markdown
- **税後 CAGR 乗数**: `0.8273 = 1 − 0.85 × 0.20315`（利益の約85%相当が課税対象との想定。
  `src/g17_trade_cost_adjustment.py` の `EFFECTIVE_TAX_FACTOR = 1.0 - 0.85 * TAX_RATE` 定義参照）。
  計算スクリプト: `scripts/compute_aftertax_cagr_v3.py`（または `compute_aftertax_cagr_v3_20260607.py`）。
  本スクリプトが canonical split（2021-05-07）での pretax → aftertax 変換の正典実装。
```

### 提案 B — CFD 財務課金ベース（full L×）を Scenario D に追記

`EVALUATION_STANDARD §1.1` の製品別パラメータ表の下に以下の注記を追加する：

```markdown
- **CFD 建玉金利の課金ベース（2026-06-10 確定）**: Scenario D の CFD 財務コストは
  **想定元本全額（full L×）** に課されることを公式情報で確認済み（出典: `audit_results/SBI_CFD_FINANCING_BASIS_20260610.md`）。
  - くりっく株365: 公式計算式「清算価格 × 取引単位 × 利率 × 日数/365」より full L× 確定。
  - IG証券: 「ポジション全体の金額が対象、証拠金を除いた借入相当分ではない」と公式明示。
  - Saxo Bank: 「約定代金の全額に対して課される」と公式明示。
  - **(L-1)× borrowed モデルは CFD 構造の誤解に基づく下限感度値として扱う**（参考のみ）。
  - realistic Scenario での `spread_cost` 実装: `wn × lev × L` 全体変化量への適用が正しい
    （`src/audit/strategy_runners.py` DELAY=2・spread=wn×lev×L 版）。
```

### 反映先

`EVALUATION_STANDARD.md` §1.1（`src/product_costs.py` パラメータ表の直後）。

---

## 反映先候補リスト

### CURRENT_BEST_STRATEGY.md

| 修正ID | 対象箇所 | 変更行イメージ |
|---|---|---|
| R4 | v4.5 推奨表 DH-W1 行 `Tradeⓞ/yr` | `68.7` → `17.6` |
| R2/R10 | v4.5 推奨表 冒頭 `> **全値⓽税後**...` 行 | `CAGR は⓽税後。Sharpe/MaxDD はⓒ税前。Worst10Y★/P10 は⓽税後。` に置換 |
| R7 | v4.5 推奨表 E4 行 Status 列 / 注記 | realistic CAGR_OOS +21.83%（full L×）/ WFA CI95+16.64%/WFE1.211 を追記 |
| R7 | v4.5 推奨表 vz065_l5 行 Status 列 / 注記 | realistic +25.20% / CI95+16.92%/WFE1.477 追記。`⚠ 要再計算` フラグ解消 |
| R9 | v4.5 推奨表 vz065_l7 行 CI95 / WFE 列 | `N/A` → `+17.37%` / `1.461（CAUTION）` |
| R3 | 投信環境 Cash Sleeve 表 DH-W1 baseline †注記 | 旧 split +13.66% と canonical +15.74% の関係を明示（本提案 R3 節文言を使用） |
| 二面性 | v4.5 推奨表末尾（棄却案の前） | 二面性ブロック新規追加 |

### EVALUATION_STANDARD.md

| 修正ID | 対象箇所 | 変更行イメージ |
|---|---|---|
| §1.1-A | §1.1「日本居住者税」直下 | 税乗数 0.8273 の出典スクリプト明記 |
| §1.1-B | §1.1 製品別パラメータ表の直後 | CFD 建玉金利 full L× 確定・BASIS 出典明記・borrowed は下限感度扱いと明記 |

### tasks.md

| 対応 | 変更行イメージ |
|---|---|
| R9（N/A 解消）を Completed に移動 | `- [x] vz065_l7 CI95/WFE N/A 補填 → +17.37%/1.461（realistic）確定（2026-06-10）` |
| R7 反映を Pending として追記 | `- [ ] CURRENT_BEST_STRATEGY.md CFD 行に realistic full L× 実値を反映（R7）` |
| R4 反映を Pending として追記 | `- [ ] CURRENT_BEST_STRATEGY.md DH-W1 Trades/yr 68.7→17.6 修正（R4）` |
| P7 照合（残課題）を Pending として追記 | `- [ ] P7 CAGR⓽ +14.90%（旧split・cash_sleeve sim）と canonical +19.33% の差分を個別照合` |

---

## 未解決の残課題

以下の事項は本 2026-06-10 検証では解消されておらず、次回セッションで照合が必要である：

1. **P7 「+14.90%」の旧 split 税後値の個別照合**  
   - 現在 `CURRENT_BEST_STRATEGY.md` 投信環境表の P7 CAGR_OOS は `+14.90%`（旧 OOS split 基準・cash_sleeve sim）。  
   - 検証ハーネスの scenarioD 再計算では `+19.33%`（VR §1）、realistic では `+19.14%`（VR §2）。  
   - この乖離（約 4.4pp）は split 日付・コスト体系・シミュレーション実装の違いが複合している可能性があり、**単純な誤りとは断定できない**。個別に出典を追跡し、どちらの値がどの条件で正しいかを確定する必要がある。

2. **`PRODUCT_COST_COMPARISON_2026-06-10.md` の main ブランチへのマージ**  
   - 現在 `claude/review-repository-gKLyi` ブランチのみ存在（VR §4 末尾記載）。
   - 内容の確認後、main へのマージまたは audit_results への再作成を検討する。

3. **P7 投信 t_p（permutation 検定）・block bootstrap の未実施**  
   - WFA 50窓 α∩β PASS は確認済みだが、正式 Active 昇格には統計検定の追加実施が必要（`tasks.md` Pending）。

4. **CFD 財務 full L× vs (L-1)× の SBI 約款一次確認**  
   - BASIS §5 限界事項：SBI 証券店頭 CFD の公式計算式 PDF は取得できていない。公式ページの文言・業界標準から full L× を採用しているが、約款 PDF による一次確認が望ましい。

---

## 適用に関する注意事項

> **本提案ファイルは修正の「提案」を記録するものであり、正典ファイルを一切変更していない。**  
> `CURRENT_BEST_STRATEGY.md`、`EVALUATION_STANDARD.md`、`tasks.md` への実際の変更は、  
> **ユーザー承認後に別セッションで実施すること。**

本提案の各修正項目は、`audit_results/VERIFICATION_REPORT_20260610.md`（v3 確定）に記録された検証成果物からの確定値のみを使用しており、捏造・推測値は含まない。

---

*管理者: 男座員也（Kazuya Oza）*
