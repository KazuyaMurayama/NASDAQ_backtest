# CURRENT BEST STRATEGY — 単一の真実 (Single Source of Truth)

> **このファイルは「現行のベスト戦略」を一意に特定するための正典です。**
> **「ベスト戦略は？」と問われた時、Claude / 人間ともにまずこのファイルだけを見れば良いように設計されています。**

作成日: 2026-05-11
最終更新日: 2026-06-07 (v4.9.2: 税後 CAGR を canonical split (IS_END=2021-05-07) で統一。`scripts/compute_aftertax_cagr_v3_20260607.py` で全戦略を pretax と同じ canonical split に同期、calendar/canonical split duality を廃止) / 2026-06-08 (標準10指標・IS/OOS min 表記・ⓒ/⓽コスト前提明確化 全セクション適用) / 2026-06-10 (v4.5推奨表: 全値⓽に統一・比較注意書き削除、V0/V7 overlay・P7投信の3行追加) / 2026-06-10 v2 (v4.5推奨表: E4 ⓽行追加で§1 Active vs vz065_l5 を同一基準で直接比較可能化) / **2026-06-10 v3 (コスト誤謬修正: E4 CAGR⓽ を CFD_SPREAD_LOW=0.20%/yr の誤値 +27.41% から SBI CFD 3.0% 正値 +20.0% に修正)** / **2026-06-10 v4 (構成・コスト注意事項削除、一次根拠を SBI CFD g14 ベースに更新、Shortlisted から CFD_SPREAD_LOW 誤値を除去)** / **2026-06-11 v5 (v4.5表を realistic full L×(SBI CFD建玉金利=想定元本全額, 一次確認済) + 正典窓(49窓)WFA で更新。E4 ⓽OOS: g14 (L-1)×借入基準 +22.4% → full L×正基準 +18.06% に訂正。R4: DH-W1 Trades 68.7→17.6 訂正(NAV符号反転の疑似指標を実リバランス値に修正)。R9: vz065_l7 N/A→CI95+16.45%/WFE1.328 補填。⚠マーク除去・vz065 WFE CAUTION注記に置換)** / **2026-06-11 v6 (ETF指定戦略 DH-W1/V0/V7/P7 の NASDAQ脚コストを CFD→TQQQ に是正。設計商品TQQQの正値で点指標を更新: DH-W1 min +14.73→+15.85%, V0 +13.43→+14.27%, V7 +15.07→+16.27%, P7 +15.84→+16.92%。CFD環境(E4/vz065)は不変。WFA列は†CFD基準据置・TQQQ再計算pending)**

---

## 🆕 v4.5 (2026-06-05) ─ 環境別 Active 候補 + 保守的採用基準導入 / v4.9 (2026-06-08) ─ ルール簡素化

### 保守的採用基準 min(IS, OOS) CAGR (現行 v4.9 確定ルール)
v4.5 (2026-06-05) で **min(IS, OOS) CAGR** を保守的期待リターン指標として導入、**v4.9 (2026-06-08) で標準化確定**。
- ✅ **min(IS, OOS) CAGR の標準化**: IS と OOS の低い方を保守的期待リターンとして使用、OOS 単独評価より優先
- ✅ WFE 補助判定: > 1.5 で regime luck 警告
- ❌ **削除 (v4.9)**: 当初の「3 軸 (min + Worst10Y + P10_5Y) すべて baseline 以上」必須条件は過度に restrictive と判断され撤回。Worst10Y / P10_5Y は §3.12 9 指標として参照するが強制条件ではない。総合判断はユーザー裁量
- 詳細: [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md §7-2](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md), [EVALUATION_STANDARD.md §3.13](EVALUATION_STANDARD.md)

### v4.5 推奨 Active 候補（環境別）— 標準10指標

> **全値⓽税後（手取り）**。CAGR は IS/OOS 両記載、**min(IS,OOS)** = 保守的採用基準値。
>
> **コスト前提 (v5 更新)**: 全行の CAGR⓽ / Worst10Y⓽ / P10⓽ は **realistic full L×（SBI CFD 建玉金利 = 想定元本全額）・正典窓(49窓)WFA** ベース（audit_results/audit_*_realistic.csv × 0.8273）。Sharpeⓒ / MaxDDⓒ は税前。旧 g14 (L-1)×借入基準の +22.4% は **過大評価**で廃止。NISA非課税環境では DH-W1/V0/V7 の税前ⓒ値がそのまま手取り。

| 環境 | 戦略 | **CAGR⓽ IS / OOS（min）** | IS-OOS gap⓽ | Sharpeⓒ | MaxDDⓒ | Worst10Y★⓽ | P10⓽ 5Y | Tradeⓞ/yr | WFEⓞ | CI95ⓡ_lo | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **§1 Active（比較基準）** | **E4 RegimeKLT**<br>S2+LT2, k_lo=0.1, k_hi=0.8<br>vz_thr=0.7, CFD | IS +16.93% / OOS +18.06%（**min +16.93%**） | **−1.13pp** | **0.678** | −65.05% | **+5.82%** | **−0.68%** | **27.1** | ✅ 1.094 | **+15.63%** | ◆ §1 Active（WFA G3 PASS・realistic full L× 正典窓49窓。旧 g14 (L-1)×基準 +22.4% は借入基準差・SOFR計上差により廃止） |
| **CFD Active候補 (v4.7)** | **vz=0.65+l5 (vz065_l5)** | IS +16.84% / OOS +20.85%（**min +16.84%**） | −4.01pp | 0.769 | −59.08% | +6.55% | +2.42% | 84.9 | 1.348 | +16.27% | WFE 1.35 CAUTION(postIS4窓 regime luck疑い)。full L×正典窓検証済み |
| ↳ 副候補 (攻め型) | vz=0.65+l7 (vz065_l7) | IS +16.81% / OOS +22.47%（**min +16.81%**） | −5.66pp | 0.764 | −68.44% | +6.27% | −1.64% | 104.7 | 1.328 | +16.45% | WFE 1.33 CAUTION(postIS4窓 regime luck疑い)。MaxDD深化・Trades多でリスク高 |
| **ETF only (NISA等)** | **DH-W1** (Asymm Hyst) | IS +15.85% / OOS +16.60%（**min +15.85%**） | −0.75pp | 0.883 | −34.44% | +9.54% | +4.79% | 17.6 | ✅ 0.996† | +13.69%† | 🟡 ETF 環境 Active 候補（v6: TQQQ-cost是正 +1.12pp） |
| ↳ overlay MaxDD優先 | **DH-W1 + mom63 V0 def**<br>M6 def {1.1, 1.0, 0.9, 0.8} | IS +14.27% / OOS +15.41%（**min +14.27%**） | −1.14pp | **0.912** | **−28.71%** | +9.49% | +4.81% | 31.2 | ✅ 1.043† | +12.57%† | 🟢 ETF overlay ADOPT（v6: TQQQ-cost是正 +0.84pp・MaxDD改善維持） |
| ↳ overlay CAGR死守 | **DH-W1 + mom63 V7 boost**<br>M6 def {1.2, 1.1, 1.0, 1.0} | IS +16.27% / OOS +16.80%（**min +16.27%**） | −0.53pp | 0.877 | −34.47% | +10.08% | +5.15% | 25.2 | ✅ 0.975† | +14.09%† | 🟡 ETF overlay候補（v6: TQQQ-cost是正 +1.20pp） |
| **投信環境 (NISA等)** | **DH-W1 P7** GOLD75/BOND25スリーブ | IS +18.89% / OOS +16.92%（**min +16.92%**） | +1.97pp | 0.861 | −48.10% | +11.41% | +6.38% | 17.6 | ✅ 1.042† | +16.57%† | 🟢 投信環境 Active 候補・中庸推奨⭐（v6: TQQQ-cost是正 +1.09pp） |

> **v6 注記 (2026-06-11) ─ ETF環境のNASDAQ脚コストモデル是正（CFD→TQQQ）**
> - **是正内容**: ETF指定戦略（**DH-W1 / V0 / V7 / P7**）の NASDAQ 脚は、エンジン既定の継承で **CFD財務（SOFR+3.0%×(L-1)≈19.9%/yr@3x）** で計上されていたが、これらは設計上 **TQQQ ETF（2×SOFR+swap0.5%+TER0.86%≈9.1%/yr）** で保有する戦略。設計商品 TQQQ のコストに是正した。
> - **効果（税後 min(IS,OOS)）**: DH-W1 +14.73→**+15.85%**(+1.12pp) / V0 +13.43→**+14.27%**(+0.84pp) / V7 +15.07→**+16.27%**(+1.20pp) / P7 +15.84→**+16.92%**(+1.09pp)。ランキング不変（P7>V7>DH-W1>V0）。Sharpe も +0.04〜0.05 改善、MaxDD はほぼ不変。
> - **検証**: 全戦略 harness が CFD版で既存 `run_*('realistic')` を **0.00pp** 再現（gate PASS）。ボラドラッグは両版とも同一 `L·r` 日次cumprodに内包＝差は財務コスト項のみ（二重計上なし）。一次根拠 `audit_results/tqqq_correction_etf_strategies_20260611.csv`。
> - **適用範囲**: **ETF指定戦略のみ**。§1 Active(E4)・vz065 等の **CFD環境戦略は CFD基準を維持**（CFDで実運用するため正しい）。
> - **† 注記（WFA pending）**: WFEⓞ / CI95ⓡ_lo 列は **CFD基準のまま**（TQQQでのWFA再計算は次工程）。点指標(CAGR/Sharpe/MaxDD/Worst10Y/P10)のみTQQQ是正済み。WFE方向はIS/OOS窓を等しく動かすため結論不変見込み。
> - **NISA非課税**: ETF/投信は税前ⓒ=手取り。TQQQ是正後 DH-W1 pretax min ≈ +19.16%、V7 ≈ +19.67%、P7 ≈ +20.45%（×1/0.8273）。
>
> **v5 注記 (2026-06-11)**
> - (i) **基準**: realistic full L×（SBI CFD建玉金利=想定元本全額）・正典窓(49窓)WFA。CAGR⓽ = pretax × 0.8273(税20.315%)。Sharpeⓒ / MaxDDⓒ は税前。
> - (ii) **二面性**: 税引前では CFD(E4/vz065)が CAGR 上位だが、(a) postIS4窓 regime luck 疑い(vz065 WFE>1.3 CAUTION)、(b) after-tax 20.315% 課税 vs ETF/投信 NISA非課税、(c) MaxDD −65% 超(E4)/−68%(vz065_l7) vs DH-W1 −34%/V0 −28%、(d) Sharpe 0.678(E4) vs 0.873(V0)/0.835(DH-W1) で ETF・投信が多指標優位。**CFD 一択ではない**。
> - (iii) **NISA非課税の場合**: ETF(DH-W1/V0/V7)/投信(P7)の Sharpeⓒ・MaxDDⓒ 値がそのまま手取りリターン基準になる（税後調整不要）。DH-W1 pretax CAGR +18.68% 等は NISA では税前=手取り。

全戦略 STRATEGY_REGISTRY §2 Shortlisted 登録済み。**§1 Active への正式昇格** は実運用変更を伴うためユーザー判断を要する。

### 棄却された v4.x 改善案 (min ルール下で REF を下回る)
- **vz=0.65+l7+F10ε-AH/AT/HL** (v4.4 採用→v4.5 棄却) — OOS 単独では魅力的だが min(IS, OOS) で REF 劣後、WFE>1.5 で regime luck 疑い (STRATEGY_REGISTRY §3 Rejected)
- **DH-T4** (v3 で push→v4 で破棄) — ETF レバ操作違反 (lev_mod 連続 scaling)
- **DH-Z2** (v4 採用→v4.3 で W1 に置換) — Trades 152→18 で W1 が優位

### v4.5 整理対象ファイル
- [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) v4.5 (§7-2 min ルール明文化、§0' 5 戦略 + min 表記、§6-4 AH 棄却根拠)
- [STRATEGY_DH_REFINEMENT_20260603.md](STRATEGY_DH_REFINEMENT_20260603.md) v1.1 (DH-W1 詳細検証)
- [STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md) (§2 に v4.5 推奨 2 件、§3 に AH/AT/HL 棄却 3 件)

---

## 投信環境 Active 候補 (2026-06-07 追加) ─ DH-W1 キャッシュ・スリーブ 1倍投信置換

> **位置付け**: ETF 環境 Active 候補 **DH-W1** は全営業日の **46.9%(6,171日)をキャッシュ 0% で待機**する。この待機資金を 1 倍投信(ゴールド/米国債20年超)で運用置換した派生系。1 倍投信は **SOFR/スワップなし・信託報酬<0.2%・5営業日ラグ**で実装可能（レバ脚は DELAY=2 / SOFR/スワップ込み、税 20.315%×0.8273）。§1 本番 Active (CFD: E4 Regime k_lt) の置換ではなく、**ETF/投信のみ環境(NISA 等)で DH-W1 を更に押し上げる候補**。

### OUT(キャッシュ)期 6,171日の各1倍資産の素の挙動
| 資産 | 年率リターン | 年率Vol | R/R | 含意 |
|---|---:|---:|---:|---|
| NASDAQ 1x | **−7.63%** ⚠ | 26.3% | −0.29 | キャッシュ期＝risk-off、NASDAQ は大損 → スリーブ不適 |
| Gold 1x | +4.07% | 21.5% | +0.19 | 堅実 |
| Bond 1x (米国債22yr) | **+6.51%** ✅ | **11.9%** | **+0.55** | 低ボラ＋プラスで最良 |

### 4戦略 標準指標（全コスト後・税20.315%後・WFA 50窓）

> ⚠ **DH-W1 baseline 表記注意**: 本表の DH-W1 値 (+13.66%) は **v4.5 §7-2 min(IS,OOS) CAGR = IS 値**（旧 OOS split 基準）。
> 上記 v4.5 推奨表の canonical OOS +18.91% とは **split 日付と計算基準が異なる**。各 Cash Sleeve 戦略の差分（+2.78pp 等）はこの +13.66% ベースの**相対比較値**として正しい。

| 戦略 | CAGR_OOS | Sharpe_OOS | MaxDD | Worst10Y★ | P10_5Y | IS-OOS gap | Trades/yr | WFE | CI95_lo | 性格 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DH-W1 (baseline) | +13.66%† | +0.844 | **−34.57%** | +8.40% | +5.11% | +1.62 | 17.8 | 0.997 | +13.95% | 現状(キャッシュ)† |
| 🟢 P2 GOLD100 | **+16.44%** | **+0.875** | −58.53% | +9.43% | +8.06% | **+0.97** | 18.8 | 1.229 | +16.04% | 攻め (OOS 最高) |
| 🟢 **P7 GOLD75/BOND25** ⭐ | +14.90% | +0.827 | −48.23% | +9.92% | +8.05% | +3.28 | 18.8 | 1.043 | +16.74% | **中庸推奨** |
| 🟢 P5 GOLD50/BOND50 | +13.28% | +0.758 | −35.97% | +10.08% | +8.09% | +5.50 | 18.8 | 0.875 | **+17.23%** | 守り (DD 最良) |

- **P2 GOLD100**: ベース比 +2.78pp の OOS 最高・最汎化(gap +0.97)。代償は MaxDD −58.5%(Gold 単独の宿命)。攻め優先向け。
- **P7 GOLD75/BOND25 ⭐(新規・推奨)**: P2 と P5 の中間最適。OOS +14.90% を確保しつつ Bond 混で MaxDD を −48.23% に緩和。総合バランス最良。
- **P5 GOLD50/BOND50**: MaxDD をベース同等(−36.0%)維持・CI95_lo/P10/Worst10Y 最高。守り最優先向け。gap +5.50pp(汎化やや弱)。

> **昇格保留の理由**: WFA 50窓は CI95_lo>0 (α) ∩ 0.5≤WFE≤2.0 (β) を全 PASS だが、**t_p(permutation 検定)・block bootstrap は未実施**。正式 Active 昇格にはこれらの統計検証が必要（[tasks.md](tasks.md) Pending）。全コスト表・執行ラグ検証は [CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md) §4-§6 参照。STRATEGY_REGISTRY §2 に 3 件登録済。

---

## Shortlisted（次善候補 / WFA 完了）

> CAGR 数値は SBI CFD 3.0%/yr ベースで未計算。詳細指標は [STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md) §2 参照。

| 戦略 | Tradeⓞ/yr | WFEⓞ | 棄却理由 |
|---|---:|---:|---|
| F8 R5_CALM_BOOST | 182 | ✅ 1.208 | Trades/yr E4 比 7 倍。OOS 偶然性疑い |
| F7v3+E4 A:tilt=2.0 | 183 | ✅ 1.203 | 同上 |
| LT2-N750 固定k=0.5 | 27 | ✅ 1.145 | E4 に Worst10Y★/IS-OOS gap/CAGR で劣後。fallback 候補 |

---

## 一次根拠ファイル（GitHub 上の正典）

| ファイル | 役割 |
|---|---|
| [7STRATEGY_PERFORMANCE_REPORT_20260529.md](7STRATEGY_PERFORMANCE_REPORT_20260529.md) | **E4 指標の一次根拠**（SBI CFD 3.0%/yr ベース、CAGR⓽ OOS+22.4%/min+20.0%） |
| [g14_wfa_sbi_cfd_summary.csv](g14_wfa_sbi_cfd_summary.csv) | g14 WFA SBI CFD サマリ（CI95_lo⓽=+16.3%、WFE=1.15） |
| [src/g14_wfa_sbi_cfd.py](src/g14_wfa_sbi_cfd.py) | g14 WFA 実行スクリプト（SBI_CFD_SPREAD=3.0%/yr） |
| [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md) | E4 パラメータ sweep（構造的採用根拠・64 config PASS 12） |
| [src/e4_regime_klt.py](src/e4_regime_klt.py) | E4 実装 |
| [src/product_costs.py](src/product_costs.py) | コスト定数の単一の真実 |
| [analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md) | 投信環境コスト・執行ラグ・P2/P7/P5 検証 |

---

## 「ベスト戦略は？」と問われたときの参照プロトコル (Claude 必読)

### 手順

1. **本ファイル (`CURRENT_BEST_STRATEGY.md`) の冒頭ブロックを引用する** — 最優先・最新の真実
2. `tasks.md` の最新 ✅ Completed エントリと突き合わせて整合確認
3. 矛盾があれば必ずユーザーに報告 → 本ファイル更新の提案

### 絶対にやってはいけないこと

- ❌ CSV を Sharpe 降順で並べて「トップ」を答える
- ❌ F7v3/F8 系 (高 Trades/yr) を「Sharpe が高いから」とベストとして提示する
- ❌ MEMORY.md 内の固定記述を一次根拠にする

---

## ⛔ 廃止された旧推奨 (ブラックリスト)

| 旧推奨 | 廃止日 | 廃止理由 |
|---|---|---|
| `F8 R5_CALM_BOOST` Sharpe +0.934, Trades/yr 182 | 2026-05-24 | Trades/yr 182回 (E4比7倍) のコスト負担と OOS 偶然性疑いによりShortlisted降格 |
| `F7v3+E4 A:tilt=2.0` Sharpe +0.926, Trades/yr 183 | 2026-05-24 | 同上。tilt系は全期間でE4との改善幅が軽微、IS-OOS gap拡大 |
| `S2_VZGated + LT2-N750-k0.5-modeB (固定k)` Sharpe 0.858 | 2026-05-24 | E4 が CAGR/Sharpe/Worst10Y★/IS-OOS gap で優位 |
| `S2_VZGated` CAGR_OOS +27.57%, Sharpe_OOS 0.769 | 2026-05-21 | S2+LT2 が全指標で上回ることを確認 |
| `DH Dyn 2x3x [A]` CAGR 22.50%, Sharpe 0.993 | 2026-05-21 | S2_VZGated が上回ることを確認 |
| `Ens2(Asym+Slope)` CAGR 28.58%, Sharpe 1.031 | 2026-04-21 | `DH Dyn 2x3x [A]` に置換 |

---

## 命名規則 (今後の再発防止)

1. **`FINAL_` プレフィックスは使用禁止**
2. **`<TOPIC>_YYYY-MM-DD.md` 形式を使用**
3. **新レポートが旧レポートを置き換える時は、必ず旧レポート冒頭に SUPERSEDED ヘッダを追加**
4. **本ファイル (`CURRENT_BEST_STRATEGY.md`) を必ず同時更新**

---

## メタ情報

変更履歴は git log で追跡可能 (`git log --follow CURRENT_BEST_STRATEGY.md`)

### 変更履歴
- **2026-06-07 (DH-W1 Cash-Sleeve 4戦略)**: ETF 環境 Active 候補 DH-W1 の OUT(キャッシュ 46.9%)期を 1 倍投信で運用置換する 4 戦略を **「投信環境 Active 候補」セクション** として §Shortlisted 直前に新設。**P2 GOLD100**(攻め, OOS +16.44% 最高)/**P7 GOLD75/BOND25 ⭐**(中庸推奨, MaxDD −48.23%)/**P5 GOLD50/BOND50**(守り, MaxDD −35.97% 最良)。全 WFA 50窓 α∩β PASS。検証済み: OUT 期資産挙動、全商品コスト表(SOFR/TER/スワップ/売買/税 20.315%)、執行ラグ(レバ脚 DELAY=2 / 投信スリーブ 5BD)。t_p/bootstrap 未実施で正式昇格保留。§1 本番 Active(E4 Regime k_lt)は変更なし。STRATEGY_REGISTRY §2 に 3 件登録。一次根拠: [CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md)。
- **2026-06-05 (v4.8: Session 4 + 5)**: `nasdaq_mom63 × S3 × M6 defensive` overlay を **Risk-Reduction Overlay Candidate** として §Shortlisted 直前に新設。Session 4 で S3 (DH-W1, ETF only) に対し Phase D 4 gate 全 PASS → ADOPT。Session 5 で S2 (D5) / E4 (現行 Active) への転用を audit-grade で検証 → 両方 NEEDS_FURTHER_WORK (WFE<1.0、ハードゲート不通過。P(MaxDD better)>0.94 で方向性は一貫)。結論: overlay は **S3 限定 (strategy-specific)**。§1 本番 Active (CFD: E4 Regime k_lt) は変更なし。
- **2026-06-05 (v4.7: CFD Active 候補を l7 → l5 に置換)**: ユーザー判断により vz=0.65+l5+F10ε を CFD 環境 Active 候補に確定。l7 (旧 REF) は副候補に降格。理由: l5 は min CAGR -1.30pp の trade-off と引換に **Worst10Y +12.67% (vs l7 +9.96%、+2.71pp)、P10_5Y +8.75% (vs l7 +4.05%、+4.70pp 大幅改善)、MaxDD -56.72% (vs l7 -65.95%、9.23pp 浅化)、Sharpe +0.841 (vs l7 +0.829)、Trades 86 (vs l7 105、18% 低コスト)** で防御指標が圧倒的優位。
- **2026-06-05 (v4.6: lmax sweep)**: vz=0.65+F10ε の lmax を l5/l5.5/l7 で比較、l5.5/l5 を Shortlisted 追加。
- **2026-06-05 (v4.5: 保守的採用基準 + 環境別 Active 候補導入)**: 本ファイル冒頭に **「v4.5 環境別 Active 候補」セクション** + **「命名規則」セクション** を新設。要点:
  - **min(IS, OOS) CAGR + Worst10Y + P10_5Y の 3 軸保守的尺度** を Active 昇格判断の必須条件として導入 (詳細: STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md §7-2)
  - **CFD 環境 Active 候補**: vz=0.65+l7+F10ε (min CAGR=+20.23%、5 戦略中 1 位)
  - **ETF 環境 Active 候補**: DH-W1 (DH Asymm+Hysteresis、ETF 制約下で DH 基線を +4.10pp 上回る唯一の改善)
  - 棄却: vz=0.65+l7+F10ε-AH/AT/HL (OOS のみ評価では魅力的だが min ルールで全敗、WFE>1.5 で regime luck 疑い)
  - 「vz=0.65+l7+F10ε」を「NEW/NEW CANDIDATE」と呼ぶ表記を廃止
  - §1 正式 Active は **未変更** (E4 RegimeKLT を維持)。CFD Active への昇格判断はユーザー承認後
- 2026-05-24 (tilt系棄却・E4 復帰): F7v3+E4 および F8-R5 を Trades/yr 過多（182〜183回/年、E4比7倍）・OOS偶然性疑い・IS-OOS gap拡大（−4.26〜4.28pp）を理由に Shortlisted 降格。E4 Regime k_lt (27回/年) を正式 Active に復帰。棄却判断: コスト加味の実質 CAGR は E4 と同等以下と判断。
- 2026-05-24 (G5 WFA + F8-R5 昇格→即降格): F8-R5 WFA PASS (CI95_lo=+27.92%) を確認するも上記理由で採用見送り。
- 2026-05-24 (G4 WFA + F7v3+E4 昇格→即降格): F7v3+E4 WFA PASS (CI95_lo=+27.15%) を確認するも上記理由で採用見送り。
- 2026-05-24 (G3 WFA): E4 Regime k_lt WFA PASS (CI95_lo=+26.51%, WFE=+1.131)。正式 Active 確定。
- 2026-05-24: E4 Regime k_lt 暫定昇格 (k_lo=0.1, k_hi=0.8, vz_thr=0.7)
- 2026-05-22: B6 N=1500 暫定昇格・即差し戻し
- 2026-05-21: B1 S2+LT2 採用
- 2026-05-12: Scenario D 補正適用
- 2026-05-11: 初版作成

---

*管理者: 男座員也（Kazuya Oza）*
