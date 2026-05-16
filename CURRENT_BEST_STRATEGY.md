# CURRENT BEST STRATEGY — 単一の真実 (Single Source of Truth)

> **このファイルは「現行のベスト戦略」を一意に特定するための正典です。**
> **「ベスト戦略は？」と問われた時、Claude / 人間ともにまずこのファイルだけを見れば良いように設計されています。**

作成日: 2026-05-11
最終更新日: 2026-05-16

> 📌 **CFD動的レバレッジ軸の推奨は別管理**: [CFD_DYNAMIC_LEVERAGE_GUIDE.md](CFD_DYNAMIC_LEVERAGE_GUIDE.md) 参照（S2_VZGated, 2026-05-16採用確定）

---

## 現行ベスト戦略

**戦略名: `DH Dyn 2x3x [A]`（Approach A・スリーブ独立型、リバランス閾値 0.15）**

### 主要指標 — Scenario D 補正済みベースライン (1974-01-02 〜 2026-03-26, 52.26年)

> **注意:** 下記は `corrected_strategy_backtest.py` Scenario D の出力値。
> コスト補正（SOFR financing + bond model + duration変動性）をすべて適用した最も現実的な推計。

| 指標 | 値 |
|---|---|
| CAGR (FULL) | **+22.50%** |
| Sharpe (FULL) | **0.993** |
| MaxDD (FULL) | **-45.08%** |
| Worst5Y CAGR (FULL) | +0.87% |
| WinRate (FULL) | 83.0% |
| 年間取引回数 | 約27回 (月2.3回) |

### 未補正ベースライン（参考: Scenario A）

| 指標 | 値 |
|---|---|
| CAGR (FULL) | +30.84% |
| Sharpe (FULL) | 1.299 |
| MaxDD (FULL) | -31.40% |

---

## コスト補正内訳

| 補正項目 | CAGR インパクト |
|---|---|
| SOFR financing drag（TQQQ+TMF 2xSOFR + Gold 1xSOFR） | **-8.13% CAGR**（主要因） |
| Bond model補正（dgs10+dur7 → dgs30+dur17+splice_fix） | +0.38% CAGR |
| Duration変動性補正（static D=17 → yield依存 Dmod） | -0.59% CAGR |
| **合計補正** | **-8.34% CAGR**（対 未補正ベースライン 30.84%） |

---

## 未モデル化コスト（実運用ではさらに低下）

- **売買税ドラッグ**: -2.8% to -5.2% CAGR（年27回リバランス、日本税率20.315%）
- **NISA**: TQQQ・TMF・Gold 2x いずれも新NISA不適用（3倍レバレッジ禁止）
- **Gold商品乖離**: UGL TER 0.95% vs シミュProxy 0.49%（差0.46%/yr）
- コスト定数の単一の真実: `src/product_costs.py`

---

## 構成

- シグナル: A2 Optimized (DD × AsymEWMA × TrendTV × SlopeMult × MomDecel × VIX_MR)
- 配分: `daily = wn·lev·(r_nas·3 − dc) + wg·r_g2 + wb·r_b3`
- 重み: `wn = clip(0.55 + 0.25·lev − 0.10·max(vz,0), 0.30, 0.90)`, `wg = wb = (1−wn)·0.5`
- 経費率: TQQQ 0.86% / Gold2x 0.50% (sim proxy) / Bond3x 0.91%
- SOFR financing: TQQQ 2xSOFR / TMF 2xSOFR / Gold2x 1xSOFR（v2補正済み）
- DELAY: 2営業日

### 一次根拠ファイル（GitHub 上の正典）

| ファイル | 役割 | 日付 |
|---|---|---|
| [THRESHOLD_SWEEP_A_REPORT_2026-04-21.md](THRESHOLD_SWEEP_A_REPORT_2026-04-21.md) | 閾値 0.15 採用根拠 (ダブルチェック PASS 3/3) | 2026-04-21 |
| [YEARLY_RETURNS_REPORT_2026-04-20_v3.md](YEARLY_RETURNS_REPORT_2026-04-20_v3.md) | 51年分の年次比較 + BRK ベンチマーク | 2026-04-21 |
| [APPROACH_A_PROPOSAL_2026-04-20.md](APPROACH_A_PROPOSAL_2026-04-20.md) | Approach A 切替の設計書 | 2026-04-20 |
| [src/corrected_strategy_backtest.py](src/corrected_strategy_backtest.py) | Scenario D 実装（SOFR+bond+duration+Gold補正） | 2026-05-12 |
| [src/product_costs.py](src/product_costs.py) | コスト定数の単一の真実 | 2026-05-12 |

### 実運用
- Googleスプレッドシート ID: `1YqwZ2EGKVFs36tTvUfup28g0GtXZwReiVOBI6eNzmVI`（タイトル: Dyn2x3x戦略）
- 実装リポジトリ: https://github.com/KazuyaMurayama/nasdaq-strategy-gas

---

## 「ベスト戦略は？」と問われたときの参照プロトコル (Claude 必読)

### 手順 (上から順に実行・1で答えが出たらそこで止める)

1. **本ファイル (`CURRENT_BEST_STRATEGY.md`) の冒頭ブロックを引用する** — これが最優先・最新の真実
2. `tasks.md` の最新 ✅ Completed エントリと突き合わせて整合確認
3. 矛盾があれば必ずユーザーに報告 → 本ファイル更新の提案

### 絶対にやってはいけないこと

- ❌ `R4_results.csv` を Sharpe 降順で並べて「トップ」を答える (CSV は実験ログであり結論ではない)
- ❌ `FINAL_RESULTS_2026-02-06.md` の冒頭テーブルを「最終」として答える (2026-02-06 時点のスナップショット・廃止済み)
- ❌ ファイル名に `FINAL` が含まれていることを理由に最新と判断する (このリポジトリでは `FINAL_` プレフィックスは廃止)
- ❌ MEMORY.md / nasdaq_best_strategy.md 内の固定記述を一次根拠にする (キャッシュであり一次情報ではない)

---

## ⛔ 廃止された旧推奨 (ブラックリスト)

これらが「ベスト」と答えられたら誤回答です。質問されても **過去の研究履歴として** のみ言及し、現行ベストとして提示しない:

| 旧推奨 | 出典ファイル | 廃止理由 |
|---|---|---|
| `DH Dyn 2x3x [A]` CAGR 30.81% | 2026-05-11 版 `CURRENT_BEST_STRATEGY.md` | 2026-05-12 Scenario D 補正適用（Gold SOFR fix）で CAGR 22.50% に更新 |
| `Ens2(Asym+Slope)` (CAGR 28.58%, Sharpe 1.031) | `FINAL_RESULTS_2026-02-06.md` | 2026-04-21 で `DH Dyn 2x3x [A] 閾値0.15` に置換 |
| `Ens2(Slope+TrendTV)` | `FINAL_RESULTS_2026-02-06.md` | 同上 |
| `DD+VT+VolSpike(1.5x)` | `R4_results.csv`, `R4_RESULTS_SUMMARY_2026-02-06.md` | 個別実験結果。最終結論ではない |
| `MomDecel(40/120)+Ens2(S+T)` | GAS 旧版表記 | Approach A 確立で再定義済 |
| `DD Dyn 2x3x [A] 閾値0.20` | `THRESHOLD_SWEEP_REPORT_2026-04-20.md` (古い) | 2026-04-21 で閾値0.15が優位と確認、`THRESHOLD_SWEEP_A_REPORT_2026-04-21.md` に置換 |

---

## 命名規則 (今後の再発防止)

新規レポート作成時:

1. **`FINAL_` プレフィックスは使用禁止** — 「FINAL」と名乗ったファイルが後で覆されると参照地獄になる
2. **`REPORT_YYYY-MM-DD` 形式または `<TOPIC>_YYYY-MM-DD.md` 形式を使用**
3. **新レポートが旧レポートを置き換える時は、必ず旧レポート冒頭に SUPERSEDED ヘッダを追加** (本ファイル末尾のテンプレ参照)
4. **本ファイル (`CURRENT_BEST_STRATEGY.md`) を必ず同時更新**
5. **`tasks.md` の Completed セクション末尾に1行追記**

### SUPERSEDED ヘッダのテンプレート

旧レポートの H1 直下に以下を挿入する:

```markdown
> ⛔ **このドキュメントは SUPERSEDED (置換済み) です**
> - 廃止日: YYYY-MM-DD
> - 後継ファイル: [新レポート名](新レポート.md)
> - 現行ベスト戦略: [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)
> - 廃止理由: <一行で理由>
```

---

## メタ情報

- このファイルは [tasks.md](tasks.md) と [FILE_INDEX.md](FILE_INDEX.md) で「最優先参照ファイル」として登録されています
- Claude のローカル memory (`~/.claude/projects/.../memory/nasdaq_best_strategy.md`) もこのファイルを一次根拠としています
- 変更履歴は git log で追跡可能 (`git log --follow CURRENT_BEST_STRATEGY.md`)

### 変更履歴
- 2026-05-12: Scenario D 補正適用。Gold 2x SOFR financing (1xSOFR) を追加。CAGR 30.84% → 22.50%, Sharpe 1.299 → 0.993, MaxDD -31.40% → -45.08%。`src/product_costs.py` をコスト定数の単一の真実として追加。
- 2026-05-11: 初版作成。`THRESHOLD_SWEEP_A_REPORT_2026-04-21.md` を一次根拠として `DH Dyn 2x3x [A] 閾値0.15` を現行ベストに確定。

---

*管理者: 男座員也（Kazuya Oza）*
