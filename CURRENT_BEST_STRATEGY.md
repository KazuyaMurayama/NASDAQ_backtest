# CURRENT BEST STRATEGY — 単一の真実 (Single Source of Truth)

> **このファイルは「現行のベスト戦略」を一意に特定するための正典です。**
> **「ベスト戦略は？」と問われた時、Claude / 人間ともにまずこのファイルだけを見れば良いように設計されています。**

作成日: 2026-05-11
最終更新日: 2026-05-21

---

## 現行ベスト戦略

**戦略名: `S2_VZGated`（Vol-Zone ゲート型 CFD レバレッジ、tv=0.8, k_vz=0.3, gate_min=0.5）**

### 主要指標 (Scenario D コスト補正済み基盤, 1974-01-02 〜 2026-03-26, 52.26年)

> **注意:** DH Dyn 2x3x [A] シグナル (Approach A, 閾値 0.15) の上に CFD Vol-Zone ゲートを適用したもの。
> CFD 軸の指標は `cfd_leverage_backtest.py` + `compute_cfd_worst10y.py` の出力値。

| 指標 | 値 | 備考 |
|---|---|---|
| CAGR_IS (1974-2021) | **+32.94%** | In-sample |
| CAGR_OOS (2021-2026) | **+27.57%** | Out-of-sample ← 10戦略中最高 |
| Worst5Y CAGR (FULL) | −4.75% | 日次ローリング最悪窓 |
| P10_5Y CAGR (FULL) | **+7.32%** | 下位10%パーセンタイル5年CAGR |
| Worst10Y★ CAGR (FULL) | **+17.74%** | カレンダー年ベース最悪10年窓 |
| MaxDD (FULL) | −62.4% | 最大ドローダウン |
| Sharpe_OOS | **0.769** | OOS期間Sharpe比 |
| 年間取引回数 | 約27回 (月2.3回) | 基底DH Dynシグナルと同じ |

### ベスト戦略採用の根拠 (2026-05-21)

| 評価軸 | S2_VZGated | 旧ベスト DH Dyn [A] Scenario D | 優位 |
|---|---|---|---|
| CAGR_OOS | **+27.57%** | +14.88% | S2 ◎ |
| Sharpe_OOS | **0.769** | 0.646 | S2 ◎ |
| Worst10Y★ | **+17.74%** | +14.30% | S2 ◎ |
| P10_5Y | +7.32% | +9.57% | DH Dyn ○ |
| Worst5Y | −4.75% | +0.87% | DH Dyn ○ |
| MaxDD | −62.4% | −45.08% | DH Dyn ○ |

> OOS CAGR と Sharpe_OOS で圧倒的優位。ドローダウン耐性は DH Dyn [A] の方が高いため、リスク許容度に応じた使い分けが可能。

---

## 構成

- **シグナル基盤**: DH Dyn A2 Optimized (DD × AsymEWMA × TrendTV × SlopeMult × MomDecel × VIX_MR), threshold=0.15
- **CFD レバレッジ**: `compute_L_s2_vz_gated(ret, vz, target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)`
- **CFD スプレッド**: 低スプレッド想定 (`CFD_SPREAD_LOW`)
- **配分**: `wn·lev_CFD·r_nas_cfd + wg·r_g2 + wb·r_b3`
  - `wn`, `wg`, `wb`: DH Dyn [A] Approach A と同一 (`simulate_rebalance_A`)
  - NAS スリーブのみ TQQQ → CFD に変更
- **Gold 2x**: TER 0.50% (sim proxy) + 1×SOFR（UGL実費 0.95%、差=−10.5 bps/yr, §16参照）
- **Bond 3x (TMF)**: TER 0.91% + 2×SOFR
- **DELAY**: 2営業日
- **実装**: `src/cfd_leverage_backtest.py`, `src/dynamic_leverage_strategies.py`

---

## コスト注意事項

| 補正項目 | 影響 |
|---|---|
| SOFR financing drag (NAS CFD + Gold 1xSOFR) | CFD 軸にも適用 |
| Gold TER ギャップ (proxy 0.50% → UGL 0.95%) | **−10.5 bps/yr** (§16) |
| TMF TER ギャップ (0.91% → 1.06%) | −3.5 bps/yr (§16) |
| スワップスプレッド推定差 (+20.5 bps) | −34 bps/yr 相当 (§16) |
| 合計推定コスト過少計上 | **約 −66 bps/yr** → 現実 CAGR_OOS ≈ 26.9% 相当 |
| 売買税ドラッグ (年27回、税率20.315%) | 別途 −2〜5% CAGR |
| NISA | CFD は原則 NISA 不適用 |

---

## 一次根拠ファイル（GitHub 上の正典）

| ファイル | 役割 | 日付 |
|---|---|---|
| [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) | 10戦略統合比較表 (§2 ◆BEST マーク) | 2026-05-21 |
| [src/cfd_leverage_backtest.py](src/cfd_leverage_backtest.py) | S2_VZGated NAV 実装 | - |
| [src/dynamic_leverage_strategies.py](src/dynamic_leverage_strategies.py) | `compute_L_s2_vz_gated` 定義 | - |
| [src/corrected_strategy_backtest.py](src/corrected_strategy_backtest.py) | DH Dyn シグナル基盤 (Scenario D) | 2026-05-12 |
| [src/product_costs.py](src/product_costs.py) | コスト定数の単一の真実 | 2026-05-12 |

---

## 「ベスト戦略は？」と問われたときの参照プロトコル (Claude 必読)

### 手順 (上から順に実行・1で答えが出たらそこで止める)

1. **本ファイル (`CURRENT_BEST_STRATEGY.md`) の冒頭ブロックを引用する** — これが最優先・最新の真実
2. `tasks.md` の最新 ✅ Completed エントリと突き合わせて整合確認
3. 矛盾があれば必ずユーザーに報告 → 本ファイル更新の提案

### 絶対にやってはいけないこと

- ❌ `R4_results.csv` を Sharpe 降順で並べて「トップ」を答える (CSV は実験ログであり結論ではない)
- ❌ `FINAL_RESULTS_2026-02-06.md` の冒頭テーブルを「最終」として答える (廃止済み)
- ❌ ファイル名に `FINAL` が含まれていることを理由に最新と判断する
- ❌ MEMORY.md / nasdaq_best_strategy.md 内の固定記述を一次根拠にする (キャッシュであり一次情報ではない)

---

## ⛔ 廃止された旧推奨 (ブラックリスト)

これらが「ベスト」と答えられたら誤回答です。質問されても **過去の研究履歴として** のみ言及し、現行ベストとして提示しない:

| 旧推奨 | 廃止日 | 廃止理由 |
|---|---|---|
| `DH Dyn 2x3x [A]` CAGR 22.50%, Sharpe 0.993 | 2026-05-21 | S2_VZGated が CAGR_OOS +27.57% / Sharpe_OOS 0.769 で上回ることを確認 |
| `DH Dyn 2x3x [A]` CAGR 30.81% | 2026-05-12 | Scenario D 補正適用で CAGR 22.50% に更新 |
| `Ens2(Asym+Slope)` (CAGR 28.58%, Sharpe 1.031) | 2026-04-21 | `DH Dyn 2x3x [A] 閾値0.15` に置換 |
| `Ens2(Slope+TrendTV)` | 2026-04-21 | 同上 |
| `DD+VT+VolSpike(1.5x)` | 2026-04-21 | 個別実験結果。最終結論ではない |
| `DD Dyn 2x3x [A] 閾値0.20` | 2026-04-21 | 閾値0.15が優位と確認 |

---

## 命名規則 (今後の再発防止)

新規レポート作成時:

1. **`FINAL_` プレフィックスは使用禁止**
2. **`REPORT_YYYY-MM-DD` 形式または `<TOPIC>_YYYY-MM-DD.md` 形式を使用**
3. **新レポートが旧レポートを置き換える時は、必ず旧レポート冒頭に SUPERSEDED ヘッダを追加**
4. **本ファイル (`CURRENT_BEST_STRATEGY.md`) を必ず同時更新**
5. **`tasks.md` の Completed セクション末尾に1行追記**

### SUPERSEDED ヘッダのテンプレート

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
- 2026-05-21: ベスト戦略を `DH Dyn 2x3x [A]` から `S2_VZGated` に更新。根拠: 10戦略統合比較表で CAGR_OOS +27.57% / Sharpe_OOS 0.769 が全戦略中トップ。旧ベストを廃止リストに移動。
- 2026-05-12: Scenario D 補正適用。CAGR 30.84% → 22.50%, Sharpe 1.299 → 0.993, MaxDD -31.40% → -45.08%。
- 2026-05-11: 初版作成。`DH Dyn 2x3x [A] 閾値0.15` を現行ベストに確定。

---

*管理者: Kazuya Murayama*
