# CURRENT BEST STRATEGY — 単一の真実 (Single Source of Truth)

> **このファイルは「現行のベスト戦略」を一意に特定するための正典です。**
> **「ベスト戦略は？」と問われた時、Claude / 人間ともにまずこのファイルだけを見れば良いように設計されています。**

作成日: 2026-05-11
最終更新日: 2026-05-24

---

## 現行ベスト戦略

**戦略名: `S2_VZGated + LT2-N750 + E4 Regime k_lt + F8 R5_CALM_BOOST`（Vol-Zone ゲート CFD + 長期逆張り + ボラレジーム動的 k_lt + レジーム別 cap bull-tilt）**

### 主要指標 (Scenario D コスト補正済み基盤, 1974-01-02 〜 2026-03-26, 52.26年)

> **注意:** DH Dyn 2x3x [A] シグナル (Approach A, 閾値 0.15) の上に CFD Vol-Zone ゲートを適用し、
> さらに LT2-N=750 を **vz レジーム条件付き k_lt** (k_lo=0.1 / k_mid=0.5 / k_hi=0.8, vz_thr=0.7) で lev シグナルに重畳、
> **F8 R5_CALM_BOOST Bull-Tilt** (tilt=10.0 step-func, レジーム別 cap: calm=0.15 / bull-VZ=0.10 / bear-VZ=0.05) で強気日の wn を動的傾斜させたもの。

| 指標 | 値 | 備考 |
|---|---|---|
| CAGR_OOS (2021-2026) | **+36.83%** | Out-of-sample |
| Sharpe_OOS | **+0.934** | OOS期間Sharpe比 |
| MaxDD (FULL) | **−63.07%** | 最大ドローダウン |
| Worst10Y★ CAGR (FULL) | **+18.58%** | カレンダー年ベース最悪10年窓 |
| P10_5Y▷ CAGR (FULL) | **+10.27%** | 5年CAGR 分布 P10 |
| IS-OOS gap | **−4.28 pp** | OOS が IS を上回る（優秀な汎化性） |
| Trades/yr | 約182回 (月15.2回) | bull-tilt日次判定によるリバランス |
| CAGR_IS (1974-2021) | +32.54% | In-sample |
| WFA_CI95_lo | **+27.92%** | G5 WFA 50窓、α PASS（t_p=0.0000） |
| WFA_WFE | **+1.208** | G5 WFA、β PASS（0.5 ≤ WFE ≤ 2.0）|

### ベスト戦略選定根拠 (2026-05-24 確定)

| 評価軸 | F8-R5 **◆ BEST** | F7v3+E4 (旧◆) | 優位 |
|---|---|---|---|
| CAGR_OOS | **+36.83%** | +36.30% | **F8-R5 +0.53pp** |
| Sharpe_OOS | **+0.934** | +0.926 | **F8-R5 +0.008** |
| MaxDD | −63.07% | **−61.96%** | F7v3 (−1.11pp 軽微、guardrail −65.01% 内) |
| Worst10Y★ | **+18.58%** | +18.27% | **F8-R5 +0.31pp** |
| P10_5Y▷ | **+10.27%** | +9.84% | **F8-R5 +0.43pp** |
| IS-OOS gap | −4.28pp | **−4.26pp** | ほぼ同等（両者とも OOS > IS の優秀域） |
| WFA_CI95_lo | **+27.92%** | +27.15% | **F8-R5 +0.77pp** |

> **F8-R5 採用理由**: Sharpe_OOS +0.934（PASS基準+0.023超過）、CAGR_OOS +36.83%、WFA CI95_lo +27.92%（F7v3+E4比+0.77pp）。
> 5指標（CAGR/Sharpe/Worst10Y★/P10_5Y/WFA CI95_lo）で F7v3+E4 を上回る。
> MaxDD −1.11pp 劣後は guardrail（−65.01%）内の軽微な差。
> **G5 WFA (2026-05-24) PASS**: CI95_lo = +27.92%（>0 α PASS, t_p=0.0000）/ WFE = +1.208（0.5–2.0 β PASS）→ **正式 Active 確定**。
> R5_CALM_BOOST の優位性: calm レジーム時に cap を 0.10 → 0.15 に拡大することで、VZ 安定期のリターン取り増しが WFA でも統計的に確認された。

---

## 次候補 (Shortlisted / WFA 完了)

| 戦略 | Sharpe_OOS | CAGR_OOS | MaxDD | WFA CI95_lo | 状態 |
|---|---|---|---|---|---|
| F7v3+E4 A:tilt=2.0 (flat cap=0.10) | +0.926 | +36.30% | −61.96% | +27.15% | WFA PASS (G4), MaxDD 最良 fallback |
| E4 Regime k_lt (tiltなし) | +0.891 | +33.53% | −60.01% | +26.51% | WFA PASS (G3), 保守的 fallback |

---

## 構成

- **シグナル基盤**: DH Dyn A2 Optimized (DD × AsymEWMA × TrendTV × SlopeMult × MomDecel × VIX_MR), threshold=0.15
- **LT2 オーバーレイ (modeB, regime-conditional k_lt)**:
  - `lt_sig = compute_lt2(close, N=750)` — 750日（≈3年）モメンタム z スコア
  - `k_lt_t = 0.8 if vz_t > +0.7; 0.1 if vz_t < −0.7; 0.5 otherwise` — vz レジーム依存感度
  - `lt_bias_t = (−k_lt_t × lt_sig_t × 0.5).clip(−0.5, +0.5)` — 動的バイアス
  - `lev_mod = clip(lev_A + lt_bias, 0, 1)` — DH Dyn lev に動的バイアス
- **Bull-Tilt オーバーレイ (F8 R5_CALM_BOOST, tilt=10.0 → step function)**:
  - `bull_mask = raw_a2 > 0.15` — A2 シグナルが強気閾値超
  - `tilt_raw = 10.0 × (raw_a2 − 0.15) × (1.0 − raw_a2)` → ほぼ完全ステップ関数（cap で飽和）
  - `cap_eff = 0.15 if |vz|<0.7 (calm regime) ; 0.10 if vz>+0.7 (bull-VZ) ; 0.05 if vz<−0.7 (bear-VZ)`
  - `tilt_amount = clip(tilt_raw, 0, cap_eff)` if bull_mask else 0
  - `wn_tilted = wn_A + tilt_amount` — NASDAQ 比率 +最大 15% (calm) / +10% (bull-VZ) / +5% (bear-VZ)
  - `wb_tilted = max(wb_A − tilt_amount, 0)` — Bond 比率 相応削減
  - `wg_tilted = wg_A` — Gold 変更なし
- **CFD レバレッジ**: `compute_L_s2_vz_gated(ret, vz, target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)`
- **CFD スプレッド**: 低スプレッド想定 (`CFD_SPREAD_LOW`)
- **配分**: `wn_tilted·lev_mod·L_s2·r_nas_cfd + wg_tilted·r_g2 + wb_tilted·r_b3`
- **Gold 2x**: TER 0.50% (sim proxy) + 1×SOFR
- **Bond 3x (TMF)**: TER 0.91% + 2×SOFR
- **DELAY**: 2営業日
- **実装**: `src/f8_regime_tilt.py`（F8-R5 主実装）, `src/f7v3_bull_tilt.py`（前版参照）, `src/e4_regime_klt.py`, `src/long_cycle_signal.py`

---

## コスト注意事項

| 補正項目 | 影響 |
|---|---|
| SOFR financing drag (NAS CFD + Gold 1xSOFR) | CFD 軸にも適用 |
| Gold TER ギャップ (proxy 0.50% → UGL 0.95%) | **−10.5 bps/yr** (§16) |
| TMF TER ギャップ (0.91% → 1.06%) | −3.5 bps/yr (§16) |
| スワップスプレッド推定差 (+20.5 bps) | −34 bps/yr 相当 (§16) |
| 合計推定コスト過少計上 | **約 −66 bps/yr** → 現実 CAGR_OOS ≈ 34.2% 相当 |
| 売買税ドラッグ (年182回、税率20.315%) | 別途 −5〜10% CAGR（頻繁売買コスト増に注意） |
| NISA | CFD は原則 NISA 不適用 |

---

## 一次根拠ファイル（GitHub 上の正典）

| ファイル | 役割 | 日付 |
|---|---|---|
| [F8_REGIME_TILT_2026-05-24.md](F8_REGIME_TILT_2026-05-24.md) | F8 regime-tilt sweep レポート（R5_CALM_BOOST採用根拠） | 2026-05-24 |
| [f8_regime_tilt_results.csv](f8_regime_tilt_results.csv) | F8 sweep 8行 raw 結果 | 2026-05-24 |
| [src/f8_regime_tilt.py](src/f8_regime_tilt.py) | F8 R5_CALM_BOOST 実装 | 2026-05-24 |
| [G5_WFA_F8R5_2026-05-24.md](G5_WFA_F8R5_2026-05-24.md) | G5 WFA レポート（CI95_lo=+27.92% / WFE=+1.208 / PASS） | 2026-05-24 |
| [g5_wfa_f8r5_summary.csv](g5_wfa_f8r5_summary.csv) | G5 WFA サマリ（F8-R5 + REF-F7v3A2-sanity） | 2026-05-24 |
| [g5_wfa_f8r5_per_window.csv](g5_wfa_f8r5_per_window.csv) | G5 WFA 50窓 × 2戦略 per-window 指標 | 2026-05-24 |
| [src/g5_wfa_f8r5.py](src/g5_wfa_f8r5.py) | G5 WFA 実行スクリプト | 2026-05-24 |
| [F7V3_BULL_TILT_2026-05-24.md](F7V3_BULL_TILT_2026-05-24.md) | F7-v3 sweep レポート (前版 Active 根拠) | 2026-05-24 |
| [G4_WFA_F7V3_2026-05-24.md](G4_WFA_F7V3_2026-05-24.md) | G4 WFA レポート（CI95_lo=+27.15% / WFE=+1.203 / PASS） | 2026-05-24 |
| [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md) | E4 sweep レポート | 2026-05-24 |
| [G3_WFA_E4_2026-05-24.md](G3_WFA_E4_2026-05-24.md) | G3 WFA レポート（E4 WFA PASS） | 2026-05-24 |
| [B1_S2_LT2_2026-05-21.md](B1_S2_LT2_2026-05-21.md) | B1 検証レポート | 2026-05-21 |
| [src/corrected_strategy_backtest.py](src/corrected_strategy_backtest.py) | DH Dyn シグナル基盤 (Scenario D) | 2026-05-12 |
| [src/product_costs.py](src/product_costs.py) | コスト定数の単一の真実 | 2026-05-12 |

---

## 「ベスト戦略は？」と問われたときの参照プロトコル (Claude 必読)

### 手順 (上から順に実行・1で答えが出たらそこで止める)

1. **本ファイル (`CURRENT_BEST_STRATEGY.md`) の冒頭ブロックを引用する** — これが最優先・最新の真実
2. `tasks.md` の最新 ✅ Completed エントリと突き合わせて整合確認
3. 矛盾があれば必ずユーザーに報告 → 本ファイル更新の提案

### 絶対にやってはいけないこと

- ❌ CSV を Sharpe 降順で並べて「トップ」を答える (CSV は実験ログであり結論ではない)
- ❌ `FINAL_RESULTS_2026-02-06.md` の冒頭テーブルを「最終」として答える (廃止済み)
- ❌ ファイル名に `FINAL` が含まれていることを理由に最新と判断する
- ❌ MEMORY.md / nasdaq_best_strategy.md 内の固定記述を一次根拠にする (キャッシュであり一次情報ではない)

---

## ⛔ 廃止された旧推奨 (ブラックリスト)

これらが「ベスト」と答えられたら誤回答です。質問されても **過去の研究履歴として** のみ言及し、現行ベストとして提示しない:

| 旧推奨 | 廃止日 | 廃止理由 |
|---|---|---|
| `F7v3+E4 A:tilt=2.0 (flat cap=0.10)` Sharpe_OOS +0.926 | 2026-05-24 | F8-R5 が Sharpe +0.934 / CAGR +36.83% / WFA CI95_lo +27.92% で5指標優位。G5 WFA PASS確認。MaxDD差-1.11ppはguardrail内。 |
| `S2_VZGated + LT2-N750 + E4 Regime k_lt` (tiltなし) Sharpe_OOS +0.891 | 2026-05-24 | F7v3 Bull-Tilt が Sharpe +0.926 / CAGR +36.30% / WFA CI95_lo +27.15% で全主要指標優位。G4 WFA PASS確認済。 |
| `S2_VZGated + LT2-N750-k0.5-modeB (固定k)` CAGR_OOS +31.16%, Sharpe_OOS 0.858 | 2026-05-24 | E4 Regime k_lt が CAGR_OOS +33.53% / Sharpe_OOS +0.891 で上回ることを確認。 |
| `S2_VZGated` CAGR_OOS +27.57%, Sharpe_OOS 0.769 | 2026-05-21 | S2+LT2 が全指標上回ることを確認 |
| `DH Dyn 2x3x [A]` CAGR 22.50%, Sharpe 0.993 | 2026-05-21 | S2_VZGated が上回ることを確認 |
| `DH Dyn 2x3x [A]` CAGR 30.81% | 2026-05-12 | Scenario D 補正適用で CAGR 22.50% に更新 |
| `Ens2(Asym+Slope)` (CAGR 28.58%, Sharpe 1.031) | 2026-04-21 | `DH Dyn 2x3x [A] 閾値0.15` に置換 |

---

## 命名規則 (今後の再発防止)

新規レポート作成時:

1. **`FINAL_` プレフィックスは使用禁止**
2. **`REPORT_YYYY-MM-DD` 形式または `<TOPIC>_YYYY-MM-DD.md` 形式を使用**
3. **新レポートが旧レポートを置き換える時は、必ず旧レポート冒頭に SUPERSEDED ヘッダを追加**
4. **本ファイル (`CURRENT_BEST_STRATEGY.md`) を必ず同時更新**
5. **`tasks.md` の Completed セクション末尾に1行追記**

---

## メタ情報

- このファイルは [tasks.md](tasks.md) と [FILE_INDEX.md](FILE_INDEX.md) で「最優先参照ファイル」として登録されています
- Claude のローカル memory (`~/.claude/projects/.../memory/MEMORY.md`) もこのファイルを一次根拠としています
- 変更履歴は git log で追跡可能 (`git log --follow CURRENT_BEST_STRATEGY.md`)

### 変更履歴
- 2026-05-24 (G5 WFA + F8-R5 正式昇格): F8 R5_CALM_BOOST の G5 WFA PASS (CI95_lo=+27.92%, WFE=+1.208, t_p=0.0000, α∩β)。CAGR_OOS=+36.83%, Sharpe_OOS=+0.934, IS-OOS gap=-4.28pp。旧 F7v3+E4 (Sharpe=+0.926) を廃止リストに移動。**正式 Active 確定**。詳細: `G5_WFA_F8R5_2026-05-24.md`, `F8_REGIME_TILT_2026-05-24.md`
- 2026-05-24 (G4 WFA + F7v3+E4 昇格): F7v3 A:tilt=2.0 G4 WFA PASS (CI95_lo=+27.15%, WFE=+1.203)。CAGR_OOS=+36.30%, Sharpe_OOS=+0.926。旧 E4 単体 (Sharpe=+0.891) を廃止リストに移動。(現在は F8-R5 昇格により廃止リスト入り)
- 2026-05-24 (G3 WFA): E4 Regime k_lt WFA PASS (CI95_lo=+26.51%, WFE=+1.131)。(現在は F7v3昇格により廃止リスト入り)
- 2026-05-24: E4 Regime k_lt 暫定昇格。IS-OOS gap −1.81pp は当時プロジェクト最高の汎化性。
- 2026-05-22〜2026-05-21: B6/B1 LT2 シリーズ各種更新 (省略 – git log 参照)
- 2026-05-12: Scenario D 補正適用
- 2026-05-11: 初版作成

---

*管理者: Kazuya Murayama*
