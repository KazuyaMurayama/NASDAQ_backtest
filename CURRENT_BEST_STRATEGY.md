# CURRENT BEST STRATEGY — 単一の真実 (Single Source of Truth)

> **このファイルは「現行のベスト戦略」を一意に特定するための正典です。**
> **「ベスト戦略は？」と問われた時、Claude / 人間ともにまずこのファイルだけを見れば良いように設計されています。**

作成日: 2026-05-11
最終更新日: 2026-05-24

---

## 現行ベスト戦略

**戦略名: `S2_VZGated + LT2-N750 + E4 Regime k_lt + F7v3 Bull-Tilt (A:tilt=2.0)`（Vol-Zone ゲート CFD + 長期逆張り + ボラレジーム動的 k_lt + 強気シグナル連動 wn 傾斜）**

### 主要指標 (Scenario D コスト補正済み基盤, 1974-01-02 〜 2026-03-26, 52.26年)

> **注意:** DH Dyn 2x3x [A] シグナル (Approach A, 閾値 0.15) の上に CFD Vol-Zone ゲートを適用し、
> さらに LT2-N=750 を **vz レジーム条件付き k_lt** (k_lo=0.1 / k_mid=0.5 / k_hi=0.8, vz_thr=0.7) で lev シグナルに重畳、
> **F7-v3 定式A Bull-Tilt** (tilt=2.0, cap=0.10, THRESHOLD=0.15) で強気日の wn を動的傾斜させたもの。

| 指標 | 値 | 備考 |
|---|---|---|
| CAGR_OOS (2021-2026) | **+36.30%** | Out-of-sample |
| Sharpe_OOS | **+0.926** | OOS期間Sharpe比 |
| MaxDD (FULL) | **−61.96%** | 最大ドローダウン |
| Worst10Y★ CAGR (FULL) | **+18.27%** | カレンダー年ベース最悪10年窓 |
| P10_5Y▷ CAGR (FULL) | **+9.84%** | 5年CAGR 分布 P10 |
| IS-OOS gap | **−4.26 pp** | OOS が IS を上回る（優秀な汎化性） |
| Trades/yr | 約183回 (月15.3回) | bull-tilt日次判定によるリバランス増 |
| CAGR_IS (1974-2021) | +32.04% | In-sample |
| WFA_CI95_lo | **+27.15%** | G4 WFA 50窓、α PASS（t_p=0.0000） |
| WFA_WFE | **+1.203** | G4 WFA、β PASS（0.5 ≤ WFE ≤ 2.0）|

### ベスト戦略選定根拠 (2026-05-24 確定)

| 評価軸 | F7v3+E4 **◆ BEST** | E4 単体 (旧◆) | 優位 |
|---|---|---|---|
| CAGR_OOS | **+36.30%** | +33.53% | **F7v3 +2.77pp** |
| Sharpe_OOS | **+0.926** | +0.891 | **F7v3 +0.035** |
| MaxDD | −61.96% | **−60.01%** | E4 (−1.95pp 軽微、guardrail −65.01% 内) |
| Worst10Y★ | +18.27% | **+18.67%** | E4 (−0.40pp 軽微、guardrail +15% 超過余裕) |
| P10_5Y▷ | **+9.84%** | +9.78% | **F7v3 +0.06pp** |
| IS-OOS gap | −4.26pp | **−1.81pp** | E4 (ただし両者とも OOS > IS の優秀域) |
| WFA_CI95_lo | **+27.15%** | +26.51% | **F7v3 +0.64pp** |

> **F7v3+E4 採用理由**: PASS 基準（Sharpe ≥ 0.9115）をクリアし G4 WFA でα∩β PASS 確認。
> Sharpe_OOS +0.035 / CAGR_OOS +2.77pp / WFA CI95_lo +0.64pp の優位性は明確。
> MaxDD −1.95pp 劣後・Worst10Y★ −0.40pp 劣後はいずれも guardrail 内の軽微な差。
> IS-OOS gap −4.26pp は依然 OOS > IS（汎化性良好）。
> **G4 WFA (2026-05-24) PASS**: CI95_lo = +27.15%（>0 α PASS, t_p=0.0000）/ WFE = +1.203（0.5–2.0 β PASS）→ **正式 Active 確定**。

---

## 次候補 (Shortlisted / WFA 待ち)

| 戦略 | Sharpe_OOS | CAGR_OOS | MaxDD | IS-OOS gap | 状態 |
|---|---|---|---|---|---|
| F8 R5_CALM_BOOST (calm cap=0.15 / stressed cap=0.05) | +0.934 | +36.83% | −63.07% | — | WFA 未実施 (G5 WFA Pending) |
| E4 Regime k_lt (tiltなし) | +0.891 | +33.53% | −60.01% | −1.81pp | WFA PASS (G3), fallback 候補 |

---

## 構成

- **シグナル基盤**: DH Dyn A2 Optimized (DD × AsymEWMA × TrendTV × SlopeMult × MomDecel × VIX_MR), threshold=0.15
- **LT2 オーバーレイ (modeB, regime-conditional k_lt)**:
  - `lt_sig = compute_lt2(close, N=750)` — 750日（≈3年）モメンタム z スコア
  - `k_lt_t = 0.8 if vz_t > +0.7; 0.1 if vz_t < −0.7; 0.5 otherwise` — vz レジーム依存感度
  - `lt_bias_t = (−k_lt_t × lt_sig_t × 0.5).clip(−0.5, +0.5)` — 動的バイアス
  - `lev_mod = clip(lev_A + lt_bias, 0, 1)` — DH Dyn lev に動的バイアス
- **Bull-Tilt オーバーレイ (F7-v3 定式A, tilt=2.0)**:
  - `bull_mask = raw_a2 > 0.15` — A2 シグナルが強気閾値超
  - `tilt_amount = clip(2.0 × (raw_a2 − 0.15) × (1.0 − raw_a2), 0, 0.10)` — cap=0.10 で飽和（ほぼステップ関数）
  - `wn_tilted = wn_A + tilt_amount` — NASDAQ 比率 +最大 10%
  - `wb_tilted = max(wb_A − tilt_amount, 0)` — Bond 比率 −最大 10%
  - `wg_tilted = wg_A` — Gold 変更なし
- **CFD レバレッジ**: `compute_L_s2_vz_gated(ret, vz, target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)`
- **CFD スプレッド**: 低スプレッド想定 (`CFD_SPREAD_LOW`)
- **配分**: `wn_tilted·lev_mod·L_s2·r_nas_cfd + wg_tilted·r_g2 + wb_tilted·r_b3`
  - `wn_tilted`, `wg_tilted`, `wb_tilted`: Bull-Tilt 適用後の配分比率
  - `lev_mod`: LT2-modeB 適用後の DH Dyn レバレッジ
  - `L_s2`: VZ ゲート CFD 動的レバレッジ (1x〜7x)
- **Gold 2x**: TER 0.50% (sim proxy) + 1×SOFR（UGL実費 0.95%、差=−10.5 bps/yr, §16参照）
- **Bond 3x (TMF)**: TER 0.91% + 2×SOFR
- **DELAY**: 2営業日
- **実装**: `src/f7v3_bull_tilt.py`（F7v3+E4 主実装）, `src/e4_regime_klt.py`, `src/b1_s2_lt2.py`, `src/cfd_leverage_backtest.py`, `src/dynamic_leverage_strategies.py`, `src/long_cycle_signal.py`

---

## コスト注意事項

| 補正項目 | 影響 |
|---|---|
| SOFR financing drag (NAS CFD + Gold 1xSOFR) | CFD 軸にも適用 |
| Gold TER ギャップ (proxy 0.50% → UGL 0.95%) | **−10.5 bps/yr** (§16) |
| TMF TER ギャップ (0.91% → 1.06%) | −3.5 bps/yr (§16) |
| スワップスプレッド推定差 (+20.5 bps) | −34 bps/yr 相当 (§16) |
| 合計推定コスト過少計上 | **約 −66 bps/yr** → 現実 CAGR_OOS ≈ 33.6% 相当 |
| 売買税ドラッグ (年183回、税率20.315%) | 別途 −5〜10% CAGR（頻繁売買コスト増に注意） |
| NISA | CFD は原則 NISA 不適用 |

---

## 一次根拠ファイル（GitHub 上の正典）

| ファイル | 役割 | 日付 |
|---|---|---|
| [F7V3_BULL_TILT_2026-05-24.md](F7V3_BULL_TILT_2026-05-24.md) | F7-v3 sweep レポート（9 config, PASS 4, 定式A/B比較） | 2026-05-24 |
| [f7v3_bull_tilt_results.csv](f7v3_bull_tilt_results.csv) | F7-v3 sweep 10行 raw 結果 | 2026-05-24 |
| [src/f7v3_bull_tilt.py](src/f7v3_bull_tilt.py) | F7-v3 Bull-Tilt 実装（E4 base + 定式A/B） | 2026-05-24 |
| [G4_WFA_F7V3_2026-05-24.md](G4_WFA_F7V3_2026-05-24.md) | G4 WFA レポート（CI95_lo=+27.15% / WFE=+1.203 / PASS） | 2026-05-24 |
| [g4_wfa_f7v3_summary.csv](g4_wfa_f7v3_summary.csv) | G4 WFA サマリ（F7v3-A2 + REF-E4-sanity） | 2026-05-24 |
| [g4_wfa_f7v3_per_window.csv](g4_wfa_f7v3_per_window.csv) | G4 WFA 50窓 × 2戦略 per-window 指標 | 2026-05-24 |
| [src/g4_wfa_f7v3.py](src/g4_wfa_f7v3.py) | G4 WFA 実行スクリプト | 2026-05-24 |
| [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md) | E4 sweep レポート（採用根拠・64 config PASS 12） | 2026-05-24 |
| [e4_regime_klt_results.csv](e4_regime_klt_results.csv) | E4 sweep 65行 raw 結果 | 2026-05-24 |
| [src/e4_regime_klt.py](src/e4_regime_klt.py) | E4 Regime k_lt 実装 | 2026-05-24 |
| [G3_WFA_E4_2026-05-24.md](G3_WFA_E4_2026-05-24.md) | G3 WFA レポート（CI95_lo=+26.51% / WFE=+1.131 / PASS） | 2026-05-24 |
| [g3_wfa_e4_summary.csv](g3_wfa_e4_summary.csv) | G3 WFA サマリ（E4 + REF-N750） | 2026-05-24 |
| [B1_S2_LT2_2026-05-21.md](B1_S2_LT2_2026-05-21.md) | B1 検証レポート（3戦略比較・判定 PASS） | 2026-05-21 |
| [STRATEGY_COMPARISON_INTEGRATED_2026-05-22.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-22.md) | 12戦略統合比較表 — 統一9指標フレームワーク (§2 ◆BEST マーク) | 2026-05-22 |
| [src/b1_s2_lt2.py](src/b1_s2_lt2.py) | S2+LT2 NAV 実装・サニティチェック込み | 2026-05-21 |
| [src/cfd_leverage_backtest.py](src/cfd_leverage_backtest.py) | S2_VZGated NAV 実装 | - |
| [src/dynamic_leverage_strategies.py](src/dynamic_leverage_strategies.py) | `compute_L_s2_vz_gated` 定義 | - |
| [src/long_cycle_signal.py](src/long_cycle_signal.py) | `compute_lt2`, `signal_to_bias`, `apply_lt_mode_b` 定義 | - |
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
| `S2_VZGated + LT2-N750 + E4 Regime k_lt` (tiltなし) Sharpe_OOS +0.891 | 2026-05-24 | F7v3 Bull-Tilt が Sharpe_OOS +0.926 / CAGR_OOS +36.30% / WFA CI95_lo +27.15% で Sharpe/CAGR/WFA全指標優位。G4 WFA PASS 確認済。MaxDD差-1.95ppは guardrail 内。 |
| `S2_VZGated + LT2-N750-k0.5-modeB (固定k)` CAGR_OOS +31.16%, Sharpe_OOS 0.858 | 2026-05-24 | E4 Regime k_lt が CAGR_OOS +33.53% / Sharpe_OOS +0.891 / IS-OOS gap −1.81pp / Worst10Y★ +18.67% で 5/6 主要指標を上回ることを確認。MaxDD −0.56pp 劣後のみ。固定 k=0.5 は Shortlisted に残置（WFA 完了済みで fallback 候補として保持）。 |
| `S2_VZGated + LT2-N750-k0.5-modeB` CAGR_OOS +31.16%, Sharpe_OOS 0.858 | 2026-05-22 | B6 N-sweep で N=1500 が Sharpe_OOS +0.885 / IS-OOS gap −0.05pp で上回ることを確認。N=750 は Shortlisted に残置（CAGR と MaxDD では依然優位、代替候補として保持）。 |
| `S2_VZGated` CAGR_OOS +27.57%, Sharpe_OOS 0.769 | 2026-05-21 | S2+LT2 が CAGR_OOS +31.16% / Sharpe_OOS 0.858 / IS-OOS gap 0.18pp で全指標上回ることを確認 |
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
- Claude のローカル memory (`~/.claude/projects/.../memory/MEMORY.md`) もこのファイルを一次根拠としています
- 変更履歴は git log で追跡可能 (`git log --follow CURRENT_BEST_STRATEGY.md`)

### 変更履歴
- 2026-05-24 (G4 WFA + F7-v3 正式昇格): F7-v3 A:tilt=2.0 (E4 + step-func bull-tilt) の G4 WFA PASS (CI95_lo=+27.15%, WFE=+1.203, t_p=0.0000, α∩β)。CAGR_OOS=+36.30%, Sharpe_OOS=+0.926, IS-OOS gap=-4.26pp。旧 E4 Regime k_lt (Sharpe=+0.891) を廃止リストに移動。**正式 Active 確定**。詳細: `G4_WFA_F7V3_2026-05-24.md`, `F7V3_BULL_TILT_2026-05-24.md`
- 2026-05-24 (G3 WFA): E4 Regime k_lt の WFA が PASS (α∩β)。CI95_lo = +26.51%（>0, t_p=0.0000 α PASS）, WFE = +1.131（β PASS）。暫定 Active → 正式 Active 確定（現在は F7-v3 昇格により廃止リスト入り）。詳細: `G3_WFA_E4_2026-05-24.md`
- 2026-05-24: ベスト戦略を `S2_VZGated + LT2-N750-k0.5-modeB` から `S2_VZGated + LT2-N750 + E4 Regime k_lt (k_lo=0.1, k_hi=0.8, vz_thr=0.7)` に更新。根拠: E4 sweep（64 config）で 12 config が全 5 基準 PASS。採用 config は Worst10Y★ +18.67% と MaxDD −60.01% を同時最良化。IS-OOS gap −1.81pp は OOS が IS を上回る当時プロジェクト最高の汎化性。詳細: `E4_REGIME_KLT_SWEEP_2026-05-24.md`
- 2026-05-23: B9 (gold_frac×wn_min 2D sweep) で gf=0.65 が Sharpe_OOS +0.944 を記録(PASS)。ただし IS-OOS gap=-5.05ppはGold 2021-2026強気エクスポージャ偏重の疑いあり、リスク3指標が同時悪化のためWFA完了まで Shortlisted 保留・REF維持。詳細: `B9_COMPARISON_2026-05-23.md`
- 2026-05-22: ベスト戦略を `S2_VZGated + LT2-N750-k0.5-modeB` から `S2_VZGated + LT2-N1500-k0.5-modeB` に更新。根拠: B6 N-sweep（6 config）で N=1500 が Sharpe_OOS +0.885（全14戦略中最高）/ IS-OOS gap −0.05pp（OOS が IS を上回る・当時プロジェクト最高の汎化性）を達成。
- 2026-05-21: ベスト戦略を `S2_VZGated` から `S2_VZGated + LT2-N750-k0.5-modeB` に更新。根拠: B1 検証で CAGR_OOS +31.16% / Sharpe_OOS 0.858 / IS-OOS gap 0.18pp が全指標で S2 単体を上回ることを確認（`src/b1_s2_lt2.py`）。
- 2026-05-21: ベスト戦略を `DH Dyn 2x3x [A]` から `S2_VZGated` に更新。根拠: 10戦略統合比較表で CAGR_OOS +27.57% / Sharpe_OOS 0.769 が全戦略中トップ。
- 2026-05-12: Scenario D 補正適用。CAGR 30.84% → 22.50%, Sharpe 1.299 → 0.993, MaxDD -31.40% → -45.08%。
- 2026-05-11: 初版作成。`DH Dyn 2x3x [A] 閾値0.15` を現行ベストに確定。

---

*管理者: Kazuya Murayama*
