# CURRENT BEST STRATEGY — 単一の真実 (Single Source of Truth)

> **このファイルは「現行のベスト戦略」を一意に特定するための正典です。**
> **「ベスト戦略は？」と問われた時、Claude / 人間ともにまずこのファイルだけを見れば良いように設計されています。**

作成日: 2026-05-11
最終更新日: 2026-05-24

---

## 現行ベスト戦略

**戦略名: `S2_VZGated + LT2-N750 + E4 Regime k_lt`（Vol-Zone ゲート CFD + 長期逆張り + ボラレジーム動的 k_lt）**

### 主要指標 (Scenario D コスト補正済み基盤, 1974-01-02 〜 2026-03-26, 52.26年)

> **注意:** DH Dyn 2x3x [A] シグナル (Approach A, 閾値 0.15) の上に CFD Vol-Zone ゲートを適用し、
> さらに LT2-N=750 を **vz レジーム条件付き k_lt** (k_lo=0.1 / k_mid=0.5 / k_hi=0.8, vz_thr=0.7) で lev シグナルに重畳したもの。
> CFD 軸の指標は `src/e4_regime_klt.py` の出力値（E4 sweep 64 config 中 Worst10Y★・MaxDD 同時最良の config）。

| 指標 | 値 | 備考 |
|---|---|---|
| CAGR_OOS (2021-2026) | **+33.53%** | Out-of-sample |
| Sharpe_OOS | **+0.891** | OOS期間Sharpe比 |
| MaxDD (FULL) | **−60.01%** | 最大ドローダウン |
| Worst10Y★ CAGR (FULL) | **+18.67%** | カレンダー年ベース最悪10年窓 |
| P10_5Y▷ CAGR (FULL) | **+9.78%** | 5年CAGR 分布 P10 |
| IS-OOS gap | **−1.81 pp** | OOS が IS を上回る（優秀な汎化性） |
| Trades/yr | **約27回** (月2.3回) | 基底DH Dynシグナルと同じ。コスト優位性高 |
| CAGR_IS (1974-2021) | +31.72% | In-sample |
| WFA_CI95_lo | **+26.51%** | G3 WFA 50窓、α PASS（t_p=0.0000） |
| WFA_WFE | **+1.131** | G3 WFA、β PASS（0.5 ≤ WFE ≤ 2.0）|

### ベスト戦略選定根拠 (2026-05-24 確定・tilt系棄却後)

| 評価軸 | E4 Regime k_lt **◆ BEST** | F7v3/F8 tilt 系 (Shortlisted) | 判断 |
|---|---|---|---|
| CAGR_OOS | +33.53% | +36.30〜36.83% | tilt系 +2.8〜3.3pp |
| Sharpe_OOS | +0.891 | +0.926〜0.934 | tilt系 +0.035〜0.043 |
| MaxDD | **−60.01%** | −61.96〜63.07% | **E4 優位** |
| Worst10Y★ | **+18.67%** | +18.27〜18.58% | **E4 優位** |
| IS-OOS gap | **−1.81pp** | −4.26〜4.28pp | **E4 優位** (OOS寄り乖離小さい) |
| **Trades/yr** | **約27回** | **約182回** | **E4 圧倒的優位 (1/7コスト)** |
| WFA_CI95_lo | +26.51% | +27.15〜27.92% | tilt系 +0.64〜1.41pp |

> **E4 採用・tilt系棄却理由**:
> tilt 系 (F7v3/F8) は OOS Sharpe を +0.035〜+0.043 改善するが、Trades/yr が 27→182 回（約7倍）に急増。
> 182 回/年の取引コスト（スプレッド・税率 20.315%・CFD スワップ）は CAGR を数〜10% 押し下げる可能性があり、
> OOS 期間 (2021-2026) の NASDAQ 強気相場に対するアウトサンプル偶然性が疑われる。
> 全期間（IS+OOS）での Sharpe 改善幅は軽微で、コストを加味すると実質同等以下と判断。
> IS-OOS gap が E4（−1.81pp）に対し tilt 系（−4.26〜4.28pp）と 2〜2.5 倍拡大している点も汎化性の観点でリスク。
> **G3 WFA (2026-05-24) PASS**: CI95_lo = +26.51%（>0 α PASS, t_p=0.0000）/ WFE = +1.131（0.5–2.0 β PASS）→ **正式 Active 確定**。

---

## Shortlisted（次善候補 / WFA 完了）

| 戦略 | Sharpe | CAGR_OOS | MaxDD | Trades/yr | WFA CI95_lo | 採用留保理由 |
|---|---|---|---|---|---|---|
| F8 R5_CALM_BOOST | +0.934 | +36.83% | −63.07% | 182 | +27.92% | Trades/yr過多、OOS偶然性疑い |
| F7v3+E4 A:tilt=2.0 | +0.926 | +36.30% | −61.96% | 183 | +27.15% | 同上 |
| LT2-N750 固定k=0.5 | +0.858 | +31.16% | −59.45% | 27 | +25.7% | E4に劣後、WFA PASS済み |

---

## 構成

- **シグナル基盤**: DH Dyn A2 Optimized (DD × AsymEWMA × TrendTV × SlopeMult × MomDecel × VIX_MR), threshold=0.15
- **LT2 オーバーレイ (modeB, regime-conditional k_lt)**:
  - `lt_sig = compute_lt2(close, N=750)` — 750日（≈3年）モメンタム z スコア
  - `k_lt_t = 0.8 if vz_t > +0.7; 0.1 if vz_t < −0.7; 0.5 otherwise` — vz レジーム依存感度
  - `lt_bias_t = (−k_lt_t × lt_sig_t × 0.5).clip(−0.5, +0.5)` — 動的バイアス
  - `lev_mod = clip(lev_A + lt_bias, 0, 1)` — DH Dyn lev に動的バイアス
- **CFD レバレッジ**: `compute_L_s2_vz_gated(ret, vz, target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)`
- **CFD スプレッド**: 低スプレッド想定 (`CFD_SPREAD_LOW`)
- **配分**: `wn·lev_mod·L_s2·r_nas_cfd + wg·r_g2 + wb·r_b3`
  - `wn`, `wg`, `wb`: DH Dyn [A] Approach A と同一 (`simulate_rebalance_A`)
  - `lev_mod`: LT2-modeB 適用後の DH Dyn レバレッジ
  - `L_s2`: VZ ゲート CFD 動的レバレッジ (1x〜7x)
- **Gold 2x**: TER 0.50% (sim proxy) + 1×SOFR
- **Bond 3x (TMF)**: TER 0.91% + 2×SOFR
- **DELAY**: 2営業日
- **実装**: `src/e4_regime_klt.py`（E4 主実装）, `src/b1_s2_lt2.py`, `src/cfd_leverage_backtest.py`, `src/dynamic_leverage_strategies.py`, `src/long_cycle_signal.py`

---

## コスト注意事項

| 補正項目 | 影響 |
|---|---|
| SOFR financing drag (NAS CFD + Gold 1xSOFR) | CFD 軸にも適用 |
| Gold TER ギャップ (proxy 0.50% → UGL 0.95%) | **−10.5 bps/yr** (§16) |
| TMF TER ギャップ (0.91% → 1.06%) | −3.5 bps/yr (§16) |
| スワップスプレッド推定差 (+20.5 bps) | −34 bps/yr 相当 (§16) |
| 合計推定コスト過少計上 | **約 −66 bps/yr** → 現実 CAGR_OOS ≈ 30.5% 相当 |
| 売買税ドラッグ (年27回、税率20.315%) | 別途 −2〜5% CAGR |
| NISA | CFD は原則 NISA 不適用 |

---

## 一次根拠ファイル（GitHub 上の正典）

| ファイル | 役割 | 日付 |
|---|---|---|
| [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md) | E4 sweep レポート（採用根拠・64 config PASS 12） | 2026-05-24 |
| [e4_regime_klt_results.csv](e4_regime_klt_results.csv) | E4 sweep 65行 raw 結果 | 2026-05-24 |
| [src/e4_regime_klt.py](src/e4_regime_klt.py) | E4 Regime k_lt 実装 | 2026-05-24 |
| [G3_WFA_E4_2026-05-24.md](G3_WFA_E4_2026-05-24.md) | G3 WFA レポート（CI95_lo=+26.51% / WFE=+1.131 / PASS） | 2026-05-24 |
| [g3_wfa_e4_summary.csv](g3_wfa_e4_summary.csv) | G3 WFA サマリ | 2026-05-24 |
| [src/g3_wfa_e4.py](src/g3_wfa_e4.py) | G3 WFA 実行スクリプト | 2026-05-24 |
| [B1_S2_LT2_2026-05-21.md](B1_S2_LT2_2026-05-21.md) | B1 検証レポート | 2026-05-21 |
| [src/corrected_strategy_backtest.py](src/corrected_strategy_backtest.py) | DH Dyn シグナル基盤 (Scenario D) | 2026-05-12 |
| [src/product_costs.py](src/product_costs.py) | コスト定数の単一の真実 | 2026-05-12 |

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
