# CURRENT BEST STRATEGY — 単一の真実 (Single Source of Truth)

> **このファイルは「現行のベスト戦略」を一意に特定するための正典です。**
> **「ベスト戦略は？」と問われた時、Claude / 人間ともにまずこのファイルだけを見れば良いように設計されています。**

作成日: 2026-05-11
最終更新日: 2026-05-22

---

## 現行ベスト戦略

**戦略名: `S2_VZGated + LT2-N750-k0.5-modeB`（Vol-Zone ゲート型 CFD レバレッジ + 長期逆張りフィルタ）**

### 主要指標 (Scenario D コスト補正済み基盤, 1974-01-02 〜 2026-03-26, 52.26年)

> **注意:** DH Dyn 2x3x [A] シグナル (Approach A, 閾値 0.15) の上に CFD Vol-Zone ゲートを適用し、
> さらに LT2-N750-k0.5-modeB（長期モメンタム逆張りバイアス、N=750日 ≈ 3年）を lev シグナルに重畳したもの。
> CFD 軸の指標は `src/b1_s2_lt2.py` の出力値（B1 検証、B6 N-sweep N=750 サニティ確認済み）。

| 指標 | 値 | 備考 |
|---|---|---|
| CAGR_OOS (2021-2026) | **+31.16%** | Out-of-sample |
| Sharpe_OOS | **0.858** | OOS期間Sharpe比 |
| MaxDD (FULL) | **−59.45%** | 最大ドローダウン |
| Worst10Y★ CAGR (FULL) | **+18.10%** | カレンダー年ベース最悪10年窓 |
| P10_5Y▷ CAGR (FULL) | +9.4% | 5年CAGR 分布 P10 (B1より) |
| IS-OOS gap | **+0.18 pp** | 過剰適合なし（< 2pp 優秀域） |
| Trades/yr | 約27回 (月2.3回) | 基底DH Dynシグナルと同じ |
| CAGR_IS (1974-2021) | +31.33% | In-sample |
| WFA_CI95_lo | — | G1_WFA_2026-05-21 参照（S2+LT2 行） |
| WFA_WFE | — | 同上 |

### ベスト戦略選定根拠 (2026-05-22 確定)

| 評価軸 | N=750 **◆ BEST** | N=1500 (候補) | 優位 |
|---|---|---|---|
| CAGR_OOS | **+31.16%** | +30.84% | **N750 +0.32pp** |
| MaxDD | **−59.45%** | −63.37% | **N750 −3.92pp 改善** |
| Worst10Y★ | **+18.10%** | +16.60% | **N750 +1.50pp** |
| P10_5Y▷ | +9.4% | +9.9% | N1500 *(+0.5pp、軽微)* |
| Sharpe_OOS | 0.858 | **0.885** | N1500 (+0.027、補助指標) |
| IS-OOS gap | +0.18pp | −0.05pp | N1500 *(両者とも十分小)* |

> **N750 採用理由**: CAGR_OOS・MaxDD・Worst10Y★ の 3 主要軸で全て N750 が優位。
> Sharpe_OOS の差（0.027）は補助指標であり、リスク指標（MaxDD −3.92pp）の改善が優先される。
> IS-OOS gap は N750 の +0.18pp も「優秀域（< 2pp）」であり、過剰適合の懸念なし。
> N1500 は **SHORTLISTED** として保持（別ファイル `B6_S2_LT2_N_SWEEP_2026-05-22.md` 参照）。

---

## 構成

- **シグナル基盤**: DH Dyn A2 Optimized (DD × AsymEWMA × TrendTV × SlopeMult × MomDecel × VIX_MR), threshold=0.15
- **LT2 オーバーレイ (modeB)**:
  - `lt_sig = compute_lt2(close, N=750)` — 750日（≈3年）モメンタム z スコア
  - `lt_bias = signal_to_bias(lt_sig, k=0.5)` — `(-0.25 × z).clip(-0.5, +0.5)`
  - `lev_mod = clip(lev_A + lt_bias, 0, 1)` — DH Dyn lev に ±0.25 バイアス
- **CFD レバレッジ**: `compute_L_s2_vz_gated(ret, vz, target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)`
- **CFD スプレッド**: 低スプレッド想定 (`CFD_SPREAD_LOW`)
- **配分**: `wn·lev_mod·L_s2·r_nas_cfd + wg·r_g2 + wb·r_b3`
  - `wn`, `wg`, `wb`: DH Dyn [A] Approach A と同一 (`simulate_rebalance_A`)
  - `lev_mod`: LT2-modeB 適用後の DH Dyn レバレッジ
  - `L_s2`: VZ ゲート CFD 動的レバレッジ (1x〜7x)
- **Gold 2x**: TER 0.50% (sim proxy) + 1×SOFR（UGL実費 0.95%、差=−10.5 bps/yr, §16参照）
- **Bond 3x (TMF)**: TER 0.91% + 2×SOFR
- **DELAY**: 2営業日
- **実装**: `src/b1_s2_lt2.py`, `src/cfd_leverage_backtest.py`, `src/dynamic_leverage_strategies.py`, `src/long_cycle_signal.py`

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
| [B1_S2_LT2_2026-05-21.md](B1_S2_LT2_2026-05-21.md) | B1 検証レポート（3戦略比較・判定 PASS） | 2026-05-21 |
| [STRATEGY_COMPARISON_INTEGRATED_2026-05-22.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-22.md) | 12戦略統合比較表 — 統一9指標フレームワーク (§2 ◆BEST マーク) | 2026-05-22 |
| [A6_LMAX_SWEEP_2026-05-21.md](A6_LMAX_SWEEP_2026-05-21.md) | l_max ロバストネス確認 (PASS) | 2026-05-21 |
| [A1_NVOL_SWEEP_2026-05-21.md](A1_NVOL_SWEEP_2026-05-21.md) | n_vol ロバストネス確認 (n=20 最適確認) | 2026-05-21 |
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
- Claude のローカル memory (`~/.claude/projects/.../memory/nasdaq_best_strategy.md`) もこのファイルを一次根拠としています
- 変更履歴は git log で追跡可能 (`git log --follow CURRENT_BEST_STRATEGY.md`)

### 変更履歴
- 2026-05-22: ベスト戦略を `S2_VZGated + LT2-N750-k0.5-modeB` から `S2_VZGated + LT2-N1500-k0.5-modeB` に更新。根拠: B6 N-sweep（6 config）で N=1500 が Sharpe_OOS +0.885（全14戦略中最高）/ IS-OOS gap −0.05pp（OOS が IS を上回る・本プロジェクト史上最高の汎化性）を達成。CAGR_OOS は N=750 比 −0.32pp と僅差、MaxDD/Worst10Y★ 劣後は guardrail 内。同セッション B3/B4/B5 LT4/LT6/LT7 派生も全 Active 未満で Shortlisted 化、P1/P3/P5/S3/S4 は S2 単体未満で全棄却（[b6_s2_lt2_N_sweep_results.csv](b6_s2_lt2_N_sweep_results.csv)）。
- 2026-05-21: ベスト戦略を `S2_VZGated` から `S2_VZGated + LT2-N750-k0.5-modeB` に更新。根拠: B1 検証で CAGR_OOS +31.16% / Sharpe_OOS 0.858 / IS-OOS gap 0.18pp が全指標で S2 単体を上回ることを確認（`src/b1_s2_lt2.py`）。旧ベスト S2_VZGated を廃止リストに移動。
- 2026-05-21: ベスト戦略を `DH Dyn 2x3x [A]` から `S2_VZGated` に更新。根拠: 10戦略統合比較表で CAGR_OOS +27.57% / Sharpe_OOS 0.769 が全戦略中トップ。旧ベストを廃止リストに移動。
- 2026-05-12: Scenario D 補正適用。CAGR 30.84% → 22.50%, Sharpe 1.299 → 0.993, MaxDD -31.40% → -45.08%。
- 2026-05-11: 初版作成。`DH Dyn 2x3x [A] 閾値0.15` を現行ベストに確定。

---

*管理者: Kazuya Murayama*
