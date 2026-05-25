# RESEARCH CONTEXT — 研究文脈と実験系統

> **新セッション開始時にこのファイルを読めば、リポジトリの研究文脈を一発で把握できるよう設計されています。**
> **「ベスト戦略は？」だけなら [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を読む。実験の連鎖・系統・棄却理由まで把握したい場合は本ファイル。**

作成日: 2026-05-24
最終更新日: 2026-05-24
管理者: 男座員也（Kazuya Oza）

---

## 0. 引き継ぎ読書順序（新セッション推奨）

| # | ファイル | 役割 | 所要 |
|---|---|---|---|
| 1 | [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) | 現行ベスト戦略の正典（数値・構成・WFA結果） | 3分 |
| 2 | **RESEARCH_CONTEXT.md**（本ファイル） | 実験系統・棄却理由・次に試すべき方向性 | 5分 |
| 3 | [tasks.md](tasks.md) | Pending / In Progress / Completed | 2分 |
| 4 | [STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md) | 全戦略台帳（Active / Shortlisted / Rejected / Deferred） | 必要時 |
| 5 | [EVALUATION_STANDARD.md](EVALUATION_STANDARD.md) | 評価基準 v1.1（コスト・期間・指標・9指標標準） | 必要時 |
| 6 | [FILE_INDEX.md](FILE_INDEX.md) | 全ファイル所在 | 必要時 |

> **6点セット**だが、通常は 1〜3 で十分。新規実験を着手するなら 4〜5 を、ファイルが見つからないなら 6 を引く。

---

## 1. 現行ベスト戦略サマリ

> 詳細は [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を参照。本ファイルでは1段落で要約。

**戦略名: S2_VZGated + LT2-N750 + E4 Regime k_lt** (2026-05-24 正式 Active 確定・F8 tilt系棄却後)

- 構成 = **DH Dyn 2x3x [A] (シグナル基盤)** + **CFD Vol-Zone ゲート** + **LT2-N750 長期逆張り** + **E4 ボラレジーム別 k_lt**
- 主要指標: CAGR_OOS=+33.53%, Sharpe_OOS=+0.891, MaxDD=-60.01%, Worst10Y★=+18.67%, IS-OOS gap=-1.81pp
- WFA: CI95_lo=+26.51% (α PASS, t_p=0.0000), WFE=+1.131 (β PASS) — G3 にて正式昇格
- 棄却理由（F8 R5_CALM_BOOST）: Trades/yr=182（E4比7倍）、OOS偶然性疑い、IS-OOS gap 2.4倍拡大
- 一次根拠: `E4_REGIME_KLT_SWEEP_2026-05-24.md`, `G3_WFA_E4_2026-05-24.md`
- 主要実装: `src/e4_regime_klt.py`, `src/long_cycle_signal.py`, `src/cfd_leverage_backtest.py`

---

## 2. 実験系統図（A〜H 系列の意味と連鎖）

> 各系列は「どんなパラメータ／設計次元を探索したか」で分類される。
> **「F7→F7v2→F7v3→F8→F9」のように同系内でバージョンアップする**ことが多い。

### 2.1 系列一覧

| 系列 | テーマ | 状態 | 代表結果 |
|---|---|---|---|
| **A系** | CFD レバレッジ・パラメータ最適化（n_vol / target_vol / k_vz / gate_min / l_max） | 完了 (S2_VZGated 採用) | A1〜A6 で S2_VZGated の最良パラメータ確定 |
| **B系** | LT (Long-cycle 長期逆張り) シグナル最適化 | 完了 (LT2-N750 採用) | B1 で LT2 採用, B6 で N=750/1500 候補, B7/B8 で他バリエーション棄却 |
| **C系** | HY / NASDAQ Heavy 系外部ゲート | 棄却 | C1 HY ゲート効果なし, C2 nasdaq_cap 効果限定的 |
| **D系** | OOS 境界変動・H4 拡張 | 棄却 | D1 で OOS 境界 ±1年 シフト ロバスト性確認のみ |
| **E系** | ボラレジーム条件付き動的パラメータ | **採用 (E4)** | E4 Regime k_lt が S2+LT2+E4 として Active 入り (CAGR_OOS +33.53%) |
| **F系** | wn/wb 動的傾斜（Bull-Tilt） | **Shortlisted (F8 R5_CALM_BOOST)** | F5 bond regime → F6 vol scale → F7 / F7v2 / F7v3 → **F8 R5_CALM_BOOST** (Shortlisted) → F9 threshold |
| **G系** | Walk-Forward Analysis (WFA) 検証 | 検証ツール | G1 baseline → G2 B9 → G3 E4 (PASS) → G4 F7v3 (PASS) → **G5 F8R5 (PASS)** |
| **H系** | 外部シグナル（Gold / Real Yield 等）オーバーレイ | 主に棄却 | H1 S4_sweep, H4 wgwb, H5 gold_dyn いずれも S2_VZGated 超えず |
| **P系** | 外部マクロシグナル (HY / CPI 等)・モデル系 | 一部 Shortlisted (非標準コスト) | P01/P02/P05 が Shortlisted（[非標準コスト] フラグ）。P1 SOFR/P3 Momentum/P4 Composite/P5 Kelly は全棄却 |
| **S系** | A2 ConvictionScore 直接レバ変換 | 棄却 | S1〜S4 で全て S2_VZGated に劣後 (S3 は IS-OOS gap +22pp で致命的) |

### 2.2 採用に至るチェーン（時系列）

```
2026-02-06  Ens2(Asym+Slope)  [当時のベスト]
   ↓ 2026-04-21 閾値スイープで降格
2026-04-21  DH Dyn 2x3x [A] 閾値0.15  [TQQQ参照, Scenario A]
   ↓ 2026-05-12 Scenario D 補正でCAGRが30.81%→22.50%に修正
2026-05-12  DH Dyn 2x3x [A] Scenario D  [現実コスト反映]
   ↓ 2026-05-19 CFD系の登場で降格
2026-05-21  S2_VZGated  [CFD Vol-Zone ゲート, CAGR_OOS +27.57%]
   ↓ 2026-05-21 B1 LT2 採用
2026-05-21  S2_VZGated + LT2-N750 (固定 k=0.5)  [CAGR_OOS +31.16%]
   ↓ 2026-05-22 N=1500 が Sharpe で勝つ → 一時昇格
2026-05-22  S2_VZGated + LT2-N1500
   ↓ 2026-05-23 N=750 にコミット 8503200 で差し戻し（リスク指標優位）
2026-05-23  S2_VZGated + LT2-N750  [安定版]
   ↓ 2026-05-24 E4 ボラレジーム条件 k_lt で +2.37pp 改善
2026-05-24  S2_VZGated + LT2-N750 + E4 Regime k_lt  [CAGR_OOS +33.53%]
   ↓ 2026-05-24 F7v3 Bull-Tilt + G4 WFA PASS
2026-05-24  + F7v3 (A:tilt=2.0, cap=0.10)  [CAGR_OOS +36.30%]
   ↓ 2026-05-24 F8 R5_CALM_BOOST + G5 WFA PASS → Trades/yr=182（E4比7倍）棄却・E4に差し戻し
2026-05-24  + F8 R5_CALM_BOOST (Shortlisted)  CAGR_OOS=+36.83%, Sharpe_OOS=+0.934, Trades/yr=182
2026-05-24  S2_VZGated + LT2-N750 + E4 Regime k_lt
            ★現行 Active★  CAGR_OOS=+33.53%, Sharpe_OOS=+0.891
```

---

## 3. 主要な発見と棄却理由サマリ

### 3.1 採用した改善（時系列）

| 採用要素 | 改善幅 | 発見セッション | 一次根拠 |
|---|---|---|---|
| **CFD Vol-Zone ゲート (S2_VZGated)** | CAGR +14pp 級（vs DH Dyn 単体） | 2026-05-19 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **LT2-N750 長期逆張り** | Sharpe +0.09, IS-OOS gap 改善 | 2026-05-21 (B1) | [B1_S2_LT2_2026-05-21.md](B1_S2_LT2_2026-05-21.md) |
| **E4 Regime k_lt (ボラレジーム動的 k)** | CAGR +2.37pp, IS-OOS gap -1.99pp | 2026-05-24 (E4) | [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md) |
| **F7v3 Bull-Tilt (step-func, tilt=2.0)** | CAGR +2.77pp, Sharpe +0.035 | 2026-05-24 (F7v3) | [F7V3_BULL_TILT_2026-05-24.md](F7V3_BULL_TILT_2026-05-24.md) |
| **F8 R5_CALM_BOOST (レジーム別 cap)** | Sharpe +0.008, CAGR +0.53pp, WFA CI95_lo +0.77pp | 2026-05-24 (F8) | [F8_REGIME_TILT_2026-05-24.md](F8_REGIME_TILT_2026-05-24.md) |

### 3.2 重要な棄却（カテゴリ別）

#### A. アンサンブル系 → 全棄却
- **Ens2 max_lev=1.0系**: DH Dyn [A] 単体に劣後 (Sharpe 不足)
- **Ens2 max_lev=3.0系**: IS-OOS gap 大 + MaxDD -77%超 で実運用不可
- **MajorityVote_P2**: 最終結論に組み込まれず
- **結論**: アンサンブル化のメリットは観測されず、現時点で全棄却

#### B. A2 ConvictionScore 直接レバ変換 (S系) → 全棄却
- **S1 / S2_TV/KVZ/GATEMIN sweeps**: パラメータ感度 dead or 過剰適合
- **S3 Decomposed A2**: **IS-OOS gap +22.39pp で致命的過剰適合**（同類実験を二度行わないこと）
- **S4 RelVol Gated**: 36/36 config が S2 単体未満
- **結論**: A2 構成要素を直接レバ生成に流すと IS バイアスが直撃

#### C. CFD レバレッジ動的調整 (P系)
- **P1 SOFR Adaptive**: 18/18 が S2 単体未満。ゼロ金利期で過剰デレバ
- **P3 Momentum Lev**: 16/16 が S2 単体未満。DH Dyn 既にモメンタム内包で冗長
- **P4 Composite (三因子乗算)**: 24/24 が S2 単体未満。論理AND効果で過剰デレバ
- **P5 Kelly Sizing**: 12/12 が S2 単体未満。分布安定仮定がNASDAQで破綻
- **結論**: 既存 S2_VZGated より単純な動的レバ調整に勝てるものなし

#### D. 外部シグナルゲート
- **C1 HY ゲート**: S2+LT2 への上乗せで改善なし（標準コスト下）
- **H1〜H5 (Gold等)**: いずれも S2_VZGated 超えず
- **P01/P02/P05 (HY/CPI ゲート)**: **[非標準コスト]** で Shortlisted 残置だが標準コスト下での再評価が必要

#### E. 配分・リバランス系 → 棄却
- **F1 配分スイープ**: 既存 DH Dyn [A] 配分が最適
- **Rebalance Frequency / Partial Rebalance**: 固定頻度・部分リバ いずれも劣後
- **SOXL 追加 / Portfolio Diversification**: 分散効果限定的

#### F. F系内のバージョン進化（採用に至るまでの失敗）
- **F5 Bond Regime**: 効果限定的
- **F6 Vol Scale**: 効果限定的
- **F7 / F7v2 Bull-Tilt (初期定式)**: tilt の peak が cap=0.10 に届かず、tilt パラメータが事実上不感応に
- **F7v3 Bull-Tilt 定式再設計**: tilt=10 で実質ステップ関数化 → 採用
- **F8 R5_CALM_BOOST**: F7v3 にレジーム別 cap を追加 → G5 WFA PASS → **Shortlisted**（Trades/yr=182, OOS偶然性疑いで Active 棄却）
- **F9 THRESHOLD 最適化**: THRESHOLD=0.15 が最良で現行値維持確認

#### G. LT 系内のバージョン進化
- **B3 LT4 / B4 LT6 / B5 LT7 / B7 LT1 / B8 LT3**: 各種 LT 変種試行
- **B6 LT2 N-sweep**: N=750 が安定 (Active)、N=1500 が Sharpe 最高 (Shortlisted)
- **B9 GoldFrac, B11 DualN**: 派生検証、いずれも N=750 単体を超えず

---

## 4. 現在の未着手方向性

> 詳細・進捗は [tasks.md](tasks.md) の Pending セクション参照。

### 4.1 短期 Pending（実装フェーズ）
- [x] ~~**STRATEGY_PERFORMANCE_COMPARISON に F8-R5 列追加**~~ → v1.5で追加済み → v1.6でF8棄却により削除済み（完了・不要）
- [ ] **Approach A への GAS 切替実装** (閾値 0.15 と同時変更, 実運用側 nasdaq-strategy-gas リポ)
- [ ] **2026年データへの拡張**（継続監視）
- [ ] **Ens2 戦略の OOS 検証** (2022-2026, 完全性のため)

### 4.2 中期で検討に値する方向性（未着手）

> 以下は本セッションまでの研究履歴から「次に試す価値がある」と思われる方向性。
> **着手前に必ず `STRATEGY_REGISTRY.md` で重複チェック**すること。

| 候補 | 着想 | 重複チェックすべき箇所 |
|---|---|---|
| **F8 派生**: cap の3レジーム値を更に細分（例: calm=0.15→0.20 / bull-VZ stay / bear-VZ=0 完全停止） | F8 で「calm boost が WFA でも効く」確認できた次の段階 | §3.3 F-series Rejected |
| **F8 + F7v3 ハイブリッド**: F7v3 単純 cap=0.10 + F8 レジーム別 cap を時間軸で交互適用 | 現行 F8 は静的設定。動的切替の余地あり | F-series 全般 |
| **VZ_thr 感度 (現状 ±0.7)**: ±0.5 / ±0.9 でレジーム境界感度を再評価 | E4 では vz_thr=0.7 が最良と判定済み（dead parameter かは未確認） | §3.3 E系 |
| **G5 WFA を H-系 外部シグナルに適用**: H4 wgwb や H5 gold_dyn 等を WFA で再評価（過去は単純 OOS のみ） | H 系は OOS 評価のみで棄却。WFA で復活可能性は低いが念のため | §3.4 H-series |
| **P-series の標準コスト再評価**: P01 / P02 / P05 を Scenario D 統一前提で再計算 | P-series は [非標準コスト]。標準下で残存するかは未検証 | §2 Shortlisted P-series |
| **LT2 N とF8 cap の交差感度**: 現状は別々に最適化。同時グリッドで再評価 | B6 (LT2-N) と F8 (cap regime) の交互作用は未検証 | B6 / F8 個別最適 |
| **Trades/yr 削減**: F8 で 182 trades/yr。リバランスバンド導入で半減可能か（税ドラッグ -2.8〜5.2% CAGR 軽減狙い） | 過去に Partial_Rebalance / Rebalance_Frequency_Sweep は棄却済み（リターン低下）。**新発想が必要** | §3.5 Partial_Rebalance |
| **Sharpe with Rf>0 再評価**: 現行 Sharpe は Rf=0、SOFR>4%環境で過大評価。Rf=4% 換算で順位入替えはあるか | EVALUATION_STANDARD §3.2 で注記済みだが未実装 | §3.2 注意事項 |

### 4.3 着手すべきでない方向性（再警告）

- **❌ 同種のアンサンブル**（DH Dyn と CFD系の組合せ統合済み、追加は劣後確実）
- **❌ A2 構成要素の直接レバ変換**（S系で全て致命的過剰適合確認済み）
- **❌ Scenario A 単独評価**（コスト過少推計、Scenario D 必須）
- **❌ FINAL_ プレフィックス命名**（CLAUDE.md / EVALUATION_STANDARD §5.1 で禁止）

---

## 5. 評価基準ハイライト（EVALUATION_STANDARD v1.1 から）

> 全項目は [EVALUATION_STANDARD.md](EVALUATION_STANDARD.md) 参照。ここはセッション開始時に頭に入れる最低限。

### 5.1 標準前提
- **コスト Scenario D**（TER + 製品別 SOFR × multiplier + swap spread）
- **期間: IS 1974-01-02〜2021-05-07 / OOS 2021-05-08〜2026-03-26 / FULL 1974〜2026**
- **DELAY=2 営業日**（look-ahead bias 防止）
- **252 営業日/年**（暦日 365 換算は不可）
- **Sharpe Rf=0** ※高金利期で過大評価される旨はレポートに明記

### 5.2 9指標標準（§3.12）
| # | 指標 | 種別 |
|---|---|---|
| 1 | CAGR_OOS | 一次 |
| 2 | Sharpe_OOS | 一次 |
| 3 | MaxDD(FULL) | 一次 |
| 4 | Worst10Y★ | 一次（カレンダー年方式） |
| 5 | P10_5Y▷ | 一次 |
| 6 | IS-OOS gap | 一次 |
| 7 | Trades/yr | 一次 |
| 8 | WFA_CI95_lo | WFA 補助 |
| 9 | WFA_WFE | WFA 補助 |

### 5.3 PASS 基準（参考リファレンス）
WFA で α∩β PASS が「正式 Active 昇格」の必要条件:
- **α**: WFA_CI95_lo > 0 AND t_p < 0.05
- **β**: 0.5 ≤ WFA_WFE ≤ 2.0

現行ベスト (F8 R5_CALM_BOOST) の REF として PASS 基準は以下を最低限満たすことが望ましい:
- Sharpe_OOS ≥ 0.911 (REF F7v3 BASE +0.020 程度)
- CAGR_OOS ≥ 29.5%
- IS-OOS gap ≤ 6.0pp
- MaxDD > -65.0% (guardrail)
- Worst10Y★ ≥ 15.0%

### 5.4 失格ライン (§4.3)
- IS-OOS gap > +10 pp
- MaxDD < -80%
- Worst10Y★ < 0%
- Sharpe_OOS < 0.5

### 5.5 禁止指標（ユーザー明示指示なしに評価で使用禁止）
- Stable_Sharpe / WinRate_yr / WorstK5_mean_CAGR / IR_vs_BH

---

## 6. ファイル命名・所在の規則（必読）

### 6.1 系列別ファイル探索
- **戦略レポート**: `<TOPIC>_YYYY-MM-DD.md` 形式（例: `F8_REGIME_TILT_2026-05-24.md`）
- **WFA レポート**: `G<N>_WFA_<TARGET>_YYYY-MM-DD.md`（例: `G5_WFA_F8R5_2026-05-24.md`）
- **比較表**: `STRATEGY_PERFORMANCE_COMPARISON_YYYY-MM-DD.md`
- **CSV**: 小文字スネーク（例: `f8_regime_tilt_results.csv`）

### 6.2 実装ファイルの正典（src/ 配下）
| 役割 | ファイル |
|---|---|
| シグナル基盤 (Scenario D) | `src/corrected_strategy_backtest.py` |
| コスト定数の単一の真実 | `src/product_costs.py` |
| CFD レバレッジ・Vol-Zone ゲート | `src/cfd_leverage_backtest.py`, `src/dynamic_leverage_strategies.py` |
| 長期サイクル LT | `src/long_cycle_signal.py`, `src/b1_s2_lt2.py` |
| E4 Regime k_lt | `src/e4_regime_klt.py` |
| F8 R5_CALM_BOOST (現行 Active) | `src/f8_regime_tilt.py` |
| F7v3 Bull-Tilt (Shortlisted) | `src/f7v3_bull_tilt.py` |
| WFA エンジン | `src/g1_wfa.py` (compute_summary_stats 含む) |
| MD ヘッダ標準 | `src/_sweep_format.py` (`MD_HEADER_1P/2P/STRAT` import 必須) |

### 6.3 やってはいけないこと
- ❌ `FINAL_` プレフィックスを新規ファイルに使用
- ❌ sweep / 戦略比較 MD で `CAGR_IS` / `CAGR_FULL` をヘッダに含める（v1.1 違反）
- ❌ sweep MD で手書きヘッダ使用（必ず `src/_sweep_format.py` から import）
- ❌ CSV を Sharpe 降順で並べて「トップ」を答える（CSV は実験ログ、結論ではない）

---

## 7. 重要な参考プロトコル

### 7.1 新検証着手前プロトコル（4ステップ・スキップ禁止）

> 詳細は [docs/rules/06_strategy-verification.md](docs/rules/06_strategy-verification.md)

1. **重複チェック**: STRATEGY_REGISTRY.md §3 Rejected で同種実験の有無
2. **評価基準確認**: EVALUATION_STANDARD.md §0 必須チェック
3. **差分仮説**: 何を変更してどう改善する仮説か明文化
4. **登録**: 着手前に tasks.md Pending に追加

スキップ時は重複研究扱いで結果は採用候補から除外される。

### 7.2 Active 昇格時の必須更新

新戦略が PASS した時:
1. `CURRENT_BEST_STRATEGY.md` 更新（旧 Active を廃止リストへ）
2. `STRATEGY_REGISTRY.md` §1 Active 上書き、旧 Active を §2 Shortlisted へ
3. `tasks.md` Completed に 1 行追加
4. `MEMORY.md` の "現行ベスト" セクション更新（プロジェクト記憶）
5. **`RESEARCH_CONTEXT.md` (本ファイル) §1 と §2.2 採用チェーンも更新**

---

## 改訂履歴

| 日付 | 変更 |
|---|---|
| 2026-05-24 | 初版作成。F8 R5_CALM_BOOST 昇格直後の研究文脈・系統図・棄却サマリを集約。新セッション 5分オリエンテーション用 |

---

*管理者: 男座員也（Kazuya Oza） / 本ファイルは新セッション開始時のオリエンテーションを目的とする。数値や戦略採否の一次根拠は CURRENT_BEST_STRATEGY.md / STRATEGY_REGISTRY.md / EVALUATION_STANDARD.md にあり、本ファイルはそれらへのナビゲーションを兼ねた要約である。*
