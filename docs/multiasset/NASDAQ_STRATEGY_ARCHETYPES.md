# NASDAQ タイミング戦略アーキタイプ目録（Gold/Bond 移植元）

作成日: 2026-06-08
最終更新日: 2026-06-08

> `STRATEGY_REGISTRY.md` と `src/` を精査して抽出した、NASDAQで試行済みのタイミング戦略“型”の一覧。
> これらを Gold/Bond に移植し、多層（3〜5レイヤー）合成で検証する（`MULTIASSET_GOLD_BOND_SIGNAL_PLAN_20260608.md` Phase 2 拡張）。

## レイヤー型（合成可能）
| 型 | NASDAQ実装 | 内容 | Gold/Bond移植 |
|---|---|---|---|
| **ボラ・ターゲティング** | P2_VolTarget, S2 `tv`, F6_vol_scale | target_vol/realized_vol で配分スケール | ✅ `vol_target_scale` |
| **VZ-gate（ボラゾーン）** | S2_VZGated (tv0.8,k_vz0.3,gate_min0.5), A1 n_vol, A3 k_vz, A4 gate_min | ボラzが高い局面で退避 | ✅ `vol_regime_gate` |
| **長期トレンド・フィルタ** | LT2/LT4/LT6 (N×k_lt), B6 N-sweep | N日MA上抜けで保有 | ✅ `ma_cross_position`(既) |
| **dual-MA クロス** | LT7 (N_short750×N_long1250) | 短期>長期MAで保有 | ✅ `dual_ma_position` |
| **ブレイクアウト（タートル）** | turtle_core/t1/t2 (Donchian) | N日高値更新で入る/N日安値で出る | ✅ `donchian_breakout_position` |
| **多ホライズン・モメンタム** | P3_momentum_sweep | 複数lookbackのモメンタム | ✅ `momentum_position`(既, 多N) |
| **レジーム・ゲート** | E4 Regime k_lt, a3_regime_tilt, f8_regime_tilt | レジーム別に配分（攻守） | ✅ マクロgate（資産別） |
| **デッドバンド** | F10 ε-deadband, C2 adaptive | 微小変化を無視し回転抑制 | ✅ `deadband` |
| **ヒステリシス** | AsymmHysteresis (enter≥0.7/exit≤0.3) | 非対称閾値で flip-flop 抑制 | ✅ `hysteresis` |
| **アンサンブル/投票** | E1_ensemble, Ens2, majority_vote | 複数信号の多数決/平均 | ✅ `ensemble_vote`(配分層で) |
| **Yang-Zhang ボラ** | C3_yang_zhang_vol | OHLCベース高精度ボラ | （日次終値のみのため簡易ボラ採用） |

## 棄却済みの教訓（重複回避）
- **Kelly(P5)/確信度直変換(S1)/多因子乗算(P4)** は NASDAQ で IS-OOS gap 過大により棄却。Gold/Bondでも過学習警戒。
- **離散レバ/単純倍(Marugoto)** は連続に劣後。
- 単層では弱く、**VZ-gate+LT+regime の3層スタックが最良**だった事実が「3〜5層で最良」という今回方針の根拠。

## 検証方針
1〜5層の合成を総当たりし、全期間9指標で評価 → 上位を WFA＋ブートストラップ（`run_phase_d`）で確定。
