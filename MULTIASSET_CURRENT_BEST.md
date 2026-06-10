# MULTIASSET CURRENT BEST — 3資産（NASDAQ/Gold/Bond）最終構成（単一の真実）

作成日: 2026-06-08
最終更新日: 2026-06-10

> ⚠⚠ **2026-06-10 重大訂正**: 本書の統合結果（税後CAGR 最大+8.9%）は **ベンチマーク DH-W1（min税後CAGR +18.10%）に大きく未達＝不採用**。
> 本タスクの本質は「**NASDAQ/GOLD/BOND/CASH 配分 × レバレッジ の2軸最適化でDH-W1を超える**」こと（下記は超えていない）。
> **正典扱いしないこと。** 正しい計画は `MULTIASSET_2AXIS_OPTIMIZATION_PLAN_20260609.md`、引き継ぎは `MULTIASSET_SESSION_HANDOFF_20260610.md` を参照。
> 確定シグナル（Gold=m252_tv0.10_z0.75_mo / Bond=m252_tv0.05_z1.0_wk）と構築モジュールは再利用可。配分・レバの結論は破棄。

> マルチアセット（NASDAQ/Gold/Bond）タイミング戦略の確定構成。これが単一の真実。
> NASDAQ単独のベストは別途 `CURRENT_BEST_STRATEGY.md`。本書は3資産統合版。
> 計画: `MULTIASSET_GOLD_BOND_SIGNAL_PLAN_20260608.md`。

## 1. 確定スリーブ（資産別シグナル＋商品）

| 資産 | 確定シグナル（1倍タイミング） | 確定商品 | 倍率 | 執行ラグ |
|---|---|---|---|---|
| **NASDAQ** | mom252 × VT(0.15) × VZ（週次リバランス） | TQQQ | 1〜3x（選好） | T+2 |
| **Gold** | **m252_tv0.10_z0.75_mo**（月次リバランス, 年3トレード） | SBI純金1倍投信 | 1x | T+5 |
| **Bond** | **m252_tv0.05_z1.0_wk**（週次リバランス） | TMF | 1x | T+2 |

- Gold/Bond は WFA＋ブートストラップで PASS 済（`PHASE_D_VALIDATION_20260608.md`）。
- Gold は1倍投信がレバ2036(2x)を上回り採用。Bond はレバで減益のため1x（分散役）。
- コスト定数は `src/product_costs.py`（単一の真実, 2036・1倍投信3種を収載）。

## 2. 最終ポートフォリオ（全期間1974-2026・純コスト後・税20.315%後）

> ★ **NASDAQ倍率×配分はリスク選好で選択**。下表から採用構成を1つ決める（要ユーザー判断）。

| 構成(NASDAQ倍率×配分) | CAGR(税後) | Sharpe | MaxDD | Calmar | WFE |
|---|---|---|---|---|---|
| 保守 (NQ@1x × InvVol) | +3.98% | **+1.99** | -9.1% | 0.438 | 0.57 |
| **やや保守 (NQ@1x × Equal)【推奨】** | +5.88% | +1.65 | **-9.1%** | **0.648** | 1.07 |
| 中庸 (NQ@2x × Equal) | +7.47% | +1.20 | -15.8% | 0.471 | 1.06 |
| 攻め (NQ@3x × Equal) | **+8.92%** | +1.00 | -22.4% | 0.398 | 1.04 |
| 参考: 無タイミング等加重B&H | +5.30% | +0.72 | -34.5% | 0.154 | 1.28 |

- **推奨（リスク調整・Calmar最大）= やや保守 (NQ@1x × Equal)**：税後CAGR +5.88% / Sharpe 1.65 / MaxDD −9.1%。
- 無タイミング等加重B&H（Sharpe 0.72 / MaxDD −34.5%）に対し、**同等以上のリターンで DD を 1/4 に圧縮**。
- リターン重視なら攻め(NQ@3x)で税後CAGR +8.92%（MaxDD −22.4%）。

## 3. 再現コマンド

```
PYTHONPATH=src python -m multi_asset.run_integration       # 最終統合（本表）
PYTHONPATH=src python -m multi_asset.run_leverage_decision # 商品・倍率判定
PYTHONPATH=src python -m multi_asset.run_allocation        # 配分ロジック比較
PYTHONPATH=src python -m multi_asset.run_layer_param_opt   # レイヤー最適化
```

## 4. 一次根拠（フェーズ別レポート）

| フェーズ | レポート |
|---|---|
| シグナル探索（単資産） | `BOND_SINGLE_ASSET_SWEEP_20260608.md` / `GOLD_SINGLE_ASSET_SWEEP_20260608.md` |
| 多層化 | `BOND_LAYERED_SWEEP_20260608.md` / `GOLD_LAYERED_SWEEP_20260608.md` |
| パラメータ最適化 | `BOND_LAYER_PARAM_OPT_20260608.md` / `GOLD_LAYER_PARAM_OPT_20260608.md` |
| 正式WFA/bootstrap | `PHASE_D_VALIDATION_20260608.md` |
| 配分 | `MULTIASSET_ALLOCATION_20260608.md` |
| 商品・レバレッジ | `LEVERAGE_DECISION_20260608.md` |
| 最終統合 | `MULTIASSET_INTEGRATED_20260608.md` |

## 5. 未確定・要判断（Open）

- ★ **採用構成の確定**（保守〜攻めのどれか）= ユーザー判断待ち。
- NASDAQスリーブのシグナルは1xタイミングPROXY（mom252×VT×VZ）。確定NASDAQ Active戦略（CFD系）への差し替えは将来課題。
- n_boot は 5000（house標準10000での最終確定は任意）。
