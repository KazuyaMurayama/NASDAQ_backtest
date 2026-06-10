# マルチアセット（NASDAQ/Gold/Bond）セッション ハンドオフ

作成日: 2026-06-10
最終更新日: 2026-06-10

> 新セッションはこれを最初に読むこと。本スレッド（3資産タイミング戦略）の発見・現状・次アクションを集約。
> 正典計画: `MULTIASSET_2AXIS_OPTIMIZATION_PLAN_20260609.md`（**最新・有効**）。旧 `MULTIASSET_GOLD_BOND_SIGNAL_PLAN_20260608.md` は単資産検証フェーズの記録。

## ⚠ 最重要（次セッションが最初に直すべきこと）
- **本タスクの本質は「NASDAQ/GOLD/BOND/CASH の配分 × レバレッジ比率」の2軸同時最適化**。CASHを配分軸に含める。
- **ベンチマーク = ETF環境の現行ベスト DH-W1**：min(IS,OOS) **税後CAGR +18.10%** / MaxDD −34.57%（標準10指標は下表）。**これを超えるのが必須**（3軸保守基準：min CAGR・Worst10Y・P10_5Y を同時超え）。CFDのE4 +33.53%は税前なので比較対象外。
- **前セッションの統合結果（`MULTIASSET_INTEGRATED_20260608.md`）は税後CAGR最大+8.9%でベンチマーク未達＝不採用**。`MULTIASSET_CURRENT_BEST.md` の数値もこの未達版なので**正典扱いしない**（2軸最適化で置換予定）。
- 報告形式（ユーザー厳守指定）：①**標準10指標** ②**表形式** ③**優れた値・戦略を太字**（ベンチ行併記）。

### ベンチマーク（DH-W1, 標準10指標, 税後⓽） — 出典 CURRENT_BEST_STRATEGY.md §v4.5
| 戦略 | CAGR min(IS,OOS) | gap | Sharpe | MaxDD | Worst10Y | P10_5Y | Trades/yr | WFE | CI95_lo |
|---|---|---|---|---|---|---|---|---|---|
| **DH-W1（ETF環境 現行ベスト）** | **+18.10%** | −0.81pp | 0.845 | −34.57% | +10.37% | +4.82% | 68.7 | 1.023 | +13.61% |

## 本セッションの発見（リスト）
1. **「ポンド」は「ボンド(Bond/債券)」の聞き違い**。3資産目はBond（英ポンドGBPではない）。GBP用に作った物は全撤去済み。
2. **既存リポは成熟したシグナル基盤を保有**（`src/signals/*`, `src/integration/nine_metric_eval`・`phase_d_*`）。再利用前提。
3. **データは全資産揃っている**：Gold(LBMA 1968+), Bond(`base_dataset.csv` の synth 1974-2009 + IEF 2009+, DGS10 1962+), NASDAQ 1974+。**FRED延長等は不要**。
4. **全期間 vs OOS の整合方法**：全期間(1974-2026=統合レポート§6基準)を主＋WFA(WFE/CI95)で手法整合。単一OOS分割(2021-05-08)は二次参照。
5. **単資産シグナル所見（全期間）**：
   - **Bond はキャッシュ超のリターンを出せない**（P(>cash)<0.9）→ **分散/DD制御役**。レバを上げると financing で減益。
   - **Gold はタイミングが効く**：`gold_realyield_lo`/`gold_mom126` が WFA+bootstrap PASS（cash/B&H超え）。
6. **多層(3-5層)化で Gold は Sharpe~1.0-1.5/MaxDD−14〜20%** に改善（NASDAQ流アーキタイプ移植：vol-target/VZ-gate/dual-MA/Donchian/hysteresis/deadband）。
7. **確定シグナル**（ユーザー承認済）：Gold=`m252_tv0.10_z0.75_mo`（月次・年3トレード）／Bond=`m252_tv0.05_z1.0_wk`。
8. **トレード回数メトリクスのバグ**：旧`nine_metric_eval`のNAV代理は過大計上。**実建玉ベース**(`strategy_layers.trades_per_year`)に置換。定期リバランス(週/月)＋実行ラグで現実化（月次=年3回, 週次=年7-9回）。
9. **商品コスト確定**（`product_costs.py`に収載）：TQQQ(3x,TER0.86%,2×SOFR+0.5%,T+2)／**2036**(Gold2x,0.50%,1×SOFR+0.5%,T+2)／TMF(3x,0.91%,2×SOFR+0.5%,T+2)／1倍投信=SBI NASDAQ100 0.1958%・SBI純金 0.1838%・米国債2255 0.154%（SOFR無, **T+5**）。商品は **TQQQ/2036/TMF**（EDV・UGLは不使用）。
10. **レバレッジの資本効率効果**：レバETFで実効レバk<上限を作ると「資本のk/LをETF＋残りキャッシュ(T-bill)」となり、余剰キャッシュ金利がfinancingを相殺。
11. **Markdown表の再発バグ**：列ラベル内のリテラル`|`が表を破壊（GitHubで段落化）。`report_format._esc()`でエスケープ＋`validate_markdown_tables()`で生成時自己検証（再発防止）。
12. **プロセス教訓**：成果物は「生テキスト目視」でなく**レンダリング/構造を機械検証**してから報告。代理指標(ファイル存在等)で成功宣言しない。

## 構築済みモジュール（テスト 55件 PASS, `tests/multi_asset/`）
- `single_asset_sweep.py`（保有vsキャッシュ評価・全期間9指標）／`bond_signals.py`（因果シグナル）
- `strategy_layers.py`（vol-target/VZ-gate/dual-MA/Donchian/hysteresis/deadband/rebalance/exec-lag/trades_per_year）
- `walkforward.py`（暦年WFA＋定常ブロックbootstrap）／`allocator.py`（equal/inverse-vol/sharpe-tilt）
- `leverage_eval.py`（純コスト後・税後リターン, 実効レバk）／`report_format.py`（太字表＋構造バリデータ）
- 実行: `run_bond_sweep`/`run_gold_sweep`/`run_layered_sweep`/`run_layer_param_opt`/`run_allocation`/`run_leverage_decision`/`run_integration`

## 次アクション（`MULTIASSET_2AXIS_OPTIMIZATION_PLAN_20260609.md` を実行）
1. **Phase 0**: `leverage_eval` に標準10指標一括評価器 `ten_metrics()`（min(IS,OOS)税後CAGR込, split=2021-05-08）を追加（TDD）。
2. **Phase 1**: `portfolio_engine.py` 新規＝4資産(N/G/B/CASH)×per-assetレバの純コスト後・税後ポートNAV。**NASDAQスリーブはDH-W1同等以上を担保**（劣後なら信号改良）。
3. **Phase 2**: `optimize_2axis.py` 新規＝配分グリッド×レバグリッドの結合探索→**DH-W1を3軸同時超え**でフィルタ→フロンティア。動的配分(リスクパリティ/ボラターゲット/レジーム)で上積み。
4. **Phase 3-4**: WFA/bootstrap確定→標準10指標の**太字表**で報告→`MULTIASSET_CURRENT_BEST.md`/`STRATEGY_REGISTRY.md`/`tasks.md`更新。
