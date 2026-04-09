# プロジェクトコンテキスト圧縮版

## 1. プロジェクト概要
- **目的:** 3倍レバレッジNASDAQ投資戦略のバックテスト研究
- **データ:** NASDAQ Composite 1974-2021年（47年間、11,943日）
- **制約:** 取引回数 ≤100回（年2回以下）
- **評価重視:** CAGR、最悪5年平均リターン、Sharpe

## 2. ファイル構成
```
nasdaq_backtest/
├── NASDAQ_Dairy_since1973.csv       # 元データ
├── 3x_NASDAQ_Strategy_Research_Summary.md  # R1-R3研究（255戦略）
├── STRATEGY_RESEARCH_PLAN_R4.md     # R4検証計画
├── R4_results.csv                   # R4検証結果（55戦略）
├── R4_RESULTS_SUMMARY.md            # R4サマリー
├── regime_strategy_results.csv      # Regime戦略検証
└── src/
    ├── backtest_engine.py           # バックテストエンジン
    ├── run_r4_backtest.py           # R4実行スクリプト
    └── test_regime_strategy.py      # Regime戦略テスト
```

## 3. 既存研究（R1-R3）の主要発見
- **ベースライン:** DD(-18/92)+VT_E(25%,S10) → Sharpe 0.856, MaxDD -62%, 30回
- **最良:** DD(-15/90)+VT_E(25%,S10) → Sharpe ~1.9, MaxDD -38%（元研究）
- **EWMA Span=10 + Target Vol 25%** が最適組み合わせ
- **レジームフィルター（MA200）** は高Sharpeだが取引200回超で失格

## 4. R4検証結果（新規55戦略）

### ベースライン対比改善Top 3:
| 戦略 | CAGR | Sharpe | MaxDD | 取引 | 改善 |
|------|------|--------|-------|------|------|
| DD+VT+VolSpike(1.5x) | 31.45% | 0.896 | -58.51% | 30 | Sharpe +4.7% |
| DD+VT+Q3Q4weak(0.8x) | 29.98% | 0.894 | -54.73% | 30 | MaxDD +12% |
| DD+AsymLev(up30,dn20) | 30.24% | 0.863 | -59.02% | 30 | バランス |

### 失格戦略:
- **DD+Regime(MA200)+VT:** Sharpe 0.93だが取引213回 ✗
- **TripleDD:** 過剰反応、Sharpe低下

## 5. 推奨実装順位
1. **VolSpike(1.5x)** - Vol前日比1.5倍超でLev半減（実装容易）
2. **Q3Q4weak(0.8x)** - 7-10月Lev×0.8（季節性ルール）
3. **AsymLev** - 上昇トレンドTV=30%、下落TV=20%

## 6. 主要コード関数
```python
# backtest_engine.py内
calc_dd_signal()           # DD制御シグナル
calc_ewma_vol()            # EWMAボラティリティ
calc_vt_leverage()         # VTレバレッジ（max_lev=1.0）
run_backtest()             # バックテスト実行
calc_metrics()             # 指標計算（CAGR,Sharpe,MaxDD,Worst5Y等）

# 主要戦略関数
strategy_baseline_dd_vt()  # ベースライン
strategy_dd_vt_volspike()  # Vol Spike検出
strategy_dd_quarterly_filter()  # 季節性フィルター
strategy_asymmetric_leverage()  # 非対称レバレッジ
strategy_dd_regime_vt()    # DD+Regime+VT（失格）
```

## 7. 次のアクション候補
- [ ] Top戦略のアウトオブサンプルテスト（2021-2024）
- [ ] 取引コスト感度分析
- [ ] パラメータ感度分析
- [ ] GAS/実運用コード作成
