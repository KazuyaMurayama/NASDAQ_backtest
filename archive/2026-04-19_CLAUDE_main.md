# NASDAQ 3x Leveraged Investment Strategy Project

## プロジェクトゴール

1. **より良い投資戦略の発見** - 3倍レバレッジNASDAQ投資における最適な戦略を研究
2. **実行可能なシステムの開発** - GASで自動実行可能なシステムを構築

---

## 研究結果サマリー

### データ
- **期間**: 1974-01-02 〜 2021-05-07(47年間)
- **対象**: NASDAQ Composite Index
- **ファイル**: `NASDAQ_Dairy_since1973.csv`

### 制約条件
- 取引回数 ≤ 100回(47年間)
- 実効レバレッジ上限 3倍(max_lev=1.0)

### 評価指標(重要度順)
1. **Sharpe Ratio** - リスク調整後リターン
2. **Worst 5Y** - 最悪5年ローリングCAGR
3. **MaxDD** - 最大ドローダウン
4. **CAGR** - 年率複利成長率

---

## トップ戦略ランキング

| 順位 | 戦略 | Sharpe | CAGR | MaxDD | Worst5Y | Trades |
|------|------|--------|------|-------|---------|--------|
| **1** | Ens2(Asym+Slope) | **1.031** | 28.6% | -48.2% | **+1.4%** | 30 |
| **2** | Ens2(Slope+TrendTV) | **1.014** | 28.5% | -48.2% | **+1.4%** | 30 |
| 3 | DD+VT+VolSpike(1.5x) | 0.902 | 31.7% | -58.4% | -7.8% | 30 |
| 4 | DD(-18/92)+VT(25%) | 0.861 | 30.0% | -61.9% | -8.0% | 30 |

### 推奨戦略: **Ens2(Asym+Slope)**
- Sharpe 1.03超え(従来比+14%)
- Worst5Y が初めてプラス圏(+1.4%)
- MaxDD -48%(従来比+10%改善)

---

Note: This is the archived legacy main-branch CLAUDE.md. See CLAUDE.md (current) for the new lightweight rules index.

## GitHub
https://github.com/KazuyaMurayama/NASDAQ_backtest
