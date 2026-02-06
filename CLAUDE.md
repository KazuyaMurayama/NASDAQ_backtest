# NASDAQ 3x Leveraged Investment Strategy Project

## プロジェクトゴール

1. **より良い投資戦略の発見** - 3倍レバレッジNASDAQ投資における最適な戦略を研究
2. **実行可能なシステムの開発** - GASで自動実行可能なシステムを構築

---

## 研究結果サマリー

### データ
- **期間**: 1974-01-02 〜 2021-05-07（47年間）
- **対象**: NASDAQ Composite Index
- **ファイル**: `NASDAQ_Dairy_since1973.csv`

### 制約条件
- 取引回数 ≤ 100回（47年間）
- 実効レバレッジ上限 3倍（max_lev=1.0）

### 評価指標（重要度順）
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
- Sharpe 1.03超え（従来比+14%）
- Worst5Y が初めてプラス圏（+1.4%）
- MaxDD -48%（従来比+10%改善）

---

## 戦略コンポーネント詳細

### Layer 1: DD Control（ドローダウン制御）
```
目的: 暴落時の致命的損失を回避
ロジック:
  - 200日ローリング高値を追跡
  - 現在価格/高値 ≤ 0.82 → CASH（退出）
  - 現在価格/高値 ≥ 0.92 → HOLD（復帰）
出力: 0（CASH）または 1（HOLD）
```

### Layer 2a: AsymEWMA（非対称EWMAボラティリティ）
```
目的: 下落時に素早くVol上昇を検出
ロジック:
  - 日次リターン < 0: EWMA Span = 5（高速反応）
  - 日次リターン ≥ 0: EWMA Span = 20（低速反応）
  - レバレッジ = min(Target_Vol / AsymEWMA_Vol, max_lev)
```

### Layer 2b: SlopeMult（MA傾き乗数）
```
目的: 取引回数を増やさずにトレンド情報を反映
ロジック:
  - MA200の日次変化率を計算
  - 60日窓でZ-score正規化
  - 乗数 = clip(0.7 + 0.3 × z, 0.3, 1.5)
効果:
  - MA上向き → 乗数 > 1.0 → レバレッジ増加
  - MA下向き → 乗数 < 1.0 → レバレッジ減少
```

### Layer 2c: TrendTV（トレンド連動Target Vol）
```
目的: トレンド強度に応じてTarget Volを動的調整
ロジック:
  - ratio = Price / MA150
  - ratio ≤ 0.85 → TV = 15%（守り）
  - ratio ≥ 1.15 → TV = 35%（攻め）
  - 中間は線形補間
```

---

## 実装済み戦略（GAS実装候補）

### 1. DD+VT+VolSpike(1.5x)【シンプル版】
```python
# 日次判定フロー
1. DD判定: 200日高値比で HOLD/CASH 決定
2. VT計算: leverage = min(0.25 / ewma_vol, 1.0)
3. VolSpike: if today_vol/yesterday_vol > 1.5 → leverage *= 0.5
4. 最終: final = dd_signal × leverage
```

### 2. Ens2(Asym+Slope)【推奨版】
```python
# 日次判定フロー
1. DD判定: 200日高値比で HOLD/CASH 決定
2. AsymVol: span=5(下落時) or 20(上昇時) でEWMA計算
3. VT計算: leverage = min(0.25 / asym_vol, 1.0)
4. SlopeMult: 乗数 = clip(0.7 + 0.3 × z_score, 0.3, 1.5)
5. 最終: final = dd_signal × leverage × slope_mult
```

---

## ファイル構成

```
nasdaq_backtest/
├── CLAUDE.md                    # このファイル（プロジェクト概要）
├── NASDAQ_Dairy_since1973.csv   # 元データ
├── src/
│   ├── backtest_engine.py       # コアエンジン（全戦略実装）
│   ├── test_ens2_strategies.py  # Ens2戦略テスト
│   └── ...
├── R4_results.csv               # R4検証結果（55戦略）
├── ens2_comparison_results.csv  # Ens2比較結果
└── VolSpike_Yearly_Returns.xlsx # 年次リターンExcel
```

---

## 重要な技術的注意点

### max_lev パラメータ
```
max_lev=1.0: 実効レバレッジ 0-3x（推奨、保守的）
max_lev=3.0: 実効レバレッジ 0-9x（高リスク）

※ 元研究(R3)のSharpe 1.8+はmax_lev=3.0の結果
※ 本プロジェクトではmax_lev=1.0を標準採用
```

### 取引回数カウント
```
カウント対象: バイナリ状態遷移のみ（HOLD↔CASH）
レバレッジ変動: カウントしない（連続的調整）
```

### コスト前提
```
年間コスト: 0.9%（経費率、投資時のみ日割り控除）
現金保有時: 0%リターン（金利なし）
```

---

## 次のステップ: GAS実装

### 必要機能
1. **データ取得** - Yahoo Finance API から NASDAQ終値
2. **状態管理** - スプレッドシートに履歴保存
3. **戦略計算** - DD/AsymEWMA/SlopeMult判定
4. **通知** - LINE or Email でシグナル送信
5. **定期実行** - Trigger で毎日自動実行

### 実装難易度
- DD+VT+VolSpike: ★★☆☆☆（簡単）
- Ens2(Asym+Slope): ★★★☆☆（中程度）

---

## 危機年パフォーマンス参考

| 年 | Ens2(Asym+Slope) | DD+VT+VolSpike | Buy&Hold 3x |
|----|------------------|----------------|-------------|
| 1974 | -39.2% | -36.9% | -75.8% |
| 1987 | **+30.7%** | +18.3% | -31.9% |
| 2000-02 | 0% | 0% | -89%〜-78% |
| 2008 | **-13.9%** | -31.1% | -87.6% |
| 2020 | +70.9% | +87.2% | +95.9% |

---

## GitHub
https://github.com/KazuyaMurayama/NASDAQ_backtest
