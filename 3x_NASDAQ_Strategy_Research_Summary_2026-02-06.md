# 3倍レバレッジNASDAQ投資戦略 バックテスト研究サマリー

> **目的:** このドキュメントは、3倍レバレッジNASDAQ100投資戦略の体系的バックテスト研究（R1〜R3、合計255戦略）の全成果を要約したものです。他のスレッドでこの文脈を引き継いで追加検証を行うためのリファレンスとして使用してください。

---

## 1. 研究概要

| 項目 | 内容 |
|------|------|
| **対象指数** | NASDAQ Composite（1973年〜2021年、約47年間） |
| **レバレッジ** | 3倍（日次リバランス、コスト0.9%/年含む） |
| **ベースライン戦略** | DD-18/92/200day（ドローダウン制御） |
| **テスト総数** | R1: 32戦略 → R2: 75戦略 → R3: 148戦略 = **合計255戦略** |
| **最終制約条件** | 総取引回数 ≤60回（47年間で年1.3回以下） |
| **データソース** | Yahoo Finance (^IXIC) |
| **実行環境** | Python (pandas, numpy) |

---

## 2. ベースライン：DD-18/92戦略の仕組み

```
ドローダウン制御（DD制御）:
- 200日間の最高値からの下落率を監視
- 下落率が-18%以下 → 全ポジション売却（CASH状態へ）
- 最高値の92%まで回復 → ポジション復帰（HOLD状態へ）

パラメータ:
  exit_threshold = 0.82   # 最高値から-18%で退避
  reentry_threshold = 0.92 # 最高値の92%で復帰
  lookback = 200          # 200日間の最高値を追跡
```

**ベースライン性能（DD-18/92 3x）:**
- CAGR: 31.67%, MaxDD: -44.52%, Sharpe: 0.999, 取引回数: 29回

---

## 3. 各ラウンドの検証カテゴリと主要発見

### Round 1（32戦略）：ボラティリティ管理の発見

**テストカテゴリ:**
1. ボラティリティターゲティング（VT）：レバレッジを動的に調整
2. ボラティリティスイッチ：高Vol時にポジション削減
3. 季節性フィルター：月別リターンパターン活用
4. RSI平均回帰：買われすぎ/売られすぎ判定
5. DD+VT複合：DD制御＋ボラティリティターゲティング
6. BH（Buy & Hold）ベースライン

**R1最良戦略:**
| 戦略 | CAGR | MaxDD | Sharpe | 取引回数 |
|------|------|-------|--------|----------|
| DD+VT(30%,LB20) | 33.58% | -44.52% | 1.298 | 29 |

**R1の重要発見:**
- ボラティリティ管理が3倍レバレッジの鍵（Sharpe +30%改善）
- 単純Vol計算（ルックバック20日）でも効果あり
- Target Vol 30%が最適（20%は保守的すぎ、40%は攻撃的すぎ）

### Round 2（75戦略）：EWMAボラティリティの突破

**テストカテゴリ:**
1. EWMA（指数加重移動平均）ボラティリティ
2. 段階的レバレッジ（Tiered Leverage）
3. レジームフィルター（MA200ベース）
4. ドローダウンスピード検出
5. MAE（Maximum Adverse Excursion）
6. Vol-of-Vol（ボラティリティのボラティリティ）
7. DD閾値最適化
8. 複合（DD+Regime+VT）
9. レバレッジキャップ付きVT
10. ベースライン比較

**R2最良戦略:**
| 戦略 | CAGR | MaxDD | Sharpe | 取引回数 |
|------|------|-------|--------|----------|
| DD+Regime(MA200)+VT(30%) | 42.15% | -38.90% | 1.929 | 213 ✗ |
| DD+VT_EWMA(30%,S10) | 40.15% | -41.42% | 1.670 | 29 ✓ |
| DD+VT_EWMA(25%,S15) | 37.82% | -38.27% | 1.648 | 29 ✓ |

**R2の重要発見:**
- **EWMA Span=10がSimple Volを大幅に上回る**（Sharpe +28.6%）
- レジームフィルター（MA200）は高Sharpeだが取引回数213回で実用不可
- Triple Layer（DD+Regime+VT）は過学習リスク大

### Round 3（148戦略）：低回転率制約下の最適化

**テストカテゴリ（全10種）:**

| カテゴリ | 戦略数 | 概要 |
|----------|--------|------|
| A. DDパラメータ最適化+EWMA VT | 20 | Exit/Reentry閾値のグリッドサーチ |
| B. 確認レジーム（Confirmed Regime） | 13 | N日連続でMA超過を要求 |
| C. マージンレジーム（Margin Regime） | 18 | MAからN%乖離で切替 |
| D. 適応的DD（Adaptive DD） | 16 | Vol状態でDD閾値を動的調整 |
| E. デュアルタイムフレームDD | 24 | 短期(100日)+長期(200日)DD併用 |
| F. アンサンブル | 6 | 複数戦略のレバレッジ平均 |
| G. モメンタムフィルター | 6 | N日リターン>0で投資 |
| H. R1/R2参照戦略 | 6 | 取引回数監査用 |
| I. カスタムDD+EWMA VT | 20 | DDパラメータ空間の網羅探索 |
| J. ベースライン | 2 | BH 3x, DD-18/92 3x |

---

## 4. 最終結果：Top 15戦略（取引回数≤60）

| 順位 | 戦略 | CAGR | MaxDD | Sharpe | Sortino | Calmar | 取引数 | 回/年 |
|------|------|------|-------|--------|---------|--------|--------|-------|
| 1 | DD(-15/90)+VT_E(25%,S10) | 39.02% | -38.23% | 1.903 | 2.727 | 1.021 | 51 | 1.1 |
| 2 | DD(-15/92)+VT_E(25%,S10) | 37.47% | -33.79% | 1.858 | 2.619 | 1.109 | 43 | 0.9 |
| 3 | DD(-18/92)+VT_E(25%,S10) | 37.41% | -37.32% | 1.806 | 2.607 | 1.002 | 29 | 0.6 |
| 4 | DD(-19/92)+VT_E(25%,S10) | 37.23% | -44.29% | 1.777 | 2.592 | 0.841 | 23 | 0.5 |
| 5 | DD(-15/90)+VT_E(30%,S10) | 42.07% | -42.76% | 1.768 | 2.468 | 0.984 | 51 | 1.1 |
| 6 | Ens2(EWMA30S10+Adapt) | 41.10% | -41.62% | 1.757 | 2.432 | 0.988 | 29 | 0.6 |
| 7 | Ens2(EWMA25+Adapt) | 38.22% | -39.60% | 1.749 | 2.397 | 0.965 | 29 | 0.6 |
| 8 | DD(-18/95)+VT_E(25%,S10) | 35.95% | -34.07% | 1.747 | 2.500 | 1.055 | 29 | 0.6 |
| 9 | DualDD(100/200,E87/85)+VT(25%) | 34.92% | -30.56% | 1.740 | 2.324 | 1.143 | 59 | 1.3 |
| 10 | DualDD(120/200,E87/85)+VT(25%) | 34.92% | -30.56% | 1.740 | 2.324 | 1.143 | 59 | 1.3 |
| 11 | DD(-15/90)+VT_E(25%,S15) | 36.00% | -37.54% | 1.737 | 2.394 | 0.959 | 51 | 1.1 |
| 12 | AdaptDD(E-18,Vs0.5)+VT_E(25%) | 35.66% | -38.26% | 1.727 | 2.383 | 0.932 | 37 | 0.8 |
| 13 | DD+MargReg(MA200,3%)+VT(25%) | 33.34% | -32.79% | 1.708 | 2.224 | 1.017 | 59 | 1.3 |
| 14 | Ens3(EWMA25+DD15+Adapt) | 38.07% | -38.25% | 1.701 | 2.313 | 0.995 | 29 | 0.6 |
| 15 | DD+VT_EWMA(30%,S10) [R2] | 40.15% | -41.42% | 1.670 | 2.349 | 0.969 | 29 | 0.6 |

**ベースライン比較:**
| 戦略 | CAGR | MaxDD | Sharpe | 取引数 |
|------|------|-------|--------|--------|
| BH 3x（何もしない） | 24.57% | -99.95% | 0.518 | 0 |
| DD-18/92 3x（制御のみ） | 31.67% | -44.52% | 0.999 | 29 |
| DD+VT(30%,LB20) [R1最良] | 33.58% | -44.52% | 1.298 | 29 |

---

## 5. 危機年パフォーマンス（Top 5戦略）

| 戦略 | 1974 | 1987 | 2000 | 2001 | 2002 | 2008 | 2011 | 2020 |
|------|------|------|------|------|------|------|------|------|
| DD(-15/90)+VT_E(25%,S10) | 0% | 49.1% | 9.6% | 0% | 0% | -5.1% | -2.6% | 93.2% |
| DD(-15/92)+VT_E(25%,S10) | 0% | 41.8% | 9.6% | 0% | 0% | -5.1% | -3.1% | 86.7% |
| DD(-18/92)+VT_E(25%,S10) | 0% | 57.4% | 6.3% | 0% | 0% | -9.8% | -3.0% | 86.7% |
| DualDD+VT(25%) | 0% | 38.0% | 8.7% | 0% | 0% | -4.1% | -2.6% | 71.2% |
| Ens2(EWMA30S10+Adapt) | 0% | 51.4% | 17.2% | 0% | 0% | -6.2% | 1.2% | 106.6% |
| DD-18/92 3x [ベースライン] | 0% | 47.2% | -11.1% | 0% | 0% | -28.2% | -41.4% | 114.9% |
| BH 3x [ベースライン] | -8.8% | -33.8% | -89.6% | -63.8% | -78.4% | -86.9% | -25.3% | 90.1% |

**危機年での主要ポイント:**
- 全エリート戦略が2000-2002年のドットコムバブル崩壊を完全回避（0%損失）
- 2008年金融危機：-4.1%〜-9.8%（DDベースライン -28.2%と比較して大幅改善）
- 2011年：-0.5%〜-3.0%（DDベースライン -41.4%と比較）
- DualDDが危機時の最高防御力（2008年 -4.1%）

---

## 6. 7つの主要発見

### 発見1：EWMA Span=10 + Target Vol 25%が最適組み合わせ
```
R1: Simple Vol(LB20) + TV=30% → Sharpe 1.298
R2: EWMA Vol(S10) + TV=30% → Sharpe 1.670 (+28.6%)
R3: EWMA Vol(S10) + TV=25% → Sharpe 1.806 (+39.1%)
```
- EWMA Span=10が全戦略でSpan=15を一貫して上回る
- Target Vol 25%はCAGRを若干犠牲にしMaxDDを大幅改善

### 発見2：DD Exit -15%はSharpe改善するが取引倍増
```
DD(-18/92): 29回, Sharpe 1.806
DD(-15/92): 43回, Sharpe 1.858 (+2.9%)
DD(-15/90): 51回, Sharpe 1.903 (+5.4%)
```
- Exit -15%は-18%より3%早く退避 → 暴落初期保護
- 取引回数が29→51（+76%）だが60回制約内
- ⚠ パラメータ最適化による過学習リスクあり

### 発見3：デュアルタイムフレームDDが最低MaxDD（-30.56%）達成
```
Single DD(-18/92):              MaxDD -37.32%
DualDD(100/200,E87/85): MaxDD -30.56% (+6.8%改善)
```
- 短期DD（100日）：急落検出
- 長期DD（200日）：構造的悪化検出
- 2つの独立タイムフレームが相補的保護

### 発見4：アンサンブル戦略が分散効果を提供
```
Ens2(EWMA30S10+Adapt): Sharpe 1.757, CAGR 41.1%, 29回
```
- 複数独立戦略のレバレッジ平均化
- 単一戦略パラメータ依存リスクの低減
- 過学習リスクが最も低いアプローチの一つ

### 発見5：R2のDD+Regime+VT（Sharpe 1.929）は取引回数で失格
```
R2最高Sharpe: DD+Regime(MA200)+VT(30%) Sharpe 1.929 → 213回 ✗
```
- MA200クロスが年4-5回発生 → 47年で200回超
- レジーム戦略は低回転率制約と根本的に非互換

### 発見6：実装推奨ランキング

| 分類 | 戦略 | Sharpe | MaxDD | 取引数 | 変更内容 |
|------|------|--------|-------|--------|----------|
| 最も堅実 | DD(-18/92)+VT_E(25%,S10) | 1.806 | -37.3% | 29 | VT計算のみ変更 |
| 最高Sharpe | DD(-15/90)+VT_E(25%,S10) | 1.903 | -38.2% | 51 | DD+VT変更 |
| 最低リスク | DualDD(100/200)+VT(25%) | 1.740 | -30.6% | 59 | DualDD実装 |
| 分散型 | Ens2(EWMA30S10+Adapt) | 1.757 | -41.6% | 29 | 2戦略並列 |

### 発見7：R1→R2→R3の改善パス
```
R1 DD+VT(30%,LB20):          Sharpe 1.298  MaxDD -44.5%  29回
    ↓ EWMA化 (+28.6%)
R2 DD+VT_EWMA(30%,S10):      Sharpe 1.670  MaxDD -41.4%  29回
    ↓ TV=25% + S10 (+8.1%)
R3 DD(-18/92)+VT_E(25%,S10): Sharpe 1.806  MaxDD -37.3%  29回
    ↓ DD最適化 (+5.4%)
R3 DD(-15/90)+VT_E(25%,S10): Sharpe 1.903  MaxDD -38.2%  51回

合計改善: Sharpe +46.6%, MaxDD +6.3%
```

---

## 7. 実装コード（主要戦略）

### 7.1 EWMA ボラティリティターゲティング

```python
import numpy as np
import pandas as pd

def calc_ewma_vt_leverage(close, target_vol=0.25, ewma_span=10, max_leverage=3.0):
    """EWMA Volatility Targeting: レバレッジを動的に調整"""
    returns = close.pct_change()
    ewma_var = returns.ewm(span=ewma_span).var()
    ewma_vol = np.sqrt(ewma_var * 252)  # 年率換算
    
    # Target Vol / 実現Vol でレバレッジ決定
    leverage = (target_vol / ewma_vol).clip(0, max_leverage)
    return leverage
```

### 7.2 DD制御（パラメータ可変）

```python
def calc_dd_signal(close, exit_threshold=0.82, reentry_threshold=0.92, lookback=200):
    """
    ドローダウン制御シグナル
    exit_threshold: 最高値からの比率（0.82 = -18%で退避）
    reentry_threshold: 最高値からの比率（0.92 = 92%回復で復帰）
    """
    peak = close.rolling(lookback).max()
    ratio = close / peak
    
    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'
    
    for i in range(lookback, len(close)):
        if state == 'HOLD' and ratio.iloc[i] <= exit_threshold:
            state = 'CASH'
        elif state == 'CASH' and ratio.iloc[i] >= reentry_threshold:
            state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0
    
    return position
```

### 7.3 デュアルタイムフレームDD

```python
def calc_dual_dd_signal(close, 
                         short_lookback=100, long_lookback=200,
                         short_exit=0.87, long_exit=0.85,
                         reentry=0.92):
    """
    2つの時間軸でDD監視
    - EXIT: どちらかが閾値を下回ったら退避
    - REENTRY: 両方が回復したら復帰
    """
    peak_short = close.rolling(short_lookback).max()
    peak_long = close.rolling(long_lookback).max()
    ratio_short = close / peak_short
    ratio_long = close / peak_long
    
    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'
    
    for i in range(long_lookback, len(close)):
        if state == 'HOLD':
            if ratio_short.iloc[i] <= short_exit or ratio_long.iloc[i] <= long_exit:
                state = 'CASH'
        elif state == 'CASH':
            if ratio_short.iloc[i] >= reentry and ratio_long.iloc[i] >= reentry:
                state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0
    
    return position
```

### 7.4 適応的DD（ボラティリティ感応型）

```python
def calc_adaptive_dd_signal(close, base_exit=0.82, vol_sensitivity=0.5,
                             reentry=0.92, lookback=200, vol_lookback=60):
    """
    高Vol時 → 閾値を引き上げ（早めに退避）
    低Vol時 → 標準閾値（不要な退避を回避）
    """
    returns = close.pct_change()
    realized_vol = returns.rolling(vol_lookback).std() * np.sqrt(252)
    median_vol = realized_vol.rolling(500).median()  # 長期中央値
    
    # Vol比率に応じて閾値を調整
    vol_ratio = realized_vol / median_vol
    dynamic_exit = base_exit + vol_sensitivity * (vol_ratio - 1) * (1 - base_exit)
    dynamic_exit = dynamic_exit.clip(base_exit - 0.05, base_exit + 0.10)
    
    peak = close.rolling(lookback).max()
    ratio = close / peak
    
    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'
    
    for i in range(max(lookback, 500), len(close)):
        if state == 'HOLD' and ratio.iloc[i] <= dynamic_exit.iloc[i]:
            state = 'CASH'
        elif state == 'CASH' and ratio.iloc[i] >= reentry:
            state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0
    
    return position
```

### 7.5 アンサンブル戦略

```python
def calc_ensemble_leverage(close, max_leverage=3.0):
    """2戦略のレバレッジ平均"""
    # 戦略A: DD-18/92 + EWMA VT(30%, S10)
    dd_a = calc_dd_signal(close, exit_threshold=0.82, reentry_threshold=0.92)
    vt_a = calc_ewma_vt_leverage(close, target_vol=0.30, ewma_span=10)
    lev_a = dd_a * vt_a
    
    # 戦略B: Adaptive DD + EWMA VT(30%, S15)
    dd_b = calc_adaptive_dd_signal(close, base_exit=0.82, vol_sensitivity=0.5)
    vt_b = calc_ewma_vt_leverage(close, target_vol=0.30, ewma_span=15)
    lev_b = dd_b * vt_b
    
    # アンサンブル: 単純平均
    ensemble = ((lev_a + lev_b) / 2).clip(0, max_leverage)
    return ensemble
```

### 7.6 バックテストフレームワーク

```python
def backtest_leveraged(close, leverage_series, annual_cost=0.009):
    """
    3倍レバレッジのバックテスト
    - 日次リバランス前提
    - コスト0.9%/年含む
    """
    daily_returns = close.pct_change()
    leveraged_returns = daily_returns * 3  # 3倍レバレッジ
    daily_cost = annual_cost / 252
    
    # 最終レバレッジ適用
    strategy_returns = leverage_series.shift(1) * leveraged_returns - daily_cost
    
    nav = (1 + strategy_returns).cumprod()
    nav = nav.fillna(1)
    
    # 性能指標計算
    total_days = len(nav)
    total_years = total_days / 252
    cagr = (nav.iloc[-1] ** (1 / total_years)) - 1
    
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    annual_ret = strategy_returns.mean() * 252
    annual_vol = strategy_returns.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    
    downside_vol = strategy_returns[strategy_returns < 0].std() * np.sqrt(252)
    sortino = annual_ret / downside_vol if downside_vol > 0 else 0
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'CAGR': cagr, 'MaxDD': max_dd, 'Sharpe': sharpe,
        'Sortino': sortino, 'Calmar': calmar, 'NAV': nav
    }

def count_trades(position_series):
    """取引回数カウント（バイナリ状態遷移のみ）"""
    changes = position_series.diff().abs()
    trades = (changes > 0.5).sum()  # HOLD↔CASH遷移のみ
    return trades
```

---

## 8. 取引回数の定義（重要）

```
「取引」= バイナリ状態遷移（投資中↔現金）
- DD制御シグナルの変化（HOLD→CASH or CASH→HOLD）のみカウント
- VTによるレバレッジ日次調整は取引としてカウントしない
- SBI証券の実際の執行に対応：取引はDDがポジション変更をトリガーした時のみ発生
```

---

## 9. カテゴリ別ベスト戦略

| カテゴリ | ベスト戦略 | Sharpe | 評価 |
|----------|-----------|--------|------|
| DD(Custom)+EWMA VT | DD(-15/90)+VT_E(25%,S10) | 1.903 | ★★★★★ |
| アンサンブル(2) | Ens2(EWMA30S10+Adapt) | 1.757 | ★★★★ |
| デュアルDD+VT | DualDD(100/200,E87/85)+VT(25%) | 1.740 | ★★★★ |
| 適応的DD+VT | AdaptDD(E-18,Vs0.5)+VT(25%) | 1.727 | ★★★★ |
| DD+マージンレジーム | DD+MargReg(MA200,3%)+VT(25%) | 1.708 | ★★★ |
| アンサンブル(3) | Ens3(EWMA25+DD15+Adapt) | 1.701 | ★★★ |
| DD+EWMA VT [R2] | DD+VT_EWMA(30%,S10) | 1.670 | ★★★ |
| DD+確認レジーム | DD+ConfReg(MA300,C5)+VT(25%) | 1.617 | ★★★ |
| DD+VT Simple [R1] | DD+VT(30%,LB20) | 1.298 | ★★ |

---

## 10. 失格戦略（取引回数>60で除外）

| 戦略 | Sharpe | 取引数 | 除外理由 |
|------|--------|--------|----------|
| DD+Regime(MA200)+VT(30%) | 1.929 | 213 | MA200クロス頻発 |
| DD+Regime(MA200)+VT(40%) | 1.837 | 213 | 同上 |
| DD+MAE(10d,-3%) | 1.591 | 121 | MAEシグナル頻発 |
| モメンタムフィルター全般 | — | 190+ | 根本的に高回転率 |

---

## 11. 未検証の追加テスト候補

以下は今後のラウンドで検証可能な項目：

1. **アウトオブサンプルテスト**（2021-2024年データ）
2. **取引コスト感度分析**（0.1%, 0.2%, 0.5%）
3. **他資産クラスへの適用**（S&P 500, Russell 2000）
4. **パラメータ感度分析**（DD閾値の±1%変動影響）
5. **ウォークフォワード検証**（ローリング最適化）
6. **GAS（Google Apps Script）実装コード**
7. **レジーム検出の改良**（Volベースのみでレジーム判定）
8. **月次リバランス版**（日次→月次でのコスト削減効果）
9. **リスクパリティアプローチ**（他資産との組み合わせ）
10. **テールリスクヘッジ**（VIX連動やオプション的保護）

---

## 12. 実行環境・前提条件

```
Python 3.x
ライブラリ: pandas, numpy, yfinance (データ取得用)
データ: Yahoo Finance ^IXIC (NASDAQ Composite)
期間: 1973-01-02 〜 2021-12-31（約47年）
レバレッジ: 日次3倍リバランス（信託報酬0.9%/年控除後）
無リスク金利: 0%（Sharpe計算時）
```

---

## 13. 注意事項・制限

- **過学習リスク:** DD閾値の最適化（特にExit -15%）はイン・サンプル最適化であり、将来の性能を保証しない
- **3倍レバレッジの前提:** 日次リバランス型ETF/投信のボラティリティドラッグを含む
- **取引コスト:** バックテストでは信託報酬のみ考慮。売買手数料・スプレッドは未反映
- **生存者バイアス:** NASDAQインデックスのみ検証。個別銘柄の除外・追加影響は未反映
- **流動性リスク:** 暴落時の約定価格ずれ（スリッページ）は未モデル化
- **VT実装の現実性:** 日次レバレッジ調整は投信の買増/売却で実現するため、厳密な日次調整は困難。月次/週次の近似実装が現実的
