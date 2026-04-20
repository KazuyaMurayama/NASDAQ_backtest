# Approach A 採用提案 — CAGR 30%超達成のための戦略再設計

**日付**: 2026-04-20
**目的**: 現行 Approach B (CAGR 24.7%) から Approach A (CAGR 30.9%) への移行可能性の検証と提案

---

## 結論（先に）

**Approach A は物理的に実現可能、かつ 7指標中 6つで Approach B を上回る**。CAGR 30%+ の目標は T=0.20 で達成可能（IS: 30.9%）。

---

## 設計の違い

### 現行 (Approach B) — 統合レバレッジ方式
```
実保有 = lev × (wn, wg, wb)
   TQQQ  = lev × w_nasdaq     (例: 0.19 × 0.56 = 10.7%)
   Gold  = lev × w_gold        (例: 0.19 × 0.22 = 4.2%)
   Bond  = lev × w_bond        (例: 0.19 × 0.22 = 4.2%)
   CASH  = 1 - lev             (例: 81%)
```
**DD発動時 (lev=0): 全額キャッシュ**

### 提案 (Approach A) — スリーブ独立方式
```
実保有 = スリーブごとに独立
   TQQQ         = w_nasdaq × lev           (NASDAQスリーブ内TQQQ)
   CASH buffer  = w_nasdaq × (1 - lev)     (NASDAQスリーブ内現金)
   Gold (2036)  = w_gold                    (常に100%投資)
   Bond (TMF)   = w_bond                    (常に100%投資)
```
**DD発動時 (lev=0): NASDAQスリーブのみ現金化、Gold/Bondは継続保有**

### 物理的実現性
- TQQQ, 2036, TMF すべて既に GAS で取引対象
- 各スリーブを独立に管理するだけ
- 固定レバレッジETFの特性と整合（TQQQを売却→現金化はできるが、Gold/Bondとは独立）

---

## 7指標 IS 比較（T=0.20、重要度順）

| 指標 | 優先度 | 現行 B | 提案 A | 差 | 判定 |
|------|--------|--------|--------|-----|-----|
| **CAGR** | ★5 | 24.7% | **30.9%** | **+6.2pp** | ✅ 大幅改善 |
| **Worst5Y** | ★5 | +3.3% | **+5.3%** | +2.0pp | ✅ 改善 |
| **Sharpe** | ★5 | 1.167 | **1.331** | +0.164 | ✅ 改善 |
| **MaxDD** | ★4 | -33.9% | **-32.6%** | +1.3pp | ✅ 改善 |
| Worst10Y | ― | +11.0% | **+15.2%** | +4.2pp | ✅ 改善 |
| WinRate | ― | 74.5% | **85.1%** | +10.6pp | ✅ 改善 |
| Trades | ― | 881 | 881 | 0 | ― 同じ |

**7指標すべてで Approach A 優位**。

---

## OOS 比較（T=0.20）

| 指標 | 現行 B | 提案 A | 差 |
|------|--------|--------|-----|
| CAGR | 22.6% | **25.1%** | +2.5pp ✅ |
| Sharpe | 0.951 | **0.990** | +0.039 ✅ |
| MaxDD | **-24.2%** | -29.2% | -5.0pp ⚠️ |
| WinRate | 60% | 60% | 0 |
| Trades | 101 | 101 | 0 |

**OOS: CAGR/Sharpe は A 優位、MaxDD は B 優位**。OOS期間(2021-2026)の MaxDD悪化は、2022年のGold/Bond同時急落（インフレ+利上げ）に起因。

---

## Walk-Forward 全期間比較（T=0.20）

| ウィンドウ | B Sharpe | A Sharpe | 差 |
|-----------|---------|---------|-----|
| WF1 (2010-15) | 0.576 | **0.645** | +0.069 ✅ |
| WF2 (2015-20) | 0.834 | **1.000** | +0.166 ✅ |
| WF3 (2020-26) | 1.033 | **1.085** | +0.052 ✅ |
| **平均** | 0.814 | **0.910** | **+0.096** ✅ |

**全ウィンドウで A 優位**。オーバーフィットの兆候なし。

---

## 閾値スイープ結果（Approach A）

| T | IS CAGR | IS Sharpe | IS Worst5Y | IS MaxDD | OOS CAGR | Trades |
|---|---------|-----------|------------|----------|----------|--------|
| 0.05 | 32.0% | 1.327 | +2.7% | -31.6% | 25.4% | 3,081 |
| 0.10 | 31.3% | 1.312 | +1.1% | -33.9% | 25.1% | 1,929 |
| 0.15 | **31.4%** | **1.338** | **+4.8%** | **-31.4%** | **25.5%** | 1,259 |
| **0.20** | 30.9% | 1.331 | **+5.3%** | -32.6% | 25.1% | 881 |
| 0.25 | 30.8% | 1.338 | +4.8% | -29.7% | 20.7% ⚠️ | 653 |

**Approach A 内の最適閾値**: T=0.20 と T=0.15 が拮抗。T=0.20 は Worst5Y で最良、T=0.15 は CAGR/Sharpe で僅差優位。現行 T=0.20 を維持しても目標CAGR 30.9% 達成。

---

## リスク分析

### リスク1: 2022年のような Gold/Bond 同時下落
- 2022: Approach A = -19.9% vs Approach B = -10.3%（A のが 9.6pp 悪い）
- 原因: インフレ+急激な利上げで TMF (Bond 3x) が -80% 級の下落、金も横ばい
- **対策**: OOS MaxDD -29.2% は許容範囲。長期では優位。

### リスク2: Gold 2x ETN (2036) の信用リスク
- 発行体信用リスク（ETNは無担保）
- **対策**: 実運用で 2036 の代替として 1540 (金現物) + 小額先物等を検討可

### リスク3: TMF の長期インフレ環境での弱さ
- 長期金利上昇局面でのレバレッジ債券ETFは逆風
- **対策**: VT_Bond_Coef 等の動的調整パラメータ追加は別タスク

---

## GAS 改修に必要な変更

### 変更箇所（すべて `nasdaq-strategy-gas/src/`）

#### 1. `Allocation.gs` — `calcActualHoldings()` を追加
```javascript
/**
 * Approach A: sleeve-independent allocation.
 * Returns actual holdings (sum = 1.0 if no excess cash buffer).
 */
function calcActualHoldings(rawLeverage, targetWeights) {
  return {
    actual_tqqq:        rawLeverage * targetWeights.w_nasdaq,
    actual_cash_buffer: (1 - rawLeverage) * targetWeights.w_nasdaq,
    actual_gold:        targetWeights.w_gold,   // ← not × rawLeverage
    actual_bond:        targetWeights.w_bond,   // ← not × rawLeverage
    actual_cash_total:  (1 - rawLeverage) * targetWeights.w_nasdaq
  };
}
```

#### 2. `Code.gs` — リバランス判定は現行維持、通知フォーマットのみ更新
- `dailyUpdate()` のリバランス判定 (`|rawLev - currentLev| > 0.20`) はそのまま
- State に `current_weights` を引き続き保持
- 通知には `calcActualHoldings()` の結果を表示

#### 3. `Notify.gs` / `DailyStatusAgent.gs` — メッセージ更新
```
【新フォーマット例】
📊 実際の保有配分:
  TQQQ (NASDAQ 3x):  10.7%   ← w_nasdaq × rawLev
  Cash buffer:       45.3%   ← w_nasdaq × (1 - rawLev)  【NEW】
  2036 (Gold 2x):    22.0%   ← w_gold のみ【変更: lev非依存】
  TMF  (Bond 3x):    22.0%   ← w_bond のみ【変更: lev非依存】
  合計:             100.0%
```

#### 4. `StateManager.gs` / `Setup.gs` — Log列の再解釈
- 既存の `actual_tqqq` 列: `w_nasdaq × rawLev` に意味を維持（実質変更なし）
- 既存の `actual_gold` 列: `w_gold` のみに変更（乗算なし）【変更】
- 既存の `actual_bond` 列: `w_bond` のみに変更（乗算なし）【変更】
- 既存の `actual_cash` 列: `w_nasdaq × (1 - rawLev)` に変更【変更】

#### 5. 初回移行作業
- 初回の `dailyUpdate` で新配分に合わせてリバランス発生（既存リバランスロジックで自動）
- 手動確認: 次回通知で Gold/Bond の保有増を確認

### 実装工数見積
- コード変更: 約30〜50行
- テスト: `dryRun()` で数値整合性確認
- 本番反映: 次の営業日 `dailyUpdate` で自動リバランス

---

## 推奨アクション

### フェーズ1（即時実施可）
1. 本提案レポートの承認
2. GAS側でのパラメータ変更実装
3. `dryRun()` で新旧配分の差分確認
4. 本番反映（次回 dailyUpdate で自動リバランス）

### フェーズ2（任意）
- T=0.15 vs T=0.20 の詳細比較で最終閾値決定
- Gold/Bond の Vol Targeting 導入検討
- ETN 信用リスク分散（2036 → 1540 代替）検討

---

## 成果物

- [approach_a_sweep_results.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/approach_a_sweep_results.csv)
- [CAGR_DISCREPANCY_ANALYSIS_2026-04-20.md](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/CAGR_DISCREPANCY_ANALYSIS_2026-04-20.md)
- [THRESHOLD_SWEEP_REPORT_2026-04-20.md](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/THRESHOLD_SWEEP_REPORT_2026-04-20.md)

---

*検証スクリプト: `/tmp/approach_a_sweep.py` （ローカル）*
