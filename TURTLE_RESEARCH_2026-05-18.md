# タートル流投資手法 仕様調査
作成日: 2026-05-18
最終更新日: 2026-05-18

調査対象: カーティス・フェイス著「Way of the Turtle」(2007) および Curtis Faith 公開の "Original Turtle Trading Rules" PDF に基づく純正タートル・ルール。改変版・派生版は除外。

---

## 概要

タートル・トレーディング・システムは、1983年にリチャード・デニスとウィリアム・エックハートが考案したトレンドフォロー手法。短期(System 1: 20日ブレイクアウト)と長期(System 2: 55日ブレイクアウト)の2システムを使用。ボラティリティ(ATR = "N")に基づくポジションサイジング、1/2N間隔のピラミッディング、2N逆行でのストップロス、複数市場にまたがるリスク制限を組み合わせた定量的・完全ルールベースの手法。商品先物市場向けに設計されたが、スポット市場・ETF等へも原則転用可能。

---

## 1. System 1 完全仕様

### 1.1 エントリー条件

```
Long  Entry: 当日価格が直近20日間の最高値を1ティック超えた時点
Short Entry: 当日価格が直近20日間の最安値を1ティック下回った時点
```

**タイミング**: 原則はブレイクアウト発生「時点」（イントラデイ）。一般的な解釈では「ブレイクした足の終値または翌日の寄り付き」でも実装される。原典 (turtle-rules.pdf) は「exceeded by a single tick the high or low of the preceding 20 days」と記述しており終値限定ではない。[要確認: 終値 vs 日中ブレイクの厳密な扱い]

### 1.2 スキップ・ルール（直前シグナルが勝ちトレードの場合）

```
if 直前の System 1 ブレイクアウト が勝ちトレードだった:
    次の System 1 シグナルをスキップ
else:
    System 1 シグナルを通常通りエントリー
```

**「勝ち」の定義**:
> 「直前のブレイクアウトが、エントリー後に 10日ブレイク(System 1出口)に達する前に 2N 以上の逆行を受けなかった場合 = 勝ち」
> 換言すると: ブレイクアウト後に価格が**2N 逆行する前に**10日出口に達した場合は「勝ち」

**重要**: スキップ判定は「実際にトレードを取ったかどうか」ではなく、「仮にトレードを取っていたとしたら勝ちか負けか」で判断する。

擬似コード:
```python
def was_winner(breakout_date, direction, N_at_entry):
    entry_price = breakout_price
    stop_loss   = entry_price - 2 * N_at_entry  # long の場合
    exit_10day  = 10日最安値(breakout_date以降)  # long の場合
    
    # 2N 逆行と 10日ブレイクどちらが先か
    date_stop   = first_date_price_reaches(stop_loss)
    date_exit   = first_date_price_reaches(exit_10day)
    
    if date_exit < date_stop:
        return True   # 勝ち → 次シグナルをスキップ
    else:
        return False  # 負け → 次シグナルを通常エントリー
```

### 1.3 フェイルセーフ・ブレイクアウト

System 1 がスキップされた状態で価格がさらにトレンドし続けた場合、**55日ブレイクアウト**でエントリーする（フェイルセーフ）。

```
if System 1 スキップ中 AND 価格が55日高値/安値を1ティック超えた:
    エントリー（System 2 と同じ条件）
```

### 1.4 出口条件

```
Long  Exit: 価格が直近10日間の最安値を1ティック下回った時点
Short Exit: 価格が直近10日間の最高値を1ティック上回った時点
```

出口発動時は**全ユニットを一括クローズ**する。

---

## 2. System 2 完全仕様

### 2.1 エントリー条件

```
Long  Entry: 当日価格が直近55日間の最高値を1ティック超えた時点
Short Entry: 当日価格が直近55日間の最安値を1ティック下回った時点
```

### 2.2 スキップ・ルール

**System 2 にスキップルールはない。** すべてのシグナルを常に取る。

### 2.3 出口条件

```
Long  Exit: 価格が直近20日間の最安値を1ティック下回った時点
Short Exit: 価格が直近20日間の最高値を1ティック上回った時点
```

出口発動時は**全ユニットを一括クローズ**する。

---

## 3. ATR (N) 計算式

### 3.1 True Range の定義

```
TR_t = max(
    H_t - L_t,                    # 当日高値 - 当日安値
    |H_t - C_{t-1}|,              # 当日高値 - 前日終値 (絶対値)
    |L_t - C_{t-1}|               # 当日安値 - 前日終値 (絶対値)
)
```

ここで H_t = 当日高値, L_t = 当日安値, C_{t-1} = 前日終値。

### 3.2 N の平滑化: Wilder 型指数移動平均

```
# 初期値 (最初の20日間): 単純平均
N_20 = mean(TR_1, TR_2, ..., TR_20)

# 21日目以降: Wilder 型指数移動平均 (period=20)
N_t  = (19 × N_{t-1} + TR_t) / 20
```

これは Wilder の Smoothed Moving Average (SMMA) であり、通常のEMAとは異なる（EMAの場合は 2/(n+1) 係数を使用するが、Wilder型は 1/n 係数）。

Pythonでの実装例:
```python
def calc_N(highs, lows, closes, period=20):
    n = len(highs)
    TR = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        TR.append(tr)
    
    # 初期値: 単純平均
    N = [sum(TR[:period]) / period]
    
    # Wilder型 SMMA
    for i in range(period, len(TR)):
        N_new = (19 * N[-1] + TR[i]) / 20
        N.append(N_new)
    
    return N
```

---

## 4. ポジションサイズ (Unit)

### 4.1 基本公式

```
Dollar Volatility = N × Dollar_Per_Point

Unit_size = (Account_Equity × 0.01) / Dollar_Volatility
          = (Account_Equity × 0.01) / (N × Dollar_Per_Point)
```

- **0.01** = 1%リスク（固定。口座サイズによらず変動しない）
- **Dollar_Per_Point** = 1ポイント(1価格単位)動いた時のドル価値
- 計算結果は**切り捨て**（端数は切り捨てて整数ユニット）

### 4.2 Dollar_Per_Point の解釈

| 資産クラス | Dollar_Per_Point の扱い |
|---|---|
| 商品先物 (例: 原油) | 1枚 × 1ポイント = 契約乗数ドル (例: 原油 = $1,000/ポイント) |
| 株式・ETF (例: TQQQ) | 株数1株 × $1 = $1/ポイント |
| 株式CFD | 1ロット × 1ポイント = ロットサイズ × $1 |

**ETF (TQQQ等) の場合の Unit 計算**:
```
# TQQQ の場合: Dollar_Per_Point = 1 (1株 = 1ドル/ポイント)
# N は TQQQ の価格ベースの ATR (例: $5.00 なら N = 5.0)
Unit_shares = (Account_Equity × 0.01) / (N_TQQQ × 1)

例: Account = $100,000, N_TQQQ = $5.00
Unit_shares = ($100,000 × 0.01) / $5.00 = 200株
```

### 4.3 レバレッジ商品 (TQQQ等) 使用時の論点

TQQQ は NASDAQ-100 の**3倍レバレッジ**ETF。タートル・ルールの Unit 計算は N (ATR) ベースのため、TQQQ の高いボラティリティが自動的に Unit_shares を小さくする（ボラティリティ正規化が機能する）。ただし以下の点に注意が必要:

```
実効エクスポージャー = Unit_shares × TQQQ価格 × 3倍レバレッジ

本来の1%リスクは N(TQQQ) に基づいて計算されるが、
TQQQ の N は理論上「NASDAQ × 3」を反映するため、
等価なアンレバレッジポジションより自動的に小さいサイズになる。
→ 原則上の追加調整は不要だが、ボラティリティ急騰期 (VIX spike) には
  Unit が急減するため注意が必要。
```

[要確認: レバレッジETFのβ収縮効果(volatility decay)を ATR が捉えているかの検証]

---

## 5. ピラミッディング

### 5.1 基本ルール

```
エントリー価格 = P0 (最初のブレイクアウト価格)

Unit 1: P0         でエントリー
Unit 2: P0 + 0.5N  でエントリー (long の場合)
Unit 3: P0 + 1.0N  でエントリー
Unit 4: P0 + 1.5N  でエントリー

最大ユニット数 = 4 (単一市場あたり)
```

short の場合は価格を引く方向に調整:
```
Unit 2: P0 - 0.5N でエントリー
Unit 3: P0 - 1.0N でエントリー
Unit 4: P0 - 1.5N でエントリー
```

### 5.2 ストップの引き上げルール（ピラミッディング時）

新しいユニットを追加するたびに、**既存ユニットのストップを 0.5N 引き上げる**。

```
Unit 追加時のストップ更新:
stop_all_units = 最新追加価格 - 2N  # long の場合

具体例 (N=5, P0=100):
  Unit 1 追加時: stop = 100 - 2×5 = 90
  Unit 2 追加時 (P=102.5): 
      全ユニットのストップ = 102.5 - 10 = 92.5 (0.5N = 2.5 引き上げ)
  Unit 3 追加時 (P=105):
      全ユニットのストップ = 105 - 10 = 95
  Unit 4 追加時 (P=107.5):
      全ユニットのストップ = 107.5 - 10 = 97.5
```

これは「最後に追加したユニットの価格から 2N 下」= 全ユニットに適用される統一ストップ水準。

### 5.3 代替ストップ戦略（Whipsaw）

一部のタートルは、ストップを **1/2N** に設定する代替戦略（ whipsaw strategy）も使用した。ストップアウトされた場合は元のブレイクアウト水準に価格が戻ったときに再エントリーする。損失回数は増えるが、利益率は改善したとの記録あり。[要確認: 純正ルールに含まれるかはグレーゾーン]

---

## 6. ストップロス

### 6.1 基本ルール

```
Long  Stop = Entry_Price - 2 × N
Short Stop = Entry_Price + 2 × N
```

- **N は追加時点での当日 N 値を使用**（エントリー時の N を固定するのか、毎日再計算した N を使うかについて: 原典では**エントリー時の N を基準に設定し、新ユニット追加のたびにストップを更新**する方式）
- ストップが逆方向に動く（不利方向へのリリーフ）ことは禁止
- 毎日の N 再計算がストップ水準の自動調整には繋がらない（ストップはエントリー基準で固定）

### 6.2 ストップ発動時の処理

```
if 価格がストップ水準に達した:
    全ユニットを一括クローズ
    新たなブレイクアウトシグナル待ちへ
    System 1 スキップルールを再評価
```

---

## 7. リスク管理ルール

### 7.1 ポジション上限（複数市場）

| レベル | カテゴリ | 最大ユニット |
|---|---|---|
| 1 | 単一市場 (Single Market) | 4 ユニット |
| 2 | 密接相関市場 (Closely Correlated) | 6 ユニット (同方向合計) |
| 3 | 関連グループ (Loosely Correlated) | 10 ユニット (同方向合計) |
| 4 | 全市場合計 (Single Direction) | 12 ユニット (同方向合計) |

原典での密接相関市場の例:
- Heating Oil / Crude Oil
- Gold / Silver
- Swiss Franc / Deutschmark
- T-Bills / Eurodollar

### 7.2 口座縮小時の調整ルール

```
if 口座資産が期初から 10% 減少:
    計算上の口座資産を 20% 削減して Unit サイズを再計算
    (例: 期初 $100,000 → 10%損失で $90,000 だが、
         計算上は $80,000 として Unit を算出)

以降 10% 減少するごとに同様の 20% 削減を適用
(年次リセットあり)
```

この仕組みにより、ドローダウン時に自動的にポジションサイズが縮小される。

擬似コード:
```python
def calc_nominal_equity(actual_equity, peak_equity):
    drawdown_pct = (peak_equity - actual_equity) / peak_equity
    steps = int(drawdown_pct / 0.10)       # 10%ごとに1ステップ
    nominal = actual_equity * (0.80 ** steps)  # 20%ずつ削減
    return nominal
```

---

## 8. NASDAQ 単一銘柄への適用上の論点

### 8.1 ポジション上限ルールの解釈

原典のリスク制限（密接相関6・関連グループ10・全方向12ユニット）は**複数商品を横断的にトレードする際**の制約。NASDAQ 単一銘柄(TQQQ等)のみを運用する場合:

| 制約 | 単一銘柄への解釈 |
|---|---|
| 単一市場: 4ユニット上限 | そのまま適用。最大 4 ユニット |
| 密接相関: 6ユニット | 単一銘柄のみなら適用なし |
| 関連グループ: 10ユニット | 単一銘柄のみなら適用なし |
| 全方向: 12ユニット | 単一銘柄のみなら 4 ユニットが実質上限 |

結論: **単一 ETF 運用では最大 4 ユニット**のみ有効。相関制約は複数銘柄追加時に検討。

### 8.2 ロング/ショート両建て

タートル・ルールはロング・ショート両方向を想定。TQQQ (3倍ロングETF) でショートは不可能なため:
- ショートシグナル時は「ポジションなし」か、SQQQ (3倍インバース) を別銘柄として扱う
- SQQQ を追加する場合は「密接相関」(TQQQ と逆相関) として 6 ユニット制限の適用を検討

### 8.3 レバレッジ ETF の ATR 挙動

```
TQQQ の ATR (N) ≈ NASDAQ-100 の ATR × 3 (概算)

ただし:
- ボラティリティ急上昇時: TQQQ の実際のボラティリティが理論 3倍を超えることがある
- ボラティリティ decay: 長期保有でレバレッジ効果が減衰するリスク
- Unit サイズ: ATR が大きいため自動的に小さくなる → 1% リスクは維持される
```

### 8.4 週末ギャップ・特殊処理

原典に明示的な週末ギャップ処理規定はない。実務上の推奨:
```
週末ギャップ発生時:
  - 寄り付きがストップ水準を超えた場合: 寄り付き価格でクローズ（スリッページ許容）
  - 寄り付きがブレイクアウト水準を超えた場合: 寄り付き価格でエントリー
  - エントリー/ストップ価格の計算は「実際の約定価格」で行う
```

### 8.5 取引コスト（スリッページ・手数料）の取り扱い

原典タートル・ルールにスリッページ・手数料の明示的な扱いは記載なし（プロ向け先物想定のため）。ETF バックテストへの適用における推奨:
```
# スリッページ推奨値 (ETF・株式の場合)
slippage_per_trade = 0.01  # $0.01/株 (流動性の高い TQQQ の場合)
commission         = 0     # 現在多くのブローカーで無料

# バックテスト実装
entry_price_actual = breakout_price + slippage (long)
entry_price_actual = breakout_price - slippage (short)
```

### 8.6 ボラティリティ調整（口座変動に伴う N 再計算）

タートル・ルールでは毎日 N を再計算し、Unit サイズもリアルタイムで調整する。TQQQ 運用では:
```
daily_rebalance:
    N_today = calc_N(recent_prices)
    unit_size_today = (nominal_equity * 0.01) / (N_today * 1.0)
    # ポジション保有中も unit_size の「参照値」は更新
    # ただしストップ水準はエントリー時の N で固定
```

---

## 9. 参考文献・出典

| # | 文献 | URL | 信頼度 |
|---|---|---|---|
| 1 | Curtis Faith, "Original Turtle Trading Rules" (公開PDF, 2003) | https://oxfordstrat.com/coasdfASD32/uploads/2016/01/turtle-rules.pdf | 一次資料 |
| 2 | Curtis Faith, "Way of the Turtle" (McGraw-Hill, 2007) | https://www.amazon.com/Way-Turtle-Methods-Ordinary-Legendary/dp/007148664X | 一次資料 |
| 3 | TradingBlox - Original Turtle System Documentation | https://www.tradingblox.com/originalturtles/originalturtlerules.pdf | 一次資料 |
| 4 | TradingBlox User Guide - Turtle System | https://www.tradingblox.com/Manuals/UsersGuideHTML/turtlesystem.htm | 一次資料 |
| 5 | Trade2Win Forum - System 1 Entry詳細議論 | https://www.trade2win.com/threads/original-turtle-trading-rules-system-1-entry.117112/ | 二次資料 |
| 6 | TurtleSignals - Turtle Trading System | https://turtlesignals.com/the-turtle-trading-system/ | 二次資料 |
| 7 | MarketClutch - Position Sizing詳細 | https://marketclutch.com/quantitative-precision-original-turtle-trading-rules-for-position-sizing/ | 二次資料 |
| 8 | TurtleTrader.com - Original Rules Summary | https://www.turtletrader.com/rules/ | 二次資料 |
| 9 | Altrady - Turtle Trading Rules | https://www.altrady.com/blog/crypto-trading-strategies/turtle-trading-strategy-rules | 二次資料 |
| 10 | TurtleTrader Blogspot - Chapter 3 Position Sizing | https://turtletradersystem.blogspot.com/2009/01/original-turtle-trading-rules-chapter-3.html | 二次資料 |

---

## 付録: クイックリファレンス表

| 項目 | System 1 | System 2 |
|---|---|---|
| エントリー | 20日高値/安値ブレイク | 55日高値/安値ブレイク |
| スキップルール | あり (直前シグナルが勝ちなら次をスキップ) | なし |
| フェイルセーフ | スキップ中に55日ブレイクで再エントリー | N/A |
| 出口 | 10日安値/高値ブレイク | 20日安値/高値ブレイク |
| ストップ | 2N 逆行 | 2N 逆行 |

| 項目 | 値 |
|---|---|
| ATR 計算 | Wilder SMMA (20日), 初期値は単純平均 |
| Unit リスク | 1% of nominal equity |
| ピラミッディング間隔 | 0.5N ごとに +1 Unit |
| 最大 Unit (単一市場) | 4 Unit |
| 最大 Unit (密接相関) | 6 Unit (同方向合計) |
| 最大 Unit (関連グループ) | 10 Unit (同方向合計) |
| 最大 Unit (全方向) | 12 Unit |
| ドローダウン調整 | 10% 損失ごとに計算資産を 20% 削減 |
