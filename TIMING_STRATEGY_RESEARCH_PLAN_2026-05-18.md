# NASDAQ 3倍レバレッジ戦略: 未試験タイミング戦略 調査・バックテスト計画

作成日: 2026-05-18
最終更新日: 2026-05-18

## 0. エグゼクティブサマリー

現行 DH Dyn 2x3x [A] は **CAGR+23% / Sharpe 1.03 / Worst5Y +0.66%** と防御力は十分だが、CFD7 級の **CAGR 30%+** とは両立できていない。S2 VZGated は CAGR を稼ぐが Worst5Y -5〜-6% で過剰リスク。問題の本質は以下 3 点:

1. **2022 Triple Bear** (NASDAQ + Gold + Bond 同時下落) は資産分散では原理的に防げない → **時間軸の防御** (=現金化レジーム) が必要
2. **金利急騰局面 (1981/1994/2022)** は価格シグナルでは遅行する → **マクロ先行指標** の活用が未開拓
3. **動的相関の崩壊**を検知するメカニズムが現行戦略に存在しない

本計画では、上記を埋める **6 カテゴリ × 26 手法** を体系評価し、**Top5 を優先実装**するロードマップを提示する。

---

## 1. 優先度マトリクス (実装難易度 × 効果期待度)

| # | カテゴリ | 手法 | 実装難度 | 効果期待 | データ | 優先度 |
|---|---------|------|---------|---------|-------|--------|
| 1.1 | マクロ | 2y-10y スプレッド逆転 | ★1 | ◎ | FRED | **A+** |
| 1.2 | マクロ | Fed Funds 12M Δ | ★1 | ◎ | FRED | **A+** |
| 1.3 | マクロ | リアルレート (10y TIPS - CPI) | ★2 | ○ | FRED | A |
| 1.4 | マクロ | M2 yoy | ★1 | △ | FRED | B |
| 2.1 | テクニカル | 200日MA フィルター | ★1 | ○ | price | **A+** |
| 2.2 | テクニカル | 52w高値乖離 | ★1 | ○ | price | A |
| 2.3 | テクニカル | Donchian/Keltner | ★2 | ○ | price | B |
| 2.4 | テクニカル | 月次RSI | ★1 | △ | price | B |
| 3.1 | クロスアセット | HY-IG クレジットスプレッド | ★1 | **◎** | FRED | **A+** |
| 3.2 | クロスアセット | 銅/金 比率 | ★1 | ○ | yfinance | A |
| 3.3 | クロスアセット | DXY | ★1 | ○ | FRED | A |
| 3.4 | クロスアセット | VIX term structure | ★2 | ○ | CBOE | B |
| 3.5 | クロスアセット | Put/Call | ★2 | △ | CBOE | C |
| 4.1 | 相関 | 動的相関モニタ | ★2 | **◎** | price | **A+** |
| 4.2 | 相関 | 最小分散最適化 | ★3 | ○ | price | B |
| 4.3 | 相関 | リスクパリティ | ★3 | ○ | price | B |
| 4.4 | 相関 | 合成OTMプット | ★4 | △ | option chain | D |
| 5.1 | レジーム | HMM (2-3状態) | ★4 | ○ | price+macro | C |
| 5.2 | レジーム | ハースト指数 | ★3 | △ | price | C |
| 5.3 | レジーム | MS-GARCH | ★5 | ○ | price | D |
| 5.4 | レジーム | CPI/PCE モメンタム | ★1 | **◎** | FRED | **A+** |
| 6.1 | レバレッジ | Kelly 動的 | ★3 | ○ | price | B |
| 6.2 | レバレッジ | CPPI | ★2 | ○ | price | A |
| 6.3 | レバレッジ | フラクタルvol | ★2 | △ | price | C |
| 6.4 | レバレッジ | CVaR 最適化 | ★4 | ○ | price | C |

---

## 2. 最優先 Top5 手法 — 詳細実装方針

### Top1: HY クレジットスプレッド・ガード (#3.1)

**メカニズム**: HY-IG スプレッドは金融ストレスの **先行指標** で、1994/2008/2015/2020/2022 全てで NASDAQ ドローダウン前に拡大した実績がある。**2022年早期(1〜3月)に既にHYスプレッドは拡大**しており、NASDAQ ドローダウンに先行。

```python
シグナル定義:
  cs_t = HY OAS (BAMLH0A0HYM2, FRED 日次)
  z = (cs_t - rolling_mean_252) / rolling_std_252
  cs_gate = clip(1.0 - max(0, z - 1.0)*0.5, 0.2, 1.0)
  # z=1.0 でゲート開始、z=3.0 でレバ20%まで縮小
```

**実装方針**:
- 既存 DH Dyn の最終レバレッジに **乗算ゲート** として組込む
- `z_threshold ∈ {0.5, 1.0, 1.5, 2.0}` と縮減傾斜をグリッド探索 (16通り以内で過学習防止)
- データ: FRED `BAMLH0A0HYM2EY` (1996〜) / 過去拡張は `BAA10Y` (1953〜) で代理

**期待効果**: 2022 で 60〜70% 程度のドローダウン縮減

---

### Top2: イールドカーブ + Fed Funds 加速度の複合レジームフィルター (#1.1 + #1.2)

**メカニズム**: 「逆イールド + Fed Funds 急上昇」は 1981/1994/2000/2007/2022 の全てに先行。単独では誤検知 (2019 など) が出るので **AND 条件で精度向上**。

```python
シグナル定義:
  yc = T10Y2Y (FRED日次)
  ff_accel = FF_rate_t - FF_rate_{t-252}  # 12M変化
  regime_risk = 1 if (yc < 0 AND ff_accel > 1.5%) else 0
  ramp_down = EWMA(regime_risk, span=63)   # 平滑化で離散的切替を避ける
  macro_gate = 1.0 - 0.5 * ramp_down      # 最大50%レバ削減
```

**期待効果**: 2022 ピークの 30〜50% 防御。1994 も大きく改善見込み。

---

### Top3: 動的相関 NASDAQ-Bond ガード + Cash 退避 (#4.1)

**メカニズム**: 2022 の本質は「Bond ヘッジが効かなかった」こと。インフレ加速期は NASDAQ-Bond 相関が正転する。これを**動的検知**すれば Bond を Cash に切替できる。

```python
シグナル定義:
  ρ_nb = corr(NASDAQ_ret, TMF_ret, window=60)
  ρ_ng = corr(NASDAQ_ret, Gold_ret, window=60)
  hedge_health = max(0, -ρ_nb) + max(0, -ρ_ng)
  bond_weight = base_bond_w * clip(hedge_health, 0.2, 1.0)
  cash_weight = 1.0 - bond_weight  # 不足分はSOFR金利で代替
```

**期待効果**: 2022 Bond 損失をほぼ回避。ドローダウン -10〜-12% → -3〜-5% に圧縮。

---

### Top4: CPI/PCE モメンタム・ガード (#5.4)

**メカニズム**: インフレ加速期 (CPI yoy 急上昇) は Triple Bear の必要条件。1981/2022 の最大下落年は全て CPI yoy が 6〜10% 超。

```python
シグナル定義:
  cpi_yoy = CPIAUCSL (FRED月次, 12M変化)
  cpi_accel = cpi_yoy_t - cpi_yoy_{t-3}   # 3ヶ月変化
  infl_regime = clip((cpi_yoy - 3.0) / 5.0, 0, 1)  # 3%→0, 8%→1
  infl_gate = 1.0 - 0.4 * max(infl_regime, clip(cpi_accel,0,2)/2)
```

**実装方針**: 発表ラグ 15〜30日を必ず考慮して適用。1948〜の長期データで検証可能。

**期待効果**: 2022 ドローダウン 40〜50% 軽減。1981 も大幅改善。

---

### Top5: 200日MA + CPPI フロアの統合 (#2.1 + #6.2)

**メカニズム**: マクロ指標が遅行/誤検知した後の**被害拡大を止める**最終防衛線。

```python
ma_gate = 1.0 if price > SMA(close, 200) else 0.5
floor = peak_NAV * 0.85
cushion = (NAV - floor) / NAV
cppi_lev_cap = m * cushion   # m=3〜5
final_lev = min(target_lev, cppi_lev_cap) * ma_gate
```

**期待効果**: 全ての 20% 超下落年を最低でも半減。未知の急落にも確定的に作動。

---

## 3. データ入手可能性

| データ | ソース | 期間 | コスト |
|--------|--------|------|--------|
| HY OAS `BAMLH0A0HYM2` | FRED | 1996〜 | 無料 (1996前: `BAA10Y` 代理) |
| 2y-10y `T10Y2Y` | FRED | 1976〜 | 無料 |
| Fed Funds `DFF` | FRED | 1954〜 | 無料 |
| CPI `CPIAUCSL` | FRED | 1947〜 | 無料 |
| Core PCE `PCEPILFE` | FRED | 1959〜 | 無料 |
| DXY `DTWEXBGS` | FRED | 1971〜 | 無料 |
| 銅先物 `HG=F` | yfinance | 2000〜 | 無料 |

**全 Top5 シグナルが FRED + yfinance で完全無料取得可能。コスト障壁なし。**

---

## 4. バックテスト設計フレームワーク

### 4.1 期間分割

```
TRAIN/IS:    1980-01 〜 2010-12  (31年)
VALIDATION:  2011-01 〜 2017-12  (7年)
OOS:         2018-01 〜 2026-03  (8年, 2022 Triple Bear 含む)
```

### 4.2 過学習防止プロトコル

1. 各新規シグナルは **3パラメータ以内** に抑制
2. グリッド粒度: 各パラメータ 3〜4 点のみ
3. Combinatorial Purged CV (5-fold)
4. Deflated Sharpe Ratio (多重検定対策)
5. Block Bootstrap (月次ブロック 1000回)

### 4.3 評価指標

| メトリック | 目標 | 重み |
|----------|-----|------|
| CAGR (FULL) | ≥ 22% | 0.30 |
| Sharpe (FULL) | ≥ 1.10 | 0.20 |
| **Worst5Y (FULL)** | **≥ -3%** | **0.30** |
| Max DD | ≥ -35% | 0.10 |
| Calmar | ≥ 0.7 | 0.10 |

**ペナルティ**: 大暴落年 (1981/1988/1994/2015/2022) のいずれかで -25% 超で自動失格。

---

## 5. 統合アーキテクチャ (階層型ゲート)

```
Layer 0: DH Dyn [A] ベース
  L_base = TQQQ_target × (DD × AsymEWMA × TrendTV × SlopeMult × MomDecel × VIX_MR)

Layer 1: マクロレジームゲート [Top2 + Top4]
  G_macro = min(yc_ff_gate, infl_gate)

Layer 2: クレジット/相関ヘルスゲート [Top1 + Top3]
  G_health = cs_gate × hedge_health_gate

Layer 3: 価格防衛 [Top5]
  G_price = min(ma_gate, cppi_cap / L_base)

L_final = L_base × G_macro × G_health × G_price
```

---

## 6. 2022年シミュレーション (事前推論)

| シグナル | 2022検知時期 | 推定削減効果 |
|---------|-------------|------------|
| #3.1 HYスプレッド | **1〜2月** (z>1.5) | NASDAQ損失 -50% |
| #1.1+1.2 YC+FF | **3〜4月** | NASDAQ損失 -25% |
| #4.1 動的相関 | **4〜5月** (ρ_nb>0) | Bondヘッジ損失 ほぼ回避 |
| #5.4 CPIモメンタム | **1月** (7.5%) | 早期防御 -40% |
| #2.1+6.2 MA+CPPI | **2月** | 価格確認後の被害止め |

**統合後 2022 推定**: -8〜-10% → **-2〜-4%** 程度

---

## 7. 実装ロードマップ (8週間)

| Phase | 期間 | 内容 |
|-------|------|------|
| **P1: データ取得** | 1週 | FRED API で全シグナルデータ取得・長期延長 |
| **P2: 単独バックテスト** | 2週 | Top5 各シグナル単独効果を IS/OOS で測定 |
| **P3: 組合せ探索** | 2週 | 5C2=10ペア → トリプレット → 全5統合 |
| **P4: 過学習確認** | 1週 | Combinatorial Purged CV、Deflated Sharpe |
| **P5: ストレステスト** | 1週 | Block Bootstrap、仮想シナリオ |
| **P6: 記録** | 1週 | DH Dyn 2x3x [B] として CURRENT_BEST_STRATEGY.md 更新 |

---

## 8. リスクと留意点

1. **Look-ahead バイアス**: CPI/M2 は発表ラグ (15〜30日) あり、必ず発表日基準で適用
2. **データ改定**: M2 は 2020年に定義変更。期間で扱いを分ける
3. **シグナル相関**: Top1/2/4 は高相関の可能性 → Shapley で寄与確認、冗長なら統合
4. **取引コスト**: ゲート切替頻度が高すぎると実コストが膨らむ → EWMA 平滑化前提

---

## 9. 既存コードの活用可能性

| 既存ファイル | 活用内容 |
|------------|----------|
| `generate_macro_features.py` | yield curve・credit spread・VIX 特徴量は **既実装**。Top2/Top4 の基盤として流用 |
| `fetch_fred_data.py` | FRED データ取得パターン |
| `dynamic_leverage_strategies.py` | ゲート組込みの基盤 (`compute_L_kelly` 等) |
| `regime_analysis.py` | レジーム分析フレームワーク |
| `corrected_strategy_backtest.py` | IS/OOS 分割・メトリクス計算の正典 |

**注目**: `generate_macro_features.py` に yield curve・credit spread・VIX 特徴量が既実装。Top2・Top4 はほぼそのまま流用できる可能性あり。

---

## 10. 期待される最終成果

統合戦略 **DH Dyn 2x3x [B]** (仮称):

| 指標 | 現行 [A] | 目標 [B] |
|------|---------|----------|
| CAGR (FULL) | +22〜23% | +22〜25% |
| Sharpe (FULL) | 1.03 | 1.15〜1.25 |
| **Worst5Y** | **+0.66%** | **-2〜+1%** (大暴落年を防御強化) |
| Max DD | -45% | -30%以内 |

CFD7 級 (CAGR 30%超) を狙う場合は **VIX term structure (#3.4) + Kelly 動的 (#6.1)** を追加する第2段階を検討。ただし Worst5Y -5% 級の覚悟が必要なため、**防御版 [B] / 攻撃版 [C]** の 2系統運用が現実的。

---

*計画立案: Opus (2026-05-18)*  
*実行予定: 本文書を参照して段階的に実装*
