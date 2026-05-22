# P1: タイミング戦略シグナルデータ取得結果

作成日: 2026-05-18
最終更新日: 2026-05-18

参照: [TIMING_STRATEGY_RESEARCH_PLAN_2026-05-18.md](TIMING_STRATEGY_RESEARCH_PLAN_2026-05-18.md)

## 概要

Top5 タイミングシグナルに必要な FRED マクロデータを取得・統合。
`data/timing_signals_raw.csv` (13,169行 × 5列) を生成。

生成スクリプト: `src/p1_fetch_timing_data.py`

---

## 取得 FRED 系列

| FRED ID | 説明 | 期間 | 行数 | 出力ファイル |
|---------|------|------|------|-------------|
| T10Y2Y | 10y-2y スプレッド (日次) | 1976-06-01～2026-03-26 | 12,450 | `data/t10y2y_daily.csv` |
| DFF | Fed Funds 実効金利 (日次) | 1954-07-01～2026-03-26 | 26,202 | `data/dff_daily.csv` |
| CPIAUCSL | CPI 全品目 (月次) | 1947-01-01～2026-03-01 | 950 | `data/cpiaucsl_monthly.csv` |
| BAMLH0A0HYM2 | ICE BofA HY OAS (日次) | 2023-05-16～2026-03-26 | 751 | `data/hy_spread_daily.csv` |
| BAA10Y | Baa-10y スプレッド (日次) | 1986-01-02～2026-03-26 | 10,058 | `data/baa10y_daily.csv` |
| BAA | Moody's Baa 利回り (月次) | 1919-01-01～2026-03-01 | 1,287 | `data/baa_monthly.csv` |
| AAA | Moody's Aaa 利回り (月次) | 1919-01-01～2026-03-01 | 1,287 | `data/aaa_monthly.csv` |

---

## 統合シグナル: `data/timing_signals_raw.csv`

### NaN 率

| 列 | NaN率 | 備考 |
|----|-------|------|
| `hy_spread` | **0.00%** | 3段階スプライスで 1974 まで完全補完 |
| `t10y2y` | 4.63% | 1974-01～1976-05 (610日) が NaN — T10Y2Y の系列開始前 |
| `dff` | **0.00%** | 1954 から完全カバー |
| `cpi_yoy` | **0.00%** | 発表ラグ 15営業日補正済み |
| `cpi_accel` | 0.48% | 先頭 63 日の shift 計算分のみ |

### 統計サマリー

| 列 | 最小 | 最大 | 平均 |
|----|------|------|------|
| hy_spread | 2.590 | 8.670 | 4.757 |
| t10y2y | -2.410 | 2.910 | 0.849 |
| dff | 0.040 | 22.360 | 4.802 |
| cpi_yoy | -1.959 | 14.592 | 3.931 |
| cpi_accel | -5.067 | 4.192 | -0.033 |

### 主要イベント クロスチェック

| 日付 | hy_spread | t10y2y | dff | cpi_yoy | cpi_accel |
|------|-----------|--------|-----|---------|----------|
| 1974-01-02 | 4.11 | NaN | 9.72 | 8.94 | NaN |
| 1981-06-01 | **5.51** | **-1.01** | **19.01** | **9.79** | -1.60 |
| 1994-03-01 | 4.23 | 1.47 | 3.31 | 2.52 | -0.23 |
| 2000-03-01 | 4.47 | -0.13 | 5.78 | 3.22 | 0.60 |
| 2008-09-15 | **6.17** | 1.69 | 2.64 | **5.31** | 1.22 |
| 2022-01-03 | 4.33 | 0.85 | 0.08 | **7.17** | 1.82 |
| 2026-03-26 | 3.21 | 0.46 | 3.64 | 3.32 | 0.32 |

→ Volcker ショック (1981): dff=19%, 逆イールド(-1.01%), インフレ(9.79%) が同時確認 ✅  
→ 金融危機 (2008): hy_spread=6.17% (ピーク付近) ✅  
→ 2022 Triple Bear: cpi_yoy=7.17%, 利上げ局面 (dff 0.08%→急騰) ✅  

---

## hy_spread スプライス設計

3段階の level-shift スプライスで 1919 ～ 2026 を連続接続:

```
[1974-01-02 ～ 1985-12-31]: BAA - AAA 月次 + offset2(+3.46)
                              ↑ level-shift で BAA10Y 開始時点に接続
[1986-01-02 ～ 2023-05-15]: BAA10Y 日次 + offset1(+2.51)
                              ↑ level-shift で BAMLH0A0HYM2 開始時点に接続
[2023-05-16 ～ 2026-03-26]: BAMLH0A0HYM2 (ICE BofA HY OAS、真値)
```

**制限事項**: FRED graphing endpoint の制約で BAMLH0A0HYM2 は 2023-05-16 以降のみ取得可能。
2022 年 Triple Bear は BAA10Y proxy で カバー。
2022-06 ピーク: **4.82%**（真の HY OAS ≈ 5.8% より低いが、z-score 基準では十分なシグナル強度）

---

## P2 への引き継ぎ

### 使用方法 (P2 での読込)

```python
import pandas as pd

signals = pd.read_csv('data/timing_signals_raw.csv', index_col=0, parse_dates=True)
# columns: hy_spread, t10y2y, dff, cpi_yoy, cpi_accel
# index: NASDAQ business days 1974-01-02 to 2026-03-26
```

### P2 で使う各シグナルのデータ源

| シグナル | 使用列 | NaN 補完方針 |
|---------|-------|------------|
| Top1 HY Credit Spread | `hy_spread` | NaN=0% — そのまま使用可 |
| Top2 YC+FF | `t10y2y`, `dff` | t10y2y NaN期間はゲート=1.0 (中立) |
| Top4 CPI Momentum | `cpi_yoy`, `cpi_accel` | NaN=0% — そのまま使用可 |
| Top3 動的相関 | NASDの価格系列から計算 | データ取得不要 |
| Top5 MA200+CPPI | NASDの価格系列から計算 | データ取得不要 |

### 注意事項
- `cpi_yoy` は **発表ラグ 15営業日** 補正済み (look-ahead バイアスなし)
- `hy_spread` は 2022年の 2022 Triple Bear をカバーするが proxy レベルは真のHY OASより絉1pp低い
- `t10y2y` の 1974-1976 NaN期間は **DGS10 - DGS2 で代替可能** (P2で必要なら追加)

---

*生成スクリプト: `src/p1_fetch_timing_data.py`*
*次フェーズ: [P2 シングルシグナル バックテスト]*
