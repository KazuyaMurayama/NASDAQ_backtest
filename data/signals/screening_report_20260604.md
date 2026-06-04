# Phase B Screening Report

作成日: 2026-06-04
最終更新日: 2026-06-04

## サマリ

- 評価triple数: 63
- PASS: **17** triples (6 unique signals)
- FAIL: 46 triples
- ユニーク信号数 (全体): 7
- BH-FDR alpha: 0.10

## 採用 (PASS) 信号

| signal_id | name | asset | horizon | mean_IC | p_BH | hit_rate | lift_pp | direction |
|---|---|---|---|---|---|---|---|---|
| 21 | ICE BofA HY OAS | NDX | 60d | +0.5327 | 0.0000 | 0.791 | +4.08pp | positive |
| 6 | VIX level | NDX | 60d | +0.3370 | 0.0000 | 0.810 | +6.01pp | positive |
| 21 | ICE BofA HY OAS | NDX | 20d | +0.2715 | 0.0000 | 0.723 | +4.81pp | positive |
| 6 | VIX level | NDX | 20d | +0.2157 | 0.0000 | 0.696 | +2.05pp | positive |
| 23 | BAA-10Y credit spread (HY-IG proxy) | NDX | 60d | +0.2047 | 0.0000 | 0.783 | +3.33pp | positive |
| 41 | DXY | NDX | 60d | +0.2012 | 0.0000 | 0.778 | +2.77pp | positive |
| 23 | BAA-10Y credit spread (HY-IG proxy) | NDX | 20d | +0.1638 | 0.0000 | 0.706 | +3.13pp | positive |
| 21 | ICE BofA HY OAS | IEF | 60d | -0.1087 | 0.0000 | 0.508 | -9.53pp | negative (inverse) |
| 28 | 10Y real yield (DGS10 - CPI YoY) | IEF | 20d | +0.1048 | 0.0000 | 0.569 | +0.93pp | positive |
| 41 | DXY | GLD | 20d | +0.0970 | 0.0000 | 0.661 | +9.65pp | positive |
| 41 | DXY | NDX | 20d | +0.0936 | 0.0000 | 0.688 | +1.32pp | positive |
| 28 | 10Y real yield (DGS10 - CPI YoY) | GLD | 20d | +0.0745 | 0.0004 | 0.785 | +22.06pp | positive |
| 28 | 10Y real yield (DGS10 - CPI YoY) | IEF | 60d | +0.0740 | 0.0084 | 0.604 | +0.04pp | positive |
| 26 | 2s10s spread (DGS10-DGS2) | NDX | 20d | -0.0736 | 0.0001 | 0.667 | -0.80pp | negative (inverse) |
| 21 | ICE BofA HY OAS | IEF | 20d | -0.0697 | 0.1116 | 0.384 | -17.59pp | negative (inverse) |
| 26 | 2s10s spread (DGS10-DGS2) | NDX | 60d | -0.0641 | 0.0086 | 0.758 | +0.79pp | negative (inverse) |
| 28 | 10Y real yield (DGS10 - CPI YoY) | GLD | 60d | +0.0492 | 0.0725 | 0.931 | +30.93pp | positive |

### 解釈ガイド

- **positive IC**: 信号値↑ (より高いバケット) → forward return↑ (順方向先行指標)
- **negative IC**: 信号値↑ → forward return↓ (逆張り先行指標)
- **lift_pp**: max信号レベル条件下のヒット率 - base rate (パーセントポイント)
- **p_BH**: BH-FDR 補正済 p 値 (`alpha=0.10`)

## 棄却 (FAIL) 信号 — 主要因

| signal_id | name | asset | horizon | mean_IC | p_BH | 主要因 |
|---|---|---|---|---|---|---|
| 6 | VIX level | GLD | 5d | +0.0266 | 0.0005 | |IC|<=0.05, Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 6 | VIX level | GLD | 20d | +0.0479 | 0.0005 | |IC|<=0.05, Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 6 | VIX level | GLD | 60d | -0.0229 | 0.2799 | FDR>=10%, |IC|<=0.05, Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 6 | VIX level | IEF | 5d | -0.0266 | 0.0012 | |IC|<=0.05, Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 6 | VIX level | IEF | 20d | -0.0177 | 0.2556 | FDR>=10%, |IC|<=0.05, Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 6 | VIX level | IEF | 60d | -0.0646 | 0.0018 | Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 6 | VIX level | NDX | 5d | +0.1337 | 0.0000 | Wilson_lower<=base+3pp |
| 21 | ICE BofA HY OAS | GLD | 5d | -0.0050 | 0.8178 | FDR>=10%, |IC|<=0.05, Wilson_lower<=base+3pp |
| 21 | ICE BofA HY OAS | GLD | 20d | -0.0189 | 0.6199 | FDR>=10%, |IC|<=0.05, Wilson_lower<=base+3pp |
| 21 | ICE BofA HY OAS | GLD | 60d | -0.2258 | 0.0002 | (complex) |
| 21 | ICE BofA HY OAS | IEF | 5d | -0.0066 | 0.6990 | FDR>=10%, |IC|<=0.05, Wilson_lower<=base+3pp |
| 21 | ICE BofA HY OAS | NDX | 5d | +0.2000 | 0.0000 | Wilson_lower<=base+3pp |
| 23 | BAA-10Y credit spread (HY-IG proxy) | GLD | 5d | -0.0163 | 0.0353 | |IC|<=0.05, Wilson_lower<=base+3pp, half-sample sign flip |
| 23 | BAA-10Y credit spread (HY-IG proxy) | GLD | 20d | -0.0264 | 0.0535 | |IC|<=0.05, Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 23 | BAA-10Y credit spread (HY-IG proxy) | GLD | 60d | -0.0816 | 0.0002 | Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 23 | BAA-10Y credit spread (HY-IG proxy) | IEF | 5d | -0.0291 | 0.0029 | |IC|<=0.05, Wilson_lower<=base+3pp, decade sign mixed |
| 23 | BAA-10Y credit spread (HY-IG proxy) | IEF | 20d | -0.0807 | 0.0000 | Wilson_lower<=base+3pp, decade sign mixed |
| 23 | BAA-10Y credit spread (HY-IG proxy) | IEF | 60d | -0.1032 | 0.0002 | Wilson_lower<=base+3pp |
| 23 | BAA-10Y credit spread (HY-IG proxy) | NDX | 5d | +0.0733 | 0.0000 | Wilson_lower<=base+3pp, decade sign mixed |
| 26 | 2s10s spread (DGS10-DGS2) | GLD | 5d | +0.0085 | 0.3484 | FDR>=10%, |IC|<=0.05, Wilson_lower<=base+3pp, decade sign mixed |
| 26 | 2s10s spread (DGS10-DGS2) | GLD | 20d | +0.0501 | 0.0084 | Wilson_lower<=base+3pp, half-sample sign flip |
| 26 | 2s10s spread (DGS10-DGS2) | GLD | 60d | +0.0329 | 0.2320 | FDR>=10%, |IC|<=0.05, Wilson_lower<=base+3pp, half-sample sign flip |
| 26 | 2s10s spread (DGS10-DGS2) | IEF | 5d | +0.0840 | 0.0000 | Wilson_lower<=base+3pp |
| 26 | 2s10s spread (DGS10-DGS2) | IEF | 20d | +0.1305 | 0.0000 | half-sample sign flip, decade sign mixed |
| 26 | 2s10s spread (DGS10-DGS2) | IEF | 60d | +0.2029 | 0.0000 | half-sample sign flip, decade sign mixed |
| 26 | 2s10s spread (DGS10-DGS2) | NDX | 5d | -0.0487 | 0.0000 | |IC|<=0.05, Wilson_lower<=base+3pp |
| 27 | 3M10Y spread (DGS10-DTB3) | GLD | 5d | +0.0575 | 0.0000 | Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 27 | 3M10Y spread (DGS10-DTB3) | GLD | 20d | +0.1088 | 0.0000 | Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 27 | 3M10Y spread (DGS10-DTB3) | GLD | 60d | +0.1142 | 0.0000 | half-sample sign flip, decade sign mixed |
| 27 | 3M10Y spread (DGS10-DTB3) | IEF | 5d | +0.0891 | 0.0000 | Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 27 | 3M10Y spread (DGS10-DTB3) | IEF | 20d | +0.1582 | 0.0000 | half-sample sign flip, decade sign mixed |
| 27 | 3M10Y spread (DGS10-DTB3) | IEF | 60d | +0.2262 | 0.0000 | half-sample sign flip, decade sign mixed |
| 27 | 3M10Y spread (DGS10-DTB3) | NDX | 5d | -0.0364 | 0.0002 | |IC|<=0.05, Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 27 | 3M10Y spread (DGS10-DTB3) | NDX | 20d | -0.0563 | 0.0048 | Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 27 | 3M10Y spread (DGS10-DTB3) | NDX | 60d | -0.0798 | 0.0019 | Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 28 | 10Y real yield (DGS10 - CPI YoY) | GLD | 5d | +0.0575 | 0.0000 | Wilson_lower<=base+3pp |
| 28 | 10Y real yield (DGS10 - CPI YoY) | IEF | 5d | +0.1022 | 0.0000 | Wilson_lower<=base+3pp |
| 28 | 10Y real yield (DGS10 - CPI YoY) | NDX | 5d | +0.0211 | 0.0207 | |IC|<=0.05, Wilson_lower<=base+3pp, decade sign mixed |
| 28 | 10Y real yield (DGS10 - CPI YoY) | NDX | 20d | +0.0375 | 0.0156 | |IC|<=0.05, decade sign mixed |
| 28 | 10Y real yield (DGS10 - CPI YoY) | NDX | 60d | +0.0554 | 0.0113 | decade sign mixed |
| 41 | DXY | GLD | 5d | +0.0437 | 0.0004 | |IC|<=0.05, Wilson_lower<=base+3pp, decade sign mixed |
| 41 | DXY | GLD | 60d | +0.1958 | 0.0000 | (complex) |
| 41 | DXY | IEF | 5d | +0.0518 | 0.0000 | Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 41 | DXY | IEF | 20d | +0.1108 | 0.0000 | Wilson_lower<=base+3pp, decade sign mixed |
| 41 | DXY | IEF | 60d | +0.0787 | 0.0023 | Wilson_lower<=base+3pp, half-sample sign flip, decade sign mixed |
| 41 | DXY | NDX | 5d | +0.0450 | 0.0000 | |IC|<=0.05, Wilson_lower<=base+3pp |

## 統計サマリ

- raw p<0.10: 56 / 63
- BH p<0.10: 55 / 63
- |IC| 中央値: 0.0740
- |IC| 95%分位: 0.2261
- |IC| max: 0.5327

## 採用判定ルール

**Primary (20d horizon):** BH-FDR<0.10 AND |IC|>0.05 AND Wilson_lower > base+3pp AND 半分割同符号

**Secondary (20d AND 60d):** |IC|>0.04 両水準 AND decade 同符号 AND 20d/60d 同符号

## 次工程 (Phase C)

PASS 信号を `phase_b_selection_<date>.csv` に出力。Phase C で overlay/standalone モードでの WFA 評価へ。
