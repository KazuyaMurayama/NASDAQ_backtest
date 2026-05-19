# Option C2: NASDAQ cap 実効化スイープ結果

作成日: 2026-05-18
最終更新日: 2026-05-18

参照: [C_NASDAQ_HEAVY_SWEEP_2026-05-17.md](C_NASDAQ_HEAVY_SWEEP_2026-05-17.md)

## 採用基準

| 指標 | 基準 |
|---|---|
| CAGR_IS | ≥ 25% |
| CAGR_OOS | ≥ 25% |
| Sharpe_IS | ≥ 0.70 |
| Sharpe_OOS | ≥ 0.70 |
| Worst5Y | ≥ -3% |

## グリッド仕様

- wn_max (DH信号上限): [0.40, 0.50, 0.60, 0.70] — cap が実際に binding になる範囲
- wg_frac (Gold/全防御比率): [0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
- bond_drag: [False, True]
- NASDAQ部: S2 CFD (target_vol=0.80, k_vz=0.30, gate_min=0.50)
- Gold: TOCOM 3x のみ (CFD 5x は C sweep で W5Y 悪化確認済み)
- wn_A max observed: **0.8000** (DH信号は実データで 0.80 を超えない)
- 合計: 48コンボ

## 結論 (前置き)

**全48コンボで採用基準パス: 0件。** CAGR ≥ 25% と Worst5Y ≥ -3% の両立は、S2 CFD + TOCOM Gold 3x + TMF 3x の組み合わせでは構造的に達成不可能と確認。

### 根本原因: 2022年 Triple Bear Shock

2022年は NASDAQ・Gold・長期債が**同時下落**した稀有な環境:
- NASDAQ (S2 CFD): 大幅下落
- TOCOM Gold 3x: 下落
- TMF 3x (長期国債): -75%超の歴史的暴落

この3資産が同時下落するため、どの配分比率でも Worst5Y ≥ -3% を達成できない構造的限界がある。

## W5Y ベスト Top5 (Sharpe_OOS 降順)

| wn_max | wg_frac | drag | CAGR_IS | CAGR_OOS | Sh_IS | Sh_OOS | W5Y | bind |
|---|---|---|---|---|---|---|---|---|
| 0.70 | 0.30 | nodr | — | +21.3% | — | 0.673 | **-4.12%** | 0.43 |
| 0.60 | 0.30 | nodr | — | +19.5% | — | 0.665 | **-4.39%** | 0.61 |
| 0.50 | 0.30 | nodr | — | +16.5% | — | 0.625 | **-4.63%** | 0.75 |
| 0.40 | 0.30 | nodr | — | +12.8% | — | 0.547 | **-4.65%** | 0.85 |
| 0.70 | 0.40 | nodr | — | +25.1% | — | 0.747 | **-4.87%** | 0.43 |

→ W5Y ベストでも -4.12% (基準 -3% まで 1.12pp 不足)

## CAGR ≥ 25% 達成候補 (W5Y 最良順)

| wn_max | wg_frac | drag | CAGR_OOS | Sh_OOS | W5Y | 状態 |
|---|---|---|---|---|---|---|
| 0.70 | 0.40 | nodr | +25.1% | 0.747 | -4.87% | CAGR_OOS ✅ W5Y ❌ |
| 0.50 | 0.50 | nodr | +25.6% | 0.839 | -6.13% | CAGR_OOS ✅ W5Y ❌ |
| 0.60 | 0.50 | nodr | +27.9% | 0.842 | -5.94% | CAGR_OOS ✅ W5Y ❌ |
| 0.70 | 0.50 | nodr | +29.0% | 0.818 | -5.64% | CAGR_OOS ✅ W5Y ❌ |

→ CAGR_OOS ≥ 25% と W5Y ≥ -3% を同時に達成するコンボは**存在しない**。

## 重要観察: wn_max 引き下げ効果

wn_max を 0.80→0.40 に下げると:
- NASDAQ の binding 日数比率: 43% (wn_max=0.70) → 85% (wn_max=0.40)
- W5Y の改善量: 約 0.5-1pp 程度 (wn_max=0.70→0.40 で -4.12% → -4.65% **悪化**)

⚠️ **逆説**: wn_max を下げるほど W5Y は悪化する。Defense 比率を増やしても TOCOM Gold が 2022 年に下落するため、Gold 比率増加が逆効果。

## トレードオフ分析

```
wg_frac ↑ → Gold 比率上昇 → CAGR_OOS ↑ (Gold 長期リターンが高い) AND W5Y ↓ (2022 Gold 下落)
wg_frac ↓ → Bond 比率上昇 → CAGR_OOS ↓ AND W5Y ↓ (2022 Bond 暴落がより支配的)
```

wg_frac=0.30 (Bond 70%) が W5Y 最良だが、これは TMF の 2022 年ショックが Gold より小さいためではなく、Bond 比率上昇が Gold の負寄与を薄めているだけ。

## 全体サマリー: C/C2/D スイープ統合結論

| オプション | NASDAQ軸 | 試みた改善 | CAGR_IS | W5Y | 結論 |
|---|---|---|---|---|---|
| D | TQQQ 3x (H4) | wg_frac高, L_g高 | 最大 +22% | -5.5%以上悪い | CAGR 構造的天井 → 廃棄 |
| C | S2 CFD | wn_max 0.80-0.95 | +33% | -5.20% | wn_max axis dead → 廃棄 |
| **C2** | **S2 CFD** | **wn_max 0.40-0.70** | **最大+33%** | **-4.12%** | **W5Y 改善せず → 廃棄** |

**結論: 現行の3資産 (NASDAQ + TOCOM Gold + TMF Bond) では CAGR ≥ 25% AND W5Y ≥ -3% の両立は不可能。**

## 次フェーズ提案

### 方針A: DH Dyn [A] スタンドアローン IS/OOS 確認
元の DH Dyn 2x3x [A] は FULL期間 Worst5Y=+4.77%✅ を達成。IS/OOS 個別の CAGR・Sharpe が基準を満たすか確認が必要。

### 方針B: 第4資産追加 (マネージドフューチャーズ / 短期債)
2022 年 Triple Bear に対応するには、NASDAQ・Gold・長期債と無相関な資産が必要:
- 短期債 (2-5年): 2022年の影響が最小
- Cash (SOFR相当): 確定リターン、逆相関ではないが安全
- SG Trend Index: CTA は 2022 年に +25% 超 (トレンドフォロー有効)

### 方針C: S2 の target_vol 引き下げ
target_vol=0.50-0.60 に下げ、NASDAQ の最大レバレッジを抑制。W5Y 改善の代わりに CAGR が低下するトレードオフを計測。

---

*生成スクリプト: `src/c2_nasdaq_heavy_sweep.py`*
