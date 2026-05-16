# セッションサマリー — 2026-05-16

作成日: 2026-05-16
最終更新日: 2026-05-16

**セッション軸**: CFD動的レバレッジ戦略 改良版検証（S1/S2/S3/S4バックテスト・P0検証）
**管理者**: 男座員也 (Kazuya Oza)
**ブランチ**: `claude/review-best-strategy-Jcjd5`

---

## TL;DR（3行）

1. **S2_VZGated 採用確定**: OOS Sharpe 0.769、Worst5Y -4.75%、IS-OOS Gap 5.4pp で全採用基準クリア
2. **P0検証完了**: SOFR単位✅ / P2/S2のtarget_volが99.7%クリップ（実質ノイズ）⚠️ / Worst5Y定義✅
3. **S1/S3/S4は不採用**: S4はWorst5Y大幅改善（-2.3%）も OOS Sharpe 0.70でP2基準届かず

---

## P0検証結果（Opus指摘対応）

### [1] SOFR単位検証 ✅ 正常

| 期間 | 年率換算値 | 期待値 | 判定 |
|---|---|---|---|
| 全期間中央値 | 4.62%/年 | 歴史的平均≈3-5% | ✅ |
| 2023年平均 | 5.07%/年 | FFレート≈5.25% | ✅ |
| 1981年平均 | 14.02%/年 | FFレート≈15-18% | ✅ |

`load_sofr()` の `/100/252` 正規化は正しい。全バックテスト結果は有効。

### [2] vt_mult クリッピング率 ⚠️ 設計上の制約（既知化済み）

**検証コード**: `src/p0_verify_critical.py`

| target_vol | ratio≥1.0の割合 | ratio中央値 | OOS期間clip率 |
|---|---|---|---|
| 0.60 | 98.6% | 4.4 | 99.3% |
| 0.70 | 99.2% | 5.1 | — |
| 0.80 | **99.7%** | **5.9** | **100.0%** |

- NASDAQの実現ボラ中央値: **13.6%/年**
- target_vol=0.80 に対し ratio中央値≈5.9 → `l_max=7`に常時クリップ
- **結論**: P2/S2の `target_vol` パラメータは実質ノイズ。戦略の本質は「高ボラ時デレバ機構」

**影響範囲**: P2, S2 の両者。S1/P4にも同様の問題。S4（相対ボラ）でこの問題を回避。

### [3] Worst5Y定義確認 ✅ 正常

`calc_7metrics` 内の計算式:
```python
r5 = (ns_f / ns_f.shift(252 * 5)) ** (1/5) - 1
w5 = float(r5.min())
```
- window = 252×5 = 1260日
- DH Dynポートフォリオ全体NAVで計算（CFDスリーブ単体ではない）
- 計算式は正しい

---

## S4戦略（新設計）仕様

### コンセプト: 相対ボラ（短期/長期EWMA比）× VIXゲート

```python
sigma_short = ewma_vol(returns, halflife=20)    # 短期EWMA（≈60日相当）
sigma_long  = ewma_vol(returns, halflife=120)   # 長期EWMA（≈252日相当）
sigma_rel   = sigma_short / sigma_long          # 相対ボラ比

rel_excess  = max(sigma_rel - rel_threshold, 0.0)
rel_factor  = clip(1.0 - k_rel × rel_excess, gate_min=0.2, 1.0)

vz_pos  = max(vz, 0.0)                         # 負のVZは無視（非対称）
vz_gate = clip(1.0 - k_vz × vz_pos, gate_min=0.2, 1.0)

L_t = clip(l_base × rel_factor × vz_gate, l_min=1, l_max=l_base)
```

**P2/S2との違い**: 絶対ボラ水準ではなく「現在ボラ÷通常ボラ」で判断
→ NASDAQが構造的高ボラレジームに移行しても自動適応

**実装場所**: `src/dynamic_leverage_strategies.py` `compute_L_s4_relvol()`

---

## バックテスト結果サマリー

**評価軸**: DH Dynポートフォリオ（NASDAQ CFD + Gold2x20% + Bond3x20%）
**IS期間**: 1974-01-02〜2021-05-07 | **OOS期間**: 2021-05-08〜2026-03-26
**採用基準**: OOS Sharpe > P2 best (0.757) AND |IS-OOS CAGR gap| < 10pp AND Worst5Y > -5%

| 戦略 | CAGR(IS) | CAGR(OOS) | Sharpe(OOS) | MaxDD | Worst5Y | IS-OOS Gap | 採用 |
|---|---|---|---|---|---|---|---|
| P2 best (tv=0.8) | +34.60% | +27.13% | 0.757 | -60.5% | -6.63% | 7.5pp | baseline |
| S1_Conviction | +43.00% | +22.47% | 0.645 | -64.3% | -1.91% | 20.5pp | ❌ Sharpe/Gap未達 |
| **S2_VZGated ★** | **+32.94%** | **+27.57%** | **0.769** | -62.4% | **-4.75%** | **5.4pp** | ✅ **全基準クリア** |
| S3_Decomposed | +32.80% | +9.60% | 0.431 | -54.8% | -3.77% | 23.2pp | ❌ Sharpe/Gap未達 |
| S4_RelVol | +40.98% | +26.19% | 0.697 | -66.1% | **-2.33%** | 14.8pp | ❌ Sharpe/Gap未達 |

---

## S2_VZGated 確定推奨パラメータ

```
target_vol = 0.80   （実質ノイズ、ただし互換性のため維持）
k_vz       = 0.30   （VIXゲート感度）
gate_min   = 0.50   （VIX高騰時の最低レバレッジ係数）
l_min      = 1.0
l_max      = 7.0
```

| 採用基準 | 値 | 判定 |
|---|---|---|
| OOS Sharpe > P2 (0.757) | 0.769 | ✅ |
| Worst5Y > -5% | -4.75% | ✅ |
| \|IS-OOS gap\| < 10pp | 5.4pp | ✅ |

---

## CURRENT_BEST_STRATEGY.md を更新しない理由

- S2はCFD専用戦略（NASDAQスリーブのみ）
- 現行ベスト（DH Dyn 2x3x [A] 閾値0.15）は TQQQ + Gold + Bond の総合戦略
- 評価軸が異なるため、CFD軸の正典は [`CFD_DYNAMIC_LEVERAGE_GUIDE.md`](CFD_DYNAMIC_LEVERAGE_GUIDE.md) で別管理
- S2 MaxDD -62%はDH Dynポートフォリオ全体の値（NASDAQスリーブが7x化するため高い）

---

## このセッションで作成・修正したファイル

| ファイル | 種別 | 内容 |
|---|---|---|
| `ENH_LEVERAGE_BACKTEST_2026-05-16.md` | 新規 | S1/S2/S3/S4バックテスト最終結果レポート |
| `src/p0_verify_critical.py` | 新規 | P0検証スクリプト（SOFR/vt_mult/Worst5Y） |
| `src/dynamic_leverage_strategies.py` | 修正 | S4 `compute_L_s4_relvol()` 追加、ドキュメント更新 |
| `src/enh_lev_backtest.py` | 修正 | S4グリッド追加、Worst5Yフィルタ（IS段階）追加 |
| `ENH_LEVERAGE_BACKTEST_2026-05-15.md` | SUPERSEDED | 後継: ENH_LEVERAGE_BACKTEST_2026-05-16.md |
| `CFD_DYNAMIC_LEVERAGE_GUIDE.md` | 新規 | CFD軸の正典（S2採用パラメータ・制約・経緯） |
| `SESSION_SUMMARY_2026-05-16.md` | 新規 | 本ファイル |
| `tasks.md` | 修正 | 完了タスク追加・新規Pendingタスク追加 |
| `FILE_INDEX.md` | 修正 | 新規ファイル追加・最終更新日更新 |
| `CURRENT_BEST_STRATEGY.md` | 1行追記 | CFD動的レバレッジ軸への相互参照追加 |

---

## 次セッションへの引継ぎ

### 優先度高（継続すべき検討）

1. **S2 Rolling Window CV 検証**
   - 現状: 単一OOS区間（2021-05-08〜2026-03-26, 約5年）のみ
   - 提案: 5yr IS → 1yr OOS のローリングウィンドウ検証でOOS安定性を確認
   - 実装: `src/enh_lev_backtest.py` に `rolling_cv()` 関数を追加

2. **target_vol パラメータの設計改善（P2*/S2*）**
   - 問題: target_vol=0.60〜0.80は99%以上クリップ（実質ノイズ）
   - 提案: target_vol ∈ {0.10, 0.13, 0.16, 0.20, 0.25} で再グリッドサーチ
   - 期待: target_vol が実際に機能するパラメータになる

3. **S4の Sharpe 改善試案**
   - Worst5Y -2.33% は S2比で大幅改善（候補として有望）
   - l_base=5以下 × k_rel/rel_threshold の再チューニングが候補
   - IS-OOS gap 14.8ppの縮小が課題

### 優先度中（DH Dyn軸との統合）

4. **S2を DH Dynポートフォリオへの正式組み込み試算**
   - 現在の S2 評価はすでに DH Dyn文脈（NASDAQ CFD + Gold2x + Bond3x）
   - CURRENT_BEST_STRATEGY.md（DH Dyn 2x3x [A]）との統合シナリオ検討

### 既存 Pending タスク（CFD軸とは独立）

5. Approach A への GAS 切替実装（閾値0.15）→ `nasdaq-strategy-gas` リポジトリ
6. Ens2 戦略の OOS 検証（2022-2026）

---

## 新セッション開始時の推奨手順

```
1. CURRENT_BEST_STRATEGY.md を読む（DH Dyn軸の現状確認）
2. CFD_DYNAMIC_LEVERAGE_GUIDE.md を読む（CFD軸の現状確認）
3. tasks.md を読む（Pending タスクの確認）
4. git checkout claude/review-best-strategy-Jcjd5（作業ブランチ確認）
```

---

*関連ファイル: [CFD_DYNAMIC_LEVERAGE_GUIDE.md](CFD_DYNAMIC_LEVERAGE_GUIDE.md) / [ENH_LEVERAGE_BACKTEST_2026-05-16.md](ENH_LEVERAGE_BACKTEST_2026-05-16.md) / [src/p0_verify_critical.py](src/p0_verify_critical.py)*
