# CFD 動的レバレッジ戦略 ガイド（正典）

作成日: 2026-05-16
最終更新日: 2026-05-16

> **本ファイルは「CFD動的レバレッジ戦略」の正典 (Single Source of Truth) です。**
> [`CURRENT_BEST_STRATEGY.md`](CURRENT_BEST_STRATEGY.md)（DH Dyn 2x3x [A]）とは**評価軸が異なります**。
> - **DH Dyn 軸**: TQQQ (3x ETF) + Gold2x + Bond3x のポートフォリオ戦略
> - **CFD軸（本ファイル）**: CFD/くりっく株365 で NASDAQ を1x〜7x 動的レバレッジ、DH Dynポートフォリオ内のNASDAQスリーブに適用
>
> 両者は独立した研究軸として**並立**します。

---

## 推奨戦略: **S2_VZGated**

採用確定日: 2026-05-16

### 数式

```
sigma_t   = rolling_std(r_nas, n=20) × sqrt(252)       — 実現ボラ（年率）
vz_t      = VIXプロキシ z-score（A2シグナルの vz 成分）

vz_gate   = clip(1 - k_vz × max(vz_t, 0), gate_min, 1.0)  — 非対称VIXゲート
L_t       = clip((target_vol / sigma_t) × vz_gate, l_min=1, l_max=7)

DELAY = 2: L_t は build_nav_strategy 内で2日シフト後に適用
```

### 確定パラメータ

| パラメータ | 値 | 説明 |
|---|---|---|
| `target_vol` | **0.80** | ボラターゲット（実質ノイズ、⚠️下記注意参照） |
| `k_vz` | **0.30** | VIXゲート感度 |
| `gate_min` | **0.50** | VIX高騰時の最低レバレッジ係数 (50%) |
| `l_min` | 1.0 | レバレッジ下限 |
| `l_max` | 7.0 | レバレッジ上限 |

### 性能指標（DH Dynポートフォリオ統合文脈、OOS: 2021-05-08〜2026-03-26）

| 指標 | S2_VZGated | P2 baseline (tv=0.8) | 基準 |
|---|---|---|---|
| CAGR (FULL) | +32.34% | +33.80% | — |
| CAGR (IS) | +32.94% | +34.60% | — |
| CAGR (OOS) | **+27.57%** | +27.13% | — |
| Sharpe (OOS) | **0.769** | 0.757 | **>P2** ✅ |
| MaxDD (FULL) | -62.36% | -60.52% | — |
| Worst5Y | **-4.75%** | -6.63% | **>-5%** ✅ |
| IS-OOS Gap | **5.4pp** | — | **<10pp** ✅ |
| 平均Lev | 5.00x | 5.34x | — |

> ⚠️ **MaxDD -62%はDH Dynポートフォリオ全体**（NASDAQ部分が最大7x化するため）。
> CFDスリーブ単体の値ではない。実際の運用は Gold2x 20% + Bond3x 20% が緩衝材になる。

### 採用基準判定（事前定義）

| 基準 | 判定 | 根拠 |
|---|---|---|
| OOS Sharpe > P2 best (0.757) | ✅ | 0.769 > 0.757 |
| \|CAGR_IS - CAGR_OOS\| < 10pp | ✅ | \|32.94% - 27.57%\| = 5.4pp |
| Worst5Y > -5% | ✅ | -4.75% > -5% |

---

## ⚠️ 設計上の既知の制約

### target_vol パラメータは実質ノイズ（P0検証確認済み）

**検証日**: 2026-05-16 (`src/p0_verify_critical.py`)

| target_vol | ratio≥1.0の割合（全期間） | ratio中央値 |
|---|---|---|
| 0.60 | 98.6% | 4.4 |
| 0.70 | 99.2% | 5.1 |
| 0.80 | **99.7%** | **5.9** |

**原因**: NASDAQの実現ボラ中央値≈13.6%/年に対し、target_vol=0.60〜0.80は5〜7倍大きいため、
`clip(target_vol/sigma, l_min=1, l_max=7)` が常に7x付近にクリップされる。

**実態**: P2/S2は「vol-targeting」ではなく「**高ボラ時デレバ機構**」として機能している。
- 平常時（σ≈14%）: ratio≈5.7 → 7xにクリップ → 最大レバ維持
- 暴落時（σ急上昇）: ratio < 7 → デレバ発動

**影響**: target_volパラメータのグリッド探索は無意味（0.60でも0.80でも結果同じ）。
ただしデレバ機構自体は正しく機能しており、**戦略の有効性は損なわれない**。

---

## 不採用戦略メモ

| 戦略 | OOS Sharpe | Worst5Y | 不採用理由 | 再検討可能性 |
|---|---|---|---|---|
| S1_Conviction | 0.645 | -1.91% | Sharpe未達・IS-OOS gap 20.5pp | 低（構造的過学習） |
| S3_Decomposed | 0.431 | -3.77% | Sharpe大幅未達・IS-OOS gap 23.2pp | なし（廃止相当） |
| S4_RelVol | 0.697 | **-2.33%** | Sharpe未達・IS-OOS gap 14.8pp | **中**（Worst5Y改善は有望、Sharpe向上策が必要） |

**S4の再検討ポイント**: Worst5Yが-2.33%と S2比で大幅改善。
Sharpe向上のためにはl_base引き下げ（5x以下）での再チューニングが候補。
ただしIS-OOS gapが14.8ppと大きく、過学習リスクあり。

---

## CFD研究経緯（時系列）

| 日付 | レポート | 内容 |
|---|---|---|
| 2026-05-15 | [CFD_LEVERAGE_BACKTEST_2026-05-15.md](CFD_LEVERAGE_BACKTEST_2026-05-15.md) | ⛔ SUPERSEDED — 6x7x版に統合 |
| 2026-05-15 | [CFD_LEVERAGE_BACKTEST_6x7x_2026-05-15.md](CFD_LEVERAGE_BACKTEST_6x7x_2026-05-15.md) | CFD 3x〜7x固定レバの DH Dyn ポートフォリオ検証 |
| 2026-05-15 | [CFD_LEVERAGE_PROCEDURE_2026-05-15.md](CFD_LEVERAGE_PROCEDURE_2026-05-15.md) | CFDレバレッジ取引の手順ガイド（初心者向け） |
| 2026-05-15 | [CFD_YEARLY_RETURNS_2026-05-15.md](CFD_YEARLY_RETURNS_2026-05-15.md) | CFD 年次リターン表 (3x/5x + ベンチマーク) |
| 2026-05-15 | [DYN_LEVERAGE_BACKTEST_2026-05-15.md](DYN_LEVERAGE_BACKTEST_2026-05-15.md) | P1〜P5 動的レバ戦略バックテスト（S1〜S4の前身） |
| 2026-05-15 | [ENH_LEVERAGE_BACKTEST_2026-05-15.md](ENH_LEVERAGE_BACKTEST_2026-05-15.md) | ⛔ SUPERSEDED — S4追加・P0検証反映版に置換 |
| 2026-05-16 | [ENH_LEVERAGE_BACKTEST_2026-05-16.md](ENH_LEVERAGE_BACKTEST_2026-05-16.md) | **S1/S2/S3/S4 バックテスト最終版（現行正典）** |

---

## 実装ファイル

| ファイル | 役割 |
|---|---|
| `src/dynamic_leverage_strategies.py` | P1〜P5, S1〜S4 の戦略関数実装 |
| `src/enh_lev_backtest.py` | S1〜S4 グリッドサーチ・DH Dynポートフォリオ評価 |
| `src/dyn_lev_backtest.py` | P1〜P5 バックテストドライバ（S1〜S4の前身） |
| `src/p0_verify_critical.py` | P0検証スクリプト（SOFR単位・vt_multクリップ率・Worst5Y） |
| `src/cfd_leverage_backtest.py` | CFD NASDAQスリーブ・DH Dynポートフォリオ構築・評価関数 |

---

## 更新ルール

- **CFD軸のベスト戦略が変わった時のみ本ファイルを更新**する
- 実験中の中間結果・検証ログはここに書かない（各バックテストレポートに記載）
- 更新時は「最終更新日」のみ変更し、「推奨戦略」セクションを書き換える

---

*管理者: 男座員也 (Kazuya Oza)*
*関連正典: [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) (DH Dyn軸)*
