# セッションサマリー 2026-05-19

作成日: 2026-05-19
最終更新日: 2026-05-19

> このドキュメントは新セッション開始時の引き継ぎ用。
> 参照順序: このファイル → [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) → [tasks.md](tasks.md)

---

## TL;DR（次セッション向け1分サマリー）

- **CFD軸のベスト戦略**: `S2_VZGated` (k_vz=0.3, gate_min=0.5, target_vol=0.8) — 採用確定
- **DH Dyn軸のベスト戦略**: `DH Dyn 2x3x [A]` (threshold=0.15) — CAGR +22.50%（2026-05-12 Gold補正後）
- **主要レポート**: [ENH_LEVERAGE_BACKTEST_2026-05-16.md](ENH_LEVERAGE_BACKTEST_2026-05-16.md) / [CFD_S2_YEARLY_RETURNS_2026-05-17.md](CFD_S2_YEARLY_RETURNS_2026-05-17.md)
- **ブランチ**: `claude/review-best-strategy-Jcjd5`（全作業はこのブランチ）

---

## このセッション（2026-05-19）でやったこと

### 1. S2_VZGated 年次リターンレポート生成

- `src/gen_s2_yearly_returns.py` を新規作成・実行
- 対象5戦略: S2_VZGated / CFD 3x固定 / CFD 7x固定 / DH Dyn 2x3x [A] / BH 1x
- FULL/IS/OOS3期間統計 + 年次リターン表（1974-2026、[OOS]マーカー付き）
- 出力: `CFD_S2_YEARLY_RETURNS_2026-05-17.md`

### 2. CFD 固定レバ定義バグ修正 → 再生成

- **バグ**: `gen_s2_yearly_returns.py` 初版が "CFD 3x/7x [固定]" を **純NASDAQのみ**（DH Dynなし）で計算
  - CFD 3x: +9.74% (誤) → **+23.20%** (正)
  - CFD 7x: -39.27% (誤) → **+41.36%** (正)
- **修正**: `ENH_LEVERAGE_BACKTEST_2026-05-15.md` と同じ定義 = DH Dyn(A) ポートフォリオ + 固定CFDレバ
- サニティチェック4件全PASS後、再push

### 3. 前セッション（2026-05-16/17）の継続確認

前セッションから引き継いでいた「S2_VZGated 年次リターン未作成」タスクを完了。

---

## CFD軸 現行ベスト: S2_VZGated 確定パラメータ

```
戦略名:     S2_VZGated
パラメータ: target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5
ポート:     NASDAQ CFD (動的L_t) + Gold 2x 20% + Bond 3x 20%
実装:       src/dynamic_leverage_strategies.py::compute_L_s2_vz_gated()
```

### S2_VZGated 採用根拠（採用基準3件全クリア）

| 基準 | 要件 | S2_VZGated | 判定 |
|------|------|-----------|------|
| OOS Sharpe | > P2 0.757 | **0.769** | ✅ |
| IS-OOS gap | < 10pp | **5.4pp** | ✅ |
| Worst5Y | > -5% | **-4.75%** | ✅ |

---

## 全戦略バックテスト結果サマリー（CFD軸、FULL期間）

| 戦略 | CAGR_IS | CAGR_OOS | Sharpe_OOS | MaxDD | Worst5Y | Worst10Y | 平均Lev | 採用 |
|------|---------|---------|-----------|-------|---------|----------|---------|------|
| **S2_VZGated** | +32.94% | +27.57% | **0.769** | -62.4% | -4.75% | +13.36% | 5.0x | ✅ |
| P2 best (tv=0.8) | +34.60% | +27.13% | 0.757 | -60.5% | -6.63% | +14.67% | 5.3x | baseline |
| S4_RelVol | +40.98% | +26.19% | 0.697 | -66.1% | -2.33% | +13.35% | 6.1x | ❌ |
| S1_Conviction | +43.00% | +22.47% | 0.645 | -64.3% | -1.91% | +18.28% | 4.8x | ❌ |
| S3_Decomposed | +32.80% | +9.60% | 0.431 | -54.8% | -3.77% | +9.85% | 3.1x | ❌ |
| CFD 7x [固定] | +43.35% | +24.44% | 0.670 | -65.0% | -5.24% | +17.77% | 7.0x | ref |
| CFD 3x [固定] | +24.07% | +15.57% | 0.670 | -44.8% | +1.43% | +10.84% | 3.0x | ref |
| DH Dyn 2x3x [A] | +23.36% | +14.88% | 0.646 | -45.1% | +0.87% | — | — | ref |
| BH 1x | +11.13% | +10.11% | 0.540 | -77.9% | -16.77% | — | — | bench |

> ⚠️ 全戦略はDH Dyn(A)ポートフォリオ（Gold 2x 20%, Bond 3x 20%）を使用。CFD固定レバも同様。

---

## P0 検証結果（前セッション 2026-05-16 確定）

| 検証項目 | 結果 |
|---------|------|
| SOFR単位（DTB3が年率%表記か） | ✅ 正しい (2023≈5.07%/yr, 1981≈14.02%/yr) |
| target_volパラメータのクリップ率 | ⚠️ 99.7%がl_max=7にクリップ（実質死パラメータ）|
| Worst5Y定義 | ✅ 正しい (ns_f.shift(252×5)による52年ローリング) |

---

## 既知の問題・未解決

1. **S1_Conviction 数値不整合疑い**: 再実行スクリプトで CAGR=+11.58% が出るが ENH レポートは +40.88%。ENHレポート値が正しい可能性が高いが原因未特定。
2. **target_vol 死パラメータ問題**: P2/S2 の target_vol=0.60-0.80 は NASDAQ ボラ中央値≈13.6%に対し99%+クリップ。実態は「高ボラ時デレバ機構」。真のvol-targetingには target_vol ∈ {0.10, 0.20} が必要（P2*/S2* 再設計タスクで対処予定）。

---

## 未完了タスク（優先順）

### CFD軸
- [ ] P2*/S2* 再設計: target_vol ∈ {0.10, 0.13, 0.16, 0.20, 0.25} で再グリッドサーチ
- [ ] S4_RelVol の Sharpe 改善 (l_base=5以下・IS-OOS gap 14.8pp 縮小)
- [ ] S2 を DH Dynポートフォリオ正式統合シナリオ試算

### DH Dyn軸
- [ ] Approach A の GAS 切替実装 (threshold=0.15 と同時)
- [ ] 2026年データへの拡張（継続監視）
- [ ] Ens2 戦略の OOS 検証 (2022-2026)

---

## 主要ファイル一覧（このセッション関連）

| ファイル | 内容 |
|---------|------|
| [CFD_DYNAMIC_LEVERAGE_GUIDE.md](CFD_DYNAMIC_LEVERAGE_GUIDE.md) | CFD軸カノニカル文書（S2_VZGated採用確定、式・実装・性能）|
| [ENH_LEVERAGE_BACKTEST_2026-05-16.md](ENH_LEVERAGE_BACKTEST_2026-05-16.md) | S1/S2/S3/S4 グリッドサーチ結果（採用判定付き）|
| [CFD_S2_YEARLY_RETURNS_2026-05-17.md](CFD_S2_YEARLY_RETURNS_2026-05-17.md) | S2年次リターン表（FULL/IS/OOS統計、[OOS]マーカー付き）|
| [CFD_YEARLY_RETURNS_2026-05-15.md](CFD_YEARLY_RETURNS_2026-05-15.md) | CFD固定レバ 3x/4x/5x 年次リターン表|
| [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) | 全軸のベスト戦略正典（DH Dyn軸: CAGR +22.50%）|
| [src/dynamic_leverage_strategies.py](src/dynamic_leverage_strategies.py) | P1-P5, S1-S4 全戦略の L_t 計算関数 |
| [src/enh_lev_backtest.py](src/enh_lev_backtest.py) | S1-S4 グリッドサーチ・採用判定スクリプト |
| [src/gen_s2_yearly_returns.py](src/gen_s2_yearly_returns.py) | S2年次リターンレポート生成スクリプト |

---

## 新セッション開始時のコマンド

```bash
git checkout claude/review-best-strategy-Jcjd5
git pull origin claude/review-best-strategy-Jcjd5
cat SESSION_SUMMARY_2026-05-19.md
cat tasks.md
```

---

*前セッションサマリー: [SESSION_SUMMARY_2026-05-16.md](SESSION_SUMMARY_2026-05-16.md)*
