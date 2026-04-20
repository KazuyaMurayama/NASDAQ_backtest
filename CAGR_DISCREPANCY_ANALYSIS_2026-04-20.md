# CAGR乖離の原因分析レポート — DH Dyn 2x3x

**日付**: 2026-04-20
**発端**: `YEARLY_RETURNS_REPORT_2026-04-01.md` の CAGR +31.40% と `THRESHOLD_SWEEP_REPORT_2026-04-20.md` の CAGR 24.7% (IS) / 22.6% (OOS) に 6.5pp の乖離あり。

---

## 結論（先に）

**計算ミスではなく、2つのレポートが実装上 *構造的に異なる戦略* をバックテストしていた。**

両者とも同じ名前「DH Dyn 2x3x」と呼んでいるが、DD発動時のキャッシュ処理が正反対。GAS実運用と整合するのは **Approach B (24.7%)** のみ。

---

## 原因特定

### Approach A ― YEARLY_RETURNS_REPORT (31.40%)
実装: `src/step_update_dyn2x3x.py:74`

```python
dyn_nav = build_dynamic_portfolio(nav_a2.values, g2, b3, wn, wg, wb)
# wn + wg + wb = 1.0（常に100%投資）
```

- NASDAQスリーブ (`nav_a2`) だけが内部で動的にレバレッジ変動
- **DD発動時もGold 2x / Bond 3x に 45% 投資継続**
- TQQQ保有残高自体は「凍結」(daily return = 0)、売却しない想定

### Approach B ― THRESHOLD_SWEEP_REPORT / GAS実運用 (24.64%)
実装: `src/test_threshold_sweep.py:223` および `src/Code.gs`

```python
daily_ret = lev_shift * (wn*(r_naq*3 - cost) + wg*r_g2 + wb*r_b3)
# 実保有 = lev × (wn, wg, wb)、残り (1-lev) はCASH
```

- **DD発動時は全額キャッシュ** (`lev=0` → 全保有ゼロ)
- GAS production ログと完全一致 (2026-04-08 実保有: TQQQ 10.7% = 0.1901 × 0.561、CASH 81%)

---

## 実測による裏付け

同じA2シグナルを両方式で 52年通期(1974-2026)計算:

| 年 | DD状態 (avg lev) | Approach A | Approach B | 差 |
|----|----|------------|------------|-----|
| **1974** | DD全期間 (0.00) | **+21.6%** | **0.0%** | +21.6pp |
| 2000 | 一部DD (0.18) | +1.5% | -4.4% | +5.9pp |
| **2001** | DD全期間 (0.00) | **+3.6%** | **0.0%** | +3.6pp |
| **2008** | DD全期間 (0.00) | **+10.4%** | **-3.1%** | +13.5pp |
| 2022 | ほぼDD (0.03) | -19.9% | -10.3% | -9.6pp |
| **全期間CAGR** | ― | **+31.00%** | **+24.64%** | **+6.35pp** |

DD年に Approach A が大きく稼いでいる(Gold/Bond収益)のが乖離の主因。

---

## どちらが正しいか

### Approach B (GAS実運用) が物理的に実現可能な戦略

- TQQQ は固定3倍レバレッジETF。保有者が内部レバレッジを操作できない。
- DD発動時の「TQQQを保有したままレバレッジ0にする」(Approach A) は現実的に不可能。
- 実取引では DD発動 → TQQQ売却 → キャッシュ。これは Approach B と一致。

### Approach A のモデリング誤り

- `build_dynamic_portfolio()` は NASDAQ側の入力として「ナスダックスリーブNAV (=日次レバレッジ可変)」を想定。
- 3倍固定ETFを使う実装とは不整合。
- 結果、DD発動時に **架空の 45% Gold/Bond 収益** を計上。52年間で複利 +6.35pp 過大。

---

## 影響範囲

### 過大評価の疑いがある資料

1. [`YEARLY_RETURNS_REPORT_2026-04-01.md`](YEARLY_RETURNS_REPORT_2026-04-01.md) — **全CAGR値が過大** (DH系のみ)
2. [`src/step_update_dyn2x3x.py`](src/step_update_dyn2x3x.py) — 生成スクリプト
3. [`src/gen_yearly_md.py`](src/gen_yearly_md.py) — 過大CAGR をハードコード (`cagrs = {'DH Dyn 2x3x': 31.40, ...}`)
4. [`src/opt_lev2x3x.py`](src/opt_lev2x3x.py) — 同じ `build_dynamic_portfolio` 方式で Dynamic grid 比較

### 既に物理正しい資料

- [`THRESHOLD_SWEEP_REPORT_2026-04-20.md`](THRESHOLD_SWEEP_REPORT_2026-04-20.md) — GAS production と一致
- [`src/test_threshold_sweep.py`](src/test_threshold_sweep.py) — GAS production と一致
- `nasdaq-strategy-gas/src/Code.gs` — GAS実運用(正)

---

## 次の判断が必要

### 選択肢

| 案 | 内容 | 影響 |
|----|------|------|
| **A (推奨)** | YEARLY_RETURNS系資料を物理正しいCAGRで再生成。GASは現状維持 | 過去レポートを再計算・修正 |
| B | GAS側を「DD時もGold/Bond保有」戦略に変更 | 運用ロジック変更、ただし固定レバETFでは非現実的 |
| C | 両実装を別名で残し「理論値 vs 実運用値」として併記 | ドキュメント整理のみ |

推奨は **A**: Approach B が物理的に唯一実現可能なため、YEARLY_RETURNS_REPORT とハードコード値をすべて Approach B で再計算し、31.40% を **~24.6%** (正しい値) に修正するべき。

---

*作成: 2026-04-20 / 検証スクリプト: `/tmp/verify_discrepancy.py`*
