# 過去ベスト戦略 + NDX 1x B&H 追加計画 — `STRATEGY_PERFORMANCE_COMPARISON_20260529.md` v3.1

> **本計画の目的**: v3 比較表（21戦略）に **過去のベスト戦略4件** と **NDX 1x B&H ベンチマーク** を、**現行 SBI CFD 前提（SOFR+3.0% + −0.66% + ×0.8273）に統一**して追加し、過去・現在のベスト戦略を同じ土俵で実コスト/税後比較できるようにする。

作成日: 2026-05-29
管理: Claude (Opus 4.7) / 承認待ち: Kazuya
基準: `EVALUATION_STANDARD.md v1.3` / §3.12 9指標標準 / `_sweep_format.MD_HEADER_STRAT`
対象ファイル: [`STRATEGY_PERFORMANCE_COMPARISON_20260529.md`](STRATEGY_PERFORMANCE_COMPARISON_20260529.md)

---

## 🎯 Goal

v3 §2 統合比較表に **5行追加**（22〜26行目）し、過去ベスト戦略の実コスト/税後パフォーマンスを正しく可視化する。

## 🏗 Architecture

- 新規スクリプト `src/g15_legacy_strategies_realistic.py` で 4戦略 + B&H の 9指標を一括計算
- `g13_realistic_cost_full_comparison.py` と同一のコスト/税モデル（SOFR+3.0% + −0.66% + ×0.8273）を再利用
- `_sweep_format.MD_HEADER_STRAT` 準拠の行を生成し、v3 §2 表に append
- B&H は CFD/レバ非該当のため **コスト=0、税のみ ×0.8273**、Trades/yr=0、Overfit/CI95=—

## 🛠 Tech Stack

- Python 3.x + pandas + numpy
- `src/backtest_engine.py`, `src/corrected_strategy_backtest.py`, `src/dynamic_leverage_strategies.py`, `src/long_cycle_signal.py`, `src/cfd_leverage_backtest.py`, `src/test_ens2_strategies.py` の既存関数を再利用
- `src/_sweep_format.py` の `MD_HEADER_STRAT` / `fmt_row_strat` を必須使用（手書きヘッダ禁止 v1.3）

---

## 📁 File Structure

| 操作 | パス | 役割 |
|---|---|---|
| Create | `src/g15_legacy_strategies_realistic.py` | 4戦略 + B&H 9指標一括算出 |
| Create | `g15_legacy_results.csv` | 各戦略 raw + cost後 + tax後 9指標 |
| Create | `audit_results/BNH_NDX_1X_AUDIT_2026-05-29.md` | B&H 計算根拠（短い監査ログ） |
| Modify | `STRATEGY_PERFORMANCE_COMPARISON_20260529.md` | §2 表に5行 append、§8 履歴に v3.1 追記、§7 一次根拠に g15 追記 |

---

## 🎯 追加対象 5戦略（最終確定）

| # | 戦略名 | 種別 | 主要参照 | 既知 raw 指標 (BT) |
|:--:|---|---|---|---|
| 22 | **S2_VZGated + LT2-N750 k=0.5 modeB ‡** | 過去ベスト (廃止 2026-05-24) | `b1_s2_lt2_results.csv` / `b1_s2_lt2.py` | Sharpe +0.858, CAGR_OOS +31.16%, MaxDD −59.45%, Trades 27, CI95_lo +25.7% |
| 23 | **S2_VZGated 単独 ‡** | 過去ベスト (廃止 2026-05-21) | `cfd_leverage_backtest.py` + `compute_L_s2_vz_gated` 直接実行 | CAGR_OOS +27.57%, Sharpe_OOS 0.769 |
| 24 | **DH Dyn 2x3x [A] ‡** | 過去ベスト (廃止 2026-05-21) | `corrected_strategy_backtest.py` + `simulate_rebalance_A` 直接実行 | CAGR +22.50%, Sharpe +0.993 |
| 25 | **Ens2(Asym+Slope) ‡** | 過去ベスト (廃止 2026-04-21) | `test_ens2_strategies.py` + `ens2_comparison_results.csv` | CAGR +28.58%, Sharpe +1.031 |
| 26 | **NDX 1x B&H 🅑** | ベンチマーク (新規) | `data/NASDAQ_extended_to_2026.csv` 直接読込 | (新規算出) |

凡例:
- `‡` = §1.3 参考値（過去ベスト・採用候補外）
- `🅑` = ベンチマーク（戦略ではなく比較基準）

---

## ❗ 留意事項（実装前に必ず認識）

1. **DH Dyn / Ens2 は CFD 非前提の戦略** — 元実装は TQQQ/TMF/UGL ETF ベース。本表は「**仮に SBI CFD で実装したらどうなるか**」の仮想評価とする。MD 注記必須。
2. **B&H に Sharpe 概念は適用可だが、Trades/yr=0 / Overfit(WFE)=— / CI95_lo=—** とする（WFA は B&H に意味なし）。
3. **税モデル**: 全戦略 §3-A（`(CAGR − 0.66%) × 0.8273`）統一。B&H も同モデル（長期繰延税金の近似誤差は数%以内）。
4. **コスト適用** は v3 §1 と完全一致: SBI CFD = SOFR+3.0% + 未含コスト −0.66%。B&H は 0 + 0。
5. **再現性確保**: 既存 BT raw 値を可能な限り CSV から読込み、新規実行は最小限に。
6. **`MD_HEADER_STRAT` を import 必須**。手書きヘッダ禁止（v1.3 違反）。
7. **`CAGR_IS / CAGR_FULL` を MD ヘッダに含めない**（v1.1 違反）。CSV 保存は OK。

---

## 📋 Phase 1: データインベントリ（事前確認）

### Task 1: 既存 CSV から raw 値を抽出

**Files:**
- Read: `b1_s2_lt2_results.csv`（S2+LT2 sweep）
- Read: `ens2_comparison_results.csv`（Ens2 sweep）

- [ ] **Step 1.1**: `b1_s2_lt2_results.csv` を pandas で読み、`N=750, k_lt=0.5, modeB` 行の **全9指標**（CAGR_OOS / Sharpe_OOS / MaxDD / Worst10Y / P10_5Y / IS-OOS gap / Trades/yr）が含まれるかを確認。欠損があれば再実行候補に分類。

実行例（Python REPL）:
```python
import pandas as pd
df = pd.read_csv('b1_s2_lt2_results.csv')
print(df.columns.tolist())
print(df[(df['N']==750) & (df['k_lt']==0.5)].to_dict('records'))
```

- [ ] **Step 1.2**: `ens2_comparison_results.csv` で `Ens2(Asym+Slope)` 行を同様に確認。

- [ ] **Step 1.3**: 欠損指標リスト（特に **Worst10Y★ カレンダー年方式 / P10_5Y▷ / IS-OOS gap**）を作成。再計算が必要な指標があれば Phase 2 で実装。

期待出力: 各戦略の raw 値マップ（dict）と「再実行が必要な戦略」リスト。

- [ ] **Step 1.4**: コミット不要（読み込み調査のみ）

---

## 📋 Phase 2: コスト/税再計算スクリプト本体

### Task 2: `g15_legacy_strategies_realistic.py` スケルトン作成

**Files:**
- Create: `src/g15_legacy_strategies_realistic.py`

- [ ] **Step 2.1**: ファイルヘッダ docstring + import を書く。

```python
"""
g15_legacy_strategies_realistic.py
====================================
過去ベスト戦略4件 + NDX 1x B&H ベンチマークを SBI CFD 前提（SOFR+3.0%）で
再シミュレーションし、v3 比較表 (§2) に追加する行を生成する。

対象:
  1. S2_VZGated + LT2-N750 k=0.5 modeB (過去ベスト, 廃止 2026-05-24)
  2. S2_VZGated 単独 (過去ベスト, 廃止 2026-05-21)
  3. DH Dyn 2x3x [A] (過去ベスト, 廃止 2026-05-21)
  4. Ens2(Asym+Slope) (過去ベスト, 廃止 2026-04-21)
  5. NDX 1x B&H (ベンチマーク, 新規)

コストモデル: SBI CFD SOFR+3.0%（g13 と同一）
税モデル: §3-A `(CAGR - 0.66%) × 0.8273`（v3 と同一）
B&H 特例: コスト=0, 税のみ ×0.8273, Trades/yr=0, Overfit/CI95=—
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr, build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_a2_signal, simulate_rebalance_A,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import build_nav_strategy, calc_7metrics
from compute_cfd_worst10y import nav_to_annual, rolling_nY_cagr
from long_cycle_signal import build_lt_signal, apply_lt_mode_b
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from a2_dyn_lmax import compute_p10_5y, calc_all_metrics, S2_BASE, K_MID
from _sweep_format import MD_HEADER_STRAT, fmt_row_strat

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# コスト/税定数 (v3 §1 と同一)
# ---------------------------------------------------------------------------
SBI_CFD_SPREAD = 0.0300        # SOFR + 3.0%/yr 業者マージン
UNINCLUDED_COST = 0.0066       # −0.66%/yr 未含コスト補正
JP_TAX_MULT = 0.8273           # ×0.8273 §3-A モデル

# 期間境界 (§2)
OOS_START = pd.Timestamp('2021-05-08')
OOS_END   = pd.Timestamp('2026-03-26')
```

- [ ] **Step 2.2**: ヘルパー関数 `apply_cost_tax(cagr_raw, is_bnh=False)` を実装。

```python
def apply_cost_tax(cagr_raw: float, is_bnh: bool = False) -> tuple:
    """raw CAGR → コスト後 → 手取り。

    Returns: (cagr_after_cost, cagr_after_tax)
    is_bnh=True の場合: コスト適用なし（B&H 特例）
    """
    if is_bnh:
        cost_drag = 0.0
    else:
        # SOFR+3.0% は build_nav_strategy 内で適用済み前提
        # g15 では CSV から既コスト後 CAGR を読む or build_nav で計算後の値を渡す
        cost_drag = 0.0  # 呼び出し側で既コスト後 CAGR を渡す
    cagr_after_cost = cagr_raw - cost_drag
    cagr_after_tax  = (cagr_after_cost - UNINCLUDED_COST) * JP_TAX_MULT
    return cagr_after_cost, cagr_after_tax
```

- [ ] **Step 2.3**: 動作確認 — Python REPL で `import g15_legacy_strategies_realistic` がエラーなく通ることを確認。

実行: `python -c "import sys; sys.path.insert(0,'src'); import g15_legacy_strategies_realistic"`
期待: 例外なし。

- [ ] **Step 2.4**: コミット
```bash
git add src/g15_legacy_strategies_realistic.py
git commit -m "feat(g15): skeleton for legacy strategy + B&H realistic comparison"
```

---

### Task 3: S2_VZGated 単独 行を実装

**Files:**
- Modify: `src/g15_legacy_strategies_realistic.py`

- [ ] **Step 3.1**: `compute_s2_vzgated_alone()` 関数を追加。`g13` の S2 セクションを参考に、LT overlay なしの純 S2 VZ ゲート版を SOFR+3.0% で実行。

```python
def compute_s2_vzgated_alone():
    """S2_VZGated 単独（LT overlay なし）を SBI CFD 前提で実行。"""
    data = load_data(DATA_PATH)
    sofr = load_sofr()
    # ... build_a2_signal で DH Dyn シグナル生成 (LT 重畳なし)
    # ... compute_L_s2_vz_gated でレバ生成 (l_max=7.0)
    # ... build_nav_strategy(spread=SBI_CFD_SPREAD) で NAV
    # ... calc_all_metrics(nav, IS/OOS/FULL) で9指標
    return metrics_dict
```

- [ ] **Step 3.2**: 関数を `__main__` ブロックから呼び、戻り値を print。値が CURRENT_BEST_STRATEGY.md の旧表値（CAGR_OOS +27.57%, Sharpe 0.769）と桁が一致することを確認（コスト変更で多少ずれる）。

- [ ] **Step 3.3**: コミット
```bash
git add src/g15_legacy_strategies_realistic.py
git commit -m "feat(g15): S2_VZGated alone metric computation"
```

---

### Task 4: S2_VZGated + LT2-N750 k=0.5 modeB 行を実装

**Files:**
- Modify: `src/g15_legacy_strategies_realistic.py`

- [ ] **Step 4.1**: `compute_s2lt2_fixed_k05()` 関数を追加。E4 の `_k_lt_regime` を **固定 k=0.5** に置換した版。

```python
def compute_s2lt2_fixed_k05():
    """S2_VZGated + LT2-N750 modeB（固定 k_lt=0.5、レジーム条件なし）。"""
    # E4 と同じ S2_FIXED_7 + LT_N=750 構成だが k_lt 固定
    # apply_lt_mode_b(lev_A, lt_sig, k=0.5) で重畳
    # 以降は S2 アロケに合流
    return metrics_dict
```

- [ ] **Step 4.2**: 戻り値の Sharpe が +0.858 付近に収束することを確認（CURRENT_BEST_STRATEGY.md Shortlisted 表値）。

- [ ] **Step 4.3**: コミット
```bash
git add src/g15_legacy_strategies_realistic.py
git commit -m "feat(g15): S2+LT2 fixed-k=0.5 modeB metric computation"
```

---

### Task 5: DH Dyn 2x3x [A] 行を実装

**Files:**
- Modify: `src/g15_legacy_strategies_realistic.py`

- [ ] **Step 5.1**: `compute_dhdyn_2x3x_A()` を追加。**LT overlay なし、S2 VZ ゲートなし**、純粋な DH Dyn シグナル × `simulate_rebalance_A`（TQQQ 3x + TMF 3x + UGL 2x ETF 配分）の NAV を生成。CFD ではないので **「仮に CFD で実装したら」の仮想評価**注記を必須に。

```python
def compute_dhdyn_2x3x_A():
    """DH Dyn 2x3x [A] (TQQQ+TMF+UGL ETF 想定)。

    SBI CFD 前提では NAS のみ CFD で TMF/UGL は ETF のまま。
    本実装は『仮に NAS 軸を SBI CFD で実装したら』のハイブリッド評価。
    """
    # build_a2_signal で DH Dyn シグナル
    # simulate_rebalance_A で TQQQ+TMF+UGL 配分（CFD ではなく ETF）
    # NAS 軸のみ SBI CFD ファイナンスコストに置換
    return metrics_dict
```

- [ ] **Step 5.2**: 値が CURRENT_BEST_STRATEGY.md 旧表（CAGR 22.50%, Sharpe 0.993）と桁が一致することを確認。

- [ ] **Step 5.3**: コミット
```bash
git add src/g15_legacy_strategies_realistic.py
git commit -m "feat(g15): DH Dyn 2x3x [A] metric (hybrid CFD assumption)"
```

---

### Task 6: Ens2(Asym+Slope) 行を実装

**Files:**
- Modify: `src/g15_legacy_strategies_realistic.py`

- [ ] **Step 6.1**: `test_ens2_strategies.py` から Asym+Slope アンサンブル生成ロジックを import or 移植。NAV を SBI CFD 前提で再構築。

```python
def compute_ens2_asym_slope():
    """Ens2(Asym+Slope) — 旧 AsymEWMA + SlopeMult アンサンブル。

    元実装は test_ens2_strategies.py。SBI CFD 前提で NAV 再構築。
    DH Dyn 系より古く、現行 a2_signal とロジック互換性に注意。
    """
    # test_ens2_strategies の calc_asym_ewma_vol + calc_slope_multiplier
    # アンサンブル → SBI CFD ファイナンス適用 → NAV
    return metrics_dict
```

- [ ] **Step 6.2**: 値が CURRENT_BEST_STRATEGY.md 旧表（CAGR 28.58%, Sharpe 1.031）と桁が一致することを確認。互換性問題で再現困難な場合は **CSV から raw を取りつつ、コスト/税のみ後付け換算**にフォールバック（注記必須）。

- [ ] **Step 6.3**: コミット
```bash
git add src/g15_legacy_strategies_realistic.py
git commit -m "feat(g15): Ens2(Asym+Slope) metric"
```

---

### Task 7: NDX 1x Buy & Hold ベンチマーク行を実装

**Files:**
- Modify: `src/g15_legacy_strategies_realistic.py`
- Create: `audit_results/BNH_NDX_1X_AUDIT_2026-05-29.md`

- [ ] **Step 7.1**: `compute_ndx_bnh_1x()` を追加。`data/NASDAQ_extended_to_2026.csv` を読込、終値リターンで NAV を構築（レバ無し、リバランス無し、コスト無し）。

```python
def compute_ndx_bnh_1x():
    """NDX 1x Buy & Hold ベンチマーク。

    純粋指数リターン。コスト=0、税は §3-A モデル（×0.8273, −0.66% は適用しない）。
    Trades/yr=0, Overfit(WFE)=None, CI95_lo=None
    """
    data = load_data(DATA_PATH)
    ret = data['Close'].pct_change().fillna(0)
    nav = (1 + ret).cumprod()

    # IS / OOS / FULL の3区間で指標算出
    def slice_metrics(nav_seg):
        cagr = (nav_seg.iloc[-1] / nav_seg.iloc[0]) ** (TRADING_DAYS / len(nav_seg)) - 1
        # Sharpe Rf=0
        daily_ret = nav_seg.pct_change().dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS)
        maxdd = (nav_seg / nav_seg.cummax() - 1).min()
        return {'CAGR': cagr, 'Sharpe': sharpe, 'MaxDD': maxdd}

    full = slice_metrics(nav)
    is_seg = slice_metrics(nav.loc[:'2021-05-07'])
    oos_seg = slice_metrics(nav.loc[OOS_START:])

    # Worst10Y★ カレンダー年方式
    worst10y_bh = compute_calendar_worst10y(nav)
    # P10_5Y▷
    p10_5y_bh = compute_p10_5y(nav)
    # IS-OOS gap
    isoos_gap = oos_seg['CAGR'] - is_seg['CAGR']

    # 税適用
    cagr_oos_net = oos_seg['CAGR'] * JP_TAX_MULT  # B&H: −0.66% 未含コスト適用なし
    worst10y_net = worst10y_bh * JP_TAX_MULT
    p10_5y_net   = p10_5y_bh * JP_TAX_MULT

    return {
        'Strategy': 'NDX 1x B&H 🅑',
        'CAGR_OOS_net': cagr_oos_net,
        'Sharpe_OOS': oos_seg['Sharpe'],
        'MaxDD': full['MaxDD'],
        'Worst10Y_net': worst10y_net,
        'P10_5Y_net': p10_5y_net,
        'IS_OOS_gap': isoos_gap,
        'Trades_yr': 0,
        'Overfit_WFE': None,
        'CI95_lo': None,
    }
```

- [ ] **Step 7.2**: 監査用 MD を作成（計算根拠の透明性確保）。

`audit_results/BNH_NDX_1X_AUDIT_2026-05-29.md` の最小内容:
```markdown
# NDX 1x B&H ベンチマーク 計算根拠 (2026-05-29)

データ: `data/NASDAQ_extended_to_2026.csv` (13,169 bars, 1974-01-02 → 2026-03-26)
方式: 終値リターン × 1x (レバ無し、リバランス無し、配当除外、コスト=0)
税: §3-A `× 0.8273` のみ（−0.66% 未含コストは B&H 非該当のため適用なし）

| 指標 | raw | 税後 |
|---|---:|---:|
| CAGR_OOS | (実値) | (実値 × 0.8273) |
| Sharpe_OOS (Rf=0) | (実値) | 不変 |
| MaxDD (FULL) | (実値) | 不変 |
| Worst10Y★ (カレンダー年) | (実値) | (実値 × 0.8273) |
| P10_5Y▷ | (実値) | (実値 × 0.8273) |
| IS-OOS gap | (実値) | 不変 |
| Trades/yr | 0 | 0 |
| Overfit(WFE) | — | — |
| CI95_lo | — | — |

留意:
- B&H 税は本来「売却時のみ実現」だが、§3-A 比例モデルで近似（誤差 ≤ 数%）
- 配当再投資なし（NDX 配当 ≈ 0.6%/yr の上振れ要因あり、本表は保守）
- NISA 適用なし前提（CFD と同等の課税条件で比較するため）
```

- [ ] **Step 7.3**: 値が直感的に妥当か検証。NDX 1974-2026 の CAGR は概ね 12-13%/yr、OOS (2021-2026) は 14-16%/yr のはず。MaxDD は 1974 / 2000 / 2008 / 2022 のいずれかで −80% 超なら異常 → 確認。

- [ ] **Step 7.4**: コミット
```bash
git add src/g15_legacy_strategies_realistic.py audit_results/BNH_NDX_1X_AUDIT_2026-05-29.md
git commit -m "feat(g15): NDX 1x B&H benchmark + audit log"
```

---

### Task 8: 5戦略を統合実行・CSV/MD 出力

**Files:**
- Modify: `src/g15_legacy_strategies_realistic.py`
- Create: `g15_legacy_results.csv`

- [ ] **Step 8.1**: `__main__` で 5 関数を順次呼び、結果を dict のリストに集約。

- [ ] **Step 8.2**: CSV 保存（raw + cost後 + tax後 全て、デバッグ/監査用）。

```python
if __name__ == '__main__':
    results = [
        compute_s2lt2_fixed_k05(),
        compute_s2_vzgated_alone(),
        compute_dhdyn_2x3x_A(),
        compute_ens2_asym_slope(),
        compute_ndx_bnh_1x(),
    ]
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(BASE, 'g15_legacy_results.csv'), index=False)
    print(df.to_string())
```

- [ ] **Step 8.3**: MD 行生成 — `fmt_row_strat` を使用してフォーマット。v3 §2 表の列順（`Strategy | CAGR⓽ | Sharpeⓒ | MaxDDⓒ | Worst10Y★⓽ | P10⓽ 5Y | IS-OOSⓒ gap | Tradeⓞ | Overfitⓞ | CI95ⓔ`）に従う。

```python
    print('\n--- MD ROWS (v3 §2 末尾に append) ---')
    for r in results:
        print(fmt_row_strat(
            r['Strategy'],
            r['CAGR_OOS_net'],
            r['Sharpe_OOS'],
            r['MaxDD'],
            r['Worst10Y_net'],
            r['P10_5Y_net'],
            r['IS_OOS_gap'],
            r['Trades_yr'],
            r['Overfit_WFE'],
            r['CI95_lo'],
        ))
```

- [ ] **Step 8.4**: 実行 — `cd src && python g15_legacy_strategies_realistic.py`。期待: CSV 生成 + MD 行5本標準出力。

- [ ] **Step 8.5**: 出力 MD 行を目視確認 — `CAGR⓽_OOS` 列が手取り値（×0.8273 適用済み）になっているか、`Worst10Y★` が ⓽（手取り）か、Trades/yr が 0 (B&H) か、Overfit/CI95 が `—` (B&H) か。

- [ ] **Step 8.6**: コミット
```bash
git add src/g15_legacy_strategies_realistic.py g15_legacy_results.csv
git commit -m "feat(g15): execute legacy strategies + B&H, output CSV"
```

---

## 📋 Phase 3: v3 比較表への追記

### Task 9: `STRATEGY_PERFORMANCE_COMPARISON_20260529.md` を更新

**Files:**
- Modify: `STRATEGY_PERFORMANCE_COMPARISON_20260529.md`:
  - Line 144 直後（最終戦略行 `[D5] vz=0.70/lmax=5.0` の下）に 5行 append
  - Line 146 (注記行) を更新（`🅑` ベンチマーク記号と「過去ベスト戦略 ‡」の意味追加）
  - §7 (一次根拠) に `src/g15_legacy_strategies_realistic.py` / `g15_legacy_results.csv` / `audit_results/BNH_NDX_1X_AUDIT_2026-05-29.md` を追記
  - §8 (改訂履歴) に v3.1 行を追加
  - §2 見出しの「21 戦略 × 9 指標 / 10 列」→ 「26 戦略 × 9 指標 / 10 列（過去ベスト4 + ベンチマーク1 含む）」に更新

- [ ] **Step 9.1**: g15 実行で得た MD 行を line 145 の前に Edit ツールで挿入。

`old_string` (line 144-145 周辺):
```markdown
| [D5] vz=0.70/lmax=5.0 | +20.7% | +0.80 M | -55.2% |  +9.9% |  +5.7% | -0.87pp |  27 | ✅ LOW<br>(1.2) | +0.16 |

*Sharpeマーカー: ...
```

`new_string`:
```markdown
| [D5] vz=0.70/lmax=5.0 | +20.7% | +0.80 M | -55.2% |  +9.9% |  +5.7% | -0.87pp |  27 | ✅ LOW<br>(1.2) | +0.16 |
| **— 過去ベスト戦略（参考値・採用候補外）—** |  |  |  |  |  |  |  |  |  |
| [Legacy] S2_VZGated + LT2-N750 k=0.5 modeB ‡ | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | 27 | (g15値) | (g15値) |
| [Legacy] S2_VZGated 単独 ‡ | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) |
| [Legacy] DH Dyn 2x3x [A] ‡ | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) |
| [Legacy] Ens2(Asym+Slope) ‡ | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) |
| **— ベンチマーク —** |  |  |  |  |  |  |  |  |  |
| **NDX 1x Buy & Hold 🅑** | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | (g15値) | 0 | — | — |

*Sharpeマーカー: ...
```

- [ ] **Step 9.2**: line 146 注記に `🅑` 説明追加。`*‡ = §1.3 参考値戦略（採用候補外）.*` の隣に `*🅑 = ベンチマーク（戦略ではなく比較基準）。コスト=0、税のみ適用。*` 追記。

- [ ] **Step 9.3**: §7 一次根拠表（line 247以降）に3行追加。

- [ ] **Step 9.4**: §8 改訂履歴（line 346-350 周辺）に v3.1 行追加:
```markdown
| **v3.1-draft** | 2026-05-29 | **過去ベスト戦略4件 + NDX 1x B&H ベンチマーク追加**。S2_VZGated+LT2 k=0.5 / S2_VZGated 単独 / DH Dyn 2x3x [A] / Ens2(Asym+Slope) を SBI CFD 前提（SOFR+3.0%）で再計算し §2 §22-25 に追加。B&H は §26 にベンチマークとして追加（コスト=0、税のみ）。DH Dyn / Ens2 は ETF ベース戦略のため『仮 CFD 実装』の仮想評価注記付き。 |
```

- [ ] **Step 9.5**: §2 見出し更新（line 108）。

`old_string`: `## 📊 §2 全戦略 統合比較表（手取りベース・改訂版）— 21 戦略 × 9 指標 / 10 列`

`new_string`: `## 📊 §2 全戦略 統合比較表（手取りベース・改訂版）— 26 戦略 × 9 指標 / 10 列（過去ベスト4 + ベンチマーク1 含む）`

- [ ] **Step 9.6**: コミット
```bash
git add STRATEGY_PERFORMANCE_COMPARISON_20260529.md
git commit -m "docs(strategy-comparison): v3.1 add 4 legacy strategies + NDX 1x B&H benchmark"
```

---

## 📋 Phase 4: 検証 + Push + 報告

### Task 10: 整合性最終検証

- [ ] **Step 10.1**: MD ファイルを Read で再読。
  - 5行が正しく22-26行目（テーブル内）に挿入されているか
  - 列数が10列で一致しているか
  - `Strategy` 列の `‡` / `🅑` マークが反映されているか
  - `Trades/yr` 列が B&H で 0、その他で実値が入っているか
  - `Overfit(WFE)` / `CI95_lo` が B&H で `—`、その他で値（または `—` の場合は理由注記）

- [ ] **Step 10.2**: §RISK セクションを再確認し、過去戦略の参考値性とB&Hベンチマーク性に関する追加注記が必要なら最後の §RISK-12 として追加検討（任意）。

### Task 11: Push + GitHub 上での URL 検証

- [ ] **Step 11.1**: `git push origin main`
- [ ] **Step 11.2**: `git rev-parse --abbrev-ref HEAD` でブランチ取得
- [ ] **Step 11.3**: `gh api repos/KazuyaMurayama/NASDAQ_backtest/contents/STRATEGY_PERFORMANCE_COMPARISON_20260529.md?ref=main` で存在確認
- [ ] **Step 11.4**: 同様に `g15_legacy_results.csv`, `src/g15_legacy_strategies_realistic.py`, `audit_results/BNH_NDX_1X_AUDIT_2026-05-29.md` を確認

### Task 12: 成果物 3列表でユーザー報告（CLAUDE.md 必須フォーマット）

| 成果物 | 説明 | リンク |
|---|---|---|
| STRATEGY_PERFORMANCE_COMPARISON_20260529.md | v3.1 — 26戦略へ拡張 | (URL検証後) |
| src/g15_legacy_strategies_realistic.py | 過去戦略+B&H 9指標算出 | (URL検証後) |
| g15_legacy_results.csv | 算出結果 raw | (URL検証後) |
| audit_results/BNH_NDX_1X_AUDIT_2026-05-29.md | B&H 監査ログ | (URL検証後) |

---

## ⚠️ Decision Points（実装中に判断保留中の点）

| # | 判断項目 | デフォルト方針 | 代替案 |
|:--:|---|---|---|
| D1 | DH Dyn 2x3x [A] の CFD 化スコープ — NAS のみ CFD か、TMF/UGL も CFD で代替か | NAS のみ CFD、TMF/UGL は ETF のまま（部分 CFD） | 全資産を「仮想 CFD」化（公平比較性は上がるが現実性は下がる） |
| D2 | Ens2 が現行 a2_signal と非互換なら | CSV から raw を取りつつ後付けでコスト/税のみ換算（保守的） | 完全に再実装し直す（時間がかかる） |
| D3 | B&H に Sharpe 高水準マーカ H/M を付与するか | 付与しない（戦略ではないので比較対象外） | 付与する（基準として明示） |
| D4 | 過去戦略の `‡` 行を §5 ランキング表にも反映するか | §2 のみに追加、§5 ランキングは現行Active候補のみ維持 | §5 にも参考値として末尾追加 |

→ Phase 2-3 実行中に方針を Quick decision で固める。

---

## 📊 想定スケジュール

| Phase | 想定時間 | 注 |
|---|---:|---|
| Phase 1 (インベントリ) | 10分 | CSV 確認のみ |
| Phase 2 (g15 実装) | 60-90分 | 5戦略、各 15-20分 |
| Phase 3 (MD 追記) | 15分 | Edit 操作 |
| Phase 4 (検証/Push/報告) | 10分 | URL 検証含む |
| **合計** | **約 1.5-2 時間** | コーディング集中 |

---

## ✅ Self-Review チェックリスト

- [ ] 過去ベスト戦略4件の Strategy 列名が決定済み
- [ ] B&H の Trades/yr / Overfit / CI95 の表記ルールが固定済み（0 / — / —）
- [ ] `MD_HEADER_STRAT` import を使用するスクリプト設計（手書きヘッダ禁止 v1.3 厳守）
- [ ] `CAGR_IS / CAGR_FULL` を MD ヘッダに含めない（v1.1 厳守）
- [ ] 各 Task に具体的なファイル/コマンド/期待出力が記載済み
- [ ] §RISK / §8 履歴 / §7 一次根拠 の連動更新が Task 9 にすべて含まれる
- [ ] Decision Points が明示済み（実装中の判断ブレ防止）

---

*管理: Claude (Opus 4.7)*
*準拠: EVALUATION_STANDARD.md v1.3 / §3.12 9指標標準 / sp-writing-plans skill*
