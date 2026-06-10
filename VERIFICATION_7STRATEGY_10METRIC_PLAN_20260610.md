# 7戦略 × 10指標 批判的検証 実装計画

作成日: 2026-06-10
最終更新日: 2026-06-10

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`（推奨）または `superpowers:executing-plans` をタスク単位で適用すること。ステップは checkbox（`- [ ]`）で進捗管理する。

**Goal:** `CURRENT_BEST_STRATEGY.md` の v4.5 推奨表（7戦略 × 10指標）の各値を、生データからの完全再実行 + 2026-06-10 現実コスト再計算で批判的に検証し、「真実度を高めた」差分レポートと現実コスト版の再計算表を提示する（正典ファイルは変更しない・報告のみ）。

**Architecture:** 既存 src/ の正典指標関数を**改変せず**ラップする「統一検証ハーネス」を新規 `src/audit/` に構築する。全7戦略の NAV を生データ（`data/NASDAQ_extended_to_2026.csv` 他）から再生成し、(A) 現行 Scenario D で再現性を確認（誤り検出）→ (B) 現実コスト（`product_costs_realistic_20260610.py`）で再計算（真実度向上）の2系統を、同一の split / 税モデル / 指標関数に通す。戦略はバッチ投入（Phase 1〜3）。

**Tech Stack:** Python 3.10+, pandas, numpy, scipy.stats（既存 requirements.txt）。新規コードは `src/audit/` と `audit_results/` のみに追加（既存 src/・正典 MD は read-only）。

**ユーザー確定方針（2026-06-10）:**
1. 検証深さ = **生データから完全再実行**
2. コスト基準 = **2026-06-10 現実コストで再計算**
3. 誤り発見時 = **報告のみ（正典は触らない）**

---

## 検証対象（CURRENT_BEST_STRATEGY.md 行26–32）

| # | 戦略 | 環境 | 主実装 | 結果CSV |
|---|---|---|---|---|
| 1 | §1 Active = E4 RegimeKLT | CFD | `src/e4_regime_klt.py` | `e4_regime_klt_results.csv` |
| 2 | CFD候補 = vz=0.65+l5+F10ε | CFD | `src/g27_vz065_lmax_sweep.py` (lmax=5) | `g27_vz065_lmax_sweep_metrics.csv` |
| 3 | 副候補 = vz=0.65+l7+F10ε | CFD | `src/g27_vz065_lmax_sweep.py` (lmax=7) | 同上 |
| 4 | ETF only = DH-W1 | ETF | `src/g23a_dh_refinement_variants.py::build_W1` | `g23b_dh_refinement_metrics.csv` |
| 5 | overlay V0 def = DH-W1 + mom63 {1.1,1.0,0.9,0.8} | ETF | `data/signals/expansion/` + s3_overlay_tuning | `s3_overlay_tuning_20260607.csv` |
| 6 | overlay V7 boost = DH-W1 + mom63 {1.2,1.1,1.0,1.0} | ETF | 同上 | 同上 |
| 7 | 投信 P7 = DH-W1 CashSleeve GOLD75/BOND25 | 投信 | `analysis_cash_sleeve/cash_sleeve_sim.py` | `cash_sleeve_4strategies_metrics.csv` |

### 10指標（列）
CAGR⓽ IS / CAGR⓽ OOS（= 2指標）/ IS-OOS gap⓽ / Sharpeⓒ / MaxDDⓒ / Worst10Y★ⓒ / P10ⓒ 5Y / Tradeⓞ/yr / WFEⓞ / CI95ⓡ_lo
→ 7戦略 × 10指標 = **70セル**

---

## 事前に特定済みの「赤信号」（検証で必ず白黒つける論点）

| ID | 論点 | 一次根拠 |
|---|---|---|
| R1 | 7戦略が CFD/ETF/投信の**別コスト環境**。同一表での行間比較は基準非統一 | EVALUATION_STANDARD §1 / 構成記述 |
| R2 | Worst10Y★/P10_5Y のシンボルが表（ⓒ税前）と §3.12（⓽税後）で**矛盾** | CURRENT_BEST 行24 vs EVAL §3.12 |
| R3 | DH-W1 の CAGR が同一ファイル内で **+15.31%（canonical）と +13.66%（旧split）の2値** | CURRENT_BEST 行29 vs 行216/EVAL §3.13 |
| R4 | DH-W1 の Trades/yr が **68.7 / 17.6 / 17.8 と4倍乖離**（計上方式不統一の疑い） | 行29 vs 行30 vs 行216 |
| R5 | 日次取引コストの反映が**不均一**（E4/F10/D5/DH-W1=反映、S2 raw=年定額0.20%のみ） | g18 調査 / EVAL §3.12 v1.4 |
| R6 | doc自身が「**−66bps/yr 過少計上**、現実CAGR≈+30.5%」と認めるが、表の⓽は補正前+33.53%から導出 | CURRENT_BEST 行274 |
| R7 | **CFD財務コストの過少計上**: sim=2×SOFR(≈7.3%) vs 現実CFD=(SOFR+3.0%)×L（L=3で19.9%/L=7で46%） | product_costs.py vs 2026-06-10 §1-1 |
| R8 | WFA窓設計の差（E4=G3 約49窓 vs P7=50窓）。窓長/step/評価開始(1977 warmup)が全戦略統一か | g1_wfa / g3_wfa_e4 / cash_sleeve |
| R9 | 近似値（`~+12.67%`）と N/A（副候補 CI95）の未確定セル | CURRENT_BEST 行27,28 |

---

## 2026-06-10 現実コスト定数（Phase 0 で `product_costs_realistic_20260610.py` に固定）

`PRODUCT_COST_COMPARISON_2026-06-10.md` 本文より抽出（前提: ¥3,000万 / 年20回 / 1取引 $22上限）:

| 商品/脚 | 現実コスト | sim 現行値 | 差分（要再コスト） |
|---|---|---|---|
| SOFR (2026-06 実測) | 3.63%/yr | DTB3 時変（52yr平均4.37%） | OOS期は実測寄せ。財務 r_$=SOFR+0.40%=4.03% |
| FXヘッジ（円建て国内商品のみ） | 2.9%/yr | 0（モデル外） | 投信脚 P7 のヘッジ有無で要判定 |
| NASDAQ 3x ETF (TQQQ) | TER0.82% + 暗黙(n-1)×r_$=8.06% ≈ **8.9%** | TER0.86% + 2×SOFR + swap0.50% | ほぼ整合（要確認） |
| NASDAQ CFD ×L | **(SOFR+3.0%)×L** = L3:19.9% / L7:46.4% | 2×SOFR + swap0.50% ≈ 7.3% | **R7: 最大の乖離** |
| CFD スプレッド | ~0.028%/片道 | CFD_SPREAD_LOW 0.20%/yr | 比例型へ変更 |
| Gold 2x | UGL 0.95% / WisdomTree2036 0.49% | sim proxy 0.49% | proxy差 −10.5bps（R6一部） |
| Bond 3x (TMF) | TER0.90%(実1.06%) + 暗黙8.06% ≈ 9.0–9.2% | TER0.91% + 2×SOFR + swap | 20Y利回り>SOFR の相殺は保守的に無視 |
| 1x 投信 (Gold/Bond/NASDAQ) | TER 0.18–0.20% + ヘッジ2.9%(ヘッジ有時) | product_costs.py 1x funds (TER のみ) | ヘッジコスト要追加判定 |
| 米国ETF 売買コスト | $22上限 ≈ ¥3,190/取引、¥30M年20回=0.21%/yr | per_unit 0.10% turnover | 規模・頻度依存へ |
| 約定ラグ | US ETF T+2（=DELAY2整合）/ 国内投信 T+1 / CFD T+0 | DELAY=2 一律 | CFD は2日保守・投信1日保守 |
| 税 | 20.315%（米ETFは配当に米10%源泉追加） | 0.8273 一律 | 配当源泉の扱い要確認 |

---

## ファイル構成（新規・すべて additive）

```
src/audit/
  __init__.py
  product_costs_realistic_20260610.py   # 現実コスト定数（dataclass、product_costs.py と同形）
  unified_metrics.py                     # 既存指標関数の統一ラッパ（split/税/10指標を一括）
  unified_wfa.py                         # g1_wfa を統一窓設計でラップ
  strategy_runners.py                    # 7戦略の NAV を生データから再生成する共通インタフェース
  run_audit.py                           # CLI: 戦略名 × コスト基準(D/realistic) を指定し全10指標出力
tests/audit/
  test_unified_metrics.py                # E4 既知値を fixture に回帰テスト
  test_product_costs_realistic.py        # 現実コスト定数の算術検証
audit_results/
  audit_<strategy>_<basis>.csv           # 各戦略×基準の再計算10指標
  VERIFICATION_REPORT_20260610.md        # 最終差分レポート（成果物）
  REALISTIC_7x10_TABLE_20260610.md       # 現実コスト版 7×10 再計算表（成果物）
```

> 既存 `src/*.py`・`CURRENT_BEST_STRATEGY.md`・`EVALUATION_STANDARD.md` 等は **read-only**。ハーネスは import で正典関数を再利用し、改変しない。

---

## Phase 0: 基盤構築 + パイロット（E4）

### Task 0.1: 現実コスト定数モジュール

**Files:**
- Create: `src/audit/product_costs_realistic_20260610.py`
- Test: `tests/audit/test_product_costs_realistic.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/audit/test_product_costs_realistic.py
from src.audit.product_costs_realistic_20260610 import (
    SOFR_2026, R_USD_FINANCING, FX_HEDGE_COST, cfd_overnight_annual,
)

def test_sofr_and_financing():
    assert SOFR_2026 == 0.0363
    assert abs(R_USD_FINANCING - 0.0403) < 1e-9   # SOFR + 0.40% mgr spread

def test_cfd_overnight_scales_with_leverage():
    # (SOFR + 3.0%) × L
    assert abs(cfd_overnight_annual(3.0) - (0.0363 + 0.030) * 3.0) < 1e-9  # 0.1989
    assert abs(cfd_overnight_annual(7.0) - (0.0363 + 0.030) * 7.0) < 1e-9  # 0.4641

def test_fx_hedge_only_for_hedged():
    assert FX_HEDGE_COST == 0.029
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python -m pytest tests/audit/test_product_costs_realistic.py -v`
Expected: FAIL（モジュール未作成 ImportError）

- [ ] **Step 3: 最小実装**

```python
# src/audit/product_costs_realistic_20260610.py
"""2026-06-10 realistic cost constants. Source: PRODUCT_COST_COMPARISON_2026-06-10.md.
ANNUAL decimal rates. Daily = annual / 252."""
SOFR_2026 = 0.0363                 # 米翌日物 2026-06 実測
MGR_SPREAD = 0.0040               # 運用会社スプレッド
R_USD_FINANCING = SOFR_2026 + MGR_SPREAD   # 4.03%
FX_HEDGE_COST = 0.029             # 円建てヘッジ商品のみ
CFD_SPREAD_ONE_WAY = 0.00028      # ~0.028% per side（比例型）
CFD_OVERNIGHT_SPREAD = 0.030      # CFD金利 = SOFR + 3.0%
US_ETF_TRADE_CAP_JPY = 3190.0     # $22 上限
PORTFOLIO_JPY = 30_000_000.0
JP_TAX = 0.20315
US_WHT_DIV = 0.10                 # 米ETF配当の米源泉

# 暗黙金利 = (leverage - 1) × r_$
def implicit_financing_annual(leverage: float) -> float:
    return (leverage - 1.0) * R_USD_FINANCING

# CFD オーバーナイト年率 = (SOFR + 3.0%) × L
def cfd_overnight_annual(leverage: float) -> float:
    return (SOFR_2026 + CFD_OVERNIGHT_SPREAD) * leverage

# US ETF 売買コスト年率（規模・頻度依存）
def us_etf_trade_cost_annual(trades_per_year: float, portfolio_jpy: float = PORTFOLIO_JPY) -> float:
    return (US_ETF_TRADE_CAP_JPY * trades_per_year) / portfolio_jpy
```

- [ ] **Step 4: テスト成功を確認**

Run: `python -m pytest tests/audit/test_product_costs_realistic.py -v`
Expected: PASS

- [ ] **Step 5: commit**

```bash
git add src/audit/product_costs_realistic_20260610.py tests/audit/test_product_costs_realistic.py
git commit -m "feat(audit): add 2026-06-10 realistic cost constants module"
```

### Task 0.2: 統一指標ハーネス（既存正典関数のラッパ）

**Files:**
- Create: `src/audit/unified_metrics.py`
- Test: `tests/audit/test_unified_metrics.py`

正典関数を改変せず import し、NAV(pd.Series, DatetimeIndex) を入力に10指標を1関数で返す。split は EVALUATION_STANDARD §2.1（IS_END=2021-05-07 / OOS_START=2021-05-08）固定。

- [ ] **Step 1: 失敗するテスト（E4 既知値を許容誤差つき fixture 化）**

```python
# tests/audit/test_unified_metrics.py
import pandas as pd, numpy as np
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START

def test_split_constants():
    assert IS_END == pd.Timestamp('2021-05-07')
    assert OOS_START == pd.Timestamp('2021-05-08')

def test_cagr_sharpe_maxdd_on_known_series():
    # 既知の定数複利 NAV: 年率10%, 252日×2年
    idx = pd.bdate_range('2020-01-01', periods=504)
    nav = pd.Series((1.10) ** (np.arange(504)/252), index=idx)
    m = compute_10metrics(nav, trades_per_year=27.0)
    assert abs(m['CAGR_FULL'] - 0.10) < 0.005
    assert m['MaxDD_FULL'] >= -1e-6   # 単調増加なら DD≈0
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python -m pytest tests/audit/test_unified_metrics.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 実装（正典関数を import して合成）**

```python
# src/audit/unified_metrics.py
"""Unified metric harness. Wraps canonical metric functions WITHOUT modifying them."""
import pandas as pd, numpy as np
from src.cfd_leverage_backtest import calc_7metrics          # CAGR/Sharpe/MaxDD (IS/OOS/FULL)
from src.compute_cfd_worst10y import nav_to_annual, rolling_nY_cagr  # Worst10Y★ calendar-year
from src.calculate_p10_5y import compute_p10_5y, compute_worst5y     # P10_5Y / Worst5Y (daily roll)

IS_END = pd.Timestamp('2021-05-07')
OOS_START = pd.Timestamp('2021-05-08')

def compute_10metrics(nav: pd.Series, trades_per_year: float) -> dict:
    nav = nav.dropna().sort_index()
    dates = pd.Series(nav.index, index=nav.index)
    base = calc_7metrics(nav, dates, trades_per_year=trades_per_year)  # 正典 CAGR/Sharpe/MaxDD
    ann = nav_to_annual(nav, dates)
    worst10 = float(rolling_nY_cagr(ann, n=10).min())
    p10_5y = compute_p10_5y(nav.values)
    is_cagr = base.get('CAGR_IS'); oos_cagr = base.get('CAGR_OOS')
    gap = (is_cagr - oos_cagr) * 100.0 if (is_cagr is not None and oos_cagr is not None) else None
    return dict(
        CAGR_IS=is_cagr, CAGR_OOS=oos_cagr, CAGR_FULL=base.get('CAGR_FULL'),
        IS_OOS_gap_pp=gap, Sharpe_OOS=base.get('Sharpe_OOS'),
        MaxDD_FULL=base.get('MaxDD'), Worst10Y_star=worst10, P10_5Y=p10_5y,
        Worst5Y=compute_worst5y(nav.values), Trades_yr=trades_per_year,
    )
```

> 注: `calc_7metrics` の返却キー名は実装（`src/cfd_leverage_backtest.py:267`）に合わせて Step 実装時に確認・調整する。返却が IS/OOS 個別 Sharpe を持たない場合は `_metrics_period` を IS/OOS で個別呼び出すラッパを足す。

- [ ] **Step 4: テスト成功を確認**

Run: `python -m pytest tests/audit/test_unified_metrics.py -v`
Expected: PASS

- [ ] **Step 5: commit**

```bash
git add src/audit/unified_metrics.py tests/audit/test_unified_metrics.py
git commit -m "feat(audit): unified 10-metric harness wrapping canonical functions"
```

### Task 0.3: 統一WFAラッパ（CI95_lo / WFE）

**Files:**
- Create: `src/audit/unified_wfa.py`
- Test: `tests/audit/test_unified_wfa.py`

`src/g1_wfa.py::compute_summary_stats` を import し、全戦略で同一の窓設計（窓長252・非重複・評価開始=各戦略のwarmup後・IS境界=2021-05-08・短窓<201日除外）を強制する per-window builder を提供。

- [ ] **Step 1: テスト（窓設計の定数と、既知 per-window CSV からの集計一致）**

```python
# tests/audit/test_unified_wfa.py
import pandas as pd
from src.audit.unified_wfa import summarize_wfa, WINDOW_LEN, OOS_START_REF

def test_window_constants():
    assert WINDOW_LEN == 252
    assert OOS_START_REF == pd.Timestamp('2021-05-08')

def test_summary_matches_existing_g3():
    # 既存 g3_wfa_e4_per_window.csv を入力に CI95_lo/WFE が既知値に一致
    per = pd.read_csv('g3_wfa_e4_per_window.csv', parse_dates=['start_date'])
    s = summarize_wfa(per)
    assert abs(s['WFA_WFE'] - 1.131) < 0.01
    assert abs(s['WFA_CI95_lo'] - 26.51) < 0.5   # %表記。単位は実CSVに合わせ調整
```

- [ ] **Step 2: テスト失敗を確認** — Run: `python -m pytest tests/audit/test_unified_wfa.py -v` → FAIL
- [ ] **Step 3: 実装** — `compute_summary_stats` を呼ぶ薄いラッパ + per-window 生成関数。単位（小数/％）は `g3_wfa_e4_per_window.csv` を読んで整合させる。
- [ ] **Step 4: テスト成功を確認** — Run 同上 → PASS
- [ ] **Step 5: commit**

```bash
git add src/audit/unified_wfa.py tests/audit/test_unified_wfa.py
git commit -m "feat(audit): unified WFA wrapper with fixed window design"
```

### Task 0.4: パイロット — E4（§1 Active）を両基準で再計算・突合【チェックポイント】

**Files:**
- Create: `src/audit/strategy_runners.py`（E4 ランナーのみ先行）
- Create: `src/audit/run_audit.py`
- Output: `audit_results/audit_e4_scenarioD.csv`, `audit_results/audit_e4_realistic.csv`

- [ ] **Step 1: E4 ランナー実装** — `src/e4_regime_klt.py` のシグナル/レバ/配分ロジックを呼び出し、コスト適用層だけ差し替え可能にして NAV を生データから再生成（Scenario D / realistic の2系統）。CFD財務は Scenario D=2×SOFR+swap、realistic=`cfd_overnight_annual(L_s2_t)` を日次適用（**R7の核心**）。
- [ ] **Step 2: 実行**

Run: `python -m src.audit.run_audit --strategy e4 --basis scenarioD --out audit_results/audit_e4_scenarioD.csv`
Run: `python -m src.audit.run_audit --strategy e4 --basis realistic --out audit_results/audit_e4_realistic.csv`

- [ ] **Step 3: 表値との突合**（Scenario D 再現性）
Scenario D 再計算が表の §1 値（CAGR_OOSⓒ +33.53% / Sharpe 0.891 / MaxDD −60.01% / Worst10Y★ +18.67% / Trades 27 / WFE 1.131 / CI95 +26.51%）と **±0.5pp / ±0.01** 以内で一致するか確認。差があれば原因（split・税・日次コスト・データ）を1行で特定。
- [ ] **Step 4: realistic との差分記録** — CFD overnight 適用で CAGR_OOS がどれだけ低下するか（R7 の定量化）を記録。
- [ ] **Step 5: commit + チェックポイント**

```bash
git add src/audit/strategy_runners.py src/audit/run_audit.py audit_results/audit_e4_*.csv
git commit -m "feat(audit): E4 pilot recompute under Scenario D and realistic cost"
```

> **チェックポイント**: パイロットで「ハーネスが表値を再現できる」ことを確認してから Phase 1 以降へ。再現できない指標があれば、harness のバグか表の誤りかを切り分けてユーザーに報告。

---

## Phase 1: CFD群（vz=0.65+l5 / l7）

R7（CFD財務過少計上）の影響が最大の群。両戦略は L_s2 が 1x〜5x/7x で変動するため、realistic では日次に `cfd_overnight_annual(L_s2_t)` を適用。

### Task 1.1: vz065_l5 / l7 ランナーと再計算

**Files:**
- Modify: `src/audit/strategy_runners.py`（vz065 ランナー追加）
- Output: `audit_results/audit_vz065_l5_scenarioD.csv` 他3本（l5/l7 × D/realistic）

- [ ] **Step 1:** `src/g27_vz065_lmax_sweep.py` のロジックを呼ぶランナー追加（F10 ε=0.015 deadband は `g19a_f10_eps_extended.py` を再利用）。
- [ ] **Step 2:** 4本実行（l5/l7 × scenarioD/realistic）。Run例: `python -m src.audit.run_audit --strategy vz065_l5 --basis realistic --out audit_results/audit_vz065_l5_realistic.csv`
- [ ] **Step 3:** Scenario D 再計算を表値（l5: CAGR_OOS +18.93% / Sharpe 0.841 / MaxDD −56.72% / Trades 86 / WFE 1.389、l7: 行28）と突合。**副候補 l7 の CI95「N/A」を実際に計算して埋める（R9）**。`~+12.67%` 等の近似 Worst10Y を厳密値に置換。
- [ ] **Step 4:** realistic で CAGR がどれだけ崩れるか定量化（CFD L=7 は財務46%/yr）。
- [ ] **Step 5: commit**

```bash
git add src/audit/strategy_runners.py audit_results/audit_vz065_*.csv
git commit -m "feat(audit): vz065 l5/l7 recompute, fill N/A CI95, exact Worst10Y"
```

### Task 1.2: CFD群 WFA（統一窓）

- [ ] **Step 1:** `src/audit/unified_wfa.py` で l5/l7 の per-window を再生成し CI95_lo/WFE を算出。
- [ ] **Step 2:** 既存 `g27`/`g9_wfa_vz065_lmax5_summary.csv` 値と突合、窓数・step が E4(G3) と統一されているか（R8）を確認。
- [ ] **Step 3: commit** — `git commit -m "feat(audit): unified WFA for CFD group, R8 window-design check"`

---

## Phase 2: ETF群（DH-W1 / overlay V0 / V7）

R3（split不連続）・R4（Trades/yr 乖離）・R2（シンボル基準）の主戦場。

### Task 2.1: DH-W1 ベース再計算

**Files:**
- Modify: `src/audit/strategy_runners.py`（DH-W1 ランナー）
- Output: `audit_results/audit_dhw1_scenarioD.csv`, `audit_results/audit_dhw1_realistic.csv`

- [ ] **Step 1:** `src/g23a_dh_refinement_variants.py::build_W1` + `g18_daily_trade_cost_wfa.py::build_dh_nav_with_cost` を呼ぶランナー。realistic では US ETF 売買コスト（$22上限・規模依存）と TMF 実TER 1.06%・Gold UGL 0.95% を適用。
- [ ] **Step 2:** 実行（D/realistic）。
- [ ] **Step 3（R3）:** canonical split で CAGR_IS/OOS を出し、表の「+15.31%/+15.74%（canonical）」と「+13.66%（旧split）」の**どちらが正しい canonical 値か**を確定。両split値を併記し不連続を定量化。
- [ ] **Step 4（R4）:** Trades/yr を「リバランスイベント数」と「turnover換算」の両定義で算出し、表の 68.7 / 17.8 がどちらの定義かを特定。**同一戦略で68.7と17.8が混在する原因を1行で結論**。
- [ ] **Step 5: commit** — `git commit -m "feat(audit): DH-W1 recompute, resolve R3 split & R4 trades/yr"`

### Task 2.2: overlay V0 / V7 再計算

**Files:**
- Modify: `src/audit/strategy_runners.py`（mom63 overlay 適用層）
- Output: `audit_results/audit_v0_*.csv`, `audit_results/audit_v7_*.csv`

- [ ] **Step 1:** `data/signals/expansion/` の nasdaq_mom63（macro_features.csv）+ M6 defensive/boost mapping を DH-W1 NAV に適用するランナー。quantile_cut(4) → publication_lag(+1BD) → `lev × mask_W1 × mult`。
- [ ] **Step 2:** 実行（V0/V7 × D/realistic = 4本）。
- [ ] **Step 3:** 表値（V0: IS+14.07/OOS+15.02, Sharpe0.892, MaxDD−28.74, WFE1.005, CI95+13.00 / V7: IS+15.74/OOS+15.96, MaxDD−34.57, WFE1.029）と突合。
- [ ] **Step 4（R2）:** Worst10Y★/P10_5Y を税前ⓒ・税後⓽の両方で出し、表記（表はⓒ）が §3.12（⓽）と矛盾する件を定量化。
- [ ] **Step 5: commit** — `git commit -m "feat(audit): overlay V0/V7 recompute, R2 pretax/aftertax symbol check"`

---

## Phase 3: 投信群（P7 GOLD75/BOND25）

### Task 3.1: P7 キャッシュスリーブ再計算

**Files:**
- Modify: `src/audit/strategy_runners.py`（P7 ランナー）
- Output: `audit_results/audit_p7_scenarioD.csv`, `audit_results/audit_p7_realistic.csv`

- [ ] **Step 1:** `analysis_cash_sleeve/cash_sleeve_sim.py` のロジック（OUT期=cash 46.9% を Gold1x75%/Bond1x25% で置換、レバ脚DELAY=2 / 投信スリーブLAG=5BD）を呼ぶランナー。realistic では国内投信のFXヘッジ2.9%（ヘッジ有商品の場合）を追加適用するか判定。
- [ ] **Step 2:** 実行（D/realistic）。
- [ ] **Step 3:** 表値（CAGR_OOS+14.90, Sharpe0.827, MaxDD−48.23, Worst10Y+9.92, WFE1.043, CI95+16.74）と突合。
- [ ] **Step 4（R8）:** P7 の WFA「50窓」と E4「約49窓」の窓数差の原因（評価開始日/warmup）を確認。
- [ ] **Step 5: commit** — `git commit -m "feat(audit): P7 cash-sleeve recompute, FX-hedge & window check"`

---

## Phase 4: 横断監査 + 成果物

### Task 4.1: 70セル差分マトリクスの生成

**Files:**
- Create: `src/audit/build_diff_matrix.py`
- Output: `audit_results/diff_matrix_20260610.csv`

- [ ] **Step 1:** 全 `audit_results/audit_*_scenarioD.csv` を集約し、表の公称値 vs 再計算値（Scenario D）の差分マトリクス（70セル）を生成。各セルに `OK / 軽微差(±許容) / 不一致 / 近似→厳密化 / N/A→確定` のステータス。
- [ ] **Step 2:** realistic 版マトリクスも生成（公称値 vs 現実コスト再計算）。
- [ ] **Step 3: commit** — `git commit -m "feat(audit): 70-cell diff matrix (Scenario D & realistic)"`

### Task 4.2: 質問①②③への結論 + 最終レポート（成果物）

**Filesः**
- Create: `audit_results/VERIFICATION_REPORT_20260610.md`（成果物）
- Create: `audit_results/REALISTIC_7x10_TABLE_20260610.md`（成果物）

- [ ] **Step 1:** VERIFICATION_REPORT に以下を記載:
  - **質問①（同一基準か）**: R1/R8 の結論（CFD/ETF/投信で別環境・窓数差の有無）
  - **質問②（コスト反映は妥当か）**: R5/R6/R7 の結論（日次コスト反映の不均一、−66bps過少、**CFD財務過少計上の定量インパクト**）
  - **質問③（各数字は正しいか）**: 70セル差分マトリクスの要約（一致/不一致/近似/N/A）、R2/R3/R4 の自己矛盾3件
  - 各赤信号 R1–R9 に「確定/否定/要追加検証」の判定
- [ ] **Step 2:** REALISTIC_7x10_TABLE に現実コスト版の 7戦略×10指標を再掲（正典は変更せず、新ファイルとして提示）。
- [ ] **Step 3:** EVALUATION_STANDARD §5 のレポート様式（必須ヘッダ・期間・コスト Scenario・コード参照）に準拠。
- [ ] **Step 4: commit** — `git commit -m "docs(audit): verification report + realistic 7x10 table (report-only)"`
- [ ] **Step 5:** ユーザーへ成果物報告（CLAUDE.md §9: 3列表・URL検証・push後）。push 可否はユーザー確認。

---

## モデル使い分け（工程別）

| 工程 | モデル | 担当タスク |
|---|---|---|
| 全体指揮・批判的判断・赤信号の解釈・最終合成・チェックポイント審査 | **Fable 5（メイン）** | Phase 0.4/2.1 の split・Trades定義の裁定、Phase 4 レポート結論 |
| ハーネス実装・各戦略ランナー・バックテスト再実行・TDD | **Sonnet サブエージェント** | Task 0.1–0.3, 各 Phase の Step1–2（実装・実行） |
| CSV値の一括抽出・公称値突合・差分マトリクス機械生成 | **Haiku サブエージェント** | Task 4.1 の集約、表値の grep 突合 |

> サブエージェント委託時は CLAUDE.md §9-A に従い「deliverables-policy 厳守・3列表・URL検証」を prompt に明記し、報告 URL はメイン（Fable）が再検証する。

---

## Self-Review（計画の自己点検）

- **スコープ網羅**: 質問①=R1/R8、②=R5/R6/R7、③=70セル+R2/R3/R4 を各 Phase に割当済み。
- **生データ完全再実行**: 各ランナーは結果CSV転記でなく `data/` から NAV 再生成（Step1 で明記）。
- **現実コスト再計算**: Task 0.1 で定数固定、各ランナーが basis=realistic で適用。
- **報告のみ**: 全出力は `src/audit/`・`audit_results/` の新規ファイル。正典 MD/既存 src は read-only。
- **未確定セルの扱い**: 副候補 CI95「N/A」(R9) は Task 1.1 Step3 で計算、近似 Worst10Y は厳密化。

---

---

## 更新反映（2026-06-10 v3 — 実行開始時に追補）

実行直前に `CURRENT_BEST_STRATEGY.md` が更新されたため、検証対象と比較基準を以下に同期する（計画の手法・Phase構成は不変）。

### 変更点（コミット `7ca89fc` / `5515be4`）
1. **「現行ベスト戦略（§1 単一BEST）」章を削除** → 単一BESTは存在せず、**全て環境別Active候補**。検証も「BEST判定」ではなく「各Active候補の各セルの正否」に限定。
2. **E4 RegimeKLT の数値修正**: `CFD_SPREAD_LOW=0.20%/yr`（誤値）→ `SBI_CFD_SPREAD=3.0%/yr`（正値）への修正で CAGR⓽ が約7pp低下。これは本計画 **R7（CFD財務過少計上）の一部をリポジトリ側が先行修正**したもの。

### E4 比較基準値の差し替え（修正後・出典 `7STRATEGY_PERFORMANCE_REPORT_20260529.md` / `g14_wfa_sbi_cfd_summary.csv`）

| 指標 | 旧基準（計画初版） | **新基準（v3修正後）** |
|---|---|---|
| CAGR⓽ IS / OOS（min） | +27.41% / +28.01%（min+27.41%） | **+20.0% / +22.4%（min+20.0%）** |
| IS-OOS gap | −0.60pp | **−2.41pp** |
| Sharpe | 0.891 | **0.79** |
| MaxDD | −60.01% | **−62.0%** |
| Worst10Y★ | +18.67% | **+9.8%** |
| P10_5Y | +9.78% | **+2.2%** |
| Trades/yr | 27 | **28** |
| WFE / CI95_lo | 1.131 / +26.51% | **1.15 / +16.3%** |

### R7 の精緻化（残存する過少計上の疑い）
リポジトリの修正は `SBI_CFD_SPREAD=3.0%/yr` を**レバレッジ非依存の定額**で適用している。一方、`PRODUCT_COST_COMPARISON_2026-06-10.md` の現実式は **CFDオーバーナイト = (SOFR+3.0%)×L**（レバ L に比例、E4 の L_s2 は 1〜7x で変動）。
→ **本計画 realistic 基準は引き続き `(SOFR+3.0%)×L`（Task 0.1 の `cfd_overnight_annual(L)`）を採用**し、定額3.0%修正でも残る過少計上を定量化する。リポジトリの定額3.0%は中間値として併記。

### スコープ確認
- v4.5表は現在 **6戦略行**（E4 / vz065_l5 / vz065_l7 / DH-W1 / V0 / V7）。**投信 P7** は別セクション扱いのため、Phase 3 着手時に表内の有無を確認し、対象から除外されていれば「投信環境候補」として別途検証する（計画の Phase 構成は不変）。
- 比較基準値は流動的なため、各 Phase の突合ステップで `CURRENT_BEST_STRATEGY.md` の**当日の値をライブ取得**して比較する。

---

*管理者: 男座員也（Kazuya Oza）*
