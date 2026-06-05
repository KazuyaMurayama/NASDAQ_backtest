# DH-W1 キャッシュ・スリーブ 1倍投信置換シミュレーション 実装計画

作成日: 2026-06-05
最終更新日: 2026-06-05

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans でタスク単位に実行。各ステップは `- [ ]` チェックボックスで追跡。
> **生成者**: Claude (Opus 4.8) / **著者**: 男座員也 (Kazuya Oza)

**Goal:** DH-W1 (Asymm+Hyst) のキャッシュ保有期間（全営業日の約47%）を、1倍の NASDAQ / ゴールド / ボンド投信で置き換えた場合のパフォーマンス押し上げ効果を、手数料・税・5営業日遅延込みで6パターン定量評価する。

**Architecture:** 既存 `build_W1()` が返す HOLD/OUT マスクを取得 → OUT 日のリターン（現状0%）を、コードに既存の1倍原資産（NASDAQ=`close` / Gold=`prepare_gold_local` / Bond=`build_bond_1x_nav_corrected` 22yr duration）の日次リターンで置換 → 5営業日ラグ・信託報酬・約20%税を適用 → 6配分パターン × フル期間/OOS で9指標を再計算。**外部データ取得は不要**（全原資産がコード内に存在）。

**Tech Stack:** Python 3.13, numpy, pandas。既存モジュール `g23a_dh_refinement_variants`, `g14_wfa_sbi_cfd`, `g18_daily_trade_cost_wfa`, `corrected_strategy_backtest`, `compute_cfd_worst10y` を import。

---

## 確定した調査結果（着手前提）

### DH-W1 のキャッシュ機構（`src/g23a_dh_refinement_variants.py` 実読で確定）
- `hold_mask_W1`: 状態機械。`lev_mod_065 ≥ 0.7` で HOLD 入り、`≤ 0.3` で OUT 入り、中間はヒステリシス維持。
- **HOLD 比率 = 53.1% → OUT（キャッシュ）≈ 46.9%**。OUT 時は `wn=wg=wb=0, lev_raw=0` → 日次リターン **0%（SOFR も無し）**。
- フル OUT 年（年次表で 0.0%）: 2001, 2002, 2008, 2022。これらが「キャッシュ保有期間」の代表。

### 1倍原資産（すべて `load_shared_assets()` 内に存在・USD建て）
| 資産 | 取得関数 | 備考 |
|---|---|---|
| NASDAQ 1x | `a['close']`（NASDAQ_extended_to_2026.csv, 1974–2026） | 価格指数 |
| Gold 1x | `prepare_gold_local(dates)`（`compute_cfd_worst10y`） | LBMA現物ベース |
| Bond 1x | `build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)` | **22yr duration 長期米国債** TR |

### 実在する SBI証券 1倍商品（手数料グラウンディング用、いずれも信託報酬 <1%・SOFR無し）
| 資産 | 商品 | 信託報酬(税込) | 備考 |
|---|---|---|---|
| NASDAQ 1x | SBI NASDAQ100インデックス・ファンド | **0.1958%** | 2026/5設定・最安。代替: ニッセイNASDAQ100 0.2035% |
| Gold 1x | SBI・iシェアーズ・ゴールドファンド（為替ヘッジなし）「サクっと純金」 | **0.1838%** | LBMA金連動 |
| Bond 1x | iシェアーズ 米国債20年超 ETF（為替ヘッジなし, 2255） | **0.154%** | 22yr duration の最適合致。代替(ヘッジあり): 2621 同0.154% |

> **手数料の基本ケース**: 各資産の実信託報酬（上表）を採用。感度確認として一律 0.50% / 1.00% も併走。

---

## 重要な設計判断（明示・要ユーザー確認可）

1. **通貨基準 = USD**: 既存 DH-W1 NAV が USD 建て（TQQQ/TMF/2036 は米国ETF）。整合のためスリーブも USD で評価する。実在 SBI 商品は「為替ヘッジなし」円建てだが、ここでは①信託報酬の現実値グラウンディング、②USD建てでの戦略全体との apple-to-apple 比較、の2目的で USD 基準を採用（円投資家視点では別途 USD/JPY 変動が上乗せされる旨をレポートに注記）。

2. **5営業日ラグの数理モデル（資本保存・両端ラグ）**:
   `fund_active(t) = out(t−5)`（OUT マスクを5営業日前方シフト）として、
   - OUT 開始 [s, s+5): 投信買付が決済中 → **キャッシュ 0%**（買いラグ）
   - [s+5, e): **1倍投信リターン**
   - HOLD 再開後 [e, e+5): 投信売却が決済中 → **まだ投信を保有**（売りラグ、HOLD開始リターンを取り逃す現実的フリクション）
   - [e+5, …): HOLD（DH-W1 レバ）リターンに復帰
   常に資本は1脚に100%（保存則OK）。

3. **税モデル**: リポ §3-A 標準 `apply_tax_etf_decimal`（年次プラスリターンに `×0.8273`、実効≈17–20%税）を採用し DH-W1 ベースラインと比較可能にする。ユーザー指定「約20%」に整合。プレ税 NAV も併記。

4. **6配分パターン（OUT 日に保有する1倍スリーブの中身、日次リバランス）**:
   | # | 配分 |
   |---|---|
   | P1 | NASDAQ 1x 100% |
   | P2 | Gold 1x 100% |
   | P3 | Bond 1x 100% |
   | P4 | NASDAQ 50% / Gold 50% |
   | P5 | Gold 50% / Bond 50% |
   | P6 | NASDAQ 1/3 / Gold 1/3 / Bond 1/3 |

5. **比較基準**: ベースライン = DH-W1（OUT=キャッシュ0%）。各パターンの「押し上げ」= フル期間/OOS の CAGR・最終倍率・MaxDD・Sharpe・Worst10Y 差分。

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| Create: `analysis_cash_sleeve/cash_sleeve_sim.py` | 本体。マスク取得→1倍置換→ラグ/手数料/税→6パターン×指標。 |
| Output: `analysis_cash_sleeve/cash_sleeve_part1_asset_during_cash.csv` | Q1回答：OUT日における各1倍資産の年率リターン/リスク。 |
| Output: `analysis_cash_sleeve/cash_sleeve_6patterns_metrics.csv` | 6パターン×(プレ税/税後)×指標、ベースライン差分付き。 |
| Output: `analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260605.md` | 日本語レポート（Q1表＋6パターン表＋SBI商品＋注意点＋結論）。 |

---

### Task 1: シミュレーション本体の作成と実行

**Files:** Create `analysis_cash_sleeve/cash_sleeve_sim.py`

- [ ] **Step 1: スクリプトを作成**（下記を verbatim）

```python
"""DH-W1 キャッシュ期間を 1倍投信(NASDAQ/Gold/Bond)で置換するシミュレーション。
Part1: OUT日における各1倍資産の年率リターン/リスク。
Part2: 6配分パターン × 5営業日ラグ + 信託報酬 + 約20%税 で押し上げ効果を評価。
原資産は全てコード内に存在 (外部取得なし)。USD建て。
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import load_shared_assets
from g23a_dh_refinement_variants import build_W1
from corrected_strategy_backtest import build_bond_1x_nav_corrected
from compute_cfd_worst10y import prepare_gold_local
from g18_daily_trade_cost_wfa import metrics_from_nav, apply_tax_etf_decimal

LAG_DAYS = 5
TRADING_DAYS = 252
OUT_DIR = os.path.join(ROOT, 'analysis_cash_sleeve')

# 信託報酬 基本ケース(実SBI商品, 年率)
FEE_NDX  = 0.001958   # SBI NASDAQ100
FEE_GOLD = 0.001838   # SBI iシェアーズ・ゴールド(ヘッジなし)
FEE_BOND = 0.00154    # iシェアーズ 米国債20年超(2255)

# 6配分パターン: (NDX, Gold, Bond) ウェイト
PATTERNS = {
    'P1_NDX100':        (1.0, 0.0, 0.0),
    'P2_GOLD100':       (0.0, 1.0, 0.0),
    'P3_BOND100':       (0.0, 0.0, 1.0),
    'P4_NDX50_GOLD50':  (0.5, 0.5, 0.0),
    'P5_GOLD50_BOND50': (0.0, 0.5, 0.5),
    'P6_THIRDS':        (1/3, 1/3, 1/3),
}


def ann_ret_vol_on_mask(ret, mask_bool):
    """mask_bool=True の日だけで年率リターン(複利)と年率vol(リスク)を算出。"""
    r = np.asarray(ret, dtype=float)[mask_bool]
    if len(r) < 5:
        return np.nan, np.nan, np.nan, 0
    n_days = len(r)
    cum = float(np.prod(1.0 + r))
    yrs = n_days / TRADING_DAYS
    cagr = cum ** (1.0 / yrs) - 1.0 if cum > 0 else -1.0
    vol = float(np.std(r, ddof=1)) * np.sqrt(TRADING_DAYS)
    return cagr, vol, cum, n_days


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 78)
    print('DH-W1 キャッシュ・スリーブ 1倍投信置換シミュレーション')
    print('=' * 78)

    a = load_shared_assets()
    dates = a['dates'].reset_index(drop=True)
    n = len(dates)

    # --- DH-W1 (baseline: OUT=cash 0%) ---
    nav_w1, cost_w1, mask, wn, lev_raw = build_W1(a)
    mask = np.asarray(mask, dtype=float)          # 1=HOLD, 0=OUT
    r_w1 = pd.Series(nav_w1).pct_change().fillna(0).values
    out = (mask < 0.5)                            # True on OUT(cash) days
    hold_ratio = float(mask.mean())
    print(f'  HOLD={hold_ratio*100:.1f}%  OUT(cash)={100-hold_ratio*100:.1f}%  n={n}')

    # --- 1x underlyings (USD), aligned to dates ---
    close = np.asarray(a['close'], dtype=float)
    ret_ndx = np.concatenate([[0.0], np.diff(close) / close[:-1]])

    gold_1x = np.asarray(prepare_gold_local(a['dates']), dtype=float)
    ret_gold = np.concatenate([[0.0], np.diff(gold_1x) / gold_1x[:-1]])

    bond_1x = np.asarray(build_bond_1x_nav_corrected(
        a['dates'], use_time_varying_duration=True, bond_maturity=22.0), dtype=float)
    ret_bond = np.concatenate([[0.0], np.diff(bond_1x) / bond_1x[:-1]])

    ret_ndx  = np.nan_to_num(ret_ndx,  nan=0.0)
    ret_gold = np.nan_to_num(ret_gold, nan=0.0)
    ret_bond = np.nan_to_num(ret_bond, nan=0.0)

    # ================= PART 1: OUT日の各資産 年率リターン/リスク =================
    print('\n[Part1] OUT(cash)日のみで各1倍資産の年率リターン/リスク')
    p1_rows = []
    for name, ret in [('NASDAQ_1x', ret_ndx), ('Gold_1x', ret_gold), ('Bond_1x', ret_bond)]:
        cagr, vol, cum, nd = ann_ret_vol_on_mask(ret, out)
        sharpe = cagr / vol if vol and vol > 0 else np.nan
        p1_rows.append(dict(asset=name, out_days=nd,
                            ann_return_pct=round(cagr*100, 2),
                            ann_vol_pct=round(vol*100, 2),
                            ret_per_risk=round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
                            cum_mult=round(cum, 3)))
        print(f'  {name:10s} 年率Ret={cagr*100:+6.2f}%  年率Vol={vol*100:5.2f}%  '
              f'累積×{cum:.2f}  (OUT {nd}日)')
    pd.DataFrame(p1_rows).to_csv(
        os.path.join(OUT_DIR, 'cash_sleeve_part1_asset_during_cash.csv'),
        index=False, float_format='%.4f')

    # ================= PART 2: 6パターン × ラグ + 手数料 + 税 =================
    # fund_active(t) = out(t-LAG): 5営業日前方シフト (買い/売り両端ラグ)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out[:-LAG_DAYS]

    def blended_ret_fee(w_ndx, w_gold, w_bond):
        r_blend = w_ndx*ret_ndx + w_gold*ret_gold + w_bond*ret_bond
        ann_fee = w_ndx*FEE_NDX + w_gold*FEE_GOLD + w_bond*FEE_BOND
        return r_blend, ann_fee / TRADING_DAYS

    print('\n[Part2] 6配分パターン (5d lag + 信託報酬 + 税後 ×0.8273)')
    p2_rows = []

    # baseline metrics (DH-W1, after-tax)
    nav_base_pre = pd.Series(np.cumprod(1.0 + r_w1), index=dates.index)
    m_base = metrics_from_nav(nav_base_pre, dates, a['ret'])
    yr_base_aft = m_base['yearly'].apply(apply_tax_etf_decimal)
    base_mult_aft = float(np.prod(1.0 + yr_base_aft.values))

    def yearly_after_tax_nav(r_daily):
        nav_pre = pd.Series(np.cumprod(1.0 + r_daily), index=dates.index)
        m = metrics_from_nav(nav_pre, dates, a['ret'])
        yr_aft = m['yearly'].apply(apply_tax_etf_decimal)
        return nav_pre, m, yr_aft

    nav_b_pre, m_b, _ = yearly_after_tax_nav(r_w1)
    p2_rows.append(dict(pattern='BASELINE_DH-W1_cash', **_fmt_metrics(m_b, base_mult_aft)))
    print(f"  {'BASELINE (cash 0%)':22s} CAGR_F={m_b['CAGR_FULL']*100:+6.2f}% "
          f"MaxDD={m_b['MaxDD_FULL']*100:+6.1f}% 税後倍率×{base_mult_aft:.1f}")

    for name, (wn_, wg_, wb_) in PATTERNS.items():
        r_blend, fee_daily = blended_ret_fee(wn_, wg_, wb_)
        r_enh = np.where(fund_active, r_blend - fee_daily, r_w1)
        nav_pre, m, yr_aft = yearly_after_tax_nav(r_enh)
        mult_aft = float(np.prod(1.0 + yr_aft.values))
        p2_rows.append(dict(pattern=name, **_fmt_metrics(m, mult_aft)))
        print(f"  {name:22s} CAGR_F={m['CAGR_FULL']*100:+6.2f}% "
              f"CAGR_OOS={m['CAGR_OOS']*100:+6.2f}% MaxDD={m['MaxDD_FULL']*100:+6.1f}% "
              f"税後倍率×{mult_aft:.1f}  vs base {(mult_aft/base_mult_aft-1)*100:+.1f}%")

    df2 = pd.DataFrame(p2_rows)
    df2.to_csv(os.path.join(OUT_DIR, 'cash_sleeve_6patterns_metrics.csv'),
               index=False, float_format='%.4f')
    print('\n[DONE] CSV written to analysis_cash_sleeve/')


def _fmt_metrics(m, mult_aft):
    return dict(
        CAGR_FULL_pct=round(m['CAGR_FULL']*100, 2),
        CAGR_IS_pct=round(m['CAGR_IS']*100, 2),
        CAGR_OOS_pct=round(m['CAGR_OOS']*100, 2),
        Sharpe_OOS=round(m['Sharpe_OOS'], 3),
        MaxDD_pct=round(m['MaxDD_FULL']*100, 2),
        Worst10Y_pct=round(m['Worst10Y_star']*100, 2),
        aftertax_mult=round(mult_aft, 3),
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 実行**（ネットワーク不要だが env により sandbox 無効化で安全）

Run: `python analysis_cash_sleeve/cash_sleeve_sim.py`
期待: HOLD≈53.1% / OUT≈46.9% を表示、Part1 で3資産の年率Ret/Vol、Part2 で BASELINE + 6パターンの CAGR/MaxDD/税後倍率、CSV2本出力、例外なし。

- [ ] **Step 3: サニティ確認**
  - `HOLD=53.1%` が §0' Sanity と一致（不一致なら mask 計算ズレ）。
  - BASELINE 税後倍率がフル期間で DH-REF 系と桁整合（DH-W1 は OOS ×2.16）。
  - Part1: 2008/2022 を含む OUT 日で Bond_1x の年率Ret がプラス（risk-off で債券上昇）、NASDAQ_1x はマイナス寄り（下落局面で OUT になりやすい）になる方向性を確認。
  - 5営業日ラグ有効化で、ラグ無し版（`fund_active=out`）よりわずかに押し上げが減衰することをスポット確認。

---

### Task 2: レポート作成

**Files:** Create `analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260605.md`

- [ ] **Step 1: レポート執筆**（H1直下に `作成日/最終更新日`）。構成:
  1. §1 Q1回答表（OUT日における NASDAQ/Gold/Bond の年率リターン・年率リスク・累積倍率）+ 解釈。
  2. §2 6パターン比較表（CAGR_FULL/IS/OOS, Sharpe_OOS, MaxDD, Worst10Y, 税後倍率, ベースライン比）。
  3. §3 実在 SBI 1倍商品表（信託報酬）+ 「投信は <1% 手数料・SOFR無し・約5営業日遅延」前提の妥当性。
  4. §4 前提・注意（USD建て / 5営業日ラグ両端モデル / 税×0.8273≈20% / 価格指数 vs TR）。
  5. §5 結論：どのパターンが押し上げ最大か、リスク（MaxDD悪化）とのトレードオフ、推奨。

---

## Self-Review

- **Spec coverage**: Q1(OUT日の資産Ret/Risk)=Part1 ✓ / 6パターン=PATTERNS ✓ / SBI実商品=調査済み&§3 ✓ / 手数料<1%=実信託報酬 ✓ / 税約20%=×0.8273 ✓ / 5営業日遅延=fund_active shift ✓。
- **No placeholders**: 全関数実装済み、ウェイト・手数料は実数値 ✓。
- **Type consistency**: `metrics_from_nav` は dict返し（`CAGR_FULL/IS/OOS, Sharpe_OOS, MaxDD_FULL, Worst10Y_star, yearly`）→ `_fmt_metrics`/`apply_tax_etf_decimal` で整合 ✓。`build_W1` は5タプル `(nav,cost,mask,wn,lev_raw)` ✓。

## 未確定・要ユーザー判断（実行前に確認可）
1. 通貨基準 USD で良いか（円建て・為替ヘッジなしの上乗せFXを別途モデルするか）。
2. 税モデルはリポ標準 ×0.8273 で良いか（厳密 20.315% を明示適用するか）。
3. 手数料は実SBI信託報酬を基本ケースとし 0.5%/1.0% を感度併走で良いか。
