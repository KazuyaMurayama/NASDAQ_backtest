"""
gen_strategy_comparison.py — 戦略パフォーマンス比較表（6戦略 統一フォーマット）
EVALUATION_STANDARD §3.12 準拠 (v1.6, 2026-05-24)

出力: STRATEGY_PERFORMANCE_COMPARISON_2026-05-24.md
  §1 比較前提
  §2 9指標比較表 (6戦略 × 9指標)  ← MD_HEADER_STRAT 使用
  §3 年次リターン表 (1974–2026)  ← [OOS] マーク付き（E4 列が最左）
  §4 OOS期間 詳細分析
  §5 採用判断サマリー
  §6 一次根拠ファイル
  §7 改訂履歴

# E4 列は gen_e4_yearly_returns.py で単独生成可能。
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sweep_format import MD_HEADER_STRAT, fmt_row_strat, MD_METRIC_GLOSSARY

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv(fname):
    return list(csv.DictReader(open(os.path.join(BASE, fname), encoding='utf-8')))


# ---------------------------------------------------------------------------
# 9指標データ収集
# ---------------------------------------------------------------------------

b9_rows  = read_csv('b9_s2lt2_goldfrac_results.csv')
b1_rows  = read_csv('b1_s2_lt2_results.csv')
e4_rows  = read_csv('e4_regime_klt_results.csv')
g1_map   = {r['strategy']: r for r in read_csv('g1_wfa_summary.csv')}
g2_map   = {r['strategy']: r for r in read_csv('g2_wfa_b9_summary.csv')}
g3_map   = {r['strategy']: r for r in read_csv('g3_wfa_e4_summary.csv')}


def find_b9(gf, wn):
    for r in b9_rows:
        if abs(float(r['gold_frac']) - gf) < 0.001 and abs(float(r['wn_min']) - wn) < 0.001:
            return r
    raise ValueError(f'B9 row not found: gf={gf}, wn={wn}')


def find_e4(klo, khi, vzthr):
    for r in e4_rows:
        if (abs(float(r['k_lo']) - klo) < 0.001 and
                abs(float(r['k_hi']) - khi) < 0.001 and
                abs(float(r['vz_thr']) - vzthr) < 0.001):
            return r
    raise ValueError(f'E4 row not found: k_lo={klo}, k_hi={khi}, vz_thr={vzthr}')


def metrics(cagr_oos, sharpe, maxdd, worst10y, p10, gap, tr, ci95=None, wfe=None):
    nan = float('nan')
    return {
        'CAGR_OOS':      cagr_oos,
        'Sharpe_OOS':    sharpe,
        'MaxDD_FULL':    maxdd,
        'Worst10Y_star': worst10y,
        'P10_5Y':        p10,
        'IS_OOS_gap':    gap,
        'Trades_yr':     tr,
        'WFA_CI95_lo':   ci95 if ci95 is not None else nan,
        'WFA_WFE':       wfe if wfe is not None else nan,
    }


# 0. E4 Regime k_lt ◆ (BEST, k_lo=0.1, k_hi=0.8, vz_thr=0.7, G3 WFA PASS 確定)
_e4r = find_e4(0.1, 0.8, 0.7)
_g3e4 = g3_map.get('E4-RegimeKLT') or list(g3_map.values())[0]
e4_klt = metrics(
    float(_e4r['CAGR_OOS']), float(_e4r['Sharpe_OOS']),
    float(_e4r['MaxDD_FULL']), float(_e4r['Worst10Y_star']),
    float(_e4r['P10_5Y']), float(_e4r['IS_OOS_gap']),
    float(_e4r['Trades_yr']),
    float(_g3e4['WFA_CI95_lo']), float(_g3e4['WFA_WFE']),
)

# 1. B9-Winner (gf=0.65, wn_min=0.20)
_b9r = find_b9(0.65, 0.20)
_g2w = g2_map['B9-Winner']
b9_winner = metrics(
    float(_b9r['CAGR_OOS']), float(_b9r['Sharpe_OOS']),
    float(_b9r['MaxDD_FULL']), float(_b9r['Worst10Y_star']),
    float(_b9r['P10_5Y']), float(_b9r['IS_OOS_gap']),
    float(_b9r['Trades_yr']),
    float(_g2w['WFA_CI95_lo']), float(_g2w['WFA_WFE']),
)

# 2. S2+LT2-N750
_n750r = next(r for r in b1_rows if 'N750' in r.get('strategy', ''))
_g1n750 = g1_map['S2+LT2']
n750 = metrics(
    float(_n750r['CAGR_OOS']), float(_n750r['Sharpe_OOS']),
    float(_n750r['MaxDD_FULL']), float(_n750r['Worst10Y_star']),
    float(_n750r['P10_5Y']), float(_n750r['IS_OOS_gap']),
    float(_n750r['n_trades_yr']),
    float(_g1n750['WFA_CI95_lo']), float(_g1n750['WFA_WFE']),
)

# 3. BH 1x (ベンチマーク)
_g1bh = g1_map['BH1x']
bh_1x = metrics(
    0.1011, 0.540, -0.779, -0.057, 0.007, 0.0102, 0,
    float(_g1bh['WFA_CI95_lo']), float(_g1bh['WFA_WFE']),
)

# 4. S2_VZGated (b1 CSV + g1_wfa)
_s2r = next(r for r in b1_rows if 'S2_VZGated' in r.get('strategy', ''))
_g1s2 = g1_map['S2']
s2_vzg = metrics(
    float(_s2r['CAGR_OOS']), float(_s2r['Sharpe_OOS']),
    float(_s2r['MaxDD_FULL']), float(_s2r['Worst10Y_star']),
    float(_s2r['P10_5Y']), float(_s2r['IS_OOS_gap']),
    float(_s2r['n_trades_yr']),
    float(_g1s2['WFA_CI95_lo']), float(_g1s2['WFA_WFE']),
)

# 5. DH Dyn 2x3x [A]
_g1dha = g1_map['DHA']
dh_a = metrics(
    0.1488, 0.646, -0.451, 0.143, 0.096, 0.0848, 27,
    float(_g1dha['WFA_CI95_lo']), float(_g1dha['WFA_WFE']),
)


# ---------------------------------------------------------------------------
# §2 9指標テーブル
# ---------------------------------------------------------------------------

hdr1, hdr2 = MD_HEADER_STRAT

row_e4   = fmt_row_strat('**E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7) ◆**',  e4_klt)
row_b9w  = fmt_row_strat('B9-Winner (gf=0.65, wn=0.20) ✅⚠',              b9_winner)
row_n750 = fmt_row_strat('S2+LT2-N750',                                    n750)
row_bh   = fmt_row_strat('BH 1x（ベンチマーク）',                          bh_1x)
row_s2   = fmt_row_strat('S2_VZGated',                                     s2_vzg)
row_dha  = fmt_row_strat('DH Dyn 2x3x [A]',                                dh_a)


# ---------------------------------------------------------------------------
# §3 年次リターン表
# ---------------------------------------------------------------------------
# E4 列が最左（年の次, ◆ 列）。
# CSV が存在しない場合は static fallback を使用。

# E4 静的フォールバック（gen_e4_yearly_returns.py 2026-05-24 実行結果）
_E4_STATIC = {
    1974:  +19.1, 1975:   -1.8, 1976: +103.0, 1977:  -11.7, 1978: +150.2,
    1979:  +20.9, 1980:  +82.5, 1981:  -38.0, 1982:  +99.2, 1983:   +9.2,
    1984:  -19.4, 1985: +174.0, 1986:  +52.7, 1987:  +71.9, 1988:  -25.6,
    1989:  +63.9, 1990:  -40.5, 1991:  +83.6, 1992:  +62.5, 1993:   -1.9,
    1994:   -8.8, 1995: +164.6, 1996:  +49.9, 1997:  +63.4, 1998:  +91.6,
    1999:  +81.1, 2000:   -9.9, 2001:   -6.9, 2002:  +14.6, 2003: +117.9,
    2004:  +14.8, 2005:  -14.0, 2006:  +51.7, 2007:  +28.0, 2008:  +21.4,
    2009:  +52.8, 2010: +107.2, 2011:   -5.1, 2012:  +39.3, 2013:  +36.1,
    2014:  +13.8, 2015:  -32.3, 2016:   -7.9, 2017:  +95.2, 2018:   -8.3,
    2019:  +66.0, 2020:  +81.3,
    2021:  +30.5, 2022:  -25.9, 2023:  +98.3, 2024:  +58.5, 2025:  +54.4,
    2026:   -7.7,
}


def _load_yearly_csv(fname, year_col='year', val_col=None):
    """CSV があれば優先、なければ None。"""
    csv_path = os.path.join(BASE, fname)
    if not os.path.exists(csv_path):
        return None
    out = {}
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        # 値カラムを推定
        if val_col is None:
            val_col = next(c for c in cols if c != year_col)
        for r in reader:
            out[int(r[year_col])] = float(r[val_col])
    return out


def _load_e4_yearly():
    """e4_yearly_returns.csv があれば優先、なければ static fallback。"""
    loaded = _load_yearly_csv('e4_yearly_returns.csv')
    return loaded if loaded is not None else dict(_E4_STATIC)


_E4_YR   = _load_e4_yearly()

# fmt: (year, b9_winner, n750, bh_1x, s2_vzg, dh_a)
_BASE_YEARLY = [
    # yr   B9-W     N750     BH     S2-VZG   DH-A
    (1974, +26.2,  +19.1,  -35.4,  +10.4,  +10.4),
    (1975,  -5.3,   -2.5,  +29.8,   -3.2,   -5.0),
    (1976, +104.5, +106.4,  +26.1, +103.9,  +45.0),
    (1977,  -11.2,  -12.7,   +7.3,  -10.6,   +0.6),
    (1978, +147.4, +138.5,  +12.3, +136.3,  +49.1),
    (1979,  +31.6,  +21.7,  +28.1,  +35.1,  +38.3),
    (1980,  +66.2,  +71.9,  +33.9,  +79.5,  +44.3),
    (1981,  -38.9,  -36.9,   -3.2,  -42.8,  -30.6),
    (1982,  +92.0,  +97.0,  +18.7, +107.3,  +87.6),
    (1983,  +10.5,  +10.2,  +19.9,  +52.1,  +17.1),
    (1984,  -31.8,  -28.2,  -11.3,  -11.9,   -9.7),
    (1985, +165.0, +172.7,  +31.5, +126.0,  +59.2),
    (1986,  +58.2,  +55.2,   +7.4,  +54.8,  +34.6),
    (1987,  +77.5,  +73.3,   -5.2,  +37.2,  +19.5),
    (1988,  -29.4,  -24.2,  +15.4,  -26.2,  -15.8),
    (1989,  +56.1,  +56.2,  +19.2,  +44.0,  +25.8),
    (1990,  -38.6,  -38.1,  -17.8,  -25.3,  -14.2),
    (1991,  +85.3,  +91.5,  +56.9,  +88.3,  +52.9),
    (1992,  +61.2,  +63.0,  +15.5,  +62.0,  +30.8),
    (1993,   -3.2,   -2.6,  +14.7,  -10.5,   +6.8),
    (1994,   -7.9,   -8.9,   -3.2,  -18.5,  -10.3),
    (1995, +152.6, +163.3,  +39.9, +175.3,  +76.5),
    (1996,  +54.2,  +53.5,  +22.7,  +46.7,  +19.1),
    (1997,  +38.1,  +45.9,  +21.6, +101.6,  +51.7),
    (1998,  +79.6,  +83.7,  +39.6, +100.8,  +78.2),
    (1999,  +62.2,  +57.6,  +85.6, +141.9, +119.1),
    (2000,   -3.6,   +0.6,  -39.3,  -10.8,   +1.1),
    (2001,   -5.9,   -7.0,  -21.1,   +1.5,   +1.5),
    (2002,  +15.0,   +7.3,  -31.5,  +26.3,  +26.3),
    (2003, +125.9, +127.5,  +50.0,  +91.8,  +70.2),
    (2004,  +14.9,  +14.9,   +8.6,  +18.0,  +11.2),
    (2005,  -19.0,  -19.4,   +1.4,  -19.3,   +0.7),
    (2006,  +66.1,  +65.2,   +9.5,  +66.5,  +34.9),
    (2007,  +36.1,  +35.0,   +9.8,  +36.4,  +23.2),
    (2008,  +15.2,  +22.2,  -40.5,  +20.9,  +21.5),
    (2009,  +62.6,  +49.2,  +43.9,  +32.2,  +24.7),
    (2010, +133.1, +131.2,  +16.9, +124.9,  +64.5),
    (2011,   +0.4,   +4.7,   -1.8,   -4.6,   -2.0),
    (2012,  +25.7,  +25.8,  +15.9,  +48.6,  +28.3),
    (2013,  +40.5,  +42.4,  +38.3,  +65.0,  +32.9),
    (2014,   -0.8,   +2.6,  +13.4,   +4.4,  +13.6),
    (2015,  -30.2,  -30.5,   +5.7,  -39.5,  -19.2),
    (2016,  -11.9,  -13.1,   +7.5,  -16.8,   +6.6),
    (2017, +102.7, +103.4,  +28.2,  +84.6,  +35.5),
    (2018,   +4.0,   +3.8,   -3.9,   -9.5,   +0.8),
    (2019,  +62.7,  +62.7,  +35.2,  +56.1,  +45.1),
    (2020,  +95.2,  +97.0,  +43.6, +123.8,  +84.2),
    # --- OOS ---
    (2021,  +28.6,  +28.2,  +21.4,  +48.9,  +24.7),
    (2022,  -20.6,  -26.6,  -33.1,  -30.2,  -30.0),
    (2023, +111.8, +110.2,  +43.4,  +74.6,  +41.2),
    (2024,  +45.1,  +40.2,  +28.6,  +27.6,  +24.4),
    (2025,  +59.0,  +49.1,  +20.4,  +66.4,  +40.2),
    (2026,   -9.9,   -9.7,   -7.9,  -18.3,  -11.7),
]

# 列順: (year, e4, b9w, n750, bh, s2_vzg, dh_a)
YEARLY = [(yr,
           _E4_YR.get(yr, float('nan')),
           b9w, n750r, bhr, s2r, dhar)
          for (yr, b9w, n750r, bhr, s2r, dhar) in _BASE_YEARLY]

OOS_START = 2021


def yr_row(rec):
    yr, e4y, b9w, n750r, bhr, s2r, dhar = rec
    tag = ' [OOS]' if yr >= OOS_START else ''
    return (
        f'| {yr}{tag} '
        f'| {e4y:+.1f} '
        f'| {b9w:+.1f} '
        f'| {n750r:+.1f} '
        f'| {bhr:+.1f} '
        f'| {s2r:+.1f} '
        f'| {dhar:+.1f} |'
    )


yearly_rows = '\n'.join(yr_row(r) for r in YEARLY)


# ---------------------------------------------------------------------------
# §4 OOS詳細分析
# ---------------------------------------------------------------------------

def pct(v):
    return f'{v * 100:+.2f}%'


def pp(v):
    return f'{v * 100:+.2f}pp'


def f3(v):
    return f'{v:+.4f}'


oos_recs = [r for r in YEARLY if r[0] >= OOS_START]
oos_lines = []
for yr, e4y, b9w, n750r, bhr, s2r, dhar in oos_recs:
    oos_lines.append(
        f'| {yr} [OOS] | {e4y:+.1f}% | {b9w:+.1f}% | {n750r:+.1f}% | {bhr:+.1f}% | {s2r:+.1f}% | {dhar:+.1f}% |'
    )
oos_table = '\n'.join(oos_lines)


# ---------------------------------------------------------------------------
# §5 サマリー
# ---------------------------------------------------------------------------

def _best_label(vals_dict):
    return max(vals_dict, key=lambda k: vals_dict[k])


metric_summary_rows = []
for metric_name, extract, fmt_fn in [
    ('CAGR_OOS', lambda m: m['CAGR_OOS'], pct),
    ('Sharpe_OOS', lambda m: m['Sharpe_OOS'], f3),
    ('MaxDD', lambda m: -m['MaxDD_FULL'], lambda v: pct(-v)),   # 絶対値小が優位
    ('Worst10Y★', lambda m: m['Worst10Y_star'], pct),
    ('P10_5Y▷', lambda m: m['P10_5Y'], pct),
    ('IS-OOS gap', lambda m: -abs(m['IS_OOS_gap']), lambda v: pp(-v)),   # gap絶対値小が優位
    ('WFA CI95_lo', lambda m: m['WFA_CI95_lo'], pct),
]:
    strats = {
        'E4 ◆':         extract(e4_klt),
        'B9-Winner':    extract(b9_winner),
        'N750':         extract(n750),
        'S2_VZGated':   extract(s2_vzg),
        'DH Dyn [A]':   extract(dh_a),
        'BH 1x':        extract(bh_1x),
    }
    best = _best_label(strats)
    metric_summary_rows.append(f'| {metric_name} | {best} |')

metric_summary = '\n'.join(metric_summary_rows)


# ---------------------------------------------------------------------------
# レポート組み立て
# ---------------------------------------------------------------------------

report = f"""\
# 戦略パフォーマンス比較表 — 6戦略 統一評価フレームワーク

作成日: 2026-05-23
最終更新日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**
生成スクリプト: `src/gen_strategy_comparison.py`

> ### ◆ 現行ベスト戦略
> **E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7)**
> CAGR_OOS **+{e4_klt['CAGR_OOS']*100:.2f}%** | Sharpe **+{e4_klt['Sharpe_OOS']:.3f}** | MaxDD **−{abs(e4_klt['MaxDD_FULL'])*100:.1f}%** | Trades/yr **{e4_klt['Trades_yr']:.0f}** | G3 WFA PASS ✓

---

## 📋 §1 比較前提

| 項目 | 定義 |
|------|------|
| **IS** | 1974-01-02 〜 2021-05-07（47.3年） |
| **OOS** | 2021-05-08 〜 2026-03-26（4.9年） |
| **FULL** | 1974-01-02 〜 2026-03-26（52.26年） |
| **コスト** | Scenario D（`src/product_costs.py` 2026-05-12 基準） |
| **DELAY** | 2営業日（look-ahead bias 対策） |
| **Sharpe Rf** | 0 |
| **CURRENT_BEST** | E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7)（◆, G3 WFA PASS 確定） |
| **WFA** | G1: 49窓, G2: 49窓, G3: 49窓（252日 calendar-year-anchored non-overlapping） |

| 凡例 | 意味 |
|------|------|
| ◆ | 現行ベスト戦略 |
| ✅ | Shortlisted（WFA PASS・ベスト昇格候補 / または棄却理由付き次善候補） |
| ⚠ | OOS期間バイアス警戒（Gold 2021-2026 強気エクスポージャ疑い） |
| [OOS] | OOS期間（2021年以降） |

---

## 📊 §2 9指標比較表（6戦略 × 9指標）

> **単位**: CAGR_OOS / Worst10Y★ / P10▷ / MaxDD = %、IS-OOS gap = pp、Tr = 回/年
> ★ = Sharpe_OOS > +0.885 / ◎ = > +0.770（S2ベースライン）
> WFA_CI95_lo / WFA_WFE に `—` は TBD または未実施

{hdr1}
{hdr2}
{row_e4}
{row_b9w}
{row_n750}
{row_bh}
{row_s2}
{row_dha}

{MD_METRIC_GLOSSARY}

---

## 📈 §3 年次リターン表（1974–2026）[単位: %]

> `[OOS]` = OOS期間（2021年以降）
> E4 列は `src/gen_e4_yearly_returns.py` で生成（暦年ベース、CFD レバレッジ S2_VZGated + LT2-N750 + Regime k_lt）。

| 年 | E4 Regime<br>k_lt ◆ | B9-Winner<br>✅⚠ | S2+LT2<br>N750 | BH 1x<br>ベンチマーク | S2_VZGated | DH Dyn<br>2x3x[A] |
|:---|---:|---:|---:|---:|---:|---:|
{yearly_rows}

---

## 🔍 §4 OOS期間（2021–2026）詳細

| 年 | E4 Regime ◆ | B9-Winner | S2+LT2-N750 | BH 1x | S2_VZGated | DH Dyn [A] |
|:---|---:|---:|---:|---:|---:|---:|
{oos_table}

**注目ポイント**:

> **コスト最優先で E4 ◆ 確定**: E4 Trades/yr={e4_klt['Trades_yr']:.0f} は他戦略に対し最低水準。
> スプレッド・税率20.315%・スワップを考慮するとコスト負担差は実運用 Sharpe に直接影響する。

> **2022年防御**: E4 -25.9% / B9-Winner -20.6% / N750 -26.6% / S2_VZGated -30.2% / DH[A] -30.0%。
> B9-Winner が最良の下落抑制。

> **B9 IS-OOS gap = {b9_winner['IS_OOS_gap']*100:+.2f}pp**: OOS期間（2021-2026）はGold ETF累積+60%超の強気相場。
> gold_frac増加 → OOS側だけ有利なため gap が負方向に傾く。Gold overfit 疑い（`B9_COMPARISON_2026-05-23.md §4-(a)` 参照）。

> **N750 gap = {n750['IS_OOS_gap']*100:+.2f}pp**: 最小 gap → 単純設計ながら最も汎化性が高い。

---

## 🏆 §5 採用判断サマリー

| 指標 | 優勝戦略 |
|------|---------|
{metric_summary}

**判定**:

> **◆ 現行ベスト確定: E4 Regime k_lt (lo=0.1/hi=0.8/vz=0.7)**
> - CAGR_OOS +{e4_klt['CAGR_OOS']*100:.2f}%, Sharpe +{e4_klt['Sharpe_OOS']:.3f}, Trades/yr={e4_klt['Trades_yr']:.0f}（コスト最小）
> - MaxDD −{abs(e4_klt['MaxDD_FULL'])*100:.2f}%（guardrail −65.0% 内）
> - IS-OOS gap {e4_klt['IS_OOS_gap']*100:+.2f}pp（最小 gap 群、汎化性最良）
> - **G3 WFA PASS**: CI95_lo=+{e4_klt['WFA_CI95_lo']*100:.2f}%（α PASS）/ WFE=+{e4_klt['WFA_WFE']:.3f}（β PASS）→ 正式 Active 確定。

- **S2+LT2-N750 Shortlisted（汎化最強, WFA 完了済み fallback）**
  - WFA CI95_lo={n750['WFA_CI95_lo']*100:.1f}%, WFE={n750['WFA_WFE']:.3f}（PASS α∩β）
  - IS-OOS gap {n750['IS_OOS_gap']*100:+.2f}pp = 最小群。Sharpe +{n750['Sharpe_OOS']:.3f}, MaxDD {n750['MaxDD_FULL']*100:.2f}%
- **B9-Winner ✅⚠ Shortlisted（保留）**
  - CAGR_OOS・Sharpe・2022年防御では有力だが IS-OOS gap {b9_winner['IS_OOS_gap']*100:+.2f}pp は Gold OOS bias 疑い
- **S2_VZGated**: G1 WFA CI95_lo最高({s2_vzg['WFA_CI95_lo']*100:.1f}%), WFE=0.830 PASS。Sharpe +0.770（基準線）
- **DH Dyn [A]**: 低MaxDD優位(-45.1%)。Sharpe/WFE 最低クラスだが下方リスク管理として参照価値あり

---

## 📁 §6 一次根拠ファイル

| ファイル | 役割 |
|----------|------|
| [e4_regime_klt_results.csv](e4_regime_klt_results.csv) | E4 ◆ 9指標ソース（64 config） |
| [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md) | E4 sweep レポート（採用根拠） |
| [e4_yearly_returns.csv](e4_yearly_returns.csv) | E4 年次リターン |
| [g3_wfa_e4_summary.csv](g3_wfa_e4_summary.csv) | E4 Regime k_lt WFAデータ（G3 PASS） |
| [G3_WFA_E4_2026-05-24.md](G3_WFA_E4_2026-05-24.md) | G3 WFA レポート（E4 Active 正式確定） |
| [b9_s2lt2_goldfrac_results.csv](b9_s2lt2_goldfrac_results.csv) | B9-Winner 9指標ソース |
| [b1_s2_lt2_results.csv](b1_s2_lt2_results.csv) | N750 / S2_VZGated 9指標ソース |
| [g1_wfa_summary.csv](g1_wfa_summary.csv) | N750/S2_VZGated/DH[A]/BH1x WFAデータ |
| [g2_wfa_b9_summary.csv](g2_wfa_b9_summary.csv) | B9-Winner WFAデータ |
| [B9_YEARLY_RETURNS_2026-05-23.md](B9_YEARLY_RETURNS_2026-05-23.md) | B9-Winner/N750/BH年次リターン元データ |
| [CFD_S2_YEARLY_RETURNS_2026-05-17.md](CFD_S2_YEARLY_RETURNS_2026-05-17.md) | S2_VZGated/DH[A]年次リターン元データ |
| [B9_COMPARISON_2026-05-23.md](B9_COMPARISON_2026-05-23.md) | B9 詳細比較（OOSバイアス論拠 §4） |
| [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) | ベスト戦略単一の真実 |
| [EVALUATION_STANDARD.md](EVALUATION_STANDARD.md) | 評価基準 v1.1 |

---

## 📝 §7 改訂履歴

| 版 | 日付 | 変更内容 |
|----|------|---------|
| **v1.6** | 2026-05-24 | F8-R5 CALM_BOOST・F7v3+E4 列を削除（採用不採用確定、Trades/yr過多）。6戦略統一評価フレームワークに縮小。 |
| v1.5 | 2026-05-24 | E4 を ◆ BEST に復帰。F8-R5 CALM_BOOST (✅) を新列追加（8戦略へ拡張）。F7v3+E4 を ✅ Shortlisted へ降格。Trades/yr 過多（182-183回）によるコスト問題が棄却理由。`src/gen_f8r5_yearly_returns.py` で F8-R5 年次リターン生成。 |
| v1.4 | 2026-05-24 | F7v3+E4 (定式A tilt=2.0/cap=0.10) を新 BEST ◆ として追加（7戦略へ拡張）。G4 WFA PASS 確定。E4 Regime k_lt は Shortlisted ✅ へ降格 fallback。`src/gen_f7v3_yearly_returns.py` で年次リターン生成。 |
| v1.3 | 2026-05-24 | G3 WFA 完了。E4 Regime k_lt の CI95_lo=+26.51% / WFE=+1.131 を §2 表に反映。暫定 Active → 正式 Active 確定。 |
| v1.2 | 2026-05-24 | E4 年次リターン列を §3/§4 に追加（`src/gen_e4_yearly_returns.py` にて計算）。 |
| v1.1 | 2026-05-24 | E4 Regime k_lt を新 BEST ◆ として追加（6戦略へ拡張）。N750 を旧◆参照行へ。WFA TBD。 |
| v1.0 | 2026-05-23 | 初版。5戦略統一フォーマット（9指標＋年次リターン）。 |

---

*管理者: Kazuya Murayama*
*準拠: `EVALUATION_STANDARD.md v1.1` / `src/_sweep_format.py MD_HEADER_STRAT`*
*今後このフォーマット（§2 9指標表 ＋ §3 年次リターン表）を戦略評価の標準レポートとして使用する*
"""

# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------

out_path = os.path.join(BASE, 'STRATEGY_PERFORMANCE_COMPARISON_2026-05-24.md')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f'Written: {out_path}')
print()
print('=== 9指標確認 ===')
for label, m in [
    ('E4 Regime◆', e4_klt),
    ('B9-Winner', b9_winner),
    ('N750     ', n750),
    ('BH 1x    ', bh_1x),
    ('S2_VZGate', s2_vzg),
    ('DH Dyn[A]', dh_a),
]:
    print(
        f'  {label}: CAGR_OOS={m["CAGR_OOS"]*100:+.2f}%'
        f'  Sharpe={m["Sharpe_OOS"]:+.4f}'
        f'  MaxDD={m["MaxDD_FULL"]*100:+.1f}%'
        f'  Trades/yr={m["Trades_yr"]:.0f}'
        f'  CI95_lo={m["WFA_CI95_lo"]*100:+.1f}%'
        f'  WFE={m["WFA_WFE"]:+.3f}'
    )
