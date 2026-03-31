"""Generate YEARLY_RETURNS_REPORT.md with colored cells"""
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yr_csv = os.path.join(BASE, 'yearly_returns_7strategies.csv')
mo_csv = os.path.join(BASE, 'monthly_returns_oos.csv')

yr = pd.read_csv(yr_csv, index_col=0)
mo = pd.read_csv(mo_csv, index_col=0)

order = ['DH Static (35/30/35)', 'DH Dynamic CAGR25+', 'A2 Optimized',
         'Ens2(Asym+Slope)', 'DD Only', 'BH 3x', 'BH 1x']
short = ['DH Static', 'DH Dyn 25+', 'A2 Opt', 'Ens2', 'DD Only', 'BH 3x', 'BH 1x']

cagrs = {'DH Static': 16.07, 'DH Dyn 25+': 25.23, 'A2 Opt': 29.19,
         'Ens2': 22.20, 'DD Only': 25.58, 'BH 3x': 19.21, 'BH 1x': 10.98}

def colored(v):
    """Return HTML-colored value string"""
    if v > 0:
        return f'<span style="color:blue">+{v:.1f}%</span>'
    elif v < 0:
        return f'<span style="color:red">{v:.1f}%</span>'
    else:
        return f'{v:.1f}%'

md = """# 7戦略 年次リターン（1974-2026）

## 検証条件

| 項目 | 値 |
|------|-----|
| データ期間 | 1974-01-02 〜 2026-03-26 |
| 実行遅延 | 2営業日 |
| 経費率 | 年0.86%（TQQQ準拠） |
| リバランス閾値 | 20% |

---

## 各戦略の概要

| 戦略 | 概要 |
|------|------|
| **DH Static (35/30/35)** * | A2のNAVに Gold 30% / Bond 35% を加えた3資産ポートフォリオ。四半期リバランス |
| **DH Dyn CAGR25+** * | A2のレバレッジとVIXシグナルで NASDAQ/Gold/Bond 比率を動的調整（CAGR 25%+制約版） |
| **A2 Optimized** | DD制御 + AsymEWMA VT + SlopeMult + MomDecel(60/180) + VIX Mean Reversion。単一資産最良 |
| **Ens2(Asym+Slope)** | DD制御 + AsymEWMA(20/5) + SlopeMult(0.7/0.3)。旧推奨戦略 |
| **DD Only** | 200日高値から-18%でCASH退避、92%回復でHOLD。最もシンプルな管理戦略 |
| **BH 3x** | NASDAQ 3倍レバレッジ（TQQQ相当）を無管理で保有 |
| **BH 1x** | レバレッジなしのNASDAQ指数をそのまま保有。ベンチマーク |

> \\* 3資産ポートフォリオ（NASDAQ 3x + Gold 447A + Bond 2621）

---

## 統計サマリー

"""

# Stats table
md += "| 統計量 | DH Static | DH Dyn 25+ | A2 Opt | Ens2 | DD Only | BH 3x | BH 1x |\n"
md += "|--------|-----------|------------|--------|------|---------|-------|-------|\n"

for stat_name, func in [
    ('CAGR', None),
    ('中央値', lambda s: s.median()),
    ('最大', lambda s: s.max()),
    ('最小', lambda s: s.min()),
    ('標準偏差', lambda s: s.std()),
    ('プラス年数', lambda s: (s > 0).sum()),
    ('マイナス年数', lambda s: (s <= 0).sum()),
]:
    md += f"| {stat_name} |"
    for name, sn in zip(order, short):
        if stat_name == 'CAGR':
            md += f" {cagrs[sn]:+.2f}% |"
        elif stat_name in ['プラス年数', 'マイナス年数']:
            md += f" {int(func(yr[name]))} |"
        else:
            md += f" {func(yr[name]):+.1f}% |"
    md += "\n"

md += """
---

## 年次リターン表（%）

"""

md += "| Year | DH Static | DH Dyn 25+ | A2 Opt | Ens2 | DD Only | BH 3x | BH 1x |\n"
md += "|------|-----------|------------|--------|------|---------|-------|-------|\n"

for year in yr.index:
    md += f"| {year} |"
    for name in order:
        v = yr.loc[year, name]
        md += f" {colored(v)} |"
    md += "\n"

md += """
---

## 月次リターン表（2021-2026, OOS期間）（%）

"""

md += "| Year-Month | DH Static | DH Dyn 25+ | A2 Opt | Ens2 | DD Only | BH 3x | BH 1x |\n"
md += "|------------|-----------|------------|--------|------|---------|-------|-------|\n"

for ym in mo.index:
    md += f"| {ym} |"
    for name in order:
        v = mo.loc[ym, name]
        md += f" {colored(v)} |"
    md += "\n"

md += """
---

*Generated: 2026-03-31*
"""

out = os.path.join(BASE, 'YEARLY_RETURNS_REPORT.md')
with open(out, 'w', encoding='utf-8') as f:
    f.write(md)
print(f"Written: {out}")
