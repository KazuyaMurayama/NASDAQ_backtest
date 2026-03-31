# 7戦略 年次リターン（1974-2026）

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

> \* 3資産ポートフォリオ（NASDAQ 3x + Gold 447A + Bond 2621）

---

## 統計サマリー

| 統計量 | DH Static | DH Dyn 25+ | A2 Opt | Ens2 | DD Only | BH 3x | BH 1x |
|--------|-----------|------------|--------|------|---------|-------|-------|
| CAGR | +16.07% | +25.23% | +29.19% | +22.20% | +25.58% | +19.21% | +10.98% |
| 中央値 | +16.5% | +28.8% | +28.8% | +15.0% | +21.7% | +39.4% | +14.9% |
| 最大 | +56.1% | +132.3% | +214.8% | +142.2% | +394.8% | +394.8% | +84.3% |
| 最小 | -11.4% | -14.9% | -27.7% | -42.8% | -47.9% | -89.7% | -40.2% |
| 標準偏差 | +16.8% | +29.4% | +44.2% | +40.0% | +74.2% | +88.9% | +24.9% |
| プラス年数 | 43 | 44 | 40 | 39 | 35 | 39 | 39 |
| マイナス年数 | 10 | 9 | 13 | 14 | 18 | 14 | 14 |

---

## 年次リターン表（%）

| Year | DH Static | DH Dyn 25+ | A2 Opt | Ens2 | DD Only | BH 3x | BH 1x |
|------|-----------|------------|--------|------|---------|-------|-------|
| 1974 | <span style="color:blue">+12.9%</span> | <span style="color:blue">+11.1%</span> | 0.0% | <span style="color:red">-42.8%</span> | <span style="color:red">-42.8%</span> | <span style="color:red">-75.8%</span> | <span style="color:red">-35.4%</span> |
| 1975 | <span style="color:red">-0.0%</span> | <span style="color:blue">+5.3%</span> | <span style="color:blue">+14.4%</span> | <span style="color:blue">+15.0%</span> | <span style="color:blue">+3.8%</span> | <span style="color:blue">+96.8%</span> | <span style="color:blue">+27.9%</span> |
| 1976 | <span style="color:blue">+28.9%</span> | <span style="color:blue">+49.7%</span> | <span style="color:blue">+66.3%</span> | <span style="color:blue">+34.3%</span> | <span style="color:blue">+91.3%</span> | <span style="color:blue">+91.3%</span> | <span style="color:blue">+25.4%</span> |
| 1977 | <span style="color:blue">+10.2%</span> | <span style="color:blue">+7.4%</span> | <span style="color:blue">+3.2%</span> | <span style="color:blue">+6.9%</span> | <span style="color:blue">+21.7%</span> | <span style="color:blue">+21.7%</span> | <span style="color:blue">+7.5%</span> |
| 1978 | <span style="color:blue">+30.9%</span> | <span style="color:blue">+67.7%</span> | <span style="color:blue">+65.6%</span> | <span style="color:blue">+52.1%</span> | <span style="color:blue">+17.8%</span> | <span style="color:blue">+39.0%</span> | <span style="color:blue">+13.4%</span> |
| 1979 | <span style="color:blue">+55.3%</span> | <span style="color:blue">+45.8%</span> | <span style="color:blue">+20.7%</span> | <span style="color:blue">+14.1%</span> | <span style="color:blue">+54.6%</span> | <span style="color:blue">+101.9%</span> | <span style="color:blue">+28.3%</span> |
| 1980 | <span style="color:blue">+22.6%</span> | <span style="color:blue">+63.1%</span> | <span style="color:blue">+87.5%</span> | <span style="color:blue">+69.6%</span> | <span style="color:blue">+57.3%</span> | <span style="color:blue">+133.5%</span> | <span style="color:blue">+36.6%</span> |
| 1981 | <span style="color:red">-11.4%</span> | <span style="color:red">-10.2%</span> | <span style="color:red">-8.3%</span> | <span style="color:red">-20.2%</span> | <span style="color:red">-33.1%</span> | <span style="color:red">-15.6%</span> | <span style="color:red">-3.8%</span> |
| 1982 | <span style="color:blue">+56.1%</span> | <span style="color:blue">+66.2%</span> | <span style="color:blue">+82.4%</span> | <span style="color:blue">+77.3%</span> | <span style="color:blue">+87.8%</span> | <span style="color:blue">+58.2%</span> | <span style="color:blue">+18.9%</span> |
| 1983 | <span style="color:blue">+19.4%</span> | <span style="color:blue">+43.6%</span> | <span style="color:blue">+76.1%</span> | <span style="color:blue">+80.6%</span> | <span style="color:blue">+54.8%</span> | <span style="color:blue">+67.6%</span> | <span style="color:blue">+20.8%</span> |
| 1984 | <span style="color:red">-2.1%</span> | <span style="color:red">-2.2%</span> | <span style="color:red">-4.7%</span> | <span style="color:red">-6.8%</span> | <span style="color:red">-4.7%</span> | <span style="color:red">-33.6%</span> | <span style="color:red">-11.0%</span> |
| 1985 | <span style="color:blue">+45.8%</span> | <span style="color:blue">+68.0%</span> | <span style="color:blue">+84.4%</span> | <span style="color:blue">+72.0%</span> | <span style="color:blue">+124.3%</span> | <span style="color:blue">+124.3%</span> | <span style="color:blue">+32.1%</span> |
| 1986 | <span style="color:blue">+26.1%</span> | <span style="color:blue">+34.8%</span> | <span style="color:blue">+28.8%</span> | <span style="color:blue">+34.4%</span> | <span style="color:blue">+18.7%</span> | <span style="color:blue">+18.7%</span> | <span style="color:blue">+7.3%</span> |
| 1987 | <span style="color:blue">+10.9%</span> | <span style="color:blue">+20.8%</span> | <span style="color:blue">+11.1%</span> | <span style="color:blue">+3.0%</span> | <span style="color:red">-29.6%</span> | <span style="color:red">-34.4%</span> | <span style="color:red">-6.4%</span> |
| 1988 | <span style="color:red">-3.5%</span> | <span style="color:red">-5.3%</span> | <span style="color:red">-4.0%</span> | <span style="color:red">-3.3%</span> | <span style="color:red">-5.9%</span> | <span style="color:blue">+37.8%</span> | <span style="color:blue">+12.7%</span> |
| 1989 | <span style="color:blue">+23.7%</span> | <span style="color:blue">+40.8%</span> | <span style="color:blue">+44.9%</span> | <span style="color:blue">+43.6%</span> | <span style="color:blue">+67.9%</span> | <span style="color:blue">+67.9%</span> | <span style="color:blue">+20.1%</span> |
| 1990 | <span style="color:red">-3.0%</span> | <span style="color:red">-5.2%</span> | <span style="color:red">-13.0%</span> | <span style="color:red">-22.4%</span> | <span style="color:red">-47.9%</span> | <span style="color:red">-50.5%</span> | <span style="color:red">-18.6%</span> |
| 1991 | <span style="color:blue">+30.2%</span> | <span style="color:blue">+54.8%</span> | <span style="color:blue">+86.5%</span> | <span style="color:blue">+79.9%</span> | <span style="color:blue">+123.5%</span> | <span style="color:blue">+262.1%</span> | <span style="color:blue">+57.5%</span> |
| 1992 | <span style="color:blue">+16.8%</span> | <span style="color:blue">+29.8%</span> | <span style="color:blue">+45.5%</span> | <span style="color:blue">+22.4%</span> | <span style="color:blue">+44.9%</span> | <span style="color:blue">+44.9%</span> | <span style="color:blue">+15.4%</span> |
| 1993 | <span style="color:blue">+11.4%</span> | <span style="color:blue">+9.3%</span> | <span style="color:blue">+4.0%</span> | <span style="color:blue">+10.0%</span> | <span style="color:blue">+47.2%</span> | <span style="color:blue">+47.2%</span> | <span style="color:blue">+15.6%</span> |
| 1994 | <span style="color:red">-2.3%</span> | <span style="color:blue">+0.1%</span> | <span style="color:blue">+0.2%</span> | <span style="color:blue">+0.7%</span> | <span style="color:red">-11.6%</span> | <span style="color:red">-11.6%</span> | <span style="color:red">-2.4%</span> |
| 1995 | <span style="color:blue">+47.4%</span> | <span style="color:blue">+73.8%</span> | <span style="color:blue">+114.8%</span> | <span style="color:blue">+91.0%</span> | <span style="color:blue">+166.0%</span> | <span style="color:blue">+166.0%</span> | <span style="color:blue">+41.5%</span> |
| 1996 | <span style="color:blue">+12.2%</span> | <span style="color:blue">+27.0%</span> | <span style="color:blue">+43.0%</span> | <span style="color:blue">+34.1%</span> | <span style="color:blue">+67.2%</span> | <span style="color:blue">+67.2%</span> | <span style="color:blue">+22.0%</span> |
| 1997 | <span style="color:blue">+28.3%</span> | <span style="color:blue">+63.4%</span> | <span style="color:blue">+99.1%</span> | <span style="color:blue">+97.2%</span> | <span style="color:blue">+64.4%</span> | <span style="color:blue">+64.4%</span> | <span style="color:blue">+22.6%</span> |
| 1998 | <span style="color:blue">+31.9%</span> | <span style="color:blue">+68.4%</span> | <span style="color:blue">+84.1%</span> | <span style="color:blue">+69.0%</span> | <span style="color:blue">+21.5%</span> | <span style="color:blue">+112.9%</span> | <span style="color:blue">+38.6%</span> |
| 1999 | <span style="color:blue">+55.3%</span> | <span style="color:blue">+132.3%</span> | <span style="color:blue">+214.8%</span> | <span style="color:blue">+142.2%</span> | <span style="color:blue">+394.8%</span> | <span style="color:blue">+394.8%</span> | <span style="color:blue">+84.3%</span> |
| 2000 | <span style="color:blue">+7.3%</span> | <span style="color:red">-3.2%</span> | <span style="color:red">-8.6%</span> | <span style="color:red">-16.0%</span> | <span style="color:red">-36.7%</span> | <span style="color:red">-89.7%</span> | <span style="color:red">-40.2%</span> |
| 2001 | <span style="color:blue">+2.5%</span> | <span style="color:blue">+1.6%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-64.1%</span> | <span style="color:red">-14.9%</span> |
| 2002 | <span style="color:blue">+12.3%</span> | <span style="color:blue">+10.5%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-78.6%</span> | <span style="color:red">-32.5%</span> |
| 2003 | <span style="color:blue">+39.5%</span> | <span style="color:blue">+67.6%</span> | <span style="color:blue">+99.3%</span> | <span style="color:blue">+127.9%</span> | <span style="color:blue">+175.5%</span> | <span style="color:blue">+159.7%</span> | <span style="color:blue">+44.7%</span> |
| 2004 | <span style="color:blue">+6.5%</span> | <span style="color:blue">+10.7%</span> | <span style="color:blue">+6.4%</span> | <span style="color:blue">+5.5%</span> | <span style="color:red">-20.5%</span> | <span style="color:blue">+15.9%</span> | <span style="color:blue">+8.4%</span> |
| 2005 | <span style="color:blue">+8.5%</span> | <span style="color:blue">+5.8%</span> | <span style="color:blue">+1.9%</span> | <span style="color:blue">+8.3%</span> | <span style="color:blue">+1.8%</span> | <span style="color:blue">+1.8%</span> | <span style="color:blue">+2.5%</span> |
| 2006 | <span style="color:blue">+22.6%</span> | <span style="color:blue">+35.0%</span> | <span style="color:blue">+45.2%</span> | <span style="color:blue">+14.0%</span> | <span style="color:blue">+16.6%</span> | <span style="color:blue">+16.6%</span> | <span style="color:blue">+7.6%</span> |
| 2007 | <span style="color:blue">+17.9%</span> | <span style="color:blue">+17.3%</span> | <span style="color:blue">+12.3%</span> | <span style="color:blue">+9.0%</span> | <span style="color:blue">+18.7%</span> | <span style="color:blue">+18.7%</span> | <span style="color:blue">+9.5%</span> |
| 2008 | <span style="color:blue">+5.4%</span> | <span style="color:blue">+3.1%</span> | <span style="color:red">-6.2%</span> | <span style="color:red">-11.9%</span> | <span style="color:red">-33.2%</span> | <span style="color:red">-87.0%</span> | <span style="color:red">-39.6%</span> |
| 2009 | <span style="color:blue">+22.3%</span> | <span style="color:blue">+43.3%</span> | <span style="color:blue">+54.3%</span> | <span style="color:blue">+48.7%</span> | <span style="color:blue">+59.2%</span> | <span style="color:blue">+110.4%</span> | <span style="color:blue">+39.0%</span> |
| 2010 | <span style="color:blue">+19.4%</span> | <span style="color:blue">+30.2%</span> | <span style="color:blue">+20.3%</span> | <span style="color:blue">+36.1%</span> | <span style="color:blue">+33.9%</span> | <span style="color:blue">+33.9%</span> | <span style="color:blue">+14.9%</span> |
| 2011 | <span style="color:red">-3.1%</span> | <span style="color:red">-14.9%</span> | <span style="color:red">-27.7%</span> | <span style="color:red">-30.7%</span> | <span style="color:red">-44.0%</span> | <span style="color:red">-25.9%</span> | <span style="color:red">-3.2%</span> |
| 2012 | <span style="color:blue">+16.5%</span> | <span style="color:blue">+27.0%</span> | <span style="color:blue">+34.6%</span> | <span style="color:blue">+22.5%</span> | <span style="color:blue">+37.5%</span> | <span style="color:blue">+37.5%</span> | <span style="color:blue">+14.0%</span> |
| 2013 | <span style="color:blue">+7.9%</span> | <span style="color:blue">+30.0%</span> | <span style="color:blue">+68.5%</span> | <span style="color:blue">+79.7%</span> | <span style="color:blue">+129.4%</span> | <span style="color:blue">+129.4%</span> | <span style="color:blue">+34.2%</span> |
| 2014 | <span style="color:blue">+5.7%</span> | <span style="color:blue">+5.4%</span> | <span style="color:blue">+10.6%</span> | <span style="color:blue">+10.4%</span> | <span style="color:blue">+39.4%</span> | <span style="color:blue">+39.4%</span> | <span style="color:blue">+14.3%</span> |
| 2015 | <span style="color:red">-7.3%</span> | <span style="color:red">-11.3%</span> | <span style="color:red">-13.2%</span> | <span style="color:blue">+3.7%</span> | <span style="color:blue">+8.2%</span> | <span style="color:blue">+8.2%</span> | <span style="color:blue">+5.9%</span> |
| 2016 | <span style="color:blue">+3.3%</span> | <span style="color:blue">+5.5%</span> | <span style="color:blue">+0.1%</span> | <span style="color:red">-0.1%</span> | <span style="color:red">-13.7%</span> | <span style="color:blue">+21.7%</span> | <span style="color:blue">+9.8%</span> |
| 2017 | <span style="color:blue">+20.4%</span> | <span style="color:blue">+31.2%</span> | <span style="color:blue">+46.5%</span> | <span style="color:blue">+36.3%</span> | <span style="color:blue">+98.2%</span> | <span style="color:blue">+98.2%</span> | <span style="color:blue">+27.2%</span> |
| 2018 | <span style="color:blue">+0.9%</span> | <span style="color:blue">+7.5%</span> | <span style="color:blue">+2.1%</span> | <span style="color:red">-12.1%</span> | <span style="color:red">-28.7%</span> | <span style="color:red">-26.2%</span> | <span style="color:red">-5.3%</span> |
| 2019 | <span style="color:blue">+24.7%</span> | <span style="color:blue">+38.0%</span> | <span style="color:blue">+47.6%</span> | <span style="color:blue">+33.6%</span> | <span style="color:blue">+61.2%</span> | <span style="color:blue">+124.4%</span> | <span style="color:blue">+34.6%</span> |
| 2020 | <span style="color:blue">+31.0%</span> | <span style="color:blue">+57.2%</span> | <span style="color:blue">+62.0%</span> | <span style="color:blue">+64.4%</span> | <span style="color:blue">+78.6%</span> | <span style="color:blue">+88.5%</span> | <span style="color:blue">+41.8%</span> |
| 2021 | <span style="color:blue">+11.6%</span> | <span style="color:blue">+28.8%</span> | <span style="color:blue">+45.7%</span> | <span style="color:blue">+28.1%</span> | <span style="color:blue">+68.2%</span> | <span style="color:blue">+68.2%</span> | <span style="color:blue">+23.2%</span> |
| 2022 | <span style="color:red">-8.7%</span> | <span style="color:red">-12.4%</span> | <span style="color:red">-14.4%</span> | <span style="color:red">-21.5%</span> | <span style="color:red">-41.0%</span> | <span style="color:red">-79.0%</span> | <span style="color:red">-33.9%</span> |
| 2023 | <span style="color:blue">+23.9%</span> | <span style="color:blue">+39.7%</span> | <span style="color:blue">+51.4%</span> | <span style="color:blue">+40.4%</span> | <span style="color:blue">+84.9%</span> | <span style="color:blue">+173.5%</span> | <span style="color:blue">+44.5%</span> |
| 2024 | <span style="color:blue">+19.6%</span> | <span style="color:blue">+30.0%</span> | <span style="color:blue">+31.7%</span> | <span style="color:blue">+36.0%</span> | <span style="color:blue">+100.9%</span> | <span style="color:blue">+100.9%</span> | <span style="color:blue">+30.8%</span> |
| 2025 | <span style="color:blue">+32.3%</span> | <span style="color:blue">+37.0%</span> | <span style="color:blue">+29.4%</span> | <span style="color:blue">+13.5%</span> | <span style="color:red">-13.8%</span> | <span style="color:blue">+46.4%</span> | <span style="color:blue">+20.5%</span> |
| 2026 | <span style="color:red">-3.7%</span> | <span style="color:red">-8.4%</span> | <span style="color:red">-11.1%</span> | <span style="color:red">-15.2%</span> | <span style="color:red">-23.5%</span> | <span style="color:red">-23.5%</span> | <span style="color:red">-7.9%</span> |

---

## 月次リターン表（2021-2026, OOS期間）（%）

| Year-Month | DH Static | DH Dyn 25+ | A2 Opt | Ens2 | DD Only | BH 3x | BH 1x |
|------------|-----------|------------|--------|------|---------|-------|-------|
| 2021-01 | <span style="color:blue">+0.4%</span> | <span style="color:blue">+4.2%</span> | <span style="color:blue">+6.7%</span> | <span style="color:blue">+8.0%</span> | <span style="color:blue">+8.0%</span> | <span style="color:blue">+8.0%</span> | <span style="color:blue">+2.9%</span> |
| 2021-02 | <span style="color:red">-4.6%</span> | <span style="color:red">-4.6%</span> | <span style="color:red">-4.4%</span> | <span style="color:red">-5.7%</span> | <span style="color:red">-5.5%</span> | <span style="color:red">-5.5%</span> | <span style="color:red">-1.6%</span> |
| 2021-03 | <span style="color:red">-2.4%</span> | <span style="color:red">-3.0%</span> | <span style="color:red">-3.9%</span> | <span style="color:red">-1.7%</span> | <span style="color:red">-9.2%</span> | <span style="color:red">-9.2%</span> | <span style="color:red">-2.5%</span> |
| 2021-04 | <span style="color:blue">+2.2%</span> | <span style="color:blue">+2.8%</span> | <span style="color:blue">+3.9%</span> | <span style="color:blue">+2.2%</span> | <span style="color:blue">+10.5%</span> | <span style="color:blue">+10.5%</span> | <span style="color:blue">+3.6%</span> |
| 2021-05 | <span style="color:blue">+0.1%</span> | <span style="color:red">-1.8%</span> | <span style="color:red">-5.2%</span> | <span style="color:red">-4.7%</span> | <span style="color:red">-4.1%</span> | <span style="color:red">-4.1%</span> | <span style="color:red">-1.1%</span> |
| 2021-06 | <span style="color:blue">+1.6%</span> | <span style="color:blue">+5.3%</span> | <span style="color:blue">+10.1%</span> | <span style="color:blue">+6.5%</span> | <span style="color:blue">+17.3%</span> | <span style="color:blue">+17.3%</span> | <span style="color:blue">+5.6%</span> |
| 2021-07 | <span style="color:blue">+2.2%</span> | <span style="color:blue">+2.4%</span> | <span style="color:blue">+2.6%</span> | <span style="color:blue">+2.4%</span> | <span style="color:blue">+2.7%</span> | <span style="color:blue">+2.7%</span> | <span style="color:blue">+1.0%</span> |
| 2021-08 | <span style="color:blue">+3.8%</span> | <span style="color:blue">+8.3%</span> | <span style="color:blue">+11.2%</span> | <span style="color:blue">+8.0%</span> | <span style="color:blue">+11.9%</span> | <span style="color:blue">+11.9%</span> | <span style="color:blue">+3.9%</span> |
| 2021-09 | <span style="color:red">-4.6%</span> | <span style="color:red">-6.6%</span> | <span style="color:red">-8.7%</span> | <span style="color:red">-8.5%</span> | <span style="color:red">-16.5%</span> | <span style="color:red">-16.5%</span> | <span style="color:red">-5.6%</span> |
| 2021-10 | <span style="color:blue">+3.5%</span> | <span style="color:blue">+6.1%</span> | <span style="color:blue">+9.4%</span> | <span style="color:blue">+7.1%</span> | <span style="color:blue">+19.7%</span> | <span style="color:blue">+19.7%</span> | <span style="color:blue">+6.4%</span> |
| 2021-11 | <span style="color:blue">+1.2%</span> | <span style="color:blue">+2.3%</span> | <span style="color:blue">+3.5%</span> | <span style="color:blue">+0.1%</span> | <span style="color:red">-1.8%</span> | <span style="color:red">-1.8%</span> | <span style="color:red">-0.4%</span> |
| 2021-12 | <span style="color:blue">+1.3%</span> | <span style="color:blue">+0.7%</span> | <span style="color:blue">+1.7%</span> | <span style="color:blue">+2.1%</span> | <span style="color:blue">+6.4%</span> | <span style="color:blue">+6.4%</span> | <span style="color:blue">+2.6%</span> |
| 2022-01 | <span style="color:red">-5.1%</span> | <span style="color:red">-9.7%</span> | <span style="color:red">-13.1%</span> | <span style="color:red">-18.1%</span> | <span style="color:red">-28.8%</span> | <span style="color:red">-28.8%</span> | <span style="color:red">-10.1%</span> |
| 2022-02 | <span style="color:blue">+1.1%</span> | <span style="color:blue">+1.3%</span> | <span style="color:red">-1.8%</span> | <span style="color:red">-4.7%</span> | <span style="color:red">-19.0%</span> | <span style="color:red">-14.0%</span> | <span style="color:red">-4.1%</span> |
| 2022-03 | <span style="color:red">-1.3%</span> | <span style="color:red">-1.2%</span> | 0.0% | 0.0% | 0.0% | <span style="color:blue">+12.9%</span> | <span style="color:blue">+5.1%</span> |
| 2022-04 | <span style="color:red">-1.3%</span> | <span style="color:red">-1.0%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-37.2%</span> | <span style="color:red">-13.5%</span> |
| 2022-05 | <span style="color:blue">+0.1%</span> | <span style="color:blue">+0.1%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-14.4%</span> | <span style="color:red">-3.6%</span> |
| 2022-06 | <span style="color:red">-0.6%</span> | <span style="color:red">-0.6%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-25.0%</span> | <span style="color:red">-8.1%</span> |
| 2022-07 | <span style="color:blue">+0.1%</span> | <span style="color:red">-0.0%</span> | 0.0% | 0.0% | 0.0% | <span style="color:blue">+35.6%</span> | <span style="color:blue">+11.3%</span> |
| 2022-08 | <span style="color:red">-2.2%</span> | <span style="color:red">-1.8%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-14.3%</span> | <span style="color:red">-4.5%</span> |
| 2022-09 | <span style="color:red">-1.8%</span> | <span style="color:red">-1.6%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-29.3%</span> | <span style="color:red">-10.3%</span> |
| 2022-10 | <span style="color:red">-1.9%</span> | <span style="color:red">-1.9%</span> | 0.0% | 0.0% | 0.0% | <span style="color:blue">+2.2%</span> | <span style="color:blue">+1.6%</span> |
| 2022-11 | <span style="color:blue">+2.8%</span> | <span style="color:blue">+2.4%</span> | 0.0% | 0.0% | 0.0% | <span style="color:blue">+13.1%</span> | <span style="color:blue">+5.3%</span> |
| 2022-12 | <span style="color:red">-0.5%</span> | <span style="color:red">-0.2%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-25.3%</span> | <span style="color:red">-8.8%</span> |
| 2023-01 | <span style="color:blue">+2.2%</span> | <span style="color:blue">+1.8%</span> | 0.0% | 0.0% | 0.0% | <span style="color:blue">+37.1%</span> | <span style="color:blue">+11.5%</span> |
| 2023-02 | <span style="color:red">-6.7%</span> | <span style="color:red">-10.3%</span> | <span style="color:red">-11.6%</span> | <span style="color:red">-11.6%</span> | <span style="color:red">-11.6%</span> | <span style="color:red">-10.0%</span> | <span style="color:red">-3.1%</span> |
| 2023-03 | <span style="color:blue">+11.3%</span> | <span style="color:blue">+18.2%</span> | <span style="color:blue">+22.4%</span> | <span style="color:blue">+22.4%</span> | <span style="color:blue">+22.4%</span> | <span style="color:blue">+22.4%</span> | <span style="color:blue">+7.4%</span> |
| 2023-04 | <span style="color:blue">+0.2%</span> | <span style="color:blue">+0.3%</span> | <span style="color:blue">+0.3%</span> | <span style="color:red">-1.5%</span> | <span style="color:blue">+0.3%</span> | <span style="color:blue">+0.3%</span> | <span style="color:blue">+0.3%</span> |
| 2023-05 | <span style="color:blue">+4.5%</span> | <span style="color:blue">+8.7%</span> | <span style="color:blue">+12.8%</span> | <span style="color:blue">+11.1%</span> | <span style="color:blue">+17.9%</span> | <span style="color:blue">+17.9%</span> | <span style="color:blue">+5.9%</span> |
| 2023-06 | <span style="color:blue">+5.1%</span> | <span style="color:blue">+11.0%</span> | <span style="color:blue">+15.9%</span> | <span style="color:blue">+13.7%</span> | <span style="color:blue">+15.9%</span> | <span style="color:blue">+15.9%</span> | <span style="color:blue">+5.2%</span> |
| 2023-07 | <span style="color:blue">+4.7%</span> | <span style="color:blue">+8.9%</span> | <span style="color:blue">+11.4%</span> | <span style="color:blue">+11.0%</span> | <span style="color:blue">+11.4%</span> | <span style="color:blue">+11.4%</span> | <span style="color:blue">+3.8%</span> |
| 2023-08 | <span style="color:red">-4.2%</span> | <span style="color:red">-8.7%</span> | <span style="color:red">-10.9%</span> | <span style="color:red">-6.4%</span> | <span style="color:red">-6.0%</span> | <span style="color:red">-6.0%</span> | <span style="color:red">-1.7%</span> |
| 2023-09 | <span style="color:red">-5.0%</span> | <span style="color:red">-6.3%</span> | <span style="color:red">-7.9%</span> | <span style="color:red">-10.4%</span> | <span style="color:red">-16.9%</span> | <span style="color:red">-16.9%</span> | <span style="color:red">-5.8%</span> |
| 2023-10 | <span style="color:red">-1.4%</span> | <span style="color:red">-5.1%</span> | <span style="color:red">-10.2%</span> | <span style="color:red">-12.5%</span> | <span style="color:red">-10.8%</span> | <span style="color:red">-10.8%</span> | <span style="color:red">-3.4%</span> |
| 2023-11 | <span style="color:blue">+6.0%</span> | <span style="color:blue">+8.4%</span> | <span style="color:blue">+11.7%</span> | <span style="color:blue">+11.9%</span> | <span style="color:blue">+28.4%</span> | <span style="color:blue">+28.4%</span> | <span style="color:blue">+8.9%</span> |
| 2023-12 | <span style="color:blue">+5.3%</span> | <span style="color:blue">+9.7%</span> | <span style="color:blue">+13.1%</span> | <span style="color:blue">+12.5%</span> | <span style="color:blue">+15.1%</span> | <span style="color:blue">+15.1%</span> | <span style="color:blue">+4.9%</span> |
| 2024-01 | <span style="color:blue">+2.1%</span> | <span style="color:blue">+4.7%</span> | <span style="color:blue">+6.7%</span> | <span style="color:blue">+6.9%</span> | <span style="color:blue">+7.6%</span> | <span style="color:blue">+7.6%</span> | <span style="color:blue">+2.7%</span> |
| 2024-02 | <span style="color:blue">+3.6%</span> | <span style="color:blue">+8.3%</span> | <span style="color:blue">+12.4%</span> | <span style="color:blue">+13.0%</span> | <span style="color:blue">+14.1%</span> | <span style="color:blue">+14.1%</span> | <span style="color:blue">+4.8%</span> |
| 2024-03 | <span style="color:blue">+1.0%</span> | <span style="color:red">-0.7%</span> | <span style="color:red">-2.0%</span> | <span style="color:red">-2.9%</span> | <span style="color:blue">+1.5%</span> | <span style="color:blue">+1.5%</span> | <span style="color:blue">+0.6%</span> |
| 2024-04 | <span style="color:red">-2.0%</span> | <span style="color:red">-3.3%</span> | <span style="color:red">-5.9%</span> | <span style="color:red">-4.1%</span> | <span style="color:red">-13.8%</span> | <span style="color:red">-13.8%</span> | <span style="color:red">-4.5%</span> |
| 2024-05 | <span style="color:blue">+2.6%</span> | <span style="color:blue">+3.4%</span> | <span style="color:blue">+5.9%</span> | <span style="color:blue">+6.3%</span> | <span style="color:blue">+22.7%</span> | <span style="color:blue">+22.7%</span> | <span style="color:blue">+7.2%</span> |
| 2024-06 | <span style="color:blue">+4.8%</span> | <span style="color:blue">+10.4%</span> | <span style="color:blue">+14.1%</span> | <span style="color:blue">+16.0%</span> | <span style="color:blue">+16.5%</span> | <span style="color:blue">+16.5%</span> | <span style="color:blue">+5.4%</span> |
| 2024-07 | <span style="color:red">-0.3%</span> | <span style="color:red">-2.9%</span> | <span style="color:red">-7.3%</span> | <span style="color:red">-8.2%</span> | <span style="color:red">-6.0%</span> | <span style="color:red">-6.0%</span> | <span style="color:red">-1.6%</span> |
| 2024-08 | <span style="color:blue">+0.5%</span> | <span style="color:blue">+0.8%</span> | <span style="color:red">-1.6%</span> | <span style="color:blue">+1.4%</span> | <span style="color:blue">+7.6%</span> | <span style="color:blue">+7.6%</span> | <span style="color:blue">+3.0%</span> |
| 2024-09 | <span style="color:blue">+3.2%</span> | <span style="color:blue">+4.1%</span> | <span style="color:blue">+3.4%</span> | <span style="color:blue">+8.0%</span> | <span style="color:blue">+18.7%</span> | <span style="color:blue">+18.7%</span> | <span style="color:blue">+6.1%</span> |
| 2024-10 | <span style="color:red">-1.3%</span> | <span style="color:red">-2.9%</span> | <span style="color:red">-2.8%</span> | <span style="color:red">-0.7%</span> | <span style="color:blue">+2.4%</span> | <span style="color:blue">+2.4%</span> | <span style="color:blue">+1.0%</span> |
| 2024-11 | <span style="color:blue">+2.8%</span> | <span style="color:blue">+4.3%</span> | <span style="color:blue">+9.4%</span> | <span style="color:blue">+5.1%</span> | <span style="color:blue">+16.2%</span> | <span style="color:blue">+16.2%</span> | <span style="color:blue">+5.4%</span> |
| 2024-12 | <span style="color:red">-3.0%</span> | <span style="color:red">-4.9%</span> | <span style="color:red">-6.3%</span> | <span style="color:red">-8.4%</span> | <span style="color:red">-2.4%</span> | <span style="color:red">-2.4%</span> | <span style="color:red">-0.5%</span> |
| 2025-01 | <span style="color:blue">+1.6%</span> | <span style="color:red">-1.0%</span> | <span style="color:red">-0.9%</span> | <span style="color:red">-2.2%</span> | <span style="color:blue">+4.3%</span> | <span style="color:blue">+4.3%</span> | <span style="color:blue">+1.8%</span> |
| 2025-02 | <span style="color:red">-0.6%</span> | <span style="color:red">-3.1%</span> | <span style="color:red">-4.9%</span> | <span style="color:red">-4.7%</span> | <span style="color:red">-9.0%</span> | <span style="color:red">-9.0%</span> | <span style="color:red">-2.8%</span> |
| 2025-03 | <span style="color:blue">+2.0%</span> | <span style="color:blue">+2.3%</span> | <span style="color:red">-1.4%</span> | <span style="color:red">-4.2%</span> | <span style="color:red">-17.8%</span> | <span style="color:red">-17.8%</span> | <span style="color:red">-5.7%</span> |
| 2025-04 | <span style="color:blue">+1.3%</span> | <span style="color:blue">+1.7%</span> | <span style="color:red">-1.6%</span> | <span style="color:red">-7.9%</span> | <span style="color:red">-30.3%</span> | <span style="color:red">-7.7%</span> | <span style="color:red">-0.0%</span> |
| 2025-05 | <span style="color:blue">+0.5%</span> | <span style="color:blue">+0.4%</span> | <span style="color:blue">+0.1%</span> | <span style="color:blue">+0.6%</span> | <span style="color:blue">+1.3%</span> | <span style="color:blue">+24.3%</span> | <span style="color:blue">+7.9%</span> |
| 2025-06 | <span style="color:blue">+6.0%</span> | <span style="color:blue">+13.4%</span> | <span style="color:blue">+18.1%</span> | <span style="color:blue">+18.1%</span> | <span style="color:blue">+18.1%</span> | <span style="color:blue">+18.1%</span> | <span style="color:blue">+5.9%</span> |
| 2025-07 | <span style="color:blue">+4.5%</span> | <span style="color:blue">+10.4%</span> | <span style="color:blue">+14.0%</span> | <span style="color:blue">+14.0%</span> | <span style="color:blue">+14.0%</span> | <span style="color:blue">+14.0%</span> | <span style="color:blue">+4.6%</span> |
| 2025-08 | <span style="color:blue">+5.5%</span> | <span style="color:blue">+9.6%</span> | <span style="color:blue">+11.5%</span> | <span style="color:blue">+9.6%</span> | <span style="color:blue">+11.5%</span> | <span style="color:blue">+11.5%</span> | <span style="color:blue">+3.9%</span> |
| 2025-09 | <span style="color:blue">+8.7%</span> | <span style="color:blue">+12.0%</span> | <span style="color:blue">+15.4%</span> | <span style="color:blue">+11.9%</span> | <span style="color:blue">+20.4%</span> | <span style="color:blue">+20.4%</span> | <span style="color:blue">+6.5%</span> |
| 2025-10 | <span style="color:blue">+4.1%</span> | <span style="color:blue">+6.1%</span> | <span style="color:blue">+8.7%</span> | <span style="color:blue">+7.7%</span> | <span style="color:blue">+12.1%</span> | <span style="color:blue">+12.1%</span> | <span style="color:blue">+4.3%</span> |
| 2025-11 | <span style="color:red">-2.7%</span> | <span style="color:red">-8.3%</span> | <span style="color:red">-12.6%</span> | <span style="color:red">-12.1%</span> | <span style="color:red">-6.9%</span> | <span style="color:red">-6.9%</span> | <span style="color:red">-2.0%</span> |
| 2025-12 | <span style="color:red">-0.2%</span> | <span style="color:red">-1.7%</span> | <span style="color:red">-2.6%</span> | <span style="color:red">-1.7%</span> | <span style="color:red">-0.9%</span> | <span style="color:red">-0.9%</span> | <span style="color:red">-0.1%</span> |
| 2026-01 | <span style="color:blue">+3.5%</span> | <span style="color:blue">+2.8%</span> | <span style="color:blue">+2.2%</span> | <span style="color:blue">+2.5%</span> | <span style="color:blue">+2.5%</span> | <span style="color:blue">+2.5%</span> | <span style="color:blue">+1.0%</span> |
| 2026-02 | <span style="color:blue">+1.2%</span> | <span style="color:red">-5.3%</span> | <span style="color:red">-10.6%</span> | <span style="color:red">-11.4%</span> | <span style="color:red">-12.0%</span> | <span style="color:red">-12.0%</span> | <span style="color:red">-3.9%</span> |
| 2026-03 | <span style="color:red">-8.2%</span> | <span style="color:red">-7.2%</span> | <span style="color:red">-4.7%</span> | <span style="color:red">-8.6%</span> | <span style="color:red">-17.4%</span> | <span style="color:red">-17.4%</span> | <span style="color:red">-5.9%</span> |

---

*Generated: 2026-03-31*
