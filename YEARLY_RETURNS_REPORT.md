# 7戦略 年次リターン（1974-2026）と月次リターン

## 検証条件

| 項目 | 値 |
|------|-----|
| データ期間 | 1974-01-02 〜 2026-03-26 |
| 実行遅延 | 2営業日 |
| 経費率 | TQQQ 0.86%, 金ETN 0.5%(推定), TMF 0.91% |
| リバランス閾値 | 20% |

---

## 各戦略の概要

| 戦略 | 概要 |
|------|------|
| **DH Dyn 2x3x** * | A2シグナルでTQQQ/Gold2x(2036)/Bond3x(TMF)を動的配分。CAGR・Sharpe両立の最良戦略 |
| **DH Dyn CAGR25+** * | A2シグナルでTQQQ/Gold1x/Bond1xを動的配分（CAGR 25%+制約版） |
| **A2 Optimized** | DD制御 + AsymEWMA VT + SlopeMult + MomDecel(60/180) + VIX Mean Reversion。単一資産最良 |
| **Ens2(Asym+Slope)** | DD制御 + AsymEWMA(20/5) + SlopeMult(0.7/0.3)。旧推奨戦略 |
| **DD Only** | 200日高値から-18%でCASH退避、92%回復でHOLD。最もシンプルな管理戦略 |
| **BH 3x** | NASDAQ 3倍レバレッジ（TQQQ相当）を無管理で保有 |
| **BH 1x** | レバレッジなしのNASDAQ指数をそのまま保有。ベンチマーク |

> \* 3資産ポートフォリオ。DH Dyn 2x3x: TQQQ + Gold2x(2036) + Bond3x(TMF)。DH Dyn CAGR25+: TQQQ + Gold1x(447A) + Bond1x(2621)

---

## 統計サマリー

| 統計量 | DH Dyn 2x3x | DH Dyn 25+ | A2 Opt | Ens2 | DD Only | BH 3x | BH 1x |
|--------|-------------|------------|--------|------|---------|-------|-------|
| CAGR | +30.67% | +25.23% | +29.19% | +22.20% | +25.58% | +19.21% | +10.98% |
| 中央値 | +31.8% | +27.2% | +26.2% | +19.0% | +24.0% | +44.4% | +15.5% |
| 最大 | +144.3% | +135.9% | +221.4% | +147.2% | +405.2% | +405.2% | +85.6% |
| 最小 | -16.6% | -12.7% | -25.1% | -42.8% | -46.4% | -89.2% | -40.5% |
| 標準偏差 | +34.5% | +29.4% | +44.5% | +40.0% | +75.2% | +90.6% | +25.3% |
| プラス年数 | 44 | 44 | 37 | 38 | 34 | 38 | 39 |
| マイナス年数 | 9 | 9 | 16 | 15 | 19 | 15 | 14 |

---

## 年次リターン表（%）

| Year | DH Dyn 2x3x | DH Dyn 25+ | A2 Opt | Ens2 | DD Only | BH 3x | BH 1x |
|------|-------------|------------|--------|------|---------|-------|-------|
| 1974 | <span style="color:blue">+24.3%</span> | <span style="color:blue">+11.1%</span> | 0.0% | <span style="color:red">-42.8%</span> | <span style="color:red">-42.8%</span> | <span style="color:red">-75.8%</span> | <span style="color:red">-35.4%</span> |
| 1975 | <span style="color:blue">+0.9%</span> | <span style="color:blue">+5.2%</span> | <span style="color:blue">+14.4%</span> | <span style="color:blue">+15.0%</span> | <span style="color:blue">+3.8%</span> | <span style="color:blue">+105.5%</span> | <span style="color:blue">+29.8%</span> |
| 1976 | <span style="color:blue">+60.5%</span> | <span style="color:blue">+50.8%</span> | <span style="color:blue">+68.3%</span> | <span style="color:blue">+35.2%</span> | <span style="color:blue">+94.5%</span> | <span style="color:blue">+94.5%</span> | <span style="color:blue">+26.1%</span> |
| 1977 | <span style="color:blue">+11.2%</span> | <span style="color:blue">+6.9%</span> | <span style="color:blue">+2.6%</span> | <span style="color:blue">+6.4%</span> | <span style="color:blue">+21.0%</span> | <span style="color:blue">+21.0%</span> | <span style="color:blue">+7.3%</span> |
| 1978 | <span style="color:blue">+74.3%</span> | <span style="color:blue">+64.4%</span> | <span style="color:blue">+61.1%</span> | <span style="color:blue">+48.2%</span> | <span style="color:blue">+14.2%</span> | <span style="color:blue">+34.8%</span> | <span style="color:blue">+12.3%</span> |
| 1979 | <span style="color:blue">+78.3%</span> | <span style="color:blue">+46.2%</span> | <span style="color:blue">+20.7%</span> | <span style="color:blue">+14.1%</span> | <span style="color:blue">+54.6%</span> | <span style="color:blue">+101.1%</span> | <span style="color:blue">+28.1%</span> |
| 1980 | <span style="color:blue">+68.2%</span> | <span style="color:blue">+56.5%</span> | <span style="color:blue">+77.2%</span> | <span style="color:blue">+62.5%</span> | <span style="color:blue">+48.1%</span> | <span style="color:blue">+119.8%</span> | <span style="color:blue">+33.9%</span> |
| 1981 | <span style="color:red">-13.8%</span> | <span style="color:red">-9.6%</span> | <span style="color:red">-7.3%</span> | <span style="color:red">-18.8%</span> | <span style="color:red">-31.9%</span> | <span style="color:red">-14.0%</span> | <span style="color:red">-3.2%</span> |
| 1982 | <span style="color:blue">+101.3%</span> | <span style="color:blue">+65.5%</span> | <span style="color:blue">+82.4%</span> | <span style="color:blue">+77.3%</span> | <span style="color:blue">+87.8%</span> | <span style="color:blue">+57.4%</span> | <span style="color:blue">+18.7%</span> |
| 1983 | <span style="color:blue">+38.7%</span> | <span style="color:blue">+41.5%</span> | <span style="color:blue">+72.5%</span> | <span style="color:blue">+76.7%</span> | <span style="color:blue">+51.1%</span> | <span style="color:blue">+63.6%</span> | <span style="color:blue">+19.9%</span> |
| 1984 | <span style="color:red">-3.3%</span> | <span style="color:red">-2.3%</span> | <span style="color:red">-4.7%</span> | <span style="color:red">-6.8%</span> | <span style="color:red">-4.7%</span> | <span style="color:red">-34.3%</span> | <span style="color:red">-11.3%</span> |
| 1985 | <span style="color:blue">+85.4%</span> | <span style="color:blue">+66.5%</span> | <span style="color:blue">+82.4%</span> | <span style="color:blue">+69.4%</span> | <span style="color:blue">+121.0%</span> | <span style="color:blue">+121.0%</span> | <span style="color:blue">+31.5%</span> |
| 1986 | <span style="color:blue">+50.5%</span> | <span style="color:blue">+34.9%</span> | <span style="color:blue">+28.9%</span> | <span style="color:blue">+34.5%</span> | <span style="color:blue">+18.8%</span> | <span style="color:blue">+18.8%</span> | <span style="color:blue">+7.4%</span> |
| 1987 | <span style="color:blue">+31.8%</span> | <span style="color:blue">+22.3%</span> | <span style="color:blue">+13.2%</span> | <span style="color:blue">+4.1%</span> | <span style="color:red">-26.9%</span> | <span style="color:red">-31.9%</span> | <span style="color:red">-5.2%</span> |
| 1988 | <span style="color:red">-7.9%</span> | <span style="color:red">-5.5%</span> | <span style="color:red">-4.0%</span> | <span style="color:red">-3.3%</span> | <span style="color:red">-5.9%</span> | <span style="color:blue">+47.8%</span> | <span style="color:blue">+15.4%</span> |
| 1989 | <span style="color:blue">+48.8%</span> | <span style="color:blue">+39.4%</span> | <span style="color:blue">+43.2%</span> | <span style="color:blue">+42.6%</span> | <span style="color:blue">+64.2%</span> | <span style="color:blue">+64.2%</span> | <span style="color:blue">+19.2%</span> |
| 1990 | <span style="color:red">-5.8%</span> | <span style="color:red">-4.8%</span> | <span style="color:red">-12.2%</span> | <span style="color:red">-21.7%</span> | <span style="color:red">-46.4%</span> | <span style="color:red">-49.1%</span> | <span style="color:red">-17.8%</span> |
| 1991 | <span style="color:blue">+62.4%</span> | <span style="color:blue">+55.0%</span> | <span style="color:blue">+86.5%</span> | <span style="color:blue">+79.9%</span> | <span style="color:blue">+123.5%</span> | <span style="color:blue">+257.4%</span> | <span style="color:blue">+56.9%</span> |
| 1992 | <span style="color:blue">+32.6%</span> | <span style="color:blue">+29.7%</span> | <span style="color:blue">+45.6%</span> | <span style="color:blue">+22.5%</span> | <span style="color:blue">+45.0%</span> | <span style="color:blue">+45.0%</span> | <span style="color:blue">+15.5%</span> |
| 1993 | <span style="color:blue">+14.8%</span> | <span style="color:blue">+7.5%</span> | <span style="color:blue">+1.6%</span> | <span style="color:blue">+7.5%</span> | <span style="color:blue">+43.8%</span> | <span style="color:blue">+43.8%</span> | <span style="color:blue">+14.7%</span> |
| 1994 | <span style="color:red">-2.3%</span> | <span style="color:red">-0.7%</span> | <span style="color:red">-0.9%</span> | <span style="color:red">-1.0%</span> | <span style="color:red">-13.6%</span> | <span style="color:red">-13.6%</span> | <span style="color:red">-3.2%</span> |
| 1995 | <span style="color:blue">+88.4%</span> | <span style="color:blue">+72.3%</span> | <span style="color:blue">+112.0%</span> | <span style="color:blue">+88.6%</span> | <span style="color:blue">+157.1%</span> | <span style="color:blue">+157.1%</span> | <span style="color:blue">+39.9%</span> |
| 1996 | <span style="color:blue">+28.0%</span> | <span style="color:blue">+27.2%</span> | <span style="color:blue">+43.4%</span> | <span style="color:blue">+34.9%</span> | <span style="color:blue">+70.3%</span> | <span style="color:blue">+70.3%</span> | <span style="color:blue">+22.7%</span> |
| 1997 | <span style="color:blue">+62.5%</span> | <span style="color:blue">+61.7%</span> | <span style="color:blue">+96.3%</span> | <span style="color:blue">+95.4%</span> | <span style="color:blue">+60.4%</span> | <span style="color:blue">+60.4%</span> | <span style="color:blue">+21.6%</span> |
| 1998 | <span style="color:blue">+82.0%</span> | <span style="color:blue">+69.2%</span> | <span style="color:blue">+85.3%</span> | <span style="color:blue">+71.5%</span> | <span style="color:blue">+24.0%</span> | <span style="color:blue">+117.4%</span> | <span style="color:blue">+39.6%</span> |
| 1999 | <span style="color:blue">+144.3%</span> | <span style="color:blue">+135.9%</span> | <span style="color:blue">+221.4%</span> | <span style="color:blue">+147.2%</span> | <span style="color:blue">+405.2%</span> | <span style="color:blue">+405.2%</span> | <span style="color:blue">+85.6%</span> |
| 2000 | <span style="color:blue">+2.7%</span> | <span style="color:blue">+0.3%</span> | <span style="color:red">-4.5%</span> | <span style="color:red">-12.9%</span> | <span style="color:red">-33.8%</span> | <span style="color:red">-89.2%</span> | <span style="color:red">-39.3%</span> |
| 2001 | <span style="color:blue">+3.6%</span> | <span style="color:blue">+1.5%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-71.9%</span> | <span style="color:red">-21.1%</span> |
| 2002 | <span style="color:blue">+22.1%</span> | <span style="color:blue">+10.3%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-77.6%</span> | <span style="color:red">-31.5%</span> |
| 2003 | <span style="color:blue">+73.8%</span> | <span style="color:blue">+66.8%</span> | <span style="color:blue">+99.3%</span> | <span style="color:blue">+127.9%</span> | <span style="color:blue">+175.5%</span> | <span style="color:blue">+188.5%</span> | <span style="color:blue">+50.0%</span> |
| 2004 | <span style="color:blue">+14.0%</span> | <span style="color:blue">+10.8%</span> | <span style="color:blue">+6.8%</span> | <span style="color:blue">+5.9%</span> | <span style="color:red">-20.1%</span> | <span style="color:blue">+16.5%</span> | <span style="color:blue">+8.6%</span> |
| 2005 | <span style="color:blue">+5.7%</span> | <span style="color:blue">+3.0%</span> | <span style="color:red">-1.2%</span> | <span style="color:blue">+4.8%</span> | <span style="color:red">-1.4%</span> | <span style="color:red">-1.4%</span> | <span style="color:blue">+1.4%</span> |
| 2006 | <span style="color:blue">+48.2%</span> | <span style="color:blue">+40.8%</span> | <span style="color:blue">+52.8%</span> | <span style="color:blue">+19.0%</span> | <span style="color:blue">+22.7%</span> | <span style="color:blue">+22.7%</span> | <span style="color:blue">+9.5%</span> |
| 2007 | <span style="color:blue">+26.2%</span> | <span style="color:blue">+18.0%</span> | <span style="color:blue">+13.4%</span> | <span style="color:blue">+9.9%</span> | <span style="color:blue">+19.9%</span> | <span style="color:blue">+19.9%</span> | <span style="color:blue">+9.8%</span> |
| 2008 | <span style="color:blue">+8.4%</span> | <span style="color:blue">+2.7%</span> | <span style="color:red">-7.9%</span> | <span style="color:red">-15.4%</span> | <span style="color:red">-36.4%</span> | <span style="color:red">-87.6%</span> | <span style="color:red">-40.5%</span> |
| 2009 | <span style="color:blue">+47.5%</span> | <span style="color:blue">+42.7%</span> | <span style="color:blue">+54.3%</span> | <span style="color:blue">+48.7%</span> | <span style="color:blue">+59.2%</span> | <span style="color:blue">+132.5%</span> | <span style="color:blue">+43.9%</span> |
| 2010 | <span style="color:blue">+45.8%</span> | <span style="color:blue">+35.4%</span> | <span style="color:blue">+26.2%</span> | <span style="color:blue">+43.2%</span> | <span style="color:blue">+40.9%</span> | <span style="color:blue">+40.9%</span> | <span style="color:blue">+16.9%</span> |
| 2011 | <span style="color:red">-6.2%</span> | <span style="color:red">-12.7%</span> | <span style="color:red">-25.1%</span> | <span style="color:red">-28.4%</span> | <span style="color:red">-41.5%</span> | <span style="color:red">-22.7%</span> | <span style="color:red">-1.8%</span> |
| 2012 | <span style="color:blue">+35.1%</span> | <span style="color:blue">+32.0%</span> | <span style="color:blue">+41.3%</span> | <span style="color:blue">+27.4%</span> | <span style="color:blue">+44.4%</span> | <span style="color:blue">+44.4%</span> | <span style="color:blue">+15.9%</span> |
| 2013 | <span style="color:blue">+27.0%</span> | <span style="color:blue">+33.2%</span> | <span style="color:blue">+76.7%</span> | <span style="color:blue">+84.7%</span> | <span style="color:blue">+150.6%</span> | <span style="color:blue">+150.6%</span> | <span style="color:blue">+38.3%</span> |
| 2014 | <span style="color:blue">+4.7%</span> | <span style="color:blue">+4.0%</span> | <span style="color:blue">+7.9%</span> | <span style="color:blue">+7.7%</span> | <span style="color:blue">+36.1%</span> | <span style="color:blue">+36.1%</span> | <span style="color:blue">+13.4%</span> |
| 2015 | <span style="color:red">-14.1%</span> | <span style="color:red">-11.5%</span> | <span style="color:red">-13.7%</span> | <span style="color:blue">+3.1%</span> | <span style="color:blue">+7.6%</span> | <span style="color:blue">+7.6%</span> | <span style="color:blue">+5.7%</span> |
| 2016 | <span style="color:blue">+7.1%</span> | <span style="color:blue">+2.2%</span> | <span style="color:red">-5.2%</span> | <span style="color:red">-3.5%</span> | <span style="color:red">-19.1%</span> | <span style="color:blue">+14.1%</span> | <span style="color:blue">+7.5%</span> |
| 2017 | <span style="color:blue">+37.4%</span> | <span style="color:blue">+33.3%</span> | <span style="color:blue">+49.5%</span> | <span style="color:blue">+39.0%</span> | <span style="color:blue">+103.3%</span> | <span style="color:blue">+103.3%</span> | <span style="color:blue">+28.2%</span> |
| 2018 | <span style="color:blue">+12.5%</span> | <span style="color:blue">+10.1%</span> | <span style="color:blue">+6.0%</span> | <span style="color:red">-10.3%</span> | <span style="color:red">-25.5%</span> | <span style="color:red">-22.9%</span> | <span style="color:red">-3.9%</span> |
| 2019 | <span style="color:blue">+47.1%</span> | <span style="color:blue">+38.2%</span> | <span style="color:blue">+47.6%</span> | <span style="color:blue">+33.6%</span> | <span style="color:blue">+61.2%</span> | <span style="color:blue">+127.5%</span> | <span style="color:blue">+35.2%</span> |
| 2020 | <span style="color:blue">+72.3%</span> | <span style="color:blue">+62.0%</span> | <span style="color:blue">+68.5%</span> | <span style="color:blue">+68.1%</span> | <span style="color:blue">+85.8%</span> | <span style="color:blue">+96.0%</span> | <span style="color:blue">+43.6%</span> |
| 2021 | <span style="color:blue">+27.3%</span> | <span style="color:blue">+25.7%</span> | <span style="color:blue">+40.4%</span> | <span style="color:blue">+22.5%</span> | <span style="color:blue">+60.8%</span> | <span style="color:blue">+60.8%</span> | <span style="color:blue">+21.4%</span> |
| 2022 | <span style="color:red">-16.6%</span> | <span style="color:red">-10.9%</span> | <span style="color:red">-11.8%</span> | <span style="color:red">-19.4%</span> | <span style="color:red">-38.9%</span> | <span style="color:red">-78.2%</span> | <span style="color:red">-33.1%</span> |
| 2023 | <span style="color:blue">+47.0%</span> | <span style="color:blue">+40.3%</span> | <span style="color:blue">+51.4%</span> | <span style="color:blue">+40.4%</span> | <span style="color:blue">+84.9%</span> | <span style="color:blue">+167.3%</span> | <span style="color:blue">+43.4%</span> |
| 2024 | <span style="color:blue">+35.7%</span> | <span style="color:blue">+25.8%</span> | <span style="color:blue">+26.1%</span> | <span style="color:blue">+30.3%</span> | <span style="color:blue">+91.1%</span> | <span style="color:blue">+91.1%</span> | <span style="color:blue">+28.6%</span> |
| 2025 | <span style="color:blue">+58.9%</span> | <span style="color:blue">+37.1%</span> | <span style="color:blue">+29.0%</span> | <span style="color:blue">+13.2%</span> | <span style="color:red">-14.2%</span> | <span style="color:blue">+45.7%</span> | <span style="color:blue">+20.4%</span> |
| 2026 | <span style="color:red">-11.0%</span> | <span style="color:red">-8.5%</span> | <span style="color:red">-11.2%</span> | <span style="color:red">-15.2%</span> | <span style="color:red">-23.6%</span> | <span style="color:red">-23.6%</span> | <span style="color:red">-7.9%</span> |

---

## 月次リターン表（2021-2026, OOS期間）（%）

| Year-Month | DH Dyn 2x3x | DH Dyn 25+ | A2 Opt | Ens2 | DD Only | BH 3x | BH 1x |
|------------|-------------|------------|--------|------|---------|-------|-------|
| 2021-01 | <span style="color:blue">+1.4%</span> | <span style="color:blue">+1.7%</span> | <span style="color:blue">+2.9%</span> | <span style="color:blue">+3.2%</span> | <span style="color:blue">+3.2%</span> | <span style="color:blue">+3.2%</span> | <span style="color:blue">+1.4%</span> |
| 2021-02 | <span style="color:red">-1.1%</span> | <span style="color:blue">+0.1%</span> | <span style="color:blue">+1.6%</span> | <span style="color:blue">+1.5%</span> | <span style="color:blue">+1.7%</span> | <span style="color:blue">+1.7%</span> | <span style="color:blue">+0.9%</span> |
| 2021-03 | <span style="color:red">-1.7%</span> | <span style="color:red">-1.0%</span> | <span style="color:red">-0.4%</span> | <span style="color:blue">+0.3%</span> | <span style="color:red">-1.1%</span> | <span style="color:red">-1.1%</span> | <span style="color:blue">+0.4%</span> |
| 2021-04 | <span style="color:blue">+5.4%</span> | <span style="color:blue">+4.3%</span> | <span style="color:blue">+5.9%</span> | <span style="color:blue">+3.5%</span> | <span style="color:blue">+16.4%</span> | <span style="color:blue">+16.4%</span> | <span style="color:blue">+5.4%</span> |
| 2021-05 | <span style="color:red">-0.6%</span> | <span style="color:red">-2.1%</span> | <span style="color:red">-5.9%</span> | <span style="color:red">-5.4%</span> | <span style="color:red">-5.5%</span> | <span style="color:red">-5.5%</span> | <span style="color:red">-1.5%</span> |
| 2021-06 | <span style="color:blue">+4.5%</span> | <span style="color:blue">+5.1%</span> | <span style="color:blue">+9.8%</span> | <span style="color:blue">+6.4%</span> | <span style="color:blue">+17.0%</span> | <span style="color:blue">+17.0%</span> | <span style="color:blue">+5.5%</span> |
| 2021-07 | <span style="color:blue">+3.3%</span> | <span style="color:blue">+2.7%</span> | <span style="color:blue">+3.0%</span> | <span style="color:blue">+2.7%</span> | <span style="color:blue">+3.1%</span> | <span style="color:blue">+3.1%</span> | <span style="color:blue">+1.2%</span> |
| 2021-08 | <span style="color:blue">+8.9%</span> | <span style="color:blue">+8.5%</span> | <span style="color:blue">+11.4%</span> | <span style="color:blue">+8.1%</span> | <span style="color:blue">+12.1%</span> | <span style="color:blue">+12.1%</span> | <span style="color:blue">+4.0%</span> |
| 2021-09 | <span style="color:red">-7.0%</span> | <span style="color:red">-6.0%</span> | <span style="color:red">-7.9%</span> | <span style="color:red">-7.7%</span> | <span style="color:red">-15.7%</span> | <span style="color:red">-15.7%</span> | <span style="color:red">-5.3%</span> |
| 2021-10 | <span style="color:blue">+7.4%</span> | <span style="color:blue">+6.7%</span> | <span style="color:blue">+10.2%</span> | <span style="color:blue">+8.1%</span> | <span style="color:blue">+22.7%</span> | <span style="color:blue">+22.7%</span> | <span style="color:blue">+7.3%</span> |
| 2021-11 | <span style="color:blue">+3.9%</span> | <span style="color:blue">+3.5%</span> | <span style="color:blue">+5.0%</span> | <span style="color:blue">+0.8%</span> | <span style="color:blue">+0.1%</span> | <span style="color:blue">+0.1%</span> | <span style="color:blue">+0.3%</span> |
| 2021-12 | <span style="color:blue">+1.0%</span> | <span style="color:blue">+0.5%</span> | <span style="color:blue">+0.8%</span> | <span style="color:blue">+0.0%</span> | <span style="color:blue">+0.6%</span> | <span style="color:blue">+0.6%</span> | <span style="color:blue">+0.7%</span> |
| 2022-01 | <span style="color:red">-9.4%</span> | <span style="color:red">-8.0%</span> | <span style="color:red">-10.4%</span> | <span style="color:red">-15.9%</span> | <span style="color:red">-26.2%</span> | <span style="color:red">-26.2%</span> | <span style="color:red">-9.0%</span> |
| 2022-02 | <span style="color:blue">+3.4%</span> | <span style="color:blue">+1.4%</span> | <span style="color:red">-1.6%</span> | <span style="color:red">-4.2%</span> | <span style="color:red">-17.2%</span> | <span style="color:red">-12.0%</span> | <span style="color:red">-3.4%</span> |
| 2022-03 | <span style="color:red">-1.1%</span> | <span style="color:red">-0.1%</span> | 0.0% | 0.0% | 0.0% | <span style="color:blue">+7.5%</span> | <span style="color:blue">+3.4%</span> |
| 2022-04 | <span style="color:red">-3.5%</span> | <span style="color:red">-1.6%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-36.6%</span> | <span style="color:red">-13.3%</span> |
| 2022-05 | <span style="color:red">-2.3%</span> | <span style="color:red">-1.0%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-10.2%</span> | <span style="color:red">-2.1%</span> |
| 2022-06 | <span style="color:red">-1.6%</span> | <span style="color:red">-0.8%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-26.7%</span> | <span style="color:red">-8.7%</span> |
| 2022-07 | <span style="color:blue">+0.4%</span> | <span style="color:blue">+0.1%</span> | 0.0% | 0.0% | 0.0% | <span style="color:blue">+39.2%</span> | <span style="color:blue">+12.3%</span> |
| 2022-08 | <span style="color:red">-3.5%</span> | <span style="color:red">-1.6%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-14.8%</span> | <span style="color:red">-4.6%</span> |
| 2022-09 | <span style="color:red">-4.6%</span> | <span style="color:red">-2.1%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-29.9%</span> | <span style="color:red">-10.5%</span> |
| 2022-10 | <span style="color:red">-2.7%</span> | <span style="color:red">-1.2%</span> | 0.0% | 0.0% | 0.0% | <span style="color:blue">+9.2%</span> | <span style="color:blue">+3.9%</span> |
| 2022-11 | <span style="color:blue">+5.6%</span> | <span style="color:blue">+2.6%</span> | 0.0% | 0.0% | 0.0% | <span style="color:blue">+10.1%</span> | <span style="color:blue">+4.4%</span> |
| 2022-12 | <span style="color:blue">+2.1%</span> | <span style="color:blue">+1.1%</span> | 0.0% | 0.0% | 0.0% | <span style="color:red">-25.1%</span> | <span style="color:red">-8.7%</span> |
| 2023-01 | <span style="color:blue">+4.8%</span> | <span style="color:blue">+2.3%</span> | 0.0% | 0.0% | 0.0% | <span style="color:blue">+34.0%</span> | <span style="color:blue">+10.7%</span> |
| 2023-02 | <span style="color:red">-12.2%</span> | <span style="color:red">-10.1%</span> | <span style="color:red">-11.6%</span> | <span style="color:red">-11.6%</span> | <span style="color:red">-11.6%</span> | <span style="color:red">-4.6%</span> | <span style="color:red">-1.1%</span> |
| 2023-03 | <span style="color:blue">+18.9%</span> | <span style="color:blue">+16.4%</span> | <span style="color:blue">+20.0%</span> | <span style="color:blue">+20.0%</span> | <span style="color:blue">+20.0%</span> | <span style="color:blue">+20.0%</span> | <span style="color:blue">+6.7%</span> |
| 2023-04 | <span style="color:red">-0.1%</span> | <span style="color:red">-0.2%</span> | <span style="color:red">-0.5%</span> | <span style="color:red">-2.2%</span> | <span style="color:red">-0.5%</span> | <span style="color:red">-0.5%</span> | <span style="color:blue">+0.0%</span> |
| 2023-05 | <span style="color:blue">+8.5%</span> | <span style="color:blue">+8.3%</span> | <span style="color:blue">+12.4%</span> | <span style="color:blue">+10.7%</span> | <span style="color:blue">+17.5%</span> | <span style="color:blue">+17.5%</span> | <span style="color:blue">+5.8%</span> |
| 2023-06 | <span style="color:blue">+14.8%</span> | <span style="color:blue">+14.2%</span> | <span style="color:blue">+20.3%</span> | <span style="color:blue">+16.0%</span> | <span style="color:blue">+20.3%</span> | <span style="color:blue">+20.3%</span> | <span style="color:blue">+6.6%</span> |
| 2023-07 | <span style="color:blue">+10.1%</span> | <span style="color:blue">+9.4%</span> | <span style="color:blue">+12.1%</span> | <span style="color:blue">+11.7%</span> | <span style="color:blue">+12.1%</span> | <span style="color:blue">+12.1%</span> | <span style="color:blue">+4.0%</span> |
| 2023-08 | <span style="color:red">-10.5%</span> | <span style="color:red">-9.8%</span> | <span style="color:red">-12.0%</span> | <span style="color:red">-7.5%</span> | <span style="color:red">-7.2%</span> | <span style="color:red">-7.2%</span> | <span style="color:red">-2.2%</span> |
| 2023-09 | <span style="color:red">-8.4%</span> | <span style="color:red">-6.4%</span> | <span style="color:red">-7.9%</span> | <span style="color:red">-10.5%</span> | <span style="color:red">-16.9%</span> | <span style="color:red">-16.9%</span> | <span style="color:red">-5.8%</span> |
| 2023-10 | <span style="color:red">-4.6%</span> | <span style="color:red">-5.1%</span> | <span style="color:red">-9.7%</span> | <span style="color:red">-11.8%</span> | <span style="color:red">-9.0%</span> | <span style="color:red">-9.0%</span> | <span style="color:red">-2.8%</span> |
| 2023-11 | <span style="color:blue">+11.2%</span> | <span style="color:blue">+8.9%</span> | <span style="color:blue">+12.6%</span> | <span style="color:blue">+13.5%</span> | <span style="color:blue">+34.7%</span> | <span style="color:blue">+34.7%</span> | <span style="color:blue">+10.7%</span> |
| 2023-12 | <span style="color:blue">+12.7%</span> | <span style="color:blue">+11.3%</span> | <span style="color:blue">+14.8%</span> | <span style="color:blue">+13.9%</span> | <span style="color:blue">+17.0%</span> | <span style="color:blue">+17.0%</span> | <span style="color:blue">+5.5%</span> |
| 2024-01 | <span style="color:blue">+1.3%</span> | <span style="color:blue">+1.3%</span> | <span style="color:blue">+2.1%</span> | <span style="color:blue">+2.4%</span> | <span style="color:blue">+2.3%</span> | <span style="color:blue">+2.3%</span> | <span style="color:blue">+1.0%</span> |
| 2024-02 | <span style="color:blue">+11.4%</span> | <span style="color:blue">+10.7%</span> | <span style="color:blue">+16.2%</span> | <span style="color:blue">+17.0%</span> | <span style="color:blue">+18.6%</span> | <span style="color:blue">+18.6%</span> | <span style="color:blue">+6.1%</span> |
| 2024-03 | <span style="color:blue">+3.4%</span> | <span style="color:blue">+1.9%</span> | <span style="color:blue">+0.9%</span> | <span style="color:blue">+0.1%</span> | <span style="color:blue">+4.9%</span> | <span style="color:blue">+4.9%</span> | <span style="color:blue">+1.8%</span> |
| 2024-04 | <span style="color:red">-3.9%</span> | <span style="color:red">-3.2%</span> | <span style="color:red">-5.8%</span> | <span style="color:red">-4.0%</span> | <span style="color:red">-13.6%</span> | <span style="color:red">-13.6%</span> | <span style="color:red">-4.4%</span> |
| 2024-05 | <span style="color:blue">+5.6%</span> | <span style="color:blue">+3.7%</span> | <span style="color:blue">+5.7%</span> | <span style="color:blue">+5.9%</span> | <span style="color:blue">+21.5%</span> | <span style="color:blue">+21.5%</span> | <span style="color:blue">+6.9%</span> |
| 2024-06 | <span style="color:blue">+12.6%</span> | <span style="color:blue">+11.6%</span> | <span style="color:blue">+15.4%</span> | <span style="color:blue">+17.9%</span> | <span style="color:blue">+18.5%</span> | <span style="color:blue">+18.5%</span> | <span style="color:blue">+6.0%</span> |
| 2024-07 | <span style="color:red">-0.9%</span> | <span style="color:red">-1.3%</span> | <span style="color:red">-5.0%</span> | <span style="color:red">-6.0%</span> | <span style="color:red">-3.6%</span> | <span style="color:red">-3.6%</span> | <span style="color:red">-0.8%</span> |
| 2024-08 | <span style="color:blue">+2.8%</span> | <span style="color:blue">+0.5%</span> | <span style="color:red">-4.0%</span> | <span style="color:red">-2.6%</span> | <span style="color:blue">+0.2%</span> | <span style="color:blue">+0.2%</span> | <span style="color:blue">+0.6%</span> |
| 2024-09 | <span style="color:blue">+2.2%</span> | <span style="color:blue">+0.2%</span> | <span style="color:red">-2.9%</span> | <span style="color:blue">+3.6%</span> | <span style="color:blue">+7.1%</span> | <span style="color:blue">+7.1%</span> | <span style="color:blue">+2.7%</span> |
| 2024-10 | <span style="color:red">-4.0%</span> | <span style="color:red">-3.5%</span> | <span style="color:red">-4.3%</span> | <span style="color:red">-3.5%</span> | <span style="color:red">-2.3%</span> | <span style="color:red">-2.3%</span> | <span style="color:red">-0.5%</span> |
| 2024-11 | <span style="color:blue">+5.9%</span> | <span style="color:blue">+5.6%</span> | <span style="color:blue">+12.0%</span> | <span style="color:blue">+6.5%</span> | <span style="color:blue">+18.9%</span> | <span style="color:blue">+18.9%</span> | <span style="color:blue">+6.2%</span> |
| 2024-12 | <span style="color:red">-3.9%</span> | <span style="color:red">-3.0%</span> | <span style="color:red">-3.5%</span> | <span style="color:red">-7.1%</span> | <span style="color:blue">+0.4%</span> | <span style="color:blue">+0.4%</span> | <span style="color:blue">+0.5%</span> |
| 2025-01 | <span style="color:blue">+1.1%</span> | <span style="color:red">-0.9%</span> | <span style="color:red">-1.1%</span> | <span style="color:red">-2.5%</span> | <span style="color:blue">+3.8%</span> | <span style="color:blue">+3.8%</span> | <span style="color:blue">+1.6%</span> |
| 2025-02 | <span style="color:red">-4.9%</span> | <span style="color:red">-4.7%</span> | <span style="color:red">-7.5%</span> | <span style="color:red">-6.6%</span> | <span style="color:red">-12.3%</span> | <span style="color:red">-12.3%</span> | <span style="color:red">-4.0%</span> |
| 2025-03 | <span style="color:blue">+6.0%</span> | <span style="color:blue">+2.2%</span> | <span style="color:red">-3.4%</span> | <span style="color:red">-6.1%</span> | <span style="color:red">-24.3%</span> | <span style="color:red">-24.3%</span> | <span style="color:red">-8.2%</span> |
| 2025-04 | <span style="color:blue">+5.2%</span> | <span style="color:blue">+1.9%</span> | <span style="color:red">-1.5%</span> | <span style="color:red">-7.3%</span> | <span style="color:red">-28.5%</span> | <span style="color:red">-5.3%</span> | <span style="color:blue">+0.9%</span> |
| 2025-05 | <span style="color:red">-1.3%</span> | <span style="color:red">-0.6%</span> | <span style="color:blue">+0.1%</span> | <span style="color:blue">+0.6%</span> | <span style="color:blue">+1.3%</span> | <span style="color:blue">+30.0%</span> | <span style="color:blue">+9.6%</span> |
| 2025-06 | <span style="color:blue">+16.7%</span> | <span style="color:blue">+15.5%</span> | <span style="color:blue">+20.5%</span> | <span style="color:blue">+20.5%</span> | <span style="color:blue">+20.5%</span> | <span style="color:blue">+20.5%</span> | <span style="color:blue">+6.6%</span> |
| 2025-07 | <span style="color:blue">+8.9%</span> | <span style="color:blue">+8.5%</span> | <span style="color:blue">+11.2%</span> | <span style="color:blue">+11.2%</span> | <span style="color:blue">+11.2%</span> | <span style="color:blue">+11.2%</span> | <span style="color:blue">+3.7%</span> |
| 2025-08 | <span style="color:blue">+4.7%</span> | <span style="color:blue">+4.2%</span> | <span style="color:blue">+4.0%</span> | <span style="color:blue">+2.2%</span> | <span style="color:blue">+4.0%</span> | <span style="color:blue">+4.0%</span> | <span style="color:blue">+1.6%</span> |
| 2025-09 | <span style="color:blue">+13.0%</span> | <span style="color:blue">+10.6%</span> | <span style="color:blue">+12.5%</span> | <span style="color:blue">+10.2%</span> | <span style="color:blue">+17.4%</span> | <span style="color:blue">+17.4%</span> | <span style="color:blue">+5.6%</span> |
| 2025-10 | <span style="color:blue">+9.0%</span> | <span style="color:blue">+7.1%</span> | <span style="color:blue">+9.9%</span> | <span style="color:blue">+8.4%</span> | <span style="color:blue">+13.5%</span> | <span style="color:blue">+13.5%</span> | <span style="color:blue">+4.7%</span> |
| 2025-11 | <span style="color:red">-6.9%</span> | <span style="color:red">-7.4%</span> | <span style="color:red">-11.5%</span> | <span style="color:red">-11.1%</span> | <span style="color:red">-5.6%</span> | <span style="color:red">-5.6%</span> | <span style="color:red">-1.5%</span> |
| 2025-12 | <span style="color:red">-1.8%</span> | <span style="color:red">-2.0%</span> | <span style="color:red">-3.0%</span> | <span style="color:red">-2.5%</span> | <span style="color:red">-2.0%</span> | <span style="color:red">-2.0%</span> | <span style="color:red">-0.5%</span> |
| 2026-01 | <span style="color:blue">+3.6%</span> | <span style="color:blue">+2.7%</span> | <span style="color:blue">+2.1%</span> | <span style="color:blue">+2.4%</span> | <span style="color:blue">+2.4%</span> | <span style="color:blue">+2.4%</span> | <span style="color:blue">+0.9%</span> |
| 2026-02 | <span style="color:red">-3.1%</span> | <span style="color:red">-4.5%</span> | <span style="color:red">-9.3%</span> | <span style="color:red">-9.9%</span> | <span style="color:red">-10.5%</span> | <span style="color:red">-10.5%</span> | <span style="color:red">-3.4%</span> |
| 2026-03 | <span style="color:red">-11.3%</span> | <span style="color:red">-6.7%</span> | <span style="color:red">-4.1%</span> | <span style="color:red">-8.1%</span> | <span style="color:red">-16.5%</span> | <span style="color:red">-16.5%</span> | <span style="color:red">-5.6%</span> |

---

*Generated: 2026-03-31*
