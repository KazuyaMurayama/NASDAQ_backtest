"""
recalc_no6_no7_aftertax.py
==========================
No.6 (DH Dyn 2x3x [A]) と No.7 (BH 1x) の手取り指標を正確に再計算する。

問題: 7STRATEGY_PERFORMANCE_REPORT_20260529.md の §2 で
     CAGR⓽_OOS = CAGR_pre-tax × 0.8273 という近似を使っている。
     正しくは: 年次リターンに逐年 ×0.8273 を適用してから複利計算する。

修正対象指標:
  - CAGR⓽_OOS  : OOS期間の手取りCAGR（主要修正対象）
  - CAGR⓽_IS   : IS期間の手取りCAGR
  - CAGR⓽_FULL : 全期間の手取りCAGR
  - MaxDD⓽      : 手取りNAVシリーズから計算した最大ドローダウン
  - Worst10Y★⓽ : 手取り年次リターンから計算した最悪10年CAGR
  - P10_5Y⓽    : 手取り年次リターンから計算した5年CAGRのP10
  - IS-OOS gap⓽: CAGR⓽_IS - CAGR⓽_OOS

税モデル: §3-A (v3.1 継続)
  各年次リターンに ×0.8273 を適用 (= 1 - 0.20315 × 0.85)
  SBI CFD 戦略 (No.1-5) の場合は追加で -0.66%/yr コスト控除があるが、
  No.6/7 はコスト既反映 or ゼロなのでコスト控除はなし。

データソース: YEARLY_RETURNS_REPORT_2026-05-12_v4.md の年次リターン
期間設定:
  FULL : 1974-01-02 〜 2026-03-26  (52.26 years)
  IS   : 1974-01-02 〜 2021-05-07  (47.35 years)
  OOS  : 2021-05-08 〜 2026-03-26  (4.879 years)
"""
import numpy as np
import sys
import os

# ---------------------------------------------------------------------------
# 年次リターンデータ (v4 report より。小数表記)
# ---------------------------------------------------------------------------
# 2026年は部分年 (Jan 1 - Mar 26 = 84/365 = 0.2301yr) の実績値
# 2021年は全年データ (OOS分割は下記で処理)
# DH Dyn 2x3x [A]: SOFR+Bond補正済み (v4 Scenario D)
# BH 1x          : 無レバ補正なし

DH_A_PRETAX = {
    1974: +0.104, 1975: -0.050, 1976: +0.450, 1977: +0.006, 1978: +0.491,
    1979: +0.383, 1980: +0.443, 1981: -0.306, 1982: +0.876, 1983: +0.171,
    1984: -0.097, 1985: +0.592, 1986: +0.346, 1987: +0.195, 1988: -0.158,
    1989: +0.258, 1990: -0.142, 1991: +0.529, 1992: +0.308, 1993: +0.068,
    1994: -0.103, 1995: +0.765, 1996: +0.191, 1997: +0.517, 1998: +0.782,
    1999: +1.191, 2000: +0.011, 2001: +0.015, 2002: +0.263, 2003: +0.702,
    2004: +0.112, 2005: +0.007, 2006: +0.349, 2007: +0.232, 2008: +0.215,
    2009: +0.247, 2010: +0.645, 2011: -0.020, 2012: +0.283, 2013: +0.329,
    2014: +0.136, 2015: -0.192, 2016: +0.066, 2017: +0.355, 2018: +0.008,
    2019: +0.451, 2020: +0.842, 2021: +0.247, 2022: -0.300, 2023: +0.412,
    2024: +0.244, 2025: +0.402, 2026: -0.117,
}

BH_1X_PRETAX = {
    1974: -0.354, 1975: +0.298, 1976: +0.261, 1977: +0.073, 1978: +0.123,
    1979: +0.281, 1980: +0.339, 1981: -0.032, 1982: +0.187, 1983: +0.199,
    1984: -0.113, 1985: +0.315, 1986: +0.074, 1987: -0.052, 1988: +0.154,
    1989: +0.192, 1990: -0.178, 1991: +0.569, 1992: +0.155, 1993: +0.147,
    1994: -0.032, 1995: +0.399, 1996: +0.227, 1997: +0.216, 1998: +0.396,
    1999: +0.856, 2000: -0.393, 2001: -0.211, 2002: -0.315, 2003: +0.500,
    2004: +0.086, 2005: +0.014, 2006: +0.095, 2007: +0.098, 2008: -0.405,
    2009: +0.439, 2010: +0.169, 2011: -0.018, 2012: +0.159, 2013: +0.383,
    2014: +0.134, 2015: +0.057, 2016: +0.075, 2017: +0.282, 2018: -0.039,
    2019: +0.352, 2020: +0.436, 2021: +0.214, 2022: -0.331, 2023: +0.434,
    2024: +0.286, 2025: +0.204, 2026: -0.079,
}

# ---------------------------------------------------------------------------
# 期間定義
# ---------------------------------------------------------------------------
# OOS 開始: 2021-05-08 → 2021年のうち OOS に属する割合
OOS_2021_DAYS = 237          # May 8 - Dec 31
OOS_2021_FRAC = OOS_2021_DAYS / 365  # 0.6493
IS_2021_FRAC  = 1.0 - OOS_2021_FRAC  # 0.3507

# 2026年は部分年 (実績値 Jan 1 - Mar 26 = 84日)
# → v4 の年次リターンは既に 84日分の実績値なので、そのまま使う
YEAR_2026_ACTUAL_DAYS = 84
YEAR_2026_YR_FRAC = YEAR_2026_ACTUAL_DAYS / 365  # 0.2301

# 全期間・IS・OOSの実年数
FULL_YEARS = 47.35 + 4.879          # ≈ 52.26 yr
IS_YEARS   = 47.35                  # 1974-01-02 〜 2021-05-07
OOS_YEARS  = 4.879                  # 2021-05-08 〜 2026-03-26

TAX_FACTOR = 0.8273  # §3-A

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def partial_return(r_annual: float, frac: float) -> float:
    """年次リターン r_annual の frac 割合分のリターン（幾何的比例配分）"""
    return (1.0 + r_annual) ** frac - 1.0


def after_tax(r: float) -> float:
    """§3-A: 年次リターンに税適用"""
    return r * TAX_FACTOR


def build_nav_and_returns(pretax: dict, oos_2021_frac: float = OOS_2021_FRAC) -> dict:
    """
    年次リターンに税適用し、各セクション(IS/OOS/FULL)の
    NAVシリーズ・年次リターン列を構築する。

    Returns: {
        'nav_full'  : list of NAV values (1 entry per year, 2026 is partial)
        'returns_at': dict {period: list of after-tax annual returns}
        'years'     : list of years used
    }
    全年 (1974-2026) のデータを使い、2021 を OOS/IS に分割する。
    """
    years = sorted(pretax.keys())  # 1974..2026

    nav_full   = 1.0   # FULL period NAV (starts 1974)
    nav_is     = 1.0   # IS period NAV
    nav_oos    = 1.0   # OOS period NAV

    returns_full = []
    returns_is   = []
    returns_oos  = []

    nav_full_series = [1.0]
    nav_oos_series  = [1.0]

    for y in years:
        r_pre = pretax[y]

        if y < 2021:
            # IS only
            r_at = after_tax(r_pre)
            returns_full.append(r_at)
            returns_is.append(r_at)
            nav_full *= (1.0 + r_at)
            nav_is   *= (1.0 + r_at)
            nav_full_series.append(nav_full)

        elif y == 2021:
            # 2021 is split: IS 前半 + OOS 後半
            # IS 分: IS_2021_FRAC 割合の部分年リターン
            r_is_partial = partial_return(r_pre, IS_2021_FRAC)
            r_is_at = after_tax(r_is_partial)
            nav_is *= (1.0 + r_is_at)
            returns_is.append(r_is_at)

            # OOS 分: OOS_2021_FRAC 割合の部分年リターン
            r_oos_partial = partial_return(r_pre, oos_2021_frac)
            r_oos_at = after_tax(r_oos_partial)
            nav_oos *= (1.0 + r_oos_at)
            returns_oos.append(r_oos_at)

            # FULL 用: 全年 (OOS後半も含む)
            r_full_at = after_tax(r_pre)
            returns_full.append(r_full_at)
            nav_full *= (1.0 + r_full_at)
            nav_full_series.append(nav_full)
            nav_oos_series.append(nav_oos)

        elif y == 2026:
            # 2026 は部分年の実績値 (84日分) → そのまま適用
            # (幾何的比例配分はしない: v4 の値は既に実績)
            r_at = after_tax(r_pre)
            returns_full.append(r_at)
            returns_oos.append(r_at)
            nav_full *= (1.0 + r_at)
            nav_oos  *= (1.0 + r_at)
            nav_full_series.append(nav_full)
            nav_oos_series.append(nav_oos)

        else:  # 2022-2025: OOS full years
            r_at = after_tax(r_pre)
            returns_full.append(r_at)
            returns_oos.append(r_at)
            nav_full *= (1.0 + r_at)
            nav_oos  *= (1.0 + r_at)
            nav_full_series.append(nav_full)
            nav_oos_series.append(nav_oos)

    return {
        'nav_full':         nav_full,
        'nav_is':           nav_is,
        'nav_oos':          nav_oos,
        'nav_full_series':  nav_full_series,
        'nav_oos_series':   nav_oos_series,
        'returns_full':     returns_full,
        'returns_is':       returns_is,
        'returns_oos':      returns_oos,
    }


def compute_maxdd_annual(nav_series: list) -> float:
    """年次NAVシリーズからMaxDDを計算（年次粒度・近似）"""
    peak = nav_series[0]
    max_dd = 0.0
    for nav in nav_series:
        if nav > peak:
            peak = nav
        dd = nav / peak - 1.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def compute_worst10y_annual(pretax_dict: dict) -> float:
    """
    カレンダー年ベース最悪10年CAGR (Worst10Y★) を年次リターンから計算。
    全ての10年窓 (連続10年) の CAGR の最小値を返す。
    手取り後の年次リターンを使う。
    """
    years = sorted(pretax_dict.keys())
    at_returns = {y: after_tax(r) for y, r in pretax_dict.items()}

    # 2026 は部分年なので 10年窓計算から除外 (or 含める場合は注意)
    full_years = [y for y in years if y < 2026]

    worst = float('inf')
    n = 10
    for i in range(len(full_years) - n + 1):
        window = full_years[i:i+n]
        # 10年間の複利
        nav_10 = 1.0
        for y in window:
            nav_10 *= (1.0 + at_returns[y])
        cagr_10 = nav_10 ** (1.0/n) - 1.0
        if cagr_10 < worst:
            worst = cagr_10
    return worst if worst != float('inf') else float('nan')


def compute_p10_5y_annual(pretax_dict: dict) -> float:
    """5年CAGR分布のP10 (10パーセンタイル) を年次リターンから計算"""
    years = sorted(pretax_dict.keys())
    at_returns = {y: after_tax(r) for y, r in pretax_dict.items()}
    full_years = [y for y in years if y < 2026]

    cagrs_5y = []
    n = 5
    for i in range(len(full_years) - n + 1):
        window = full_years[i:i+n]
        nav_5 = 1.0
        for y in window:
            nav_5 *= (1.0 + at_returns[y])
        cagr_5 = nav_5 ** (1.0/n) - 1.0
        cagrs_5y.append(cagr_5)

    if not cagrs_5y:
        return float('nan')
    return float(np.percentile(cagrs_5y, 10))


def compute_sharpe_annual(returns_list: list) -> float:
    """年次リターンリストからSharpe比を計算 (Rf=0)。
    注: 年次データは数点しかないため OOS Sharpeは参考値。"""
    if len(returns_list) < 2:
        return float('nan')
    arr = np.array(returns_list)
    mean = np.mean(arr)
    std  = np.std(arr, ddof=1)
    if std == 0:
        return float('nan')
    return mean / std


# ---------------------------------------------------------------------------
# メイン計算
# ---------------------------------------------------------------------------

def calc_metrics(name: str, pretax: dict, verbose: bool = True) -> dict:
    r = build_nav_and_returns(pretax)

    # CAGR
    cagr_full = r['nav_full'] ** (1.0 / FULL_YEARS) - 1.0
    cagr_is   = r['nav_is']   ** (1.0 / IS_YEARS)   - 1.0
    cagr_oos  = r['nav_oos']  ** (1.0 / OOS_YEARS)  - 1.0

    # MaxDD (annual approximation)
    max_dd = compute_maxdd_annual(r['nav_full_series'])

    # Worst10Y★
    worst10y = compute_worst10y_annual(pretax)

    # P10_5Y
    p10_5y = compute_p10_5y_annual(pretax)

    # IS-OOS gap (after-tax)
    is_oos_gap = cagr_is - cagr_oos

    # Sharpe_OOS from annual after-tax (参考値のみ)
    sharpe_oos_annual = compute_sharpe_annual(r['returns_oos'])

    result = dict(
        name=name,
        cagr_full=cagr_full, cagr_is=cagr_is, cagr_oos=cagr_oos,
        max_dd=max_dd, worst10y=worst10y, p10_5y=p10_5y,
        is_oos_gap=is_oos_gap, sharpe_oos_annual=sharpe_oos_annual,
    )

    if verbose:
        _print_results(result, pretax)

    return result


def _print_results(res: dict, pretax: dict):
    name = res['name']
    print(f"\n{'='*60}")
    print(f"  {name}  — 手取り指標 (§3-A ×0.8273 逐年適用)")
    print(f"{'='*60}")
    print(f"  CAGR⓽_OOS   (correct, compounded): {res['cagr_oos']*100:+.2f}%")
    print(f"  CAGR⓽_IS                         : {res['cagr_is']*100:+.2f}%")
    print(f"  CAGR⓽_FULL                        : {res['cagr_full']*100:+.2f}%")
    print(f"  IS-OOS gap⓽                       : {res['is_oos_gap']*100:+.2f}pp")
    print(f"  MaxDD⓽ (annual approx.)           : {res['max_dd']*100:+.2f}%")
    print(f"  Worst10Y★⓽                        : {res['worst10y']*100:+.2f}%")
    print(f"  P10_5Y⓽                           : {res['p10_5y']*100:+.2f}%")
    print(f"")
    print(f"  参考: 現行(誤)近似値  CAGR⓽_OOS  = CAGR_pre × 0.8273")
    cagr_pre_oos_dh  = 0.1488 if 'DH' in name else 0.1011
    cagr_wrong = cagr_pre_oos_dh * 0.8273
    print(f"    CAGR_pre_OOS (v4) = {cagr_pre_oos_dh*100:.2f}% → ×0.8273 = {cagr_wrong*100:.2f}%")
    print(f"  ΔError = {(res['cagr_oos'] - cagr_wrong)*100:+.2f}pp  (correct - wrong)")
    print(f"")
    print(f"  ⚠ MaxDD は年次NAVから計算 (日次より大きめに出る場合あり)")
    print(f"  ⚠ Sharpe_OOS(annual): {res['sharpe_oos_annual']:.3f}  "
          f"(年次データ5点のみ — 参考値, 信頼性低)")


# ---------------------------------------------------------------------------
# BH 1x を日次NASDAQデータで再計算 (より正確なMaxDD用)
# ---------------------------------------------------------------------------

def calc_bh1x_daily(verbose: bool = True) -> dict:
    """
    BH 1x を日次NASDAQデータから計算。
    各年末に §3-A 税適用 (年間リターン × 0.8273) でNAVを調整。
    """
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_dir, 'data', 'NASDAQ_extended_to_2026.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(repo_dir, 'NASDAQ_extended_to_2026.csv')
    if not os.path.exists(csv_path):
        print("  ⚠ NASDAQ_extended_to_2026.csv が見つかりません — 日次計算をスキップ")
        return {}

    import pandas as pd

    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    # Close価格列を特定
    close_col = None
    for c in ['Close', 'close', 'Adj Close', 'adj_close']:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        close_col = df.columns[0]

    prices = df[close_col].sort_index().dropna()

    # OOS / IS / FULL 期間
    IS_START  = pd.Timestamp('1974-01-02')
    IS_END    = pd.Timestamp('2021-05-07')
    OOS_START = pd.Timestamp('2021-05-08')
    OOS_END   = pd.Timestamp('2026-03-26')
    FULL_START = IS_START

    prices = prices.loc[FULL_START:OOS_END]

    # 年次後税NAVシリーズを構築
    # アプローチ: 各年末時点で「当年リターン × 0.8273」を反映した
    # 調整係数をかけていく (年内は税なし、年末に確定)
    nav_raw = prices / prices.iloc[0]  # tax-before NAV

    # 年次後税NAV
    nav_after = nav_raw.copy() * 0.0  # same index, zero-fill
    nav_after.iloc[0] = 1.0

    years_in_data = sorted(nav_raw.index.year.unique())

    # pre-tax annual returns from daily data
    pretax_daily_annual = {}
    for yr in years_in_data:
        yr_prices = prices[prices.index.year == yr]
        if len(yr_prices) == 0:
            continue
        # previous year-end price (or start of data)
        prev = prices[prices.index < yr_prices.index[0]]
        if len(prev) == 0:
            start_p = yr_prices.iloc[0]
        else:
            start_p = prev.iloc[-1]
        end_p = yr_prices.iloc[-1]
        pretax_daily_annual[yr] = (end_p / start_p) - 1.0

    # Build after-tax annual NAV series (annual endpoints)
    cum_nav = 1.0
    annual_nav_pts = {min(years_in_data) - 1: 1.0}  # fictional start=1

    for yr in years_in_data:
        r_pre = pretax_daily_annual.get(yr, 0.0)
        r_at  = r_pre * TAX_FACTOR
        cum_nav *= (1.0 + r_at)
        annual_nav_pts[yr] = cum_nav

    # MaxDD from annual after-tax NAV
    nav_vals = [annual_nav_pts[y] for y in sorted(annual_nav_pts.keys())]
    max_dd_annual = compute_maxdd_annual(nav_vals)

    # OOS CAGR from daily data (after-tax)
    oos_years_data = [y for y in years_in_data if y >= 2021]
    oos_nav = 1.0
    # 2021 OOS portion: May 8 to Dec 31
    p_may8 = prices[prices.index >= OOS_START].iloc[0] if len(prices[prices.index >= OOS_START]) > 0 else None
    p_dec31_2021 = prices[prices.index.year == 2021].iloc[-1]
    p_oos_end   = prices.iloc[-1]  # 2026-03-26

    if p_may8 is not None:
        # 2021 OOS raw return (May 8 - Dec 31, 2021)
        yr21_oos_raw_r = p_dec31_2021 / p_may8 - 1.0
        oos_nav *= (1.0 + yr21_oos_raw_r * TAX_FACTOR)

        # 2022-2025 full years
        for yr in [2022, 2023, 2024, 2025]:
            r = pretax_daily_annual.get(yr, 0.0)
            oos_nav *= (1.0 + r * TAX_FACTOR)

        # 2026 partial year (Jan 1 - Mar 26)
        p_jan1_2026 = prices[prices.index.year == 2026].iloc[0] if len(prices[prices.index.year == 2026]) > 0 else None
        if p_jan1_2026 is not None:
            prev_2026 = prices[prices.index < prices[prices.index.year == 2026].index[0]]
            start_2026 = prev_2026.iloc[-1] if len(prev_2026) > 0 else p_jan1_2026
            r_2026 = p_oos_end / start_2026 - 1.0
            oos_nav *= (1.0 + r_2026 * TAX_FACTOR)

    oos_years_actual = (OOS_END - OOS_START).days / 365.25
    cagr_oos_daily = oos_nav ** (1.0 / oos_years_actual) - 1.0

    # Full CAGR daily
    full_nav = annual_nav_pts.get(max(years_in_data), cum_nav)
    full_years_actual = (OOS_END - IS_START).days / 365.25
    cagr_full_daily = full_nav ** (1.0 / full_years_actual) - 1.0

    # IS CAGR daily
    is_nav = annual_nav_pts.get(2020, 1.0)
    # add 2021 IS portion (Jan 1 - May 7, 2021)
    p_jan1_2021 = prices[prices.index.year == 2021].iloc[0] if len(prices[prices.index.year == 2021]) > 0 else None
    if p_jan1_2021 is not None:
        prev_2021 = prices[prices.index < prices[prices.index.year == 2021].index[0]]
        start_2021 = prev_2021.iloc[-1] if len(prev_2021) > 0 else p_jan1_2021
        p_may7_2021 = prices[prices.index <= IS_END].iloc[-1]
        r_is_2021 = p_may7_2021 / start_2021 - 1.0
        is_nav *= (1.0 + r_is_2021 * TAX_FACTOR)

    is_years_actual = (IS_END - IS_START).days / 365.25
    cagr_is_daily = is_nav ** (1.0 / is_years_actual) - 1.0

    # Sharpe_OOS from daily after-tax returns
    oos_prices = prices.loc[OOS_START:OOS_END]
    if len(oos_prices) > 1:
        oos_daily_ret = oos_prices.pct_change().dropna()
        oos_daily_ret_at = oos_daily_ret * TAX_FACTOR
        mean_d = float(oos_daily_ret_at.mean())
        std_d  = float(oos_daily_ret_at.std())
        sharpe_oos = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else float('nan')
    else:
        sharpe_oos = float('nan')

    result = dict(
        name='BH 1x (daily)',
        cagr_oos=cagr_oos_daily,
        cagr_is=cagr_is_daily,
        cagr_full=cagr_full_daily,
        max_dd_annual=max_dd_annual,
        sharpe_oos=sharpe_oos,
        is_oos_gap=cagr_is_daily - cagr_oos_daily,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  BH 1x (日次NASDAQデータ使用) — より正確なMaxDD")
        print(f"{'='*60}")
        print(f"  CAGR⓽_OOS   (daily): {cagr_oos_daily*100:+.2f}%")
        print(f"  CAGR⓽_IS    (daily): {cagr_is_daily*100:+.2f}%")
        print(f"  CAGR⓽_FULL  (daily): {cagr_full_daily*100:+.2f}%")
        print(f"  MaxDD⓽ (annual)    : {max_dd_annual*100:+.2f}%")
        print(f"  Sharpe_OOS (daily) : {sharpe_oos:+.3f}")
        print(f"  IS-OOS gap⓽        : {(cagr_is_daily - cagr_oos_daily)*100:+.2f}pp")

    return result


# ---------------------------------------------------------------------------
# 比較サマリ
# ---------------------------------------------------------------------------

def print_summary(dh_res: dict, bh_annual: dict, bh_daily: dict):
    print(f"\n{'='*70}")
    print(f"  比較サマリ: 旧値(×CAGR_pre 近似) vs 新値(逐年複利)")
    print(f"{'='*70}")
    print(f"  {'指標':<22} {'旧 DH Dyn':>12} {'新 DH Dyn':>12}  |  {'旧 BH 1x':>10} {'新 BH 1x':>10}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*12}  |  {'-'*10}  {'-'*10}")

    # DH Dyn
    dh_old_cagr_oos = 0.1488 * 0.8273
    bh_old_cagr_oos = 0.1011 * 0.8273

    dh_new = dh_res
    bh_new_oos = bh_daily.get('cagr_oos', bh_annual.get('cagr_oos', float('nan')))

    rows = [
        ('CAGR⓽_OOS',   dh_old_cagr_oos,    dh_new['cagr_oos'],
                         bh_old_cagr_oos,    bh_new_oos),
        ('CAGR⓽_IS',    0.2336*0.8273,       dh_new['cagr_is'],
                         0.1113*0.8273,       bh_daily.get('cagr_is', bh_annual.get('cagr_is', float('nan')))),
        ('CAGR⓽_FULL',  0.2250*0.8273,       dh_new['cagr_full'],
                         0.1098*0.8273,       bh_daily.get('cagr_full', bh_annual.get('cagr_full', float('nan')))),
        ('MaxDD⓽',       -0.451,              dh_new['max_dd'],
                          -0.779,             bh_daily.get('max_dd_annual', bh_annual.get('max_dd', float('nan')))),
        ('Worst10Y★⓽',  float('nan'),         dh_new['worst10y'],
                          float('nan'),        bh_annual.get('worst10y', float('nan'))),
        ('P10_5Y⓽',     float('nan'),         dh_new['p10_5y'],
                          float('nan'),        bh_annual.get('p10_5y', float('nan'))),
    ]

    for label, dh_old, dh_new_val, bh_old, bh_new_val in rows:
        def fmt(v):
            if v != v: return '    —   '  # NaN
            if 'DD' in label or 'MaxDD' in label:
                return f'{v*100:+.1f}%'
            if 'gap' in label.lower():
                return f'{v*100:+.2f}pp'
            return f'{v*100:+.1f}%'
        print(f"  {label:<22}  {fmt(dh_old):>10}  {fmt(dh_new_val):>10}  |  {fmt(bh_old):>8}  {fmt(bh_new_val):>8}")


if __name__ == '__main__':
    print("No.6 DH Dyn 2x3x [A] — 手取り指標再計算")
    dh_res = calc_metrics('DH Dyn 2x3x [A]', DH_A_PRETAX)

    print("\nNo.7 BH 1x — 年次データベース")
    bh_annual = calc_metrics('BH 1x (annual)', BH_1X_PRETAX)

    print("\nNo.7 BH 1x — 日次NASDAQデータ（より正確なMaxDD・Sharpe）")
    bh_daily = calc_bh1x_daily()

    print_summary(dh_res, bh_annual, bh_daily)
    print()
