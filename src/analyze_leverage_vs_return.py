"""
raw_leverage ビン別 × 前方1年3倍NASDAQリターン分析

GAS実装 (NASDAQ-strategy-gas) と同一パラメータで raw_leverage を算出し、
その値を10%刻みでビニングして、各ビンの前方1年間リターンを集計する。

raw_leverage = DD × VT × SlopeMult × MomDecel  (clip [0, 1])
"""

import pandas as pd
import numpy as np
import os
import sys

# =============================================================================
# CONFIG — GAS Code.gs と完全一致
# =============================================================================
CONFIG = {
    'DD': {'LOOKBACK': 200, 'EXIT_THRESHOLD': 0.82, 'REENTRY_THRESHOLD': 0.92},
    'ASYM_EWMA': {'SPAN_DOWN': 5, 'SPAN_UP': 20},
    'TREND_TV': {'MA': 150, 'TV_MIN': 0.15, 'TV_MAX': 0.35, 'RATIO_LOW': 0.85, 'RATIO_HIGH': 1.15},
    'SLOPE_MULT': {'MA': 200, 'NORM_WINDOW': 60, 'BASE': 0.7, 'SENSITIVITY': 0.3, 'MIN': 0.3, 'MAX': 1.5},
    'MOM_DECEL': {'SHORT': 40, 'LONG': 120, 'SENSITIVITY': 0.3, 'MIN': 0.5, 'MAX': 1.3, 'Z_WINDOW': 120},
}

ANNUAL_COST = 0.015  # 年1.5%
BASE_LEVERAGE = 3.0
FORWARD_DAYS = 250   # 約1年


# =============================================================================
# Layer 1: DD（ドローダウン制御）
# =============================================================================
def calc_dd(close: pd.Series) -> pd.Series:
    lookback = CONFIG['DD']['LOOKBACK']
    exit_th = CONFIG['DD']['EXIT_THRESHOLD']
    reentry_th = CONFIG['DD']['REENTRY_THRESHOLD']

    peak = close.rolling(lookback, min_periods=1).max()
    ratio = close / peak

    dd = pd.Series(1.0, index=close.index)
    state = 'HOLD'
    for i in range(len(close)):
        if state == 'HOLD' and ratio.iloc[i] <= exit_th:
            state = 'CASH'
        elif state == 'CASH' and ratio.iloc[i] >= reentry_th:
            state = 'HOLD'
        dd.iloc[i] = 1.0 if state == 'HOLD' else 0.0
    return dd


# =============================================================================
# Layer 2: AsymEWMA → TrendTV → VT
# =============================================================================
def calc_asym_ewma_vol(close: pd.Series) -> pd.Series:
    span_down = CONFIG['ASYM_EWMA']['SPAN_DOWN']
    span_up = CONFIG['ASYM_EWMA']['SPAN_UP']
    alpha_down = 2.0 / (span_down + 1)
    alpha_up = 2.0 / (span_up + 1)

    returns = close.pct_change()
    n = len(close)

    # 初期化: 最初の20日で分散推定
    init_period = 20
    init_rets = returns.iloc[1:init_period + 1].dropna()
    if len(init_rets) == 0:
        init_var = 0.0001
    else:
        init_var = (init_rets ** 2).mean()

    variance = np.full(n, np.nan)
    variance[init_period] = init_var

    # ウォームアップ: init_period までは初期分散で埋める
    for i in range(init_period + 1, n):
        ret = returns.iloc[i]
        if np.isnan(ret):
            variance[i] = variance[i - 1] if not np.isnan(variance[i - 1]) else init_var
            continue
        alpha = alpha_down if ret < 0 else alpha_up
        prev_var = variance[i - 1] if not np.isnan(variance[i - 1]) else init_var
        variance[i] = (1 - alpha) * prev_var + alpha * ret * ret

    vol = np.sqrt(np.array(variance) * 252)
    return pd.Series(vol, index=close.index)


def calc_trend_tv(close: pd.Series) -> pd.Series:
    ma_period = CONFIG['TREND_TV']['MA']
    tv_min = CONFIG['TREND_TV']['TV_MIN']
    tv_max = CONFIG['TREND_TV']['TV_MAX']
    ratio_low = CONFIG['TREND_TV']['RATIO_LOW']
    ratio_high = CONFIG['TREND_TV']['RATIO_HIGH']

    ma = close.rolling(ma_period, min_periods=ma_period).mean()
    ratio = close / ma
    trend_tv = tv_min + (tv_max - tv_min) * (ratio - ratio_low) / (ratio_high - ratio_low)
    trend_tv = trend_tv.clip(tv_min, tv_max)
    # MA未算出期間はデフォルト
    trend_tv = trend_tv.fillna((tv_min + tv_max) / 2)
    return trend_tv


def calc_vt(trend_tv: pd.Series, asym_vol: pd.Series) -> pd.Series:
    vt = (trend_tv / asym_vol).clip(0, 1.0)
    vt = vt.fillna(1.0)
    return vt


# =============================================================================
# Layer 3: SlopeMult（MA200傾き乗数）
# =============================================================================
def calc_slope_mult(close: pd.Series) -> pd.Series:
    ma_period = CONFIG['SLOPE_MULT']['MA']
    norm_window = CONFIG['SLOPE_MULT']['NORM_WINDOW']
    base = CONFIG['SLOPE_MULT']['BASE']
    sensitivity = CONFIG['SLOPE_MULT']['SENSITIVITY']
    min_val = CONFIG['SLOPE_MULT']['MIN']
    max_val = CONFIG['SLOPE_MULT']['MAX']

    ma = close.rolling(ma_period, min_periods=ma_period).mean()
    # MA の日次変化率
    ma_slope = ma.pct_change()
    # norm_window でZスコア化
    slope_mean = ma_slope.rolling(norm_window, min_periods=norm_window).mean()
    slope_std = ma_slope.rolling(norm_window, min_periods=norm_window).std()

    slope_z = (ma_slope - slope_mean) / slope_std.replace(0, np.nan)
    slope_mult = (base + sensitivity * slope_z).clip(min_val, max_val)
    slope_mult = slope_mult.fillna(1.0)
    return slope_mult


# =============================================================================
# Layer 4: MomDecel（モメンタム減速）
# =============================================================================
def calc_mom_decel(close: pd.Series) -> pd.Series:
    short_p = CONFIG['MOM_DECEL']['SHORT']
    long_p = CONFIG['MOM_DECEL']['LONG']
    sens = CONFIG['MOM_DECEL']['SENSITIVITY']
    min_val = CONFIG['MOM_DECEL']['MIN']
    max_val = CONFIG['MOM_DECEL']['MAX']
    z_window = CONFIG['MOM_DECEL']['Z_WINDOW']

    mom_short = close / close.shift(short_p) - 1
    mom_long = close / close.shift(long_p) - 1
    mom_long_norm = mom_long * (short_p / long_p)
    decel = mom_short - mom_long_norm

    decel_mean = decel.rolling(z_window, min_periods=z_window).mean()
    decel_std = decel.rolling(z_window, min_periods=z_window).std()
    decel_z = (decel - decel_mean) / decel_std.replace(0, np.nan)

    mom_decel = (1.0 + sens * decel_z).clip(min_val, max_val)
    mom_decel = mom_decel.fillna(1.0)
    return mom_decel


# =============================================================================
# raw_leverage 算出
# =============================================================================
def calc_raw_leverage(close: pd.Series) -> pd.DataFrame:
    dd = calc_dd(close)
    asym_vol = calc_asym_ewma_vol(close)
    trend_tv = calc_trend_tv(close)
    vt = calc_vt(trend_tv, asym_vol)
    slope_mult = calc_slope_mult(close)
    mom_decel = calc_mom_decel(close)

    raw_lev = (dd * vt * slope_mult * mom_decel).clip(0, 1.0)

    df = pd.DataFrame({
        'dd': dd,
        'asym_vol': asym_vol,
        'trend_tv': trend_tv,
        'vt': vt,
        'slope_mult': slope_mult,
        'mom_decel': mom_decel,
        'raw_leverage': raw_lev,
    })
    return df


# =============================================================================
# 前方1年 3倍レバレッジリターン算出
# =============================================================================
def calc_forward_1y_return(close: pd.Series) -> pd.Series:
    """各日から250営業日後までの3倍レバレッジ投信のリターン"""
    daily_ret = close.pct_change()
    leveraged_ret = daily_ret * BASE_LEVERAGE
    daily_cost = ANNUAL_COST / 252

    # 日次の純リターン（3倍リターン - コスト）
    net_daily = leveraged_ret - daily_cost

    n = len(close)
    fwd_return = pd.Series(np.nan, index=close.index)

    # cumulative product を使って高速化
    cum_prod = (1 + net_daily).cumprod()
    cum_prod = cum_prod.fillna(method='ffill')

    for i in range(1, n - FORWARD_DAYS):
        # i日目の終値でエントリー → i+FORWARD_DAYS日目にイグジット
        fwd_return.iloc[i] = cum_prod.iloc[i + FORWARD_DAYS] / cum_prod.iloc[i] - 1

    return fwd_return


# =============================================================================
# メイン分析
# =============================================================================
def main():
    # データ読み込み
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(data_dir, 'NASDAQ_Dairy_since1973.csv')

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    close = df['Close'].astype(float)
    dates = df['Date']

    print(f"Data range: {dates.iloc[0].strftime('%Y-%m-%d')} ~ {dates.iloc[-1].strftime('%Y-%m-%d')}")
    print(f"Total rows: {len(df)}")

    # Step 1: raw_leverage 算出
    print("\nCalculating raw_leverage (4 layers)...")
    layers = calc_raw_leverage(close)
    layers['date'] = dates
    layers['close'] = close

    # Step 2: 前方1年リターン算出
    print("Calculating forward 1-year 3x leveraged returns...")
    layers['fwd_1y_return'] = calc_forward_1y_return(close)

    # ウォームアップ期間（MA200+NORM60+Z_WINDOW120 = ~380日）を除外
    warmup = 400
    valid = layers.iloc[warmup:].dropna(subset=['fwd_1y_return']).copy()
    print(f"Valid data points: {len(valid)} (after warmup={warmup} and forward window)")
    print(f"Valid date range: {valid['date'].iloc[0].strftime('%Y-%m-%d')} ~ {valid['date'].iloc[-1].strftime('%Y-%m-%d')}")

    # Step 3: ビニング
    bins = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
              '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    valid['leverage_bin'] = pd.cut(valid['raw_leverage'], bins=bins, labels=labels, include_lowest=True)

    # Step 4: 集計
    print("\n" + "=" * 80)
    print("raw_leverage ビン別 × 前方1年 3倍NASDAQリターン")
    print(f"年コスト: {ANNUAL_COST*100:.1f}%  |  前方期間: {FORWARD_DAYS}営業日  |  ベースレバレッジ: {BASE_LEVERAGE}x")
    print("=" * 80)

    results = []
    for label in labels:
        subset = valid[valid['leverage_bin'] == label]['fwd_1y_return']
        if len(subset) == 0:
            results.append({
                'bin': label, 'count': 0, 'mean': np.nan, 'median': np.nan,
                'std': np.nan, 'positive_pct': np.nan, 'worst': np.nan, 'best': np.nan,
                'q25': np.nan, 'q75': np.nan,
            })
            continue

        results.append({
            'bin': label,
            'count': len(subset),
            'mean': subset.mean(),
            'median': subset.median(),
            'std': subset.std(),
            'positive_pct': (subset > 0).mean(),
            'worst': subset.min(),
            'best': subset.max(),
            'q25': subset.quantile(0.25),
            'q75': subset.quantile(0.75),
        })

    results_df = pd.DataFrame(results)

    # 表示
    print(f"\n{'Bin':<12} {'N':>6} {'Mean':>9} {'Median':>9} {'Std':>9} {'P(>0)':>8} {'Worst':>9} {'Best':>9} {'Q25':>9} {'Q75':>9}")
    print("-" * 100)
    for _, r in results_df.iterrows():
        if r['count'] == 0:
            print(f"{r['bin']:<12} {int(r['count']):>6}    ---       ---       ---      ---       ---       ---       ---       ---")
        else:
            print(f"{r['bin']:<12} {int(r['count']):>6} {r['mean']:>+8.1%} {r['median']:>+8.1%} {r['std']:>8.1%} {r['positive_pct']:>7.1%} {r['worst']:>+8.1%} {r['best']:>+8.1%} {r['q25']:>+8.1%} {r['q75']:>+8.1%}")

    # 全体統計
    total = valid['fwd_1y_return']
    print("-" * 100)
    print(f"{'ALL':<12} {len(total):>6} {total.mean():>+8.1%} {total.median():>+8.1%} {total.std():>8.1%} {(total>0).mean():>7.1%} {total.min():>+8.1%} {total.max():>+8.1%} {total.quantile(0.25):>+8.1%} {total.quantile(0.75):>+8.1%}")

    # レイヤー別分布
    print("\n" + "=" * 60)
    print("raw_leverage 分布サマリー")
    print("=" * 60)
    print(f"{'Bin':<12} {'N':>6} {'%':>7}")
    print("-" * 30)
    for label in labels:
        count = len(valid[valid['leverage_bin'] == label])
        pct = count / len(valid) * 100
        print(f"{label:<12} {count:>6} {pct:>6.1f}%")

    # 各レイヤーの統計
    print("\n" + "=" * 60)
    print("各レイヤーの統計量")
    print("=" * 60)
    for col in ['dd', 'vt', 'slope_mult', 'mom_decel', 'raw_leverage']:
        v = valid[col]
        print(f"{col:<15} mean={v.mean():.3f}  std={v.std():.3f}  min={v.min():.3f}  max={v.max():.3f}")

    # CSV出力
    out_path = os.path.join(data_dir, 'leverage_bin_analysis.csv')
    results_df.to_csv(out_path, index=False, float_format='%.4f')
    print(f"\nResults saved to: {out_path}")

    # 詳細データも出力（検証用）
    detail_path = os.path.join(data_dir, 'leverage_daily_detail.csv')
    detail_cols = ['date', 'close', 'dd', 'asym_vol', 'trend_tv', 'vt', 'slope_mult', 'mom_decel', 'raw_leverage', 'fwd_1y_return']
    valid[detail_cols].to_csv(detail_path, index=False, float_format='%.6f')
    print(f"Daily detail saved to: {detail_path}")


if __name__ == '__main__':
    main()
