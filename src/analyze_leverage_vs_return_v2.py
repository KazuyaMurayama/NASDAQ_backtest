"""
raw_leverage ビン別 × 前方リターン分析 V2

V1からの変更点:
  1. ビニング改善: 0%ちょうど / 100%ちょうど を独立ビンに分離
  2. 複数前方期間: 10営業日 / 20営業日 / 60営業日 / 250営業日

raw_leverage = DD × VT × SlopeMult × MomDecel  (clip [0, 1])
"""

import pandas as pd
import numpy as np
import os

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

ANNUAL_COST = 0.015
BASE_LEVERAGE = 3.0
FORWARD_PERIODS = [10, 20, 60, 250]  # 複数前方期間

# ビン定義: 0%ちょうど / (0,10%] / ... / [90%,100%) / 100%ちょうど
BIN_LABELS = ['=0%', '0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
              '50-60%', '60-70%', '70-80%', '80-90%', '90-100%', '=100%']


# =============================================================================
# Layer計算（V1と同一）
# =============================================================================
def calc_dd(close):
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


def calc_asym_ewma_vol(close):
    alpha_down = 2.0 / (CONFIG['ASYM_EWMA']['SPAN_DOWN'] + 1)
    alpha_up = 2.0 / (CONFIG['ASYM_EWMA']['SPAN_UP'] + 1)
    returns = close.pct_change()
    n = len(close)
    init_period = 20
    init_rets = returns.iloc[1:init_period + 1].dropna()
    init_var = (init_rets ** 2).mean() if len(init_rets) > 0 else 0.0001
    variance = np.full(n, np.nan)
    variance[init_period] = init_var
    for i in range(init_period + 1, n):
        ret = returns.iloc[i]
        if np.isnan(ret):
            variance[i] = variance[i - 1] if not np.isnan(variance[i - 1]) else init_var
            continue
        alpha = alpha_down if ret < 0 else alpha_up
        prev_var = variance[i - 1] if not np.isnan(variance[i - 1]) else init_var
        variance[i] = (1 - alpha) * prev_var + alpha * ret * ret
    return pd.Series(np.sqrt(np.array(variance) * 252), index=close.index)


def calc_trend_tv(close):
    c = CONFIG['TREND_TV']
    ma = close.rolling(c['MA'], min_periods=c['MA']).mean()
    ratio = close / ma
    tv = c['TV_MIN'] + (c['TV_MAX'] - c['TV_MIN']) * (ratio - c['RATIO_LOW']) / (c['RATIO_HIGH'] - c['RATIO_LOW'])
    return tv.clip(c['TV_MIN'], c['TV_MAX']).fillna((c['TV_MIN'] + c['TV_MAX']) / 2)


def calc_vt(trend_tv, asym_vol):
    return (trend_tv / asym_vol).clip(0, 1.0).fillna(1.0)


def calc_slope_mult(close):
    c = CONFIG['SLOPE_MULT']
    ma = close.rolling(c['MA'], min_periods=c['MA']).mean()
    ma_slope = ma.pct_change()
    slope_mean = ma_slope.rolling(c['NORM_WINDOW'], min_periods=c['NORM_WINDOW']).mean()
    slope_std = ma_slope.rolling(c['NORM_WINDOW'], min_periods=c['NORM_WINDOW']).std()
    slope_z = (ma_slope - slope_mean) / slope_std.replace(0, np.nan)
    return (c['BASE'] + c['SENSITIVITY'] * slope_z).clip(c['MIN'], c['MAX']).fillna(1.0)


def calc_mom_decel(close):
    c = CONFIG['MOM_DECEL']
    mom_short = close / close.shift(c['SHORT']) - 1
    mom_long = close / close.shift(c['LONG']) - 1
    decel = mom_short - mom_long * (c['SHORT'] / c['LONG'])
    decel_mean = decel.rolling(c['Z_WINDOW'], min_periods=c['Z_WINDOW']).mean()
    decel_std = decel.rolling(c['Z_WINDOW'], min_periods=c['Z_WINDOW']).std()
    decel_z = (decel - decel_mean) / decel_std.replace(0, np.nan)
    return (1.0 + c['SENSITIVITY'] * decel_z).clip(c['MIN'], c['MAX']).fillna(1.0)


def calc_raw_leverage(close):
    dd = calc_dd(close)
    asym_vol = calc_asym_ewma_vol(close)
    trend_tv = calc_trend_tv(close)
    vt = calc_vt(trend_tv, asym_vol)
    slope_mult = calc_slope_mult(close)
    mom_decel = calc_mom_decel(close)
    raw_lev = (dd * vt * slope_mult * mom_decel).clip(0, 1.0)
    return pd.DataFrame({
        'dd': dd, 'asym_vol': asym_vol, 'trend_tv': trend_tv,
        'vt': vt, 'slope_mult': slope_mult, 'mom_decel': mom_decel,
        'raw_leverage': raw_lev,
    })


# =============================================================================
# 前方リターン算出（任意期間対応）
# =============================================================================
def calc_forward_return(close, fwd_days):
    daily_ret = close.pct_change()
    net_daily = daily_ret * BASE_LEVERAGE - ANNUAL_COST / 252
    cum_prod = (1 + net_daily).cumprod().ffill()
    n = len(close)
    fwd = pd.Series(np.nan, index=close.index)
    for i in range(1, n - fwd_days):
        fwd.iloc[i] = cum_prod.iloc[i + fwd_days] / cum_prod.iloc[i] - 1
    return fwd


# =============================================================================
# ビニング（0%ちょうど / 100%ちょうど を分離）
# =============================================================================
def assign_bin(val):
    if val == 0.0:
        return '=0%'
    if val >= 1.0:
        return '=100%'
    # (0, 1) の範囲を10%刻み
    idx = int(val * 10)  # 0..9
    if idx >= 10:
        idx = 9
    inner_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
                    '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    return inner_labels[idx]


# =============================================================================
# 集計関数
# =============================================================================
def summarize_bin(subset):
    if len(subset) == 0:
        return {'count': 0, 'mean': np.nan, 'median': np.nan, 'std': np.nan,
                'positive_pct': np.nan, 'worst': np.nan, 'best': np.nan,
                'q25': np.nan, 'q75': np.nan}
    return {
        'count': len(subset),
        'mean': subset.mean(),
        'median': subset.median(),
        'std': subset.std(),
        'positive_pct': (subset > 0).mean(),
        'worst': subset.min(),
        'best': subset.max(),
        'q25': subset.quantile(0.25),
        'q75': subset.quantile(0.75),
    }


# =============================================================================
# メイン
# =============================================================================
def main():
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(data_dir, 'NASDAQ_Dairy_since1973.csv')

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    close = df['Close'].astype(float)
    dates = df['Date']

    print(f"Data: {dates.iloc[0].strftime('%Y-%m-%d')} ~ {dates.iloc[-1].strftime('%Y-%m-%d')} ({len(df)} rows)")

    # raw_leverage 算出
    print("\nCalculating raw_leverage (4 layers)...")
    layers = calc_raw_leverage(close)
    layers['date'] = dates
    layers['close'] = close

    # 前方リターン算出（全期間）
    for fwd in FORWARD_PERIODS:
        col = f'fwd_{fwd}d'
        print(f"Calculating forward {fwd}-day 3x leveraged returns...")
        layers[col] = calc_forward_return(close, fwd)

    # ウォームアップ除外
    warmup = 400
    # 最短前方期間（10日）があればOK
    valid = layers.iloc[warmup:].dropna(subset=['fwd_10d']).copy()
    print(f"\nValid data: {len(valid)} rows ({valid['date'].iloc[0].strftime('%Y-%m-%d')} ~ {valid['date'].iloc[-1].strftime('%Y-%m-%d')})")

    # ビン割当
    valid['leverage_bin'] = valid['raw_leverage'].apply(assign_bin)

    # ============================
    # 各前方期間ごとに集計
    # ============================
    all_results = {}

    for fwd in FORWARD_PERIODS:
        col = f'fwd_{fwd}d'
        # この前方期間のデータがある行のみ
        sub = valid.dropna(subset=[col])

        print(f"\n{'='*90}")
        print(f"前方 {fwd} 営業日 ({fwd/250:.1f}年)  |  有効データ: {len(sub)}日  |  コスト: {ANNUAL_COST*100:.1f}%/年  |  3倍レバレッジ")
        print(f"{'='*90}")
        print(f"{'Bin':<12} {'N':>6} {'Mean':>9} {'Median':>9} {'Std':>9} {'P(>0)':>8} {'Worst':>9} {'Best':>9} {'Q25':>9} {'Q75':>9}")
        print("-" * 100)

        results = []
        for label in BIN_LABELS:
            subset = sub[sub['leverage_bin'] == label][col]
            stats = summarize_bin(subset)
            stats['bin'] = label
            stats['fwd_days'] = fwd
            results.append(stats)

            if stats['count'] == 0:
                print(f"{label:<12} {0:>6}    ---       ---       ---      ---       ---       ---       ---       ---")
            else:
                print(f"{label:<12} {stats['count']:>6} {stats['mean']:>+8.1%} {stats['median']:>+8.1%} {stats['std']:>8.1%} {stats['positive_pct']:>7.1%} {stats['worst']:>+8.1%} {stats['best']:>+8.1%} {stats['q25']:>+8.1%} {stats['q75']:>+8.1%}")

        # 全体行
        total = sub[col]
        print("-" * 100)
        print(f"{'ALL':<12} {len(total):>6} {total.mean():>+8.1%} {total.median():>+8.1%} {total.std():>8.1%} {(total>0).mean():>7.1%} {total.min():>+8.1%} {total.max():>+8.1%} {total.quantile(0.25):>+8.1%} {total.quantile(0.75):>+8.1%}")

        all_results[fwd] = pd.DataFrame(results)

    # ビン分布（共通）
    print(f"\n{'='*60}")
    print("raw_leverage 分布")
    print(f"{'='*60}")
    print(f"{'Bin':<12} {'N':>6} {'%':>7}")
    print("-" * 30)
    for label in BIN_LABELS:
        c = len(valid[valid['leverage_bin'] == label])
        print(f"{label:<12} {c:>6} {c/len(valid)*100:>6.1f}%")

    # レイヤー統計
    print(f"\n{'='*60}")
    print("各レイヤーの統計量")
    print(f"{'='*60}")
    for col_name in ['dd', 'vt', 'slope_mult', 'mom_decel', 'raw_leverage']:
        v = valid[col_name]
        print(f"{col_name:<15} mean={v.mean():.3f}  std={v.std():.3f}  min={v.min():.3f}  max={v.max():.3f}")

    # CSV出力
    combined = pd.concat(all_results.values(), ignore_index=True)
    out_path = os.path.join(data_dir, 'leverage_bin_analysis_v2.csv')
    combined.to_csv(out_path, index=False, float_format='%.4f')
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
