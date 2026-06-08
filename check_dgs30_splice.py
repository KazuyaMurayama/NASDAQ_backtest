"""Check dgs30/dgs10 splice for discontinuity."""
import os, sys
import numpy as np
import pandas as pd

DATA = r'C:\Users\user\Desktop\投資・不動産\nasdaq_backtest\data'

def load_yield(name):
    path = os.path.join(DATA, f'{name}_daily.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df.columns = ['yield_pct']
    return pd.to_numeric(df['yield_pct'], errors='coerce').ffill(limit=5)

dgs10 = load_yield('dgs10')
dgs30 = load_yield('dgs30')

print("DGS10 range:", dgs10.dropna().index[0].date(), "to", dgs10.dropna().index[-1].date())
print("DGS30 range:", dgs30.dropna().index[0].date(), "to", dgs30.dropna().index[-1].date())

# Find first available dgs30 date
first_dgs30 = dgs30.dropna().index[0]
print(f"\nFirst dgs30 date: {first_dgs30.date()}")

# Show levels around splice point
window = pd.date_range(first_dgs30 - pd.Timedelta(days=10),
                       first_dgs30 + pd.Timedelta(days=10), freq='D')
print("\nYield levels around splice (dgs10 vs dgs30):")
for d in window:
    if d in dgs10.index or d in dgs30.index:
        v10 = dgs10.get(d, float('nan'))
        v30 = dgs30.get(d, float('nan'))
        marker = " <-- FIRST DGS30" if d == first_dgs30 else ""
        print(f"  {d.date()}: dgs10={v10:.4f}%  dgs30={v30:.4f}%  diff={v30-v10:.4f}%{marker}")

# Simulate the splice as in build_bond_1x_nav_corrected
y_full = dgs30.reindex(dgs10.index.union(dgs30.index))
mask_no = y_full.isna()
y_full[mask_no] = dgs10[mask_no]
y_full = y_full.ffill(limit=5)

# Check the yield change at the splice point
y_dec = y_full / 100.0
# Find the step at first_dgs30
before_idx = y_full.index.get_loc(first_dgs30) - 1
if before_idx >= 0:
    prev_date = y_full.index[before_idx]
    y_prev = y_dec.iloc[before_idx]
    y_curr = y_dec.loc[first_dgs30]
    dy = y_curr - y_prev
    price_ret_7  = -7  * dy
    price_ret_17 = -17 * dy
    print(f"\nSplice yield change on {first_dgs30.date()}:")
    print(f"  prev ({prev_date.date()}): {y_prev*100:.4f}%  curr: {y_curr*100:.4f}%  dy={dy*100:+.4f}%")
    print(f"  Bond price return (D=7):  {price_ret_7*100:+.4f}%")
    print(f"  Bond price return (D=17): {price_ret_17*100:+.4f}%")
    print(f"  Is this clipped? |{price_ret_17*100:.2f}%| > 20%: {abs(price_ret_17) > 0.20}")

# Check the bond_1x_nav daily return at the splice
print("\nSimulating bond_1x_nav returns around splice (D=17):")
y_arr = y_dec.values
n_around = 20
idx_splice = y_full.index.get_loc(first_dgs30)
for i in range(idx_splice - 5, idx_splice + 5):
    if i < 1 or i >= len(y_arr): continue
    dy = y_arr[i] - y_arr[i-1]
    price_ret  = -17 * dy
    coupon_ret = y_arr[i-1] / 252
    daily_ret  = np.clip(price_ret + coupon_ret, -0.20, 0.20)
    print(f"  {y_full.index[i].date()}: dy={dy*100:+.4f}%  price_ret={price_ret*100:+.5f}%  daily_ret={daily_ret*100:+.5f}%  CLIPPED:{abs(price_ret+coupon_ret)>0.20}")

print("\nDone.")
