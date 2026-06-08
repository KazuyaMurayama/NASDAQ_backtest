"""Check coupon basis, clip frequency, and duration mismatch."""
import os, sys
import numpy as np
import pandas as pd

DATA = r'C:\Users\user\Desktop\投資・不動産\nasdaq_backtest\data'

def load_yield(name):
    path = os.path.join(DATA, f'{name}_daily.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df.columns = ['yield_pct']
    return pd.to_numeric(df['yield_pct'], errors='coerce').ffill(limit=5)

dgs30 = load_yield('dgs30')
dgs10 = load_yield('dgs10')

# Build spliced yield series (same logic as corrected code)
y_full_30 = dgs30.reindex(dgs10.index.union(dgs30.index))
mask_no = y_full_30.isna()
y_full_30[mask_no] = dgs10[mask_no]
y_full_30 = y_full_30.ffill(limit=5)
y_dec = y_full_30 / 100.0

print("=== Issue A: Coupon /252 vs /365 ===")
# Build two bond NAVs with same duration, different coupon basis
def build_nav(y_dec_arr, duration, coupon_basis):
    nav = np.ones(len(y_dec_arr))
    clips = 0
    for i in range(1, len(y_dec_arr)):
        dy = y_dec_arr[i] - y_dec_arr[i-1]
        price_ret  = -duration * dy
        coupon_ret = y_dec_arr[i-1] / coupon_basis
        dr = price_ret + coupon_ret
        if abs(dr) > 0.20:
            clips += 1
        nav[i] = nav[i-1] * (1 + np.clip(dr, -0.20, 0.20))
    return nav, clips

n17_252, clips_17_252 = build_nav(y_dec.values, 17, 252)
n17_365, clips_17_365 = build_nav(y_dec.values, 17, 365)
n7_252,  clips_7_252  = build_nav(y_dec.values,  7, 252)

yrs = len(y_dec) / 252
cagr_17_252 = n17_252[-1]**(1/yrs) - 1
cagr_17_365 = n17_365[-1]**(1/yrs) - 1
cagr_7_252  = n7_252[-1]**(1/yrs)  - 1

print(f"  D=17, /252: CAGR={cagr_17_252*100:.2f}%  clips={clips_17_252}")
print(f"  D=17, /365: CAGR={cagr_17_365*100:.2f}%  clips={clips_17_365}")
print(f"  D=7,  /252: CAGR={cagr_7_252*100:.2f}%   clips={clips_7_252}")
print(f"  Coupon basis effect (17, 252vs365): {(cagr_17_252-cagr_17_365)*100:+.2f}%/yr CAGR")

print("\n=== Issue B: Clip frequency ===")
print(f"  Clips D=17 /252: {clips_17_252}")
print(f"  Clips D=7  /252: {clips_7_252}")

print("\n=== Issue C: Static D=17 vs time-varying for 1979-1982 ===")
# Compare D=17 vs D=9 for 1979-1982
mask_volcker = (y_full_30.index >= '1979-01-01') & (y_full_30.index <= '1982-12-31')
y_volcker = y_dec.values[mask_volcker]

def cum_return(y_arr, duration, coupon_basis=252):
    nav = 1.0
    for i in range(1, len(y_arr)):
        dy = y_arr[i] - y_arr[i-1]
        dr = np.clip(-duration*dy + y_arr[i-1]/coupon_basis, -0.20, 0.20)
        nav *= (1 + dr)
    return nav

cum_17 = cum_return(y_volcker, 17)
cum_9  = cum_return(y_volcker, 9)
cum_7  = cum_return(y_volcker, 7)
n_yrs_v = int(mask_volcker.sum()) / 252

print(f"  Volcker period 1979-1982 ({n_yrs_v:.1f} yrs):")
print(f"    D=17: cum return {(cum_17-1)*100:+.1f}%   CAGR {(cum_17**(1/n_yrs_v)-1)*100:+.2f}%/yr")
print(f"    D=9:  cum return {(cum_9-1)*100:+.1f}%    CAGR {(cum_9**(1/n_yrs_v)-1)*100:+.2f}%/yr")
print(f"    D=7:  cum return {(cum_7-1)*100:+.1f}%    CAGR {(cum_7**(1/n_yrs_v)-1)*100:+.2f}%/yr")
print(f"  Impact of D=17 vs D=9: {(cum_17-cum_9)*100:+.1f}% over period")

# Check 30yr bond duration formula at peak yields
print("\nApproximate modified duration for 20yr coupon bond at various yields:")
for ytm in [0.05, 0.08, 0.10, 0.15, 0.17]:
    T = 20; c = ytm  # coupon = yield for par bond
    # Macaulay duration: sum(t*cf*discount) / price
    times = np.arange(1, T+1)
    cfs = np.ones(T)*c; cfs[-1] += 1
    disc = (1+ytm)**(-times)
    mac_dur = (times * cfs * disc).sum() / (cfs * disc).sum()
    mod_dur = mac_dur / (1+ytm)
    print(f"  YTM={ytm*100:.0f}%: Mac.Dur={mac_dur:.2f}  Mod.Dur={mod_dur:.2f}")

print("\nDone.")
