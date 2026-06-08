"""Compare prepare_bond_data() vs build_bond_1x_nav_baseline() output."""
import sys, os, types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.append(r'C:\Users\user\Desktop\投資・不動産\nasdaq_backtest\src')

import numpy as np
import pandas as pd
from backtest_engine import load_data
from test_portfolio_diversification import prepare_bond_data
from corrected_strategy_backtest import build_bond_1x_nav_baseline

BASE = r'C:\Users\user\Desktop\投資・不動産\nasdaq_backtest'
df = load_data(os.path.join(BASE, 'NASDAQ_extended_to_2026.csv'))
dates = df['Date']

print("Building bond NAVs...")
nav_orig = prepare_bond_data(dates)           # original
nav_new  = build_bond_1x_nav_baseline(dates)  # new

s_orig = pd.Series(nav_orig, index=range(len(nav_orig)))
s_new  = pd.Series(nav_new,  index=range(len(nav_new)))

# Compare
diff = s_new - s_orig
print(f"\nFirst date: {dates.iloc[0].date()}  Last: {dates.iloc[-1].date()}")
print(f"nav_orig final: {s_orig.iloc[-1]:.6f}")
print(f"nav_new  final: {s_new.iloc[-1]:.6f}")
print(f"Max abs diff:   {diff.abs().max():.8f}")
print(f"Mean abs diff:  {diff.abs().mean():.8f}")
print(f"Days with diff > 1e-6: {(diff.abs() > 1e-6).sum()}")

# Show first few divergence points
mask = diff.abs() > 1e-6
if mask.any():
    idx = mask.idxmax()
    print(f"\nFirst divergence at row {idx}, date {dates.iloc[idx].date()}")
    print(f"  orig: {s_orig.iloc[idx]:.8f}  new: {s_new.iloc[idx]:.8f}")
    print("Context rows around divergence (orig vs new):")
    for j in range(max(0,idx-2), min(len(s_orig), idx+5)):
        print(f"  row {j} ({dates.iloc[j].date()}): orig={s_orig.iloc[j]:.8f}  new={s_new.iloc[j]:.8f}  diff={s_new.iloc[j]-s_orig.iloc[j]:.2e}")

print("\nDone.")
