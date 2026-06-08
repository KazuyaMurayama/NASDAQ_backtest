import pandas as pd, numpy as np
df = pd.read_csv(r'C:\Users\user\Desktop\投資・不動産\nasdaq_backtest\NASDAQ_extended_to_2026.csv')
close = pd.to_numeric(df['Close'], errors='coerce').dropna()
print(f"Rows: {len(close)}")
print(f"Start: {df['Date'].iloc[0]}  End: {df['Date'].iloc[-1]}")
print(f"First close: {close.iloc[0]:.2f}  Last close: {close.iloc[-1]:.2f}")
# Annual return stats to spot-check
ret = close.pct_change().dropna()
print(f"Daily ret mean: {ret.mean()*252*100:.2f}%/yr  std: {ret.std()*np.sqrt(252)*100:.2f}%/yr")
print(f"\nSample (2020-2024 range):")
mask = df['Date'].between('2020-01-01','2024-12-31')
print(df.loc[mask, ['Date','Close']].head(3).to_string())
print(df.loc[mask, ['Date','Close']].tail(3).to_string())

# Additional: known NDX price-only reference points
# NDX ~4069 on Jan 3, 2000; ~19000+ end of 2024
mask2000 = df['Date'].between('2000-01-01','2000-01-05')
mask2024 = df['Date'].between('2024-12-28','2024-12-31')
print(f"\nNDX ~Jan 2000: {df.loc[mask2000, 'Close'].values}")
print(f"NDX ~Dec 2024: {df.loc[mask2024, 'Close'].values}")
if len(df.loc[mask2000]) > 0 and len(df.loc[mask2024]) > 0:
    ratio = df.loc[mask2024, 'Close'].values[-1] / df.loc[mask2000, 'Close'].values[0]
    print(f"2000→2024 ratio: {ratio:.1f}x  (price-only NDX expected ~4.5-5x)")
