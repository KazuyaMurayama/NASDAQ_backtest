"""
Extend NASDAQ data to 2026 by merging original CSV with Yahoo Finance data.
"""
import pandas as pd
import yfinance as yf
import os

def extend_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_path = os.path.join(base_dir, 'NASDAQ_Dairy_since1973.csv')
    output_path = os.path.join(base_dir, 'NASDAQ_extended_to_2026.csv')

    # Load original data
    orig = pd.read_csv(original_path, parse_dates=['Date'])
    orig = orig.sort_values('Date').reset_index(drop=True)
    last_date = orig['Date'].iloc[-1]
    print(f"Original data: {orig['Date'].iloc[0].date()} to {last_date.date()} ({len(orig)} rows)")

    # Fetch new data from Yahoo Finance
    ticker = yf.Ticker('^IXIC')
    new = ticker.history(start=last_date.strftime('%Y-%m-%d'), end='2026-03-28')
    new = new.reset_index()
    new['Date'] = pd.to_datetime(new['Date']).dt.tz_localize(None)

    # Rename columns to match original format
    new = new.rename(columns={'Stock Splits': 'Stock_Splits'})
    new_formatted = pd.DataFrame({
        'Date': new['Date'],
        'Open': new['Open'],
        'High': new['High'],
        'Low': new['Low'],
        'Close': new['Close'],
        'Adj Close': new['Close'],  # Use Close as Adj Close for index
        'Volume': new['Volume']
    })

    # Remove overlap (keep only rows after the last original date)
    new_formatted = new_formatted[new_formatted['Date'] > last_date]
    print(f"New data: {len(new_formatted)} additional rows")

    # Combine
    combined = pd.concat([orig, new_formatted], ignore_index=True)
    combined = combined.sort_values('Date').reset_index(drop=True)
    combined.to_csv(output_path, index=False)

    print(f"Combined data: {combined['Date'].iloc[0].date()} to {combined['Date'].iloc[-1].date()} ({len(combined)} rows)")
    print(f"Saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    extend_data()
