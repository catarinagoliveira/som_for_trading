import os
import yfinance as yf
import pandas as pd

def download_numerical_data(tickers, start_date, end_date, save_dir='data/raw'):
    os.makedirs(save_dir, exist_ok=True)
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                df.to_csv(f"{save_dir}/{ticker}_data.csv")
            else:
                print(f"No data for {ticker}.")
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")