import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def load_data(ticker):
    """
    Fetches historical stock data for a given ticker from Yahoo Finance.
    
    Args:
        ticker (str): The stock symbol (e.g., 'AAPL', 'GOOGL').
        
    Returns:
        pd.DataFrame: A DataFrame containing the stock history.
    """
    print(f"Downloading data for {ticker}...")
    
    # Dynamic Date Range: Today back to 10 years ago
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

if __name__ == "__main__":
    # Test the function with Apple Inc.
    ticker_symbol = "AAPL"
    stock_data = load_data(ticker_symbol)
    
    if not stock_data.empty:
        print("\nSUCCESS: Data loaded!")
        print(f"Loaded {len(stock_data)} rows of data.")
        print("\nHere are the first 5 rows (The DataFrame):")
        print(stock_data.head())
    else:
        print("ERROR: No data found.")
