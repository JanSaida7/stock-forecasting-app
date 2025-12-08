import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_data

def plot_data(df, ticker):
    """
    Plots the closing price of the stock.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label=f'{ticker} Close Price')
    plt.title(f'{ticker} Stock Price History')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    print("Close the plot window to continue...")
    plt.show()

def scale_data(df):
    """
    Scales the 'Close' price column to be between 0 and 1.
    Returns the scaler object and the scaled data.
    """
    # yfinance returns a MultiIndex (Price, Ticker)
    # df['Close'] returns the Close prices (as a DataFrame or Series)
    data = df['Close']
    
    # Ensure data is 2D for the scaler (rows, 1)
    dataset = data.values
    if len(dataset.shape) == 1:
        dataset = dataset.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    return scaler, scaled_data

if __name__ == "__main__":
    ticker = "AAPL"
    df = load_data(ticker)
    
    if not df.empty:
        # 1. Visualize
        print("Plotting data...")
        plot_data(df, ticker)
        
        # 2. Scale
        print("Scaling data...")
        scaler, scaled_data = scale_data(df)
        
        print("\nData Scaled successfully!")
        print(f"Original value (First row): {df['Close'].iloc[0]}")
        print(f"Scaled value (First row): {scaled_data[0]}")
