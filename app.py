import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model

# Import our helper functions
from data_loader import load_data
from preprocess import scale_data, plot_data

# Set page configuration
st.set_page_config(page_title="Stock Forecaster", layout="wide")

st.title("📈 Stock Market Forecasting App")

# Sidebar for user input
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")

if st.sidebar.button("Run Forecast"):
    st.subheader(f"Analyzing {ticker}...")
    
    # 1. Load Data
    with st.spinner("Downloading data..."):
        df = load_data(ticker)
    
    if df.empty:
        st.error(f"Could not load data for {ticker}. Please check the ticker symbol.")
    else:
        # Display Summary
        st.write(f"### {ticker} Stock Price History (2020-2024)")
        st.write(df.tail())
        
        # Plot Raw Data
        st.subheader("Closing Price vs Time Chart")
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Close Price')
        plt.title(f"{ticker} Close Price")
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        st.pyplot(fig)
        
        # 2. Prepare Data for Model
        st.subheader("Model Predictions")
        with st.spinner("Loading Model & Calculating..."):
            # Scale data
            scaler, scaled_data = scale_data(df)
            
            # Load trained model
            try:
                model = load_model('stock_model.keras')
            except:
                st.error("Model not found! Please train the model first (Step 5).")
                st.stop()
                
            # Prepare test data (Past 60 days to predict)
            # We want to see how well it fits the existing data first
            # We used 60 days as sequence length
            seq_length = 60
            x_test = []
            y_test = [] # Actual values to compare against
            
            # Create sequences from the entire dataset to visualize fit
            for i in range(seq_length, len(scaled_data)):
                x_test.append(scaled_data[i-seq_length:i, 0])
                y_test.append(scaled_data[i, 0])
                
            x_test, y_test = np.array(x_test), np.array(y_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            # Make Predictions
            predictions = model.predict(x_test)
            
            # Inverse Scale (Convert 0-1 back to USD)
            predictions = scaler.inverse_transform(predictions)
            y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Plot Predictions vs Actual
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(y_test_scaled, color='blue', label='Actual Price')
            plt.plot(predictions, color='red', label='Predicted Price')
            plt.title(f"{ticker} Price Prediction")
            plt.xlabel('Time')
            plt.ylabel('Price (USD)')
            plt.legend()
            st.pyplot(fig2)
            
            st.success("Analysis Complete!")
