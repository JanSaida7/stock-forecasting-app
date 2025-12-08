import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import our helper functions
from data_loader import load_data
from preprocess import scale_data

# Set page configuration
st.set_page_config(page_title="Stock Forecaster", layout="wide")

st.title("📈 Stock Market Forecasting App")

# Sidebar for user input
st.sidebar.header("User Input")

# Dropdown for Company Selection
companies = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Google (GOOGL)": "GOOGL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Nvidia (NVDA)": "NVDA",
    "Meta (META)": "META",
    "Bitcoin (BTC-USD)": "BTC-USD"
}

selected_company = st.sidebar.selectbox("Select Company", list(companies.keys()))
ticker = companies[selected_company]

if st.sidebar.button("Run Forecast"):
    st.subheader(f"Analyzing {ticker}...")
    
    # 1. Load Data
    with st.spinner("Downloading data..."):
        df = load_data(ticker)
    
    if df.empty:
        st.error(f"Could not load data for {ticker}. Please check the ticker symbol.")
    else:
        # Display Summary
        st.write(f"### {selected_company} - Price History")
        st.dataframe(df.tail())
        
        # Plot Raw Data (Interactive with Plotly)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title=f"{ticker} Close Price History", xaxis_title='Date', yaxis_title='Price (USD)')
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Prepare Data for Model
        with st.spinner("Calculating Future Prices..."):
            # Scale data
            scaler, scaled_data = scale_data(df)
            
            # Load trained model
            try:
                model = load_model('stock_model.keras')
            except:
                st.error("Model not found! Please train the model first (Step 5).")
                st.stop()
            
            # --- NEXT DAY PREDICTION ---
            # We need the last 100 days of data to predict the next 1 day
            seq_length = 100
            
            # Get the last 100 days from the scaled data
            last_100_days = scaled_data[-seq_length:]
            
            # Reshape for the model (1 sample, 100 time steps, 1 feature)
            last_100_days = np.reshape(last_100_days, (1, seq_length, 1))
            
            # Predict
            prediction = model.predict(last_100_days)
            
            # Inverse Scale (Convert 0.xxxx back to dollars)
            predicted_price = scaler.inverse_transform(prediction)
            
            # Display Big Metric
            st.markdown("---")
            st.subheader("🔮 AI Prediction")
            st.metric(label="Predicted Close Price for Tomorrow", value=f"${predicted_price[0][0]:.2f}")
            st.markdown("---")

            # --- MODEL PERFORMANCE (Historical vs Predicted) ---
            st.write("### Model Performance (Test on Past Data)")
            # Generate predictions for the visualization graph
            # Create sequences from the existing data
            x_test = []
            y_test = [] 
            
            for i in range(seq_length, len(scaled_data)):
                x_test.append(scaled_data[i-seq_length:i, 0])
                y_test.append(scaled_data[i, 0])
                
            x_test, y_test = np.array(x_test), np.array(y_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Plot Predictions vs Actual (Interactive with Plotly)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=y_test_scaled.flatten(), mode='lines', name='Actual Price', line=dict(color='blue')))
            fig2.add_trace(go.Scatter(y=predictions.flatten(), mode='lines', name='AI Predicted Price', line=dict(color='red')))
            fig2.update_layout(title=f"{ticker} - Actual vs Predicted", xaxis_title='Time', yaxis_title='Price (USD)')
            st.plotly_chart(fig2, use_container_width=True)

            # --- ACCURACY METRICS ---
            st.subheader("Accuracy Metrics")
            mae = mean_absolute_error(y_test_scaled, predictions)
            rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
            
            col1, col2 = st.columns(2)
            col1.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
            col2.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")
