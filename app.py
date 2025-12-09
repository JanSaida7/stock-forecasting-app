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

# Define cached data loading function
@st.cache_data
def load_data_cached(ticker):
    return load_data(ticker)

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


# Session State for persisting data across reruns (e.g. when changing dates)
if 'run_forecast' not in st.session_state:
    st.session_state['run_forecast'] = False

if st.sidebar.button("Run Forecast"):
    st.session_state['run_forecast'] = True

if st.session_state['run_forecast']:
    st.subheader(f"Analyzing {ticker}...")
    
    # 1. Load Data
    with st.spinner("Downloading data..."):
        df = load_data_cached(ticker)
    
    if df.empty:
        st.error(f"Could not load data for {ticker}. Please check the ticker symbol.")
    else:

        # Date Filter
        st.sidebar.subheader("Date Range")
        min_date = df.index.min().date()
        max_date = df.index.max().date()

        start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

        if start_date > end_date:
            st.error("Start Date must be before End Date")
        else:
            # Filter data for display
            # Convert to datetime for comparison if needed, but pandas usually handles dates well
            filtered_df = df.loc[str(start_date):str(end_date)]
            
            # Display Summary
            st.write(f"### {selected_company} - Price History")
            st.dataframe(filtered_df.tail())
            
            # Extract Close Price properly (Handle MultiIndex)
            if isinstance(filtered_df.columns, pd.MultiIndex):
                close_price = filtered_df['Close'].iloc[:, 0]
            else:
                close_price = filtered_df['Close']
            
            # Plot Raw Data (Interactive with Plotly)
            fig = go.Figure()
            
            # Check if we have Open, High, Low, Close data for Candlestick
            # yfinance normally provides these
            if 'Open' in filtered_df.columns and 'High' in filtered_df.columns and 'Low' in filtered_df.columns:
                # Handle MultiIndex if present
                if isinstance(filtered_df.columns, pd.MultiIndex):
                    open_p = filtered_df['Open'].iloc[:, 0]
                    high_p = filtered_df['High'].iloc[:, 0]
                    low_p = filtered_df['Low'].iloc[:, 0]
                    close_p = filtered_df['Close'].iloc[:, 0]
                else:
                    open_p = filtered_df['Open']
                    high_p = filtered_df['High']
                    low_p = filtered_df['Low']
                    close_p = filtered_df['Close']
                    
                fig.add_trace(go.Candlestick(x=filtered_df.index,
                                open=open_p,
                                high=high_p,
                                low=low_p,
                                close=close_p,
                                name='Market Data'))
            else:
                # Fallback to Line chart if data is missing
                fig.add_trace(go.Scatter(x=filtered_df.index, y=close_price, mode='lines', name='Close Price'))
                
            fig.update_layout(
                title=f"{ticker} Historical Prices",
                yaxis_title='Price (USD)',
                xaxis_rangeslider_visible=False,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. Prepare Data for Model
            with st.spinner("Calculating Future Prices..."):
                # Scale data
                scaler, target_scaler, scaled_data = scale_data(df)
                
                # Load trained model
                try:
                    model = load_model('stock_model.keras')
                except:
                    st.error("Model not found! Please train the model first (Step 5).")
                    st.stop()
                
                # --- NEXT DAY PREDICTION ---
                # We need the last 100 days of data to predict the next 1 day
                max_idx = len(scaled_data) 
                # But we only need the VERY LAST 100 days for tomorrow's prediction
                # The filter shouldn't affect the model's ability to predict tomorrow, 
                # so we use the full 'scaled_data' here.
                
                seq_length = 100
                
                # Get the last 100 days from the scaled data
                last_100_days = scaled_data[-seq_length:]
                
                # Reshape for the model (1 sample, 100 time steps, 3 features)
                last_100_days = np.reshape(last_100_days, (1, seq_length, 3))
                
                # Predict
                prediction = model.predict(last_100_days)
                
                # Inverse Scale (Convert 0.xxxx back to dollars)
                predicted_price = target_scaler.inverse_transform(prediction)
                
                # Display Big Metric
                st.markdown("---")
                st.subheader("🔮 AI Prediction")
                st.metric(label="Predicted Close Price for Tomorrow", value=f"${predicted_price[0][0]:.2f}")
                st.markdown("---")

                # --- MODEL PERFORMANCE (Historical vs Predicted) ---
                st.write("### Model Performance (Filter Applied)")
                # Generate predictions for the visualization graph
                # Create sequences from the existing data
                x_test = []
                y_test = [] 
                
                for i in range(seq_length, len(scaled_data)):
                    x_test.append(scaled_data[i-seq_length:i])
                    y_test.append(scaled_data[i, 0])
                    
                x_test, y_test = np.array(x_test), np.array(y_test)
                # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) # No longer needed, already 3D
                
                predictions = model.predict(x_test)
                predictions = target_scaler.inverse_transform(predictions)
                y_test_scaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))
                
                # Filter predictions for display based on date
                pred_dates = df.index[seq_length:]
                
                # Create DataFrame for easier filtering
                pred_df = pd.DataFrame({
                    'Actual': y_test_scaled.flatten(),
                    'Predicted': predictions.flatten()
                }, index=pred_dates)
                
                filtered_pred_df = pred_df.loc[str(start_date):str(end_date)]
                
                # Plot Predictions vs Actual (Interactive with Plotly)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=filtered_pred_df.index, y=filtered_pred_df['Actual'], mode='lines', name='Actual Price', line=dict(color='blue')))
                fig2.add_trace(go.Scatter(x=filtered_pred_df.index, y=filtered_pred_df['Predicted'], mode='lines', name='AI Predicted Price', line=dict(color='red')))
                fig2.update_layout(title=f"{ticker} - Actual vs Predicted", xaxis_title='Time', yaxis_title='Price (USD)')
                st.plotly_chart(fig2, use_container_width=True)

                # --- LAST 10 DAYS PERFORMANCE (New Feature) ---
                st.markdown("---")
                st.subheader("🔍 Model Performance (Last 10 Days)")
                
                # Get last 10 days of data
                if len(y_test_scaled) >= 10:
                    y_last_10 = y_test_scaled[-10:].flatten()
                    pred_last_10 = predictions[-10:].flatten()
                    
                    # Create a date range for these points (Assuming consistent daily data ending mostly recently)
                    # We trace back from the last available date in our dataframe
                    last_date_idx = df.index[-1]
                    dates_last_10 = df.index[-10:] 

                    # 1. Metrics for Last 10 Days
                    mae_10 = mean_absolute_error(y_last_10, pred_last_10)
                    rmse_10 = np.sqrt(mean_squared_error(y_last_10, pred_last_10))

                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Last 10 Days MAE", f"${mae_10:.2f}")
                    m_col2.metric("Last 10 Days RMSE", f"${rmse_10:.2f}")

                    # 2. Specific Chart for Last 10 Days
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(x=dates_last_10, y=y_last_10, mode='lines+markers', name='Actual Price', line=dict(color='blue')))
                    fig3.add_trace(go.Scatter(x=dates_last_10, y=pred_last_10, mode='lines+markers', name='Predicted Price', line=dict(color='red', dash='dash')))
                    fig3.update_layout(title="Last 10 Days: Actual vs Predicted", xaxis_title='Date', yaxis_title='Price (USD)')
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # 3. Data Table
                    comparison_df = pd.DataFrame({
                        'Date': dates_last_10,
                        'Actual Price': y_last_10,
                        'Predicted Price': pred_last_10,
                        'Difference': y_last_10 - pred_last_10,
                        'Error %': np.abs((y_last_10 - pred_last_10) / y_last_10) * 100
                    })
                    comparison_df.set_index('Date', inplace=True)
                    st.write("### Detailed Data (Last 10 Days)")
                    st.dataframe(comparison_df.style.format("{:.2f}"))
                    
                    # Export Button
                    csv = comparison_df.to_csv()
                    st.download_button(
                        label="📥 Download Data as CSV",
                        data=csv,
                        file_name=f'{ticker}_last_10_days_prediction.csv',
                        mime='text/csv',
                    )
                else:
                    st.warning("Not enough data to show last 10 days verification.")

                # --- ACCURACY METRICS ---
                st.subheader("Accuracy Metrics")
                mae = mean_absolute_error(y_test_scaled, predictions)
                rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
                
                col1, col2 = st.columns(2)
                col1.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
                col2.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")
