import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import our helper functions
from data_loader import load_data
from preprocess import scale_data
from model import train_universal_model # Import training logic

# Define cached data loading function
@st.cache_data
def load_data_cached(ticker):
    return load_data(ticker)

# Train Linear Regression Model (On-the-fly)
@st.cache_resource
def train_lr_model(scaled_data):
    # Prepare data for LR (Flattened input)
    # Use existing sequence length of 100
    seq_length = 100
    x_train = []
    y_train = []
    
    for i in range(seq_length, len(scaled_data)):
        x_train.append(scaled_data[i-seq_length:i].flatten()) # Flatten (100, 3) -> (300,)
        y_train.append(scaled_data[i, 0])      # Close price
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Train
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    return lr_model

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

# Model Selection
model_type = st.sidebar.radio("Select Model Type", ["LSTM (Deep Learning)", "Linear Regression (Baseline)"])

# Retrain Button (Only for LSTM)
if "LSTM" in model_type:
    if st.sidebar.button("🔄 Retrain LSTM Model"):
        with st.spinner("Retraining Universal LSTM Model... This may take 2-5 minutes."):
            success, message = train_universal_model(epochs=5) # 5 epochs for speed in demo, can be 10
            if success:
                st.success(message)
                # Clear cache to ensure model reloads
                st.cache_resource.clear()
            else:
                st.error(f"Training Failed: {message}")

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
                
                # --- MODEL SELECTION LOGIC ---
                model = None
                
                if "LSTM" in model_type:
                    # Load trained LSTM model
                    try:
                        model = load_model('stock_model.keras')
                    except:
                        st.error("LSTM Model not found! Please train the model first.")
                        st.stop()
                else:
                    # Train Linear Regression on-the-fly
                    model = train_lr_model(scaled_data)
                    st.success("Trained Linear Regression Baseline!")
                
                # --- PREDICTION PREP ---
                seq_length = 100
                
                # Get the last 100 days from the scaled data
                last_100_days = scaled_data[-seq_length:]
                
                # Prepare input shape based on model type
                if "LSTM" in model_type:
                    # Reshape for LSTM (1 sample, 100 time steps, 3 features)
                    input_data = np.reshape(last_100_days, (1, seq_length, 3))
                else:
                    # Flatten for LR (1 sample, 300 features)
                    input_data = last_100_days.flatten().reshape(1, -1)
                
                # Predict
                prediction = model.predict(input_data)
                
                # Inverse Scale (Convert 0.xxxx back to dollars)
                predicted_price = target_scaler.inverse_transform(prediction.reshape(-1, 1))
                
                # Display Big Metric
                st.markdown("---")
                st.subheader("🔮 AI Prediction")
                
                col_pred, col_signal = st.columns(2)
                
                with col_pred:
                    st.metric(label="Predicted Close Price for Tomorrow", value=f"${predicted_price[0][0]:.2f}")
                    
                with col_signal:
                    # Calculate Signal
                    # Handle MultiIndex for Last Actual Price
                    if isinstance(df.columns, pd.MultiIndex):
                        last_actual_price = float(df['Close'].iloc[-1, 0])
                    else:
                        last_actual_price = float(df['Close'].iloc[-1])
                        
                    predicted_value = predicted_price[0][0]
                    
                    if predicted_value > last_actual_price:
                        st.markdown(f"<h2 style='color: green;'>✅ BUY SIGNAL</h2>", unsafe_allow_html=True)
                        st.write(f"Predicted to RISE from ${last_actual_price:.2f}")
                    elif predicted_value < last_actual_price:
                        st.markdown(f"<h2 style='color: red;'>🔻 SELL SIGNAL</h2>", unsafe_allow_html=True)
                        st.write(f"Predicted to FALL from ${last_actual_price:.2f}")
                    else:
                        st.markdown(f"<h2 style='color: gray;'>⏸️ HOLD SIGNAL</h2>", unsafe_allow_html=True)
                        st.write(f"Predicted to STAY FLAT at ${last_actual_price:.2f}")
                        
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
                
                # Reshape if Linear Regression
                if "Linear Regression" in model_type:
                    x_test = x_test.reshape(x_test.shape[0], -1)
                
                predictions = model.predict(x_test)
                # Reshape for inverse transform (Scaler expects 2D)
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
                
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
                    
                    # --- PROFIT ANALYSIS (Baseline Comparison) ---
                    st.markdown("---")
                    st.subheader("💰 Profit Analysis (Model vs Buy & Hold)")
                    
                    # 1. Calculate Daily Returns
                    profit_df = filtered_pred_df.copy()
                    profit_df['Daily_Return'] = profit_df['Actual'].pct_change()
                    
                    # 2. Signal: Buy (1) if Predicted > Previous Actual
                    profit_df['Prev_Actual'] = profit_df['Actual'].shift(1)
                    profit_df['Signal'] = np.where(profit_df['Predicted'] > profit_df['Prev_Actual'], 1, 0)
                    
                    # 3. Strategy Return
                    profit_df['Strategy_Return'] = profit_df['Signal'].shift(1) * profit_df['Daily_Return']
                    
                    # 4. Cumulative Returns
                    profit_df['Buy_Hold_Cum'] = (1 + profit_df['Daily_Return']).cumprod() - 1
                    profit_df['Strategy_Cum'] = (1 + profit_df['Strategy_Return']).cumprod() - 1
                    
                    profit_df.fillna(0, inplace=True)
                    
                    # Plot
                    fig_profit = go.Figure()
                    fig_profit.add_trace(go.Scatter(x=profit_df.index, y=profit_df['Buy_Hold_Cum']*100, mode='lines', name='Buy & Hold (%)', line=dict(color='blue')))
                    fig_profit.add_trace(go.Scatter(x=profit_df.index, y=profit_df['Strategy_Cum']*100, mode='lines', name='Model Strategy (%)', line=dict(color='green')))
                    fig_profit.update_layout(title="Cumulative Return Comparison (Selected Period)", xaxis_title='Date', yaxis_title='Return (%)')
                    st.plotly_chart(fig_profit, use_container_width=True)
                    
                    # Metrics
                    total_buy_hold = profit_df['Buy_Hold_Cum'].iloc[-1] * 100
                    total_strategy = profit_df['Strategy_Cum'].iloc[-1] * 100
                    
                    p_col1, p_col2 = st.columns(2)
                    p_col1.metric("Total Buy & Hold Return", f"{total_buy_hold:.2f}%")
                    p_col2.metric("Total Model Return", f"{total_strategy:.2f}%", delta=f"{total_strategy-total_buy_hold:.2f}%")
                    
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
