import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from textblob import TextBlob
import yfinance as yf

# Import our helper functions
from data_loader import load_data
from preprocess import scale_data
from model import train_universal_model # Import training logic

# Define cached data loading function
@st.cache_data
def load_data_cached(ticker):
    return load_data(ticker)

# Fetch News function
# @st.cache_data(ttl=3600) # CMC: Disabled for debugging
def get_stock_news(ticker):
    # st.write(f"Debug: Fetching news for {ticker}")
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        
        processed_news = []
        if news_list:
            for item in news_list[:5]: # Top 5 news
                # Handle nested content structure usually found in yfinance
                content = item.get('content', item) # Fallback to item if not nested
                
                title = content.get('title', '')
                # Attempt to find link in various fields
                link = '#'
                candidates = [
                    content.get('clickThroughUrl'),
                    content.get('canonicalUrl'),
                    content.get('link')
                ]
                
                for candidate in candidates:
                    if candidate:
                        if isinstance(candidate, dict):
                            found_url = candidate.get('url')
                            if found_url:
                                link = found_url
                                break
                        elif isinstance(candidate, str):
                            link = candidate
                            break
                
                # Provider/Publisher might be a dict or string
                pub_data = content.get('provider', {})
                if isinstance(pub_data, dict):
                    publisher = pub_data.get('displayName', 'Unknown')
                else:
                    publisher = str(pub_data)
                
                # Sentiment Analysis
                analysis = TextBlob(title)
                polarity = analysis.sentiment.polarity
                
                if polarity > 0.1:
                    sentiment = "Positive"
                    color = "green"
                elif polarity < -0.1:
                    sentiment = "Negative"
                    color = "red"
                else:
                    sentiment = "Neutral"
                    color = "gray"
                    
                processed_news.append({
                    'title': title,
                    'link': link,
                    'publisher': publisher,
                    'sentiment': sentiment,
                    'color': color,
                    'score': polarity
                })
        return processed_news
    except Exception as e:
        return []

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

if 'balance' not in st.session_state:
    st.session_state['balance'] = 10000.0
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = {} # Dict to store shares per ticker e.g. {'AAPL': 10}

# Set page configuration
st.set_page_config(page_title="Stock Forecaster", layout="wide")

st.title("📈 Stock Market Forecasting App")

# Sidebar for user input
st.sidebar.header("User Input")

# --- APP GUIDE ---
with st.sidebar.expander("ℹ️ How to use", expanded=False):
    st.markdown("""
    1. **Select Company**: Choose a stock from the dropdown.
    2. **Choose Model**:
        * *LSTM*: Deep Learning (slower, more complex).
        * *Linear Regression*: Fast baseline.
    3. **Run Forecast**: Click to fetch data and predict.
    4. **Paper Trading**: Buy/Sell virtual stocks to test strategies.
    """)

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

selected_company = st.sidebar.selectbox(
    "Select Company", 
    list(companies.keys()), 
    help="Choose the stock ticker you want to analyze."
)
ticker = companies[selected_company]

# Model Selection
model_type = st.sidebar.radio(
    "Select Model Type", 
    ["LSTM (Deep Learning)", "Linear Regression (Baseline)"],
    help="LSTM: Good for complex patterns.\nLinear Regression: Good for trends."
)

# Retrain Button (Only for LSTM)
if "LSTM" in model_type:
    if st.sidebar.button("🔄 Retrain LSTM Model", help="Click to retrain the model with the latest data (approx 2-5 mins)."):
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

if st.sidebar.button("Run Forecast", type="primary", help="Fetch data and generate predictions."):
    st.session_state['run_forecast'] = True

if st.session_state['run_forecast']:
    st.subheader(f"Analyzing {ticker}...")
    
    # 1. Load Data
    with st.spinner("Downloading data..."):
        df = load_data_cached(ticker)
    
    if df.empty:
        st.error(f"Could not load data for {ticker}. Please check the ticker symbol.")
    else:

        display_df = df.reset_index()
        display_df['Date'] = display_df['Date'].dt.date
        
        # --- PRE-CALCULATE LATEST PRICE ---
        # Handle MultiIndex for Last Actual Price
        if isinstance(df.columns, pd.MultiIndex):
            last_actual_price = float(df['Close'].iloc[-1, 0])
        else:
            last_actual_price = float(df['Close'].iloc[-1])
            
        # --- PAPER TRADING SIMULATOR (Sidebar) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 Paper Trading")
        
        # 1. Dashboard
        current_shares = st.session_state['portfolio'].get(ticker, 0)
        total_balance = st.session_state['balance']
        position_value = current_shares * last_actual_price
        
        st.sidebar.metric("Cash Balance", f"${total_balance:,.2f}")
        st.sidebar.metric(f"Shares ({ticker})", f"{current_shares}")
        st.sidebar.metric("Position Value", f"${position_value:,.2f}")
        
        # 2. Trade Controls
        trade_qty = st.sidebar.number_input("Quantity", min_value=1, value=1, step=1)
        
        col_buy, col_sell = st.sidebar.columns(2)
        
        with col_buy:
            if st.button("Buy", type="primary"):
                cost = trade_qty * last_actual_price
                if total_balance >= cost:
                    st.session_state['balance'] -= cost
                    st.session_state['portfolio'][ticker] = current_shares + trade_qty
                    st.sidebar.success(f"Bought {trade_qty} {ticker}!")
                    st.rerun()
                else:
                    st.sidebar.error("Insufficient Funds!")
                    
        with col_sell:
            if st.button("Sell"):
                if current_shares >= trade_qty:
                    revenue = trade_qty * last_actual_price
                    st.session_state['balance'] += revenue
                    st.session_state['portfolio'][ticker] = current_shares - trade_qty
                    st.sidebar.success(f"Sold {trade_qty} {ticker}!")
                    st.rerun()
                else:
                    st.sidebar.error("Not enough shares!")
        
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
                    st.metric(
                        label="Predicted Close Price for Tomorrow", 
                        value=f"${predicted_price[0][0]:.2f}",
                        help="The AI's best guess for the stock's closing price on the next trading day."
                    )
                    
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
                        
                
                # --- TABS SETUP ---
                tab1, tab2, tab3 = st.tabs(["📅 Forecast & News", "📊 Model Performance", "📥 Raw Data"])
                
                with tab1:
                    # --- 7-DAY FORECAST (Recursive) ---
                    st.subheader("📅 7-Day Forecast (Experimental)")
                    st.info("Note: Recursive forecasting uses the model's own predictions as input for the next day. Accuracy may decrease over longer horizons.")
                    
                    future_days = 7
                    future_predictions = []
                    
                    # We start with the same last_100_days 
                    current_batch = scaled_data[-seq_length:].copy() 
                    
                    for i in range(future_days):
                        # 1. Prepare input
                        if "LSTM" in model_type:
                            input_feed = np.reshape(current_batch, (1, seq_length, 3))
                        else:
                            input_feed = current_batch.flatten().reshape(1, -1)
                            
                        # 2. Predict next step
                        next_pred_scaled = model.predict(input_feed)[0] 
                        
                        # Ensure scalar/1D extraction strictly
                        if isinstance(next_pred_scaled, np.ndarray):
                            val_0 = next_pred_scaled.flatten()[0]
                        else:
                            val_0 = next_pred_scaled
                        
                        # 3. Store Result (Inverse Scale)
                        val_inv = target_scaler.inverse_transform([[val_0]])[0][0]
                        future_predictions.append(val_inv)
                        
                        # 4. Update Batch
                        last_row = current_batch[-1]
                        new_row = np.array([val_0, last_row[1], last_row[2]]) # [NewPrice, OldRSI, OldEMA]
                        
                        current_batch = np.vstack([current_batch[1:], new_row])
                        
                    # Display 7-Day Results
                    dates_future = [datetime.today() + timedelta(days=i+1) for i in range(future_days)]
                    
                    # 1. Chart
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(x=[df.index[-1]] + dates_future, 
                                                    y=[last_actual_price] + future_predictions, 
                                                    mode='lines+markers', 
                                                    name='7-Day Forecast',
                                                    line=dict(color='purple', dash='dot')))
                                                    
                    fig_forecast.update_layout(title="7-Day Price Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # 2. Table
                    forecast_df = pd.DataFrame({
                        "Date": [d.strftime('%Y-%m-%d') for d in dates_future],
                        "Predicted Price": [f"${p:.2f}" for p in future_predictions]
                    })
                    st.table(forecast_df)
                            
                    st.markdown("---")
                    
                    # --- NEWS & SENTIMENT ---
                    st.subheader("📰 Recent News & Sentiment")
                    news_items = get_stock_news(ticker)
                    
                    if news_items:
                        for news in news_items:
                            st.markdown(f"""
                            <a href='{news['link']}' target='_blank' style='text-decoration: none; color: inherit;'>
                                <div style='
                                    padding: 15px; 
                                    border-radius: 8px; 
                                    border: 1px solid #e0e0e0; 
                                    margin-bottom: 10px; 
                                    transition: background-color 0.2s; 
                                    background-color: transparent;'
                                    onmouseover="this.style.backgroundColor='#f0f2f6';" 
                                    onmouseout="this.style.backgroundColor='transparent';">
                                    
                                    <b style='font-size: 1.1em;'>{news['title']}</b><br>
                                    <div style='margin-top: 5px; font-size: 0.9em; color: gray;'>
                                        <span>{news['publisher']}</span> • 
                                        <span style='color: {news['color']}; font-weight: bold;'>{news['sentiment']}</span>
                                    </div>
                                </div>
                            </a>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("No recent news found.")

                with tab2:
                    # --- MODEL PERFORMANCE (Historical vs Predicted) ---
                    st.write("### Model Performance (Filter Applied)")
                    
                    # Generate predictions for the visualization graph
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
                    if predictions.ndim == 1:
                        predictions = predictions.reshape(-1, 1)
                    
                    predictions = target_scaler.inverse_transform(predictions)
                    y_test_scaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Filter predictions for display based on date
                    pred_dates = df.index[seq_length:]
                    
                    pred_df = pd.DataFrame({
                        'Actual': y_test_scaled.flatten(),
                        'Predicted': predictions.flatten()
                    }, index=pred_dates)
                    
                    filtered_pred_df = pred_df.loc[str(start_date):str(end_date)]
                    
                    # Plot Predictions vs Actual
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=filtered_pred_df.index, y=filtered_pred_df['Actual'], mode='lines', name='Actual Price', line=dict(color='blue')))
                    fig2.add_trace(go.Scatter(x=filtered_pred_df.index, y=filtered_pred_df['Predicted'], mode='lines', name='AI Predicted Price', line=dict(color='red')))
                    fig2.update_layout(title=f"{ticker} - Actual vs Predicted", xaxis_title='Time', yaxis_title='Price (USD)')
                    st.plotly_chart(fig2, use_container_width=True)
    
                    # --- LAST 10 DAYS PERFORMANCE ---
                    st.markdown("---")
                    st.subheader("🔍 Model Performance (Last 10 Days)")
                    
                    if len(y_test_scaled) >= 10:
                        y_last_10 = y_test_scaled[-10:].flatten()
                        pred_last_10 = predictions[-10:].flatten()
                        dates_last_10 = df.index[-10:] 
    
                        # Metrics
                        mae_10 = mean_absolute_error(y_last_10, pred_last_10)
                        rmse_10 = np.sqrt(mean_squared_error(y_last_10, pred_last_10))
    
                        m_col1, m_col2 = st.columns(2)
                        m_col1.metric("Last 10 Days MAE", f"${mae_10:.2f}")
                        m_col2.metric("Last 10 Days RMSE", f"${rmse_10:.2f}")
    
                        # Chart
                        fig3 = go.Figure()
                        fig3.add_trace(go.Scatter(x=dates_last_10, y=y_last_10, mode='lines+markers', name='Actual Price', line=dict(color='blue')))
                        fig3.add_trace(go.Scatter(x=dates_last_10, y=pred_last_10, mode='lines+markers', name='Predicted Price', line=dict(color='red', dash='dash')))
                        fig3.update_layout(title="Last 10 Days: Actual vs Predicted", xaxis_title='Date', yaxis_title='Price (USD)')
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        # --- PROFIT ANALYSIS ---
                        st.markdown("---")
                        st.subheader("💰 Profit Analysis (Model vs Buy & Hold)")
                        
                        profit_df = filtered_pred_df.copy()
                        profit_df['Daily_Return'] = profit_df['Actual'].pct_change()
                        profit_df['Prev_Actual'] = profit_df['Actual'].shift(1)
                        profit_df['Signal'] = np.where(profit_df['Predicted'] > profit_df['Prev_Actual'], 1, 0)
                        profit_df['Strategy_Return'] = profit_df['Signal'].shift(1) * profit_df['Daily_Return']
                        profit_df['Buy_Hold_Cum'] = (1 + profit_df['Daily_Return']).cumprod() - 1
                        profit_df['Strategy_Cum'] = (1 + profit_df['Strategy_Return']).cumprod() - 1
                        profit_df.fillna(0, inplace=True)
                        
                        fig_profit = go.Figure()
                        fig_profit.add_trace(go.Scatter(x=profit_df.index, y=profit_df['Buy_Hold_Cum']*100, mode='lines', name='Buy & Hold (%)', line=dict(color='blue')))
                        fig_profit.add_trace(go.Scatter(x=profit_df.index, y=profit_df['Strategy_Cum']*100, mode='lines', name='Model Strategy (%)', line=dict(color='green')))
                        fig_profit.update_layout(title="Cumulative Return Comparison", xaxis_title='Date', yaxis_title='Return (%)')
                        st.plotly_chart(fig_profit, use_container_width=True)
                        
                        p_col1, p_col2 = st.columns(2)
                        p_col1.metric("Total Buy & Hold Return", f"{profit_df['Buy_Hold_Cum'].iloc[-1] * 100:.2f}%")
                        p_col2.metric("Total Model Return", f"{profit_df['Strategy_Cum'].iloc[-1] * 100:.2f}%")
                    else:
                        st.warning("Not enough data to show last 10 days verification.")
    
                    # --- ACCURACY METRICS ---
                    st.subheader("Accuracy Metrics")
                    mae = mean_absolute_error(y_test_scaled, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
                    col2.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")

                with tab3:
                    if len(y_test_scaled) >= 10:
                        # 3. Data Table (Last 10 Days)
                         # Review: Re-creating comparison df as it was inside the if block earlier
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
                        st.write("No detailed data available.")
