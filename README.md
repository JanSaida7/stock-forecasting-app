# Stock Market Forecasting App 📈

A Machine Learning powered application that predicts stock prices using an LSTM (Long Short-Term Memory) Neural Network. Built with Python, TensorFlow, and Streamlit.

## 🚀 Features
### 1. **Interactive Dashboard**
   - **Company Selector**: Choose from popular tech giants (Apple, Tesla, Google, Microsoft, etc.) and Bitcoin.
   - **Professional Charts**: **Candlestick charts** with 1M, 6M, YTD, and 1Y range selectors.
   - **Interactive Date Filter**: Custom date range picker to zoom in on specific historical periods.
   - **Tomorrow's Prediction**: Shows a clear, predicted stock price for the next trading day.

### 2. **Advanced AI & Analytics**
   - **Multi-Model Support**: Toggle between deep learning (**LSTM**) and statistical baselines (**Linear Regression**).
   - **Model Retraining**: One-click **Retrain** button to update the model with the latest data directly from the UI.
   - **Trading Signals**: Actionable **BUY**, **SELL**, or **HOLD** signals based on predicted price movement.
   - **7-Day Forecast**: Recursive multi-step prediction to see the trend for the upcoming week.
   - **Profit Analysis**: "Buy & Hold" vs "Model Strategy" comparison to validate profitability.
   - **News Sentiment**: Real-time news headlines with **Sentiment Analysis** (Positive/Negative/Neutral) powered by TextBlob.

### 3. **Paper Trading Simulator** 💰
   - **Virtual Portfolio**: Start with a **$10,000** virtual balance.
   - **Practice Trading**: Buy and Sell stocks at real-time prices to test your strategies risk-free.
   - **Session Tracking**: Tracks your shares, cash balance, and total portfolio value (session-based).

### 4. **Data Transparency & UX**
   - **In-App Guide**: Built-in "How to Use" guide and helpful tooltips for every feature.
   - **Tabbed Layout**: Clean interface with separate tabs for Forecasts, Performance Metrics, and Raw Data.
   - **Accuracy Metrics**: Live **MAE** and **RMSE** scores for the selected period.
   - **Detailed Data**: View and **Export** historical predictions and errors as CSV.

## 🛠️ Tech Stack
- **Python**: Core programming language.
- **TensorFlow/Keras**: Deep Learning for the LSTM model.
- **Streamlit**: Frontend infrastructure.
- **yfinance**: Data acquisition.
- **Pandas & NumPy**: Data manipulation & Technical Indicators.
- **Plotly**: Interactive Candle & Line Charts.

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/JanSaida7/stock-forecasting-app.git
   cd stock-forecasting-app
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

1. **Train the Model** (Optional, if you want to retrain)
   ```bash
   python model.py
   ```
   *This saves the model as `stock_model.keras`.*

2. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 🧠 Model Details
- **Architecture**: LSTM (2 layers, 50 units each) + Dense Layers.
- **Features**: Close Price, RSI (14-day), EMA (50-day).
- **Sequence Length**: Looks back 100 days of multi-feature data to predict the next Close price.
- **Optimizer**: Adam.
- **Loss Function**: Mean Squared Error.

---
*Created by [JanSaida7](https://github.com/JanSaida7)*
