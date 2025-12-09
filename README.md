# Stock Market Forecasting App 📈

A Machine Learning powered application that predicts stock prices using an LSTM (Long Short-Term Memory) Neural Network. Built with Python, TensorFlow, and Streamlit.

## 🚀 Features
### 1. **Interactive Dashboard**
   - **Company Selector**: Choose from popular tech giants (Apple, Tesla, Google, Microsoft, etc.).
   - **Professional Charts**: **Candlestick charts** with 1M, 6M, YTD, and 1Y range selectors for deep analysis.
   - **Tomorrow's Prediction**: Shows a clear, predicted stock price for the next trading day.

### 2. **Advanced AI Model**
   - **Multi-Feature Learning**: Trained not just on Price, but also on **RSI (14)** and **EMA (50)** technical indicators.
   - **Generalist Training**: Learn patterns from **10 Years** of combined data (AAPL, MSFT, GOOGL, AMZN).
   - **Robust Architecture**: Uses a multi-input **LSTM (Long Short-Term Memory)** neural network.
   - **Performance**: Instant load times thanks to **Smart Caching**.

### 3. **Accuracy Reporting**
   - **Scoreboard**: Displays **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error) to quantify performance.
   - **Visual Verification**: Comparing "Actual vs Predicted" prices on historical test data.

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
