# Stock Market Forecasting App 📈

A Machine Learning powered application that predicts stock prices using an LSTM (Long Short-Term Memory) Neural Network. Built with Python, TensorFlow, and Streamlit.

## 🚀 Features
- **Real-time Data Loading**: Fetches specific stock data from Yahoo Finance.
- **Interactive Visualization**: Plots historical closing prices.
- **AI Predictions**: Uses a trained LSTM model to forecast future trends.
- **User-Friendly UI**: Simple web interface built with Streamlit.

## 🛠️ Tech Stack
- **Python**: Core programming language.
- **TensorFlow/Keras**: Deep Learning for the LSTM model.
- **Streamlit**: Frontend infrastructure.
- **yfinance**: Data acquisition.
- **Pandas & NumPy**: Data manipulation.
- **Matplotlib**: Plotting and graphs.

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
- **Sequence Length**: Looks back 60 days to predict the next day.
- **Optimizer**: Adam.
- **Loss Function**: Mean Squared Error.

---
*Created by [JanSaida7](https://github.com/JanSaida7)*
