# Stock Market Forecasting App 📈

A Machine Learning powered application that predicts stock prices using an LSTM (Long Short-Term Memory) Neural Network. Built with Python, TensorFlow, and Streamlit.

## 🚀 Features
### 1. **Interactive Dashboard**
   - **Company Selector**: Choose from popular tech giants (Apple, Tesla, Google, Microsoft, etc.).
   - **Interactive Charts**: Zoom, pan, and hover over data points using **Plotly**.
   - **Tomorrow's Prediction**: Shows a clear, predicted stock price for the next trading day.

### 2. **Universal AI Model**
   - **Generalist Training**: Trained on **10 Years** of combined data from **AAPL, MSFT, GOOGL, and AMZN**.
   - **Robustness**: Understands general tech sector trends rather than just one company's history.
   - **Deep Learning**: Uses an **LSTM (Long Short-Term Memory)** neural network with 50 Epochs of training.

### 3. **Accuracy Reporting**
   - **Scoreboard**: Displays **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error) to quantify performance.
   - Real-time comparison of "Actual vs Predicted" prices.

## 🛠️ Tech Stack
- **Python**: Core programming language.
- **TensorFlow/Keras**: Deep Learning for the LSTM model.
- **Streamlit**: Frontend infrastructure.
- **yfinance**: Data acquisition.
- **Pandas & NumPy**: Data manipulation.
- **Plotly**: Interactive Data Visualization.

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
- **Sequence Length**: Looks back 100 days to predict the next day.
- **Optimizer**: Adam.
- **Loss Function**: Mean Squared Error.

---
*Created by [JanSaida7](https://github.com/JanSaida7)*
