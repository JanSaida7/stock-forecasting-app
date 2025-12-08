import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_loader import load_data
from preprocess import scale_data

def create_sequences(data, seq_length=60):
    """
    Creates sequences of data for the LSTM model.
    X: Past 60 days
    y: Next day's price
    """
    x = []
    y = []
    
    # We need at least seq_length days of data
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
        
    return np.array(x), np.array(y)

def build_model(input_shape):
    """
    Builds the LSTM Neural Network.
    """
    model = Sequential()
    
    # Layer 1: LSTM with 50 units
    # return_sequences=True because we have another LSTM layer after this
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) # Prevents overfitting
    
    # Layer 2: LSTM with 50 units
    # return_sequences=False because this is the last LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Layer 3: Dense layers to consolidate features
    model.add(Dense(units=25))
    
    # Layer 4: Output layer (1 unit for the predicted price)
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    ticker = "AAPL"
    
    # 1. Load Data
    print(f"Loading data for {ticker}...")
    df = load_data(ticker)
    
    # 2. Scale Data
    print("Scaling data...")
    scaler, scaled_data = scale_data(df)
    
    # 3. Create Sequences
    print("Creating sequences...")
    SEQ_LENGTH = 100
    x_train, y_train = create_sequences(scaled_data, SEQ_LENGTH)
    
    # Reshape x_train for LSTM [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    print(f"Training data shape: {x_train.shape}")
    
    # 4. Build Model
    print("Building model...")
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_loader import load_data
from preprocess import scale_data

def create_sequences(data, seq_length=60):
    """
    Creates sequences of data for the LSTM model.
    X: Past 60 days
    y: Next day's price
    """
    x = []
    y = []
    
    # We need at least seq_length days of data
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
        
    return np.array(x), np.array(y)

def build_model(input_shape):
    """
    Builds the LSTM Neural Network.
    """
    model = Sequential()
    
    # Layer 1: LSTM with 50 units
    # return_sequences=True because we have another LSTM layer after this
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) # Prevents overfitting
    
    # Layer 2: LSTM with 50 units
    # return_sequences=False because this is the last LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Layer 3: Dense layers to consolidate features
    model.add(Dense(units=25))
    
    # Layer 4: Output layer (1 unit for the predicted price)
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    ticker = "AAPL"
    
    # 1. Load Data
    print(f"Loading data for {ticker}...")
    df = load_data(ticker)
    
    # 2. Scale Data
    print("Scaling data...")
    scaler, scaled_data = scale_data(df)
    
    # 3. Create Sequences
    print("Creating sequences...")
    SEQ_LENGTH = 100
    x_train, y_train = create_sequences(scaled_data, SEQ_LENGTH)
    
    # Reshape x_train for LSTM [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    print(f"Training data shape: {x_train.shape}")
    
    # 4. Build Model
    print("Building model...")
    model = build_model((x_train.shape[1], 1))
    
    # 5. Train Model
    print("Training model (This may take a minute)...")
    # Epochs = how many times the model sees the data
    # Batch_size = how many samples to process at once
    model.fit(x_train, y_train, batch_size=32, epochs=50)
    
    # 6. Save Model
    model.save('stock_model.keras') # .keras is the new standard format
    print("\nModel trained and saved successfully as 'stock_model.keras'!")
