
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_loader import load_data
from preprocess import scale_data

def create_sequences(data, seq_length=60):
    """
    Creates sequences of data for the LSTM model.
    X: Past 60 days (all features)
    y: Next day's price (only Close price, usually col 0)
    """
    x = []
    y = []
    
    # We need at least seq_length days of data
    for i in range(seq_length, len(data)):
        # Take all features for the sequence
        x.append(data[i-seq_length:i])
        # Predict only the Close price (Column 0)
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

def train_universal_model(epochs=10):
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    all_x_train = []
    all_y_train = []
    
    SEQ_LENGTH = 100
    
    print(f"Starting Generalist Training on: {tickers}")
    
    for ticker in tickers:
        print(f"\n--- Processing {ticker} ---")
        try:
            # 1. Load Data
            df = load_data(ticker)
            if df.empty:
                print(f"Skipping {ticker} (No data)")
                continue
                
            # 2. Scale Data
            scaler, target_scaler, scaled_data = scale_data(df)
            
            # 3. Create Sequences
            x, y = create_sequences(scaled_data, SEQ_LENGTH)
            
            all_x_train.append(x)
            all_y_train.append(y)
            print(f"Added {len(x)} samples from {ticker}")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            return False, str(e)
            
    # Combine all data
    if not all_x_train:
        print("No data collected. Exiting.")
        return False, "No data collected"

    x_train = np.vstack(all_x_train)
    y_train = np.concatenate(all_y_train)
    
    print(f"\nTOTAL Training Data Shape: {x_train.shape}")
    
    # 4. Build Model
    print("Building model...")
    model = build_model((x_train.shape[1], x_train.shape[2]))
    
    # 5. Train Model
    print("Training Universal Model (This will take a few minutes)...")
    model.fit(x_train, y_train, batch_size=32, epochs=epochs)
    
    # 6. Save Model
    model.save('stock_model.keras')
    print("\nUniversal Model trained and saved successfully!")
    return True, "Model trained successfully!"

if __name__ == "__main__":
    train_universal_model()
