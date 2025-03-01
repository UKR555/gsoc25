import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load financial dataset (Replace with actual data source)
def load_data(file_path, feature="Close"):
    df = pd.read_csv(file_path)
    return df[feature].values.reshape(-1, 1)

# Preprocess data for LSTM
def prepare_data(series, time_steps=50):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series)

    X, y = [], []
    for i in range(len(scaled_series) - time_steps):
        X.append(scaled_series[i : i + time_steps])
        y.append(scaled_series[i + time_steps])

    return np.array(X), np.array(y), scaler

# Define LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Train model
def train_model(model, X_train, y_train, epochs=20, batch_size=16):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Predict future values
def predict(model, X_test, scaler):
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions)

# Main function
def main():
    file_path = "stock_data.csv"  # Replace with actual dataset
    time_steps = 50

    # Load and preprocess data
    series = load_data(file_path)
    X, y, scaler = prepare_data(series, time_steps)

    # Split into train and test
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Build and train LSTM model
    model = build_lstm_model((time_steps, 1))
    train_model(model, X_train, y_train)

    # Make predictions
    predictions = predict(model, X_test, scaler)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label="Actual Prices")
    plt.plot(predictions, label="Predicted Prices", linestyle="dashed")
    plt.legend()
    plt.title("Time Series Prediction using LSTM")
    plt.show()

if __name__ == "__main__":
    main()
