from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    return model
