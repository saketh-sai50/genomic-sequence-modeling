from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

def build_bilstm(input_shape):
    model = Sequential([
        Bidirectional(LSTM(128), input_shape=input_shape),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    return model
