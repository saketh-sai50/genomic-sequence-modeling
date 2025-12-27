from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

def build_gru(input_shape):
    model = Sequential([
        GRU(128, input_shape=input_shape),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    return model
