import numpy as np
from encoding import one_hot_encode_sequence

def preprocess_data(sequences, labels):
    """
    Encodes sequences and converts them to NumPy arrays
    """
    X = np.array([one_hot_encode_sequence(seq) for seq in sequences])
    y = np.array(labels)
    return X, y
