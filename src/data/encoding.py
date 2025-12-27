import numpy as np

def one_hot_encode_sequence(sequence):
    """
    Converts nucleotide sequence into one-hot encoded format.
    A, C, G, T → 4-dimensional vectors
    """
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }

    encoded = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]
    return np.array(encoded)
