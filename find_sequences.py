import numpy as np
from tensorflow.keras.models import load_model
import random

print("Loading Hybrid BiLSTM Model...")
hybrid_model = load_model('models/genomic_hybrid_bilstm.keras')

# Standard one-hot encoding function
def preprocess_batch(dna_list):
    # Mapping A, C, G, T to integers (0 is reserved for padding/unknown)
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    batch_array = []
    for seq in dna_list:
        integers = [mapping.get(base, 0) for base in seq]
        batch_array.append(integers)
    # Output shape will now be (Batch_Size, 60)
    return np.array(batch_array, dtype=np.int32)

print("Generating 50,000 synthetic DNA sequences...")
bases = ['A', 'C', 'G', 'T']
test_sequences = [''.join(random.choices(bases, k=60)) for _ in range(50000)]

print("Feeding sequences to the model...")
processed_batch = preprocess_batch(test_sequences)

predictions = hybrid_model.predict(processed_batch, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

class_0_indices = np.where(predicted_classes == 0)[0]
class_1_indices = np.where(predicted_classes == 1)[0]

print("\n" + "="*50)
print("🎯 CRACKED SEQUENCES FOUND 🎯")
print("="*50)

if len(class_0_indices) > 0:
    print(f"\n✅ CLASS 0 (DONOR) TRIGGER SEQUENCE:")
    best_0 = class_0_indices[np.argmax(predictions[class_0_indices, 0])]
    print(test_sequences[best_0])
else:
    print("\n❌ No Class 0 sequences found. (Run again)")

if len(class_1_indices) > 0:
    print(f"\n✅ CLASS 1 (ACCEPTOR) TRIGGER SEQUENCE:")
    best_1 = class_1_indices[np.argmax(predictions[class_1_indices, 1])]
    print(test_sequences[best_1])
else:
    print("\n❌ No Class 1 sequences found.")
print("="*50)