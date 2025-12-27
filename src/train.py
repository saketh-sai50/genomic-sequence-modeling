import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from models.bilstm import build_bilstm

# Dummy data (replace later)
X = np.random.rand(1000, 60, 4)
y = np.random.randint(0, 2, 1000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = build_bilstm(input_shape=(60, 4))

model.compile(
    optimizer=Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

model.save("bilstm_model.h5")
