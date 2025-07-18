import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


WINDOW_SIZE = 60  # 2 seconds if ~30Hz
THRESHOLD_STD_MULTIPLIER = 3  
EPOCHS = 20
BATCH_SIZE = 32
df_path = os.path.join(BASE_DIR, '..','training_data.csv')
df = pd.read_csv(df_path)
df = df.drop(columns=["timestamp_ms", "timestamp_iso"])
data = df.values.astype(np.float32)  

def create_windows(data, window_size):
    X = []
    for i in range(len(data) - window_size):
        window = data[i:i+window_size].flatten()  # flatten to 1D
        X.append(window)
    return np.array(X)

X = create_windows(data, WINDOW_SIZE)


X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)


input_dim = X.shape[1]
print("\ninput dim: ", input_dim, "\n") # 1620 = 60 samples * 27 sensors/actuators

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(input_dim, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_data=(X_val, X_val), verbose=1)


reconstructions = model.predict(X_train)
errors = np.mean(np.square(X_train - reconstructions), axis=1) # (original - reconstructed)^2
threshold = np.mean(errors) + THRESHOLD_STD_MULTIPLIER * np.std(errors)
print(f"Anomaly threshold set at: {threshold:.6f}")


def check_for_anomaly(window_data):
    """Input: 2s raw window (shape: 60 x num_features), Output: bool"""
    flat = window_data.flatten().reshape(1, -1)
    recon = model.predict(flat)
    error = np.mean(np.square(flat - recon))
    is_anomaly = error > threshold
    return is_anomaly, error

latest_window = data[-WINDOW_SIZE:]  # most recent 2 seconds
anomaly, err = check_for_anomaly(latest_window)
print("Anomaly Detected:" if anomaly else "No Anomaly Detected", f"(error={err:.6f})")

# Save the model
model.save('ver1.keras')