import numpy as np
import pandas as pd
import tensorflow as tf


model = tf.keras.models.load_model('/mnt/c/Users/nut/CudaTensorflow/AnomalyDetectorV1/ver1/ver1.keras')

#(must be exactly 60 rows)
test_df = pd.read_csv("/mnt/c/Users/nut/CudaTensorflow/AnomalyDetectorV1/ANOMALY.csv")
test_df = test_df.drop(columns=["timestamp_ms", "timestamp_iso"])
data = test_df.values.astype(np.float32)

assert data.shape[0] == 60, "CSV must have exactly 60 rows (2 seconds of data)."


flat = data.flatten().reshape(1, -1)


recon = model.predict(flat, verbose=0)
error_vector = np.square(flat - recon).reshape(60, -1)  # shape: (60 rows, 27 features)
mean_error = np.mean(error_vector)

#total error 
print(f"\nTotal reconstruction error: {mean_error:.6f}\n")


row_col_errors = [
    (row, col, error_vector[row, col])
    for row in range(error_vector.shape[0])
    for col in range(error_vector.shape[1])
]


row_col_errors.sort(key=lambda x: x[2], reverse=True)


print("Sorted reconstruction errors (highest to lowest):")
for r, c, err in row_col_errors:
    print(f"Row {r}, Feature {c} → Error: {err:.6f}")
