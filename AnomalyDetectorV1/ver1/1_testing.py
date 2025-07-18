import tensorflow as tf
import os
import pandas as pd
import numpy as np
import random

def create_random_window(training_data_csv_path,window_size):
    df = pd.read_csv(training_data_csv_path)
    data = df.to_numpy()
    n_rows, n_cols = data.shape
    chosen_row = random.randint(0, n_rows - window_size)
    window = data[chosen_row:chosen_row + window_size]
    window_df = pd.DataFrame(window, columns=df.columns)
    return window_df

def inject_binary_flip_errors(training_data_csv_path,window_size=60, n_errors=100):
    # Select and initialise a random window from the training data
    window_df = create_random_window(training_data_csv_path, window_size)

    data = window_df.to_numpy()
    
    n_rows, n_cols = data.shape
    error_locations = set()

    while len(error_locations) < n_errors:
        row = random.randint(0, n_rows - 1)
        col = random.randint(2, n_cols - 1)
        if (row, col) not in error_locations:
            val = data[row, col]
            if val == 0:
                data[row, col] = 1
                error_locations.add((row + 2, col + 1))
            else:
                data[row, col] = 0
                error_locations.add((row + 2, col + 1))

    infected_df = pd.DataFrame(data, columns=window_df.columns)

    return infected_df, np.array(list(error_locations))
'''
Note:  [ 0 18] => [ 2 19]
'''
def PredictionBenchmark(model,training_data_csv_path,window_size=60,n_errors=10, detect_thresh=0.9999):
    
    infected_df, error_coords = inject_binary_flip_errors(training_data_csv_path,window_size, n_errors)
    test_df = infected_df.drop(columns=["timestamp_ms", "timestamp_iso"])
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

    # print("Sorted reconstruction errors (highest to lowest):")
    highest_errors = [err for _, _, err in row_col_errors if err > detect_thresh]
    print(f"Number of errors higher than {detect_thresh}: {len(highest_errors)}")
    # if len(row_col_errors) != 0:
    #     for r, c, err in row_col_errors:
    #         if err > detect_thresh:
    #             print(f"Row {r + 2}, Feature {c + 3} → Error: {err:.6f}") # adjusted to have the same index as the csv file.
                
    # else:
    #     print("No reconstruction errors detected\n")
    
    correct_detections = 0
    false_positives = 0
    matched_errors = set()
    
    for r, c, err in row_col_errors:
        if err > detect_thresh:
            adjusted_row = r + 2
            adjusted_col = c + 3
            key = (adjusted_row, adjusted_col)
    
            if key in map(tuple, error_coords) and key not in matched_errors:
                correct_detections += 1
                matched_errors.add(key)
            elif key not in map(tuple, error_coords):
                false_positives += 1

    return correct_detections, false_positives

TotalErrors = 0
TotalCorrectDetections = 0
TotalFalsePositives = 0
n_errors = 20
n_cycles = 100
detectionThreshold = 0.9999
Results = []
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
training_data_csv_path = os.path.join(BASE_DIR, '..', 'modbus_log_20250710_101858_10ms.csv')
model_path = os.path.join(BASE_DIR, 'ver1.keras')
model = tf.keras.models.load_model(model_path)
for i in range(0, n_cycles):    
    correct_detections, false_positives = PredictionBenchmark(model,training_data_csv_path,window_size=60,n_errors=10, detect_thresh=0.9999)
    print(f"Cycle {i}: {correct_detections} correct detections")
    Results.append(correct_detections)
    TotalErrors += n_errors
    TotalCorrectDetections += correct_detections
    TotalFalsePositives += false_positives

precision = TotalCorrectDetections / (TotalCorrectDetections + TotalFalsePositives) if (TotalCorrectDetections + TotalFalsePositives) > 0 else 0
recall = TotalCorrectDetections / TotalErrors if TotalErrors > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Detection threshold used: {detectionThreshold}")
print(f"Total Errors inserted: {TotalErrors}")
print(f"Total correct detections (True Positives): {TotalCorrectDetections}")
print(f"Total false positives: {TotalFalsePositives}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(Results)


    
    