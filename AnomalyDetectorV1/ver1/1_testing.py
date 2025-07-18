import numpy as np
import pandas as pd
import tensorflow as tf
import os
import random
import pandas as pd
import numpy as np
import random

def inject_binary_flip_errors(csv_path, n_errors=100):

    df = pd.read_csv(csv_path)
    data = df.to_numpy()
    
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

    modified_df = pd.DataFrame(data, columns=df.columns)

 
    modified_df.to_csv(csv_path, index=False)

    return modified_df, np.array(list(error_locations))
'''
Note:  [ 0 18] => [ 2 19]
'''
def PredictionBenchmark(n_errors=10, detect_thresh=0.9999):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    test_df_path = os.path.join(BASE_DIR, '..', 'LiveAnomalyEdition.csv')
    mod_df, error_coords = inject_binary_flip_errors(test_df_path, n_errors=n_errors)
    # print("Injected binary flips at:\n", error_coords)

    model_path = os.path.join(BASE_DIR, 'ver1.keras')
    model = tf.keras.models.load_model(model_path)

    # #(must be exactly 60 rows)
    test_df = pd.read_csv(test_df_path)
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

    # print("Sorted reconstruction errors (highest to lowest):")
    highest_errors = [err for _, _, err in row_col_errors if err > detect_thresh]
    print(f"Number of errors higher than {detect_thresh}: {len(highest_errors)}")
    # if len(row_col_errors) != 0:
    #     for r, c, err in row_col_errors:
    #         if err > detect_thresh:
    #             print(f"Row {r + 2}, Feature {c + 3} → Error: {err:.6f}") # adjusted to have the same index as the csv file.
                
    # else:
    #     print("No reconstruction errors detected\n")

    print("Resetting the csv file...")
    original_df_path = os.path.join(BASE_DIR, '..', 'LiveAnomalyEditionOriginal.csv')
    df = pd.read_csv(original_df_path)
    df.to_csv(test_df_path, index=False)
    
    
    correct_detections = 0
    for r, c, err in row_col_errors:
        if err > detect_thresh:
            adjusted_row = r + 2
            adjusted_col = c + 3
            if [adjusted_row, adjusted_col] in error_coords.tolist():
                correct_detections += 1
    
    return correct_detections
    
TotalErrors = 0
TotalCorrectDetections = 0
n_errors = 10
detectionThreshold = 0.999
Results = []
for i in range(0, 100):    
    correct_detections = PredictionBenchmark(n_errors=n_errors, detect_thresh=detectionThreshold)
    print(f"Cycle {i}: {correct_detections} correct detections")
    Results.append(correct_detections)
    TotalErrors += n_errors
    TotalCorrectDetections += correct_detections

print(f"Detection threshold: {detectionThreshold}")
print(f"Total Errors inserted: {TotalErrors}")
print(f"Total correct detections: {TotalCorrectDetections}")
print(Results)


    
    