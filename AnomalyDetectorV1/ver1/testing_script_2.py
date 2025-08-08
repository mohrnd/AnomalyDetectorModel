import tensorflow as tf
import os
import pandas as pd
import numpy as np
import random

def create_random_window(training_data_csv_path, window_size):
    df = pd.read_csv(training_data_csv_path)
    data = df.to_numpy()
    n_rows = data.shape[0]
    chosen_row = random.randint(0, n_rows - window_size)
    window = data[chosen_row:chosen_row + window_size]
    return pd.DataFrame(window, columns=df.columns)

def inject_binary_flip_errors(training_data_csv_path, window_size=60, n_errors=100):
    window_df = create_random_window(training_data_csv_path, window_size)
    data = window_df.to_numpy()
    n_rows, n_cols = data.shape
    error_locations = set()

    while len(error_locations) < n_errors:
        row = random.randint(0, n_rows - 1)
        col = random.randint(2, n_cols - 1)
        if (row, col) not in error_locations:
            val = data[row, col]
            data[row, col] = 1 - val if val in [0, 1] else val  # binary flip
            error_locations.add((row + 2, col + 1))

    infected_df = pd.DataFrame(data, columns=window_df.columns)
    return infected_df, np.array(list(error_locations))

def analyze_input_data(data, name="Input"):
    print(f"\n📊 {name.upper()} ANALYSIS:")
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    print(f"  Mean: {np.mean(data):.3f}")
    print(f"  Std: {np.std(data):.3f}")
    unique = np.unique(data)
    if len(unique) <= 10:
        print(f"  Unique values: {unique}")

def PredictionBenchmark(model, training_data_csv_path, window_size=60, n_errors=10, detect_thresh=0.999):
    infected_df, error_coords = inject_binary_flip_errors(training_data_csv_path, window_size, n_errors)
    test_df = infected_df.drop(columns=["timestamp_ms", "timestamp_iso"])
    data = test_df.values.astype(np.float32)

    assert data.shape[0] == window_size, f"Expected {window_size} rows, got {data.shape[0]}"
    flat = data.flatten().reshape(1, -1)
    
    # Analyze input
    analyze_input_data(flat, "Test Window")

    recon = model.predict(flat, verbose=0)
    error_vector = np.square(flat - recon).reshape(window_size, -1)
    mean_error = np.mean(error_vector)

    print(f"\n📈 Total reconstruction error: {mean_error:.6f}")
    
    # Detailed error stats
    row_col_errors = [
        (r, c, error_vector[r, c])
        for r in range(error_vector.shape[0])
        for c in range(error_vector.shape[1])
    ]
    sorted_errors = sorted(row_col_errors, key=lambda x: x[2], reverse=True)

    high_errors = [(r, c, err) for (r, c, err) in row_col_errors if err > detect_thresh]
    print(f"  Nodes with error > {detect_thresh}: {len(high_errors)}")

    # Threshold distribution
    thresholds = [0.99, 0.999, 0.9999, 0.99999]
    print("\n📊 Threshold-based error distribution:")
    for t in thresholds:
        count = sum(err > t for _, _, err in row_col_errors)
        print(f"  > {t:.5f}: {count} nodes")

    # Detection logic
    correct_detections, false_positives = 0, 0
    matched_errors = set()
    for r, c, err in high_errors:
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
n_errors = 10
n_cycles = 1000
detectionThreshold = 0.999
Results = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
training_data_csv_path = os.path.join(BASE_DIR, '..', 'testing_data.csv')
model_path = os.path.join(BASE_DIR, 'ver1.keras')

model = tf.keras.models.load_model(model_path)

print("\n🚀 Starting Diagnostic Benchmark")
print("=" * 60)

for i in range(n_cycles):
    print(f"\n🔄 Cycle {i+1}/{n_cycles}")
    try:
        correct_detections, false_positives = PredictionBenchmark(
            model, training_data_csv_path, window_size=60, n_errors=n_errors, detect_thresh=detectionThreshold
        )
        Results.append(correct_detections)
        TotalErrors += n_errors
        TotalCorrectDetections += correct_detections
        TotalFalsePositives += false_positives
        print(f"  ✅ Detected: {correct_detections}, ❌ False Positives: {false_positives}")
    except Exception as e:
        print(f"  ❌ Error in cycle {i+1}: {e}")

# ================= FINAL RESULTS ==================
print("=" * 60)
print("\n📊 FINAL DIAGNOSTIC SUMMARY")
print("=" * 60)
precision = TotalCorrectDetections / (TotalCorrectDetections + TotalFalsePositives) if (TotalCorrectDetections + TotalFalsePositives) > 0 else 0
recall = TotalCorrectDetections / TotalErrors if TotalErrors > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Detection threshold: {detectionThreshold}")
print(f"Total errors inserted: {TotalErrors}")
print(f"True Positives: {TotalCorrectDetections}")
print(f"False Positives: {TotalFalsePositives}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"Mean detections per cycle: {np.mean(Results):.2f}")
print(f"Std Dev: {np.std(Results):.2f}, Min: {np.min(Results)}, Max: {np.max(Results)}")
print(Results)