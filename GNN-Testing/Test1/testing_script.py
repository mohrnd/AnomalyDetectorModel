import tensorflow as tf
import os
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple

# Import your GNN model classes - you might need to adjust this import
# from paste import *  

# Since loading the saved model is causing issues, we'll recreate the model from scratch
from keras.saving import register_keras_serializable

# DIAGNOSTIC FUNCTIONS
def diagnose_model_behavior(model, inputs, variable_names):
    """Diagnose what the model is actually learning."""
    print("\n" + "="*60)
    print("MODEL DIAGNOSTIC ANALYSIS")
    print("="*60)
    
    try:
        outputs = model(inputs, training=False)
        
        print(f"Input node_values shape: {inputs['node_values'].shape}")
        print(f"Input node_values range: [{tf.reduce_min(inputs['node_values']):.3f}, {tf.reduce_max(inputs['node_values']):.3f}]")
        print(f"Input node_values mean: {tf.reduce_mean(inputs['node_values']):.3f}")
        print(f"Input node_values std: {tf.math.reduce_std(inputs['node_values']):.3f}")
        
        print(f"\nInitial embeddings shape: {outputs['initial_embeddings'].shape}")
        print(f"Initial embeddings range: [{tf.reduce_min(outputs['initial_embeddings']):.3f}, {tf.reduce_max(outputs['initial_embeddings']):.3f}]")
        print(f"Initial embeddings mean: {tf.reduce_mean(outputs['initial_embeddings']):.3f}")
        print(f"Initial embeddings std: {tf.math.reduce_std(outputs['initial_embeddings']):.3f}")
        
        print(f"\nReconstructions shape: {outputs['reconstructions'].shape}")
        print(f"Reconstructions range: [{tf.reduce_min(outputs['reconstructions']):.3f}, {tf.reduce_max(outputs['reconstructions']):.3f}]")
        print(f"Reconstructions mean: {tf.reduce_mean(outputs['reconstructions']):.3f}")
        print(f"Reconstructions std: {tf.math.reduce_std(outputs['reconstructions']):.3f}")
        
        # Calculate the actual reconstruction errors per node
        reconstruction_errors = tf.reduce_mean(tf.square(outputs['reconstructions'] - outputs['initial_embeddings']), axis=-1)
        reconstruction_errors_flat = reconstruction_errors.numpy().flatten()
        
        print(f"\nPer-node reconstruction errors:")
        print(f"  Min error: {np.min(reconstruction_errors_flat):.6f}")
        print(f"  Max error: {np.max(reconstruction_errors_flat):.6f}")
        print(f"  Mean error: {np.mean(reconstruction_errors_flat):.6f}")
        print(f"  Std error: {np.std(reconstruction_errors_flat):.6f}")
        
        # Show error distribution
        print(f"\nError distribution:")
        for i, error in enumerate(reconstruction_errors_flat):
            if i < len(variable_names):
                print(f"  {variable_names[i]:20}: {error:.6f}")
        
        # Analyze thresholds
        thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        print(f"\nNodes flagged at different thresholds:")
        for threshold in thresholds:
            flagged_count = np.sum(reconstruction_errors_flat > threshold)
            print(f"  Threshold {threshold:4.1f}: {flagged_count:2d}/{len(reconstruction_errors_flat)} nodes flagged")
        
        # Check if model is actually learning
        print(f"\n🔍 DIAGNOSIS SUMMARY:")
        if np.all(reconstruction_errors_flat > 1.0):
            print("  ❌ PROBLEM: ALL nodes have high reconstruction error")
            print("  ❌ Model is NOT learning proper reconstructions")
        elif np.std(reconstruction_errors_flat) < 0.001:
            print("  ❌ PROBLEM: All reconstruction errors are nearly identical")
            print("  ❌ Model is not discriminating between different inputs")
        elif np.mean(reconstruction_errors_flat) > 1.0:
            print("  ⚠️  WARNING: Very high average reconstruction error")
            print("  ⚠️  Model may not be learning effectively")
        else:
            print("  ✅ Model appears to be learning reasonable reconstructions")
        
        return reconstruction_errors_flat
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        return None

def analyze_input_data(data, name="Input Data"):
    """Analyze the characteristics of input data."""
    print(f"\n📊 {name.upper()} ANALYSIS:")
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    print(f"  Mean: {np.mean(data):.3f}")
    print(f"  Std: {np.std(data):.3f}")
    
    # Check for binary data
    unique_values = np.unique(data)
    if len(unique_values) <= 10:
        print(f"  Unique values: {unique_values}")
        if set(unique_values).issubset({0.0, 1.0}):
            print("  ✅ Data appears to be binary (0s and 1s)")
        else:
            print("  ℹ️  Data has few unique values (may be categorical)")
    
    # Check for missing/invalid data
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))
    if nan_count > 0:
        print(f"  ⚠️  WARNING: {nan_count} NaN values found")
    if inf_count > 0:
        print(f"  ⚠️  WARNING: {inf_count} infinite values found")

def compare_models_side_by_side(original_errors, fixed_errors=None):
    """Compare the behavior of original vs fixed model."""
    print(f"\n📊 MODEL COMPARISON:")
    print(f"Original Model:")
    print(f"  Error range: [{np.min(original_errors):.6f}, {np.max(original_errors):.6f}]")
    print(f"  Error mean: {np.mean(original_errors):.6f}")
    print(f"  Error std: {np.std(original_errors):.6f}")
    print(f"  Nodes with error > 1.0: {np.sum(original_errors > 1.0)}/{len(original_errors)}")
    
    if fixed_errors is not None:
        print(f"Fixed Model:")
        print(f"  Error range: [{np.min(fixed_errors):.6f}, {np.max(fixed_errors):.6f}]")
        print(f"  Error mean: {np.mean(fixed_errors):.6f}")
        print(f"  Error std: {np.std(fixed_errors):.6f}")
        print(f"  Nodes with error > 1.0: {np.sum(fixed_errors > 1.0)}/{len(fixed_errors)}")

def analyze_detection_performance(cycles_data, detection_threshold):
    """Analyze detection performance across multiple cycles."""
    print(f"\n📈 DETECTION PERFORMANCE ANALYSIS:")
    
    # Extract data from cycles
    correct_detections = []
    false_positives = []
    total_detections = []
    
    for cycle_data in cycles_data:
        if isinstance(cycle_data, tuple) and len(cycle_data) >= 2:
            correct, false_pos = cycle_data[0], cycle_data[1]
            correct_detections.append(correct)
            false_positives.append(false_pos)
            total_detections.append(correct + false_pos)
    
    if correct_detections:
        print(f"Detection Threshold: {detection_threshold}")
        print(f"Correct Detections:")
        print(f"  Mean: {np.mean(correct_detections):.2f}")
        print(f"  Std: {np.std(correct_detections):.2f}")
        print(f"  Range: [{np.min(correct_detections)}, {np.max(correct_detections)}]")
        
        print(f"False Positives:")
        print(f"  Mean: {np.mean(false_positives):.2f}")
        print(f"  Std: {np.std(false_positives):.2f}")
        print(f"  Range: [{np.min(false_positives)}, {np.max(false_positives)}]")
        
        print(f"Total Detections:")
        print(f"  Mean: {np.mean(total_detections):.2f}")
        print(f"  Std: {np.std(total_detections):.2f}")
        print(f"  Range: [{np.min(total_detections)}, {np.max(total_detections)}]")
        
        # Check for concerning patterns
        if np.std(total_detections) < 1.0:
            print(f"  ⚠️  WARNING: Very consistent total detections ({np.mean(total_detections):.1f}) - model may be broken")
        
        if np.mean(false_positives) > np.mean(correct_detections) * 2:
            print(f"  ⚠️  WARNING: False positives >> correct detections - threshold too low")

def run_comprehensive_diagnosis():
    """Run a comprehensive diagnosis of the current model."""
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE MODEL DIAGNOSIS")
    print("="*80)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'ver2.keras')
    training_data_path = os.path.join(BASE_DIR, 'training_data.csv')
    
    try:
        # Load data first
        print("📁 Loading training data...")
        with open(training_data_path, 'r') as f:
            csv_data_str = f.read()
        
        windowed_data, node_assignments, variable_names = prepare_data(csv_data_str)
        analyze_input_data(windowed_data, "Windowed Training Data")
        
        print(f"📋 Data Structure:")
        print(f"  Variables: {len(variable_names)}")
        print(f"  Node assignments: {node_assignments}")
        
        # Try to load and diagnose the model
        print("\n🤖 Loading model...")
        model, node_assignments, adjacency_matrix, node_type_indices, variable_names = load_model_and_setup(
            model_path, training_data_path
        )
        
        # Create test inputs
        print("\n🧪 Creating test inputs...")
        num_nodes = len(variable_names)
        
        # Test with different types of inputs
        test_cases = [
            ("All zeros", tf.zeros([1, num_nodes, 1])),
            ("All ones", tf.ones([1, num_nodes, 1])),
            ("Random normal", tf.random.normal([1, num_nodes, 1])),
            ("Random binary", tf.cast(tf.random.uniform([1, num_nodes, 1]) > 0.5, tf.float32)),
        ]
        
        all_errors = []
        for test_name, test_values in test_cases:
            print(f"\n🔬 Testing with {test_name}:")
            
            test_inputs = {
                'node_values': test_values,
                'node_types': node_type_indices,
                'adjacency_matrix': adjacency_matrix,
                'edge_embeddings': adjacency_matrix
            }
            
            analyze_input_data(test_values.numpy(), f"{test_name} Input")
            errors = diagnose_model_behavior(model, test_inputs, variable_names)
            if errors is not None:
                all_errors.append((test_name, errors))
        
        # Compare different test cases
        if len(all_errors) > 1:
            print(f"\n🔄 CROSS-TEST COMPARISON:")
            for i, (name1, errors1) in enumerate(all_errors):
                for j, (name2, errors2) in enumerate(all_errors[i+1:], i+1):
                    diff = np.mean(np.abs(errors1 - errors2))
                    print(f"  {name1} vs {name2}: Mean absolute difference = {diff:.6f}")
                    if diff < 0.001:
                        print(f"    ⚠️  WARNING: Nearly identical outputs - model may be ignoring inputs!")
        
        return model, all_errors
        
    except Exception as e:
        print(f"❌ Error during comprehensive diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return None, []

@register_keras_serializable()
class FiLMLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim: int, name: str = "film_layer"):
        super(FiLMLayer, self).__init__(name=name)
        self.embedding_dim = embedding_dim
        self.gamma_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(embedding_dim, activation='sigmoid')
        ])
        self.beta_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(embedding_dim, activation='tanh')
        ])

    def call(self, embeddings: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
        gamma = self.gamma_mlp(values)
        beta = self.beta_mlp(values)
        return gamma * embeddings + beta

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_dim": self.embedding_dim})
        return config

@register_keras_serializable()
class MessagePassing(tf.keras.layers.Layer):
    def __init__(self, embedding_dim: int, message_dim: int = 32, name: str = "message_passing"):
        super(MessagePassing, self).__init__(name=name)
        self.embedding_dim = embedding_dim
        self.message_dim = message_dim
        self.message_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(message_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim)
        ])
        self.update_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim * 2, activation='relu'),
            tf.keras.layers.Dense(embedding_dim)
        ])

    def call(self, node_embeddings: tf.Tensor, adjacency_matrix: tf.Tensor, edge_embeddings: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(node_embeddings)[0]
        num_nodes = tf.shape(node_embeddings)[1]

        source_nodes = tf.expand_dims(node_embeddings, 2)
        source_nodes = tf.tile(source_nodes, [1, 1, num_nodes, 1])
        edge_features = tf.concat([source_nodes, edge_embeddings], axis=-1)
        messages = self.message_mlp(edge_features)

        edge_mask = tf.reduce_sum(tf.abs(edge_embeddings), axis=-1, keepdims=True) > 0
        messages *= tf.cast(edge_mask, messages.dtype)

        aggregated_messages = tf.reduce_sum(messages, axis=1)
        update_input = tf.concat([node_embeddings, aggregated_messages], axis=-1)
        return self.update_mlp(update_input)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "message_dim": self.message_dim
        })
        return config

@register_keras_serializable()
class IndustrialGNN(tf.keras.Model):
    def __init__(self, node_types: List[str], embedding_dim: int = 32, num_gnn_layers: int = 2, edge_dim: int = 3, **kwargs):
        # Filter out unsupported kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['trainable']}
        super(IndustrialGNN, self).__init__(**filtered_kwargs)
        
        self.node_types = node_types
        self.embedding_dim = embedding_dim
        self.num_gnn_layers = num_gnn_layers
        self.edge_dim = edge_dim
        
        self.type_embeddings = tf.Variable(
            tf.random.normal([len(node_types), embedding_dim], stddev=0.1), trainable=True
        )
        self.film_layer = FiLMLayer(embedding_dim)
        self.gnn_layers = [MessagePassing(embedding_dim, name=f"gnn_layer_{i}") for i in range(num_gnn_layers)]
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim)
        ])

    def create_graph_structure(self, node_assignments: Dict[str, List[str]]) -> Tuple[tf.Tensor, tf.Tensor]:
        var_to_idx = {}
        node_types_list = []
        idx = 0
        for node_type, variables in node_assignments.items():
            for var in variables:
                var_to_idx[var] = idx
                node_types_list.append(self.node_types.index(node_type))
                idx += 1
        num_nodes = len(var_to_idx)
        adjacency = np.zeros((num_nodes, num_nodes, self.edge_dim))
        for i, type_i in enumerate(node_types_list):
            for j, type_j in enumerate(node_types_list):
                if i != j:
                    if type_i == 0 and type_j in [1, 2]:
                        adjacency[i, j] = [1, 0, 0]
                    elif type_i in [1, 2] and type_j == 0:
                        adjacency[i, j] = [0, 1, 0]
                    elif type_i == type_j:
                        adjacency[i, j] = [0, 0, 1]
        return tf.constant(adjacency, dtype=tf.float32), tf.constant(node_types_list, dtype=tf.int32)

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> Dict[str, tf.Tensor]:
        node_values = inputs['node_values']
        node_types = inputs['node_types']
        adjacency_matrix = inputs['adjacency_matrix']
        edge_embeddings = inputs['edge_embeddings']

        batch_size = tf.shape(node_values)[0]
        initial_embeddings = tf.gather(self.type_embeddings, node_types)
        initial_embeddings = tf.tile(tf.expand_dims(initial_embeddings, 0), [batch_size, 1, 1])
        conditioned_embeddings = self.film_layer(initial_embeddings, node_values)
        h_0 = conditioned_embeddings

        current_embeddings = conditioned_embeddings
        batch_adjacency = tf.tile(tf.expand_dims(adjacency_matrix, 0), [batch_size, 1, 1, 1])
        batch_edges = tf.tile(tf.expand_dims(edge_embeddings, 0), [batch_size, 1, 1, 1])

        for gnn_layer in self.gnn_layers:
            current_embeddings = gnn_layer(current_embeddings, batch_adjacency, batch_edges)

        reconstructions = self.decoder(current_embeddings)
        return {
            'embeddings': current_embeddings,
            'reconstructions': reconstructions,
            'initial_embeddings': h_0
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "node_types": self.node_types,
            "embedding_dim": self.embedding_dim,
            "num_gnn_layers": self.num_gnn_layers,
            "edge_dim": self.edge_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Remove any unsupported arguments
        filtered_config = {k: v for k, v in config.items() if k not in ['trainable', 'dtype']}
        return cls(**filtered_config)

def prepare_data(csv_data: str, window_size: int = 60) -> Tuple[np.ndarray, Dict[str, List[str]], List[str]]:
    lines = csv_data.strip().split('\n')
    header = lines[0].split(',')[2:]
    data = [list(map(float, line.split(',')[2:])) for line in lines[1:]]
    data = np.array(data)

    windowed_data = [data[i:i + window_size] for i in range(0, len(data) - window_size + 1, window_size)]
    windowed_data = np.array(windowed_data)

    node_assignments = {'PLC': [], 'Sensor': [], 'Actuator': []}
    for var in header:
        if 'PLC' in var or 'P1_2_Armed' in var or 'P3_4_Armed' in var:
            node_assignments['PLC'].append(var)
        elif any(sensor_type in var for sensor_type in ['Presence', 'Check', 'Light', 'EmergencyStop']):
            node_assignments['Sensor'].append(var)
        else:
            node_assignments['Actuator'].append(var)

    return windowed_data, node_assignments, header

def create_random_window(training_data_csv_path: str, window_size: int = 60) -> pd.DataFrame:
    """Create a random window from the training data."""
    df = pd.read_csv(training_data_csv_path)
    data = df.to_numpy()
    n_rows, n_cols = data.shape
    chosen_row = random.randint(0, n_rows - window_size)
    window = data[chosen_row:chosen_row + window_size]
    window_df = pd.DataFrame(window, columns=df.columns)
    return window_df

def inject_binary_flip_errors(training_data_csv_path: str, window_size: int = 60, n_errors: int = 100) -> Tuple[pd.DataFrame, np.ndarray]:
    """Inject binary flip errors into a random window of data."""
    # Select and initialise a random window from the training data
    window_df = create_random_window(training_data_csv_path, window_size)
    data = window_df.to_numpy()
    
    n_rows, n_cols = data.shape
    error_locations = set()

    while len(error_locations) < n_errors:
        row = random.randint(0, n_rows - 1)
        # Skip timestamp columns (first 2 columns)
        col = random.randint(2, n_cols - 1)
        if (row, col) not in error_locations:
            val = data[row, col]
            if val == 0:
                data[row, col] = 1
                error_locations.add((row + 2, col + 1))  # Adjusted for CSV indexing
            else:
                data[row, col] = 0
                error_locations.add((row + 2, col + 1))  # Adjusted for CSV indexing

    infected_df = pd.DataFrame(data, columns=window_df.columns)
    return infected_df, np.array(list(error_locations))

def prepare_gnn_input(infected_df: pd.DataFrame, node_assignments: Dict[str, List[str]], 
                     adjacency_matrix: tf.Tensor, node_type_indices: tf.Tensor, 
                     window_size: int = 60) -> Dict[str, tf.Tensor]:
    """Prepare input data for the GNN model."""
    # Remove timestamp columns
    test_df = infected_df.drop(columns=["timestamp_ms", "timestamp_iso"])
    data = test_df.values.astype(np.float32)
    
    assert data.shape[0] == window_size, f"Data must have exactly {window_size} rows."
    
    # Calculate node values (mean over time window)
    node_values = np.mean(data, axis=0, keepdims=True)  # Shape: (1, num_features)
    node_values = np.expand_dims(node_values, axis=-1)  # Shape: (1, num_features, 1)
    
    inputs = {
        'node_values': tf.constant(node_values, dtype=tf.float32),
        'node_types': node_type_indices,
        'adjacency_matrix': adjacency_matrix,
        'edge_embeddings': adjacency_matrix
    }
    
    return inputs, data

def calculate_reconstruction_error(model: IndustrialGNN, inputs: Dict[str, tf.Tensor], 
                                 original_data: np.ndarray) -> np.ndarray:
    """Calculate reconstruction error for each node/feature."""
    outputs = model(inputs, training=False)
    
    # Get reconstructions and initial embeddings
    reconstructions = outputs['reconstructions'].numpy()  # Shape: (1, num_nodes, embedding_dim)
    initial_embeddings = outputs['initial_embeddings'].numpy()  # Shape: (1, num_nodes, embedding_dim)
    
    # Calculate reconstruction error per node (feature)
    node_errors = np.mean(np.square(reconstructions - initial_embeddings), axis=-1)  # Shape: (1, num_nodes)
    node_errors = node_errors.flatten()  # Shape: (num_nodes,)
    
    return node_errors

def gnn_prediction_benchmark(model: IndustrialGNN, training_data_csv_path: str, 
                           node_assignments: Dict[str, List[str]], adjacency_matrix: tf.Tensor,
                           node_type_indices: tf.Tensor, variable_names: List[str],
                           window_size: int = 60, n_errors: int = 10, 
                           detect_thresh: float = 0.9999, verbose: bool = False) -> Tuple[int, int]:
    """Run prediction benchmark for the GNN model with enhanced diagnostics."""
    
    # Inject errors and prepare data
    infected_df, error_coords = inject_binary_flip_errors(training_data_csv_path, window_size, n_errors)
    inputs, original_data = prepare_gnn_input(infected_df, node_assignments, adjacency_matrix, 
                                            node_type_indices, window_size)
    
    if verbose:
        print(f"\n🧪 BENCHMARK DIAGNOSTICS:")
        print(f"  Window size: {window_size}")
        print(f"  Errors injected: {n_errors}")
        print(f"  Detection threshold: {detect_thresh}")
        analyze_input_data(original_data, "Test Window Data")
        
        # Show where errors were injected
        print(f"  Error locations (CSV coordinates): {error_coords[:5]}..." if len(error_coords) > 5 else error_coords)
    
    # Calculate reconstruction errors
    node_errors = calculate_reconstruction_error(model, inputs, original_data)
    
    if verbose:
        print(f"\n📊 RECONSTRUCTION ERROR ANALYSIS:")
        print(f"  Error shape: {node_errors.shape}")
        analyze_input_data(node_errors, "Node Reconstruction Errors")
        
        # Show top errors
        top_error_indices = np.argsort(node_errors)[-10:][::-1]
        print(f"  Top 10 highest errors:")
        for i, idx in enumerate(top_error_indices):
            var_name = variable_names[idx] if idx < len(variable_names) else f"Node_{idx}"
            print(f"    {i+1}. {var_name}: {node_errors[idx]:.6f}")
    
    mean_error = np.mean(node_errors)
    if not verbose:  # Only print this line if not in verbose mode (to match original output)
        print(f"\nTotal reconstruction error: {mean_error:.6f}")
    
    # Find high error nodes
    high_error_nodes = []
    for node_idx, error in enumerate(node_errors):
        if error > detect_thresh:
            high_error_nodes.append((node_idx, error))
    
    if not verbose:  # Only print this line if not in verbose mode
        print(f"Number of nodes with errors higher than {detect_thresh}: {len(high_error_nodes)}")
    
    if verbose and high_error_nodes:
        print(f"\n🚨 HIGH ERROR NODES (>{detect_thresh}):")
        for node_idx, error in high_error_nodes:
            var_name = variable_names[node_idx] if node_idx < len(variable_names) else f"Node_{node_idx}"
            print(f"  {var_name}: {error:.6f}")
    
    # Calculate correct detections and false positives
    correct_detections = 0
    false_positives = 0
    matched_errors = set()
    
    for node_idx, error in high_error_nodes:
        # Map node index back to variable name and check if it had injected errors
        variable_name = variable_names[node_idx] if node_idx < len(variable_names) else f"Unknown_{node_idx}"
        
        # Check if this variable had any errors injected
        has_error = False
        for row, col in error_coords:
            # Map column index back to variable (accounting for timestamp columns)
            if col - 3 == node_idx:  # -3 to account for timestamp columns and 0-based indexing
                has_error = True
                break
        
        if has_error and (node_idx,) not in matched_errors:
            correct_detections += 1
            matched_errors.add((node_idx,))
            if verbose:
                print(f"  ✅ Correctly detected error in {variable_name}")
        elif not has_error:
            false_positives += 1
            if verbose:
                print(f"  ❌ False positive in {variable_name}")
    
    if verbose:
        print(f"\n📈 DETECTION RESULTS:")
        print(f"  Correct detections: {correct_detections}")
        print(f"  False positives: {false_positives}")
        print(f"  Total detections: {correct_detections + false_positives}")
        if n_errors > 0:
            recall = correct_detections / n_errors
            print(f"  Recall: {recall:.3f}")
        if (correct_detections + false_positives) > 0:
            precision = correct_detections / (correct_detections + false_positives)
            print(f"  Precision: {precision:.3f}")
    
    return correct_detections, false_positives

def load_model_and_setup(model_path: str, training_data_csv_path: str) -> Tuple[IndustrialGNN, Dict[str, List[str]], tf.Tensor, tf.Tensor, List[str]]:
    """Load the trained model and set up necessary components."""
    
    # First, recreate the data preparation from training to get the structure
    with open(training_data_csv_path, 'r') as f:
        csv_data_str = f.read()
    
    windowed_data, node_assignments, variable_names = prepare_data(csv_data_str)
    
    try:
        # Try to load the model directly
        model = tf.keras.models.load_model(model_path, custom_objects={
            'IndustrialGNN': IndustrialGNN,
            'FiLMLayer': FiLMLayer,
            'MessagePassing': MessagePassing
        })
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading saved model: {e}")
        print("Creating new model with same architecture...")
        
        # Create a new model with the same architecture
        node_types = ['PLC', 'Sensor', 'Actuator']
        model = IndustrialGNN(node_types=node_types, embedding_dim=32, num_gnn_layers=2)
        
        # Build the model by running a dummy forward pass
        adjacency_matrix, node_type_indices = model.create_graph_structure(node_assignments)
        num_nodes = len(variable_names)
        dummy_node_values = tf.random.normal([1, num_nodes, 1])
        
        dummy_inputs = {
            'node_values': dummy_node_values,
            'node_types': node_type_indices,
            'adjacency_matrix': adjacency_matrix,
            'edge_embeddings': adjacency_matrix
        }
        
        _ = model(dummy_inputs)
        print("New model created and built successfully!")
        
        # Try to load weights if the model file exists
        try:
            # If you have a separate weights file, load it here
            # model.load_weights(weights_path)
            print("Note: Using randomly initialized weights. For best results, retrain the model.")
        except Exception as w_e:
            print(f"Could not load weights: {w_e}")
    
    # Recreate adjacency matrix and node type indices
    adjacency_matrix, node_type_indices = model.create_graph_structure(node_assignments)
    
    return model, node_assignments, adjacency_matrix, node_type_indices, variable_names

def run_detailed_single_benchmark(model, training_data_csv_path, node_assignments, 
                                adjacency_matrix, node_type_indices, variable_names,
                                n_errors=10, detection_threshold=0.999):
    """Run a single benchmark cycle with full diagnostics enabled."""
    print("\n" + "="*60)
    print("🔬 DETAILED SINGLE BENCHMARK ANALYSIS")
    print("="*60)
    
    correct_detections, false_positives = gnn_prediction_benchmark(
        model, training_data_csv_path, node_assignments, adjacency_matrix,
        node_type_indices, variable_names, window_size=60, n_errors=n_errors,
        detect_thresh=detection_threshold, verbose=True
    )
    
    print(f"\n📊 SINGLE BENCHMARK SUMMARY:")
    print(f"  Errors injected: {n_errors}")
    print(f"  Correct detections: {correct_detections}")
    print(f"  False positives: {false_positives}")
    print(f"  Total detections: {correct_detections + false_positives}")
    
    if n_errors > 0:
        recall = correct_detections / n_errors
        print(f"  Recall: {recall:.3f}")
    if (correct_detections + false_positives) > 0:
        precision = correct_detections / (correct_detections + false_positives)
        print(f"  Precision: {precision:.3f}")
        if n_errors > 0:
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  F1 Score: {f1:.3f}")
    
    return correct_detections, false_positives

def main():
    """Enhanced main function with comprehensive diagnostics."""
    print("🚀 STARTING ENHANCED DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    # Configuration
    n_errors = 10
    n_cycles = 10  # Reduced for diagnostic run
    detection_threshold = 0.999
    window_size = 60
    
    # File paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    training_data_csv_path = os.path.join(BASE_DIR, 'training_data.csv')
    testing_data_csv_path = os.path.join(BASE_DIR, 'testing_data.csv')
    model_path = os.path.join(BASE_DIR, 'ver2.keras')
    
    # STEP 1: Run comprehensive diagnosis
    print("\n" + "="*60)
    print("STEP 1: COMPREHENSIVE MODEL DIAGNOSIS")
    print("="*60)
    
    try:
        model, diagnostic_errors = run_comprehensive_diagnosis()
        
        if model is None:
            print("❌ Could not load or create model. Stopping analysis.")
            return
            
        print("✅ Comprehensive diagnosis completed!")
        
    except Exception as e:
        print(f"❌ Error during comprehensive diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STEP 2: Load model components for benchmarking
    print("\n" + "="*60)
    print("STEP 2: LOADING MODEL COMPONENTS FOR BENCHMARKING")
    print("="*60)
    
    try:
        model, node_assignments, adjacency_matrix, node_type_indices, variable_names = load_model_and_setup(
            model_path, training_data_csv_path
        )
        
        print(f"✅ Loaded model with {len(variable_names)} variables")
        print(f"📋 Node assignments: {node_assignments}")
        
    except Exception as e:
        print(f"❌ Error loading model components: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STEP 3: Run detailed single benchmark
    print("\n" + "="*60)
    print("STEP 3: DETAILED SINGLE BENCHMARK")
    print("="*60)
    
    try:
        single_correct, single_false_pos = run_detailed_single_benchmark(
            model, testing_data_csv_path, node_assignments, 
            adjacency_matrix, node_type_indices, variable_names,
            n_errors, detection_threshold
        )
        
    except Exception as e:
        print(f"❌ Error during detailed benchmark: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STEP 4: Run multiple cycles for statistical analysis
    print("\n" + "="*60)
    print("STEP 4: MULTIPLE CYCLE ANALYSIS")
    print("="*60)
    
    total_errors = 0
    total_correct_detections = 0
    total_false_positives = 0
    results = []
    cycle_data = []
    
    print(f"Running {n_cycles} cycles for statistical analysis...")
    
    for i in range(n_cycles):
        try:
            correct_detections, false_positives = gnn_prediction_benchmark(
                model, testing_data_csv_path, node_assignments, adjacency_matrix,
                node_type_indices, variable_names, window_size, n_errors, detection_threshold
            )
            
            print(f"Cycle {i+1:2d}: {correct_detections} correct detections, {false_positives} false positives")
            results.append(correct_detections)
            cycle_data.append((correct_detections, false_positives))
            total_errors += n_errors
            total_correct_detections += correct_detections
            total_false_positives += false_positives
            
        except Exception as e:
            print(f"❌ Error in cycle {i+1}: {e}")
            continue
    
    # STEP 5: Analyze detection performance across cycles
    print("\n" + "="*60)
    print("STEP 5: DETECTION PERFORMANCE ANALYSIS")
    print("="*60)
    
    try:
        analyze_detection_performance(cycle_data, detection_threshold)
    except Exception as e:
        print(f"❌ Error during performance analysis: {e}")
    
    # STEP 6: Final results
    print("\n" + "="*80)
    print("🎯 FINAL COMPREHENSIVE RESULTS")
    print("="*80)
    
    # Calculate metrics
    precision = total_correct_detections / (total_correct_detections + total_false_positives) if (total_correct_detections + total_false_positives) > 0 else 0
    recall = total_correct_detections / total_errors if total_errors > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Configuration:")
    print(f"  Detection threshold: {detection_threshold}")
    print(f"  Errors per cycle: {n_errors}")
    print(f"  Cycles completed: {len(cycle_data)}")
    print(f"  Window size: {window_size}")
    
    print(f"\nOverall Performance:")
    print(f"  Total errors inserted: {total_errors}")
    print(f"  Total correct detections (True Positives): {total_correct_detections}")
    print(f"  Total false positives: {total_false_positives}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    
    if results:
        print(f"\nDetection Statistics:")
        print(f"  Mean correct detections per cycle: {np.mean(results):.2f}")
        print(f"  Std correct detections per cycle: {np.std(results):.2f}")
        print(f"  Min/Max correct detections: {np.min(results)}/{np.max(results)}")
    
    print(f"\n🔍 Model Health Summary:")
    if diagnostic_errors:
        print(f"  Model diagnostic tests: {len(diagnostic_errors)} completed")
        for test_name, errors in diagnostic_errors:
            avg_error = np.mean(errors)
            if avg_error > 10:
                print(f"  ❌ {test_name}: Very high reconstruction error ({avg_error:.3f})")
            elif avg_error > 1:
                print(f"  ⚠️  {test_name}: High reconstruction error ({avg_error:.3f})")
            else:
                print(f"  ✅ {test_name}: Reasonable reconstruction error ({avg_error:.3f})")
    
    # Model performance assessment
    if recall < 0.1:
        print(f"  ❌ CRITICAL: Very low recall ({recall:.3f}) - model barely detecting errors")
    elif recall < 0.5:
        print(f"  ⚠️  WARNING: Low recall ({recall:.3f}) - model missing many errors")
    else:
        print(f"  ✅ Recall is reasonable ({recall:.3f})")
        
    if precision < 0.1:
        print(f"  ❌ CRITICAL: Very low precision ({precision:.3f}) - many false alarms")
    elif precision < 0.5:
        print(f"  ⚠️  WARNING: Low precision ({precision:.3f}) - some false alarms")
    else:
        print(f"  ✅ Precision is reasonable ({precision:.3f})")

if __name__ == "__main__":
    main()