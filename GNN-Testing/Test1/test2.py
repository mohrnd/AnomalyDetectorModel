import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
from keras.saving import register_keras_serializable

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("\n\n\nGPU memory growth enabled. Mixed precision set.\n\n\n")
    except Exception as e:
        print("Failed to enable memory growth or mixed precision:", e)

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

@register_keras_serializable()
class MessagePassing(tf.keras.layers.Layer):
    def __init__(self, embedding_dim: int, message_dim: int = 32, name: str = "message_passing"):
        super(MessagePassing, self).__init__(name=name)
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

@register_keras_serializable()
class IndustrialGNN(tf.keras.Model):
    def __init__(self, node_types: List[str], embedding_dim: int = 64, num_gnn_layers: int = 2, edge_dim: int = 3):
        super(IndustrialGNN, self).__init__()
        self.node_types = node_types
        self.embedding_dim = embedding_dim
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

def train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    csv_data_path = os.path.join(BASE_DIR, 'training_data.csv')
    with open(csv_data_path, 'r') as f:
        csv_data_str = f.read()

    windowed_data, node_assignments, variable_names = prepare_data(csv_data_str)
    print(f"Data shape: {windowed_data.shape}")
    print(f"Node assignments: {node_assignments}")

    node_types = ['PLC', 'Sensor', 'Actuator']
    model = IndustrialGNN(node_types=node_types, embedding_dim=64, num_gnn_layers=2)
    adjacency_matrix, node_type_indices = model.create_graph_structure(node_assignments)

    assert len(variable_names) == len(node_type_indices), "Mismatch between variables and node type indices"
    num_nodes = len(variable_names)

    node_values = np.mean(windowed_data, axis=1) # ERROR HERE, it is averaging all of the values in the 2 sec timeframe
    node_values = np.expand_dims(node_values, axis=-1)

    dataset = tf.data.Dataset.from_tensor_slices({
        'node_values': tf.constant(node_values, dtype=tf.float32)
    }).batch(8)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def train_step(batch_node_values):
        inputs = {
            'node_values': batch_node_values,
            'node_types': node_type_indices,
            'adjacency_matrix': adjacency_matrix,
            'edge_embeddings': adjacency_matrix
        }
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(outputs['reconstructions'] - outputs['initial_embeddings']))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    print("Starting training...")
    for epoch in tqdm(range(40)):
        total_loss = 0.0
        for batch in dataset:
            loss = train_step(batch['node_values'])
            total_loss += loss.numpy()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    print("Training completed!")

    # Evaluation on entire set
    inputs = {
        'node_values': tf.constant(node_values, dtype=tf.float32),
        'node_types': node_type_indices,
        'adjacency_matrix': adjacency_matrix,
        'edge_embeddings': adjacency_matrix
    }
    outputs = model(inputs, training=False)
    final_loss = tf.reduce_mean(tf.square(outputs['reconstructions'] - outputs['initial_embeddings']))
    print(f"Final embeddings shape: {outputs['embeddings'].shape}")
    print(f"Reconstruction loss: {final_loss:.4f}")

    return model, inputs, variable_names

if __name__ == "__main__":
    model, inputs, variable_names = train_model()
    model.save('ver2.keras')
