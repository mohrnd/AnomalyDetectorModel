import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np

# https://blog.tensorflow.org/2021/11/introducing-tensorflow-gnn.html
class WeightedSumConvolution(tf.keras.layers.Layer):
    """Weighted sum of source nodes states."""

    def call(self, graph: tfgnn.GraphTensor, edge_set_name: tfgnn.EdgeSetName) -> tfgnn.Field:
        messages = tfgnn.broadcast_node_to_edges(
            graph,
            edge_set_name,
            tfgnn.SOURCE,
            feature_name=tfgnn.DEFAULT_STATE_NAME)
        weights = graph.edge_sets[edge_set_name]['weight']
        weighted_messages = tf.expand_dims(weights, -1) * messages
        pooled_messages = tfgnn.pool_edges_to_node(
            graph,
            edge_set_name,
            tfgnn.TARGET,
            reduce_type='sum',
            feature_value=weighted_messages)
        return pooled_messages

# Model hyper-parameters
h_dims = {'user': 256, 'movie': 64, 'genre': 128}

gnn = tfgnn.keras.ConvGNNBuilder(
    lambda edge_set_name: WeightedSumConvolution(),
    lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
        tf.keras.layers.Dense(h_dims[node_set_name], activation='relu'))
)

# Build the model with two rounds of message passing
model = tf.keras.models.Sequential([
    gnn.Convolve({'genre'}),  # sends messages from movie to genre
    gnn.Convolve({'user'}),   # sends messages from movie and genre to users
    tfgnn.keras.layers.Readout(node_set_name="user"),
    tf.keras.layers.Dense(1, activation='sigmoid')  # For binary prediction
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def create_sample_graph():
    """Create a sample heterogeneous graph with users, movies, and genres."""
    
    # Node features
    user_features = tf.constant([[1.0, 0.5], [0.8, 0.3], [0.2, 0.9]], dtype=tf.float32)  # 3 users
    movie_features = tf.constant([[0.5, 0.5, 0.1], [0.9, 0.2, 0.8]], dtype=tf.float32)   # 2 movies
    genre_features = tf.constant([[1.0, 0.0]], dtype=tf.float32)                          # 1 genre
    
    # Edge connections and weights
    # movie -> genre edges (movies belong to genres)
    movie_to_genre_source = tf.constant([0, 1], dtype=tf.int32)  # both movies connect to genre 0
    movie_to_genre_target = tf.constant([0, 0], dtype=tf.int32)
    movie_to_genre_weights = tf.constant([0.8, 0.6], dtype=tf.float32)
    
    # movie -> user edges (users rate movies)
    movie_to_user_source = tf.constant([0, 0, 1, 1], dtype=tf.int32)  # movie connections
    movie_to_user_target = tf.constant([0, 1, 1, 2], dtype=tf.int32)  # user connections
    movie_to_user_weights = tf.constant([0.9, 0.7, 0.8, 0.5], dtype=tf.float32)
    
    # genre -> user edges (users like certain genres)
    genre_to_user_source = tf.constant([0, 0], dtype=tf.int32)  # genre 0 connects to users
    genre_to_user_target = tf.constant([0, 2], dtype=tf.int32)  # users 0 and 2
    genre_to_user_weights = tf.constant([0.6, 0.4], dtype=tf.float32)
    
    # Create the graph tensor
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'user': tfgnn.NodeSet.from_fields(
                sizes=tf.constant([3]),
                features={tfgnn.DEFAULT_STATE_NAME: user_features}
            ),
            'movie': tfgnn.NodeSet.from_fields(
                sizes=tf.constant([2]),
                features={tfgnn.DEFAULT_STATE_NAME: movie_features}
            ),
            'genre': tfgnn.NodeSet.from_fields(
                sizes=tf.constant([1]),
                features={tfgnn.DEFAULT_STATE_NAME: genre_features}
            )
        },
        edge_sets={
            'movie_to_genre': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([2]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('movie', movie_to_genre_source),
                    target=('genre', movie_to_genre_target)
                ),
                features={'weight': movie_to_genre_weights}
            ),
            'movie_to_user': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([4]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('movie', movie_to_user_source),
                    target=('user', movie_to_user_target)
                ),
                features={'weight': movie_to_user_weights}
            ),
            'genre_to_user': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([2]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('genre', genre_to_user_source),
                    target=('user', genre_to_user_target)
                ),
                features={'weight': genre_to_user_weights}
            )
        }
    )
    
    return graph

# Create sample data
sample_graph = create_sample_graph()
print("Sample graph created successfully!")
print(f"Users: {sample_graph.node_sets['user'].sizes}")
print(f"Movies: {sample_graph.node_sets['movie'].sizes}")
print(f"Genres: {sample_graph.node_sets['genre'].sizes}")

# Test the model
print("\nTesting the model...")
try:
    # Make a prediction
    prediction = model(sample_graph)
    print(f"Model output shape: {prediction.shape}")
    print(f"Sample predictions: {prediction.numpy().flatten()}")
    
    # Create some dummy labels for training demonstration
    labels = tf.constant([[1.0], [0.0], [1.0]], dtype=tf.float32)  # 3 users, binary labels
    
    # Train for a few steps
    print("\nTraining for a few steps...")
    for step in range(5):
        with tf.GradientTape() as tape:
            predictions = model(sample_graph)
            loss = tf.keras.losses.binary_crossentropy(labels, predictions)
            mean_loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(mean_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(f"Step {step + 1}, Loss: {mean_loss.numpy():.4f}")
    
    print("\nGNN model is working successfully!")
    
except Exception as e:
    print(f"Error: {e}")