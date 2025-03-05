import numpy as np

# Defining Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Forward propagation through a 3-hidden-layer neural network.
def forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4):

    # First layer, Linear - ReLU
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    
    # Second layer, Linear - ReLU
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    
    # Third layer, Linear - ReLU
    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)
    
    # Output Layer: Linear - Softmax
    Z4 = np.dot(A3, W4) + b4
    A4 = softmax(Z4)
    
    return Z1, A1, Z2, A2, Z3, A3, Z4, A4

# Initialize weights and biases for the network.
def initialize_parameters(input_size, hidden1_size, hidden2_size, hidden3_size, output_size, seed=42):
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Initialize weights with small random values and biases with zeros
    W1 = np.random.randn(input_size, hidden1_size) * 0.01
    b1 = np.zeros((1, hidden1_size))
    W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
    b2 = np.zeros((1, hidden2_size))
    W3 = np.random.randn(hidden2_size, hidden3_size) * 0.01
    b3 = np.zeros((1, hidden3_size))
    W4 = np.random.randn(hidden3_size, output_size) * 0.01
    b4 = np.zeros((1, output_size))
    
    return W1, b1, W2, b2, W3, b3, W4, b4
