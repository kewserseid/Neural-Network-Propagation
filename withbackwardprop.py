
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load Wine dataset
wine = datasets.load_wine()
X = wine.data  
y = wine.target

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Define network structure
input_size = X_train.shape[1] 
hidden_size = 5  
output_size = len(np.unique(y))  # Number of classes


# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)  
b1 = np.zeros((1, hidden_size))  # Bias for hidden layer
W2 = np.random.randn(hidden_size, output_size)  # Weights between hidden and output layers
b2 = np.zeros((1, output_size))  # Bias for output layer

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1) 
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)  
    return Z1, A1, Z2, A2 

def compute_loss(output, y):
    m = y.shape[0]
    return -np.sum(np.log(output[np.arange(m), y])) / m  # Cross-entropy loss

def backpropagation(X, y, W1, b1, W2, b2, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        Z1, A1, Z2, output = forward_propagation(X, W1, b1, W2, b2)
        loss = compute_loss(output, y)
        
        m = X.shape[0]
        dZ2 = output
        dZ2[np.arange(m), y] -= 1  # Adjust for correct labels
        dW2 = np.dot(A1.T, dZ2) / m  # Gradient for W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # Gradient for b2

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * (Z1 > 0)  # Derivative of ReLU
        dW1 = np.dot(X.T, dZ1) / m  # Gradient for W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # Gradient for b1

        # Update weights and biases
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return W1, b1, W2, b2

# Train the model with backpropagation
W1, b1, W2, b2 = backpropagation(X_train, y_train, W1, b1, W2, b2)

# Evaluate the model
Z1_test, A1_test, Z2_test, test_output = forward_propagation(X_test, W1, b1, W2, b2)
test_predictions = np.argmax(test_output, axis=1)  
accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy (with backpropagation): {accuracy * 100:.2f}%")

# Save the backpropagation-trained model
np.savez("backprop_model.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Backpropagation model saved successfully!")
