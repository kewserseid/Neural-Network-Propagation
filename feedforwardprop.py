import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data  
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

input_size = X_train.shape[1] 
hidden_size = 5  
output_size = 3  

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

# Forward propagation function
def forward_propagation(X, W1, b1, W2, b2):
    # Input to hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1) 

    # Hidden to output layer
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)  # Apply Softmax activation for multi-class classification

    return Z1, A1, Z2, A2 

learning_rate = 0.01
epochs = 1000
for epoch in range(epochs):
    Z1, A1, Z2, output = forward_propagation(X_train, W1, b1, W2, b2)
    m = X_train.shape[0]  # Number of training examples
    loss = -np.sum(np.log(output[np.arange(m), y_train])) / m  # Cross-entropy loss
    dZ2 = output
    dZ2[np.arange(m), y_train] -= 1  # Adjust for correct labels
    dW2 = np.dot(A1.T, dZ2) / m  # Gradient for W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # Gradient for b2

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)  # Derivative of ReLU
    dW1 = np.dot(X_train.T, dZ1) / m  # Gradient for W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # Gradient for b1

    # Update weights and biases using gradient descent
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

Z1_test, A1_test, Z2_test, test_output = forward_propagation(X_test, W1, b1, W2, b2)
test_predictions = np.argmax(test_output, axis=1)  

accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
