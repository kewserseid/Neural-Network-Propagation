# Implement backpropagation for the model previously built using forward propagation, and
# evaluate the differences in performance between the two approaches.

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
y_encoded = np.eye(len(np.unique(y_encoded)))[y_encoded]

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Activation Functions
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Loss Function
def cross_entropy(y_pred, y_true):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Adding epsilon to avoid log(0)


# Forward Propagation
def forward_propagation(X, w1, b1, w2, b2):
    z1 = np.dot(X, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


# Backward Propagation
def backward_propagation(X, y, w1, b1, w2, b2, z1, a1, z2, a2, learning_rate=0.01):
    m = X.shape[0]
    # Gradients
    dz2 = a2 - y
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    dz1 = np.dot(dz2, w2.T) * (z1 > 0)  # ReLU derivative
    dw1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    # Update Weights & Biases
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    return w1, b1, w2, b2


# **1. Feedforward-Only Model (No Backpropagation)**
def feedforward_only_model(X_train, y_train, X_test, y_test, epochs=1000, step_size=0.01):
    np.random.seed(42)
    w1 = np.random.randn(4, 10)
    b1 = np.zeros((1, 10))
    w2 = np.random.randn(10, 3)
    b2 = np.zeros((1, 3))

    for epoch in range(epochs):
        _, _, _, a2 = forward_propagation(X_train, w1, b1, w2, b2)
        loss = cross_entropy(a2, y_train)

        if epoch % 100 == 0:
            print(f"[Feedforward] Epoch {epoch}, Loss: {loss:.4f}")

    # Evaluate Model
    _, _, _, test_output = forward_propagation(X_test, w1, b1, w2, b2)
    test_predictions = np.argmax(test_output, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, test_predictions)
    print(f"Feedforward-Only Model Test Accuracy: {accuracy * 100:.2f}%")


# Model 2: Training with Forward & Backward Propagation
def train_model(X_train, y_train, X_test, y_test, epochs=1000, learning_rate=0.01):
    np.random.seed(42)
    w1 = np.random.randn(4, 10)
    b1 = np.zeros((1, 10))
    w2 = np.random.randn(10, 3)
    b2 = np.zeros((1, 3))

    # Training Loop
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward_propagation(X_train, w1, b1, w2, b2)
        loss = cross_entropy(a2, y_train)
        w1, b1, w2, b2 = backward_propagation(X_train, y_train, w1, b1, w2, b2, z1, a1, z2, a2, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Evaluate Model
    _, _, _, test_output = forward_propagation(X_test, w1, b1, w2, b2)
    test_predictions = np.argmax(test_output, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, test_predictions)
    print(f"Trained Model Test Accuracy: {accuracy * 100:.2f}%")


# Run Both Models
print("\nRunning FeedForward-Only Model")
feedforward_only_model(X_train, y_train, X_test, y_test, 1000)
# Running FeedForward-Only Model
# [Feedforward] Epoch 0, Loss: 1.2293
# [Feedforward] Epoch 100, Loss: 1.2293
# [Feedforward] Epoch 200, Loss: 1.2293
# [Feedforward] Epoch 300, Loss: 1.2293
# [Feedforward] Epoch 400, Loss: 1.2293
# [Feedforward] Epoch 500, Loss: 1.2293
# [Feedforward] Epoch 600, Loss: 1.2293
# [Feedforward] Epoch 700, Loss: 1.2293
# [Feedforward] Epoch 800, Loss: 1.2293
# [Feedforward] Epoch 900, Loss: 1.2293
# Feedforward-Only Model Test Accuracy: 43.33%


print("\nTraining and Evaluating Backpropagation Model:")
train_model(X_train, y_train, X_test, y_test, 1000)
# Training and Evaluating Backpropagation Model:
# Epoch 0, Loss: 1.2293
# Epoch 100, Loss: 0.4524
# Epoch 200, Loss: 0.3542
# Epoch 300, Loss: 0.3031
# Epoch 400, Loss: 0.2684
# Epoch 500, Loss: 0.2437
# Epoch 600, Loss: 0.2248
# Epoch 700, Loss: 0.2094
# Epoch 800, Loss: 0.1963
# Epoch 900, Loss: 0.1849
# Trained Model Test Accuracy: 96.67%
