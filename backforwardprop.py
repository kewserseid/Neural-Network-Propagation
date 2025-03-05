# Implement backpropagation for the model previously built using forward propagation, and 
# evaluate the differences in performance between the two approaches.
import numpy as np
from feedforwardprop import forward_propagation

def backward_propagation(X, y, Z1, A1, Z2, A2, Z3, A3, Z4, output, W1, b1, W2, b2, W3, b3, W4, b4, learning_rate):

    m = X.shape[0]
    loss = -np.sum(np.log(output[np.arange(m), y] + 1e-15)) / m
    total_loss = loss

    dZ4 = output
    dZ4[np.arange(m), y] -= 1
    dW4 = np.dot(A3.T, dZ4) / m
    db4 = np.sum(dZ4, axis=0, keepdims=True) / m
    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * (Z3 > 0)
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * (Z2 > 0)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4

    return W1, b1, W2, b2, W3, b3, W4, b4, total_loss


def train(X_train, y_train, W1, b1, W2, b2, W3, b3, W4, b4, epochs, learning_rate):
    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3, Z4, output = forward_propagation(X_train, W1, b1, W2, b2, W3, b3, W4, b4)
        W1, b1, W2, b2, W3, b3, W4, b4, total_loss = backward_propagation(
            X_train, y_train, Z1, A1, Z2, A2, Z3, A3, Z4, output, W1, W2, W3, W4, learning_rate)
        if epoch % 10 == 0:
            print(f"Backprop Epoch {epoch}, Total Loss: {total_loss:.4f}")
    return W1, b1, W2, b2, W3, b3, W4, b4