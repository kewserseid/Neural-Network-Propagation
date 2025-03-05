import numpy as np
from feedforwardprop import forward_propagation

# Perform backward propagation and compute gradients.

def backward_propagation(X, y, Z1, A1, Z2, A2, Z3, A3, Z4, output, 
                         W1, b1, W2, b2, W3, b3, W4, b4, learning_rate):
    
    m = X.shape[0] 

    # Compute cross-entropy loss
    loss = -np.sum(np.log(output[np.arange(m), y] + 1e-15)) / m
    
    # Compute gradient of the output layer
    dZ4 = output
    dZ4[np.arange(m), y] -= 1  # Adjust softmax gradient for cross-entropy loss
    
    # Compute gradients for W4 and b4
    dW4 = np.dot(A3.T, dZ4) / m
    db4 = np.sum(dZ4, axis=0, keepdims=True) / m
    
    # Backpropagate through the third layer
    
    dA3 = np.dot(dZ4, W4.T)  # Gradient of activation A3
    dZ3 = dA3 * (Z3 > 0)  # Apply ReLU derivative
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m
    
    # Backpropagate through the second layer

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * (Z2 > 0)  # Apply ReLU derivative
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    # Backpropagate through the first layer

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)  # Apply ReLU derivative
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    # Update weights and biases using gradient descent
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4
    
    return W1, b1, W2, b2, W3, b3, W4, b4, loss

# Train the neural network with backpropagation.

def train(X_train, y_train, W1, b1, W2, b2, W3, b3, W4, b4, epochs, learning_rate):
    
    for epoch in range(epochs):
        # Forward pass to compute activations and output
        Z1, A1, Z2, A2, Z3, A3, Z4, output = forward_propagation(X_train, W1, b1, W2, b2, W3, b3, W4, b4)
        
        # Backward pass to compute gradients and update weights
        W1, b1, W2, b2, W3, b3, W4, b4, loss = backward_propagation(
                X_train, y_train, Z1, A1, Z2, A2, Z3, A3, Z4, output, 
                W1, b1, W2, b2, W3, b3, W4, b4, learning_rate
            )
        
        # Print loss every 10 epochs for monitoring training progress
        if epoch % 10 == 0:
            print(f"Backprop Epoch {epoch}, Loss: {loss:.4f}")
    
    return W1, b1, W2, b2, W3, b3, W4, b4
