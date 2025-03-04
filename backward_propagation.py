import numpy as np

def backward_propagation(X, y, Z1, A1, Z2, A2, W2):
    m = X.shape[0]
    loss = -np.sum(np.log(A2[np.arange(m), y])) / m
    dZ2 = A2
    dZ2[np.arange(m), y] -= 1
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2, loss
