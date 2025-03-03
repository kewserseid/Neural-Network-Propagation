# Implement backpropagation for the model previously built using forward propagation, and 
# evaluate the differences in performance between the two approaches.

# Includes both only forward propagation and forward with backward propagation
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# used load_wine dataset
wine = load_wine()
X = wine.data
y = wine.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42)

input_size = X_train.shape[1]  # has 13 features
hidden_size = 10
output_size = len(np.unique(y_train))  # has 3 classes

# Initialize weights and biases
np.random.seed(42) 
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Forward propagation function
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Backward propagation function
def backward_propagation(X, y, Z1, A1, output, W2):
    m = X.shape[0]

    dZ2 = output.copy()
    dZ2[np.arange(m), y] -= 1
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

# Evaluation: Forward Propagation Only
_, _, _, test_output_untrained = forward_propagation(X_test, W1, b1, W2, b2)
test_predictions_untrained = np.argmax(test_output_untrained, axis=1)
accuracy_untrained = accuracy_score(y_test, test_predictions_untrained)
print(f"Wine Dataset - Test Accuracy (Forward Propagation Only): {accuracy_untrained * 100:.2f}%")

# Training: Forward Propagation with Backpropagation
learning_rate = 0.01
epochs = 1000
loss_history = []

for epoch in range(epochs):
    Z1, A1, Z2, output = forward_propagation(X_train, W1, b1, W2, b2)
    
    m = X_train.shape[0]
    loss = -np.sum(np.log(output[np.arange(m), y_train])) / m
    loss_history.append(loss)
    
    dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, Z1, A1, output, W2)
    
    # Update weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluate after training
_, _, _, test_output_trained = forward_propagation(X_test, W1, b1, W2, b2)
test_predictions_trained = np.argmax(test_output_trained, axis=1)
accuracy_trained = accuracy_score(y_test, test_predictions_trained)
print(f"Wine Dataset - Test Accuracy (After Training with Backpropagation): {accuracy_trained * 100:.2f}%")