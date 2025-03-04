# Implement backpropagation for the model previously built using forward propagation, and 
# evaluate the differences in performance between the two approaches.
import numpy as np
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
b1 = np.zeros((1, hidden_size))  
W2 = np.random.randn(hidden_size, output_size)  
b2 = np.zeros((1, output_size))  

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1) 
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)  
    return Z1, A1, Z2, A2

def backpropagation(X, y, Z1, A1, Z2, A2, W1, W2, b1, b2, learning_rate):
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
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2, loss

learning_rate = 0.01
epochs = 1000
for epoch in range(epochs):
    Z1, A1, Z2, output = forward_propagation(X_train, W1, b1, W2, b2)
    W1, b1, W2, b2, loss = backpropagation(X_train, y_train, Z1, A1, Z2, output, W1, W2, b1, b2, learning_rate)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

Z1_test, A1_test, Z2_test, test_output = forward_propagation(X_test, W1, b1, W2, b2)
test_predictions = np.argmax(test_output, axis=1)
accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
