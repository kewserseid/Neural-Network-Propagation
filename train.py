import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from forward_propagation import forward_propagation
from backward_propagation import backward_propagation

wine = datasets.load_wine()
X = wine.data
y = wine.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

input_size = X_train.shape[1]
hidden_size = 6
output_size = len(np.unique(y))

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    Z1, A1, Z2, output = forward_propagation(X_train, W1, b1, W2, b2)
    dW1, db1, dW2, db2, loss = backward_propagation(X_train, y_train, Z1, A1, Z2, output, W2)
    
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
