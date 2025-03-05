import numpy as np
import streamlit as st

# Load both models
forward_data = np.load("forward_model.npz")
W1_forward, b1_forward, W2_forward, b2_forward = forward_data["W1"], forward_data["b1"], forward_data["W2"], forward_data["b2"]

backprop_data = np.load("backprop_model.npz")
W1_backprop, b1_backprop, W2_backprop, b2_backprop = backprop_data["W1"], backprop_data["b1"], backprop_data["W2"], backprop_data["b2"]

# Define activation functions
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
    return np.argmax(A2, axis=1)

# Streamlit UI
st.title("Wine Classification: Forward vs Backpropagation")
st.header("Predicting Wine Quality with Neural Networks")
st.write("This application allows you to input wine features and predict the wine quality class using two different neural network models: one trained with forward propagation and the other with backpropagation.")
st.write("There are 3 classes to predict: Class 0, Class 1, and Class 2.")

# User input for feature values
feature_names = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "OD280/OD315 of Diluted Wines", "Proline"]
features = [st.number_input(f"{feature_names[i]}") for i in range(W1_forward.shape[0])]

if st.button("Predict"):
    input_data = np.array(features).reshape(1, -1)  # Reshape for model input

    # Predictions
    pred_forward = forward_propagation(input_data, W1_forward, b1_forward, W2_forward, b2_forward)
    pred_backprop = forward_propagation(input_data, W1_backprop, b1_backprop, W2_backprop, b2_backprop)

    # Display results
    st.write(f"Prediction (Forward Only): Class {pred_forward[0]}")
    st.write(f"Prediction (Backpropagation): Class {pred_backprop[0]}")