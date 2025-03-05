import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score 


"""
    I picked sklearn's load_breast_cancer dataset. It has 30 features and binary target, whether the indicated cell is infected or not.
    Since the logloss(binary cross-entropy loss) is suitable for binary classifiers,I used it as the cost function.
    The first hiddle layer uses the Relu function as its activation function, while the second one uses Sigmoid function. I normalized the input data
    to prevent outliers and higher learning rate.
"""

np.random.seed(2)

data = load_breast_cancer()
X = data.data
X_normalized = (X - np.mean(X))/ np.std(X)
y = data.target

X_train , X_test, y_train , y_test = train_test_split(X_normalized, y)

variables = {"a1": np.array([]) , "a2": np.array([])  , "w1": np.array([]) , "w2": np.array([]) , "b1": np.array([]) , "b2": np.array([]) }

X_train = X_train.T


#initialization for the datasets with 30 input layers, 2 hidden layers and 1 output layers
#The biases are initialized to 0

variables["w1"] = np.random.randn(2, X_train.shape[0]) * 0.01
variables["w2"] = np.random.randn(1, 2) * 0.01
variables["b1"] = np.zeros((2, 1))
variables["b2"] = np.zeros((1, 1))



def relu_fn(z):
    return np.maximum(0, z)
def sigmoid_fn(z):
    return 1 / (1 + np.exp(-z))

def log_loss_derivative(X, y, variables):
    dimension = X.shape[0]
    dLogLossdW2 =  (1 / dimension) * np.dot((variables["a2"] - y),  variables["a1"].T)
    dLogLossdW1 =  (1 / dimension) * np.dot(np.dot(variables["w2"].T * (variables["a2"] - y), np.dot(variables["a1"].T, 1 - variables["a1"])) ,  X.T)
    dLogLossdb2 =  (1 / dimension) * np.dot((variables["a2"] - y) , np.ones((y.shape[0], 1)))
    dLogLossdb1 =  (1 / dimension) * np.dot(np.dot(variables["w2"].T * (variables["a2"] - y), np.dot(variables["a1"].T, 1 - variables["a1"])),  np.ones((y.shape[0], 1)))

    return (dLogLossdb1, dLogLossdb2, dLogLossdW1, dLogLossdW2)


def forward_propagation(X, variables):
    z1 = np.dot(variables["w1"], X) + variables["b1"]
    variables["a1"] = relu_fn(z1)
    z2 = np.dot(variables["w2"], variables["a1"]) + variables["b2"]
    variables["a2"] = sigmoid_fn(z2)
    return variables
    
def backward_propagation(X, y, variables):
    learning_rate = 0.0015
    for _ in range(1000):
        dLogLossdb1, dLogLossdb2, dLogLossdW1, dLogLossdW2 = log_loss_derivative(X, y, variables)
        variables["w1"] -= learning_rate * dLogLossdW1 
        variables["w2"] -= learning_rate * dLogLossdW2
        variables["b1"] -= learning_rate * dLogLossdb1
        variables["b2"] -= learning_rate * dLogLossdb2
    return variables


variables = forward_propagation(X_train, variables)
"""The following result inidicate the accuracy of forward propagation without the backward propagation"""
prediction =  np.round(variables["a2"]).T

print("Accuracy :", end = " ")
print(accuracy_score(y_train, prediction) * 100)

variables = backward_propagation(X_train,y_train,  variables)
variables = forward_propagation(X_train, variables)


"""The following result inidicate the accuracy of forward propagation after updating the weights and biases with backward propagation"""


prediction =  np.round(variables["a2"]).T
print("Accuracy :", end = " ")
print(accuracy_score(y_train, prediction) * 100)

"""Even though it was expected the backward propagation would result in more accuracy, for this case the forward propagation is  exhibited to have better accuracy"""