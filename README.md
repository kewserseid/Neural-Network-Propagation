# Neural-Network-Forward-Propagation

## Description

This work is about comparing the training of a neural network with and without backpropagation. It includes implementations of forward propagation, backward propagation, and training algorithms, as well as a demonstration of training the neural network on the MNIST dataset(its subset). The goal is to highlight the importance of backpropagation in achieving high accuracy in neural network training.

## Description of Files

- `backforwardprop.py`: Contains functions for backward propagation and training the neural network.
    - `backward_propagation`: Implements the backward propagation algorithm, which calculates the gradient of the loss function with respect to each weight by the chain rule, allowing the network to adjust the weights to minimize the loss.
    - `train`: Trains the neural network using backpropagation, iteratively updating the weights to improve the model's performance.

- `feedforwardprop.py`: Contains functions for forward propagation and initializing parameters.
    - `forward_propagation`: Implements the forward propagation algorithm, which calculates the output of the neural network by passing the input through each layer.
    - `initialize_parameters`: Initializes the parameters of the neural network, setting up the weights and biases before training begins.

- `Train_main.ipynb`: Training the neural network on the MNIST dataset.
    - Demonstrates loading a subset of the MNIST dataset, a large database of handwritten digits commonly used for training various image processing systems.
    - Preprocesses the dataset, including normalization and train test splitting, to prepare it for training.
    - Trains the neural network with and without backpropagation, providing a side-by-side comparison of the two methods.

## Summary of Findings

- **Test Accuracy (With Backpropagation)**: 98.21%
- **Test Accuracy (Without Backpropagation)**: 31.05%

The results clearly show that training the neural network with backpropagation significantly improves the test accuracy compared to training without backpropagation. This demonstrates the effectiveness of backpropagation in optimizing neural network performance by efficiently minimizing the loss function.