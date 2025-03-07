# CNN Forward vs Backpropagation: A Comparative Study

## Overview

This project demonstrates the **difference between a CNN with and without backpropagation** by comparing their performance on the **MNIST handwritten digit dataset**.

We implement:

- A **forward-only CNN** (random weights, no backpropagation)
- A **trained CNN** (uses backpropagation to adjust weights)

By visually and quantitatively analyzing the results, we highlight the **importance of backpropagation** in training deep neural networks.

---

## Dataset

The dataset consists of **grayscale images of handwritten digits (0-9)** from the MNIST dataset.

- **Image size:** 28x28 pixels
- **Pixel range:** 0 to 255 (grayscale intensity)
- **Train set:** 60,000 images
- **Test set:** 10,000 images

Each sample has **784 features** (28x28 pixels) and a **label** indicating the digit.

---

## Model Architecture

Both models share the same CNN architecture:

1. **Convolutional Layer 1**: 16 filters, 3x3 kernel, ReLU activation
2. **Max Pooling Layer**: 2x2 kernel
3. **Convolutional Layer 2**: 32 filters, 3x3 kernel, ReLU activation
4. **Max Pooling Layer**: 2x2 kernel
5. **Fully Connected Layer**: 128 neurons, ReLU activation
6. **Output Layer**: 10 neurons (Softmax activation for classification)

---

## Methodology

### **1️⃣ Forward-Only CNN (Without Backpropagation)**

- The model **randomly initializes weights** and runs a forward pass.
- No weight updates occur (i.e., no learning).
- Predictions are compared to actual labels to observe random classification.

### **2️⃣ Trained CNN (With Backpropagation)**

- The model undergoes training using **cross-entropy loss and SGD optimizer**.
- **Gradient descent updates** weights using backpropagation.
- Performance is evaluated after training.

---

## Results & Comparison

| Feature                  | **Without Backpropagation** | **With Backpropagation**     |
| ------------------------ | --------------------------- | ---------------------------- |
| **Test Accuracy**        | **9.12%** (random guessing) | **98.89%** (trained model)   |
| **Feature Maps**         | Random, unstructured        | Detects edges, digits        |
| **Activations**          | Weak, noisy response        | Focuses on relevant features |
| **Loss Trend**           | Constant (no learning)      | Decreases over epochs        |
| **Accuracy Improvement** | None                        | Steadily increases           |

### **📌 Visualizations**

#### **1️⃣ Sample Predictions Before & After Training**

| Before Training (Random Weights) | After Training (Learned Weights) |
| -------------------------------- | -------------------------------- |
|                                  |                                  |

#### **2️⃣ Feature Maps (Filters) Before & After Training**

| Before Training | After Training |
| --------------- | -------------- |
|                 |                |

#### **3️⃣ Activation Maps (Neural Responses)**

| Before Training | After Training |
| --------------- | -------------- |
|                 |                |

#### **4️⃣ Loss & Accuracy Over Time**

| Loss Reduction | Accuracy Improvement |
| -------------- | -------------------- |
|                |                      |

---

## Conclusion

### **Key Findings**

- **Without backpropagation, CNNs cannot learn** meaningful patterns from data.
- **Trained CNNs develop structured feature maps**, enabling accurate classification.
- **Backpropagation significantly improves performance** from **random guessing (9.12%) to state-of-the-art accuracy (98.89%)**.

### **Takeaway**

💡 **Backpropagation is the key mechanism that allows CNNs to learn and generalize from data, making deep learning practical and effective!**

---

## Running the Notebook

### **Requirements**

- Python 3.8+
- PyTorch
- Matplotlib
- Jupyter Notebook

### **Installation**

```bash
pip install torch torchvision matplotlib jupyter
```

### **Usage**

1. Run the Jupyter Notebook: `jupyter notebook`
2. Open `cnn_forward_vs_backprop.ipynb`
3. Execute the cells to see comparisons and visualizations.

---

## Credits

- **Author:** Ofgeha Gelana
- **Dataset:** [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- **Framework:** PyTorch

🚀 **Happy Learning!** 🎯

