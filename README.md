# 🧠 CNN Forward & Backward Propagation - Handwritten Digits Classification  

This project implements **Convolutional Neural Networks (CNNs)** to classify handwritten digits (0-9) using the **MNIST-like dataset** from `train.csv` and `test.csv`.  

## 📖 **Project Overview**  
We perform:  
✔ **Forward Propagation** – Pass data through CNN to make predictions.  
✔ **Backward Propagation** – Compute gradients to update model weights.  
✔ **Comparison** – Visualize differences between forward and backward propagation.  

---

## 📊 **Dataset Description**  
- **Source:** The dataset contains **grayscale images** of handwritten digits (0-9).  
- **Image Size:** `28 × 28 pixels` (Flattened into `784` features).  
- **Format:**  
    - `train.csv` → **785 columns** (`1 label + 784 pixel values`).  
    - `test.csv` → **784 columns** (Only pixel values, no labels).  

### 📌 **Dataset Example**  
| Label | pixel0 | pixel1 | ... | pixel782 | pixel783 |
|--------|--------|--------|----|----------|----------|
| 5      | 0      | 0      | ... | 12       | 0        |
| 3      | 0      | 10     | ... | 50       | 0        |

---

## 🔥 **Forward Propagation in CNN**  
**Definition:**  
Forward propagation is the process where input images pass **through CNN layers** to generate predictions.  

### **Steps in Forward Propagation**  
1. **Input Layer** – Raw `28x28` image pixels.  
2. **Convolutional Layers** – Extract features using filters.  
3. **Activation Functions (ReLU)** – Introduce non-linearity.  
4. **Pooling Layers** – Reduce spatial dimensions (downsampling).  
5. **Fully Connected (FC) Layers** – Perform classification.  
6. **Output Layer** – Predict digit (0-9) using **Softmax activation**.  

### 📌 **Forward Propagation Visualization**  
We visualize how the CNN **extracts features** layer by layer.  

```python
visualize_forward(model, sample_image)
