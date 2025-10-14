
#  Machine Learning Algorithms from Scratch  

This repository contains implementations of core **Machine Learning algorithms** — built completely **from scratch** using only **NumPy and Pandas** (no pre-built ML libraries).  
Each project demonstrates how fundamental algorithms work internally, including their mathematics, data preprocessing, and training loops.

---

##  Table of Contents
1. [Project 1: Linear Regression](#project-1-linear-regression)
2. [Project 2: Logistic Regression](#project-2-logistic-regression)
3. [Project 3: K-Means Clustering](#project-3-k-means-clustering)
4. [Project 4: Support Vector Machine (SVM)](#project-4-support-vector-machine-svm)
5. [Summary](#summary)

---

##  Project 1: Linear Regression (House Price Prediction)

###  Objective
Predict house prices using **Linear Regression**, implemented entirely from scratch using **Gradient Descent**, **Matrix Operations**, and **Mean Squared Error (MSE)** for optimization.

###  Libraries Used
```python
import numpy as np
import pandas as pd
```

- **NumPy** → for mathematical operations (arrays, matrices, mean, etc.)  
- **Pandas** → for loading and handling datasets  

###  Steps

1. **Load Dataset**
   ```python
   data = pd.read_csv('HousingData.csv')
   X = data.drop('MEDV', axis=1).values
   y = data['MEDV'].values.reshape(-1, 1)
   ```

2. **Normalize Data**
   ```python
   X = (X - X.mean(axis=0)) / X.std(axis=0)
   ```

3. **Train-Test Split**
   ```python
   def train_test_split(X, y, test_size=0.2):
       n = len(X)
       idx = np.arange(n)
       np.random.shuffle(idx)
       split = int(n * (1 - test_size))
       return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]
   ```

4. **Initialize Parameters**
   ```python
   W = np.zeros((X_train.shape[1], 1))
   b = 0
   lr = 0.01
   epochs = 500
   ```

5. **Train Using Gradient Descent**
   ```python
   for i in range(epochs):
       y_pred = X_train @ W + b
       error = y_pred - y_train
       loss = np.mean(error ** 2)
       dW = (2/len(X_train)) * (X_train.T @ error)
       db = (2/len(X_train)) * np.sum(error)
       W -= lr * dW
       b -= lr * db
   ```

6. **Evaluate**
   ```python
   y_pred_test = X_test @ W + b
   mse = np.mean((y_test - y_pred_test)**2)
   r2 = 1 - np.sum((y_test - y_pred_test)**2) / np.sum((y_test - y_test.mean())**2)
   ```

**Output Example**
```
Epoch 0, Loss: 532.74
Epoch 100, Loss: 50.43
Test MSE: 25.84
R2: 0.86
```

---

## Project 2: Logistic Regression (Iris Classification)

###  Objective
Classify iris flower species using **Logistic Regression** — a probabilistic linear model trained with **Binary Cross-Entropy Loss**.

###  Steps

1. **Load and Preprocess Data**
   ```python
   iris = pd.read_csv('iris_dataset.csv')
   X = iris.iloc[:, :-1].values
   y = iris.iloc[:, -1].values
   ```

2. **Encode & Normalize**
   ```python
   X = (X - X.mean(axis=0)) / X.std(axis=0)
   ```

3. **Initialize Parameters**
   ```python
   W = np.zeros((X_train.shape[1], 1))
   b = 0
   lr = 0.1
   epochs = 500
   ```

4. **Sigmoid Function**
   ```python
   def sigmoid(z):
       return 1 / (1 + np.exp(-z))
   ```

5. **Training**
   ```python
   for i in range(epochs):
       z = X_train @ W + b
       y_pred = sigmoid(z)
       loss = -np.mean(y_train*np.log(y_pred+1e-8) + (1-y_train)*np.log(1-y_pred+1e-8))
       dW = (1/len(X_train)) * (X_train.T @ (y_pred - y_train))
       db = (1/len(X_train)) * np.sum(y_pred - y_train)
       W -= lr * dW
       b -= lr * db
   ```

6. **Evaluation**
   ```python
   y_pred_test = sigmoid(X_test @ W + b) >= 0.5
   acc = np.mean(y_pred_test == y_test)
   ```

**Output Example**
```
Epoch 0, Loss: 0.6931
Epoch 100, Loss: 0.2789
Test Accuracy: 0.92
```

---

##  Project 3: K-Means Clustering (Customer Segmentation)

###  Objective
Cluster mall customers into groups based on **Annual Income** and **Spending Score**.

###  Steps

1. **Load Dataset**
   ```python
   mall = pd.read_csv('Mall_Customers.csv')
   X = mall[['Annual Income (k$)', 'Spending Score (1-100)']].values
   ```

2. **K-Means Concept**
   - Choose number of clusters (e.g., k = 3)
   - Randomly initialize centroids
   - Assign points to nearest centroid
   - Recalculate centroids as mean of clusters
   - Repeat until centroids stabilize

---

##  Project 4: Support Vector Machine (SVM)

###  Objective
Implement a **Support Vector Machine** from scratch for **binary classification**, finding the hyperplane that maximizes the margin between two classes.

###  Key Concepts
```
Loss = (1/N) * Σ max(0, 1 - y*(w·x + b)) + λ‖W‖²
```

###  Implementation
```python
import numpy as np

X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values
labels, y = np.unique(y, return_inverse=True)
mask = y < 2
X, y = X[mask], y[mask]
y = np.where(y == 0, -1, 1)

X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1,1))
y_train, y_test = y_train.flatten(), y_test.flatten()

w = np.zeros(X_train.shape[1])
b = 0.0

def svm_predict(X):
    scores = X @ w + b
    return np.where(scores >= 0, 1, -1)
```

###  Evaluation
```python
accuracy = np.mean(y_pred == y_test)
```

---

##  Summary
| Algorithm | Type | Core Concept | From Scratch |
|------------|------|---------------|---------------|
| Linear Regression | Supervised | Gradient Descent + MSE |
| Logistic Regression | Supervised | Sigmoid + Cross-Entropy |
| K-Means | Unsupervised | Distance + Centroid Update |
| SVM | Supervised | Hinge Loss + Margin Maximization |

---

## Author
**Prem**  
_Data Scientist | ML Engineer | Software Developer (C++, C, Python)_  
