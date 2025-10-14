import numpy as np
import pandas as pd

# PROJECT 1: Linear Regression (House Price Prediction)
print("\nPROJECT 1: Linear Regression")

# Load dataset
data = pd.read_csv("HousingData.csv")  
X = data.drop("MEDV", axis=1).values
y = data["MEDV"].values.reshape(-1, 1)


# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train-test split
def train_test_split(X, y, test_size=0.2):
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Initialize
n_features = X_train.shape[1]
W = np.zeros((n_features, 1))
b = 0
lr = 0.01
epochs = 500

# Training
for i in range(epochs):
    y_pred = X_train @ W + b
    error = y_pred - y_train
    loss = np.mean(error**2)

    dW = (2/len(X_train)) * (X_train.T @ error)
    db = (2/len(X_train)) * np.sum(error)

    W -= lr * dW
    b -= lr * db

    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")

# Evaluation
y_pred_test = X_test @ W + b
mse = np.mean((y_test - y_pred_test)**2)
r2 = 1 - np.sum((y_test - y_pred_test)**2) / np.sum((y_test - y_test.mean())**2)

print("Test MSE:", mse)
print("Test R2:", r2)


# PROJECT 2: Logistic Regression (Iris Classification)
print("\nPROJECT 2: Logistic Regression")

# Load dataset
iris = pd.read_csv("iris_dataset.csv")
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values   

# Encode labels (Setosa=0, Versicolor=1, Virginica=2)
labels, y = np.unique(y, return_inverse=True)

# Binary classification example (only 0 and 1 classes)
mask = y < 2
X, y = X[mask], y[mask].reshape(-1, 1)

# Normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Initialize
W = np.zeros((X_train.shape[1], 1))
b = 0
lr = 0.1
epochs = 500

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training
for i in range(epochs):
    z = X_train @ W + b
    y_pred = sigmoid(z)

    loss = -np.mean(y_train*np.log(y_pred+1e-8) + (1-y_train)*np.log(1-y_pred+1e-8))

    dW = (1/len(X_train)) * (X_train.T @ (y_pred - y_train))
    db = (1/len(X_train)) * np.sum(y_pred - y_train)

    W -= lr * dW
    b -= lr * db

    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")

# Evaluation
y_pred_test = sigmoid(X_test @ W + b) >= 0.5
acc = np.mean(y_pred_test == y_test)
print("Test Accuracy:", acc)


# PROJECT 3: K-Means Clustering (Customer Segmentation)
print("\nPROJECT 3: K-Means Clustering")

# Load dataset
mall = pd.read_csv("Mall_Customers.csv")
X = mall[['Annual Income (k$)', 'Spending Score (1-100)']].values

# K-Means
k = 3
np.random.seed(42)
centroids = X[np.random.choice(len(X), k, replace=False)]

def closest_centroid(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

for _ in range(100):
    labels = closest_centroid(X, centroids)
    new_centroids = np.array([X[labels==j].mean(axis=0) for j in range(k)])
    if np.allclose(centroids, new_centroids):
        break
    centroids = new_centroids

print("Final Centroids:\n", centroids)
print("Cluster Counts:", np.bincount(labels))

print("\nPROJECT 4: Support Vector Machine (SVM) — human-style")

# Reuse iris loaded above
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values
labels, y = np.unique(y, return_inverse=True)
mask = y < 2
X, y = X[mask], y[mask]
y = np.where(y == 0, -1, 1)  # convert 0 -> -1, 1 -> 1

# normalize (simple, same as before)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# train-test split (reuse function)
X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1))
y_train = y_train.flatten()
y_test = y_test.flatten()

# initialize weights
w = np.zeros(X_train.shape[1])
b = 0.0

lr = 0.001
reg = 0.01   # regularization strength
epochs = 1000

# Simple SGD for linear SVM (hinge loss)
for ep in range(epochs):
    # shuffle data each epoch
    perm = np.random.permutation(len(X_train))
    X_sh = X_train[perm]
    y_sh = y_train[perm]

    for xi, yi in zip(X_sh, y_sh):
        # compute margin (note: using dot product)
        margin = yi * (np.dot(w, xi) + b)
        if margin >= 1:
            # only regularization gradient
            dw = 2 * reg * w
            db = 0.0
        else:
            # hinge loss gradient + regularization
            dw = 2 * reg * w - yi * xi
            db = -yi

        # update weights and bias
        w -= lr * dw
        b -= lr * db

    if ep % 200 == 0:
        # compute training accuracy
        preds = np.sign(X_train @ w + b)
        train_acc = np.mean(preds == y_train)
        print(f"Epoch {ep}: train_acc={train_acc:.3f}")


def svm_predict(X):
    scores = X @ w + b
    return np.where(scores >= 0, 1, -1)

y_pred = svm_predict(X_test)
acc = np.mean(y_pred == y_test)
print("SVM Test Accuracy:", acc)
