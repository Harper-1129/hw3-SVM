import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Step 1: Generate 300 random variables X(i) in the range [0, 1000]
np.random.seed(42)
X = np.random.randint(0, 1001, 300).reshape(-1, 1)

# Step 2: Create binary classification labels Y(i)
Y = np.where((X > 500) & (X < 800), 1, 0).ravel()

# Step 3: Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y1 = logreg.predict(X_test)

# SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, Y_train)
y2 = svm.predict(X_test)

# Create a range of X values for decision boundary and probability visualization
X_range = np.linspace(0, 1000, 1000).reshape(-1, 1)

# Logistic Regression probabilities and predictions
logreg_probs = logreg.predict_proba(X_range)[:, 1]
logreg_preds = logreg.predict(X_range)

# SVM probabilities and decision function
svm_probs = svm.predict_proba(X_range)[:, 1]
svm_decision = svm.decision_function(X_range)

# Plot 1: Logistic Regression with probability curve and decision boundary
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='gray', label='True Labels')  # True data points
plt.scatter(X_test, y1, color='blue', marker='x', label='Logistic Regression Predictions')  # Predictions
plt.plot(X_range, logreg_probs, color='red', linestyle='--', label='Probability Curve')  # Probability curve
plt.axhline(0.5, color='black', linestyle=':', label='Decision Threshold (0.5)')  # Decision threshold
plt.axvline(500, color='green', linestyle='--', label='True Lower Boundary (500)')
plt.axvline(800, color='purple', linestyle='--', label='True Upper Boundary (800)')
plt.xlabel('X')
plt.ylabel('Y/Probability')
plt.title('Logistic Regression: Predictions and Decision Boundary')
plt.legend()

# Plot 2: SVM with probability curve and decision boundary
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color='gray', label='True Labels')  # True data points
plt.scatter(X_test, y2, color='green', marker='s', label='SVM Predictions')  # Predictions
plt.plot(X_range, svm_probs, color='orange', linestyle='--', label='Probability Curve')  # Probability curve
plt.axhline(0.5, color='black', linestyle=':', label='Decision Threshold (0.5)')  # Decision threshold
plt.axvline(500, color='green', linestyle='--', label='True Lower Boundary (500)')
plt.axvline(800, color='purple', linestyle='--', label='True Upper Boundary (800)')
plt.xlabel('X')
plt.ylabel('Y/Probability')
plt.title('SVM: Predictions and Decision Boundary')
plt.legend()

plt.tight_layout()
plt.show()
