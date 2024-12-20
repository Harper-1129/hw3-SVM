import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Step 1: Generate random data and labels
np.random.seed(42)
X = np.random.randint(0, 1001, 300)  # Random data
Y = np.where((X > 500) & (X < 800), 1, 0)  # Binary labels

# Step 2: Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X.reshape(-1, 1), Y, test_size=0.2, random_state=42)

# Step 3: Train Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y1 = logreg.predict(X_test)

# Step 4: Train SVM model
svm = SVC(probability=True)
svm.fit(X_train, Y_train)
y2 = svm.predict(X_test)

# Step 5: Plot results
plt.figure(figsize=(15, 6))

# Logistic Regression
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='gray', label='True Labels')
plt.scatter(X_test, y1, color='blue', marker='x', label='Logistic Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Logistic Regression')
plt.legend()

# SVM
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color='gray', label='True Labels')
plt.scatter(X_test, y2, color='green', marker='s', label='SVM')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVM')
plt.legend()

plt.tight_layout()
plt.show()
