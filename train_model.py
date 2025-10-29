# Script for training ML model
"""
Iris Flower Classification - Model Training Script
Trains a Random Forest model on the Iris dataset
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import numpy as np

# Load the Iris dataset
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print(f"Dataset shape: {X.shape}")
print(f"Features: {iris.feature_names}")
print(f"Classes: {target_names}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"{'='*50}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print("\nFeature Importance:")
for feature, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Save the trained model
print("\nSaving model to model.pkl...")
pickle.dump(model, open("model.pkl", "wb"))
print("✅ Model saved successfully!")

# Test with sample prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Sample Setosa
prediction = model.predict(sample)
print(f"\nSample Prediction Test:")
print(f"Input: {sample[0]}")
print(f"Predicted Class: {target_names[prediction[0]]}")
