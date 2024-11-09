# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 21:09:14 2024

@author: azan
"""

# Import libraries for model training
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import numpy as np



# Load the feature-selected training and testing data
train_df = pd.read_excel(r"E:\Thesis\excel data\train_features_selected.xlsx")
test_df = pd.read_excel(r"E:\Thesis\excel data\test_features_selected.xlsx")

# Separate features and labels
X_train = train_df.drop(columns=['Image_ID', 'Class'])
y_train = train_df['Class']
X_test = test_df.drop(columns=['Image_ID', 'Class'])
y_test = test_df['Class']

# Encode the target labels (if not already encoded)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


n_components = 15  # Adjust based on the explained variance required
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# Initialize the RF model with Grid Search for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],   # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],    # Minimum samples required at a leaf node
    'bootstrap': [True, False]        # Method of selecting samples for training each tree
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_pca, y_train_encoded)

# Best model from grid search
best_rf = grid_search.best_estimator_
print("Best parameters from GridSearchCV:", grid_search.best_params_)

# Fit PCA on training data and transform both train and test sets


# Initialize the model (RandomForest as an example)
#best_rf = RandomForestClassifier(random_state=42)

# Train the model
best_rf.fit(X_train_pca, y_train_encoded)

# Make predictions on the test set
y_pred = best_rf.predict(X_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
precision = precision_score(y_test_encoded, y_pred, average='weighted')
recall = recall_score(y_test_encoded, y_pred, average='weighted')  # Sensitivity
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Convert class names to strings for the classification report
class_names = list(map(str, label_encoder.classes_))
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test_encoded, y_pred)
# Classification report and confusion matrix
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
disp.plot()
plt.show()

# Optionally, save the trained model
import joblib
joblib.dump(model, "trained_RF_model.joblib")

print("Model training, evaluation, and saving completed.")
