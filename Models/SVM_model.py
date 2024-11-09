# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:58:01 2024

@author: azan
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import joblib

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

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA for dimensionality reduction 
pca = PCA(n_components=15)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Initialize the SVM model with Grid Search for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],      # Regularization parameter
    'kernel': ['linear', 'rbf'],  # SVM kernel type
    'gamma': ['scale', 'auto']    # Kernel coefficient
}

grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5)
grid_search.fit(X_train_pca, y_train_encoded)

# Best model from grid search
best_svm = grid_search.best_estimator_
print("Best parameters from GridSearchCV:", grid_search.best_params_)

# Train the model with the best parameters on the full training data
best_svm.fit(X_train_pca, y_train_encoded)

# Make predictions on the test set
y_pred = best_svm.predict(X_test_pca)
y_pred_proba = best_svm.predict_proba(X_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy of SVM: {accuracy:.2f}")
# Precision, Recall (Sensitivity), and F1-score
precision = precision_score(y_test_encoded, y_pred, average='weighted')
recall = recall_score(y_test_encoded, y_pred, average='weighted')  # Sensitivity
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Classification report and confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred))



# Save the model, scaler, PCA, and label encoder
joblib.dump(best_svm, r"E:\Thesis\best_svm_model.joblib")
joblib.dump(scaler, r"E:\Thesis\scaler.joblib")
joblib.dump(pca, r"E:\Thesis\pca.joblib")
joblib.dump(label_encoder, r"E:\Thesis\label_encoder.joblib")



