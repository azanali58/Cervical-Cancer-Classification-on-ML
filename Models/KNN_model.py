# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:47:47 2024

@author: azan
"""

# Import libraries for model training
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

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

# Step 1: Apply PCA
n_components = 15  # Adjust based on explained variance required
pca = PCA(n_components=n_components)

# Fit PCA on training data and transform both train and test sets
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initialize the KNN model with Grid Search for hyperparameter tuning
param_grid = {
    'n_neighbors': [i for i in range(3,33,2)],  # Number of neighbors to consider
    'weights': ['uniform', 'distance'],  # Weight function for neighbors
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metrics
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_pca, y_train_encoded)

# Best model from grid search
best_knn = grid_search.best_estimator_
print("Best parameters from GridSearchCV:", grid_search.best_params_)

# Train the model with the best parameters on the full training data
best_knn.fit(X_train_pca, y_train_encoded)

score_train_knn = best_knn.score(X_train_pca, y_train_encoded)  #  goodness of fit
score_test_knn = best_knn.score(X_test_pca,y_test_encoded)  #  goodness of fit
# Make predictions on the test set
y_pred = best_knn.predict(X_test_pca)
y_pred_proba = best_knn.predict_proba(X_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy of KNN: {accuracy:.2f}")
# Precision, Recall (Sensitivity), and F1-score
precision = precision_score(y_test_encoded, y_pred, average='weighted')
recall = recall_score(y_test_encoded, y_pred, average='weighted')  # Sensitivity
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test_encoded, y_pred)
# Classification report and confusion matrix
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
disp.plot()
plt.show()