# -*- coding: utf-8 -*-
"""

@author: azan
"""


import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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

# Initialize and train the Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train_pca, y_train_encoded)

# Make predictions on the test set
y_pred = gnb.predict(X_test_pca)
y_pred_proba = gnb.predict_proba(X_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred, average='weighted')
recall = recall_score(y_test_encoded, y_pred, average='weighted')
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print(f"Accuracy of Naive Bayes: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test_encoded, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
disp.plot()
plt.show()

# Save the model, scaler, PCA, and label encoder
joblib.dump(gnb, r"E:\Thesis\joblib\naive_bayes_model.joblib")
joblib.dump(scaler, r"E:\Thesis\joblib\scaler.joblib")
joblib.dump(pca, r"E:\Thesis\joblib\pca.joblib")
joblib.dump(label_encoder, r"E:\Thesis\joblib\label_encoder.joblib")

print("Naive Bayes model training and evaluation complete.")


