# -*- coding: utf-8 -*-
"""
Created on Sat sept  2 11:58:01 2024

@author: azan
"""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier; # importing neural network classifier
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


MLP_classifier = MLPClassifier(hidden_layer_sizes=(10,9),activation='tanh',max_iter=1000)
MLP_classifier.fit(X_train_pca,y_train_encoded)

#MLP_classifier = grid_search.best_estimator_

#score_test_MLP = MLP_classifier.score(X_test,y_test);  #  goodness of fit

# Make predictions on the test set
y_pred = MLP_classifier.predict(X_test_pca);   # spam prediction
y_pred_proba = MLP_classifier.predict_proba(X_test_pca);  #  prediction probabilities


# Evaluate the model
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy of MLP: {accuracy:.2f}")
precision = precision_score(y_test_encoded, y_pred, average='weighted')
recall = recall_score(y_test_encoded, y_pred, average='weighted')  # Sensitivity
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# Confusion Matrix
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test_encoded, y_pred)
# Classification report and confusion matrix
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
disp.plot()
plt.show()

# Save the XGBoost model, scaler, PCA, and label encoder
joblib.dump(MLP_classifier, r"E:\Thesis\MLP_model.joblib")
joblib.dump(scaler, r"E:\Thesis\scaler.joblib")
joblib.dump(pca, r"E:\Thesis\pca.joblib")
joblib.dump(label_encoder, r"E:\Thesis\label_encoder.joblib")
