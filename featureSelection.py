# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE,mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the Excel files
train_df = pd.read_excel(r"E:\Thesis\excel data\training_features_1.xlsx")
test_df = pd.read_excel(r"E:\Thesis\excel data\testing_features.xlsx")

# Separate features and target variable
X_train = train_df.drop(columns=['Image_ID', 'Class'])
y_train = train_df['Class']
X_test = test_df.drop(columns=['Image_ID', 'Class'])
y_test = test_df['Class']

# Standardize features (helps some feature selection methods)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## 1. Filter-Based Feature Selection
# Select top features using ANOVA F-value
k_best = 18  # Choose the number of features to select
filter_selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
X_train_filter_selected = filter_selector.fit_transform(X_train_scaled, y_train)

# Get the selected feature names after filter selection
selected_filter_features = X_train.columns[filter_selector.get_support()]
print(f"Top {k_best} features selected by filter method: {selected_filter_features.tolist()}")

# Use selected RFE features to create new DataFrames for training and testing
train_df_selected = train_df[['Image_ID'] + selected_filter_features.tolist() + ['Class']]
test_df_selected = test_df[['Image_ID'] + selected_filter_features.tolist() + ['Class']]


"""
# Narrow down to just the selected features from Step 1
X_train_filtered = X_train[selected_filter_features]
X_test_filtered = X_test[selected_filter_features]

## Step 2: Wrapper-Based Selection (RFE) on Filtered Features
wrapper_k = 18  # Choose final number of features after RFE
clf = RandomForestClassifier(random_state=42)
rfe_selector = RFE(estimator=clf, n_features_to_select=wrapper_k)
X_train_rfe_selected = rfe_selector.fit_transform(X_train_filtered, y_train)

# Get the selected feature names after RFE selection
selected_rfe_features = selected_filter_features[rfe_selector.get_support()]
print(f"Top {wrapper_k} features selected by wrapper method: {selected_rfe_features.tolist()}")

# Use selected RFE features to create new DataFrames for training and testing
train_df_selected = train_df[['Image_ID'] + selected_rfe_features.tolist() + ['Class']]
test_df_selected = test_df[['Image_ID'] + selected_rfe_features.tolist() + ['Class']]
"""
# Export to Excel
train_df_selected.to_excel(r"E:\Thesis\excel data\train_features_selected.xlsx", index=False)
test_df_selected.to_excel(r"E:\Thesis\excel data\test_features_selected.xlsx", index=False)

print("Combined feature selection completed and saved to Excel.")