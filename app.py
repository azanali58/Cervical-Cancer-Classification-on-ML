# app.py

import sys
import numpy as np
import pandas as pd
import joblib
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QFileDialog, QWidget, QTextEdit
from ImageProcessing import extract_features # Ensure this function is properly defined

class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load the trained model, scaler, PCA, and label encoder
        self.model = joblib.load(r"E:\Thesis\joblib\best_svm_model.joblib")
        self.scaler = joblib.load(r"E:\Thesis\joblib\scaler.joblib")
        self.pca = joblib.load(r"E:\Thesis\joblib\pca.joblib")
        self.label_encoder = joblib.load(r"E:\Thesis\joblib\label_encoder.joblib")
        
        
        all_features_df = pd.read_excel(r"E:\Thesis\excel data\testing_features.xlsx")
        self.all_feature_names = all_features_df.columns[1:-1].tolist() 
        
        # Load the selected features DataFrame for reference
        selected_features_df = pd.read_excel(r"E:\Thesis\excel data\train_features_selected.xlsx")
        self.selected_feature_names = selected_features_df.columns[1:-1].tolist()  # Adjust based on your DataFrame structure
    
        # Layout and Widgets
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Class Prediction App')

        self.layout = QVBoxLayout()
        
        self.label = QLabel("Upload Image(s) for Prediction")
        self.layout.addWidget(self.label)

        self.upload_button = QPushButton("Upload Images")
        self.upload_button.clicked.connect(self.upload_images)
        self.layout.addWidget(self.upload_button)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.layout.addWidget(self.result_display)

        self.setLayout(self.layout)

    def upload_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Image Files (*.jpg *.jpeg *.bmp *.png)")

        if files:
            results = []
            for file in files:
                # Read the image using OpenCV
                image = cv2.imread(file)

                # Make predictions
                predictions = self.predict_image(image)

                # Display results
                result_text = f"Predictions for {file}:\n"
                for class_name, prob in predictions.items():
                    result_text += f"{class_name}: {prob * 100:.2f}%\n"
                results.append(result_text)

            self.result_display.setPlainText("\n".join(results))

    def predict_image(self, image):
        # Preprocess the image to extract features
        processed_image = extract_features(image)

        # Convert to DataFrame for consistency with the model input
        all_features_df = pd.DataFrame(processed_image.reshape(1, -1), columns=self.all_feature_names)  # Use the full list of feature names
        filtered_features_df = all_features_df[self.selected_feature_names] 


        # Scale and apply PCA
        processed_image_scaled = self.scaler.transform(filtered_features_df[self.selected_feature_names])
        processed_image_pca = self.pca.transform(processed_image_scaled)

        # Predict using the loaded model
        probabilities = self.model.predict_proba(processed_image_pca)

            
        # Map class names to probabilities
        class_probabilities = {class_name: probability for class_name, probability in zip(self.label_encoder.classes_, probabilities[0])}
    
        return class_probabilities

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ImageClassifierApp()
    window.resize(400, 300)
    window.show()
    sys.exit(app.exec_())
