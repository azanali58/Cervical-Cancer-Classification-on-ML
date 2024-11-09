# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:23:28 2024

@author: azan
"""

# Cell 1: Import Libraries
import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops, label
from sklearn.preprocessing import LabelEncoder
from skimage.feature import graycomatrix, graycoprops
import statistics


# Define paths to training and testing directories
train_path = r"E:\Thesis\SiPakMed Dataset\Training"
test_path = r"E:\Thesis\SiPakMed Dataset\Test"

# Preprocess the image
def preprocess_image_nucleus(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_image = cv2.medianBlur(grayscale_image, 13)  # Adjust the kernel size 
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(median_image)    
    _, binary_image = cv2.threshold(clahe_image,95, 130, cv2.THRESH_BINARY_INV)
    return binary_image

def preprocess_image_cytoplasm(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_image = cv2.medianBlur(grayscale_image, 7)  # Adjust the kernel size (5x5 in this case)
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)    
    _, binary_image = cv2.threshold(median_image,120, 180, cv2.THRESH_BINARY_INV)
    return binary_image

def preprocess_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_image = cv2.medianBlur(grayscale_image, 13)  # Adjust the kernel size (5x5 in this case)
      # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(median_image)
      
    binary_image = cv2.adaptiveThreshold(clahe_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY_INV, 11, 2)

    return binary_image
def preprocess_image1(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_image = cv2.medianBlur(grayscale_image, 7)  # Adjust the kernel size (5x5 in this case)
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)        
    binary_image = cv2.adaptiveThreshold(median_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)

    return binary_image

# Feature extraction function
def extract_features(image):
    try:
        # Initialize default values for features
        nucleus_area = cytoplasm_area = nucleus_brightness = cytoplasm_brightness = 0
        nc_ratio = nucleus_roundness = cytoplasm_roundness = nucleus_perimeter = 0
        nucleus_position_x = nucleus_position_y = 0
        cytoplasm_maxima = cytoplasm_minima = nucleus_maxima = nucleus_minima = 0
        cytoplasm_perimeter = mean_intensity = std_dev_intensity = 0
        LBP_mean = LBP_var = 0
        
        preprocessed_image = preprocess_image_nucleus(image)
        labeled_image = label(preprocessed_image)
        props = regionprops(labeled_image, intensity_image=image)
        if not props:
            preprocessed_image = preprocess_image(image)
            labeled_image = label(preprocessed_image)
            props = regionprops(labeled_image, intensity_image=image)

        contours, hierarchy = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cnt = contours[0]  # Assuming the largest contour is the object of interest
        nucleus_area = cv2.contourArea(cnt)
        nucleus_perimeter = cv2.arcLength(cnt, True)
        nucleus_roundness = (4 * np.pi * nucleus_area) / (nucleus_perimeter ** 2) if nucleus_perimeter > 0 else 0
        for region in props:
            nucleus_brightness =statistics.mean(region.mean_intensity)
            nucleus_position_x, nucleus_position_y = region.centroid
            nucleus_maxima = max(region.max_intensity)
            nucleus_minima = min(region.min_intensity)
        
        LBP = local_binary_pattern(preprocessed_image, 8, 16, 'default')
        LBP_mean = np.mean(LBP)
        LBP_var = np.var(preprocessed_image)
        
        # Extract GLCM features
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(preprocessed_image, distances=distances, angles=angles, symmetric=True, normed=True)

        glcm_contrast = graycoprops(glcm, 'contrast').mean()
        glcm_correlation = graycoprops(glcm, 'correlation').mean()
        glcm_energy = graycoprops(glcm, 'energy').mean()
        glcm_homogeneity = graycoprops(glcm, 'homogeneity').mean()

        
        #cytoplasm textural features
        preprocessed_image1 = preprocess_image_cytoplasm(image)
        labeled_image1 = label(preprocessed_image1)
        props1 = regionprops(labeled_image1, intensity_image=image)
        if not props1:
            preprocessed_image1 = preprocess_image1(image)
            labeled_image1 = label(preprocessed_image1)
            props1 = regionprops(labeled_image1, intensity_image=image)
            
            
        contours, hierarchy = cv2.findContours(preprocessed_image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cnt = contours[0]  # Assuming the largest contour is the object of interest
        cytoplasm_area = cv2.contourArea(cnt)
        cytoplasm_perimeter = cv2.arcLength(cnt, True)
        cytoplasm_roundness = (4 * np.pi * cytoplasm_area) / (cytoplasm_perimeter ** 2) if cytoplasm_perimeter > 0 else 0
        for region in props1:
            cytoplasm_brightness = statistics.mean(region.mean_intensity)
            cytoplasm_maxima = max(region.max_intensity)
            cytoplasm_minima = min(region.min_intensity)
    


        nc_ratio = nucleus_area / cytoplasm_area if cytoplasm_area != 0 else 0
        # Image intensity statistics
        mean_intensity = np.mean(image)
        std_dev_intensity = np.std(image)

     

    

        # Combine features into one list
        features =  [
            nucleus_area, cytoplasm_area, nucleus_brightness, cytoplasm_brightness, nc_ratio,
            nucleus_roundness, cytoplasm_roundness, nucleus_perimeter, nucleus_position_x,
            nucleus_position_y, cytoplasm_maxima, cytoplasm_minima, nucleus_maxima, nucleus_minima,
            cytoplasm_perimeter, mean_intensity, std_dev_intensity,LBP_mean,LBP_var,
            glcm_contrast, glcm_correlation, glcm_energy, glcm_homogeneity
        ]
        
    except Exception as e:
        print(f"Error extracting features for an image: {e}")
        features = [0] * (23)  # Fill with zeros if extraction fails
   
    #print(f"Extracted feature vector length: {len(features)}")
    #return np.array(features)
    return np.array(features)

# Load images and labels
def load_images_and_labels(directory):
    images, labels, image_filenames = [], [], []
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith(".bmp"):
                    img_path = os.path.join(class_dir, filename)
                    image = cv2.imread(img_path)
                    if image is not None:
                        images.append(image)
                        labels.append(class_name)
                        image_filenames.append(filename)
    return images, labels, image_filenames

def load_images_and_labels_single(directory):
    images, labels, image_filenames = [], [], []
    head_tail= os.path.split(directory)
    filename = head_tail[1]
    path = head_tail[0]
    head_tail1= os.path.split(path)
    classname =  head_tail1[1]
    image = cv2.imread(directory)
    images.append(image)
    labels.append(0)
    image_filenames.append(filename)
    return images, labels, image_filenames

# Load training and testing data
train_images, train_labels, train_filenames = load_images_and_labels(train_path)
test_images, test_labels, test_filenames = load_images_and_labels(test_path)


"""
#for testing perpose 
"img_path= r"E:\SingleCellPAP\Training\im_Parabasal\097_03.bmp"
train_images, train_labels, train_filenames = load_images_and_labels_single(img_path)
img_path1= r"E:\SingleCellPAP\Test\im_Metaplastic\001_03.bmp"
test_images, test_labels, test_filenames = load_images_and_labels_single(img_path1)
"""
# Extract features for training images
train_features = []
for image in train_images:
    features = extract_features(image)
    #print(f"Extracted features for an image: {features}")  # Debugging line
    if features.size > 0:  # Only append if features are valid
        train_features.append(features)

# Extract features for testing images
test_features = []
for image in test_images:
    features = extract_features(image)
    #print(f"Extracted features for a test image: {features}")  # Debugging line
    if features.size > 0:  # Only append if features are valid
        test_features.append(features)

# Check the number of features extracted
print(f"Number of training feature sets: {len(train_features)}")
print(f"Number of testing feature sets: {len(test_features)}")

# Convert to NumPy arrays
X_train = np.array(train_features) if train_features else np.empty((0, 0))
y_train = np.array(train_labels) if train_labels else np.empty((0,))
X_test = np.array(test_features) if test_features else np.empty((0, 0))
y_test = np.array(test_labels) if test_labels else np.empty((0,))


# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train) if y_train.size > 0 else np.empty((0,))
y_test_encoded = label_encoder.transform(y_test) if y_test.size > 0 else np.empty((0,))


# Define feature names
feature_names = [
    'Nucleus_Area', 'Cytoplasm_Area', 'Nucleus_Brightness', 'Cytoplasm_Brightness', 'N/C_Ratio',
    'Nucleus_Roundness', 'Cytoplasm_Roundness', 'Nucleus_Perimeter', 'Nucleus_X_Position',
    'Nucleus_Y_Position', 'Cytoplasm_Maxima', 'Cytoplasm_Minima', 'Nucleus_Maxima', 'Nucleus_Minima',
    'Cytoplasm_Perimeter', 'Mean_Intensity', 'Std_Dev_Intensity', 'LBP_mean','LBP_var',
    'glcm_contrast', 'glcm_correlation', 'glcm_energy', 'glcm_homogeneity'

]

# Create DataFrames for features
train_df = pd.DataFrame(X_train, columns=feature_names)
train_df['Image_ID'] = train_filenames
train_df['Class'] = y_train
train_df = train_df[['Image_ID'] + feature_names + ['Class']]

test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['Image_ID'] = test_filenames
test_df['Class'] = y_test
test_df = test_df[['Image_ID'] + feature_names + ['Class']]

# Save to Excel
train_df.to_excel(r"E:\Thesis\excel data\training_features_1.xlsx", index=False)
test_df.to_excel(r"E:\Thesis\excel data\testing_features.xlsx", index=False)

print("Feature extraction and saving completed successfully.")