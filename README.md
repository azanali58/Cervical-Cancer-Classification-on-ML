# Cervical Cell Classification Using Machine Learning

This project applies machine learning techniques to classify cervical cell images and assists in the early detection of cervical cancer. By using image processing and various machine learning classifiers, the system aims to accurately identify 
and differentiate between different types of cervical cells, potentially improving diagnostic efficiency, especially in low-resource settings.

---


## Overview
Cervical cancer is a leading cause of cancer-related deaths worldwide, particularly in under-resourced regions. Traditional diagnostic methods like the Pap smear test are prone to human error.

- Implement image preprocessing, enhancement, and feature extraction techniques.
- Evaluate multiple machine learning models (SVM, Random Forest, MLP, KNN, Naïve Bayes) for optimal classification accuracy.
- Develop a user-friendly GUI for easy interaction with the classification model.

## Dataset
This project uses the **SIPaKMeD** dataset, a publicly available dataset containing 4,049 single-cell images of cervical cells. The images are categorized into five classes based on the level of abnormality.

---

## Methods
1. **Preprocessing**: Image filtering and segmentation to enhance image quality.
2. **Feature Extraction**: Extracts morphological, color, and texture-based features (e.g., Local Binary Patterns, Gray Level Co-occurrence Matrix).
3. **Feature Selection**: Selects the most relevant features to optimize model performance using filter and wrapper methods.
4. **Model Training**: Trains multiple classifiers, including SVM, Random Forest, and MLP, to identify the best-performing model.
5. **Evaluation**: Evaluates models on accuracy, precision, recall, and F1 score.
6. **Deployment**: Provides an interactive GUI for end-users to upload images and view predictions.

---

## Requirements
This project requires the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `opencv-python`
- `scikit-image`
- `PyQt5`
- `joblib`

A complete list of dependencies can be found in `requirements.txt`.

---

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/cervical-cell-classification.git
   cd cervical-cell-classification
   install the packages from the Requirement file
