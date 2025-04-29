# Coral Bleaching Classification with Machine Learning

This project is focused on classifying coral images as **healthy** or **bleached** based on color data. The model uses different machine learning algorithms including **SVM**, **KNN**, **Logistic Regression**, **XGBoost**, and **Random Forest** for binary classification (CORAL vs CORAL_BL). The data consists of multiple columns representing color values in hexadecimal format, which are converted to RGB for model input. Our goal is to develop accurate, efficient, and accessible models that can assist in large-scale coral reef monitoring, particularly in resource-constrained field conditions.

## Team
Allison Ebsen, Yu Fang Ma, Frances Uy  
ICS 435 – Machine Learning Methods  
Spring 2025 Final Project

---

## Project Overview

Coral reefs play a vital ecological and cultural role, especially in Hawaiʻi. However, with increasing global temperatures, **coral bleaching** is a growing threat. Traditional monitoring methods are manual and slow. In this project, we compare three different machine learning approaches for automating coral health classification:

- **CNN (Convolutional Neural Network)** – deep learning baseline
- **KNN with Color Feature Extraction** – novel lightweight approach
- **Gradient Boosting with Feature Engineering** – structured ML solution

Our models were trained on a dataset of over 9,900 labeled coral images provided by NOAA. All code and feature extraction pipelines are included in this repository.

---

## Structure

```bash
📁 ICS435-CORAL-BLEACHING-PROJECT/
├── gradient-boosting.ipynb        # Notebook for feature extraction + Gradient Boosting
├── cnn_model.ipynb                # (Optional) Notebook for CNN baseline
├── color_knn_model.ipynb          # (Optional) Notebook for color-based KNN model
├── coral_train_features.csv       # Extracted features for Gradient Boosting
├── sample_images.png              # Sample coral image visualizations
├── confusion_matrix_test.png      # Model evaluation visual
├── requirements.txt               # Required Python packages
└── README.md                      # This file
```

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models](#models)
  - [Random Forest](#random-forest)
  - [SVM](#svm)
  - [KNN](#knn)
  - [Logistic Regression](#logistic-regression)
  - [XGBoost](#xgboost)
- [Results](#results)
- [Summary](#summary)

---

## Dataset
- Source: [NOAA ESD Coral Bleaching Dataset](https://huggingface.co/datasets/akridge/NOAA-ESD-CORAL-Bleaching-Dataset)
- 6,488 healthy coral images labeled CORAL
- 3,432 bleached coral images labeled CORAL_BL
- All images sourced from the Hawaiian Archipelago

---

## Installation

### Requirements

To run this project, you'll need Python 3.x and the following dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn xgboost
```

Or install them using:
```bash
pip install -r requirements.txt
```


If using Jupyter Notebooks:
```bash
pip install notebook
```

---

## Running the Project
1. Clone the repository or download the files.
2. Install the dependencies listed in requirements.txt.
3. Open desired notebook file in Jupyter.
4. Run all cells to reproduce our feature extraction, model training, and evaluation.

## Models

### Random Forest

- **Grid Search Parameters**:
  - Number of trees (`n_estimators`)
  - Maximum depth of trees (`max_depth`)
  - Minimum samples split (`min_samples_split`)
  - **Best Parameters**: Selected through Grid Search to find the optimal random forest configuration.

### SVM (Support Vector Machine)

- **Grid Search Parameters**:
  - Regularization (`C`)
  - Kernel type (`rbf`, `poly`)
  - Polynomial degree for `poly` kernel
  - **Best Parameters**: Found through Grid Search, yielding the best model for classification.

### KNN (K-Nearest Neighbors)

- **Grid Search Parameters**:
  - Number of neighbors (`n_neighbors`)
  - Weighting function (`uniform`, `distance`)
  - Distance metric (`Manhattan` or `Euclidean`)
  - **Best Parameters**: Found through Grid Search for optimal KNN configuration.

### Logistic Regression

- **Grid Search Parameters**:
  - Regularization strength (`C`)
  - Penalty type (`l2`)
  - Solver type (`liblinear`)
  - **Best Parameters**: Identified through Grid Search for logistic regression.

### XGBoost

- **Grid Search Parameters**:
  - Number of trees (`n_estimators`)
  - Maximum depth of trees (`max_depth`)
  - Learning rate (`learning_rate`)
  - Subsampling rate (`subsample`)
  - **Best Parameters**: Selected through Grid Search for optimal boosting performance.

### Gradient Boosting
  - n_estimators
  - learning_rate
  - max_depth
  - min_samples_split
  - min_samples_leaf
  - subsample
  - random_state

---

## Model Performance (Test Set)

| Model                  | Accuracy | AUC   | Pros                                  | Cons                                |
|------------------------|----------|-------|---------------------------------------|-------------------------------------|
| CNN (Convolutional)    | ~0.85    | N/A   | High accuracy, captures spatial data  | Requires GPU, high storage/compute  |
| KNN + Color Extraction | ~0.82    | ~0.89 | Lightweight, interpretable, fast      | May miss texture/structural cues    |
| Gradient Boosting      | 0.8185   | ~0.89 | Strong generalization, interpretable  | Requires manual feature extraction  |

---

## Results

After training and evaluating the models, the following results were obtained:
The script evaluates the models using the following metrics:

1. **Accuracy**
2. **Precision**
3. **Recall**
4. **F1-score**
- **AUC**: AUC scores are plotted to demonstrate how well the models distinguish between the two classes.
- **Confusion Matrices**: Confusion matrices for each model were plotted to visualize the number of true positives, false positives, true negatives, and false negatives.

---

## Summary

This repository contains two key files:

1. **`get_coral_color_data.py`**: A Python script that processes images of coral to extract their color data. The script converts hexadecimal color values from the images into RGB values and generates a dataset containing color information, which will later be used for training machine learning models. It is the first step in preparing data for analysis.

2. **`coral_bleaching_models.ipynb`**: A Jupyter Notebook that implements machine learning models to predict coral bleaching (or coral and coral-like conditions). It uses the dataset generated by the `get_coral_color_data.py` script and applies multiple machine learning models like SVM, KNN, Logistic Regression, Random Forest, and XGBoost to classify the coral conditions based on the color data.

---

## Instructions

### Step 1: Run `get_coral_color_data.py`

This script extracts color information from coral images and prepares the dataset for training.

1. **Download the Images**: Place the coral images that you want to analyze in a folder (ensure that the images are in a format like PNG, JPEG, etc.).

2. **Run the Script**:
   - Open a terminal or command prompt and navigate to the directory containing `get_coral_color_data.py`.
   - Execute the script by running the following command:
     ```bash
     python get_coral_color_data.py --image_folder <path_to_image_folder> --output_file <output_csv_file>
     ```
     - Replace `<path_to_image_folder>` with the path to the folder containing the coral images.
     - Replace `<output_csv_file>` with the name of the output CSV file (e.g., `coral_colors.csv`).

   - The script will extract the color information from the images, convert the hex values to RGB, and output a CSV file with the dataset.

3. **Review the Output**:
   - The script will generate a CSV file containing the color data. This file can be used in the next step for training the machine learning models.


### Step 2: Run `coral_bleaching_models.ipynb`

This Jupyter Notebook uses the dataset generated in Step 1 to train and evaluate machine learning models for predicting coral bleaching based on color data.

1. **Prepare the Dataset**:
   - Ensure that the CSV file generated by `get_coral_color_data.py` is available. It will be used as input for the notebook.

2. **Open the Notebook**:
   - Open the `coral_bleaching_models.ipynb` file in a Jupyter Notebook or Google Colab environment.

3. **Load the Data**:
   - The notebook will load the dataset (generated in Step 1) and preprocess it, including converting color columns to RGB format.

4. **Train the Models**:
   - The notebook applies multiple machine learning models (SVM, KNN, Logistic Regression, Random Forest, and XGBoost) to predict coral bleaching based on the color data.
   - It performs hyperparameter tuning via grid search to identify the best performing model and evaluates the models using various metrics like accuracy, precision, recall, F1-score, and AUC.

5. **Results and Evaluation**:
   - After training, the notebook will output performance metrics for each model, including accuracy, precision, recall, F1-score, and confusion matrices.
   - ROC curves for each model will be plotted to visualize performance.
   
6. **Make Predictions**:
   - After the models are trained, you can use them to make predictions on new data

---

## Acknowledgments

- NOAA & Pacific Islands Fisheries Science Center for the dataset
- Professor Haopeng Zhang for the project inspiration and foundational ideas