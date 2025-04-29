# ICS435-CORAL-BLEACHING-PROJECT
# Coral Classification with Machine Learning

This project is focused on classifying coral species based on color data. The model uses different machine learning algorithms including **SVM**, **KNN**, **Logistic Regression**, **XGBoost**, and **Random Forest** for binary classification (CORAL vs CORAL_BL). The data consists of multiple columns representing color values in hexadecimal format, which are converted to RGB for model input.

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
- [License](#license)

## Installation

### Requirements

To run this project, you'll need Python 3.x and the following dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn xgboost
```

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

## Results

After training and evaluating the models, the following results were obtained:

- **Accuracy**: The models showed competitive accuracy on the validation and test sets.
- **AUC**: AUC scores are plotted to demonstrate how well the models distinguish between the two classes.
- **Confusion Matrices**: Confusion matrices for each model were plotted to visualize the number of true positives, false positives, true negatives, and false negatives.


