# Coral Bleaching Classification Using Machine Learning

This project explores multiple machine learning methods to classify coral images as **healthy** or **bleached**. Our goal is to develop accurate, efficient, and accessible models that can assist in large-scale coral reef monitoring, particularly in resource-constrained field conditions.

## Team
Allison Ebsen, Yu Fang Ma, Frances Uy  
ICS 485 – Machine Learning Methods  
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

---

## Dataset
- Source: [NOAA ESD Coral Bleaching Dataset](https://huggingface.co/datasets/akridge/NOAA-ESD-CORAL-Bleaching-Dataset)
- 6,488 healthy coral images labeled CORAL
- 3,432 bleached coral images labeled CORAL_BL
- All images sourced from the Hawaiian Archipelago

---

## Requirements
To run this project, you'll need Python 3.8+ and the following packages:

_Install them using:_
```bash
pip install -r requirements.txt
```

_If using Jupyter Notebooks:_
```bash
pip install notebook
```

---

## Running the Project
1. Clone the repository or download the files.
2. Install the dependencies listed in requirements.txt.
3. Open desired notebook file in Jupyter.
4. Run all cells to reproduce our feature extraction, model training, and evaluation.

---

## Model Performance (Test Set)

| Model                  | Accuracy | AUC   | Pros                                  | Cons                                |
|------------------------|----------|-------|---------------------------------------|-------------------------------------|
| CNN (Convolutional)    | ~0.85    | N/A   | High accuracy, captures spatial data  | Requires GPU, high storage/compute  |
| KNN + Color Extraction | ~0.82    | ~0.89 | Lightweight, interpretable, fast      | May miss texture/structural cues    |
| Gradient Boosting      | 0.8185   | ~0.89 | Strong generalization, interpretable  | Requires manual feature extraction  |


---

## Acknowledgments

- NOAA & Pacific Islands Fisheries Science Center for the dataset
- Professor Haopeng Zhang for the project inspiration and foundational ideas