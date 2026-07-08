# 🧠 Stroke Prediction Index (SPI)

An interpretable machine learning framework for predicting ischemic stroke lesion volume and severity using multimodal MRI features extracted from the ISLES 2022 dataset.

## Overview

This project uses MRI scans from the **ISLES 2022 (Ischemic Stroke Lesion Segmentation Challenge)** dataset to predict stroke lesion volume and estimate stroke severity through a novel **Stroke Prediction Index (SPI)**.

The framework combines information from three MRI modalities:

- Diffusion Weighted Imaging (DWI)
- Apparent Diffusion Coefficient (ADC)
- Fluid Attenuated Inversion Recovery (FLAIR)

The extracted imaging features are used to train machine learning models for regression and severity classification.

## Dataset

- **Dataset:** ISLES 2022
- **Input:** Multimodal MRI scans (DWI, ADC, FLAIR)
- **Target:** Stroke lesion volume

After feature extraction, the generated dataset (`stroke_dataset_final.csv`) is used for model training and evaluation.

## Pipeline

1. Load MRI-derived feature dataset.
2. Preprocess and clean the data.
3. Engineer additional imaging features.
4. Normalize and balance features.
5. Train regression models for lesion volume prediction.
6. Generate the Stroke Prediction Index (SPI).
7. Classify stroke severity into Mild, Moderate, and Severe.
8. Evaluate model performance and visualize the results.

## MRI Features Used

The framework utilizes imaging-derived features including:

- DWI Mean Intensity
- ADC Mean Intensity
- FLAIR Mean Intensity
- Heterogeneity
- Intensity Ratio
- FLAIR-ADC Ratio
- Normalized Spatial Spread

## Models

### Regression

- Random Forest Regressor
- Ridge Regression (used for SPI generation)

### Classification

- Gradient Boosting Classifier

## Evaluation Metrics

Regression:

- R² Score
- Root Mean Squared Error (RMSE)

Classification:

- Accuracy
- Precision
- Confusion Matrix
- Classification Report

## Visualizations

The project generates:

- Model comparison
- Actual vs Predicted values
- SPI vs Lesion Volume
- Feature Importance
- Residual Plot
- Volume Distribution
- Confusion Matrix

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- NiBabel

## Future Work

- Improve lesion volume prediction using larger datasets.
- Incorporate additional clinical features.
- Extend the framework with deep learning-based MRI feature extraction.
