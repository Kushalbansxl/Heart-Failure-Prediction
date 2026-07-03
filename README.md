# CardioCare Diagnostics: Heart Failure Prediction System

## Overview
This project implements a dual-model approach for heart failure prediction using both clinical tabular data and MRI scans. The system combines a Random Forest model for analyzing clinical parameters and a Convolutional Neural Network (CNN) for processing medical imaging data, providing medical professionals with comprehensive diagnostic support.


## Features
- **Dual Prediction Methods**:
  - Clinical data analysis using a Random Forest Classifier.
  - MRI image analysis using a Convolutional Neural Network (CNN).
- **Interactive Web Interface**:
  - User-friendly Streamlit application featuring a modern, dark-themed UI.
  - Real-time diagnostic predictions with probability scores and risk assessments.

## Project Structure
```text
heart_failure_pred/
├── artifacts/
│   ├── data/
│   │   ├── img_data/
│   │   │   ├── normal/
│   │   │   └── failure/
│   │   └── tabular_data/
│   │       └── heart_failure.csv
│   └── models/
│       ├── cnn/
│       │   ├── cnn_model.keras
│       │   └── cnn_score.json
│       └── rf/
│           ├── rf_model.pkl
│           └── rf_score.json
├── app.py
├── constants.py
├── main.py
├── train_cnn.py
├── train_rf.py
├── utils.py
├── requirements.txt
└── README.md
