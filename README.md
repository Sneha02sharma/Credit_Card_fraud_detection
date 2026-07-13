# Credit Card Fraud Detection 💳

A machine learning project that detects fraudulent credit card transactions using transaction data. Built as part of the CodSoft Data Science Internship.

## Overview

This project covers a complete, beginner-friendly ML workflow:
- Cleaning and exploring a large, real-world transaction dataset
- Checking for missing values and duplicates
- Feature selection using correlation analysis
- Exploratory data analysis (EDA) with visualizations
- Training a Random Forest model to classify transactions as fraud or not
- Evaluating the model's performance

## Dataset

The dataset (`creditcard.csv`) contains **284,807 transactions** with 31 columns, including anonymized features `V1`–`V28`, `Time`, `Amount`, and `Class` (0 = No Fraud, 1 = Fraud).

## Approach

**1. Data Cleaning**
- Checked for missing values — none were found.
- Found and removed **1,081 duplicate rows**, reducing the dataset from 284,807 to 283,726 rows.

**2. Correlation & Feature Selection**
- Since the `V1`–`V28` features were already anonymized/encoded (PCA-transformed), no additional encoding was needed.
- Checked correlation of `Time` and `Amount` with the target `Class`.
- `Time` had very low correlation with `Class` and was dropped.
- Visualized correlations across all features using a heatmap.

**3. Exploratory Data Analysis**
- Used a count plot to see the class distribution: **473 fraud cases vs. 283,253 non-fraud cases** — a highly imbalanced dataset.
- Used a boxplot to compare transaction `Amount` across classes — non-fraud transactions tended to have higher amounts than fraud cases.

**4. Model Building**
- Split the data into training (70%) and testing (30%) sets.
- Applied `StandardScaler` for feature scaling.
- Trained a **Random Forest Classifier** (with `class_weight='balanced'` to handle the class imbalance).

**5. Evaluation**
- Achieved an accuracy of **0.99**.
- Weighted average precision and recall also came out to **0.99**.
- Looking closer at the classification report: class 0 (non-fraud) was predicted near-perfectly, while class 1 (fraud) had a precision of 0.93 and recall of 0.74 — reflecting the challenge of detecting the rarer fraud cases in an imbalanced dataset.

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## How to Run

1. Clone this repository
   ```bash
   git clone https://github.com/Sneha02sharma/Credit_Card_fraud_detection.git
   cd Credit_Card_fraud_detection
   ```
2. Install dependencies
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Open and run the notebook
   ```bash
   jupyter notebook Credit_Card_fraud_detection.ipynb
   ```

> **Note:** This notebook currently loads the dataset from a local path (`C:\Users\Dell\Downloads\creditcard.csv`). To run it yourself, download `creditcard.csv` from rom Kaggle and update that path in the notebook to match where you've saved it on your machine.
## Acknowledgements

Dataset based on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) originally released by the Machine Learning Group at ULB (Université Libre de Bruxelles) on Kaggle.

---
*This project was completed as part of the CodSoft Data Science Internship.*
