# Mobile Money Fraud Detection with CatBoost

## Overview

This project focuses on detecting fraudulent mobile money transactions using machine learning. It was developed as part of the DataTour 2026 challenge.

The objective is to identify potentially fraudulent transactions, engineer useful predictive features, and evaluate a classification model on an imbalanced dataset.

## Objectives

- Explore and analyze mobile money transaction data.
- Engineer relevant predictive features.
- Train a machine learning model for fraud detection.
- Evaluate model performance using Precision-Recall Area Under the Curve (PR-AUC).
- Generate predictions for unseen transactions.

## Dataset

The project uses a large transaction dataset:

| Dataset | Number of rows | Number of columns |
|---|---:|---:|
| Training set | 1,290,081 | 11 |
| Test set | 430,100 | 10 |

The target variable is `fraud_flag`.

**Note:** Competition datasets are not included in this repository. Obtain them from an authorized source and respect the competition's data-sharing rules.

## Methodology

1. Exploratory data analysis.
2. Feature engineering.
3. Model training with CatBoost.
4. Model evaluation.
5. Prediction generation.

## Model and Results

- **Algorithm:** CatBoost Classifier
- **Best validation PR-AUC:** 0.4382
- **Best iteration:** Approximately 253
- **Target variable:** `fraud_flag`

PR-AUC is an appropriate evaluation metric for imbalanced classification problems such as fraud detection.

These results reflect the current experiment and depend on the validation procedure used.

## Project Structure

```text
mobile-money-fraud-detection/
├── datatour_column_descriptions.csv
├── eda.py
├── features.py
├── train.py
├── evaluate.py
├── predict.py
├── starter_notebook.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

The `requirements.txt` file should list the Python dependencies needed to run the project.

## Technologies

- Python
- Pandas
- NumPy
- CatBoost
- Scikit-learn
- Jupyter Notebook

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/freud-ahodonou/mobile-money-fraud-detection.git
cd mobile-money-fraud-detection
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

After creating `requirements.txt`, install the required libraries:

```powershell
pip install -r requirements.txt
```

### 4. Prepare the dataset

Obtain the competition dataset from an authorized source and place the required files in the locations expected by the scripts.

### 5. Run the project

Review the scripts and their dependencies before running them:

```powershell
python eda.py
python features.py
python train.py
python evaluate.py
python predict.py
```

Run the scripts in the order required by their actual implementation.

## Future Improvements

- Compare CatBoost with other machine learning algorithms.
- Improve feature engineering.
- Optimize the classification threshold.
- Analyze precision, recall, and the confusion matrix.
- Improve model interpretability.
- Evaluate robustness on unseen transactions.

## Author

**Sègan Freud AHODONOU**

Applied Mathematics and Computer Science student interested in Data Science, Machine Learning, and Artificial Intelligence.

- GitHub: https://github.com/freud-ahodonou

## Disclaimer

This project is intended for educational and portfolio purposes. Its predictions should not be considered a complete production-grade fraud detection system.
