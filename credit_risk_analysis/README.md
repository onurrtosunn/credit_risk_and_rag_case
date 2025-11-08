# Credit Risk Analysis

This project provides a comprehensive machine learning solution for credit risk prediction. The project has a modular structure that includes data cleaning, feature engineering, model training, and evaluation steps.

## Project Structure

```
credit_risk_analysis/
├── config.py                 # Configuration parameters
├── data_cleaning.py          # Data cleaning module
├── feature_engineering.py    # Feature engineering module
├── eda.py                    # Exploratory data analysis module
├── train.py                  # Model training module
├── testing.py                # Model testing and evaluation module
├── main.ipynb                # Main working notebook
├── feature_engineering_test.ipynb  # Feature engineering test notebook
├── data/                     # Data files
│   └── credit_risk_case.xlsx
├── models/                   # Trained models
│   ├── pd_logistic_pipeline.pkl
│   ├── pd_gbm_pipeline.pkl
│   ├── pd_knn_pipeline.pkl
│   ├── pd_rf_pipeline.pkl
│   └── pd_xgb_pipeline.pkl
├── plots/                    # Visualizations
└── reports/                  # Reports and results
```

## Installation

### Requirements

To install the required Python packages for running the project:

```bash
pip install -r requirements.txt
```

### Data Preparation

Place the data file at `data/credit_risk_case.xlsx`.

## Usage

### 1. Data Loading and Cleaning

By loading raw data and applying cleaning operations:

- Removing ID columns (id, member_id)
- Converting employment length (emp_length) to numeric value
- Applying business logic constraints
- Filling missing values:
  - Filling categorical features with 'Unknown'
  - Filling numerical features with median
  - Filling delinquency features with 0

### 2. Exploratory Data Analysis (EDA)

By performing comprehensive analysis on the dataset:

- Calculating dataset overview statistics
- Analyzing numerical features by target variable
- Analyzing categorical features by target variable
- Calculating correlation matrix
- Visualizing feature distributions:
  - Creating bar charts
  - Drawing correlation heatmap
  - Creating violin plots

### 3. Feature Engineering

By creating new features and transforming existing features; DTI, Credit Utilization, Payment-to-Income Ratio etc.


### 3.1 Feature Engineering Test Module

The `feature_engineering_test.ipynb` notebook provides comprehensive testing and analysis of all engineered features defined in the configuration. By performing the following operations:

This module helps in understanding which engineered features are most predictive and provides comprehensive analysis before model training.

### 4. Model Training

By training five different machine learning models:

- **Logistic Regression**: Training linear model, performing hyperparameter optimization with GridSearchCV
- **HistGradientBoosting**: Training gradient boosting model, performing hyperparameter optimization with GridSearchCV
- **KNN**: Training K-Nearest Neighbors algorithm, performing hyperparameter optimization with GridSearchCV
- **Random Forest**: Training tree-based ensemble model, performing hyperparameter optimization with GridSearchCV
- **XGBoost**: Training gradient boosting model, performing hyperparameter optimization with GridSearchCV

For each model:
- Addressing class imbalance with SMOTE
- Evaluating model performance with cross-validation
- Finding best hyperparameters with GridSearchCV
- Saving trained models

### 5. Model Evaluation

By evaluating trained models on test set:

- Calculating ROC-AUC score
- Calculating Gini coefficient
- Calculating KS statistic
- Calculating accuracy, precision, recall, and F1 score
- Creating confusion matrix
- Drawing ROC curves
- Performing feature importance analysis

### 6. Visualization

By visualizing model results:

- Creating ROC curves and confusion matrices for each model
- Visualizing cross-validation results
- Creating feature importance plots
- Creating model comparison plots
- Visualizing all metrics comparatively

### 7. Reporting

By saving results to CSV files:

- Saving test set results (`reports/model_test_results.csv`)
- Saving cross-validation results (`reports/model_cv_results.csv`)
- Saving model comparison summary (`reports/model_comparison_summary.csv`)

### Key Findings:

- Data preprocessing and preparation completed for model training
- Missing values imputed using appropriate techniques
- Data types standardized and optimized
- Feature engineering performed using feature_engineering_test.py module
- Correlation analysis conducted; no significant multicollinearity detected
- Machine learning algorithms selected and implemented
- Results visualized with comprehensive plots
- Model performance metrics evaluated (ROC-AUC, Gini Coefficient, KS Statistic, Accuracy, Precision, Recall, F1-Score)
- Feature importance analysis performed to identify key drivers of default prediction across different model architectures
- Comparative model evaluation and selection recommendations provided based on performance metrics
- Detailed results exported to CSV files for further analysis and documentation
