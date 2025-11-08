"""
Configuration file for Credit Risk Analysis
This file contains all configurable parameters for data cleaning, feature engineering, and modeling.
"""

# ============================================================================
# DATA CLEANING CONFIGURATION
# ============================================================================

# Columns to drop (Data Leakage and Multicollinearity)
COLUMNS_TO_DROP = {
    # Data Leakage Variables (variables that contain information about the target)
    'data_leakage': [
        'total_pymnt',           # Total payment - contains future information
        'total_pymnt_inv',       # Total payment to investors - contains future information
        'last_pymnt_amnt'        # Last payment amount - contains future information
    ],
    # Multicollinearity Variables (highly correlated with other features)
    'multicollinearity': [
        'funded_amnt',           # Highly correlated with loan_amnt
        'funded_amnt_inv'        # Highly correlated with loan_amnt
        # Note: installment is kept for DTI and Payment-to-Income Ratio features
    ],
    # ID Columns (not useful for modeling)
    'id_columns': [
        'id',
        'member_id'
    ]
}

# Missing Value Imputation Strategy
MISSING_VALUE_IMPUTATION = {
    # Fill with 'Unknown' (categorical features)
    'fill_unknown': [
        'emp_title'             # Employment title
    ],
    # Fill with 0 (numerical features that should be 0 if missing)
    'fill_zero': [
        'delinq_2yrs'            # Number of delinquencies - should be 0 if missing
    ],
    # Fill with median (numerical features)
    'fill_median': [
        # All other numerical features will be filled with median in the pipeline
        # This is handled automatically in the preprocessing pipeline
    ]
}

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# Features to use in modeling
FEATURE_LISTS = {
    # Numerical features
    'numeric_features': [
        'loan_amnt',             # Loan amount
        'installment',           # Monthly installment payment
        'annual_inc',            # Annual income
        'delinq_2yrs',           # Number of delinquencies in last 2 years
        'open_acc',              # Number of open accounts
        'total_acc',             # Total number of accounts
        'tot_coll_amt',          # Total collection amount
        'tot_cur_bal',           # Total current balance
        'total_rev_hi_lim'       # Total revolving credit limit
    ],
    # Ordinal features (ordered categorical)
    'ordinal_features': [
        'emp_length'             # Employment length (ordered)
    ],
    # Low cardinality nominal features (unordered categorical with few categories)
    'low_cardinality_nominal': [
        'term'                   # Loan term (36/60 months)
    ],
    # High cardinality nominal features (unordered categorical with many categories)
    'high_cardinality_nominal': [
        'emp_title'              # Employment title (many unique values)
    ]
}

# Engineered features to create - Çok fazla feature üretelim (alakalı alakasız)
ENGINEERED_FEATURES = {
    # Basic ratio features
    'DTI': {
        'formula': 'installment / (annual_inc + 1e-6)',
        'description': 'Debt-to-Income ratio (DTI)'
    },
    'Credit_Utilization': {
        'formula': 'tot_cur_bal / (total_rev_hi_lim + 1e-6)',
        'description': 'Credit Utilization'
    },
    'Payment_to_Income_Ratio': {
        'formula': 'installment / ((annual_inc / 12) + 1e-6)',
        'description': 'Payment-to-Income Ratio'
    },
    'Average_Credit_Line_Size': {
        'formula': 'tot_cur_bal / (open_acc + 1e-6)',
        'description': 'Average Credit Line Size'
    },
    'lti': {
        'formula': 'loan_amnt / (annual_inc + 1e-6)',
        'description': 'Loan-to-Income ratio (LTI)'
    },
    'debt_to_income': {
        'formula': 'tot_cur_bal / annual_inc',
        'description': 'Debt to income ratio'
    },
    'loan_to_income': {
        'formula': 'loan_amnt / (annual_inc + 1e-6)',
        'description': 'Loan to income ratio (LTI)'
    },
    'utilization_rate': {
        'formula': 'tot_cur_bal / (total_rev_hi_lim + 1e-6)',
        'description': 'Credit utilization rate'
    },
    'account_age_ratio': {
        'formula': 'open_acc / (total_acc + 1e-6)',
        'description': 'Open account ratio'
    },
    'has_delinquency': {
        'formula': '(delinq_2yrs > 0).astype(int)',
        'description': 'Binary indicator for delinquency'
    },
    'partial_payment': {
        'formula': 'tot_coll_amt / loan_amnt',
        'description': 'Partial payment ratio'
    },
    'income_per_account': {
        'formula': 'annual_inc / total_acc',
        'description': 'Income per account'
    },
    'loan_per_month': {
        'formula': 'loan_amnt / term',
        'description': 'Loan amount per month'
    },
    'collection_ratio': {
        'formula': 'tot_coll_amt / tot_cur_bal',
        'description': 'Collection to balance ratio'
    },
    # Additional ratio features
    'balance_to_income': {
        'formula': 'tot_cur_bal / annual_inc',
        'description': 'Balance to income ratio'
    },
    'collection_to_income': {
        'formula': 'tot_coll_amt / annual_inc',
        'description': 'Collection to income ratio'
    },
    'limit_to_income': {
        'formula': 'total_rev_hi_lim / annual_inc',
        'description': 'Credit limit to income ratio'
    },
    'loan_to_balance': {
        'formula': 'loan_amnt / tot_cur_bal',
        'description': 'Loan to balance ratio'
    },
    'loan_to_limit': {
        'formula': 'loan_amnt / total_rev_hi_lim',
        'description': 'Loan to credit limit ratio'
    },
    'collection_to_limit': {
        'formula': 'tot_coll_amt / total_rev_hi_lim',
        'description': 'Collection to limit ratio'
    },
    'balance_to_limit': {
        'formula': 'tot_cur_bal / total_rev_hi_lim',
        'description': 'Balance to limit ratio'
    },
    # Account-based features
    'closed_accounts': {
        'formula': 'total_acc - open_acc',
        'description': 'Number of closed accounts'
    },
    'closed_account_ratio': {
        'formula': '(total_acc - open_acc) / total_acc',
        'description': 'Ratio of closed accounts'
    },
    'accounts_per_year': {
        'formula': 'total_acc / (emp_length + 1)',
        'description': 'Accounts per year of employment'
    },
    'open_accounts_per_year': {
        'formula': 'open_acc / (emp_length + 1)',
        'description': 'Open accounts per year of employment'
    },
    # Income-based features
    'income_squared': {
        'formula': 'annual_inc ** 2',
        'description': 'Income squared'
    },
    'income_log': {
        'formula': 'np.log1p(annual_inc)',
        'description': 'Log of income'
    },
    'income_sqrt': {
        'formula': 'np.sqrt(annual_inc)',
        'description': 'Square root of income'
    },
    'income_per_month': {
        'formula': 'annual_inc / 12',
        'description': 'Monthly income'
    },
    'loan_to_monthly_income': {
        'formula': 'loan_amnt / (annual_inc / 12)',
        'description': 'Loan to monthly income ratio'
    },
    # Loan-based features
    'loan_squared': {
        'formula': 'loan_amnt ** 2',
        'description': 'Loan amount squared'
    },
    'loan_log': {
        'formula': 'np.log1p(loan_amnt)',
        'description': 'Log of loan amount'
    },
    'loan_sqrt': {
        'formula': 'np.sqrt(loan_amnt)',
        'description': 'Square root of loan amount'
    },
    'loan_per_year': {
        'formula': 'loan_amnt / (emp_length + 1)',
        'description': 'Loan per year of employment'
    },
    # Balance-based features
    'balance_squared': {
        'formula': 'tot_cur_bal ** 2',
        'description': 'Balance squared'
    },
    'balance_log': {
        'formula': 'np.log1p(tot_cur_bal)',
        'description': 'Log of balance'
    },
    'balance_sqrt': {
        'formula': 'np.sqrt(tot_cur_bal)',
        'description': 'Square root of balance'
    },
    'balance_per_account': {
        'formula': 'tot_cur_bal / total_acc',
        'description': 'Balance per account'
    },
    'balance_per_open_account': {
        'formula': 'tot_cur_bal / open_acc',
        'description': 'Balance per open account'
    },
    # Collection-based features
    'collection_squared': {
        'formula': 'tot_coll_amt ** 2',
        'description': 'Collection amount squared'
    },
    'collection_log': {
        'formula': 'np.log1p(tot_coll_amt)',
        'description': 'Log of collection amount'
    },
    'collection_per_account': {
        'formula': 'tot_coll_amt / total_acc',
        'description': 'Collection per account'
    },
    # Limit-based features
    'limit_squared': {
        'formula': 'total_rev_hi_lim ** 2',
        'description': 'Credit limit squared'
    },
    'limit_log': {
        'formula': 'np.log1p(total_rev_hi_lim)',
        'description': 'Log of credit limit'
    },
    'limit_per_account': {
        'formula': 'total_rev_hi_lim / total_acc',
        'description': 'Credit limit per account'
    },
    'available_credit': {
        'formula': 'total_rev_hi_lim - tot_cur_bal',
        'description': 'Available credit'
    },
    'available_credit_ratio': {
        'formula': '(total_rev_hi_lim - tot_cur_bal) / total_rev_hi_lim',
        'description': 'Available credit ratio'
    },
    # Employment-based features
    'emp_length_squared': {
        'formula': 'emp_length ** 2',
        'description': 'Employment length squared'
    },
    'emp_length_log': {
        'formula': 'np.log1p(emp_length)',
        'description': 'Log of employment length'
    },
    'income_per_emp_year': {
        'formula': 'annual_inc / (emp_length + 1)',
        'description': 'Income per year of employment'
    },
    # Delinquency-based features
    'delinq_per_account': {
        'formula': 'delinq_2yrs / total_acc',
        'description': 'Delinquency per account'
    },
    'delinq_per_open_account': {
        'formula': 'delinq_2yrs / open_acc',
        'description': 'Delinquency per open account'
    },
    'has_multiple_delinq': {
        'formula': '(delinq_2yrs > 1).astype(int)',
        'description': 'Has multiple delinquencies'
    },
    # Interaction features
    'loan_income_interaction': {
        'formula': 'loan_amnt * annual_inc',
        'description': 'Loan and income interaction'
    },
    'balance_limit_interaction': {
        'formula': 'tot_cur_bal * total_rev_hi_lim',
        'description': 'Balance and limit interaction'
    },
    'loan_emp_interaction': {
        'formula': 'loan_amnt * emp_length',
        'description': 'Loan and employment length interaction'
    },
    'income_emp_interaction': {
        'formula': 'annual_inc * emp_length',
        'description': 'Income and employment length interaction'
    },
    'loan_accounts_interaction': {
        'formula': 'loan_amnt * total_acc',
        'description': 'Loan and accounts interaction'
    },
    'balance_accounts_interaction': {
        'formula': 'tot_cur_bal * total_acc',
        'description': 'Balance and accounts interaction'
    },
    # Polynomial features
    'loan_income_ratio_squared': {
        'formula': '(loan_amnt / annual_inc) ** 2',
        'description': 'Loan to income ratio squared'
    },
    'utilization_rate_squared': {
        'formula': '(tot_cur_bal / total_rev_hi_lim) ** 2',
        'description': 'Utilization rate squared'
    },
    # Statistical features
    'total_debt': {
        'formula': 'tot_cur_bal + loan_amnt',
        'description': 'Total debt'
    },
    'debt_to_income_total': {
        'formula': '(tot_cur_bal + loan_amnt) / annual_inc',
        'description': 'Total debt to income ratio'
    },
    'net_worth_estimate': {
        'formula': 'annual_inc - tot_cur_bal - loan_amnt',
        'description': 'Net worth estimate'
    },
    # Term-based features
    'term_36': {
        'formula': '(term == 36).astype(int)',
        'description': 'Is 36 month term'
    },
    'term_60': {
        'formula': '(term == 60).astype(int)',
        'description': 'Is 60 month term'
    },
    'loan_per_term_month': {
        'formula': 'loan_amnt / term',
        'description': 'Loan per term month'
    },
    # Binning features
    'high_income': {
        'formula': '(annual_inc > annual_inc.quantile(0.75)).astype(int)',
        'description': 'High income indicator'
    },
    'low_income': {
        'formula': '(annual_inc < annual_inc.quantile(0.25)).astype(int)',
        'description': 'Low income indicator'
    },
    'high_loan': {
        'formula': '(loan_amnt > loan_amnt.quantile(0.75)).astype(int)',
        'description': 'High loan indicator'
    },
    'low_loan': {
        'formula': '(loan_amnt < loan_amnt.quantile(0.25)).astype(int)',
        'description': 'Low loan indicator'
    },
    'high_utilization': {
        'formula': '((tot_cur_bal / total_rev_hi_lim) > 0.8).astype(int)',
        'description': 'High utilization indicator'
    },
    'low_utilization': {
        'formula': '((tot_cur_bal / total_rev_hi_lim) < 0.2).astype(int)',
        'description': 'Low utilization indicator'
    },
    # Account status features
    'many_accounts': {
        'formula': '(total_acc > total_acc.quantile(0.75)).astype(int)',
        'description': 'Many accounts indicator'
    },
    'few_accounts': {
        'formula': '(total_acc < total_acc.quantile(0.25)).astype(int)',
        'description': 'Few accounts indicator'
    },
    'many_open_accounts': {
        'formula': '(open_acc > open_acc.quantile(0.75)).astype(int)',
        'description': 'Many open accounts indicator'
    },
    'few_open_accounts': {
        'formula': '(open_acc < open_acc.quantile(0.25)).astype(int)',
        'description': 'Few open accounts indicator'
    },
    # Employment features
    'long_employment': {
        'formula': '(emp_length >= 5).astype(int)',
        'description': 'Long employment indicator'
    },
    'short_employment': {
        'formula': '(emp_length < 2).astype(int)',
        'description': 'Short employment indicator'
    },
    'new_employee': {
        'formula': '(emp_length == 0).astype(int)',
        'description': 'New employee indicator'
    },
    # Risk indicators
    'high_risk_loan': {
        'formula': '((loan_amnt / annual_inc) > 0.5).astype(int)',
        'description': 'High risk loan indicator'
    },
    'high_risk_utilization': {
        'formula': '((tot_cur_bal / total_rev_hi_lim) > 0.9).astype(int)',
        'description': 'High risk utilization indicator'
    },
    'high_risk_debt': {
        'formula': '((tot_cur_bal / annual_inc) > 2).astype(int)',
        'description': 'High risk debt indicator'
    },
    # Additional risk ratios with safety
    'monthly_inc': {
        'formula': 'annual_inc / 12',
        'description': 'Monthly income'
    },
    'lti': {
        'formula': 'loan_amnt / (annual_inc + 1e-6)',
        'description': 'Loan-to-Income ratio'
    },
    'util_ratio': {
        'formula': 'tot_cur_bal / (total_rev_hi_lim + 1e-6)',
        'description': 'Credit utilization ratio'
    },
    'open_acc_ratio': {
        'formula': 'open_acc / (total_acc + 1e-6)',
        'description': 'Open account ratio'
    },
    # Interaction features
    'inc_per_open_acc': {
        'formula': 'annual_inc / (open_acc + 1)',
        'description': 'Income per open account'
    },
    'inc_x_delinq': {
        'formula': 'annual_inc * (delinq_2yrs + 1)',
        'description': 'Income times delinquency'
    },
    'loan_x_term': {
        'formula': 'loan_amnt * term',
        'description': 'Loan amount times term'
    },
    # Polynomial features for ratios
    'lti_sq': {
        'formula': '(loan_amnt / (annual_inc + 1e-6)) ** 2',
        'description': 'Loan-to-Income ratio squared'
    },
    'util_ratio_sq': {
        'formula': '(tot_cur_bal / (total_rev_hi_lim + 1e-6)) ** 2',
        'description': 'Utilization ratio squared'
    },
    # Behavioral features
    'has_collection': {
        'formula': '(tot_coll_amt > 0).astype(int)',
        'description': 'Has collection amount'
    },
    'credit_history_proxy': {
        'formula': 'total_acc - open_acc',
        'description': 'Credit history proxy (closed accounts)'
    },
    # Creative features
    'is_loan_amnt_rounded': {
        'formula': '(loan_amnt % 1000 == 0).astype(int)',
        'description': 'Is loan amount rounded to thousands'
    },
    'is_annual_inc_rounded': {
        'formula': '(annual_inc % 1000 == 0).astype(int)',
        'description': 'Is annual income rounded to thousands'
    },
    # Income groups (will be handled in feature engineering with training stats)
    'inc_group_high': {
        'formula': '(annual_inc > annual_inc.quantile(0.8)).astype(int)',
        'description': 'Very high income group'
    },
    'inc_group_low': {
        'formula': '(annual_inc < annual_inc.quantile(0.2)).astype(int)',
        'description': 'Very low income group'
    },
    # Employment title features (will be handled separately)
    'emp_title_length': {
        'formula': 'emp_title.astype(str).str.len()',
        'description': 'Employment title length'
    }
}

# ============================================================================
# MODELING CONFIGURATION
# ============================================================================

# Train/Test Split
TRAIN_TEST_SPLIT = {
    'test_size': 0.2,
    'random_state': 42
}

# Cross-Validation
CROSS_VALIDATION = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}

# Model Hyperparameters (GridSearchCV)
MODEL_PARAM_GRIDS = {
    'logistic_regression': {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    },
    'hist_gradient_boosting': {
        'classifier__max_iter': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    },
    'knn': {
        'classifier__n_neighbors': [5, 10, 15, 20],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'random_forest': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'xgboost': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0]
    }
}

# Model Evaluation Metrics
EVALUATION_METRICS = [
    'roc_auc',
    'accuracy',
    'precision',
    'recall',
    'f1_score'
]

# ============================================================================
# FILE PATHS
# ============================================================================

# Data paths
DATA_PATHS = {
    'raw_data': 'data/credit_risk_case.xlsx',
    'cleaned_data': 'data/cleaned_data.csv',
    'train_data': 'data/train_data.csv',
    'test_data': 'data/test_data.csv'
}

# Model paths
MODEL_PATHS = {
    'logistic_regression': 'models/pd_logistic_pipeline.pkl',
    'hist_gradient_boosting': 'models/pd_gbm_pipeline.pkl',
    'knn': 'models/pd_knn_pipeline.pkl',
    'random_forest': 'models/pd_rf_pipeline.pkl',
    'xgboost': 'models/pd_xgb_pipeline.pkl'
}

# Output paths
OUTPUT_PATHS = {
    'results': 'results/',
    'plots': 'plots/',
    'reports': 'reports/'
}

