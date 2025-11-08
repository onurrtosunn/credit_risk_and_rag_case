"""
Data Cleaning Module
This module handles all data cleaning operations including:
- Removing data leakage variables
- Handling missing values
- Data type conversions
- Business logic constraints
"""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple
import config

warnings.filterwarnings('ignore')

EMPLOYMENT_LENGTH_MAP = {
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10
}


def _apply_business_constraints(X: pd.DataFrame) -> pd.DataFrame:
    """Apply business logic constraints to ensure data consistency."""
    
    if 'loan_amnt' in X.columns and 'funded_amnt' in X.columns:
        count = (X['funded_amnt'] > X['loan_amnt']).sum()
        if count > 0:
            print(f"   [Constraint] Corrected {count} rows where 'funded_amnt > loan_amnt'")
            X['funded_amnt'] = np.minimum(X['loan_amnt'], X['funded_amnt'])
    
    if 'funded_amnt' in X.columns and 'funded_amnt_inv' in X.columns:
        count = (X['funded_amnt_inv'] > X['funded_amnt']).sum()
        if count > 0:
            print(f"   [Constraint] Corrected {count} rows where 'funded_amnt_inv > funded_amnt'")
            X['funded_amnt_inv'] = np.minimum(X['funded_amnt'], X['funded_amnt_inv'])
    
    if 'total_pymnt' in X.columns and 'total_pymnt_inv' in X.columns:
        count = (X['total_pymnt_inv'] > X['total_pymnt']).sum()
        if count > 0:
            print(f"   [Constraint] Corrected {count} rows where 'total_pymnt_inv > total_pymnt'")
            X['total_pymnt_inv'] = np.minimum(X['total_pymnt'], X['total_pymnt_inv'])
    
    if 'total_pymnt' in X.columns and 'last_pymnt_amnt' in X.columns:
        count = (X['last_pymnt_amnt'] > X['total_pymnt']).sum()
        if count > 0:
            print(f"   [Constraint] Corrected {count} rows where 'last_pymnt_amnt > total_pymnt'")
            X['last_pymnt_amnt'] = np.minimum(X['total_pymnt'], X['last_pymnt_amnt'])
    
    print("   ✓ Business logic constraints applied")
    return X


def _clean_emp_length(X: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert emp_length to numeric using mapping.
    """
    if 'emp_length' not in X.columns:
        return X
    
    X['emp_length'] = X['emp_length'].astype(str).str.strip()
    X['emp_length'] = X['emp_length'].map(EMPLOYMENT_LENGTH_MAP)
    
    return X


def clean_data(
    df: pd.DataFrame,
    target_column: str = 'default',
    config_module=None) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Clean the dataset by removing problematic columns and handling missing values.
    """
    if config_module is None:
        config_module = config
    
    print("=" * 80)
    print("DATA CLEANING")
    print("=" * 80)
    
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    removed_columns = []
    
    print("\n1. Removing ID columns...")
    id_cols = [col for col in config_module.COLUMNS_TO_DROP['id_columns'] 
               if col in X.columns]
    if id_cols:
        X = X.drop(columns=id_cols)
        removed_columns.extend(id_cols)
        print(f"   ✓ Removed {len(id_cols)} columns: {id_cols}")
    
    print("\n2. Converting emp_length...")
    X = _clean_emp_length(X)
    if 'emp_length' in X.columns:
        missing_count = X['emp_length'].isna().sum()
        if missing_count > 0:
            median_emp_length = X['emp_length'].median()
            X['emp_length'] = X['emp_length'].fillna(median_emp_length)
            print(f"   ✓ Converted emp_length to numeric using mapping, filled {missing_count} missing values with median ({median_emp_length:.1f})")
        else:
            print("   ✓ Converted emp_length to numeric using mapping")
    
    print("\n3. Applying business logic constraints...")
    X = _apply_business_constraints(X)
    
    print("\n4. Handling missing values...")
    
    if 'emp_title' in X.columns:
        X['emp_title'] = X['emp_title'].fillna('Unknown')
        print("   ✓ Filled emp_title missing values with 'Unknown'")
    
    if 'delinq_2yrs' in X.columns:
        X['delinq_2yrs'] = X['delinq_2yrs'].fillna(0)
        print("   ✓ Filled delinq_2yrs missing values with 0")
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    missing_numeric = [col for col in numeric_cols if X[col].isna().any()]
    if missing_numeric:
        for col in missing_numeric:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
        print(f"   ✓ Filled {len(missing_numeric)} numerical columns with median: {missing_numeric}")
    
    print(f"\n✓ Data cleaning complete!")
    print(f"  - Original features: {df.shape[1] - 1}")
    print(f"  - Remaining features: {X.shape[1]}")
    print(f"  - Removed columns: {len(removed_columns)}")
    
    return X, y, removed_columns

def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of missing values in the dataset.

    """
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).values,
        'Data_Type': df.dtypes.values
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
    missing_summary = missing_summary.sort_values('Missing_Percentage', ascending=False)
    return missing_summary

def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Perform basic data quality checks.
    """
    quality_report = {
        'shape': df.shape,
        'duplicates': df.duplicated().sum(),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'missing_by_column': get_missing_value_summary(df)
    }
    return quality_report


