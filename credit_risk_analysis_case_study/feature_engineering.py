"""
Feature Engineering Module
This module handles feature engineering operations including:
- Creating new engineered features
- Feature transformations
"""

import pandas as pd
import numpy as np
import warnings
import re
from typing import Optional
import config

warnings.filterwarnings('ignore')


def create_engineered_features(df: pd.DataFrame, config_module=None, training_stats=None, selected_features=None) -> pd.DataFrame:
    """
    Create engineered features based on configuration.
    """
    if config_module is None:
        config_module = config
    
    print("=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    
    df_eng = df.copy()
    engineered_features_created = []
    
    if training_stats is None:
        training_stats = {}
        for col in ['annual_inc', 'loan_amnt', 'total_acc', 'open_acc']:
            if col in df_eng.columns:
                training_stats[col] = {
                    'q20': df_eng[col].quantile(0.2),
                    'q25': df_eng[col].quantile(0.25),
                    'q75': df_eng[col].quantile(0.75),
                    'q80': df_eng[col].quantile(0.8)
                }
    
    features_to_create = config_module.ENGINEERED_FEATURES
    if selected_features is not None:
        features_to_create = {k: v for k, v in features_to_create.items() if k in selected_features}
        print(f"Creating only {len(selected_features)} selected features: {selected_features}")
    
    for feature_name, feature_config in features_to_create.items():
        try:
            formula = feature_config['formula']
            description = feature_config.get('description', '')
            if 'quantile' in formula and training_stats:
                for col_name in ['annual_inc', 'loan_amnt', 'total_acc', 'open_acc']:
                    if col_name in training_stats:
                        formula = formula.replace(f'{col_name}.quantile(0.75)', str(training_stats[col_name]['q75']))
                        formula = formula.replace(f'{col_name}.quantile(0.25)', str(training_stats[col_name]['q25']))
                        formula = formula.replace(f'{col_name}.quantile(0.8)', str(training_stats[col_name].get('q80', df_eng[col_name].quantile(0.8))))
                        formula = formula.replace(f'{col_name}.quantile(0.2)', str(training_stats[col_name].get('q20', df_eng[col_name].quantile(0.2))))
            
            if '.astype(str).str.' in formula or '.str.' in formula:
                for col in df_eng.columns:
                    if f'{col}.astype(str)' in formula or f'{col}.str.' in formula:
                        formula = formula.replace(f'{col}.astype(str)', f"df_eng['{col}'].astype(str)")
                        formula = formula.replace(f'{col}.str.', f"df_eng['{col}'].str.")
            
            safe_formula = formula
            cols_sorted = sorted(df_eng.columns, key=len, reverse=True)
            for col in cols_sorted:
                if f"df_eng['{col}']" in safe_formula:
                    continue
                pattern = r'\b' + re.escape(col) + r'\b'
                safe_formula = re.sub(pattern, f"df_eng['{col}']", safe_formula)
            
            df_eng[feature_name] = eval(safe_formula)
            df_eng[feature_name] = df_eng[feature_name].replace([np.inf, -np.inf], np.nan)
            engineered_features_created.append(feature_name)
            print(f"   ✓ Created '{feature_name}': {description}")
            
        except Exception as e:
            print(f"   ✗ Failed to create '{feature_name}': {str(e)}")
            continue
    
    print(f"\n✓ Feature engineering complete!")
    print(f"  - Original features: {len(df.columns)}")
    print(f"  - Engineered features created: {len(engineered_features_created)}")
    print(f"  - Total features: {len(df_eng.columns)}")
    
    return df_eng

def get_feature_lists(config_module=None) -> dict:
    """
    Get feature lists from configuration.
    """
    if config_module is None:
        config_module = config
    
    return {
        'numeric_features': config_module.FEATURE_LISTS['numeric_features'],
        'ordinal_features': config_module.FEATURE_LISTS['ordinal_features'],
        'low_cardinality_nominal': config_module.FEATURE_LISTS['low_cardinality_nominal'],
        'high_cardinality_nominal': config_module.FEATURE_LISTS['high_cardinality_nominal']
    }


def validate_features(df: pd.DataFrame, feature_lists: dict) -> dict:
    """
    Validate that all required features exist in the dataset.
    """
    validation_results = {
        'missing_features': {},
        'available_features': {},
        'all_valid': True }
    
    for feature_type, features in feature_lists.items():
        missing = [f for f in features if f not in df.columns]
        available = [f for f in features if f in df.columns]
        
        validation_results['missing_features'][feature_type] = missing
        validation_results['available_features'][feature_type] = available
        
        if missing:
            validation_results['all_valid'] = False
            print(f"⚠ Warning: Missing {len(missing)} {feature_type}: {missing}")
    
    if validation_results['all_valid']:
        print("✓ All required features are available")
    
    return validation_results


def get_selected_features_for_training(df: pd.DataFrame, config_module=None) -> list:
    """
    Get only selected engineered features for training (DTI, Credit_Utilization, Payment_to_Income_Ratio, 
    Average_Credit_Line_Size, lti).
    Only returns the selected engineered features, not original features or other engineered features.
    """
    if config_module is None:
        config_module = config
    
    selected_engineered_features = [
        'DTI',
        'Credit_Utilization',
        'Payment_to_Income_Ratio',
        'Average_Credit_Line_Size',
        'lti'
    ]
    
    available_selected = [f for f in selected_engineered_features if f in df.columns]
    missing_selected = [f for f in selected_engineered_features if f not in df.columns]
    
    if missing_selected:
        print(f"⚠ Warning: {len(missing_selected)} selected engineered features are missing: {missing_selected}")
        print(f"   Available selected features: {available_selected}")
    
    if not available_selected:
        print("⚠ Error: None of the selected engineered features are available in the dataset!")
        print(f"   Dataset columns: {list(df.columns)}")
        return []
    
    print(f"✓ Selected {len(available_selected)} engineered features for training: {available_selected}")
    
    return available_selected

