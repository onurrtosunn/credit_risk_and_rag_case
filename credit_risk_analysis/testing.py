"""
Model Testing/Evaluation Module
This module handles model evaluation on test data including:
- Model predictions
- Performance metrics calculation
- Model comparison
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
import config

warnings.filterwarnings('ignore')


def load_model(model_path: str) -> object:
    """
    Load a trained model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model") -> Dict:
    """
    Evaluate a model on test data.
    """
    print("=" * 80)
    print(f"EVALUATING {model_name.upper()} ON TEST SET")
    print("=" * 80)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    evaluation_results = {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'roc_curve': (fpr, tpr, thresholds)
    }
    
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(class_report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    return evaluation_results


def evaluate_all_models(X_test: pd.DataFrame, y_test: pd.Series, config_module=None, model_paths: Optional[Dict[str, str]] = None) -> Dict[str, Dict]:
    """
    Evaluate all models on test data.
    """
    if config_module is None:
        config_module = config
    
    if model_paths is None:
        model_paths = config_module.MODEL_PATHS
    
    evaluation_results = {}
    
    for model_name, model_path in model_paths.items():
        try:
            model = load_model(model_path)
            results = evaluate_model(model, X_test, y_test, model_name)
            evaluation_results[model_name] = results
            
        except Exception as e:
            print(f"\n✗ Error evaluating {model_name}: {str(e)}")
            continue
    
    return evaluation_results

def compare_models(evaluation_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare all models based on evaluation metrics.
    """
    comparison_data = []
    
    for model_name, results in evaluation_results.items():
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'ROC-AUC': results['roc_auc'],
            'F1-Score': results['f1_score'],
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    return comparison_df


def get_best_model(evaluation_results: Dict[str, Dict], metric: str = 'roc_auc') -> Tuple[str, Dict]:
    """
    Get the best model based on a metric.
    """
    best_model_name = None
    best_score = -np.inf
    
    for model_name, results in evaluation_results.items():
        score = results.get(metric, -np.inf)
        if score > best_score:
            best_score = score
            best_model_name = model_name
    
    if best_model_name is None:
        raise ValueError("No models found in evaluation results")
    
    return best_model_name, evaluation_results[best_model_name]

