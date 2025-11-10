"""
Model Training Module
This module handles model training with GridSearchCV for hyperparameter tuning.
Supports multiple models: Logistic Regression, HistGradientBoosting, KNN, 
Random Forest, and XGBoost.
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import (
    StandardScaler, OrdinalEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, confusion_matrix
)
import xgboost as xgb
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import config
import feature_engineering

# Optional: WandB for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

warnings.filterwarnings('ignore')


def create_preprocessors(X_train: pd.DataFrame, config_module=None):
    """
    Create preprocessing pipelines for different model types.
    """
    if config_module is None:
        config_module = config
    
    # Get available columns from X_train
    available_columns = set(X_train.columns)
    
    feature_lists = feature_engineering.get_feature_lists(config_module)
    
    # Filter feature lists to only include features that exist in X_train
    numeric_features = [f for f in feature_lists['numeric_features'] 
                       if f in available_columns]
    ordinal_features = [f for f in feature_lists['ordinal_features'] 
                        if f in available_columns]
    low_cardinality_nominal = [f for f in feature_lists['low_cardinality_nominal'] 
                              if f in available_columns]
    high_cardinality_nominal = [f for f in feature_lists['high_cardinality_nominal'] 
                                if f in available_columns]
    
    # For any remaining features in X_train that are not in config lists,
    # treat them as numeric features (for engineered features)
    remaining_features = available_columns - set(numeric_features) - set(ordinal_features) - set(low_cardinality_nominal) - set(high_cardinality_nominal)
    if remaining_features:
        # Check if they are numeric or categorical
        for feat in remaining_features:
            if X_train[feat].dtype in ['int64', 'float64']:
                numeric_features.append(feat)
            else:
                # Treat as low cardinality nominal if not too many unique values
                if X_train[feat].nunique() <= 10:
                    low_cardinality_nominal.append(feat)
                else:
                    high_cardinality_nominal.append(feat)
    
    # Numeric transformer for Logistic Regression (with scaling)
    numeric_transformer_logistic = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Numeric transformer for tree-based models (no scaling needed)
    numeric_transformer_gbm = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # Ordinal transformer
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Low cardinality nominal transformer
    low_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    # High cardinality nominal transformer (Target Encoding)
    high_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('target', TargetEncoder())
    ])
    
    # Build transformers list dynamically (only include non-empty feature lists)
    transformers_logistic = []
    transformers_gbm = []
    
    if numeric_features:
        transformers_logistic.append(('num', numeric_transformer_logistic, numeric_features))
        transformers_gbm.append(('num', numeric_transformer_gbm, numeric_features))
    if ordinal_features:
        transformers_logistic.append(('ord', ordinal_transformer, ordinal_features))
        transformers_gbm.append(('ord', ordinal_transformer, ordinal_features))
    if low_cardinality_nominal:
        transformers_logistic.append(('low_card', low_card_transformer, low_cardinality_nominal))
        transformers_gbm.append(('low_card', low_card_transformer, low_cardinality_nominal))
    if high_cardinality_nominal:
        transformers_logistic.append(('high_card', high_card_transformer, high_cardinality_nominal))
        transformers_gbm.append(('high_card', high_card_transformer, high_cardinality_nominal))
    
    # Validate that we have at least some features
    total_features = len(numeric_features) + len(ordinal_features) + len(low_cardinality_nominal) + len(high_cardinality_nominal)
    if total_features == 0:
        raise ValueError("No features found in X_train that match the feature lists. Please check your data and configuration.")
    
    # Print feature categorization for debugging
    if len(numeric_features) > 0:
        print(f"  Numeric features ({len(numeric_features)}): {numeric_features}")
    if len(ordinal_features) > 0:
        print(f"  Ordinal features ({len(ordinal_features)}): {ordinal_features}")
    if len(low_cardinality_nominal) > 0:
        print(f"  Low cardinality nominal features ({len(low_cardinality_nominal)}): {low_cardinality_nominal}")
    if len(high_cardinality_nominal) > 0:
        print(f"  High cardinality nominal features ({len(high_cardinality_nominal)}): {high_cardinality_nominal}")
    
    # Combine transformers for Logistic Regression
    preprocessor_logistic = ColumnTransformer(
        transformers=transformers_logistic,
        remainder='drop'
    )
    
    # Combine transformers for tree-based models
    preprocessor_gbm = ColumnTransformer(
        transformers=transformers_gbm,
        remainder='drop'
    )
    
    return preprocessor_logistic, preprocessor_gbm


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config_module=None,
    use_wandb: bool = True,
    wandb_project: str = "credit-risk-analysis",
    wandb_run_name: Optional[str] = None
) -> Tuple[Pipeline, Dict]:
    """
    Train a single model with GridSearchCV.
    
    Args:
        model_name: Name of the model to train
        X_train: Training features
        y_train: Training target
        config_module: Configuration module (default: config)
        use_wandb: Whether to log to WandB (default: False)
        wandb_project: WandB project name
        wandb_run_name: WandB run name (default: model_name with timestamp)
    """
    if config_module is None:
        config_module = config
    
    # Initialize WandB if requested
    if use_wandb and WANDB_AVAILABLE:
        if wandb_run_name is None:
            wandb_run_name = f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_name": model_name,
                "n_samples": len(X_train),
                "n_features": X_train.shape[1],
                "class_distribution": {
                    "default": int(y_train.sum()),
                    "non_default": int((y_train == 0).sum())
                },
                "param_grid": config_module.MODEL_PARAM_GRIDS.get(model_name, {})
            }
        )
        print(f"✓ WandB initialized: {wandb_project}/{wandb_run_name}")
    elif use_wandb and not WANDB_AVAILABLE:
        print("⚠ WandB requested but not available. Install with: pip install wandb")
        use_wandb = False
    
    print("=" * 80)
    print(f"TRAINING MODEL: {model_name.upper()}")
    print("=" * 80)
    
    preprocessor_logistic, preprocessor_gbm = create_preprocessors(X_train, config_module)
    
    # Create model pipeline based on model type (with SMOTE)
    if model_name == 'logistic_regression':
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor_logistic),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ))
        ])
        param_grid = config_module.MODEL_PARAM_GRIDS['logistic_regression']
        
    elif model_name == 'hist_gradient_boosting':
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor_gbm),
            ('smote', SMOTE(random_state=42)),
            ('classifier', HistGradientBoostingClassifier(
                class_weight='balanced',
                random_state=42
            ))
        ])
        param_grid = config_module.MODEL_PARAM_GRIDS['hist_gradient_boosting']
        
    elif model_name == 'knn':
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor_logistic),
            ('smote', SMOTE(random_state=42)),
            ('classifier', KNeighborsClassifier())
        ])
        param_grid = config_module.MODEL_PARAM_GRIDS['knn']
        
    elif model_name == 'random_forest':
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor_gbm),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])
        param_grid = config_module.MODEL_PARAM_GRIDS['random_forest']
        
    elif model_name == 'xgboost':
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor_gbm),
            ('smote', SMOTE(random_state=42)),
            ('classifier', xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight
            ))
        ])
        param_grid = config_module.MODEL_PARAM_GRIDS['xgboost']
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # GridSearchCV
    cv = StratifiedKFold(
        n_splits=config_module.CROSS_VALIDATION['n_splits'],
        shuffle=config_module.CROSS_VALIDATION['shuffle'],
        random_state=config_module.CROSS_VALIDATION['random_state']
    )
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\nPerforming GridSearchCV...")
    print(f"Parameter grid: {param_grid}")
    
    # Fit model
    print("\nTraining model...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    training_info = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'model_name': model_name
    }
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")
    
    # Log to WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "best_cv_score": grid_search.best_score_,
            "best_params": grid_search.best_params_,
            "n_cv_splits": config_module.CROSS_VALIDATION['n_splits']
        })
        print("✓ Results logged to WandB")
    
    return best_model, training_info


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config_module=None,
    models_to_train: Optional[list] = None,
    use_wandb: bool = True,
    wandb_project: str = "credit-risk-analysis"
) -> Dict[str, Tuple[Pipeline, Dict]]:
    """
    Train all models.
    
    Args:
        X_train: Training features
        y_train: Training target
        config_module: Configuration module (default: config)
        models_to_train: List of models to train (default: all models)
        use_wandb: Whether to log to WandB (default: False)
        wandb_project: WandB project name
    """
    if config_module is None:
        config_module = config
    
    if models_to_train is None:
        models_to_train = [
            'logistic_regression',
            'hist_gradient_boosting',
            'knn',
            'random_forest',
            'xgboost'
        ]
    
    trained_models = {}
    
    for model_name in models_to_train:
        try:
            best_model, training_info = train_model(
                model_name, X_train, y_train, config_module,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_run_name=f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )
            trained_models[model_name] = (best_model, training_info)
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {str(e)}")
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({"error": str(e), "model": model_name})
            continue
    
    # Log comparison of all models to WandB
    if use_wandb and WANDB_AVAILABLE and trained_models:
        comparison_data = []
        for model_name, (model, info) in trained_models.items():
            comparison_data.append({
                "model": model_name,
                "cv_score": info['best_cv_score']
            })
        comparison_df = pd.DataFrame(comparison_data)
        wandb.log({"model_comparison": wandb.Table(dataframe=comparison_df)})
        print("✓ Model comparison logged to WandB")
    
    return trained_models


def save_models(
    trained_models: Dict[str, Pipeline],
    config_module=None,
    output_dir: str = 'models'
) -> None:
    """
    Save trained models to disk.
    """
    if config_module is None:
        config_module = config
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)
    
    for model_name, model_value in trained_models.items():
        try:
            # Handle both cases: model only or (model, training_info) tuple
            if isinstance(model_value, tuple):
                model, training_info = model_value
            else:
                model = model_value
            
            model_path = config_module.MODEL_PATHS.get(model_name, f'{output_dir}/{model_name}.pkl')
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"✓ Saved: {model_path}")
        except Exception as e:
            print(f"✗ Error saving {model_name}: {str(e)}")


def evaluate_model_cv(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    config_module=None,
    cv: Optional[int] = None,
    use_wandb: bool = False,
    model_name: str = "Model"
) -> Dict:
    """
    Evaluate a model using cross-validation with multiple metrics.
    
    Args:
        model: Trained model pipeline
        X: Features
        y: Target
        config_module: Configuration module (default: config)
        cv: Cross-validation strategy (default: StratifiedKFold)
        use_wandb: Whether to log to WandB (default: False)
        model_name: Name of the model for logging
    """
    if config_module is None:
        config_module = config
    
    if cv is None:
        cv = StratifiedKFold(
            n_splits=config_module.CROSS_VALIDATION['n_splits'],
            shuffle=config_module.CROSS_VALIDATION['shuffle'],
            random_state=config_module.CROSS_VALIDATION['random_state']
        )
    
    # Define scoring metrics
    scoring = {
        'roc_auc': 'roc_auc',
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    
    # Calculate mean and std for each metric
    cv_summary = {}
    for metric in scoring.keys():
        scores = cv_results[f'test_{metric}']
        cv_summary[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
    
    # Log to WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb_metrics = {}
        for metric, values in cv_summary.items():
            wandb_metrics[f"cv_{metric}_mean"] = values['mean']
            wandb_metrics[f"cv_{metric}_std"] = values['std']
        wandb.log(wandb_metrics)
        print(f"✓ CV results logged to WandB for {model_name}")
    
    return {
        'cv_summary': cv_summary,
        'cv_results': cv_results
    }


def get_feature_importance(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model"
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    """
    from sklearn.inspection import permutation_importance
    
    # Get the classifier from the pipeline
    classifier = model.named_steps['classifier']
    
    # Extract feature importance based on model type
    if hasattr(classifier, 'feature_importances_'):
        # Tree-based models (Random Forest, XGBoost, HistGradientBoosting)
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        # Linear models (Logistic Regression)
        # Use absolute value of coefficients
        importances = np.abs(classifier.coef_[0])
    else:
        # Models without feature importance (KNN) - use permutation importance
        print(f"⚠ Warning: {model_name} does not support native feature importance.")
        print(f"   Using permutation importance instead...")
        
        # Use permutation importance for KNN
        try:
            # Get a sample for faster computation
            sample_size = min(1000, len(X_train))
            X_sample = X_train.sample(n=sample_size, random_state=42)
            y_sample = y_train.loc[X_sample.index]
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_sample, y_sample,
                n_repeats=10,
                random_state=42,
                scoring='roc_auc',
                n_jobs=-1
            )
            importances = perm_importance.importances_mean
        except Exception as e:
            print(f"   ⚠ Error calculating permutation importance: {str(e)}")
            return pd.DataFrame({
                'feature': list(X_train.columns),
                'importance': [0] * len(X_train.columns)
            })
    
    # Get feature names from preprocessor - use get_feature_names_out if available
    preprocessor = model.named_steps['preprocessor']
    
    try:
        # Try to use get_feature_names_out (sklearn >= 1.0)
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names_processed = preprocessor.get_feature_names_out()
        else:
            # Fallback to manual extraction
            feature_names_processed = []
            if hasattr(preprocessor, 'transformers_'):
                for name, transformer, features in preprocessor.transformers_:
                    if name == 'num':
                        # Numeric features
                        feature_names_processed.extend(features)
                    elif name == 'ord':
                        # Ordinal features
                        feature_names_processed.extend(features)
                    elif name == 'low_card':
                        # One-hot encoded features
                        if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                            onehot_features = transformer.named_steps['onehot'].get_feature_names_out(features)
                            feature_names_processed.extend(onehot_features)
                        else:
                            # Fallback: use original feature names
                            feature_names_processed.extend(features)
                    elif name == 'high_card':
                        # Target encoded features
                        feature_names_processed.extend(features)
    except Exception as e:
        # If extraction fails, use original feature names
        print(f"⚠ Warning: Could not extract processed feature names: {str(e)}")
        feature_names_processed = list(X_train.columns)
    
    # If we couldn't extract feature names or count doesn't match, use original feature names
    if len(feature_names_processed) != len(importances):
        print(f"⚠ Warning: Feature count mismatch ({len(feature_names_processed)} vs {len(importances)}). Using original feature names.")
        feature_names_processed = list(X_train.columns)
        # If still mismatch, use indices
        if len(feature_names_processed) != len(importances):
            feature_names_processed = [f'feature_{i}' for i in range(len(importances))]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names_processed[:len(importances)],
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


def visualize_model_results(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize model results including ROC curve and confusion matrix.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0].plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].set_title(f'ROC Curve - {model_name}', fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'])
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('Actual', fontsize=11)
    axes[1].set_title(f'Confusion Matrix - {model_name}', fontsize=13, fontweight='bold')
    
    # Add metrics text
    metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
    axes[1].text(0.5, -0.15, metrics_text, transform=axes[1].transAxes,
                 ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{model_name}_results.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {os.path.join(save_path, f'{model_name}_results.png')}")
    
    plt.show()


def visualize_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str = "Model",
    top_n: int = 15,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize feature importance.
    """
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Feature Importance', fontsize=11)
    plt.ylabel('Features', fontsize=11)
    plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(top_features['importance'].values):
        plt.text(v, i, f' {v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{model_name}_feature_importance.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {os.path.join(save_path, f'{model_name}_feature_importance.png')}")
    
    plt.show()


def visualize_cv_results(
    cv_results: Dict,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize cross-validation results.
    """
    cv_summary = cv_results['cv_summary']
    
    # Prepare data for plotting
    metrics = list(cv_summary.keys())
    means = [cv_summary[m]['mean'] for m in metrics]
    stds = [cv_summary[m]['std'] for m in metrics]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(metrics))
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.4f}\n±{std:.4f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Metrics', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Cross-Validation Results - {model_name}', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics], rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, max(means) + max(stds) + 0.1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{model_name}_cv_results.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {os.path.join(save_path, f'{model_name}_cv_results.png')}")
    
    plt.show()


def summarize_cv_results(
    cv_results: Dict,
    model_name: str = "Model"
) -> pd.DataFrame:
    """
    Summarize cross-validation results in a DataFrame.
    """
    cv_summary = cv_results['cv_summary']
    
    summary_data = []
    for metric, values in cv_summary.items():
        summary_data.append({
            'Model': model_name,
            'Metric': metric.upper().replace('_', ' '),
            'Mean': values['mean'],
            'Std': values['std'],
            'Min': np.min(values['scores']),
            'Max': np.max(values['scores'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


def save_model_results_to_csv(
    test_results: Dict[str, Dict],
    cv_results_all: Optional[Dict[str, Dict]] = None,
    config_module=None,
    output_dir: str = 'reports'
) -> None:
    """
    Save model results to CSV files.
    """
    if config_module is None:
        config_module = config
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SAVING MODEL RESULTS TO CSV")
    print("=" * 80)
    
    # Save test set results
    test_results_df = pd.DataFrame(test_results).T
    test_results_df = test_results_df.sort_values('roc_auc', ascending=False)
    # Handle columns dynamically based on available metrics
    available_columns = []
    column_mapping = {
        'roc_auc': 'ROC-AUC',
        'gini': 'Gini',
        'ks_statistic': 'KS-Statistic',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score'
    }
    for col in ['roc_auc', 'gini', 'ks_statistic', 'accuracy', 'precision', 'recall', 'f1_score']:
        if col in test_results_df.columns:
            available_columns.append(column_mapping[col])
    test_results_df.columns = available_columns
    test_results_df.index.name = 'Model'
    
    test_csv_path = os.path.join(output_dir, 'model_test_results.csv')
    test_results_df.to_csv(test_csv_path, index=True)
    print(f"✓ Saved test set results to: {test_csv_path}")
    
    # Save cross-validation results if provided
    if cv_results_all:
        cv_summaries_all = []
        for model_name, cv_results in cv_results_all.items():
            cv_summary_df = summarize_cv_results(cv_results, model_name=model_name)
            cv_summaries_all.append(cv_summary_df)
        
        if cv_summaries_all:
            cv_summaries_combined = pd.concat(cv_summaries_all, ignore_index=True)
            cv_csv_path = os.path.join(output_dir, 'model_cv_results.csv')
            cv_summaries_combined.to_csv(cv_csv_path, index=False)
            print(f"✓ Saved cross-validation results to: {cv_csv_path}")
    
    # Save combined summary
    summary_data = []
    for model_name, results in test_results.items():
        summary_dict = {
            'Model': model_name.replace('_', ' ').title(),
            'ROC-AUC': results['roc_auc'],
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score']
        }
        # Add Gini and KS if they exist
        if 'gini' in results:
            summary_dict['Gini'] = results['gini']
        if 'ks_statistic' in results:
            summary_dict['KS-Statistic'] = results['ks_statistic']
        summary_data.append(summary_dict)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('ROC-AUC', ascending=False)
    
    summary_csv_path = os.path.join(output_dir, 'model_comparison_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✓ Saved model comparison summary to: {summary_csv_path}")


if __name__ == "__main__":
    print("Model Training Module")
    print("=" * 80)
    print("\nThis module provides functions for model training:")
    print("  - create_preprocessors(): Create preprocessing pipelines")
    print("  - train_model(): Train a single model")
    print("  - train_all_models(): Train all models")
    print("  - save_models(): Save trained models to disk")
    print("  - evaluate_model_cv(): Evaluate model using cross-validation")
    print("  - get_feature_importance(): Extract feature importance")
    print("  - visualize_model_results(): Visualize model results")
    print("  - visualize_feature_importance(): Visualize feature importance")
    print("  - visualize_cv_results(): Visualize cross-validation results")
    print("  - summarize_cv_results(): Summarize cross-validation results")
    print("  - save_model_results_to_csv(): Save model results to CSV files")
    print("\nImport this module in your scripts to use these functions.")
