"""
Exploratory Data Analysis (EDA) Module
This module provides comprehensive EDA functions including:
- Statistical analysis by default status
- Feature distribution analysis
- Correlation analysis
- Outlier detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Optional, Dict, List
import config

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def get_dataset_overview(df: pd.DataFrame, target_column: str = 'default') -> dict:
    """
    Get overview of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_column : str
        Name of the target column
    
    Returns:
    --------
    dict
        Dataset overview statistics
    """
    overview = {
        'total_records': len(df),
        'total_features': len(df.columns) - 1,
        'default_count': df[target_column].sum(),
        'non_default_count': (df[target_column] == 0).sum(),
        'default_rate': df[target_column].mean() * 100,
        'non_default_rate': (1 - df[target_column].mean()) * 100
    }
    return overview


def analyze_numerical_features_by_target(
    df: pd.DataFrame,
    target_column: str = 'default',
    numeric_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze numerical features by target status.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_column : str
        Name of the target column
    numeric_features : list, optional
        List of numerical features to analyze
    
    Returns:
    --------
    pd.DataFrame
        Statistical summary by target status
    """
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [f for f in numeric_features if f != target_column]
    
    summary_stats = []
    
    for feature in numeric_features:
        if feature not in df.columns:
            continue
        
        non_default = df[df[target_column] == 0][feature]
        default = df[df[target_column] == 1][feature]
        
        stats = {
            'Feature': feature,
            'Non-Default_Mean': non_default.mean(),
            'Default_Mean': default.mean(),
            'Non-Default_Median': non_default.median(),
            'Default_Median': default.median(),
            'Non-Default_Std': non_default.std(),
            'Default_Std': default.std(),
            'Effect_Size': (default.mean() - non_default.mean()) / non_default.std() 
                          if non_default.std() > 0 else 0
        }
        summary_stats.append(stats)
    
    stats_df = pd.DataFrame(summary_stats)
    stats_df = stats_df.sort_values('Effect_Size', key=abs, ascending=False)
    
    return stats_df


def analyze_categorical_features_by_target(
    df: pd.DataFrame,
    target_column: str = 'default',
    categorical_features: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Analyze categorical features by target status.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_column : str
        Name of the target column
    categorical_features : list, optional
        List of categorical features to analyze
    
    Returns:
    --------
    dict
        Dictionary with crosstab for each categorical feature
    """
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    crosstabs = {}
    
    for feature in categorical_features:
        if feature not in df.columns:
            continue
        
        crosstab = pd.crosstab(df[feature], df[target_column], margins=True)
        crosstab['Default_Rate'] = (crosstab[1] / crosstab['All'] * 100).round(2)
        crosstabs[feature] = crosstab
    
    return crosstabs


def calculate_correlation_matrix(
    df: pd.DataFrame,
    target_column: str = 'default',
    numeric_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate correlation matrix for numerical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_column : str
        Name of the target column
    numeric_features : list, optional
        List of numerical features to include
    
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Include target in correlation
    features_to_correlate = [f for f in numeric_features if f in df.columns]
    if target_column in df.columns:
        features_to_correlate.append(target_column)
    
    corr_matrix = df[features_to_correlate].corr()
    
    return corr_matrix


def detect_outliers_iqr(
    df: pd.DataFrame,
    feature: str,
    target_column: str = 'default'
) -> Dict[str, List[int]]:
    """
    Detect outliers using IQR method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    feature : str
        Feature name to analyze
    target_column : str
        Name of the target column
    
    Returns:
    --------
    dict
        Dictionary with outlier indices by target status
    """
    outliers = {
        'non_default': [],
        'default': []
    }
    
    for target_value, label in [(0, 'non_default'), (1, 'default')]:
        data = df[df[target_column] == target_value][feature].dropna()
        
        if len(data) == 0:
            continue
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = data[outlier_mask].index.tolist()
        outliers[label] = outlier_indices
    
    return outliers


def get_eda_summary(
    df: pd.DataFrame,
    target_column: str = 'default',
    config_module=None
) -> dict:
    """
    Get comprehensive EDA summary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_column : str
        Name of the target column
    config_module : module
        Configuration module (default: config)
    
    Returns:
    --------
    dict
        Comprehensive EDA summary
    """
    if config_module is None:
        config_module = config
    
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    # Dataset overview
    overview = get_dataset_overview(df, target_column)
    print(f"\nDataset Overview:")
    print(f"  - Total records: {overview['total_records']}")
    print(f"  - Total features: {overview['total_features']}")
    print(f"  - Default rate: {overview['default_rate']:.2f}%")
    print(f"  - Non-default rate: {overview['non_default_rate']:.2f}%")
    
    # Get all numerical features (including engineered ones)
    all_numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in all_numeric_features:
        all_numeric_features.remove(target_column)
    
    # Numerical features analysis - use all numeric features
    numeric_stats = analyze_numerical_features_by_target(df, target_column, all_numeric_features)
    
    # Categorical features analysis
    categorical_features = (
        config_module.FEATURE_LISTS.get('ordinal_features', []) +
        config_module.FEATURE_LISTS.get('low_cardinality_nominal', []) +
        config_module.FEATURE_LISTS.get('high_cardinality_nominal', [])
    )
    categorical_stats = analyze_categorical_features_by_target(df, target_column, categorical_features)
    
    # Correlation matrix - use all numeric features
    corr_matrix = calculate_correlation_matrix(df, target_column, all_numeric_features)
    
    eda_summary = {
        'overview': overview,
        'numeric_stats': numeric_stats,
        'categorical_stats': categorical_stats,
        'correlation_matrix': corr_matrix,
        'all_numeric_features': all_numeric_features
    }
    
    print("\n✓ EDA summary complete!")
    
    return eda_summary


def visualize_feature_distributions_bar(
    df: pd.DataFrame,
    target_column: str = 'default',
    save_path: Optional[str] = None
) -> None:
    """
    Visualize feature distributions using bar charts comparing default vs non-default.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features and target
    target_column : str
        Name of the target column
    save_path : str, optional
        Path to save plots
    """
    import os
    
    if save_path is None:
        save_path = 'plots/'
    
    os.makedirs(save_path, exist_ok=True)
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    n_cols = 4
    n_rows = (len(numeric_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes_list = [axes]
    elif n_rows == 1:
        axes_list = list(axes) if hasattr(axes, '__iter__') else [axes]
    else:
        axes_list = axes.flatten().tolist() if hasattr(axes, 'flatten') else list(axes.flatten())
    
    for idx, feature in enumerate(numeric_features):
        if idx >= len(axes_list):
            break
        
        ax = axes_list[idx]
        
        non_default_data = df[df[target_column] == 0][feature].dropna()
        default_data = df[df[target_column] == 1][feature].dropna()
        
        bins = np.linspace(
            min(non_default_data.min(), default_data.min()),
            max(non_default_data.max(), default_data.max()),
            20
        )
        
        non_default_counts, _ = np.histogram(non_default_data, bins=bins)
        default_counts, _ = np.histogram(default_data, bins=bins)
        
        x = np.arange(len(bins)-1)
        width = 0.35
        
        ax.bar(x - width/2, non_default_counts, width, label='Non-Default', 
               color='green', alpha=0.7)
        ax.bar(x + width/2, default_counts, width, label='Default', 
               color='red', alpha=0.7)
        
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')
        ax.set_xticks([])
    
    for idx in range(len(numeric_features), len(axes_list)):
        axes_list[idx].axis('off')
    
    plt.suptitle('Feature Distributions by Default Status (Bar Charts)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'all_features_bar_charts.png'), 
                    dpi=300, bbox_inches='tight')
    plt.show()


def visualize_correlation_heatmap(
    df: pd.DataFrame,
    target_column: str = 'default',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Visualize correlation heatmap for all features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features and target
    target_column : str
        Name of the target column
    save_path : str, optional
        Path to save plots
    
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    import os
    
    if save_path is None:
        save_path = 'plots/'
    
    os.makedirs(save_path, exist_ok=True)
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    corr_matrix = df[numeric_features + [target_column]].corr()
    
    plt.figure(figsize=(max(14, len(numeric_features)*0.5), max(12, len(numeric_features)*0.5)))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=True, yticklabels=True, annot_kws={"size": 8})
    plt.title('Correlation Heatmap - All Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'all_features_correlation_heatmap.png'), 
                    dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 80)
    print("CORRELATION WITH DEFAULT (sorted by absolute value)")
    print("=" * 80)
    default_corr = corr_matrix[target_column].drop(target_column).sort_values(key=abs, ascending=False)
    print(default_corr.to_string())
    
    return corr_matrix


def visualize_feature_distributions_violin(
    df: pd.DataFrame,
    target_column: str = 'default',
    save_path: Optional[str] = None
) -> None:
    """
    Visualize feature distributions using violin plots.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features and target
    target_column : str
        Name of the target column
    save_path : str, optional
        Path to save plots
    """
    import os
    
    if save_path is None:
        save_path = 'plots/'
    
    os.makedirs(save_path, exist_ok=True)
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    n_cols = 3
    n_rows = (len(numeric_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes_list = [axes]
    elif n_rows == 1:
        axes_list = list(axes) if hasattr(axes, '__iter__') else [axes]
    else:
        axes_list = axes.flatten().tolist() if hasattr(axes, 'flatten') else list(axes.flatten())
    
    for idx, feature in enumerate(numeric_features):
        if idx >= len(axes_list):
            break
        
        ax = axes_list[idx]
        
        non_default_data = df[df[target_column] == 0][feature].dropna()
        default_data = df[df[target_column] == 1][feature].dropna()
        
        parts = ax.violinplot(
            [non_default_data, default_data],
            positions=[0, 1],
            showmeans=True,
            showmedians=True
        )
        
        for pc, color in zip(parts['bodies'], ['lightgreen', 'lightcoral']):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-Default', 'Default'])
        ax.set_ylabel(feature, fontsize=9)
        ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
    
    for idx in range(len(numeric_features), len(axes_list)):
        axes_list[idx].axis('off')
    
    plt.suptitle('Violin Plots: All Feature Distributions by Default Status', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'all_features_violin_plots.png'), 
                    dpi=300, bbox_inches='tight')
    plt.show()


def create_all_feature_visualizations(
    df: pd.DataFrame,
    target_column: str = 'default',
    save_path: Optional[str] = None
) -> None:
    """
    Create all feature visualizations: bar charts, heatmap, and violin plots.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features and target
    target_column : str
        Name of the target column
    save_path : str, optional
        Path to save plots
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    print(f"Visualizing {len(numeric_features)} features...")
    
    visualize_feature_distributions_bar(df, target_column, save_path)
    visualize_correlation_heatmap(df, target_column, save_path)
    visualize_feature_distributions_violin(df, target_column, save_path)
    
    print(f"\n✓ Visualized {len(numeric_features)} features with bar charts, heatmap, and violin plots")


if __name__ == "__main__":
    # Example usage
    print("Exploratory Data Analysis (EDA) Module")
    print("=" * 80)
    print("\nThis module provides functions for EDA:")
    print("  - get_dataset_overview(): Get dataset overview")
    print("  - analyze_numerical_features_by_target(): Analyze numerical features")
    print("  - analyze_categorical_features_by_target(): Analyze categorical features")
    print("  - calculate_correlation_matrix(): Calculate correlation matrix")
    print("  - detect_outliers_iqr(): Detect outliers using IQR method")
    print("  - get_eda_summary(): Get comprehensive EDA summary")
    print("\nImport this module in your scripts to use these functions.")

