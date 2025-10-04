import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import roc_curve, auc, confusion_matrix
from src.config.config import REPORTS_PATH
from src.utils import create_directories

FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures', 'visualization')
create_directories([FIGURES_PATH])

def plot_distribution(df: pd.DataFrame, column: str, is_numerical: bool):
    """Plots and saves the distribution of a single feature."""
    plt.figure(figsize=(10, 6))
    if is_numerical:
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
    else:
        sns.countplot(y=column, data=df, order=df[column].value_counts().index)
        plt.title(f'Distribution of {column}')
    
    save_path = os.path.join(FIGURES_PATH, f'{column}_distribution.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved distribution plot for {column} to {save_path}")

def plot_correlation_heatmap(df: pd.DataFrame, numerical_features: list):
    """Plots and saves a correlation heatmap for numerical features."""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    
    save_path = os.path.join(FIGURES_PATH, 'correlation_heatmap.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved correlation heatmap to {save_path}")

def plot_class_distribution(df: pd.DataFrame, target_column: str):
    """Plots and saves the distribution of the target class."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df)
    plt.title('Class Distribution')
    
    save_path = os.path.join(FIGURES_PATH, 'class_distribution.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved class distribution plot to {save_path}")

def plot_feature_importance(feature_importances: pd.DataFrame):
    """Plots and saves feature importances."""
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importances.sort_values(by='importance', ascending=False))
    plt.title('Feature Importance')
    
    save_path = os.path.join(FIGURES_PATH, 'feature_importance.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved feature importance plot to {save_path}")

def plot_roc_curves(y_true, y_preds: dict):
    """Plots and saves ROC curves for multiple models."""
    plt.figure(figsize=(10, 8))
    for model_name, y_pred_prob in y_preds.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    save_path = os.path.join(FIGURES_PATH, 'roc_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC curves plot to {save_path}")

def plot_confusion_matrices(y_true, y_preds: dict):
    """Plots and saves confusion matrices for multiple models."""
    num_models = len(y_preds)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes] # Make it iterable
        
    for ax, (model_name, y_pred_class) in zip(axes, y_preds.items()):
        cm = confusion_matrix(y_true, y_pred_class)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix: {model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
    save_path = os.path.join(FIGURES_PATH, 'confusion_matrices.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrices plot to {save_path}")

def create_interactive_plots(df: pd.DataFrame):
    """Creates and saves an interactive scatter plot using Plotly."""
    fig = px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn",
                     title="Interactive Scatter Plot of Tenure vs. Monthly Charges")
    
    save_path = os.path.join(FIGURES_PATH, 'interactive_scatter.html')
    fig.write_html(save_path)
    print(f"Saved interactive scatter plot to {save_path}")

if __name__ == '__main__':
    # Create dummy data for demonstration
    data = {
        'tenure': [10, 20, 5, 40, 60],
        'MonthlyCharges': [50, 70, 40, 90, 100],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Two year'],
        'Churn': ['No', 'Yes', 'No', 'No', 'Yes']
    }
    dummy_df = pd.DataFrame(data)

    # Dummy model outputs
    y_true_dummy = [0, 1, 0, 0, 1]
    y_preds_prob_dummy = {'Model A': [0.1, 0.8, 0.2, 0.3, 0.9]}
    y_preds_class_dummy = {'Model A': [0, 1, 0, 0, 1]}
    feature_importances_dummy = pd.DataFrame({
        'feature': ['tenure', 'MonthlyCharges', 'Contract'],
        'importance': [0.5, 0.3, 0.2]
    })

    # --- Function Calls ---
    plot_distribution(dummy_df, 'tenure', is_numerical=True)
    plot_distribution(dummy_df, 'Contract', is_numerical=False)
    plot_correlation_heatmap(dummy_df, ['tenure', 'MonthlyCharges'])
    plot_class_distribution(dummy_df, 'Churn')
    plot_feature_importance(feature_importances_dummy)
    plot_roc_curves(y_true_dummy, y_preds_prob_dummy)
    plot_confusion_matrices(y_true_dummy, y_preds_class_dummy)
    create_interactive_plots(dummy_df)
