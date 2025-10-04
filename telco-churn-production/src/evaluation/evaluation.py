import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    roc_curve, 
    precision_recall_curve
)
from src.modeling.sklearn_pipeline import ChurnPipeline

def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """Calculates and returns a dictionary of classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plots a confusion matrix.
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names (list, optional): Names of the classes. Defaults to None.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_roc_curve(model: ChurnPipeline, X_test, y_test):
    """Plots the ROC curve for a binary classification model."""
    y_prob = model.pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(model: ChurnPipeline, X_test, y_test):
    """Plots the Precision-Recall curve for a binary classification model."""
    y_prob = model.pipeline.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

def plot_feature_importance(model: ChurnPipeline, top_n=20):
    """Plots the feature importances from the trained model."""
    try:
        feature_importance_df = model.get_feature_importance()
        if feature_importance_df is None:
            print("Feature importance is not available for this model type.")
            return

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(top_n))
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred while plotting feature importance: {e}")
        raise
