import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from imblearn.metrics import classification_report_imbalanced
from src.visualization import plot_roc_curves, plot_confusion_matrices

class ModelEvaluator:
    """A class for comprehensive evaluation of classification models."""

    def __init__(self, models: dict, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Initializes the ModelEvaluator.

        Args:
            models (dict): A dictionary of trained models, e.g., {'model_name': model_instance}.
            X_test (pd.DataFrame): The test features.
            y_test (pd.Series): The true test labels.
        """
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = {}
        self._make_predictions()

    def _make_predictions(self):
        """Helper method to make predictions for all models."""
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            self.predictions[name] = {'class': y_pred, 'proba': y_pred_proba}

    def calculate_classification_metrics(self, model_name: str) -> dict:
        """Calculates a dictionary of standard classification metrics."""
        if model_name not in self.predictions:
            raise ValueError(f"Model '{model_name}' not found.")
        
        y_pred = self.predictions[model_name]['class']
        y_pred_proba = self.predictions[model_name]['proba']
        
        pr_auc = auc(precision_recall_curve(self.y_test, y_pred_proba, pos_label=1)[1], 
                     precision_recall_curve(self.y_test, y_pred_proba, pos_label=1)[0])

        metrics = {
            'ROC_AUC': roc_auc_score(self.y_test, y_pred_proba),
            'PR_AUC': pr_auc
        }
        report = classification_report(self.y_test, y_pred, output_dict=True)
        metrics.update(report['weighted avg'])
        return metrics

    def evaluate_imbalanced_classification(self, model_name: str):
        """Prints a classification report tailored for imbalanced datasets."""
        if model_name not in self.predictions:
            raise ValueError(f"Model '{model_name}' not found.")
        
        print(f"--- Imbalanced Classification Report for {model_name} ---")
        y_pred = self.predictions[model_name]['class']
        print(classification_report_imbalanced(self.y_test, y_pred))
        print("--- End Report ---\n")

    def generate_classification_report(self, model_name: str):
        """Prints a standard classification report."""
        if model_name not in self.predictions:
            raise ValueError(f"Model '{model_name}' not found.")
        
        print(f"--- Classification Report for {model_name} ---")
        y_pred = self.predictions[model_name]['class']
        print(classification_report(self.y_test, y_pred))
        print("--- End Report ---\n")

    def create_evaluation_plots(self):
        """Creates and saves evaluation plots (ROC and Confusion Matrix) for all models."""
        print("--- Creating Evaluation Plots ---")
        # ROC Curves
        y_preds_proba = {name: preds['proba'] for name, preds in self.predictions.items()}
        plot_roc_curves(self.y_test, y_preds_proba)

        # Confusion Matrices
        y_preds_class = {name: preds['class'] for name, preds in self.predictions.items()}
        plot_confusion_matrices(self.y_test, y_preds_class)
        print("--- Plots Saved ---\n")

    def compare_multiple_models(self) -> pd.DataFrame:
        """Compares performance across multiple models and returns a summary DataFrame."""
        results = []
        for name in self.models.keys():
            metrics = self.calculate_classification_metrics(name)
            metrics['model'] = name
            results.append(metrics)
        
        results_df = pd.DataFrame(results).set_index('model')
        return results_df

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    # Create dummy data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, weights=[0.9, 0.1], random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train dummy models
    lr = LogisticRegression(random_state=42).fit(X_train, y_train)
    rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    models_dict = {'Logistic Regression': lr, 'Random Forest': rf}

    # --- Initialize and use the evaluator ---
    evaluator = ModelEvaluator(models_dict, X_test, y_test)

    # 1. Compare all models
    comparison_df = evaluator.compare_multiple_models()
    print("--- Model Comparison ---")
    print(comparison_df)
    print("--- End Comparison ---\n")

    # 2. Get detailed reports for a single model
    evaluator.generate_classification_report('Random Forest')
    evaluator.evaluate_imbalanced_classification('Random Forest')

    # 3. Create and save all evaluation plots
    evaluator.create_evaluation_plots()
