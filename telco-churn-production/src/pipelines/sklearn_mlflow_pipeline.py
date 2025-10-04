"""
Telco Churn Prediction - MLflow Tracking Pipeline
This module implements training with MLflow experiment tracking for both RandomForest and XGBoost models.
"""

import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier


class MLflowTrainingPipeline:
    """
    Complete training pipeline with MLflow experiment tracking.
    Supports training RandomForest and XGBoost models with comprehensive logging.
    """

    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize the MLflow training pipeline.

        Args:
            config_path (str): Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_mlflow()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None

    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_mlflow(self):
        """
        Configure MLflow tracking URI and experiment name.
        Sets tracking URI to local directory './mlruns' and experiment name to 'telco-churn-prediction'.
        """
        # Set tracking URI to local directory
        tracking_uri = "./mlruns"
        mlflow.set_tracking_uri(tracking_uri)

        # Set experiment name
        experiment_name = self.config['mlflow']['experiment_name']
        mlflow.set_experiment(experiment_name)

        print(f"MLflow Configuration:")
        print(f"  Tracking URI: {tracking_uri}")
        print(f"  Experiment: {experiment_name}")

    def load_and_prepare_data(self):
        """Load and preprocess data, then split into train/test sets."""
        # Load data
        data_path = self.config['data']['raw']
        print(f"\nLoading data from: {data_path}")
        df = pd.read_csv(data_path)

        # Preprocess TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna(subset=['TotalCharges'])

        # Drop customerID if exists
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)

        # Separate features and target
        target_column = self.config['target']
        X = df.drop(target_column, axis=1)
        y = df[target_column].map({'Yes': 1, 'No': 0})

        # Split data
        test_size = self.config['training']['test_size']
        random_state = self.config['training']['random_state']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"Dataset size: {len(df)}")
        print(f"Training set: {len(self.X_train)}")
        print(f"Test set: {len(self.X_test)}")
        print(f"Overall churn rate: {y.mean():.2%}")

    def create_preprocessor(self):
        """Create preprocessing pipeline with StandardScaler and OneHotEncoder."""
        numerical_features = self.config['features']['numerical']
        categorical_features = self.config['features']['categorical']

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )

        return self.preprocessor

    def create_model(self, model_name):
        """
        Create model based on model_name.

        Args:
            model_name (str): Either 'RandomForest' or 'XGBoost'

        Returns:
            Classifier instance
        """
        if model_name == 'RandomForest':
            params = self.config['models']['random_forest']
            return RandomForestClassifier(**params)
        elif model_name == 'XGBoost':
            params = self.config['models']['xgboost']
            return XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Generate and save confusion matrix plot.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model for the plot title

        Returns:
            str: Path to saved plot
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save to reports/figures/
        output_dir = 'reports/figures'
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'confusion_matrix_{model_name.lower()}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def train_model(self, model_name):
        """
        Train a model with MLflow tracking.

        Args:
            model_name (str): Either 'RandomForest' or 'XGBoost'
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name} with MLflow Tracking")
        print(f"{'='*60}")

        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_training"):

            # Log dataset parameters
            mlflow.log_param("dataset_size", len(self.X_train) + len(self.X_test))
            mlflow.log_param("train_size", len(self.X_train))
            mlflow.log_param("test_size", len(self.X_test))
            mlflow.log_param("churn_rate", float(self.y_train.mean()))
            mlflow.log_param("model_type", model_name)

            # Create and log model hyperparameters
            model = self.create_model(model_name)
            model_params = model.get_params()

            # Log all hyperparameters
            for param_name, param_value in model_params.items():
                mlflow.log_param(f"model_{param_name}", param_value)

            # Create full pipeline
            if self.preprocessor is None:
                self.create_preprocessor()

            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])

            # Train model
            print(f"\nTraining {model_name}...")
            pipeline.fit(self.X_train, self.y_train)
            print("Training completed!")

            # Make predictions
            y_pred = pipeline.predict(self.X_test)
            y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]

            # Calculate metrics
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            # Log metrics
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            print(f"\nMetrics:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")

            # Generate and log confusion matrix
            plot_path = self.plot_confusion_matrix(self.y_test, y_pred, model_name)
            mlflow.log_artifact(plot_path)
            print(f"\nConfusion matrix saved and logged: {plot_path}")

            # Log the trained model
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=f"telco_churn_{model_name.lower()}"
            )
            print(f"\nModel logged to MLflow")

            # Print classification report
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred,
                                      target_names=['No Churn', 'Churn']))

            print(f"\n{'='*60}")
            print(f"{model_name} Training Complete!")
            print(f"{'='*60}")

            return {
                'model_name': model_name,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }

    def run_experiments(self):
        """Run training experiments for both RandomForest and XGBoost."""
        print("="*60)
        print("TELCO CHURN PREDICTION - MLFLOW TRAINING PIPELINE")
        print("="*60)

        # Load and prepare data
        self.load_and_prepare_data()

        # Create preprocessor (shared by both models)
        self.create_preprocessor()

        # Train both models
        results = []

        # Train RandomForest
        rf_results = self.train_model('RandomForest')
        results.append(rf_results)

        # Train XGBoost
        xgb_results = self.train_model('XGBoost')
        results.append(xgb_results)

        # Compare results
        print("\n" + "="*60)
        print("EXPERIMENT COMPARISON")
        print("="*60)

        comparison_df = pd.DataFrame(results)
        print("\n", comparison_df.to_string(index=False))

        best_model = comparison_df.loc[comparison_df['roc_auc'].idxmax(), 'model_name']
        best_auc = comparison_df['roc_auc'].max()

        print(f"\n🏆 Best Model: {best_model} (ROC AUC: {best_auc:.4f})")

        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*60)
        print(f"\nView results in MLflow UI:")
        print(f"  cd telco-churn-production")
        print(f"  mlflow ui")
        print(f"  Open: http://localhost:5000")

        return comparison_df


def main():
    """Main execution function."""
    # Initialize MLflow training pipeline
    mlflow_pipeline = MLflowTrainingPipeline(config_path='config/config.yaml')

    # Run experiments for both models
    results = mlflow_pipeline.run_experiments()


if __name__ == "__main__":
    main()
