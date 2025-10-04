from src.modeling.sklearn_pipeline import ChurnPipeline
from config.mlflow_utils import MLflowTracker
from config.mlflow_config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODEL_REGISTRY_NAME
import pandas as pd

class ChurnPipelineMLflow(ChurnPipeline):
    def __init__(self, model_type='logistic_regression', random_state=42):
        super().__init__(model_type, random_state)
        self.mlflow_tracker = MLflowTracker(tracking_uri=MLFLOW_TRACKING_URI, experiment_name=EXPERIMENT_NAME)

    def train(self, X: pd.DataFrame, y: pd.Series, cv=5, scoring='accuracy'):
        try:
            self.mlflow_tracker.start_experiment(run_name=f"{self.model_type}_training")
            
            # Log hyperparameters
            self.mlflow_tracker.log_parameters(self.param_grid)
            self.mlflow_tracker.log_parameters({'model_type': self.model_type, 'random_state': self.random_state})

            # Original training process
            super().train(X, y, cv, scoring)

            # Log metrics from cross-validation
            cv_scores = self.metadata.get('cross_validation_scores', {})
            for metric_name, values in cv_scores.items():
                if 'test_' in metric_name:
                    self.mlflow_tracker.log_metrics({f"cv_{metric_name}": pd.Series(values).mean()})

            # Log best parameters from GridSearchCV
            self.mlflow_tracker.log_parameters(self.grid_search.best_params_)

            # Log the model
            model_artifact_path = f"{self.model_type}_model"
            self.mlflow_tracker.log_model(self.pipeline, model_artifact_path)

            # Register the model
            model_uri = f"runs:/{self.mlflow_tracker.run.info.run_id}/{model_artifact_path}"
            self.mlflow_tracker.register_model(model_uri, MODEL_REGISTRY_NAME)

            print("MLflow logging and model registration complete.")

        except Exception as e:
            print(f"An error occurred during MLflow training: {e}")
            raise
        finally:
            self.mlflow_tracker.end_experiment()

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> dict:
        # Original evaluation process
        metrics = super().evaluate(X, y_true)

        # Log evaluation metrics to MLflow
        if self.mlflow_tracker.run: # Check if there is an active run
            self.mlflow_tracker.log_metrics(metrics)
            print("Evaluation metrics logged to MLflow.")
        
        return metrics
