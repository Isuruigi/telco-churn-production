import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

from src.sklearn_production_pipeline import ChurnPipelineProduction
from src.mlflow_manager import MLflowManager
from src.data_loader import TelcoDataLoader
from src.preprocessor import DataPreprocessor
from src.config.config import TARGET_COLUMN, RANDOM_STATE

class SklearnMlflowPipeline(ChurnPipelineProduction):
    """Extends the production pipeline to integrate with MLflow tracking."""

    def __init__(self, model, tracking_uri: str, experiment_name: str, model_registry_name: str, **kwargs):
        """
        Initializes the MLflow-integrated pipeline.

        Args:
            model: A scikit-learn compatible model instance.
            tracking_uri (str): The MLflow tracking server URI.
            experiment_name (str): The name of the MLflow experiment.
            model_registry_name (str): The name to use for the model in the registry.
        """
        super().__init__(model, **kwargs)
        self.mlflow_manager = MLflowManager(tracking_uri, experiment_name)
        self.model_registry_name = model_registry_name

    def train_with_validation(self, X_train, y_train, X_val=None, y_val=None, fit_params=None):
        """Overrides the parent method to wrap training in an MLflow run."""
        try:
            self.mlflow_manager.start_run(run_name=f"{self.model.__class__.__name__}_training")
            
            # Log model parameters
            self.mlflow_manager.log_hyperparameters(self.model.get_params())
            
            # Perform training
            super().train_with_validation(X_train, y_train, X_val, y_val, fit_params)
            
            # Create model signature
            signature = infer_signature(X_train, self.pipeline.predict(X_train))
            
            # Log the entire pipeline as a model artifact
            self.mlflow_manager.log_model_artifacts(
                self.pipeline, 
                artifact_path="churn_model_pipeline",
                registered_model_name=self.model_registry_name
            )
            
            print("Training run successfully logged to MLflow.")

        except Exception as e:
            print(f"An error occurred during MLflow run: {e}")
        finally:
            # Ensure the run is always ended
            self.mlflow_manager.end_run()

    def evaluate_model(self, X_test, y_test) -> dict:
        """Overrides the parent method to log evaluation metrics to MLflow."""
        metrics = super().evaluate_model(X_test, y_test)
        
        # Log metrics to the active MLflow run (if one exists)
        if self.mlflow_manager.active_run:
            self.mlflow_manager.log_evaluation_metrics(metrics['weighted avg'])
            self.mlflow_manager.log_evaluation_metrics({'accuracy': metrics['accuracy']})
            print("Evaluation metrics successfully logged to MLflow.")
        
        return metrics

    def transition_model_stage(self, version: int, stage: str, archive_existing: bool = True):
        """Transitions a model version to a new stage."""
        print(f"Transitioning version {version} of model '{self.model_registry_name}' to stage '{stage}'...")
        self.mlflow_manager.client.transition_model_version_stage(
            name=self.model_registry_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
        print("Transition complete.")

if __name__ == '__main__':
    # --- MLflow Configuration ---
    TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT = "SklearnProductionPipelineDemo"
    MODEL_REGISTRY = "ProductionChurnModel"

    # 1. Load and prepare data
    df = TelcoDataLoader().load_raw_data()
    df_processed = DataPreprocessor.preprocess_data(df)
    df_processed.dropna(inplace=True)

    X = df_processed.drop(TARGET_COLUMN, axis=1)
    y = df_processed[TARGET_COLUMN].apply(lambda x: 1 if x == 'Yes' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # 2. Initialize the MLflow-integrated pipeline
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, max_depth=10)
    mlflow_pipeline = SklearnMlflowPipeline(
        model=rf_model,
        tracking_uri=TRACKING_URI,
        experiment_name=EXPERIMENT,
        model_registry_name=MODEL_REGISTRY
    )

    # 3. Train and evaluate the pipeline (this will be one complete MLflow run)
    mlflow_pipeline.train_with_validation(X_train, y_train)
    # Note: In this combined method, evaluation should happen inside the run.
    # For a cleaner flow, the train method could also take test data for evaluation.
    
    # Let's start a new run for evaluation for clarity in the demo
    with mlflow.start_run(experiment_id=mlflow_pipeline.mlflow_manager.experiment_id, run_name="Evaluation_Run") as run:
        # Load the trained pipeline from the previous run's artifacts
        # In a real scenario, you'd use the run_id from the training step.
        # For this demo, we'll just use the pipeline in memory.
        evaluation_metrics = mlflow_pipeline.evaluate_model(X_test, y_test)

    print("\n--- Latest Model Version in Registry ---")
    client = MlflowClient(tracking_uri=TRACKING_URI)
    latest_versions = client.get_latest_versions(MODEL_REGISTRY, stages=["None"])
    if latest_versions:
        latest_version = latest_versions[0]
        print(f"Version: {latest_version.version}, Stage: {latest_version.current_stage}")

        # 4. Transition the model to "Staging"
        mlflow_pipeline.transition_model_stage(version=latest_version.version, stage="Staging")
