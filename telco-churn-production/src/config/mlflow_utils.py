import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

class MLflowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()
        try:
            self.experiment_id = self.client.create_experiment(self.experiment_name)
        except MlflowException:
            self.experiment_id = self.client.get_experiment_by_name(self.experiment_name).experiment_id
        self.run = None

    def start_experiment(self, run_name: str = None):
        """Starts a new MLflow run."""
        self.run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        return self.run

    def end_experiment(self):
        """Ends the current MLflow run."""
        mlflow.end_run()
        self.run = None

    def log_parameters(self, params: dict):
        """Logs a dictionary of parameters."""
        if self.run:
            mlflow.log_params(params)
        else:
            print("Error: No active run. Please start an experiment first.")

    def log_metrics(self, metrics: dict):
        """Logs a dictionary of metrics."""
        if self.run:
            mlflow.log_metrics(metrics)
        else:
            print("Error: No active run. Please start an experiment first.")

    def log_artifacts(self, artifact_path: str):
        """Logs an artifact (file or directory)."""
        if self.run:
            mlflow.log_artifacts(artifact_path)
        else:
            print("Error: No active run. Please start an experiment first.")

    def log_model(self, model, artifact_path: str):
        """Logs a model."""
        if self.run:
            mlflow.sklearn.log_model(model, artifact_path)
        else:
            print("Error: No active run. Please start an experiment first.
")
    def register_model(self, model_uri: str, name: str):
        """Registers a model in the MLflow Model Registry."""
        try:
            return mlflow.register_model(model_uri, name)
        except MlflowException as e:
            print(f"Error registering model: {e}")
            return None

    def get_best_model_from_registry(self, name: str, stage: str = 'None'):
        """Gets the latest version of a model for a given stage."""
        try:
            versions = self.client.get_latest_versions(name, stages=[stage])
            if not versions:
                print(f"No models found for stage '{stage}' in registry for model '{name}'.")
                return None
            return versions[0]
        except MlflowException as e:
            print(f"Error getting model from registry: {e}")
            return None

    def load_model_from_registry(self, name: str, stage: str = 'Staging'):
        """Loads a model from the MLflow Model Registry."""
        try:
            return mlflow.sklearn.load_model(model_uri=f"models:/{name}/{stage}")
        except MlflowException as e:
            print(f"Error loading model from registry: {e}")
            return None
