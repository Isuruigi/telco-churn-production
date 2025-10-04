import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

class MLflowManager:
    """Provides a comprehensive manager for MLflow experiment tracking."""

    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initializes the MLflowManager.

        Args:
            tracking_uri (str): The MLflow tracking server URI.
            experiment_name (str): The name of the experiment.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment_id = self.setup_experiment()
        self.active_run = None

    def setup_experiment(self) -> str:
        """Creates or gets an MLflow experiment."""
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                print(f"Experiment '{self.experiment_name}' not found. Creating new experiment.")
                return self.client.create_experiment(self.experiment_name)
            return experiment.experiment_id
        except MlflowException as e:
            print(f"Error setting up experiment: {e}")
            return None

    def start_run(self, run_name: str = None) -> mlflow.ActiveRun:
        """Starts a new MLflow run."""
        self.active_run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        print(f"Started MLflow run '{self.active_run.info.run_name}' (ID: {self.active_run.info.run_id})")
        return self.active_run

    def end_run(self):
        """Ends the current active MLflow run."""
        if self.active_run:
            mlflow.end_run()
            print("MLflow run ended.")
            self.active_run = None

    def log_hyperparameters(self, params: dict):
        """Logs a dictionary of hyperparameters."""
        mlflow.log_params(params)

    def log_training_metrics(self, metrics: dict, step: int = None):
        """Logs training metrics, optionally at a specific step."""
        mlflow.log_metrics(metrics, step=step)

    def log_evaluation_metrics(self, metrics: dict):
        """Logs final evaluation metrics."""
        mlflow.log_metrics(metrics)

    def log_model_artifacts(self, model, artifact_path: str, registered_model_name: str = None):
        """Logs a model as an artifact and optionally registers it."""
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        print(f"Model saved as artifact: '{artifact_path}'")
        if registered_model_name:
            print(f"Model registered with name: '{registered_model_name}'")

    def register_best_model(self, metric_to_optimize: str, model_registry_name: str, ascending=False):
        """Finds the best run in the experiment and registers its model."""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_to_optimize} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        if not runs:
            print("No runs found in the experiment.")
            return None
        
        best_run = runs[0]
        best_run_id = best_run.info.run_id
        model_uri = f"runs:/{best_run_id}/model"
        
        print(f"Registering best model from run {best_run_id} to registry as '{model_registry_name}'")
        registered_model = mlflow.register_model(model_uri, model_registry_name)
        return registered_model

    def get_model_from_registry(self, model_name: str, stage: str = "Staging"):
        """Loads a model from the MLflow Model Registry."""
        try:
            model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
            print(f"Loaded model '{model_name}' (stage: {stage}) from registry.")
            return model
        except MlflowException as e:
            print(f"Error loading model from registry: {e}")
            return None

if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    # --- MLflow Configuration ---
    # This assumes you have an MLflow tracking server running.
    # For a local demo, you can run `mlflow ui` in a terminal.
    TRACKING_URI = "sqlite:///mlflow.db" # Using a local file for the demo
    EXPERIMENT = "MLflowManagerDemo"
    MODEL_REGISTRY = "DemoChurnModel"

    # 1. Initialize the manager
    mlflow_manager = MLflowManager(tracking_uri=TRACKING_URI, experiment_name=EXPERIMENT)

    # 2. Start a run
    mlflow_manager.start_run(run_name="RandomForest_TestRun")

    # 3. Prepare data and model
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)

    # 4. Log information to MLflow
    mlflow_manager.log_hyperparameters(params)
    mlflow_manager.log_training_metrics({'training_samples': len(X_train)})
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow_manager.log_evaluation_metrics({'accuracy': accuracy})

    # 5. Log model and register it
    mlflow_manager.log_model_artifacts(rf, artifact_path="model", registered_model_name=MODEL_REGISTRY)

    # 6. End the run
    mlflow_manager.end_run()

    # 7. Load the model back from the registry
    print("\n--- Loading model from registry for inference ---")
    loaded_model = mlflow_manager.get_model_from_registry(model_name=MODEL_REGISTRY, stage="None")
    if loaded_model:
        print("Prediction with loaded model:", loaded_model.predict(X_test[:5]))
