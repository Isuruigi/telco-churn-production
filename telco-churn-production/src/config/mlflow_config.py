import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# --- MLflow Constants ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "telco-churn-prediction"
MODEL_REGISTRY_NAME = "telco-churn-models"

# --- MLflow Functions ---

def setup_mlflow_tracking(tracking_uri: str = MLFLOW_TRACKING_URI):
    """Sets up the MLflow tracking URI."""
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")

def create_experiment(experiment_name: str = EXPERIMENT_NAME) -> str:
    """
    Creates an MLflow experiment if it does not already exist.

    Returns:
        str: The ID of the experiment.
    """
    client = MlflowClient()
    try:
        experiment_id = client.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
        return experiment_id
    except MlflowException:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        print(f"Experiment '{experiment_name}' already exists with ID: {experiment_id}")
        return experiment_id

def get_experiment_by_name(experiment_name: str = EXPERIMENT_NAME):
    """
    Gets an experiment by its name.

    Returns:
        mlflow.entities.Experiment: The experiment object, or None if not found.
    """
    client = MlflowClient()
    try:
        return client.get_experiment_by_name(experiment_name)
    except MlflowException:
        print(f"Experiment '{experiment_name}' not found.")
        return None

# Example usage (can be removed or commented out)
if __name__ == "__main__":
    print("--- Setting up MLflow Tracking ---")
    setup_mlflow_tracking()
    
    print("\n--- Creating/Verifying MLflow Experiment ---")
    exp_id = create_experiment()
    
    print("\n--- Getting Experiment Details ---")
    experiment = get_experiment_by_name()
    if experiment:
        print(f"Experiment Name: {experiment.name}")
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Artifact Location: {experiment.artifact_location}")
