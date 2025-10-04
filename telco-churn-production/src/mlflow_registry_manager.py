from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

class RegistryManager:
    """Manages the complete lifecycle of models in the MLflow Model Registry."""

    def __init__(self, tracking_uri: str):
        """
        Initializes the RegistryManager.

        Args:
            tracking_uri (str): The MLflow tracking server URI.
        """
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def register_model_version(self, model_name: str, source_uri: str, run_id: str = None) -> dict:
        """
        Registers a new version of a model from a specific run.

        Args:
            model_name (str): The name of the model in the registry.
            source_uri (str): The URI of the model artifact (e.g., 'runs:/<run_id>/model').
            run_id (str, optional): The ID of the run to associate with the model version.

        Returns:
            dict: The registered model version details.
        """
        try:
            print(f"Registering model '{model_name}' from source: {source_uri}")
            registered_version = self.client.create_model_version(
                name=model_name,
                source=source_uri,
                run_id=run_id
            )
            print(f"Successfully registered version: {registered_version.version}")
            return registered_version
        except MlflowException as e:
            print(f"Error registering model version: {e}")
            return None

    def transition_model_stage(self, model_name: str, version: str, stage: str, archive_existing: bool = True):
        """Transitions a model version to a specified stage (e.g., 'Staging', 'Production')."""
        try:
            print(f"Transitioning model '{model_name}' version {version} to stage '{stage}'...")
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            print("Transition complete.")
        except MlflowException as e:
            print(f"Error transitioning model stage: {e}")

    def archive_old_versions(self, model_name: str, versions_to_keep: int = 2):
        """Archives old versions of a model, keeping a specified number of recent versions."""
        print(f"Archiving old versions of model '{model_name}', keeping the latest {versions_to_keep}...")
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            # Sort by version number (descending)
            versions.sort(key=lambda v: int(v.version), reverse=True)
            
            if len(versions) > versions_to_keep:
                for old_version in versions[versions_to_keep:]:
                    if old_version.current_stage != 'Archived':
                        self.transition_model_stage(model_name, old_version.version, 'Archived', archive_existing=False)
        except MlflowException as e:
            print(f"Error archiving old versions: {e}")

    def get_production_model(self, model_name: str):
        """Retrieves the model version currently in the 'Production' stage."""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                print(f"No model in 'Production' stage found for '{model_name}'.")
                return None
            
            prod_model = versions[0]
            print(f"Found production model: Version {prod_model.version}")
            return prod_model
        except MlflowException as e:
            print(f"Error getting production model: {e}")
            return None

    def update_model_description(self, model_name: str, description: str):
        """Updates the description of a registered model."""
        try:
            self.client.update_registered_model(name=model_name, description=description)
            print(f"Updated description for registered model '{model_name}'.")
        except MlflowException as e:
            print(f"Error updating model description: {e}")

    def update_model_version_description(self, model_name: str, version: str, description: str):
        """Updates the description of a specific model version."""
        try:
            self.client.update_model_version(name=model_name, version=version, description=description)
            print(f"Updated description for model '{model_name}' version {version}.")
        except MlflowException as e:
            print(f"Error updating model version description: {e}")

if __name__ == '__main__':
    # This is a conceptual demo. It requires an MLflow server and a registered model.
    # Assume a model has been logged in a run with ID 'some_run_id'
    
    TRACKING_URI = "sqlite:///mlflow.db"
    MODEL_NAME = "DemoChurnModelForRegistry"
    RUN_ID = "some_run_id" # Replace with a real run ID from your MLflow server
    SOURCE_URI = f"runs:/{RUN_ID}/model"

    print("--- Initializing Registry Manager ---")
    registry_manager = RegistryManager(tracking_uri=TRACKING_URI)

    # The following lines are for demonstration purposes.
    # You would uncomment and run them with a real RUN_ID.

    # 1. Register a new model version
    # print("\n--- 1. Registering New Model Version ---")
    # new_version = registry_manager.register_model_version(MODEL_NAME, SOURCE_URI, RUN_ID)
    # if new_version:
    #     VERSION = new_version.version

    #     # 2. Update descriptions
    #     print("\n--- 2. Updating Descriptions ---")
    #     registry_manager.update_model_description(MODEL_NAME, "This is a model to predict customer churn.")
    #     registry_manager.update_model_version_description(MODEL_NAME, VERSION, "Initial version trained on Q1 data.")

    #     # 3. Transition to Staging
    #     print("\n--- 3. Transitioning to Staging ---")
    #     registry_manager.transition_model_stage(MODEL_NAME, VERSION, "Staging")

    #     # 4. Get the production model (will be None initially)
    #     print("\n--- 4. Getting Production Model ---")
    #     prod_model = registry_manager.get_production_model(MODEL_NAME)

    #     # 5. Archive old versions (if you run this multiple times)
    #     print("\n--- 5. Archiving Old Versions ---")
    #     registry_manager.archive_old_versions(MODEL_NAME, versions_to_keep=1)

    print("\nRegistry Manager demo complete. Uncomment lines to run with a real MLflow server and run ID.")
