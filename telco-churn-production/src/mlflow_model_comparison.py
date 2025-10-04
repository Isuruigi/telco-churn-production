import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

class ModelComparison:
    """Uses MLflow data to compare experiment runs and manage model lifecycle."""

    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initializes the ModelComparison class.

        Args:
            tracking_uri (str): The MLflow tracking server URI.
            experiment_name (str): The name of the experiment to compare runs from.
        """
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment = self.client.get_experiment_by_name(experiment_name)
        if not self.experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found.")
        self.experiment_id = self.experiment.experiment_id

    def compare_experiment_runs(self, metric_to_optimize: str, ascending: bool = False) -> pd.DataFrame:
        """Fetches all runs from the experiment and compares them based on a metric."""
        print(f"--- Comparing runs in experiment: '{self.experiment_name}' ---")
        runs_df = mlflow.search_runs(experiment_ids=[self.experiment_id], order_by=[f"metrics.{metric_to_optimize} {'ASC' if ascending else 'DESC'}"])
        
        # Filter to relevant columns for clarity
        param_cols = [col for col in runs_df.columns if col.startswith('params.')]
        metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]
        
        return runs_df[["run_id", "status"] + metric_cols + param_cols]

    def generate_comparison_report(self, runs_df: pd.DataFrame, top_n: int = 5):
        """Generates and prints a comparison report of the top N runs."""
        print("\n--- Model Comparison Report (Top 5 Runs) ---")
        print(runs_df.head(top_n).to_markdown(index=False))

    def select_best_model_version(self, model_registry_name: str, metric_to_optimize: str, ascending: bool = False):
        """Selects the best model from the experiment and finds its version in the registry."""
        print(f"\n--- Selecting best model version from '{model_registry_name}' ---")
        runs_df = self.compare_experiment_runs(metric_to_optimize, ascending)
        if runs_df.empty:
            print("No runs found to select from.")
            return None, None

        best_run_id = runs_df.iloc[0]["run_id"]
        print(f"Best run found with ID: {best_run_id}")

        # Find the registered model version associated with this run
        versions = self.client.search_model_versions(f"run_id='{best_run_id}'")
        if not versions:
            print(f"No registered model version found for run ID: {best_run_id}")
            return None, None
        
        best_version = versions[0] # Assuming the first one is the one we want
        print(f"Found corresponding model version: {best_version.version}")
        return best_version, best_run_id

    def promote_model_to_production(self, model_name: str, version: str, archive_existing: bool = True):
        """Promotes a specific model version to the 'Production' stage."""
        print(f"\n--- Promoting model '{model_name}' version {version} to Production ---")
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=archive_existing
            )
            print("Model successfully promoted to Production.")
        except Exception as e:
            print(f"Error during model promotion: {e}")

if __name__ == '__main__':
    # This demo assumes you have already run the `sklearn_mlflow_pipeline.py` 
    # script multiple times to generate some runs and registered models.

    # --- MLflow Configuration ---
    TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT = "SklearnProductionPipelineDemo" # Use the same experiment name
    MODEL_REGISTRY = "ProductionChurnModel"
    METRIC = "metrics.accuracy" # The metric to optimize for

    try:
        # 1. Initialize the comparison tool
        model_comp = ModelComparison(tracking_uri=TRACKING_URI, experiment_name=EXPERIMENT)

        # 2. Get and report on the best runs
        all_runs_df = model_comp.compare_experiment_runs(METRIC)
        model_comp.generate_comparison_report(all_runs_df)

        # 3. Select the best model version from the registry
        best_model_ver, best_run_id = model_comp.select_best_model_version(MODEL_REGISTRY, METRIC)

        # 4. Promote the best model to Production
        if best_model_ver:
            model_comp.promote_model_to_production(MODEL_REGISTRY, best_model_ver.version)

    except ValueError as e:
        print(f"Error: {e}. Please ensure the experiment exists and has runs.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
