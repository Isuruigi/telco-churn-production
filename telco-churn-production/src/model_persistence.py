import joblib
import json
import os
from datetime import datetime
from src.config.config import MODELS_PATH
from src.utils import create_directories

class ModelPersistence:
    """Handles the saving and loading of model and pipeline artifacts."""

    def __init__(self, model_dir=MODELS_PATH):
        """
        Initializes the ModelPersistence class.

        Args:
            model_dir (str): The directory to save and load artifacts from.
        """
        self.model_dir = model_dir
        create_directories([self.model_dir])

    def save_model_with_metadata(self, model, model_name: str, metadata: dict):
        """
        Saves a model and its associated metadata.

        Args:
            model: The trained model object.
            model_name (str): The name for the model file (e.g., 'random_forest_v1').
            metadata (dict): A dictionary of metadata to save (e.g., metrics, params).
        """
        # Add timestamp to metadata
        metadata['save_timestamp'] = datetime.now().isoformat()

        # Define paths
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")

        # Save model
        try:
            joblib.dump(model, model_path)
            print(f"Model saved successfully to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            return

        # Save metadata
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadata saved successfully to {metadata_path}")
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def load_model_with_metadata(self, model_name: str) -> tuple:
        """Loads a model and its metadata."""
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")

        # Load model
        model = self._load_artifact(model_path)
        # Load metadata
        metadata = self._load_artifact(metadata_path, is_json=True)

        return model, metadata

    def save_preprocessing_pipeline(self, pipeline, pipeline_name: str):
        """Saves a preprocessing pipeline."""
        pipeline_path = os.path.join(self.model_dir, f"{pipeline_name}.joblib")
        self._save_artifact(pipeline, pipeline_path)

    def load_preprocessing_pipeline(self, pipeline_name: str):
        """Loads a preprocessing pipeline."""
        pipeline_path = os.path.join(self.model_dir, f"{pipeline_name}.joblib")
        return self._load_artifact(pipeline_path)

    def _save_artifact(self, artifact, path: str):
        try:
            joblib.dump(artifact, path)
            print(f"Artifact saved successfully to {path}")
        except Exception as e:
            print(f"Error saving artifact to {path}: {e}")

    def _load_artifact(self, path: str, is_json=False):
        if not self.validate_model_artifacts(path):
            return None
        try:
            if is_json:
                with open(path, 'r') as f:
                    artifact = json.load(f)
            else:
                artifact = joblib.load(path)
            print(f"Artifact loaded successfully from {path}")
            return artifact
        except Exception as e:
            print(f"Error loading artifact from {path}: {e}")
            return None

    def validate_model_artifacts(self, artifact_path: str) -> bool:
        """Validates the existence of a single model artifact."""
        if not os.path.exists(artifact_path):
            print(f"Validation failed: Artifact not found at {artifact_path}")
            return False
        print(f"Validation successful: Artifact found at {artifact_path}")
        return True

if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier

    # --- Dummy Artifacts for Demonstration ---
    dummy_model = RandomForestClassifier(n_estimators=5)
    dummy_metadata = {
        'model_type': 'RandomForestClassifier',
        'parameters': {'n_estimators': 5},
        'metrics': {'roc_auc': 0.85, 'accuracy': 0.90}
    }
    MODEL_NAME = "demo_model"

    # --- Initialize Persistence Manager ---
    persistence = ModelPersistence()

    # 1. Save model and metadata
    persistence.save_model_with_metadata(dummy_model, MODEL_NAME, dummy_metadata)

    # 2. Validate artifacts
    print("\n--- Validating Artifacts ---")
    persistence.validate_model_artifacts(os.path.join(persistence.model_dir, f"{MODEL_NAME}.joblib"))
    persistence.validate_model_artifacts(os.path.join(persistence.model_dir, f"{MODEL_NAME}_metadata.json"))

    # 3. Load model and metadata
    print("\n--- Loading Artifacts ---")
    loaded_model, loaded_metadata = persistence.load_model_with_metadata(MODEL_NAME)

    if loaded_model and loaded_metadata:
        print("\n--- Loaded Model and Metadata ---")
        print("Model Type:", type(loaded_model))
        print("Metadata:", loaded_metadata)
