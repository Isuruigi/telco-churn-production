import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.modeling.sklearn_pipeline import ChurnPipeline

def load_trained_model(model_path: str) -> ChurnPipeline:
    """Loads a trained model pipeline from the specified path."""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        raise
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise

def validate_input(data: pd.DataFrame, model: ChurnPipeline) -> pd.DataFrame:
    """Validates the input data to ensure it meets the model's requirements."""
    required_features = model.numerical_features + model.categorical_features
    missing_features = [feat for feat in required_features if feat not in data.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Ensure correct data types
    for col in model.numerical_features:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Handle 'TotalCharges' specifically if it's all NaNs after coercion
    if 'TotalCharges' in data.columns and data['TotalCharges'].isnull().all():
         data['TotalCharges'].fillna(0, inplace=True) # Or a more sophisticated imputation

    return data

def predict_single_sample(model: ChurnPipeline, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Makes a prediction for a single sample."""
    try:
        sample_df = pd.DataFrame([sample])
        sample_df = validate_input(sample_df, model)

        classification = model.predict(sample_df)[0]
        probabilities = model.pipeline.predict_proba(sample_df)[0]
        confidence_score = np.max(probabilities)

        return {
            "classification": classification,
            "probabilities": probabilities.tolist(),
            "confidence_score": confidence_score
        }
    except Exception as e:
        print(f"An error occurred during single sample prediction: {e}")
        raise

def predict_batch_samples(model: ChurnPipeline, samples: pd.DataFrame) -> List[Dict[str, Any]]:
    """Makes predictions for a batch of samples."""
    try:
        samples = validate_input(samples, model)

        classifications = model.predict(samples)
        probabilities = model.pipeline.predict_proba(samples)
        confidence_scores = np.max(probabilities, axis=1)

        results = []
        for i in range(len(samples)):
            results.append({
                "classification": classifications[i],
                "probabilities": probabilities[i].tolist(),
                "confidence_score": confidence_scores[i]
            })
        return results
    except Exception as e:
        print(f"An error occurred during batch prediction: {e}")
        raise
