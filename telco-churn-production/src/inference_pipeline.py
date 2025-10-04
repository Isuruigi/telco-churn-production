import joblib
import pandas as pd
import numpy as np
import shap
import os
from src.config.config import MODELS_PATH

class InferencePipeline:
    """Orchestrates the real-time prediction workflow."""

    def __init__(self, model_path: str):
        """
        Initializes the InferencePipeline.

        Args:
            model_path (str): Path to the trained model pipeline file.
        """
        self.model_path = model_path
        self.pipeline = self.load_trained_model()
        self.explainer = None

    def load_trained_model(self):
        """Loads the trained model pipeline."""
        try:
            pipeline = joblib.load(self.model_path)
            print(f"Pipeline loaded successfully from {self.model_path}")
            return pipeline
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}")
            return None
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            return None

    def _prepare_input(self, input_data) -> pd.DataFrame:
        """Converts input data (dict or DataFrame) to a DataFrame for prediction."""
        if isinstance(input_data, dict):
            return pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            return input_data
        else:
            raise ValueError("Input data must be a dictionary or a pandas DataFrame.")

    def predict_single(self, input_data: dict) -> tuple:
        """Makes a prediction on a single data point."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not loaded.")
        
        df = self._prepare_input(input_data)
        prediction = self.pipeline.predict(df)[0]
        confidence = self.calculate_prediction_confidence(df)[0]
        
        return prediction, confidence

    def predict_batch(self, input_data) -> tuple:
        """Makes predictions on a batch of data."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not loaded.")
        
        df = self._prepare_input(input_data)
        predictions = self.pipeline.predict(df)
        confidences = self.calculate_prediction_confidence(df)
        
        return predictions, confidences

    def calculate_prediction_confidence(self, input_data) -> np.ndarray:
        """Calculates the confidence score for predictions."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not loaded.")
        
        df = self._prepare_input(input_data)
        probabilities = self.pipeline.predict_proba(df)
        # Confidence is the probability of the predicted class
        confidence = np.max(probabilities, axis=1)
        return confidence

    def generate_explanations(self, input_data, training_data_sample):
        """
        Generates SHAP explanations for a single prediction.

        Args:
            input_data (dict): The single data point to explain.
            training_data_sample (pd.DataFrame): A sample of the training data for the explainer.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not loaded.")

        # The preprocessor is the first step of the pipeline
        preprocessor = self.pipeline.steps[0][1]
        # The model is the second step
        model = self.pipeline.steps[1][1]

        # Create a SHAP explainer
        # For tree models, TreeExplainer is faster. For others, KernelExplainer.
        if hasattr(model, 'feature_importances_'): # Likely a tree-based model
            self.explainer = shap.TreeExplainer(model, data=preprocessor.transform(training_data_sample))
        else: # For linear models or others
            self.explainer = shap.KernelExplainer(model.predict_proba, preprocessor.transform(training_data_sample))

        # Transform the single input data point
        input_df = self._prepare_input(input_data)
        input_transformed = preprocessor.transform(input_df)

        # Get SHAP values
        shap_values = self.explainer.shap_values(input_transformed)

        # For binary classification, shap_values can be a list of two arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # Explanations for the positive class

        # Create a SHAP explanation object
        feature_names = preprocessor.get_feature_names_out()
        explanation = shap.Explanation(values=shap_values, base_values=self.explainer.expected_value, 
                                       data=input_transformed, feature_names=feature_names)

        print("--- SHAP Explanation ---")
        print("Explanation for predicting the 'Churn' class:")
        shap.plots.waterfall(explanation[0], max_display=10)
        return explanation

if __name__ == '__main__':
    # This assumes a trained pipeline has been saved, e.g., by running training_pipeline.py
    model_file = os.path.join(MODELS_PATH, 'final_pipeline.joblib')

    if not os.path.exists(model_file):
        print(f"Model file not found at {model_file}. Please run the training pipeline first.")
    else:
        # Initialize the inference pipeline
        inference_pipe = InferencePipeline(model_path=model_file)

        # Create a sample customer data point for prediction
        sample_customer = {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 1,
            'PhoneService': 'No',
            'MultipleLines': 'No phone service',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 29.85,
            'TotalCharges': '29.85'
        }

        # 1. Make a single prediction
        prediction, confidence = inference_pipe.predict_single(sample_customer)
        print(f"\n--- Single Prediction ---")
        print(f"Predicted Churn: {'Yes' if prediction == 1 else 'No'}")
        print(f"Confidence: {confidence:.2%}")

        # 2. Generate explanations (requires a sample of training data)
        # For demonstration, we'll load the raw data again to get a sample
        from src.data_loader import TelcoDataLoader
        from sklearn.model_selection import train_test_split
        df = TelcoDataLoader().load_raw_data()
        df_processed = DataPreprocessor.preprocess_data(df.copy())
        X = df_processed.drop(TARGET_COLUMN, axis=1)
        y = df_processed[TARGET_COLUMN]
        X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        
        inference_pipe.generate_explanations(sample_customer, X_train.head(100)) # Use 100 samples for explainer background
