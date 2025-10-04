import joblib
import pandas as pd
import numpy as np
import shap
import os
from src.config.config import MODELS_PATH, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

class InferenceEngine:
    """Handles new data inference with input validation and explanations."""

    def __init__(self, model_path: str):
        """
        Initializes the InferenceEngine.

        Args:
            model_path (str): Path to the trained scikit-learn pipeline file.
        """
        self.pipeline = self.load_trained_pipeline(model_path)
        self.explainer = None
        self.feature_schema = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        self.numerical_features = NUMERICAL_FEATURES

    @staticmethod
    def load_trained_pipeline(model_path: str):
        """Loads the trained model pipeline from a file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model pipeline not found at {model_path}")
        try:
            pipeline = joblib.load(model_path)
            print("Inference pipeline loaded successfully.")
            return pipeline
        except Exception as e:
            raise IOError(f"Error loading pipeline: {e}")

    def _validate_input(self, input_data: pd.DataFrame):
        """Performs comprehensive validation on the input DataFrame."""
        # 1. Check for missing columns
        missing_cols = set(self.feature_schema) - set(input_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in input data: {list(missing_cols)}")

        # 2. Check data types (simple version)
        # In a real-world scenario, you might enforce specific dtypes
        for col in self.numerical_features:
            if not pd.api.types.is_numeric_dtype(input_data[col]):
                # Attempt to coerce, but raise error if it fails for critical fields
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
                if input_data[col].isnull().any():
                    raise TypeError(f"Column '{col}' could not be converted to a numeric type.")
        return input_data

    def predict_single_customer(self, customer_data: dict) -> dict:
        """Predicts churn for a single customer with validation."""
        df = pd.DataFrame([customer_data])
        df = self._validate_input(df)
        
        prediction = self.pipeline.predict(df)[0]
        confidence = self.get_prediction_confidence(df)[0]
        
        return {
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'confidence': confidence
        }

    def predict_batch_customers(self, customers_data: list) -> list:
        """Predicts churn for a batch of customers with validation."""
        df = pd.DataFrame(customers_data)
        df = self._validate_input(df)

        predictions = self.pipeline.predict(df)
        confidences = self.get_prediction_confidence(df)

        results = []
        for i in range(len(df)):
            results.append({
                'prediction': 'Churn' if predictions[i] == 1 else 'No Churn',
                'confidence': confidences[i]
            })
        return results

    def get_prediction_confidence(self, customer_data) -> np.ndarray:
        """Returns the prediction probability for the predicted class."""
        probabilities = self.pipeline.predict_proba(customer_data)
        return np.max(probabilities, axis=1)

    def get_feature_explanations(self, customer_data: dict, training_data_sample: pd.DataFrame):
        """Generates and returns SHAP feature explanations for a single prediction."""
        if self.explainer is None:
            print("Initializing SHAP explainer...")
            preprocessor = self.pipeline.steps[0][1]
            model = self.pipeline.steps[1][1]
            
            # Use a masker for KernelExplainer for better feature name handling
            masker = shap.maskers.Independent(preprocessor.transform(training_data_sample))
            self.explainer = shap.KernelExplainer(model.predict_proba, masker)

        input_df = pd.DataFrame([customer_data])
        input_transformed = self.pipeline.steps[0][1].transform(input_df)
        
        shap_values = self.explainer.shap_values(input_transformed)
        
        # For binary classification, get explanations for the positive class
        shap_values_positive_class = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Format the output
        feature_names = self.pipeline.steps[0][1].get_feature_names_out()
        explanations = {feature: value for feature, value in zip(feature_names, shap_values_positive_class[0])}
        
        print("--- SHAP Explanation (Top 10 Features) ---")
        sorted_explanations = sorted(explanations.items(), key=lambda item: abs(item[1]), reverse=True)
        for feature, value in sorted_explanations[:10]:
            print(f"- {feature}: {value:.4f}")

        return explanations

if __name__ == '__main__':
    from src.data_loader import TelcoDataLoader
    from sklearn.model_selection import train_test_split

    MODEL_FILE = os.path.join(MODELS_PATH, 'final_pipeline.joblib')

    if not os.path.exists(MODEL_FILE):
        print(f"Model file not found at {MODEL_FILE}. Please run the training pipeline first.")
    else:
        # 1. Initialize the engine
        engine = InferenceEngine(model_path=MODEL_FILE)

        # 2. Create a sample customer
        sample = {
            'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'Yes',
            'tenure': 72, 'PhoneService': 'Yes', 'MultipleLines': 'Yes', 'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No', 'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes',
            'TechSupport': 'Yes', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
            'Contract': 'Two year', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Credit card (automatic)',
            'MonthlyCharges': 113.25, 'TotalCharges': '8684.8'
        }

        # 3. Get a single prediction
        result = engine.predict_single_customer(sample)
        print(f"\n--- Single Prediction Result ---\n{result}")

        # 4. Get feature explanations
        print("\n--- Generating Feature Explanations ---")
        df = TelcoDataLoader().load_raw_data()
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]
        X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        
        engine.get_feature_explanations(sample, X_train.head(100))