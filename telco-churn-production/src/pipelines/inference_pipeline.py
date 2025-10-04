"""
Telco Churn Prediction - Inference Pipeline
This module implements inference capabilities for the trained churn prediction model.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Union, List


class InferencePipeline:
    """
    Inference pipeline for making churn predictions on new customer data.
    Supports both single customer and batch predictions.
    """

    def __init__(self, model_path='src/models/sklearn_pipeline.pkl'):
        """
        Initialize the inference pipeline by loading the trained model.

        Args:
            model_path (str): Path to the saved pipeline model
        """
        self.model_path = model_path
        self.pipeline = self._load_model()

    def _load_model(self):
        """
        Load the saved model from disk.

        Returns:
            Pipeline: Loaded sklearn pipeline

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path}\n"
                "Please train the model first using sklearn_pipeline.py"
            )

        print(f"Loading model from: {self.model_path}")
        pipeline = joblib.load(self.model_path)
        print("Model loaded successfully!")
        return pipeline

    def _get_risk_level(self, probability: float) -> str:
        """
        Determine risk level based on churn probability.

        Args:
            probability (float): Churn probability (0-1)

        Returns:
            str: Risk level (Low, Medium, or High)
        """
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"

    def predict_single(self, customer_data: Dict) -> Dict:
        """
        Make a prediction for a single customer.

        Args:
            customer_data (dict): Dictionary containing customer features

        Returns:
            dict: Prediction results containing probability and risk level

        Example:
            >>> customer = {
            ...     'gender': 'Male',
            ...     'SeniorCitizen': 0,
            ...     'Partner': 'Yes',
            ...     'Dependents': 'No',
            ...     'tenure': 12,
            ...     'PhoneService': 'Yes',
            ...     'MultipleLines': 'No',
            ...     'InternetService': 'Fiber optic',
            ...     'OnlineSecurity': 'No',
            ...     'OnlineBackup': 'No',
            ...     'DeviceProtection': 'No',
            ...     'TechSupport': 'No',
            ...     'StreamingTV': 'Yes',
            ...     'StreamingMovies': 'Yes',
            ...     'Contract': 'Month-to-month',
            ...     'PaperlessBilling': 'Yes',
            ...     'PaymentMethod': 'Electronic check',
            ...     'MonthlyCharges': 89.95,
            ...     'TotalCharges': 1079.40
            ... }
            >>> result = inference.predict_single(customer)
        """
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])

        # Make prediction
        churn_probability = self.pipeline.predict_proba(df)[0, 1]
        churn_prediction = self.pipeline.predict(df)[0]
        risk_level = self._get_risk_level(churn_probability)

        result = {
            'churn_prediction': 'Yes' if churn_prediction == 1 else 'No',
            'churn_probability': float(churn_probability),
            'risk_level': risk_level
        }

        return result

    def predict_batch(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for a batch of customers.

        Args:
            customers_df (pd.DataFrame): DataFrame containing customer features

        Returns:
            pd.DataFrame: DataFrame with predictions, probabilities, and risk levels

        Example:
            >>> customers = pd.read_csv('new_customers.csv')
            >>> results = inference.predict_batch(customers)
        """
        # Make predictions
        churn_probabilities = self.pipeline.predict_proba(customers_df)[:, 1]
        churn_predictions = self.pipeline.predict(customers_df)

        # Calculate risk levels
        risk_levels = [self._get_risk_level(prob) for prob in churn_probabilities]

        # Create results DataFrame
        results_df = customers_df.copy()
        results_df['churn_prediction'] = ['Yes' if pred == 1 else 'No' for pred in churn_predictions]
        results_df['churn_probability'] = churn_probabilities
        results_df['risk_level'] = risk_levels

        return results_df

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Args:
            top_n (int): Number of top features to return

        Returns:
            pd.DataFrame: DataFrame with feature names and importance scores
        """
        # Get the classifier from the pipeline
        classifier = self.pipeline.named_steps['classifier']

        # Get feature names after preprocessing
        preprocessor = self.pipeline.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()

        # Get feature importances
        importances = classifier.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df


def main():
    """Main function to demonstrate inference pipeline usage."""
    print("="*60)
    print("TELCO CHURN PREDICTION - INFERENCE PIPELINE")
    print("="*60)

    # Initialize inference pipeline
    try:
        inference = InferencePipeline(model_path='src/models/sklearn_pipeline.pkl')
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    print("\n" + "-"*60)
    print("TEST 1: Single Customer Prediction")
    print("-"*60)

    # Test with sample customer data
    sample_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 89.95,
        'TotalCharges': 1079.40
    }

    print("\nCustomer Profile:")
    for key, value in sample_customer.items():
        print(f"  {key}: {value}")

    result = inference.predict_single(sample_customer)

    print("\nPrediction Results:")
    print(f"  Churn Prediction: {result['churn_prediction']}")
    print(f"  Churn Probability: {result['churn_probability']:.2%}")
    print(f"  Risk Level: {result['risk_level']}")

    print("\n" + "-"*60)
    print("TEST 2: Batch Prediction")
    print("-"*60)

    # Create batch of customers
    batch_customers = pd.DataFrame([
        {
            'gender': 'Female',
            'SeniorCitizen': 1,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 2,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 70.70,
            'TotalCharges': 151.65
        },
        {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'Yes',
            'tenure': 48,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Bank transfer (automatic)',
            'MonthlyCharges': 103.70,
            'TotalCharges': 4820.40
        }
    ])

    batch_results = inference.predict_batch(batch_customers)

    print(f"\nBatch prediction completed for {len(batch_results)} customers")
    print("\nResults Summary:")
    print(batch_results[['tenure', 'Contract', 'MonthlyCharges',
                         'churn_prediction', 'churn_probability', 'risk_level']])

    print("\n" + "-"*60)
    print("TEST 3: Feature Importance")
    print("-"*60)

    importance_df = inference.get_feature_importance(top_n=10)
    print("\nTop 10 Most Important Features:")
    print(importance_df.to_string(index=False))

    print("\n" + "="*60)
    print("INFERENCE PIPELINE TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
