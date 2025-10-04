"""
Telco Churn Prediction - Scikit-Learn Pipeline
This module implements the complete ML pipeline for churn prediction using sklearn.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


class TelcoChurnPipeline:
    """
    Complete pipeline for Telco Churn prediction using scikit-learn.
    Handles data loading, preprocessing, training, and model persistence.
    """

    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize the pipeline with configuration.

        Args:
            config_path (str): Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_data(self):
        """
        Load data from the path specified in config file.

        Returns:
            pd.DataFrame: Loaded dataset
        """
        data_path = self.config['data']['raw']
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Data loaded: {df.shape}")
        return df

    def preprocess_data(self, df):
        """
        Preprocess the raw data.

        Args:
            df (pd.DataFrame): Raw dataframe

        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Convert TotalCharges to numeric, coercing errors
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Drop rows with missing TotalCharges
        df = df.dropna(subset=['TotalCharges'])

        # Drop customerID if it exists (not a feature)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)

        print(f"Data after preprocessing: {df.shape}")
        return df

    def create_preprocessing_pipeline(self):
        """
        Create a ColumnTransformer preprocessing pipeline.
        Applies StandardScaler to numerical features and OneHotEncoder to categorical features.

        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        numerical_features = self.config['features']['numerical']
        categorical_features = self.config['features']['categorical']

        # Numerical transformer
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )

        return preprocessor

    def build_pipeline(self):
        """
        Build the full pipeline combining preprocessor and RandomForestClassifier.

        Returns:
            Pipeline: Complete ML pipeline
        """
        preprocessor = self.create_preprocessing_pipeline()

        # Get RandomForest parameters from config
        rf_params = self.config['models']['random_forest']

        # Create full pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**rf_params))
        ])

        print("Pipeline created successfully")
        return self.pipeline

    def prepare_train_test_split(self, df):
        """
        Split data into training and testing sets.

        Args:
            df (pd.DataFrame): Preprocessed dataframe
        """
        target_column = self.config['target']
        test_size = self.config['training']['test_size']
        random_state = self.config['training']['random_state']

        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column].map({'Yes': 1, 'No': 0})

        # Split data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        print(f"Class distribution in training: {self.y_train.value_counts().to_dict()}")

    def train(self):
        """Train the pipeline on the training data."""
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")

        print("\nTraining the model...")
        self.pipeline.fit(self.X_train, self.y_train)
        print("Training completed!")

    def evaluate(self):
        """
        Evaluate the trained model and print metrics.

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not trained. Call train() first.")

        # Make predictions
        y_pred = self.pipeline.predict(self.X_test)
        y_pred_proba = self.pipeline.predict_proba(self.X_test)[:, 1]

        # Calculate metrics
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Churn', 'Churn']))

        print(f"\nROC AUC Score: {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print(f"\nTrue Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

        return {
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }

    def save_pipeline(self, output_path='src/models/sklearn_pipeline.pkl'):
        """
        Save the trained pipeline to disk using joblib.

        Args:
            output_path (str): Path to save the pipeline
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not trained. Call train() first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save pipeline
        joblib.dump(self.pipeline, output_path)
        print(f"\nPipeline saved to: {output_path}")

    def run_full_pipeline(self):
        """Execute the complete pipeline from data loading to model saving."""
        print("="*60)
        print("TELCO CHURN PREDICTION - SKLEARN PIPELINE")
        print("="*60)

        # Load and preprocess data
        df = self.load_data()
        df = self.preprocess_data(df)

        # Prepare train/test split
        self.prepare_train_test_split(df)

        # Build and train pipeline
        self.build_pipeline()
        self.train()

        # Evaluate
        metrics = self.evaluate()

        # Save pipeline
        self.save_pipeline()

        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED")
        print("="*60)

        return metrics


def main():
    """Main execution function."""
    # Initialize and run pipeline
    pipeline = TelcoChurnPipeline(config_path='config/config.yaml')
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
