import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, RANDOM_STATE

# Custom transformer for feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        # Example: tenure in bins
        X_transformed['tenure_group'] = pd.cut(X_transformed['tenure'], bins=[0, 12, 24, 36, 48, 60, np.inf], labels=['0-12', '12-24', '24-36', '36-48', '48-60', '60+'])
        return X_transformed

# Custom transformer for handling outliers
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col in self.columns:
            Q1 = X_transformed[col].quantile(0.25)
            Q3 = X_transformed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X_transformed[col] = np.clip(X_transformed[col], lower_bound, upper_bound)
        return X_transformed

class DataPreprocessor:
    """A class to create a full preprocessing pipeline for the Telco churn data."""

    def __init__(self, numerical_features=NUMERICAL_FEATURES, categorical_features=CATEGORICAL_FEATURES):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

    def _create_numerical_pipeline(self) -> Pipeline:
        """Creates the pipeline for numerical features."""
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

    def _create_categorical_pipeline(self) -> Pipeline:
        """Creates the pipeline for categorical features."""
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Creates the full preprocessing pipeline using ColumnTransformer."""
        # First, convert TotalCharges to numeric. This needs to be done before the pipeline.
        # A custom transformer could also be built for this.

        numerical_pipeline = self._create_numerical_pipeline()
        categorical_pipeline = self._create_categorical_pipeline()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ],
            remainder='passthrough' # Keep other columns (if any)
        )
        return preprocessor

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """A static method to apply initial data cleaning steps before the pipeline."""
        df_processed = df.copy()
        # Convert TotalCharges to numeric, coercing errors to NaN
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
        # Drop customerID as it is not a feature
        if 'customerID' in df_processed.columns:
            df_processed = df_processed.drop('customerID', axis=1)
        return df_processed

if __name__ == '__main__':
    from src.data_loader import TelcoDataLoader

    # 1. Load data
    loader = TelcoDataLoader()
    churn_df = loader.load_raw_data()

    if churn_df is not None:
        # 2. Apply initial preprocessing
        preprocessed_df = DataPreprocessor.preprocess_data(churn_df)

        # 3. Create the preprocessing pipeline
        preprocessor_builder = DataPreprocessor()
        full_pipeline = preprocessor_builder.create_preprocessing_pipeline()

        # 4. Fit and transform the data
        # In a real scenario, you would split into train/test before fitting
        X = preprocessed_df.drop('Churn', axis=1)
        y = preprocessed_df['Churn']
        
        print("Fitting and transforming the data...")
        X_transformed = full_pipeline.fit_transform(X)

        # Get feature names after transformation
        feature_names = full_pipeline.get_feature_names_out()
        
        # Create a DataFrame with the transformed data
        X_transformed_df = pd.DataFrame(X_transformed, index=X.index, columns=feature_names)
        
        print("Shape of transformed data:", X_transformed_df.shape)
        print("First 5 rows of transformed data:")
        print(X_transformed_df.head())
