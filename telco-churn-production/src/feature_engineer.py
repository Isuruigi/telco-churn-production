import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates new features for the Telco churn dataset."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Applies all feature engineering steps."""
        X_eng = X.copy()
        X_eng = self.create_tenure_categories(X_eng)
        X_eng = self.create_service_combinations(X_eng)
        X_eng = self.create_spending_features(X_eng)
        X_eng = self.create_interaction_features(X_eng)
        # X_eng = self.create_aggregated_features(X_eng) # This is more complex and might require fitting
        return X_eng

    def create_tenure_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bins the 'tenure' feature into categories."""
        bins = [0, 12, 24, 48, 60, np.inf]
        labels = ['Short-term', 'Mid-term', 'Long-term', 'Very-long-term', 'Loyal']
        df['tenure_category'] = pd.cut(df['tenure'], bins=bins, labels=labels)
        return df

    def create_service_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates features by combining different services."""
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['num_services'] = df[service_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)
        df['has_streaming'] = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)
        return df

    def create_spending_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates features related to customer spending."""
        # Ensure TotalCharges is numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # To avoid division by zero
        df_temp = df.replace({'tenure': 0}, np.nan)
        
        df['monthly_charge_per_tenure'] = df['MonthlyCharges'] / df_temp['tenure']
        df['total_charges_to_tenure_ratio'] = df['TotalCharges'] / df_temp['tenure']
        
        # Impute NaNs created by division by zero (for tenure=0 customers)
        df.fillna(0, inplace=True)
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates interaction features."""
        # Example: Interaction between tenure and monthly charges
        df['tenure_x_monthly_charges'] = df['tenure'] * df['MonthlyCharges']
        return df

    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates features based on group-wise statistics."""
        # This is more complex as it can lead to data leakage if not handled carefully.
        # For example, calculating group means should be done on the training set and then applied to the test set.
        # Here is a simple example of how it could be done.
        
        # Calculate average monthly charges per contract type
        contract_avg_charges = df.groupby('Contract')['MonthlyCharges'].transform('mean')
        df['monthly_charges_vs_contract_avg'] = df['MonthlyCharges'] - contract_avg_charges
        return df

if __name__ == '__main__':
    from src.data_loader import TelcoDataLoader

    # Load data
    loader = TelcoDataLoader()
    churn_df = loader.load_raw_data()

    if churn_df is not None:
        # Initialize and apply feature engineering
        feature_engineer = FeatureEngineer()
        engineered_df = feature_engineer.transform(churn_df)

        print("Shape of original data:", churn_df.shape)
        print("Shape of engineered data:", engineered_df.shape)
        print("\nNew columns:")
        new_cols = [col for col in engineered_df.columns if col not in churn_df.columns]
        print(new_cols)
        print("\nFirst 5 rows of engineered data with new columns:")
        print(engineered_df[new_cols + ['tenure', 'MonthlyCharges']].head())
