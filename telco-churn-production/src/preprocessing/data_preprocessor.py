import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(
        self,
        categorical_strategy='onehot',  # 'onehot' or 'label'
        numerical_strategy='standard',  # 'standard' or 'minmax'
        feature_engineering_enabled=True
    ):
        self.categorical_strategy = categorical_strategy
        self.numerical_strategy = numerical_strategy
        self.feature_engineering_enabled = feature_engineering_enabled
        self.preprocessor = None
        self.categorical_features = []
        self.numerical_features = []
        self.feature_names_out = None

    def _identify_features(self, df: pd.DataFrame):
        # Identify categorical and numerical features based on common Telco Churn dataset columns
        # Exclude 'customerID' and 'Churn' (target) if present
        all_features = [col for col in df.columns if col not in ['customerID', 'Churn']]

        # These are typical for the Telco Churn dataset
        self.categorical_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        self.numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

        # Filter to ensure only columns present in the DataFrame are used
        self.categorical_features = [f for f in self.categorical_features if f in df.columns]
        self.numerical_features = [f for f in self.numerical_features if f in df.columns]

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_engineering_enabled:
            return df

        df_fe = df.copy()

        # 1. Convert TotalCharges to numeric, coercing errors to NaN and imputing
        if 'TotalCharges' in df_fe.columns:
            df_fe['TotalCharges'] = pd.to_numeric(df_fe['TotalCharges'], errors='coerce')
            # Impute with median or mean, or 0 if it makes sense for the domain
            # Using median to be robust to outliers
            df_fe['TotalCharges'].fillna(df_fe['TotalCharges'].median(), inplace=True)

        # 2. Create tenure groups
        if 'tenure' in df_fe.columns:
            bins = [0, 12, 24, 36, 48, 60, 72]
            labels = ['1-12', '13-24', '25-36', '37-48', '49-60', '61-72']
            df_fe['tenure_group'] = pd.cut(df_fe['tenure'], bins=bins, labels=labels, right=False)
            self.categorical_features.append('tenure_group') # Add new feature to categorical list

        # 3. MonthlyCharges per tenure (if TotalCharges and tenure exist)
        if 'TotalCharges' in df_fe.columns and 'tenure' in df_fe.columns:
            df_fe['AvgMonthlyCharge'] = df_fe.apply(
                lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else 0,
                axis=1
            )
            self.numerical_features.append('AvgMonthlyCharge') # Add new feature to numerical list

        # 4. Has Internet Service
        if 'InternetService' in df_fe.columns:
            df_fe['HasInternetService'] = df_fe['InternetService'].apply(lambda x: 1 if x != 'No' else 0)
            self.categorical_features.append('HasInternetService')

        # 5. Combine Security/Protection features
        security_features = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        if all(f in df_fe.columns for f in security_features):
            df_fe['HasSecurityProtection'] = df_fe[security_features].apply(
                lambda x: 1 if any(val == 'Yes' for val in x) else 0,
                axis=1
            )
            self.categorical_features.append('HasSecurityProtection')

        # 6. Convert SeniorCitizen to boolean (if not already numeric)
        if 'SeniorCitizen' in df_fe.columns and df_fe['SeniorCitizen'].dtype == 'int64':
            df_fe['SeniorCitizen'] = df_fe['SeniorCitizen'].astype(bool)
            # If it's treated as categorical, ensure it's in the list
            if 'SeniorCitizen' not in self.categorical_features and 'SeniorCitizen' not in self.numerical_features:
                self.categorical_features.append('SeniorCitizen')

        return df_fe

    def _build_preprocessor(self):
        transformers = []

        # Numerical transformer
        if self.numerical_strategy == 'standard':
            numerical_transformer = StandardScaler()
        elif self.numerical_strategy == 'minmax':
            numerical_transformer = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported numerical strategy: {self.numerical_strategy}")
        transformers.append(('num', numerical_transformer, self.numerical_features))

        # Categorical transformer
        if self.categorical_strategy == 'onehot':
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        elif self.categorical_strategy == 'label':
            # LabelEncoder is typically for target variable or single column
            # For multiple columns in ColumnTransformer, it's more complex.
            # We'll stick to OneHotEncoder for simplicity in ColumnTransformer for now.
            raise ValueError("LabelEncoder is not directly supported for multiple columns in ColumnTransformer. Use OneHotEncoder.")
        else:
            raise ValueError(f"Unsupported categorical strategy: {self.categorical_strategy}")
        transformers.append(('cat', categorical_transformer, self.categorical_features))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    def fit(self, X: pd.DataFrame, y=None):
        """Fits the preprocessor to the data."""
        self._identify_features(X)
        X_fe = self._feature_engineer(X)
        self._build_preprocessor()
        self.preprocessor.fit(X_fe)
        self.feature_names_out = self.preprocessor.get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data using the fitted preprocessor."""
        X_fe = self._feature_engineer(X)
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been fitted. Call .fit() first.")
        X_transformed = self.preprocessor.transform(X_fe)
        return pd.DataFrame(X_transformed, columns=self.feature_names_out, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fits the preprocessor and then transforms the data."""
        self.fit(X, y)
        return self.transform(X)
