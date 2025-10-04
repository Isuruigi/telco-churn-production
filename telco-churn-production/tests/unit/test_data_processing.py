import pytest
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Add src directory to path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader import TelcoDataLoader
from src.preprocessor import DataPreprocessor
from src.feature_engineer import FeatureEngineer
from src.imbalance_handler import ImbalanceHandler
from config.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

@pytest.fixture
def mock_data():
    """Creates a mock DataFrame for testing."""
    data = {
        'customerID': ['1', '2', '3', '4', '5'],
        'gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
        'tenure': [1, 10, 20, 30, 40],
        'MonthlyCharges': [29.85, 50.0, 60.0, 70.0, 80.0],
        'TotalCharges': ['29.85', '500', '1200', '2100', ' '],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
        'Churn': ['No', 'Yes', 'No', 'Yes', 'No']
    }
    # Add all required features with dummy data
    for col in CATEGORICAL_FEATURES:
        if col not in data:
            data[col] = ['No'] * 5
    for col in NUMERICAL_FEATURES:
        if col not in data:
            data[col] = [0] * 5
            
    return pd.DataFrame(data)

@pytest.fixture
def temp_csv_file(tmp_path, mock_data):
    """Creates a temporary CSV file for testing data loading."""
    file_path = tmp_path / "temp_data.csv"
    mock_data.to_csv(file_path, index=False)
    return str(file_path)

class TestDataLoader:
    def test_load_raw_data(self, temp_csv_file):
        loader = TelcoDataLoader(data_path=temp_csv_file)
        df = loader.load_raw_data()
        assert df is not None
        assert not df.empty

    def test_load_invalid_path(self):
        loader = TelcoDataLoader(data_path='invalid/path/to/file.csv')
        df = loader.load_raw_data()
        assert df is None

    def test_validate_data_quality(self, mock_data):
        loader = TelcoDataLoader(data_path='')
        # Test with clean data
        assert loader.validate_data_quality(mock_data) == True
        # Test with data having duplicates
        mock_data_dups = pd.concat([mock_data, mock_data.head(1)])
        assert loader.validate_data_quality(mock_data_dups) == True # Should still pass but print a message

class TestPreprocessor:
    def test_preprocess_data(self, mock_data):
        df_processed = DataPreprocessor.preprocess_data(mock_data)
        assert 'customerID' not in df_processed.columns
        assert pd.api.types.is_numeric_dtype(df_processed['TotalCharges'])
        # Check that the row with space in TotalCharges is now NaN
        assert df_processed['TotalCharges'].isnull().sum() == 1

    def test_create_preprocessing_pipeline(self):
        preprocessor_builder = DataPreprocessor()
        pipeline = preprocessor_builder.create_preprocessing_pipeline()
        assert isinstance(pipeline, ColumnTransformer)

class TestFeatureEngineer:
    def test_transform(self, mock_data):
        fe = FeatureEngineer()
        df_engineered = fe.transform(mock_data)
        assert 'tenure_category' in df_engineered.columns
        assert 'num_services' in df_engineered.columns
        assert 'tenure_x_monthly_charges' in df_engineered.columns

class TestImbalanceHandler:
    @pytest.fixture
    def imbalanced_data(self):
        X = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series(np.array([0]*90 + [1]*10))
        return X, y

    def test_apply_smote(self, imbalanced_data):
        handler = ImbalanceHandler()
        X, y = imbalanced_data
        X_res, y_res = handler.apply_smote(X, y)
        assert len(X_res) > len(X)
        assert y_res.value_counts()[0] == y_res.value_counts()[1]

    def test_apply_random_undersampling(self, imbalanced_data):
        handler = ImbalanceHandler()
        X, y = imbalanced_data
        X_res, y_res = handler.apply_random_undersampling(X, y)
        assert len(X_res) < len(X)
        assert y_res.value_counts()[0] == y_res.value_counts()[1]

    def test_cost_sensitive_model(self):
        from sklearn.linear_model import LogisticRegression
        handler = ImbalanceHandler()
        model = handler.get_cost_sensitive_model(LogisticRegression)
        assert model.get_params()['class_weight'] == 'balanced'
