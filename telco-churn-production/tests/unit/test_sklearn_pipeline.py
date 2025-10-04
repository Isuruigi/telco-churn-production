import pytest
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add src directory to path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.sklearn_production_pipeline import ChurnPipelineProduction
from src.model_persistence import ModelPersistence
from src.inference_engine import InferenceEngine
from config.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

# Use the full feature set for integration testing
TEST_NUM_FEATURES = NUMERICAL_FEATURES
TEST_CAT_FEATURES = CATEGORICAL_FEATURES

@pytest.fixture(scope="module")
def mock_dataset():
    """Creates a mock classification dataset for the entire test module."""
    n_features = len(TEST_NUM_FEATURES) + len(TEST_CAT_FEATURES)
    X, y = make_classification(n_samples=200, n_features=n_features, 
                               n_informative=10, n_redundant=5, random_state=42)
    
    # Create a pandas DataFrame with correct column names
    columns = TEST_NUM_FEATURES + TEST_CAT_FEATURES
    X = pd.DataFrame(X, columns=columns)
    
    # Add realistic mock data for all categorical features
    for col in TEST_CAT_FEATURES:
        X[col] = np.random.choice([f'{col}_A', f'{col}_B'], 200)
    
    return X, y

class TestChurnPipelineProduction:
    @pytest.fixture
    def pipeline_instance(self):
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        return ChurnPipelineProduction(model, numerical_features=TEST_NUM_FEATURES, categorical_features=TEST_CAT_FEATURES)

    def test_pipeline_creation(self, pipeline_instance):
        pipeline = pipeline_instance.create_model_pipeline()
        assert pipeline is not None
        assert 'preprocessor' in pipeline.named_steps
        assert 'model' in pipeline.named_steps

    def test_training_and_evaluation(self, pipeline_instance, mock_dataset):
        X, y = mock_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline_instance.train_with_validation(X_train, y_train)
        metrics = pipeline_instance.evaluate_model(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_training_reproducibility(self, pipeline_instance, mock_dataset):
        X, y = mock_dataset
        
        # First run
        pipeline_instance.train_with_validation(X, y)
        preds1 = pipeline_instance.pipeline.predict(X)

        # Second run with a new instance
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        new_pipeline_instance = ChurnPipelineProduction(model, numerical_features=TEST_NUM_FEATURES, categorical_features=TEST_CAT_FEATURES)
        new_pipeline_instance.train_with_validation(X, y)
        preds2 = new_pipeline_instance.pipeline.predict(X)

        np.testing.assert_array_equal(preds1, preds2)

class TestModelPersistence:
    def test_model_saving_loading(self, tmp_path):
        persistence = ModelPersistence(model_dir=str(tmp_path))
        model = RandomForestClassifier(random_state=42)
        metadata = {"test_metric": 0.95}
        model_name = "test_model"

        persistence.save_model_with_metadata(model, model_name, metadata)
        
        # Test validation
        assert persistence.validate_model_artifacts(os.path.join(tmp_path, f"{model_name}.joblib"))

        # Test loading
        loaded_model, loaded_metadata = persistence.load_model_with_metadata(model_name)
        assert loaded_model is not None
        assert loaded_metadata["test_metric"] == 0.95

class TestInferenceEngine:
    @pytest.fixture
    def trained_pipeline_path(self, tmp_path, mock_dataset):
        X, y = mock_dataset
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        prod_pipeline = ChurnPipelineProduction(model, numerical_features=TEST_NUM_FEATURES, categorical_features=TEST_CAT_FEATURES)
        prod_pipeline.train_with_validation(X, y)
        
        pipeline_path = tmp_path / "test_pipeline.joblib"
        prod_pipeline.save_pipeline_artifacts(str(pipeline_path))
        return str(pipeline_path)

    def test_inference_engine_loading(self, trained_pipeline_path):
        engine = InferenceEngine(model_path=trained_pipeline_path)
        assert engine.pipeline is not None

    def test_inference_accuracy(self, trained_pipeline_path, mock_dataset):
        X, y = mock_dataset
        engine = InferenceEngine(model_path=trained_pipeline_path)
        
        # Get prediction from the raw pipeline
        raw_pipeline = joblib.load(trained_pipeline_path)
        raw_preds = raw_pipeline.predict(X)

        # Get prediction from the inference engine
        engine_preds_list = engine.predict_batch_customers(X.to_dict(orient='records'))
        engine_preds = [1 if res['prediction'] == 'Churn' else 0 for res in engine_preds_list]

        np.testing.assert_array_equal(raw_preds, engine_preds)

    def test_input_validation(self, trained_pipeline_path):
        engine = InferenceEngine(model_path=trained_pipeline_path)
        # Missing a feature
        invalid_sample = {'tenure': 10, 'Contract': 'One year'}
        with pytest.raises(ValueError):
            engine.predict_single_customer(invalid_sample)