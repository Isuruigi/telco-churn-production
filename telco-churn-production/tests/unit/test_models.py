import pytest
import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Add src directory to path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.base_model import LogisticRegressionModel
from src.ensemble_models import RandomForestChurnModel, XGBoostChurnModel
from src.advanced_ensemble import VotingChurnModel, StackingChurnModel
from src.hyperparameter_tuner import HyperparameterTuner

@pytest.fixture
def classification_data():
    """Creates a mock classification dataset for testing."""
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    X = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
    return X, y

class TestBaseModel:
    def test_logistic_regression_model(self, classification_data, tmp_path):
        X, y = classification_data
        model = LogisticRegressionModel(random_state=42)
        
        # Test train and predict
        model.train(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        
        # Test evaluate
        metrics = model.evaluate(X, y)
        assert 'roc_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1

        # Test save and load
        model_path = tmp_path / "lr_model.joblib"
        model.save_model(str(model_path))
        assert os.path.exists(model_path)
        loaded_model = model.load_model(str(model_path))
        assert loaded_model is not None

class TestEnsembleModels:
    def test_random_forest_model(self, classification_data):
        X, y = classification_data
        model = RandomForestChurnModel(random_state=42)
        model.train(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_xgboost_model(self, classification_data):
        X, y = classification_data
        model = XGBoostChurnModel(random_state=42)
        model.train(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)

class TestAdvancedEnsemble:
    @pytest.fixture
    def base_estimators(self):
        lr = LogisticRegression(random_state=42)
        rf = RandomForestClassifier(random_state=42)
        return [('lr', lr), ('rf', rf)]

    def test_voting_model(self, classification_data, base_estimators):
        X, y = classification_data
        model = VotingChurnModel(estimators=base_estimators)
        model.train(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_stacking_model(self, classification_data, base_estimators):
        X, y = classification_data
        model = StackingChurnModel(estimators=base_estimators)
        model.train(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)

class TestHyperparameterTuner:
    def test_grid_search_cv(self, classification_data):
        X, y = classification_data
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [10, 20], 'max_depth': [2, 3]}
        tuner = HyperparameterTuner(model, param_grid)
        best_model, best_params = tuner.grid_search_cv(X, y)
        
        assert best_model is not None
        assert 'n_estimators' in best_params
        assert best_params['n_estimators'] in [10, 20]

    def test_random_search_cv(self, classification_data):
        X, y = classification_data
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [10, 20], 'max_depth': [2, 3]}
        tuner = HyperparameterTuner(model, param_grid)
        best_model, best_params = tuner.random_search_cv(X, y, n_iter=2)
        
        assert best_model is not None
        assert 'n_estimators' in best_params
