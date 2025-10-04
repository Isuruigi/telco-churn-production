import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from src.base_model import BaseChurnModel
from src.config.config import RANDOM_STATE

class RandomForestChurnModel(BaseChurnModel):
    """A churn prediction model using Random Forest."""

    def __init__(self, **kwargs):
        super().__init__(RandomForestClassifier(random_state=RANDOM_STATE, **kwargs))

    def train(self, X_train, y_train, tune_hyperparameters=False, param_grid=None):
        """Trains the Random Forest model, with optional hyperparameter tuning."""
        if tune_hyperparameters:
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_leaf': [1, 2, 4]
                }
            grid_search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best Random Forest parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X_train, y_train)
        print("Random Forest model trained.")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class XGBoostChurnModel(BaseChurnModel):
    """A churn prediction model using XGBoost."""

    def __init__(self, **kwargs):
        super().__init__(XGBClassifier(random_state=RANDOM_STATE, **kwargs))

    def train(self, X_train, y_train, tune_hyperparameters=False, param_grid=None):
        """Trains the XGBoost model, with optional hyperparameter tuning."""
        if tune_hyperparameters:
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7]
                }
            grid_search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best XGBoost parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X_train, y_train)
        print("XGBoost model trained.")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class CatBoostChurnModel(BaseChurnModel):
    """A churn prediction model using CatBoost."""

    def __init__(self, **kwargs):
        super().__init__(CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, **kwargs))

    def train(self, X_train, y_train, tune_hyperparameters=False, param_grid=None):
        """Trains the CatBoost model, with optional hyperparameter tuning."""
        if tune_hyperparameters:
            if param_grid is None:
                param_grid = {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'depth': [3, 5, 7]
                }
            grid_search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best CatBoost parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X_train, y_train)
        print("CatBoost model trained.")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    # Create dummy data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=0, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Random Forest Example ---
    print("--- Training Random Forest ---")
    rf_model = RandomForestChurnModel()
    # Set tune_hyperparameters=True to run GridSearchCV
    rf_model.train(X_train, y_train, tune_hyperparameters=False)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    print(f"Random Forest Metrics: {rf_metrics}")

    # --- XGBoost Example ---
    print("\n--- Training XGBoost ---")
    xgb_model = XGBoostChurnModel()
    xgb_model.train(X_train, y_train, tune_hyperparameters=False)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    print(f"XGBoost Metrics: {xgb_metrics}")

    # --- CatBoost Example ---
    print("\n--- Training CatBoost ---")
    cat_model = CatBoostChurnModel()
    cat_model.train(X_train, y_train, tune_hyperparameters=False)
    cat_metrics = cat_model.evaluate(X_test, y_test)
    print(f"CatBoost Metrics: {cat_metrics}")
