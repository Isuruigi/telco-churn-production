import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import optuna
from optuna.integration import OptunaSearchCV
from src.config.config import RANDOM_STATE

class HyperparameterTuner:
    """A class to perform hyperparameter tuning for ML models."""

    def __init__(self, model, param_grid):
        """
        Initializes the HyperparameterTuner.

        Args:
            model: The machine learning model to tune.
            param_grid (dict): The hyperparameter grid or distribution.
        """
        self.model = model
        self.param_grid = param_grid

    def grid_search_cv(self, X_train, y_train, cv=3, scoring='roc_auc'):
        """Performs Grid Search Cross-Validation."""
        print("--- Starting Grid Search CV ---")
        grid_search = GridSearchCV(self.model, self.param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters (Grid Search): {grid_search.best_params_}")
        print(f"Best score (Grid Search): {grid_search.best_score_:.4f}")
        print("--- Finished Grid Search CV ---\
")
        return grid_search.best_estimator_, grid_search.best_params_

    def random_search_cv(self, X_train, y_train, n_iter=50, cv=3, scoring='roc_auc'):
        """Performs Randomized Search Cross-Validation."""
        print("--- Starting Randomized Search CV ---")
        random_search = RandomizedSearchCV(self.model, self.param_grid, n_iter=n_iter, cv=cv, 
                                           scoring=scoring, n_jobs=-1, verbose=1, random_state=RANDOM_STATE)
        random_search.fit(X_train, y_train)
        print(f"Best parameters (Random Search): {random_search.best_params_}")
        print(f"Best score (Random Search): {random_search.best_score_:.4f}")
        print("--- Finished Randomized Search CV ---\
")
        return random_search.best_estimator_, random_search.best_params_

    def bayesian_optimization(self, X_train, y_train, n_iter=50, cv=3, scoring='roc_auc'):
        """
        Performs Bayesian Optimization using scikit-optimize (skopt).
        Note: Requires `pip install scikit-optimize`
        """
        print("--- Starting Bayesian Optimization ---")
        try:
            from skopt import BayesSearchCV
            bayes_search = BayesSearchCV(self.model, self.param_grid, n_iter=n_iter, cv=cv, 
                                         scoring=scoring, n_jobs=-1, verbose=1, random_state=RANDOM_STATE)
            bayes_search.fit(X_train, y_train)
            print(f"Best parameters (Bayesian Opt): {bayes_search.best_params_}")
            print(f"Best score (Bayesian Opt): {bayes_search.best_score_:.4f}")
            print("--- Finished Bayesian Optimization ---\
")
            return bayes_search.best_estimator_, bayes_search.best_params_
        except ImportError:
            print("scikit-optimize is not installed. Please run `pip install scikit-optimize` to use this feature.")
            return None, None

    def optuna_optimization(self, X_train, y_train, n_trials=100, cv=3, scoring='roc_auc'):
        """Performs hyperparameter optimization using Optuna."""
        print("--- Starting Optuna Optimization ---")
        
        # OptunaSearchCV is a convenient wrapper
        optuna_search = OptunaSearchCV(self.model, self.param_grid, n_trials=n_trials, cv=cv, 
                                       scoring=scoring, n_jobs=-1, verbose=1, random_state=RANDOM_STATE)
        optuna_search.fit(X_train, y_train)

        print(f"Best parameters (Optuna): {optuna_search.best_params_}")
        print(f"Best score (Optuna): {optuna_search.best_score_:.4f}")
        print("--- Finished Optuna Optimization ---\
")
        return optuna_search.best_estimator_, optuna_search.best_params_

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    # Create dummy data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=RANDOM_STATE)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Define model and parameter grid for RandomForest
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize the tuner
    tuner = HyperparameterTuner(rf, param_grid_rf)

    # --- Run different tuning methods ---
    # 1. Grid Search (using a smaller grid for speed)
    small_grid = {'n_estimators': [100, 150], 'max_depth': [5, 10]}
    grid_tuner = HyperparameterTuner(rf, small_grid)
    best_model_grid, _ = grid_tuner.grid_search_cv(X_train, y_train)

    # 2. Random Search
    best_model_random, _ = tuner.random_search_cv(X_train, y_train, n_iter=20) # n_iter is small for demo

    # 3. Bayesian Optimization
    # Note: Requires scikit-optimize
    # For skopt, the param grid needs to be defined with specific types from skopt.space
    try:
        from skopt.space import Real, Categorical, Integer
        param_space_bayes = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(5, 25),
            'min_samples_leaf': Integer(1, 5),
            'min_samples_split': Integer(2, 10)
        }
        bayes_tuner = HyperparameterTuner(rf, param_space_bayes)
        best_model_bayes, _ = bayes_tuner.bayesian_optimization(X_train, y_train, n_iter=20)
    except ImportError:
        print("Skipping Bayesian optimization example.")

    # 4. Optuna Optimization
    # For Optuna, the param grid needs to be defined with specific types from optuna.distributions
    param_dist_optuna = {
        'n_estimators': optuna.distributions.IntDistribution(100, 500),
        'max_depth': optuna.distributions.IntDistribution(5, 25),
        'min_samples_leaf': optuna.distributions.IntDistribution(1, 5),
        'min_samples_split': optuna.distributions.IntDistribution(2, 10)
    }
    optuna_tuner = HyperparameterTuner(rf, param_dist_optuna)
    best_model_optuna, _ = optuna_tuner.optuna_optimization(X_train, y_train, n_trials=20)
