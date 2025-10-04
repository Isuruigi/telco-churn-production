import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from src.config.config import RANDOM_STATE

class CrossValidator:
    """Implements various cross-validation strategies for model evaluation."""

    def __init__(self, model, X, y):
        """
        Initializes the CrossValidator.

        Args:
            model: The machine learning model to evaluate.
            X (pd.DataFrame): The feature set.
            y (pd.Series): The target variable.
        """
        self.model = model
        self.X = X
        self.y = y

    def stratified_kfold_cv(self, n_splits=5, scoring='roc_auc') -> np.ndarray:
        """Performs Stratified K-Fold cross-validation."""
        print(f"--- Performing Stratified {n_splits}-Fold Cross-Validation ---")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring=scoring, n_jobs=-1)
        
        print(f"Scores for each fold: {scores}")
        print(f"Mean score: {scores.mean():.4f}")
        print(f"Standard deviation: {scores.std():.4f}")
        print("--- Finished Stratified K-Fold CV ---\n")
        return scores

    def time_series_cv(self, n_splits=5, scoring='roc_auc') -> np.ndarray:
        """
        Performs Time Series cross-validation.
        Note: This is for time-dependent data, not typically used for this dataset.
        """
        print(f"--- Performing Time Series {n_splits}-Fold Cross-Validation ---")
        cv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring=scoring, n_jobs=-1)
        
        print(f"Scores for each fold: {scores}")
        print(f"Mean score: {scores.mean():.4f}")
        print(f"Standard deviation: {scores.std():.4f}")
        print("--- Finished Time Series CV ---\n")
        return scores

    def custom_cv_strategy(self, cv_strategy, scoring='roc_auc') -> np.ndarray:
        """Performs cross-validation with a custom strategy."""
        print(f"--- Performing CV with custom strategy: {type(cv_strategy).__name__} ---")
        scores = cross_val_score(self.model, self.X, self.y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
        
        print(f"Scores for each fold: {scores}")
        print(f"Mean score: {scores.mean():.4f}")
        print(f"Standard deviation: {scores.std():.4f}")
        print("--- Finished Custom CV ---\n")
        return scores

    def validate_model_stability(self, n_splits=10, threshold=0.05):
        """
        Validates model stability by analyzing the variance of scores across folds.

        Args:
            n_splits (int): The number of folds to use.
            threshold (float): The threshold for score standard deviation to be considered stable.
        """
        print("--- Validating Model Stability ---")
        scores = self.stratified_kfold_cv(n_splits=n_splits)
        score_std = scores.std()

        if score_std < threshold:
            print(f"Model is stable. Standard deviation ({score_std:.4f}) is below the threshold ({threshold}).")
        else:
            print(f"Model may be unstable. Standard deviation ({score_std:.4f}) is above the threshold ({threshold}).")
        print("--- Finished Model Stability Validation ---")

if __name__ == '__main__':
    # Create dummy data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                               n_redundant=5, n_classes=2, weights=[0.8, 0.2], 
                               flip_y=0.05, random_state=RANDOM_STATE)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)

    # Initialize a model
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE)

    # Initialize the validator
    validator = CrossValidator(rf_model, X, y)

    # --- Run different CV strategies ---
    
    # 1. Stratified K-Fold (recommended for this type of problem)
    validator.stratified_kfold_cv()

    # 2. Time Series (for demonstration)
    validator.time_series_cv()

    # 3. Custom Strategy (e.g., simple K-Fold)
    simple_kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    validator.custom_cv_strategy(simple_kfold)

    # 4. Validate model stability
    validator.validate_model_stability()
