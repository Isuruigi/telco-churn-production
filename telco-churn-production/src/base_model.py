from abc import ABC, abstractmethod
import joblib
import pandas as pd
from typing import Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class BaseChurnModel(ABC):
    """Abstract base class for churn prediction models."""

    def __init__(self, model):
        """
        Initializes the BaseChurnModel.

        Args:
            model: A scikit-learn compatible model instance.
        """
        self.model = model

    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """Trains the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Makes predictions on new data."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Makes probability predictions on new data."""
        pass

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluates the model and returns a dictionary of metrics."""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1] # Probability of positive class

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        return metrics

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Returns feature importances as a DataFrame."""
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            importances = self.model.feature_importances_
            feature_names = self.model.feature_names_in_
            return pd.DataFrame({'feature': feature_names, 'importance': importances})
        elif hasattr(self.model, 'coef_'):
            # For linear models
            importances = self.model.coef_[0]
            feature_names = self.model.feature_names_in_
            return pd.DataFrame({'feature': feature_names, 'importance': importances})
        else:
            print("Model does not have feature_importances_ or coef_ attribute.")
            return None

    def save_model(self, file_path: str):
        """Saves the trained model to a file."""
        try:
            joblib.dump(self.model, file_path)
            print(f"Model saved to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    @classmethod
    def load_model(cls, file_path: str):
        """Loads a trained model from a file."""
        try:
            model = joblib.load(file_path)
            print(f"Model loaded from {file_path}")
            # We need to return an instance of a concrete class, not the base class.
            # This method is better implemented in the concrete classes.
            # For now, we'll just return the sklearn model itself.
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

# Example of a concrete implementation
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(BaseChurnModel):
    """A churn prediction model using Logistic Regression."""

    def __init__(self, **kwargs):
        super().__init__(LogisticRegression(**kwargs))

    def train(self, X_train, y_train, **kwargs):
        """Trains the logistic regression model."""
        self.model.fit(X_train, y_train, **kwargs)
        print("Logistic Regression model trained.")

    def predict(self, X):
        """Makes predictions using the trained model."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Makes probability predictions."""
        return self.model.predict_proba(X)

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    # Create dummy data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=0, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Usage of the concrete model class ---
    # 1. Initialize the model
    lr_model = LogisticRegressionModel(random_state=42)

    # 2. Train the model
    lr_model.train(X_train, y_train)

    # 3. Evaluate the model
    metrics = lr_model.evaluate(X_test, y_test)
    print(f"\nEvaluation metrics:\n{metrics}")

    # 4. Get feature importance
    feature_importance = lr_model.get_feature_importance()
    if feature_importance is not None:
        print("\nFeature Importances (Top 5):")
        print(feature_importance.sort_values(by='importance', ascending=False).head())

    # 5. Save the model
    model_path = './logistic_regression_model.joblib'
    lr_model.save_model(model_path)

    # 6. Load the model
    loaded_sklearn_model = LogisticRegressionModel.load_model(model_path)
    if loaded_sklearn_model:
        # To use the loaded model with our class structure, we'd re-instantiate the class
        loaded_lr_model = LogisticRegressionModel()
        loaded_lr_model.model = loaded_sklearn_model
        print("\nPredictions with loaded model:")
        print(loaded_lr_model.predict(X_test[:5]))
