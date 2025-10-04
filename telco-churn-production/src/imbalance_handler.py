import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from src.config.config import RANDOM_STATE

class ImbalanceHandler:
    """Provides methods for handling imbalanced datasets."""

    def __init__(self, random_state=RANDOM_STATE):
        """
        Initializes the ImbalanceHandler.

        Args:
            random_state (int): The random state for reproducibility.
        """
        self.random_state = random_state

    def apply_smote(self, X, y):
        """Applies the SMOTE over-sampling technique."""
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"Original dataset shape: {y.shape[0]}")
        print(f"Resampled dataset shape (SMOTE): {y_resampled.shape[0]}")
        return X_resampled, y_resampled

    def apply_adasyn(self, X, y):
        """Applies the ADASYN over-sampling technique."""
        adasyn = ADASYN(random_state=self.random_state)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        print(f"Original dataset shape: {y.shape[0]}")
        print(f"Resampled dataset shape (ADASYN): {y_resampled.shape[0]}")
        return X_resampled, y_resampled

    def apply_random_oversampling(self, X, y):
        """Applies random over-sampling."""
        ros = RandomOverSampler(random_state=self.random_state)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        print(f"Original dataset shape: {y.shape[0]}")
        print(f"Resampled dataset shape (Random Over-sampling): {y_resampled.shape[0]}")
        return X_resampled, y_resampled

    def apply_random_undersampling(self, X, y):
        """Applies random under-sampling."""
        rus = RandomUnderSampler(random_state=self.random_state)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        print(f"Original dataset shape: {y.shape[0]}")
        print(f"Resampled dataset shape (Random Under-sampling): {y_resampled.shape[0]}")
        return X_resampled, y_resampled

    def get_cost_sensitive_model(self, model_class, **kwargs):
        """
        Returns a model instance configured for cost-sensitive learning.
        This is achieved by setting class_weight='balanced'.
        
        Args:
            model_class: The scikit-learn model class (e.g., LogisticRegression).
            **kwargs: Additional arguments for the model's constructor.

        Returns:
            A model instance with class_weight='balanced'.
        """
        try:
            # Check if the model supports class_weight
            if 'class_weight' in model_class().get_params():
                return model_class(class_weight='balanced', random_state=self.random_state, **kwargs)
            else:
                print(f"Warning: {model_class.__name__} does not support 'class_weight'. Returning default model.")
                return model_class(random_state=self.random_state, **kwargs)
        except Exception as e:
            print(f"Error creating cost-sensitive model: {e}")
            return None

if __name__ == '__main__':
    # Create a dummy imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
                               n_redundant=10, n_clusters_per_class=1, weights=[0.9, 0.1],
                               flip_y=0, random_state=RANDOM_STATE)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)

    print(f"Original class distribution:\n{y.value_counts()}\n")

    handler = ImbalanceHandler()

    # --- Resampling Methods ---
    X_smote, y_smote = handler.apply_smote(X, y)
    print(f"SMOTE class distribution:\n{y_smote.value_counts()}\n")

    X_adasyn, y_adasyn = handler.apply_adasyn(X, y)
    print(f"ADASYN class distribution:\n{y_adasyn.value_counts()}\n")

    X_ros, y_ros = handler.apply_random_oversampling(X, y)
    print(f"Random Over-sampling class distribution:\n{y_ros.value_counts()}\n")

    X_rus, y_rus = handler.apply_random_undersampling(X, y)
    print(f"Random Under-sampling class distribution:\n{y_rus.value_counts()}\n")

    # --- Cost-Sensitive Learning ---
    print("--- Cost-Sensitive Learning Example ---")
    lr_cost_sensitive = handler.get_cost_sensitive_model(LogisticRegression)
    if lr_cost_sensitive:
        print(f"Created cost-sensitive model: {lr_cost_sensitive}")
        # This model can now be trained directly on the imbalanced data
        # lr_cost_sensitive.fit(X, y)
