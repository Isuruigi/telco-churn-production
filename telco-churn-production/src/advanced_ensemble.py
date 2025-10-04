import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.base_model import BaseChurnModel
from src.config.config import RANDOM_STATE

class VotingChurnModel(BaseChurnModel):
    """An ensemble model using VotingClassifier."""

    def __init__(self, estimators, voting='soft', **kwargs):
        """
        Args:
            estimators (list): A list of (name, model) tuples for the base models.
            voting (str): 'soft' or 'hard'.
        """
        model = VotingClassifier(estimators=estimators, voting=voting, **kwargs)
        super().__init__(model)

    def train(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train)
        print("VotingClassifier model trained.")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class StackingChurnModel(BaseChurnModel):
    """An ensemble model using StackingClassifier."""

    def __init__(self, estimators, final_estimator=None, **kwargs):
        """
        Args:
            estimators (list): A list of (name, model) tuples for the base models.
            final_estimator: The meta-learner.
        """
        if final_estimator is None:
            final_estimator = LogisticRegression(random_state=RANDOM_STATE)
        
        model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, **kwargs)
        super().__init__(model)

    def train(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train)
        print("StackingClassifier model trained.")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class BlendingChurnModel(BaseChurnModel):
    """An ensemble model that implements blending manually."""

    def __init__(self, base_models, meta_model=None):
        """
        Args:
            base_models (list): A list of instantiated base models.
            meta_model: The meta-learner.
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model else LogisticRegression(random_state=RANDOM_STATE)
        # The `model` attribute is used by the base class for saving, but here we have multiple models.
        # We will handle saving/loading manually in this class.
        super().__init__(model=None) 

    def train(self, X_train, y_train, validation_split_size=0.3):
        """Trains the blending model."""
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=validation_split_size, random_state=RANDOM_STATE
        )

        # Train base models on the sub-training set and make predictions on the validation set
        meta_features = []
        for model in self.base_models:
            model.fit(X_train_sub, y_train_sub)
            val_preds = model.predict_proba(X_val)[:, 1]
            meta_features.append(pd.Series(val_preds, index=X_val.index))
        
        meta_X = pd.concat(meta_features, axis=1)
        meta_X.columns = [f'model_{i}_pred' for i in range(len(self.base_models))]

        # Train the meta-model on the validation predictions
        self.meta_model.fit(meta_X, y_val)
        print("Blending model trained.")

    def predict(self, X):
        # Get predictions from base models
        base_preds = []
        for model in self.base_models:
            preds = model.predict_proba(X)[:, 1]
            base_preds.append(pd.Series(preds, index=X.index))
        
        meta_features = pd.concat(base_preds, axis=1)
        meta_features.columns = [f'model_{i}_pred' for i in range(len(self.base_models))]

        # Make final prediction with the meta-model
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X):
        base_preds = []
        for model in self.base_models:
            preds = model.predict_proba(X)[:, 1]
            base_preds.append(pd.Series(preds, index=X.index))
        
        meta_features = pd.concat(base_preds, axis=1)
        meta_features.columns = [f'model_{i}_pred' for i in range(len(self.base_models))]
        
        return self.meta_model.predict_proba(meta_features)

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier

    # Create dummy data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=0, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Base Models ---
    lr = LogisticRegression(random_state=RANDOM_STATE)
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)

    estimators = [('lr', lr), ('rf', rf), ('dt', dt)]

    # --- Voting Classifier Example ---
    print("--- Training Voting Classifier ---")
    voting_model = VotingChurnModel(estimators=estimators)
    voting_model.train(X_train, y_train)
    voting_metrics = voting_model.evaluate(X_test, y_test)
    print(f"Voting Classifier Metrics: {voting_metrics}")

    # --- Stacking Classifier Example ---
    print("\n--- Training Stacking Classifier ---")
    stacking_model = StackingChurnModel(estimators=estimators)
    stacking_model.train(X_train, y_train)
    stacking_metrics = stacking_model.evaluate(X_test, y_test)
    print(f"Stacking Classifier Metrics: {stacking_metrics}")

    # --- Blending Classifier Example ---
    print("\n--- Training Blending Classifier ---")
    # Re-instantiate models for blending
    lr_blend = LogisticRegression(random_state=RANDOM_STATE)
    rf_blend = RandomForestClassifier(random_state=RANDOM_STATE)
    dt_blend = DecisionTreeClassifier(random_state=RANDOM_STATE)
    base_models_blend = [lr_blend, rf_blend, dt_blend]
    
    blending_model = BlendingChurnModel(base_models=base_models_blend)
    blending_model.train(X_train, y_train)
    blending_metrics = blending_model.evaluate(X_test, y_test)
    print(f"Blending Classifier Metrics: {blending_metrics}")
