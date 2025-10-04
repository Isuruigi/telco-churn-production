import os
import joblib
import json
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ChurnPipeline:
    def __init__(self, model_type='logistic_regression', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = None
        self.metadata = {}
        self.grid_search = None

        self.categorical_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        self.numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        if self.model_type == 'logistic_regression':
            self.classifier = LogisticRegression(random_state=self.random_state, solver='liblinear')
            self.param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l1', 'l2']
            }
        elif self.model_type == 'random_forest':
            self.classifier = RandomForestClassifier(random_state=self.random_state)
            self.param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5]
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.classifier)
        ])

    def train(self, X: pd.DataFrame, y: pd.Series, cv=5, scoring='accuracy'):
        try:
            print(f"Training {self.model_type} with GridSearchCV...")
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
            X.fillna(X.mean(numeric_only=True), inplace=True)

            self.grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=cv, scoring=scoring, n_jobs=-1)
            self.grid_search.fit(X, y)

            self.pipeline = self.grid_search.best_estimator_
            print("GridSearchCV training complete.")
            print(f"Best parameters: {self.grid_search.best_params_}")

            print("Performing cross-validation...")
            cv_results = cross_validate(self.pipeline, X, y, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'])
            self.metadata['cross_validation_scores'] = {k: v.tolist() for k, v in cv_results.items()}
            print("Cross-validation complete.")

            self.metadata['model_type'] = self.model_type
            self.metadata['training_timestamp'] = datetime.now().isoformat()
            self.metadata['best_params'] = self.grid_search.best_params_

        except Exception as e:
            print(f"An error occurred during training: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.pipeline:
            raise RuntimeError("Pipeline is not trained. Please train the model before making predictions.")
        try:
            print("Making predictions...")
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
            X.fillna(X.mean(numeric_only=True), inplace=True)
            return self.pipeline.predict(X)
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            raise

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> dict:
        try:
            print("Evaluating model...")
            y_pred = self.predict(X)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
            self.metadata['evaluation_metrics'] = metrics

            print(f"\n--- Model Evaluation ({self.model_type}) ---")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"--- End Model Evaluation ---")
            return metrics
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            raise

    def save_model(self, base_path: str, version: str = None):
        if not self.pipeline:
            print("Error: No pipeline to save. Please train the pipeline first.")
            return

        if not version:
            version = datetime.now().strftime("%Y%m%d-%H%M%S")

        model_dir_name = f"{self.model_type}_{version}"
        full_model_path = os.path.join(base_path, model_dir_name)
        os.makedirs(full_model_path, exist_ok=True)

        pipeline_file = os.path.join(full_model_path, "pipeline.joblib")
        metadata_file = os.path.join(full_model_path, "metadata.json")

        try:
            joblib.dump(self.pipeline, pipeline_file)
            print(f"Pipeline successfully saved to {pipeline_file}")

            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=4)
            print(f"Metadata successfully saved to {metadata_file}")

        except Exception as e:
            print(f"Error saving pipeline or metadata: {e}")
            raise

    def load_model(self, version_path: str):
        pipeline_file = os.path.join(version_path, "pipeline.joblib")
        metadata_file = os.path.join(version_path, "metadata.json")

        try:
            self.pipeline = joblib.load(pipeline_file)
            print(f"Pipeline successfully loaded from {pipeline_file}")

            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"Metadata successfully loaded from {metadata_file}")

            self.model_type = self.metadata.get('model_type', self.model_type)

        except FileNotFoundError:
            print(f"Error: Model or metadata file not found in {version_path}")
            self.pipeline = None
            self.metadata = {}
            raise
        except Exception as e:
            print(f"Error loading pipeline or metadata: {e}")
            self.pipeline = None
            self.metadata = {}
            raise

    def get_feature_importance(self):
        if not self.pipeline:
            raise RuntimeError("Pipeline is not trained. Please train the model first.")

        if self.model_type == 'logistic_regression':
            importances = self.pipeline.named_steps['classifier'].coef_[0]
        elif self.model_type == 'random_forest':
            importances = self.pipeline.named_steps['classifier'].feature_importances_
        else:
            print("Feature importance not supported for this model type.")
            return None

        onehot_features = self.pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(self.categorical_features)
        feature_names = self.numerical_features + list(onehot_features)

        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance = feature_importance.sort_values(by='importance', ascending=False)
        return feature_importance