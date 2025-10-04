import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.config.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, RANDOM_STATE

class ChurnPipelineProduction:
    """Manages a reproducible scikit-learn pipeline for the churn project."""

    def __init__(self, model, numerical_features=NUMERICAL_FEATURES, categorical_features=CATEGORICAL_FEATURES):
        """
        Initializes the production pipeline manager.

        Args:
            model: A scikit-learn compatible model instance.
            numerical_features (list): List of numerical feature names.
            categorical_features (list): List of categorical feature names.
        """
        self.model = model
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.pipeline = None

    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Creates the preprocessing part of the pipeline."""
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop' # In production, we only want to use specified features
        )
        return preprocessor

    def create_model_pipeline(self) -> Pipeline:
        """Creates the full scikit-learn pipeline with preprocessing and model."""
        preprocessor = self.create_preprocessing_pipeline()
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.model)
        ])
        return self.pipeline

    def train_with_validation(self, X_train, y_train, X_val=None, y_val=None, fit_params=None):
        """
        Trains the pipeline. Optionally uses a validation set for models that support it.
        """
        if self.pipeline is None:
            self.create_model_pipeline()
        
        print("--- Training the production pipeline... ---")
        # If the model supports early stopping, validation data can be passed via fit_params
        if fit_params and X_val is not None and y_val is not None:
            self.pipeline.fit(X_train, y_train, **fit_params)
        else:
            self.pipeline.fit(X_train, y_train)
        print("--- Pipeline training complete. ---")

    def evaluate_model(self, X_test, y_test) -> dict:
        """Evaluates the trained pipeline on the test set."""
        if self.pipeline is None or not hasattr(self.pipeline, 'classes_'):
            raise RuntimeError("Pipeline has not been trained yet. Call train_with_validation() first.")
        
        print("--- Evaluating pipeline on test data... ---")
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))
        return report

    def save_pipeline_artifacts(self, file_path: str):
        """Saves the trained pipeline object to a file."""
        if self.pipeline is None:
            raise RuntimeError("No pipeline to save. Please create and train the pipeline first.")
        
        print(f"--- Saving pipeline artifacts to {file_path}... ---")
        joblib.dump(self.pipeline, file_path)
        print("--- Artifacts saved successfully. ---")

    @staticmethod
    def load_pipeline_artifacts(file_path: str) -> Pipeline:
        """Loads a trained pipeline from a file."""
        try:
            pipeline = joblib.load(file_path)
            print(f"Pipeline loaded successfully from {file_path}")
            return pipeline
        except FileNotFoundError:
            print(f"Error: Pipeline file not found at {file_path}")
            return None

if __name__ == '__main__':
    from src.data_loader import TelcoDataLoader
    from src.preprocessor import DataPreprocessor

    # 1. Load and prepare data
    df = TelcoDataLoader().load_raw_data()
    df_processed = DataPreprocessor.preprocess_data(df)
    df_processed.dropna(inplace=True) # Simple handling for demo

    X = df_processed.drop(TARGET_COLUMN, axis=1)
    y = df_processed[TARGET_COLUMN].apply(lambda x: 1 if x == 'Yes' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # 2. Initialize the production pipeline with a model
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=150, max_depth=10)
    prod_pipeline = ChurnPipelineProduction(model=rf_model)

    # 3. Create the full pipeline object
    prod_pipeline.create_model_pipeline()

    # 4. Train the pipeline
    prod_pipeline.train_with_validation(X_train, y_train)

    # 5. Evaluate the model
    evaluation_metrics = prod_pipeline.evaluate_model(X_test, y_test)

    # 6. Save the final pipeline
    pipeline_path = "./production_churn_pipeline.joblib"
    prod_pipeline.save_pipeline_artifacts(pipeline_path)

    # 7. Load the pipeline and make a prediction
    loaded_pipeline = ChurnPipelineProduction.load_pipeline_artifacts(pipeline_path)
    if loaded_pipeline:
        sample = X_test.iloc[[0]]
        prediction = loaded_pipeline.predict(sample)
        print(f"\n--- Sample Prediction with Loaded Pipeline ---")
        print(f"Sample Data: {sample.to_dict('records')[0]}")
        print(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")
