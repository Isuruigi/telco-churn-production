import joblib
import os
from sklearn.pipeline import Pipeline
from src.data_pipeline import DataPipeline
from src.base_model import LogisticRegressionModel
from src.ensemble_models import RandomForestChurnModel, XGBoostChurnModel
from src.hyperparameter_tuner import HyperparameterTuner
from src.model_evaluator import ModelEvaluator
from src.config.config import DATA_PATH, MODELS_PATH, RANDOM_STATE

class TrainingPipeline:
    """Orchestrates the end-to-end model training workflow."""

    def __init__(self, data_pipeline: DataPipeline):
        """
        Initializes the TrainingPipeline.

        Args:
            data_pipeline (DataPipeline): An instance of the data pipeline.
        """
        self.data_pipeline = data_pipeline
        self.models = {}
        self.best_model = None
        self.preprocessor = None

    def train_baseline_models(self, X_train, y_train):
        """Trains baseline models."""
        print("--- Training Baseline Models ---")
        lr = LogisticRegressionModel(random_state=RANDOM_STATE)
        lr.train(X_train, y_train)
        self.models['Logistic Regression'] = lr.model
        print("--- Baseline Models Trained ---\n")

    def train_ensemble_models(self, X_train, y_train):
        """Trains ensemble models."""
        print("--- Training Ensemble Models ---")
        rf = RandomForestChurnModel(random_state=RANDOM_STATE)
        rf.train(X_train, y_train)
        self.models['Random Forest'] = rf.model

        xgb = XGBoostChurnModel(random_state=RANDOM_STATE)
        xgb.train(X_train, y_train)
        self.models['XGBoost'] = xgb.model
        print("--- Ensemble Models Trained ---
")

    def tune_hyperparameters(self, X_train, y_train):
        """Tunes hyperparameters for a selected model."""
        print("--- Tuning Hyperparameters for RandomForest ---")
        rf_for_tuning = RandomForestChurnModel(random_state=RANDOM_STATE).model
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_leaf': [1, 2]
        }
        tuner = HyperparameterTuner(rf_for_tuning, param_grid)
        best_model, _ = tuner.grid_search_cv(X_train, y_train)
        self.models['Tuned RF'] = best_model
        print("--- Hyperparameter Tuning Complete ---\n")

    def evaluate_models(self, X_test, y_test) -> pd.DataFrame:
        """Evaluates all trained models."""
        print("--- Evaluating Models ---")
        evaluator = ModelEvaluator(self.models, X_test, y_test)
        comparison_df = evaluator.compare_multiple_models()
        evaluator.create_evaluation_plots()
        print("--- Model Evaluation Complete ---\n")
        return comparison_df

    def select_best_model(self, comparison_df: pd.DataFrame, metric='ROC_AUC'):
        """Selects the best model based on a given metric."""
        print(f"--- Selecting Best Model based on {metric} ---")
        best_model_name = comparison_df[metric].idxmax()
        self.best_model = self.models[best_model_name]
        print(f"Best model selected: {best_model_name}")
        print("--- Best Model Selected ---\n")

    def save_pipeline(self, file_name='final_pipeline.joblib'):
        """Saves the full pipeline (preprocessor + best model)."""
        if self.best_model is None or self.preprocessor is None:
            raise RuntimeError("Best model or preprocessor not available. Run the pipeline first.")

        print("--- Saving Final Pipeline ---")
        full_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', self.best_model)
        ])
        
        save_path = os.path.join(MODELS_PATH, file_name)
        joblib.dump(full_pipeline, save_path)
        print(f"Final pipeline saved to {save_path}")
        print("--- Pipeline Saved ---\n")

    def run(self):
        """Runs the full training pipeline."""
        # 1. Get data from the data pipeline
        X_train, X_test, y_train, y_test, self.preprocessor = self.data_pipeline.run(handle_imbalance_method='smote')
        
        # Convert sparse matrix to dense if necessary (e.g., for some models)
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
            X_test = X_test.toarray()

        # 2. Train models
        self.train_baseline_models(X_train, y_train)
        self.train_ensemble_models(X_train, y_train)

        # 3. Tune hyperparameters
        self.tune_hyperparameters(X_train, y_train)

        # 4. Evaluate models
        comparison_report = self.evaluate_models(X_test, y_test)
        print("--- Model Comparison Report ---")
        print(comparison_report)
        print("--- End Report ---\n")

        # 5. Select best model
        self.select_best_model(comparison_report)

        # 6. Save the final pipeline
        self.save_pipeline()

if __name__ == '__main__':
    # Initialize the data pipeline
    data_pipe = DataPipeline(data_path=DATA_PATH)
    
    # Initialize and run the training pipeline
    training_pipe = TrainingPipeline(data_pipe)
    training_pipe.run()
