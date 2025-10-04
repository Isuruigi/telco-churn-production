import itertools
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.mlflow_manager import MLflowManager
from src.data_pipeline import DataPipeline
from src.sklearn_production_pipeline import ChurnPipelineProduction
from src.config.config import DATA_PATH

class ExperimentRunner:
    """Orchestrates multiple ML experiments with comprehensive MLflow logging."""

    def __init__(self, mlflow_manager: MLflowManager, data_pipeline: DataPipeline):
        """
        Initializes the ExperimentRunner.

        Args:
            mlflow_manager (MLflowManager): An instance of the MLflow manager.
            data_pipeline (DataPipeline): An instance of the data pipeline.
        """
        self.mlflow_manager = mlflow_manager
        self.data_pipeline = data_pipeline

    def run_hyperparameter_experiments(self, model_class, param_grid: dict, X_train, y_train, X_test, y_test):
        """Runs experiments for different hyperparameter combinations."""
        print("\n--- Running Hyperparameter Tuning Experiments ---")
        # Create all combinations of parameters
        keys, values = zip(*param_grid.items())
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            run_name = f"{model_class.__name__}_params_{sum(v) if all(isinstance(i, (int, float)) for i in v) else 'combo'}"
            
            with self.mlflow_manager.start_run(run_name=run_name):
                self.mlflow_manager.log_hyperparameters(params)
                
                model = model_class(**params)
                pipeline = ChurnPipelineProduction(model)
                pipeline.train_with_validation(X_train, y_train)
                metrics = pipeline.evaluate_model(X_test, y_test)
                
                self.mlflow_manager.log_evaluation_metrics(metrics['weighted avg'])
                self.mlflow_manager.log_evaluation_metrics({'accuracy': metrics['accuracy']})

    def run_algorithm_comparison(self, models_to_compare: dict, X_train, y_train, X_test, y_test):
        """Runs experiments comparing different algorithms."""
        print("\n--- Running Algorithm Comparison Experiments ---")
        for name, model in models_to_compare.items():
            with self.mlflow_manager.start_run(run_name=f"algo_comp_{name}"):
                self.mlflow_manager.log_hyperparameters(model.get_params())
                
                pipeline = ChurnPipelineProduction(model)
                pipeline.train_with_validation(X_train, y_train)
                metrics = pipeline.evaluate_model(X_test, y_test)
                
                self.mlflow_manager.log_evaluation_metrics(metrics['weighted avg'])
                self.mlflow_manager.log_evaluation_metrics({'accuracy': metrics['accuracy']})

    def run_feature_experiments(self, model, feature_sets: dict, X_train, y_train, X_test, y_test):
        """Runs experiments with different feature sets."""
        print("\n--- Running Feature Set Experiments ---")
        for name, features in feature_sets.items():
            with self.mlflow_manager.start_run(run_name=f"feature_exp_{name}"):
                self.mlflow_manager.log_hyperparameters({'feature_set': name})
                
                # Select the features for this run
                X_train_fs = X_train[features]
                X_test_fs = X_test[features]
                
                # We need to create a pipeline with the correct feature lists
                num_fs = [f for f in features if f in self.data_pipeline.preprocessor_builder.numerical_features]
                cat_fs = [f for f in features if f in self.data_pipeline.preprocessor_builder.categorical_features]
                
                pipeline = ChurnPipelineProduction(model, numerical_features=num_fs, categorical_features=cat_fs)
                pipeline.train_with_validation(X_train_fs, y_train)
                metrics = pipeline.evaluate_model(X_test_fs, y_test)
                
                self.mlflow_manager.log_evaluation_metrics(metrics['weighted avg'])
                self.mlflow_manager.log_evaluation_metrics({'accuracy': metrics['accuracy']})

    def track_all_experiments(self):
        """Orchestrates a full suite of experiments."""
        print("====== STARTING ALL EXPERIMENTS ======")
        # 1. Get data
        X_train, X_test, y_train, y_test, _ = self.data_pipeline.run()
        # For feature experiments, we need the data before the final transformation
        df = self.data_pipeline.load_and_validate()
        df_engineered = self.data_pipeline.preprocess_and_engineer_features(df)
        X_raw_train, X_raw_test, _, _ = self.data_pipeline.split_data(df_engineered)

        # 2. Define models and parameters for experiments
        models_for_comparison = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42)
        }
        rf_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
        feature_sets_to_try = {
            'numeric_only': self.data_pipeline.preprocessor_builder.numerical_features,
            'all_features': self.data_pipeline.preprocessor_builder.numerical_features + self.data_pipeline.preprocessor_builder.categorical_features
        }

        # 3. Run experiments
        self.run_algorithm_comparison(models_for_comparison, X_train, y_train, X_test, y_test)
        self.run_hyperparameter_experiments(RandomForestClassifier, rf_param_grid, X_train, y_train, X_test, y_test)
        # self.run_feature_experiments(RandomForestClassifier(random_state=42), feature_sets_to_try, X_raw_train, y_train, X_raw_test, y_test)

        print("\n====== ALL EXPERIMENTS FINISHED ======")
        print(f"Check the MLflow UI at {self.mlflow_manager.tracking_uri} under experiment '{self.mlflow_manager.experiment_name}'")

if __name__ == '__main__':
    # --- MLflow Configuration ---
    TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT = "ChurnExperimentSuite"

    # 1. Initialize manager and pipelines
    mlflow_manager = MLflowManager(tracking_uri=TRACKING_URI, experiment_name=EXPERIMENT)
    data_pipe = DataPipeline(data_path=DATA_PATH)

    # 2. Initialize and run the experiment runner
    experiment_runner = ExperimentRunner(mlflow_manager, data_pipe)
    experiment_runner.track_all_experiments()
