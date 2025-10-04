import argparse
import joblib
import pandas as pd
import os
from model_evaluator import ModelEvaluator
from data_loader import TelcoDataLoader
from src.config.config import TARGET_COLUMN

def main(model_paths, test_data_path, metric):
    """Main function to load, compare, and evaluate models."""
    print("====== INITIALIZING MODEL COMPARISON WORKFLOW ======")

    # 1. Load Test Data
    print(f"Loading test data from: {test_data_path}")
    # In a real scenario, you would use a pre-split test set.
    # For this script, we'll load the full dataset and use the preprocessor from the first pipeline.
    df = TelcoDataLoader(data_path=test_data_path).load_raw_data()
    if df is None:
        return

    # 2. Load Models
    models_dict = {}
    preprocessor = None
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Warning: Model file not found at {path}. Skipping.")
            continue
        try:
            pipeline = joblib.load(path)
            model_name = os.path.splitext(os.path.basename(path))[0]
            models_dict[model_name] = pipeline
            print(f"Loaded model '{model_name}' from {path}")
            # Use the preprocessor from the first loaded pipeline
            if preprocessor is None:
                preprocessor = pipeline.steps[0][1]
        except Exception as e:
            print(f"Error loading model from {path}: {e}")

    if not models_dict:
        print("No models were loaded. Exiting.")
        return

    # 3. Prepare Test Data
    # The loaded pipelines contain both preprocessor and model.
    # We can use the pipeline directly for prediction.
    # For ModelEvaluator, we need to pass the transformed data.
    X_test = df.drop(TARGET_COLUMN, axis=1)
    y_test = df[TARGET_COLUMN].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # We need to get the model instance from each pipeline for the evaluator
    model_instances = {name: pipe.steps[1][1] for name, pipe in models_dict.items()}
    
    # Transform the test data using one of the preprocessors
    X_test_transformed = preprocessor.transform(X_test)

    # 4. Evaluate and Compare Models
    print("\n--- Evaluating and Comparing Models ---")
    evaluator = ModelEvaluator(model_instances, X_test_transformed, y_test)
    comparison_df = evaluator.compare_multiple_models()
    
    print("\n--- Model Performance Comparison ---")
    print(comparison_df)

    # 5. Recommend Best Model
    print(f"\n--- Recommending Best Model (based on {metric}) ---")
    try:
        best_model_name = comparison_df[metric].idxmax()
        print(f"Recommendation: The best model is '{best_model_name}' with a {metric} of {comparison_df.loc[best_model_name, metric]:.4f}.")
    except KeyError:
        print(f"Error: Metric '{metric}' not found in evaluation results. Please choose from {list(comparison_df.columns)}.")

    # 6. Generate Comparison Plots
    evaluator.create_evaluation_plots()
    print("\nComparison plots have been saved to the reports/figures/visualization directory.")

    print("\n====== MODEL COMPARISON WORKFLOW FINISHED ======")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Comparison Script for the Churn Prediction Project")

    # This assumes you have multiple models saved, e.g., final_pipeline.joblib, logistic_regression.joblib, etc.
    parser.add_argument('--model_paths', nargs='+', required=True,
                        help='A list of paths to the trained model pipeline files to compare.')
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='Path to the test data CSV file for evaluation.')
    parser.add_argument('--metric', type=str, default='ROC_AUC',
                        help='The metric to use for recommending the best model.')

    args = parser.parse_args()

    main(
        model_paths=args.model_paths,
        test_data_path=args.test_data_path,
        metric=args.metric
    )
    
    # Example Usage from terminal:
    # python compare_models.py --model_paths models/final_pipeline.joblib models/another_model.joblib --test_data_path data/WA_Fn-UseC_-Telco-Customer-Churn.csv
