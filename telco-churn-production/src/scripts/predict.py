import argparse
import json
import os
import pandas as pd
from inference_pipeline import InferencePipeline
from data_loader import TelcoDataLoader # Needed for training data sample for SHAP
from preprocessor import DataPreprocessor
from sklearn.model_selection import train_test_split
from src.config.config import MODELS_PATH, TARGET_COLUMN, RANDOM_STATE

def main(model_path, input_data_str, explain):
    """Main function to orchestrate the inference workflow."""
    print("====== INITIALIZING INFERENCE WORKFLOW ======")

    # 1. Validate model path
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please provide a valid path.")
        return

    # 2. Parse input data
    try:
        input_data = json.loads(input_data_str)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format for input data.")
        return

    # 3. Initialize the Inference Pipeline
    inference_pipe = InferencePipeline(model_path=model_path)

    # 4. Make prediction
    prediction, confidence = inference_pipe.predict_single(input_data)
    
    output = {
        'prediction': 'Churn' if prediction == 1 else 'No Churn',
        'confidence': f'{confidence:.2%}'
    }
    
    print("\n--- Prediction Result ---")
    print(json.dumps(output, indent=2))

    # 5. Generate explanations if requested
    if explain:
        print("\n--- Generating Prediction Explanation ---")
        # SHAP requires a background dataset to compute expectations.
        # We load the original data and take a sample from the training set.
        print("(Loading background data for SHAP explainer...)")
        df = TelcoDataLoader().load_raw_data()
        df_processed = DataPreprocessor.preprocess_data(df.copy())
        X = df_processed.drop(TARGET_COLUMN, axis=1)
        y = df_processed[TARGET_COLUMN]
        X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        
        # Generate and display the explanation
        inference_pipe.generate_explanations(input_data, X_train.head(100))

    print("\n====== INFERENCE WORKFLOW FINISHED ======")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main Prediction Script for the Churn Prediction Project")

    default_model_path = os.path.join(MODELS_PATH, 'final_pipeline.joblib')

    parser.add_argument('--model_path', type=str, default=default_model_path,
                        help=f'Path to the trained model pipeline. Defaults to {default_model_path}')
    
    # Example customer data as a JSON string
    example_customer = {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
        'tenure': 1, 'PhoneService': 'No', 'MultipleLines': 'No phone service', 'InternetService': 'DSL',
        'OnlineSecurity': 'No', 'OnlineBackup': 'Yes', 'DeviceProtection': 'No',
        'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85, 'TotalCharges': '29.85'
    }

    parser.add_argument('--input_data', type=str, default=json.dumps(example_customer),
                        help='New customer data for prediction, as a JSON string.')

    parser.add_argument('--explain', action='store_true',
                        help='Generate and display a SHAP explanation for the prediction.')

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        input_data_str=args.input_data,
        explain=args.explain
    )
