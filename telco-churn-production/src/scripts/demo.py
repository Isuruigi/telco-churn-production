import os
import json
from pipeline_orchestrator import PipelineOrchestrator
from report_generator import ReportGenerator
from model_evaluator import ModelEvaluator # For loading comparison data
from business_evaluator import BusinessEvaluator # For loading biz impact data
from src.config.config import DATA_PATH, MODELS_PATH, REPORTS_PATH

def run_demo():
    """Runs a complete demonstration of the project's capabilities."""

    print("=================================================")
    print("    TELCO CUSTOMER CHURN PREDICTION DEMO     ")
    print("=================================================")

    orchestrator = PipelineOrchestrator()

    # --- 1. Pipeline Health Check ---
    print("\nSTEP 1: Performing Pipeline Health Check...")
    # This will check for data, but the model may not exist yet.
    orchestrator.validate_pipeline_health()

    # --- 2. Run Full Training Pipeline ---
    print("\nSTEP 2: Running the Full Training Pipeline...")
    print("(This will load data, preprocess, train models, tune, evaluate, and save the best pipeline)...")
    orchestrator.run_training_only()
    print("\nFull Training Pipeline has been executed.")
    print(f"The best model pipeline has been saved to: {orchestrator.final_model_path}")
    print(f"Evaluation plots have been saved to: {os.path.join(REPORTS_PATH, 'figures')}")

    # --- 3. Run Inference on a Sample Customer ---
    print("\nSTEP 3: Running Inference on a Sample Customer...")
    sample_customer = {
        'gender': 'Female', 'SeniorCitizen': 1, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No',
        'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.70, 'TotalCharges': '151.65'
    }
    orchestrator.run_inference_only(sample_customer)

    # --- 4. Generate a Project Report ---
    # In a real run, we would load the actual artifacts. Here, we simulate it for the demo.
    print("\nSTEP 4: Generating a Final Project Report...")
    try:
        # This part is for demonstration; it re-creates objects that the training pipeline would have created.
        # A more robust implementation would save these artifacts to disk and load them here.
        report_gen = ReportGenerator()
        
        # Dummy data for report generation - in a real scenario, you'd load these from the training run
        best_model_name = "Tuned RF" # Assuming this was the best model
        key_metrics = {'ROC_AUC': 0.85, 'precision': 0.8, 'recall': 0.9} # Dummy metrics
        biz_impact = {
            'net_impact': 50000,
            'estimated_roi': 1.5,
            'customers_targeted': 200,
            'estimated_customers_saved': 80,
            'total_campaign_cost': 10000,
            'estimated_revenue_saved': 60000
        }
        comparison_data = {
            'Model': ['Logistic Regression', 'Random Forest', 'Tuned RF'],
            'ROC_AUC': [0.82, 0.84, 0.85],
            'f1-score': [0.78, 0.81, 0.82]
        }
        model_comparison = pd.DataFrame(comparison_data).set_index('Model')

        report_gen.generate_full_report(
            best_model_name=best_model_name,
            key_metrics=key_metrics,
            business_impact=biz_impact,
            comparison_df=model_comparison
        )
        print(f"Report saved to: {os.path.join(REPORTS_PATH, 'project_report.md')}")
    except Exception as e:
        print(f"Could not generate report. Error: {e}")

    print("\n=================================================")
    print("                 DEMO COMPLETE                 ")
    print("=================================================")

if __name__ == '__main__':
    run_demo()
