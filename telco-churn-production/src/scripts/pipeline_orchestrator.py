import os
import argparse
from data_pipeline import DataPipeline
from training_pipeline import TrainingPipeline
from inference_pipeline import InferencePipeline
from src.config.config import DATA_PATH, MODELS_PATH

class PipelineOrchestrator:
    """Manages and executes the entire ML workflow."""

    def __init__(self, data_path=DATA_PATH, model_dir=MODELS_PATH):
        """
        Initializes the PipelineOrchestrator.

        Args:
            data_path (str): Path to the raw data.
            model_dir (str): Directory where models are saved.
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.final_model_path = os.path.join(self.model_dir, 'final_pipeline.joblib')

    def run_complete_pipeline(self):
        """Runs the complete data and training pipeline."""
        print("====== STARTING COMPLETE PIPELINE RUN ======")
        # 1. Initialize Data Pipeline
        data_pipe = DataPipeline(data_path=self.data_path)
        
        # 2. Initialize and run Training Pipeline
        training_pipe = TrainingPipeline(data_pipe)
        training_pipe.run()
        print("====== COMPLETE PIPELINE RUN FINISHED ======")

    def run_training_only(self):
        """Alias for the complete pipeline run, focusing on the training aspect."""
        print("====== STARTING TRAINING-ONLY RUN ======")
        self.run_complete_pipeline()
        print("====== TRAINING-ONLY RUN FINISHED ======")

    def run_inference_only(self, sample_data: dict):
        """Runs the inference pipeline on a single sample."""
        print("====== STARTING INFERENCE-ONLY RUN ======")
        if not os.path.exists(self.final_model_path):
            print(f"Error: Model not found at {self.final_model_path}. Please run the training pipeline first.")
            return

        inference_pipe = InferencePipeline(model_path=self.final_model_path)
        prediction, confidence = inference_pipe.predict_single(sample_data)
        
        print(f"\n--- Inference Result ---")
        print(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
        print(f"Confidence: {confidence:.2%}")
        print("====== INFERENCE-ONLY RUN FINISHED ======")

    def validate_pipeline_health(self) -> bool:
        """Performs a basic health check on the pipeline components."""
        print("====== VALIDATING PIPELINE HEALTH ======")
        health_ok = True

        # 1. Check if data exists
        if not os.path.exists(self.data_path):
            print(f"[FAIL] Data file not found at: {self.data_path}")
            health_ok = False
        else:
            print(f"[OK] Data file found at: {self.data_path}")

        # 2. Check if a trained model exists
        if not os.path.exists(self.final_model_path):
            print(f"[FAIL] Trained model not found at: {self.final_model_path}")
            health_ok = False
        else:
            print(f"[OK] Trained model found at: {self.final_model_path}")
            # 3. Try to load the model and make a dummy prediction
            try:
                inference_pipe = InferencePipeline(model_path=self.final_model_path)
                # Create a full dummy record with all expected columns
                from data_loader import TelcoDataLoader
                df = TelcoDataLoader().load_raw_data()
                full_dummy_data = df.iloc[0].to_dict()
                inference_pipe.predict_single(full_dummy_data)
                print("[OK] Model loaded and dummy prediction successful.")
            except Exception as e:
                print(f"[FAIL] Failed to load or use the model: {e}")
                health_ok = False

        print(f"\nOverall Health Status: {'OK' if health_ok else 'FAIL'}")
        print("====== HEALTH VALIDATION FINISHED ======")
        return health_ok

def main():
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator for the Churn Prediction Project")
    parser.add_argument('--run', type=str, required=True, 
                        choices=['train', 'inference', 'health_check'],
                        help="The part of the pipeline to run.")
    
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator()

    if args.run == 'train':
        orchestrator.run_complete_pipeline()
    elif args.run == 'health_check':
        orchestrator.validate_pipeline_health()
    elif args.run == 'inference':
        # Example customer data for inference
        sample_customer = {
            'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': 10, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'DSL',
            'OnlineSecurity': 'Yes', 'OnlineBackup': 'No', 'DeviceProtection': 'No',
            'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
            'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Mailed check',
            'MonthlyCharges': 49.95, 'TotalCharges': '499.5'
        }
        orchestrator.run_inference_only(sample_customer)

if __name__ == '__main__':
    main()
