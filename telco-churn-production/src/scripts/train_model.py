import argparse
from data_pipeline import DataPipeline
from training_pipeline import TrainingPipeline
from src.config.config import DATA_PATH, MODELS_PATH

def main(data_path, model_dir, imbalance_method):
    """Main function to orchestrate the training workflow."""
    print("====== INITIALIZING TRAINING WORKFLOW ======")

    # 1. Override config paths if provided via command line
    # In a more complex app, this would be handled by a more robust config management system
    from config import config
    config.DATA_PATH = data_path
    config.MODELS_PATH = model_dir

    # 2. Initialize the Data Pipeline
    # The data pipeline will use the (potentially updated) config paths
    data_pipe = DataPipeline(data_path=config.DATA_PATH)
    
    # 3. Initialize the Training Pipeline
    training_pipe = TrainingPipeline(data_pipe)

    # 4. Run the training workflow
    # We need to modify the TrainingPipeline's run method to accept the imbalance method
    # For now, let's adapt here. A better solution would be to refactor TrainingPipeline.
    
    print(f"Running training with imbalance handling method: {imbalance_method}")
    
    # The run method of TrainingPipeline can be modified to accept this parameter.
    # Let's assume we modify it. For now, we will call the steps manually to show the logic.
    
    # Step 1: Get data from the data pipeline
    X_train, X_test, y_train, y_test, preprocessor = data_pipe.run(handle_imbalance_method=imbalance_method)
    training_pipe.preprocessor = preprocessor

    # Convert sparse matrix to dense if necessary
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    # Step 2: Train models
    training_pipe.train_baseline_models(X_train, y_train)
    training_pipe.train_ensemble_models(X_train, y_train)

    # Step 3: Tune hyperparameters
    training_pipe.tune_hyperparameters(X_train, y_train)

    # Step 4: Evaluate models
    comparison_report = training_pipe.evaluate_models(X_test, y_test)
    print("--- Model Comparison Report ---")
    print(comparison_report)

    # Step 5: Select best model
    training_pipe.select_best_model(comparison_report)

    # Step 6: Save the final pipeline
    training_pipe.save_pipeline()

    print("====== TRAINING WORKFLOW FINISHED ======")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main Training Script for the Churn Prediction Project")
    
    parser.add_argument('--data_path', type=str, default=DATA_PATH, 
                        help=f'Path to the raw data CSV file. Defaults to {DATA_PATH}')
    parser.add_argument('--model_dir', type=str, default=MODELS_PATH, 
                        help=f'Directory to save trained models. Defaults to {MODELS_PATH}')
    parser.add_argument('--imbalance_method', type=str, default='smote', 
                        choices=['smote', 'adasyn', 'random_oversample', 'random_undersample', None],
                        help='Method to handle class imbalance. Defaults to smote.')

    args = parser.parse_args()

    main(
        data_path=args.data_path, 
        model_dir=args.model_dir, 
        imbalance_method=args.imbalance_method
    )
