import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

from src.modeling.sklearn_pipeline import ChurnPipeline

def main():
    # Define paths
    data_path = os.path.join(project_root, "data", "telco_churn.csv")
    models_path = os.path.join(project_root, "models")

    # Load data
    df = pd.read_csv(data_path)

    # Convert TotalCharges to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows with missing values for simplicity in this training script
    df.dropna(inplace=True)

    # Define target and features
    X = df.drop("Churn", axis=1)
    y = df["Churn"].apply(lambda x: 1 if x == 'Yes' else 0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the pipeline
    churn_pipeline = ChurnPipeline(model_type='random_forest')
    churn_pipeline.train(X_train, y_train)

    # Evaluate the model
    churn_pipeline.evaluate(X_test, y_test)

    # Save the model
    churn_pipeline.save_model(models_path)

if __name__ == "__main__":
    main()
