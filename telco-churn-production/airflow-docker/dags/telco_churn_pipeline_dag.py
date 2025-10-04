"""
Telco Churn Prediction - Airflow DAG
This DAG orchestrates the complete ML pipeline for churn prediction on a weekly schedule.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score


# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def data_preprocessing(**context):
    """
    Task 1: Data Preprocessing
    Loads raw CSV, handles TotalCharges column, and saves to processed folder.
    """
    print("="*60)
    print("TASK 1: DATA PREPROCESSING")
    print("="*60)

    # Load raw data
    raw_path = '/opt/airflow/data/raw/telco_churn.csv'
    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"Initial shape: {df.shape}")

    # Handle TotalCharges column - convert to numeric, coercing errors
    print("\nHandling TotalCharges column...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows with missing TotalCharges
    initial_rows = len(df)
    df = df.dropna(subset=['TotalCharges'])
    dropped_rows = initial_rows - len(df)
    print(f"Dropped {dropped_rows} rows with missing TotalCharges")

    # Drop customerID if exists
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("Dropped customerID column")

    # Save processed data
    processed_path = '/opt/airflow/data/processed/telco_churn_processed.csv'
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"\nProcessed data saved to: {processed_path}")
    print(f"Final shape: {df.shape}")
    print("="*60)

    # Push metadata to XCom
    context['ti'].xcom_push(key='processed_rows', value=len(df))
    context['ti'].xcom_push(key='processed_path', value=processed_path)


def feature_engineering(**context):
    """
    Task 2: Feature Engineering
    Creates new features like tenure_group and avg_monthly_charge.
    """
    print("="*60)
    print("TASK 2: FEATURE ENGINEERING")
    print("="*60)

    # Load processed data
    processed_path = context['ti'].xcom_pull(key='processed_path', task_ids='data_preprocessing')
    print(f"Loading processed data from: {processed_path}")
    df = pd.read_csv(processed_path)

    # Create tenure_group (binned from 0 to 72 months)
    print("\nCreating tenure_group feature...")
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-12', '13-24', '25-48', '49-72'],
        include_lowest=True
    )
    print(f"Tenure groups created: {df['tenure_group'].value_counts().to_dict()}")

    # Create avg_monthly_charge feature
    print("\nCreating avg_monthly_charge feature...")
    df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero
    print(f"avg_monthly_charge statistics:\n{df['avg_monthly_charge'].describe()}")

    # Additional features
    print("\nCreating additional features...")

    # Service count
    service_cols = [col for col in df.columns if any(x in col.lower() for x in ['service', 'backup', 'protection', 'support', 'streaming'])]
    if service_cols:
        df['service_count'] = df[service_cols].apply(lambda x: sum(x != 'No'), axis=1)
        print(f"Service count feature created")

    # Save engineered features
    engineered_path = '/opt/airflow/data/processed/telco_churn_engineered.csv'
    df.to_csv(engineered_path, index=False)
    print(f"\nEngineered data saved to: {engineered_path}")
    print(f"Total features: {len(df.columns)}")
    print("="*60)

    # Push to XCom
    context['ti'].xcom_push(key='engineered_path', value=engineered_path)
    context['ti'].xcom_push(key='feature_count', value=len(df.columns))


def model_training(**context):
    """
    Task 3: Model Training
    Trains RandomForestClassifier using numerical features and saves the model.
    """
    print("="*60)
    print("TASK 3: MODEL TRAINING")
    print("="*60)

    # Load engineered data
    engineered_path = context['ti'].xcom_pull(key='engineered_path', task_ids='feature_engineering')
    print(f"Loading engineered data from: {engineered_path}")
    df = pd.read_csv(engineered_path)

    # Prepare features and target
    target_col = 'Churn'
    print(f"\nTarget column: {target_col}")

    # Select only numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target from features if present
    if target_col in numerical_features:
        numerical_features.remove(target_col)

    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")

    # Prepare X and y
    X = df[numerical_features]
    y = df[target_col].map({'Yes': 1, 'No': 0})

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForestClassifier
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    print("Model training completed!")

    # Save model and scaler
    model_dir = '/opt/airflow/src/models'
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'airflow_rf_model.pkl')
    scaler_path = os.path.join(model_dir, 'airflow_scaler.pkl')
    features_path = os.path.join(model_dir, 'airflow_features.pkl')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(numerical_features, features_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Features saved to: {features_path}")
    print("="*60)

    # Push to XCom
    context['ti'].xcom_push(key='model_path', value=model_path)
    context['ti'].xcom_push(key='scaler_path', value=scaler_path)
    context['ti'].xcom_push(key='features_path', value=features_path)
    context['ti'].xcom_push(key='X_test', value=X_test_scaled.tolist())
    context['ti'].xcom_push(key='y_test', value=y_test.tolist())


def model_evaluation(**context):
    """
    Task 4: Model Evaluation
    Loads trained model and prints ROC AUC score and classification report.
    """
    print("="*60)
    print("TASK 4: MODEL EVALUATION")
    print("="*60)

    # Load model
    model_path = context['ti'].xcom_pull(key='model_path', task_ids='model_training')
    scaler_path = context['ti'].xcom_pull(key='scaler_path', task_ids='model_training')

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Get test data from XCom
    X_test = np.array(context['ti'].xcom_pull(key='X_test', task_ids='model_training'))
    y_test = np.array(context['ti'].xcom_pull(key='y_test', task_ids='model_training'))

    print(f"Test set size: {X_test.shape}")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"\nROC AUC Score: {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    print("="*60)

    # Push to XCom
    context['ti'].xcom_push(key='roc_auc', value=roc_auc)


def generate_predictions(**context):
    """
    Task 5: Generate Predictions
    Loads model and predicts on full dataset, saving predictions with timestamp.
    """
    print("="*60)
    print("TASK 5: GENERATE PREDICTIONS")
    print("="*60)

    # Load model, scaler, and features
    model_path = context['ti'].xcom_pull(key='model_path', task_ids='model_training')
    scaler_path = context['ti'].xcom_pull(key='scaler_path', task_ids='model_training')
    features_path = context['ti'].xcom_pull(key='features_path', task_ids='model_training')

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    # Load full dataset
    engineered_path = context['ti'].xcom_pull(key='engineered_path', task_ids='feature_engineering')
    print(f"Loading data from: {engineered_path}")
    df = pd.read_csv(engineered_path)

    # Prepare features
    X = df[feature_names]
    X_scaled = scaler.transform(X)

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Add predictions to dataframe
    df['predicted_churn'] = ['Yes' if p == 1 else 'No' for p in predictions]
    df['churn_probability'] = probabilities
    df['risk_level'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    # Save predictions with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    predictions_dir = '/opt/airflow/data/predictions'
    os.makedirs(predictions_dir, exist_ok=True)

    predictions_path = os.path.join(predictions_dir, f'churn_predictions_{timestamp}.csv')
    df.to_csv(predictions_path, index=False)

    print(f"\nPredictions saved to: {predictions_path}")
    print(f"Total predictions: {len(df)}")
    print(f"\nPrediction distribution:")
    print(df['predicted_churn'].value_counts())
    print(f"\nRisk level distribution:")
    print(df['risk_level'].value_counts())
    print("="*60)

    # Push to XCom
    context['ti'].xcom_push(key='predictions_path', value=predictions_path)
    context['ti'].xcom_push(key='total_predictions', value=len(df))


# Define the DAG
with DAG(
    dag_id='telco_churn_ml_pipeline',
    default_args=default_args,
    description='Complete ML pipeline for Telco Churn Prediction',
    schedule_interval=timedelta(days=7),  # Weekly schedule
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'churn', 'telco'],
) as dag:

    # Task 1: Data Preprocessing
    task_data_preprocessing = PythonOperator(
        task_id='data_preprocessing',
        python_callable=data_preprocessing,
        provide_context=True,
    )

    # Task 2: Feature Engineering
    task_feature_engineering = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering,
        provide_context=True,
    )

    # Task 3: Model Training
    task_model_training = PythonOperator(
        task_id='model_training',
        python_callable=model_training,
        provide_context=True,
    )

    # Task 4: Model Evaluation
    task_model_evaluation = PythonOperator(
        task_id='model_evaluation',
        python_callable=model_evaluation,
        provide_context=True,
    )

    # Task 5: Generate Predictions
    task_generate_predictions = PythonOperator(
        task_id='generate_predictions',
        python_callable=generate_predictions,
        provide_context=True,
    )

    # Set task dependencies
    task_data_preprocessing >> task_feature_engineering >> task_model_training >> task_model_evaluation >> task_generate_predictions
