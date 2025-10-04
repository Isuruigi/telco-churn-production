# This file contains configurations specifically for the production environment.

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://production-mlflow-server:5000"
EXPERIMENT_NAME = "prod-telco-churn-prediction"
MODEL_REGISTRY_NAME = "prod-telco-churn-model"

# --- Data and Model Paths (Production) ---
# These would typically point to a cloud storage solution like S3, GCS, or Azure Blob Storage.
DATA_PATHS = {
    "raw_data": "s3://your-production-bucket/data/raw/telco_churn.csv",
    "processed_data": "s3://your-production-bucket/data/processed/",
}

MODEL_PATHS = {
    "preprocessor": "s3://your-production-bucket/models/preprocessor.joblib",
    "final_model": "s3://your-production-bucket/models/final_churn_model.joblib",
}

# --- Spark Configuration ---
# Example configuration for a production Spark cluster
SPARK_CONFIG = {
    "master": "spark://your-spark-master:7077",
    "appName": "TelcoChurnProductionPipeline",
    "spark.executor.memory": "4g",
    "spark.executor.cores": "2",
    "spark.driver.memory": "2g",
}

# --- Airflow Configuration ---
# Configuration for DAGs and connections in a production Airflow environment
AIRFLOW_CONFIG = {
    "default_args": {
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 1,
    },
    "schedule_interval": "@daily",
    "catchup": False,
}

# --- API Configuration ---
# Configuration for the prediction service API (e.g., using FastAPI)
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "title": "Telco Churn Prediction API",
    "description": "API for real-time churn predictions.",
    "version": "1.0.0",
}

# --- Monitoring Configuration ---
# Endpoints and settings for monitoring tools like Prometheus and Grafana
MONITORING_CONFIG = {
    "prometheus_endpoint": "http://your-prometheus-server/api/v1/write",
    "grafana_dashboard_url": "http://your-grafana-server/d/your-dashboard-id",
    "log_level": "INFO",
}
