import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Data paths
DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "telco_churn.csv")

# Models path
MODELS_PATH = os.path.join(ROOT_DIR, "models")

# Reports path
REPORTS_PATH = os.path.join(ROOT_DIR, "reports")

# ML model configuration
RANDOM_STATE = 42
TARGET_COLUMN = "Churn"

# Feature lists
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]
