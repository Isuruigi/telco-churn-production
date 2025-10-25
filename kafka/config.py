"""
Configuration file for Telco Churn Kafka Pipeline
"""
import os
import logging
from pathlib import Path

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:29092']

# Kafka Topics
KAFKA_TOPICS = {
    'raw_customers': 'telco-raw-customers',
    'predictions': 'telco-churn-predictions',
    'deadletter': 'telco-deadletter'
}

# Producer Configuration
PRODUCER_CONFIG = {
    'bootstrap_servers': KAFKA_BOOTSTRAP_SERVERS,
    'value_serializer': lambda v: v,  # Will be set in producer
    'key_serializer': lambda k: k.encode('utf-8') if k else None,
    'acks': 'all',
    'retries': 3,
    'max_in_flight_requests_per_connection': 1
}

# Consumer Configuration
CONSUMER_CONFIG = {
    'bootstrap_servers': KAFKA_BOOTSTRAP_SERVERS,
    'value_deserializer': lambda v: v,  # Will be set in consumer
    'key_deserializer': lambda k: k.decode('utf-8') if k else None,
    'auto_offset_reset': 'earliest',
    'enable_auto_commit': True,
    'group_id': 'telco-churn-consumer-group',
    'max_poll_records': 500,
    'session_timeout_ms': 30000
}

# Streaming Configuration
STREAMING_CONFIG = {
    'default_events_per_sec': 10,
    'min_events_per_sec': 1,
    'max_events_per_sec': 1000
}

# Batch Configuration
BATCH_CONFIG = {
    'default_batch_size': 100,
    'min_batch_size': 10,
    'max_batch_size': 10000,
    'checkpoint_dir': 'checkpoints'
}

# Model Configuration
BASE_DIR = Path(__file__).parent  # kafka/
PROJECT_ROOT = BASE_DIR.parent     # Telco churn project 1/
MODEL_CONFIG = {
    'model_path': PROJECT_ROOT / 'telco-churn-production' / 'src' / 'models' / 'sklearn_pipeline.pkl',
    'model_type': 'sklearn_pipeline'
}

# Data Configuration
DATA_CONFIG = {
    'csv_path': PROJECT_ROOT / 'telco-churn-production' / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv',
    'customer_id_field': 'customerID'
}

# Feature columns (based on Telco dataset)
FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Target column
TARGET_COLUMN = 'Churn'

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'kafka_pipeline.log',
            'mode': 'a'
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'kafka': {
            'handlers': ['console', 'file'],
            'level': 'WARNING',
            'propagate': False
        }
    }
}


def setup_logging():
    """Setup logging configuration"""
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger(__name__)


def create_directories():
    """Create necessary directories if they don't exist"""
    checkpoint_dir = Path(BATCH_CONFIG['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)

    # Create models directory if it doesn't exist
    MODEL_CONFIG['model_path'].parent.mkdir(parents=True, exist_ok=True)


# Validate paths on import
def validate_config():
    """Validate configuration paths and settings"""
    errors = []

    # Check if CSV exists
    if not DATA_CONFIG['csv_path'].exists():
        errors.append(f"CSV file not found: {DATA_CONFIG['csv_path']}")

    # Check if model exists
    if not MODEL_CONFIG['model_path'].exists():
        errors.append(f"Model file not found: {MODEL_CONFIG['model_path']}")

    if errors:
        print("Configuration warnings:")
        for error in errors:
            print(f"  - {error}")

    return len(errors) == 0


if __name__ == "__main__":
    print("Kafka Pipeline Configuration")
    print("=" * 50)
    print(f"Kafka Bootstrap Servers: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Topics: {KAFKA_TOPICS}")
    print(f"Model Path: {MODEL_CONFIG['model_path']}")
    print(f"CSV Path: {DATA_CONFIG['csv_path']}")
    print("\nValidating configuration...")
    if validate_config():
        print("✓ Configuration is valid!")
    else:
        print("✗ Configuration has warnings (see above)")
