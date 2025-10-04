import os
import json
import logging
from typing import Optional

def setup_logging(log_file='project.log'):
    """Set up logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_directories(dir_paths: list):
    """Create directories if they do not exist."""
    for path in dir_paths:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")

def save_json(data: dict, file_path: str):
    """Save a dictionary to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Successfully saved JSON to {file_path}")
    except Exception as e:
        logging.error(f"Error saving JSON to {file_path}: {e}")

def load_json(file_path: str) -> Optional[dict]:
    """Load a JSON file and return a dictionary."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Successfully loaded JSON from {file_path}")
            return data
    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading JSON from {file_path}: {e}")
        return None

def calculate_business_metrics(y_true, y_pred, monthly_charges):
    """
    Calculate business-oriented metrics for a churn model.

    Args:
        y_true (pd.Series): True churn labels.
        y_pred (pd.Series): Predicted churn labels.
        monthly_charges (pd.Series): Monthly charges for each customer.

    Returns:
        dict: A dictionary of business metrics.
    """
    results = {
        'total_customers': len(y_true),
        'actual_churners': int(y_true.sum()),
        'predicted_churners': int(y_pred.sum()),
        'correctly_identified_churners': int((y_true & y_pred).sum()),
        'total_revenue_at_risk': format_currency(monthly_charges[y_true == 1].sum()),
        'potential_revenue_saved': format_currency(monthly_charges[(y_true == 1) & (y_pred == 1)].sum()),
    }
    return results

def format_currency(amount: float) -> str:
    """Format a number as a currency string (USD)."""
    return f"${amount:,.2f}"

if __name__ == '__main__':
    # Example Usage
    setup_logging('utils_test.log')

    # Directory creation
    create_directories(['test_dir/subdir'])

    # JSON operations
    test_data = {'key': 'value', 'number': 123}
    json_path = 'test_dir/test.json'
    save_json(test_data, json_path)
    loaded_data = load_json(json_path)
    if loaded_data:
        logging.info(f"Loaded data: {loaded_data}")

    # Business metrics calculation (dummy data)
    import pandas as pd
    true_labels = pd.Series([1, 0, 1, 0, 1, 0])
    pred_labels = pd.Series([1, 0, 0, 0, 1, 1])
    charges = pd.Series([100, 50, 80, 60, 120, 40])
    metrics = calculate_business_metrics(true_labels, pred_labels, charges)
    logging.info(f"Business metrics: {metrics}")
