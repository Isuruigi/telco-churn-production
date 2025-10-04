import pandas as pd
from typing import Optional
from src.config.config import DATA_PATH

class TelcoDataLoader:
    """Data loader for the Telco Customer Churn dataset."""

    def __init__(self, data_path=DATA_PATH):
        """
        Initializes the TelcoDataLoader.

        Args:
            data_path (str): The path to the Telco Customer Churn CSV file.
        """
        self.data_path = data_path

    def load_raw_data(self) -> Optional[pd.DataFrame]:
        """
        Loads the Telco Customer Churn dataset from a CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame, or None if an error occurs.
        """
        try:
            df = pd.read_csv(self.data_path)
            print(f"Successfully loaded data from {self.data_path}")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return None

    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Performs data quality checks on the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if data quality is acceptable, False otherwise.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            print("Validation failed: Input is not a valid pandas DataFrame.")
            return False

        print("\n--- Data Quality Report ---")

        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            print("No missing values found.")
        else:
            print("Missing values found:")
            print(missing_values[missing_values > 0])

        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows == 0:
            print("No duplicate rows found.")
        else:
            print(f"{duplicate_rows} duplicate rows found.")

        print("--- End Data Quality Report ---\n")
        return True

    def get_data_info(self, df: pd.DataFrame):
        """
        Prints general information about the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to get info from.
        """
        if df is None:
            return
        print("\n--- DataFrame Info ---")
        df.info()
        print("\n--- DataFrame Description ---")
        print(df.describe(include='all'))
        print("--- End DataFrame Info ---\n")

    def generate_data_report(self, df: pd.DataFrame):
        """
        Generates a simple report for each column in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to generate the report for.
        """
        if df is None:
            return
        print("\n--- Data Column Report ---")
        for col in df.columns:
            print(f"\nColumn: {col}")
            print(f"  Data Type: {df[col].dtype}")
            print(f"  Number of Missing Values: {df[col].isnull().sum()}")
            print(f"  Number of Unique Values: {df[col].nunique()}")
            if df[col].nunique() < 10:
                print(f"  Unique Values: {df[col].unique()}")
        print("--- End Data Column Report ---\n")

if __name__ == '__main__':
    try:
        loader = TelcoDataLoader()
        churn_df = loader.load_raw_data()
        if churn_df is not None:
            loader.validate_data_quality(churn_df)
            loader.get_data_info(churn_df)
            loader.generate_data_report(churn_df)
    except ImportError as e:
        print(f"Error: {e}. Please run this script from the root of the project.")
