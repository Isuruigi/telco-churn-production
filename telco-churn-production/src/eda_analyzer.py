import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config.config import TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, REPORTS_PATH
from src.data_loader import TelcoDataLoader
from src.utils import create_directories

class TelcoEDAAnalyzer:
    """Performs exploratory data analysis on the Telco Customer Churn dataset."""

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the TelcoEDAAnalyzer.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
        """
        self.df = df
        self.figures_path = os.path.join(REPORTS_PATH, 'figures', 'eda')
        create_directories([self.figures_path])

    def analyze_target_distribution(self):
        """Analyzes and plots the distribution of the target variable."""
        print("--- Target Variable Distribution ---")
        print(self.df[TARGET_COLUMN].value_counts(normalize=True))
        
        plt.figure(figsize=(6, 4))
        sns.countplot(x=TARGET_COLUMN, data=self.df)
        plt.title('Distribution of Customer Churn')
        plt.savefig(os.path.join(self.figures_path, 'churn_distribution.png'))
        plt.close()
        print("--- End Target Variable Distribution ---\n")

    def analyze_feature_distributions(self):
        """Analyzes and plots the distributions of numerical and categorical features."""
        print("--- Feature Distributions ---")
        # Numerical features
        for col in NUMERICAL_FEATURES:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(self.figures_path, f'{col}_distribution.png'))
            plt.close()

        # Categorical features
        for col in CATEGORICAL_FEATURES:
            plt.figure(figsize=(10, 6))
            sns.countplot(y=col, data=self.df, order=self.df[col].value_counts().index)
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(self.figures_path, f'{col}_distribution.png'))
            plt.close()
        print("--- End Feature Distributions ---\n")

    def analyze_correlations(self):
        """Analyzes and plots the correlations between numerical features."""
        print("--- Correlation Analysis ---")
        # Convert TotalCharges to numeric, coercing errors
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        
        # Recalculate numerical features to avoid issues with non-numeric TotalCharges
        numerical_df = self.df[NUMERICAL_FEATURES].copy()
        correlation_matrix = numerical_df.corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Numerical Features')
        plt.savefig(os.path.join(self.figures_path, 'correlation_matrix.png'))
        plt.close()
        print("--- End Correlation Analysis ---\n")

    def detect_outliers(self):
        """Detects and visualizes outliers in numerical features using box plots."""
        print("--- Outlier Detection ---")
        for col in NUMERICAL_FEATURES:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=self.df[col])
            plt.title(f'Box Plot of {col}')
            plt.savefig(os.path.join(self.figures_path, f'{col}_boxplot.png'))
            plt.close()
        print("--- End Outlier Detection ---\n")

    def analyze_missing_values(self):
        """Analyzes and reports missing values in the dataset."""
        print("--- Missing Value Analysis ---")
        missing_values = self.df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            print("Columns with missing values:")
            print(missing_values)
        else:
            print("No missing values found.")
        print("--- End Missing Value Analysis ---\n")

if __name__ == '__main__':
    # Load the data
    # Note: This requires the config to be importable from the root of the project.
    # To run this script directly, you might need to adjust the python path.
    # Example: PYTHONPATH=. python src/eda_analyzer.py
    try:
        loader = TelcoDataLoader()
        churn_df = loader.load_raw_data()

        if churn_df is not None:
            # Initialize the analyzer
            analyzer = TelcoEDAAnalyzer(churn_df)

            # Run the analyses
            analyzer.analyze_missing_values()
            analyzer.analyze_target_distribution()
            analyzer.analyze_feature_distributions()
            analyzer.analyze_correlations()
            analyzer.detect_outliers()
    except ImportError as e:
        print(f"Error: {e}. Please run this script from the root of the project.")