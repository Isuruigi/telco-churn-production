import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import TelcoDataLoader
from src.preprocessor import DataPreprocessor
from src.feature_engineer import FeatureEngineer
from src.imbalance_handler import ImbalanceHandler
from src.config.config import TARGET_COLUMN, RANDOM_STATE

class DataPipeline:
    """Orchestrates the end-to-end data processing pipeline."""

    def __init__(self, data_path: str):
        """
        Initializes the DataPipeline.

        Args:
            data_path (str): The path to the raw data file.
        """
        self.data_path = data_path
        self.loader = TelcoDataLoader(data_path=self.data_path)
        self.preprocessor_builder = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.imbalance_handler = ImbalanceHandler()

    def load_and_validate(self) -> pd.DataFrame:
        """Loads and validates the raw data."""
        print("--- 1. Loading and Validating Data ---")
        df = self.loader.load_raw_data()
        if df is not None:
            self.loader.validate_data_quality(df)
        print("--- Data Loaded and Validated ---\n")
        return df

    def preprocess_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies preprocessing and feature engineering steps."""
        print("--- 2. Preprocessing and Engineering Features ---")
        # Initial cleaning
        df_processed = self.preprocessor_builder.preprocess_data(df)
        # Feature engineering
        df_engineered = self.feature_engineer.transform(df_processed)
        print("--- Features Preprocessed and Engineered ---\n")
        return df_engineered

    def split_data(self, df: pd.DataFrame, test_size=0.2, stratify=True):
        """Splits the data into training and testing sets."""
        print("--- 3. Splitting Data ---")
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN].apply(lambda x: 1 if x == 'Yes' else 0)
        
        stratify_col = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify_col
        )
        print(f"Data split into training ({len(X_train)}) and testing ({len(X_test)}) sets.")
        print("--- Data Splitting Complete ---\n")
        return X_train, X_test, y_train, y_test

    def apply_pipeline_and_imbalance_handling(self, X_train, y_train, X_test, handle_imbalance_method=None):
        """
        Applies the full preprocessing pipeline and optional imbalance handling.
        """
        print("--- 4. Applying Full Pipeline and Imbalance Handling ---")
        # Create the ColumnTransformer pipeline
        # Note: We need to get the correct feature lists after engineering
        engineered_cols = self.feature_engineer.transform(X_train.copy()).columns
        numerical_features = [col for col in self.preprocessor_builder.numerical_features if col in engineered_cols]
        categorical_features = [col for col in self.preprocessor_builder.categorical_features if col in engineered_cols]
        # Add new engineered features to the lists
        new_numerical = ['tenure_x_monthly_charges', 'monthly_charge_per_tenure', 'total_charges_to_tenure_ratio', 'num_services']
        new_categorical = ['tenure_category', 'has_streaming']
        numerical_features.extend([f for f in new_numerical if f in engineered_cols])
        categorical_features.extend([f for f in new_categorical if f in engineered_cols])

        preprocessor = DataPreprocessor(numerical_features=list(set(numerical_features)), 
                                      categorical_features=list(set(categorical_features))).create_preprocessing_pipeline()

        # Fit on training data and transform both train and test data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Handle imbalance on the transformed training data
        if handle_imbalance_method:
            print(f"Handling imbalance using: {handle_imbalance_method}")
            if handle_imbalance_method == 'smote':
                X_train_transformed, y_train = self.imbalance_handler.apply_smote(X_train_transformed, y_train)
            # Add other methods here...
        
        print("--- Pipeline and Imbalance Handling Complete ---\n")
        return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor

    def run(self, test_size=0.2, handle_imbalance_method=None):
        """Runs the full data pipeline."""
        df = self.load_and_validate()
        if df is None:
            return None
        
        df_engineered = self.preprocess_and_engineer_features(df)
        X_train, X_test, y_train, y_test = self.split_data(df_engineered, test_size=test_size)
        
        X_train_final, X_test_final, y_train_final, y_test_final, pipeline_fitted = self.apply_pipeline_and_imbalance_handling(
            X_train, y_train, X_test, handle_imbalance_method
        )
        
        return X_train_final, X_test_final, y_train_final, y_test_final, pipeline_fitted

if __name__ == '__main__':
    from src.config.config import DATA_PATH

    # Initialize and run the pipeline
    data_pipeline = DataPipeline(data_path=DATA_PATH)
    
    # Run without imbalance handling
    # X_train, X_test, y_train, y_test, preprocessor = data_pipeline.run()

    # Run with SMOTE for imbalance handling
    X_train_smote, X_test_smote, y_train_smote, y_test_smote, preprocessor_smote = data_pipeline.run(handle_imbalance_method='smote')

    if X_train_smote is not None:
        print("Pipeline executed successfully!")
        print(f"Shape of final training data: {X_train_smote.shape}")
        print(f"Shape of final testing data: {X_test_smote.shape}")
