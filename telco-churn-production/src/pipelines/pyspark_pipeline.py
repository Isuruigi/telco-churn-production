"""
Telco Churn Prediction - PySpark MLlib Pipeline
This module implements a complete ML pipeline using PySpark's MLlib for distributed processing.
"""

import yaml
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)


class PySparkMLPipeline:
    """
    Complete PySpark MLlib pipeline for Telco Churn prediction.
    Handles distributed data processing, feature engineering, and model training.
    """

    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize the PySpark ML pipeline.

        Args:
            config_path (str): Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.spark = None
        self.model = None
        self.pipeline = None

    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def initialize_spark(self):
        """
        Initialize SparkSession with application name 'TelcoChurnPrediction'
        and driver memory configured to 4g.
        """
        print("="*60)
        print("INITIALIZING SPARK SESSION")
        print("="*60)

        self.spark = SparkSession.builder \
            .appName("TelcoChurnPrediction") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .master("local[*]") \
            .getOrCreate()

        print(f"Spark Version: {self.spark.version}")
        print(f"Application Name: TelcoChurnPrediction")
        print(f"Driver Memory: 4g")
        print("="*60 + "\n")

        return self.spark

    def load_data(self):
        """
        Load CSV data, convert TotalCharges to numeric, and add binary label column.

        Returns:
            DataFrame: PySpark DataFrame with processed data
        """
        data_path = self.config['data']['raw']
        print(f"Loading data from: {data_path}")

        # Read CSV
        df = self.spark.read.csv(data_path, header=True, inferSchema=True)

        print(f"Initial data shape: {df.count()} rows, {len(df.columns)} columns")

        # Convert TotalCharges to numeric type (handle spaces and errors)
        df = df.withColumn(
            "TotalCharges",
            when(col("TotalCharges") == " ", None)
            .otherwise(col("TotalCharges").cast(DoubleType()))
        )

        # Drop rows with null TotalCharges
        df = df.dropna(subset=["TotalCharges"])

        # Drop customerID if exists (not a feature)
        if "customerID" in df.columns:
            df = df.drop("customerID")

        # Add binary label column (1 for Churn=Yes, 0 for Churn=No)
        target_column = self.config['target']
        df = df.withColumn(
            "label",
            when(col(target_column) == "Yes", 1.0).otherwise(0.0)
        )

        # Drop original Churn column
        df = df.drop(target_column)

        print(f"Processed data shape: {df.count()} rows, {len(df.columns)} columns")
        print(f"Label distribution:")
        df.groupBy("label").count().show()

        return df

    def create_pipeline(self, df):
        """
        Create ML pipeline with feature engineering and RandomForestClassifier.

        Pipeline stages:
        1. StringIndexer for categorical columns
        2. OneHotEncoder for indexed categorical columns
        3. VectorAssembler to combine features
        4. StandardScaler for feature scaling
        5. RandomForestClassifier

        Args:
            df: PySpark DataFrame

        Returns:
            Pipeline: Complete ML pipeline
        """
        print("\n" + "="*60)
        print("CREATING ML PIPELINE")
        print("="*60)

        numerical_features = self.config['features']['numerical']
        categorical_features = self.config['features']['categorical']

        # Filter categorical features that exist in the dataframe
        categorical_features = [f for f in categorical_features if f in df.columns]

        print(f"Numerical features: {len(numerical_features)}")
        print(f"Categorical features: {len(categorical_features)}")

        stages = []

        # Stage 1: StringIndexer for categorical columns
        indexers = []
        indexed_cols = []
        for cat_col in categorical_features:
            indexer = StringIndexer(
                inputCol=cat_col,
                outputCol=f"{cat_col}_indexed",
                handleInvalid="keep"
            )
            indexers.append(indexer)
            indexed_cols.append(f"{cat_col}_indexed")

        stages.extend(indexers)
        print(f"Added {len(indexers)} StringIndexers")

        # Stage 2: OneHotEncoder for indexed categorical columns
        encoder = OneHotEncoder(
            inputCols=indexed_cols,
            outputCols=[f"{col}_encoded" for col in indexed_cols],
            handleInvalid="keep"
        )
        stages.append(encoder)
        print(f"Added OneHotEncoder")

        # Stage 3: VectorAssembler to combine all features
        encoded_cols = [f"{col}_encoded" for col in indexed_cols]
        all_feature_cols = numerical_features + encoded_cols

        assembler = VectorAssembler(
            inputCols=all_feature_cols,
            outputCol="assembled_features",
            handleInvalid="skip"
        )
        stages.append(assembler)
        print(f"Added VectorAssembler with {len(all_feature_cols)} features")

        # Stage 4: StandardScaler
        scaler = StandardScaler(
            inputCol="assembled_features",
            outputCol="features",
            withStd=True,
            withMean=False  # False for sparse vectors
        )
        stages.append(scaler)
        print(f"Added StandardScaler")

        # Stage 5: RandomForestClassifier
        rf_params = self.config['models']['random_forest']
        rf = RandomForestClassifier(
            numTrees=rf_params['n_estimators'],
            maxDepth=rf_params['max_depth'],
            seed=rf_params['random_state'],
            featuresCol="features",
            labelCol="label"
        )
        stages.append(rf)
        print(f"Added RandomForestClassifier (numTrees={rf_params['n_estimators']}, maxDepth={rf_params['max_depth']})")

        # Create pipeline
        self.pipeline = Pipeline(stages=stages)
        print(f"\nTotal pipeline stages: {len(stages)}")
        print("="*60 + "\n")

        return self.pipeline

    def train_and_evaluate(self, df):
        """
        Train and evaluate the model.
        - Splits data into 80/20 train/test
        - Trains the pipeline
        - Evaluates using ROC AUC, accuracy, and F1 score
        - Saves the trained model

        Args:
            df: PySpark DataFrame with features and label
        """
        print("="*60)
        print("TRAINING AND EVALUATION")
        print("="*60)

        # Split data into 80/20 train/test
        test_size = self.config['training']['test_size']
        train_size = 1.0 - test_size
        random_state = self.config['training']['random_state']

        train_df, test_df = df.randomSplit([train_size, test_size], seed=random_state)

        print(f"Training set: {train_df.count()} rows")
        print(f"Test set: {test_df.count()} rows")

        # Train the pipeline
        print("\nTraining model...")
        start_time = time.time()

        self.model = self.pipeline.fit(train_df)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Make predictions on test set
        print("\nEvaluating model on test set...")
        predictions = self.model.transform(test_df)

        # Evaluate using BinaryClassificationEvaluator for ROC AUC
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        roc_auc = binary_evaluator.evaluate(predictions)

        # Evaluate using MulticlassClassificationEvaluator for accuracy and F1
        multiclass_evaluator_accuracy = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = multiclass_evaluator_accuracy.evaluate(predictions)

        multiclass_evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )
        f1_score = multiclass_evaluator_f1.evaluate(predictions)

        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print("="*60)

        # Show sample predictions
        print("\nSample Predictions:")
        predictions.select("label", "prediction", "probability").show(10, truncate=False)

        # Save the trained model
        model_path = "src/models/pyspark_model"
        print(f"\nSaving model to: {model_path}")
        self.model.write().overwrite().save(model_path)
        print("Model saved successfully!")

        return {
            'training_time': training_time,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'f1_score': f1_score
        }

    def stop_spark(self):
        """Stop the Spark session."""
        if self.spark:
            print("\nStopping Spark session...")
            self.spark.stop()
            print("Spark session stopped.")

    def run_complete_pipeline(self):
        """Execute the complete PySpark ML pipeline."""
        try:
            print("\n" + "="*60)
            print("TELCO CHURN PREDICTION - PYSPARK PIPELINE")
            print("="*60 + "\n")

            # Initialize Spark
            self.initialize_spark()

            # Load data
            df = self.load_data()

            # Create pipeline
            pipeline = self.create_pipeline(df)

            # Train and evaluate
            metrics = self.train_and_evaluate(df)

            print("\n" + "="*60)
            print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            print("="*60)

            return metrics

        except Exception as e:
            print(f"\nError during pipeline execution: {e}")
            raise

        finally:
            # Always stop Spark session
            self.stop_spark()


def main():
    """Main execution function."""
    # Initialize and run PySpark pipeline
    pyspark_pipeline = PySparkMLPipeline(config_path='config/config.yaml')
    metrics = pyspark_pipeline.run_complete_pipeline()

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Training Time: {metrics['training_time']:.2f} seconds")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
