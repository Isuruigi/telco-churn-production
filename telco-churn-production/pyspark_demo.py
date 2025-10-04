"""
Simplified PySpark demo for screenshot capture
"""
import os
os.environ['HADOOP_HOME'] = os.path.join(os.getcwd(), 'telco-churn-production', 'hadoop')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import time

print("="*60)
print("TELCO CHURN PREDICTION - PYSPARK PIPELINE")
print("="*60)
print("\nInitializing Spark Session...")

# Initialize Spark
spark = SparkSession.builder \
    .appName("TelcoChurnPrediction") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .master("local[*]") \
    .getOrCreate()

print(f"[OK] Spark Version: {spark.version}")
print(f"[OK] Application Name: TelcoChurnPrediction")
print(f"[OK] Driver Memory: 4g")
print("="*60 + "\n")

# Load data
print("Loading data...")
df = spark.read.csv("telco-churn-production/data/raw/telco_churn.csv", header=True, inferSchema=True)
print(f"[OK] Data loaded: {df.count()} rows, {len(df.columns)} columns")

# Preprocess
df = df.withColumn("TotalCharges", when(col("TotalCharges") == " ", None).otherwise(col("TotalCharges").cast(DoubleType())))
df = df.dropna(subset=["TotalCharges"])
if "customerID" in df.columns:
    df = df.drop("customerID")

# Create label column
df = df.withColumn("label", when(col("Churn") == "Yes", 1).otherwise(0))
print(f"[OK] Preprocessed data: {df.count()} rows")

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"[OK] Training set: {train_df.count()} rows")
print(f"[OK] Test set: {test_df.count()} rows\n")

# Feature engineering
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                   'PaperlessBilling', 'PaymentMethod']

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

print("Building PySpark ML Pipeline...")
print(f"[OK] Categorical features: {len(categorical_cols)}")
print(f"[OK] Numerical features: {len(numerical_cols)}")

# Create pipeline stages
stages = []

# String indexing and one-hot encoding
for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol=f"{col_name}_indexed", outputCol=f"{col_name}_encoded")
    stages += [indexer, encoder]

# Assemble features
feature_cols = [f"{col}_encoded" for col in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features", handleInvalid="skip")
stages.append(assembler)

# Scale features
scaler = StandardScaler(inputCol="assembled_features", outputCol="features")
stages.append(scaler)

# RandomForest
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=10, seed=42)
stages.append(rf)

# Create pipeline
pipeline = Pipeline(stages=stages)
print("[OK] Pipeline created with", len(stages), "stages\n")

# Train
print("="*60)
print("TRAINING MODEL...")
print("="*60)
start_time = time.time()

model = pipeline.fit(train_df)
training_time = time.time() - start_time

print(f"[OK] Training completed in {training_time:.2f} seconds")
print("="*60 + "\n")

# Evaluate
print("="*60)
print("MODEL EVALUATION")
print("="*60)

predictions = model.transform(test_df)

# ROC AUC
roc_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
roc_auc = roc_evaluator.evaluate(predictions)

# Accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)

# F1
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
f1_score = f1_evaluator.evaluate(predictions)

print(f"[OK] ROC AUC Score: {roc_auc:.4f}")
print(f"[OK] Accuracy: {accuracy:.4f}")
print(f"[OK] F1 Score: {f1_score:.4f}")
print("="*60 + "\n")

# Show sample predictions
print("Sample Predictions:")
predictions.select("label", "prediction", "probability").show(10, truncate=False)

print("\n" + "="*60)
print("PYSPARK PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)

# Note: Skipping model save due to Windows Hadoop issues
print("\nNote: Model save skipped due to Windows compatibility")
print("In production, model would be saved to: src/models/pyspark_model/")

spark.stop()
